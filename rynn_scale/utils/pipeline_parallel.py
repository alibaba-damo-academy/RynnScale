from abc import ABC, abstractmethod
from queue import Queue
from typing import List, Dict, Any
from enum import Enum

import torch
import torch.distributed as dist
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.pipe import PipelineModule as _PipelineModule
from deepspeed.runtime.pipe.topology import ProcessTopology, PipelineParallelGrid
from transformers import PreTrainedModel

from ..utils import logging


logger = logging.get_logger(__name__)


class PipelineModule(_PipelineModule):
    def __init__(
        self,
        module: PreTrainedModel,
        pipeline_model_parallel_size: int,
        pipeline_model_parallel_rank: int,
        data_parallel_size: int,
        data_parallel_rank: int,
    ):
        super(_PipelineModule, self).__init__()
        self.module = module

        # DeepSpeed compatibility
        self._topo = ProcessTopology(
            axes=["pipe", "data"],
            dims=[pipeline_model_parallel_size, data_parallel_size],
        )
        self._grid = PipelineParallelGrid(self._topo)

        assert self._grid.get_pipeline_model_parallel_rank() == pipeline_model_parallel_rank
        assert self._grid.get_data_parallel_rank() == data_parallel_rank

        self.loss_fn = None

        self.tied_comms = {}
        self.activation_checkpoint_interval = -1
        self.dynamic_shape = False

        layer_indices = sorted(self.module.language_model.layers.keys())
        self._local_start = int(layer_indices[0])
        self._local_stop = int(layer_indices[-1]) + 1

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class PipelineStage(object):
    def __init__(
        self,
        module: torch.nn.Module,
        deepspeed_engine: DeepSpeedEngine,
        group: torch.distributed.ProcessGroup,
    ):
        self.module = module
        self.group = group
        self.deepspeed_engine = deepspeed_engine

        self.using_bf16_optimizer = type(self.deepspeed_engine.optimizer) is BF16_Optimizer

        self.num_stages = dist.get_world_size(self.group)
        self.stage_index = dist.get_rank(self.group)

        if self.num_stages > 1:
            assert self.deepspeed_engine.zero_optimization_stage() < ZeroStageEnum.gradients, (
                "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"
            )

        self.hidden_size = module.config.get_text_config().hidden_size

        self.input_queue = Queue()
        self.output_queue = Queue()

        self.fwd_recv_buffer = None
        self.bwd_recv_buffer = None

        self.is_first_stage = self.stage_index == 0
        self.is_last_stage = self.stage_index == self.num_stages - 1

        self.prev_stage_rank = (
            dist.get_global_rank(self.group, self.stage_index - 1) if not self.is_first_stage else None
        )

        self.next_stage_rank = (
            dist.get_global_rank(self.group, self.stage_index + 1) if not self.is_last_stage else None
        )

    @property
    def device(self):
        return self.module.device

    @torch.cuda.nvtx.range("forward")
    def forward_one_chunk(
        self,
        batches: List[Dict[str, Any]],
        batch_index: int,
    ):
        model_inputs = {}
        for k, v in batches[batch_index].items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
            model_inputs[k] = v

        if self.is_first_stage:
            assert "input_ids" in model_inputs
        else:
            model_inputs.pop("input_ids", None)
            assert self.fwd_recv_buffer is not None
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, forward batch {batch_index + 1}, receive: {self.fwd_recv_buffer.shape}",
            )
            self.fwd_recv_buffer.requires_grad_(True)
            self.input_queue.put(self.fwd_recv_buffer)
            model_inputs["inputs_embeds"] = self.fwd_recv_buffer

        if self.is_last_stage:
            assert "labels" in model_inputs
        else:
            model_inputs.pop("labels", None)

        outputs = self.module(**model_inputs)
        self.fwd_recv_buffer = None

        send_ops = []
        if self.is_last_stage:
            self.output_queue.put(outputs.loss)
            loss = outputs.loss.clone().detach()
        else:
            self.output_queue.put(outputs.last_hidden_state)
            loss = None
            send_ops.append(
                dist.P2POp(
                    dist.isend,
                    outputs.last_hidden_state,
                    self.next_stage_rank,
                    self.group,
                )
            )
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, forward batch {batch_index + 1}, send: {outputs.last_hidden_state.shape}",
            )

        return loss, send_ops

    @torch.cuda.nvtx.range("backward")
    def backward_one_chunk(self, batch_index: int):
        assert not self.output_queue.empty()
        output = self.output_queue.get()

        if self.is_last_stage:
            self.deepspeed_engine.backward(output)
        else:
            # https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py#L805
            if self.using_bf16_optimizer:
                # manually call because we don't call optimizer.backward()
                self.deepspeed_engine.optimizer.clear_lp_grads()

            self.deepspeed_engine._running_engine_backward = True
            assert self.bwd_recv_buffer is not None
            output.backward(self.bwd_recv_buffer)
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, backward batch {batch_index + 1}, receive: {self.bwd_recv_buffer.shape}",
            )
            self.bwd_recv_buffer = None
            self.deepspeed_engine._running_engine_backward = False

            if self.using_bf16_optimizer:
                # manually call because we don't call optimizer.backward()
                if not self.deepspeed_engine._config.bfloat16_config.immediate_grad_update:
                    self.deepspeed_engine.optimizer.update_hp_grads(clear_lp_grads=False)

        send_ops = []
        if not self.is_first_stage:
            assert not self.input_queue.empty()
            grad_input = self.input_queue.get().grad
            send_ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_input,
                    self.prev_stage_rank,
                    self.group,
                )
            )
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, backward batch {batch_index + 1}, send: {grad_input.shape}",
            )

        return send_ops

    def _shape_inference(self, batch: Dict[str, Any]):
        if "input_ids" in batch:
            shape = batch["input_ids"].shape
        elif "position_ids" in batch:
            shape = batch["position_ids"].shape
        elif "labels" in batch:
            shape = batch["labels"].shape
        else:
            raise RuntimeError("Cannot infer shape from batch")
        return shape

    def get_fwd_recv_ops(self, batch: Dict[str, Any]) -> List[dist.P2POp]:
        ops = []
        if self.prev_stage_rank is not None:
            shape = self._shape_inference(batch)
            self.fwd_recv_buffer = torch.empty(
                size=(*shape, self.hidden_size),
                dtype=self.module.dtype,
                device=self.module.device,
            )
            ops.append(dist.P2POp(dist.irecv, self.fwd_recv_buffer, self.prev_stage_rank, self.group))
        return ops

    def get_bwd_recv_ops(self, batch: Dict[str, Any]) -> List[dist.P2POp]:
        ops = []
        if self.next_stage_rank is not None:
            shape = self._shape_inference(batch)
            self.bwd_recv_buffer = torch.empty(
                size=(*shape, self.hidden_size),
                dtype=self.module.dtype,
                device=self.device,
            )
            ops.append(dist.P2POp(dist.irecv, self.bwd_recv_buffer, self.next_stage_rank, self.group))
        return ops


def _batch_isend_irecv(ops: List[dist.P2POp]) -> List[dist.Work]:
    if len(ops) == 0:
        return []
    return dist.batch_isend_irecv(ops)


class BasePipelineSchedule(ABC):
    def __init__(
        self,
        stages: List[PipelineStage],
        deepspeed_engine: DeepSpeedEngine,
    ):
        assert isinstance(stages, (list, tuple)) and len(stages) > 0
        self.stages = stages
        self.deepspeed_engine = deepspeed_engine
        self.using_bf16_optimizer = type(self.deepspeed_engine.optimizer) is BF16_Optimizer

    @abstractmethod
    def _step(self, batches: List[Dict[str, Any]]):
        raise NotImplementedError

    def step(self, batches: List[Dict[str, Any]]):
        self.deepspeed_engine.set_gradient_accumulation_boundary(is_boundary=False)
        losses = self._step(batches)
        self.deepspeed_engine.set_gradient_accumulation_boundary(is_boundary=True)

        if self.using_bf16_optimizer:
            # PP+BF16 work for ZeRO Stage 1
            self.deepspeed_engine._bf16_reduce_grads()
        else:
            self.deepspeed_engine.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)

        return losses


class ScheduleNoPipelining(BasePipelineSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.stages) == 1

    def _step(self, batches: List[Dict[str, Any]]):
        losses = []
        for i in range(len(batches)):
            loss, _ = self.stages[0].forward_one_chunk(batches, batch_index=i)
            self.stages[0].backward_one_chunk(batch_index=i)
            losses.append(loss)
        return torch.stack(losses).to(torch.float32)


class ScheduleGPipe(BasePipelineSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.stages) == 1

    def _step(self, batches: List[Dict[str, Any]]):
        raise NotImplementedError


class Schedule1F1B(BasePipelineSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.stages) == 1

    def _step(self, batches: List[Dict[str, Any]]):
        stage = self.stages[0]
        losses = []

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(len(batches), stage.num_stages - stage.stage_index)

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_works: List[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            fwd_recvs = stage.get_fwd_recv_ops(batches[fwd_mb_index])
            for work in _batch_isend_irecv(fwd_recvs):
                work.wait()

            loss, fwd_sends = stage.forward_one_chunk(batches, batch_index=fwd_mb_index)

            for work in send_works:
                work.wait()

            if fwd_mb_index != warmup_chunks - 1:
                send_works = _batch_isend_irecv(fwd_sends)

            losses.append(loss)
            fwd_mb_index += 1

        # 1B1F phase
        while True:
            bwd_recvs = stage.get_bwd_recv_ops(batches[bwd_mb_index])
            for work in _batch_isend_irecv(fwd_sends + bwd_recvs):
                work.wait()

            bwd_sends = stage.backward_one_chunk(batch_index=bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == len(batches):
                break

            fwd_recvs = stage.get_fwd_recv_ops(batches[fwd_mb_index])
            for work in _batch_isend_irecv(bwd_sends + fwd_recvs):
                work.wait()

            loss, fwd_sends = stage.forward_one_chunk(batches, batch_index=fwd_mb_index)

            losses.append(loss)
            fwd_mb_index += 1

        send_works = _batch_isend_irecv(bwd_sends)

        # Cooldown phase
        while bwd_mb_index < len(batches):
            bwd_recvs = stage.get_bwd_recv_ops(batches[bwd_mb_index])
            for work in _batch_isend_irecv(bwd_recvs):
                work.wait()

            bwd_sends = stage.backward_one_chunk(batch_index=bwd_mb_index)

            for work in send_works:
                work.wait()

            send_works = _batch_isend_irecv(bwd_sends)

            bwd_mb_index += 1

        if stage.is_last_stage:
            losses = torch.stack(losses).to(torch.float32)
        else:
            losses = torch.ones(len(losses), device=stage.device, dtype=torch.float32)

        dist.broadcast(losses, group=stage.group, group_src=stage.num_stages - 1)

        for work in send_works:
            work.wait()

        return losses


class ScheduleInterleaved1F1B(BasePipelineSchedule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.stages) > 1

    def _step(self, batches: List[Dict[str, Any]]):
        raise NotImplementedError


class PipelineSchedule(Enum):
    NO_PIPELINING = None
    SCHEDULE_GPIPE = "gpipe"
    SCHEDULE_1F1B = "1f1b"
    SCHEDULE_INTERLEAVED_1F1B = "interleaved_1f1b"


ALL_PIPELINE_SCHEDULES = {
    PipelineSchedule.NO_PIPELINING: ScheduleNoPipelining,
    PipelineSchedule.SCHEDULE_GPIPE: ScheduleGPipe,
    PipelineSchedule.SCHEDULE_1F1B: Schedule1F1B,
    PipelineSchedule.SCHEDULE_INTERLEAVED_1F1B: ScheduleInterleaved1F1B,
}
