from abc import ABCMeta, abstractmethod
from enum import Enum
from queue import Queue
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from deepspeed.pipe import PipelineModule as _PipelineModule
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.engine import MEMORY_OPT_ALLREDUCE_SIZE, DeepSpeedEngine
from deepspeed.runtime.pipe.topology import PipelineParallelGrid, ProcessTopology
from deepspeed.runtime.zero.config import ZeroStageEnum
from transformers import PreTrainedModel

from .. import parallel_state as mpu
from ..utils import logging

logger = logging.get_logger(__name__)


class PipelineModule(_PipelineModule):
    def __init__(self, module: PreTrainedModel):
        super(_PipelineModule, self).__init__()
        self.module = module

        # DeepSpeed compatibility
        pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        dcp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        dcp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)

        self._topo = ProcessTopology(
            axes=["pipe", "data"],
            dims=[pp_world_size, dcp_world_size],
        )
        self._grid = PipelineParallelGrid(self._topo)

        assert self._grid.get_pipeline_model_parallel_rank() == pp_rank
        assert self._grid.get_data_parallel_rank() == dcp_rank

        self.loss_fn = None

        self.tied_comms = {}
        self.activation_checkpoint_interval = -1
        self.dynamic_shape = False

        layer_indices = sorted(self.module.get_decoder().layers.keys())
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
        self.cp_size = mpu.get_context_parallel_world_size()

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
        loss_scaling_factor: float = 1.0,
    ):
        model_inputs = {}
        for k, v in batches[batch_index].items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
            model_inputs[k] = v

        logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, forward batch {batch_index + 1}")
        if self.is_first_stage:
            assert "input_ids" in model_inputs
        else:
            model_inputs.pop("input_ids", None)
            assert self.fwd_recv_buffer is not None
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
            loss = outputs.loss * (loss_scaling_factor * self.cp_size)
            self.output_queue.put(loss)
            loss = loss.clone().detach()
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
            logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, send output: {outputs.last_hidden_state.shape}")

        return loss, send_ops

    @torch.cuda.nvtx.range("backward")
    def backward_one_chunk(self, batch_index: int):
        assert not self.output_queue.empty()
        output = self.output_queue.get()

        logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, backward batch {batch_index + 1}")
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
            logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, send grad: {grad_input.shape}")

        return send_ops

    def _shape_inference(self, batch: Dict[str, Any]):
        if "input_ids" in batch:
            shape = list(batch["input_ids"].shape)
        elif "labels" in batch:
            shape = list(batch["labels"].shape)
        elif "cu_seq_lens_q" in batch:
            shape = [1, batch["cu_seq_lens_q"][-1]]
        else:
            raise RuntimeError("Cannot infer shape from batch")

        cp_size = mpu.get_context_parallel_world_size()
        assert shape[1] % cp_size == 0
        shape[1] = shape[1] // cp_size

        return tuple(shape)

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
            logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, receive input: {self.fwd_recv_buffer.shape}")
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
            logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, receive grad: {self.bwd_recv_buffer.shape}")
        return ops


def _batch_isend_irecv(ops: List[dist.P2POp]) -> List[dist.Work]:
    if len(ops) == 0:
        return []
    return dist.batch_isend_irecv(ops)


class ScheduleNoPipelining(object):
    def __init__(
        self,
        stages: List[PipelineStage],
        deepspeed_engine: DeepSpeedEngine,
    ):
        assert isinstance(stages, (list, tuple)) and len(stages) > 0
        self.stages = stages
        self.deepspeed_engine = deepspeed_engine

    def step(self, batches: List[Dict[str, Any]]):
        self.deepspeed_engine.set_gradient_accumulation_boundary(is_boundary=False)
        losses = []
        for i in range(len(batches)):
            loss, _ = self.stages[0].forward_one_chunk(batches, batch_index=i)
            if i == len(batches) - 1:
                self.deepspeed_engine.set_gradient_accumulation_boundary(is_boundary=True)
            self.stages[0].backward_one_chunk(batch_index=i)
            losses.append(loss)
        return torch.stack(losses).to(torch.float32)


class BasePipelineSchedule(ScheduleNoPipelining, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.using_bf16_optimizer = type(self.deepspeed_engine.optimizer) is BF16_Optimizer
        self.deepspeed_engine.enable_backward_allreduce = False
        assert self.deepspeed_engine.zero_optimization_stage() < ZeroStageEnum.gradients, (
            "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"
        )

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
        loss_scaling_factor = 1 / len(batches)
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

            loss, fwd_sends = stage.forward_one_chunk(
                batches,
                batch_index=fwd_mb_index,
                loss_scaling_factor=loss_scaling_factor,
            )

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

            loss, fwd_sends = stage.forward_one_chunk(
                batches,
                batch_index=fwd_mb_index,
                loss_scaling_factor=loss_scaling_factor,
            )

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
            losses = torch.stack(losses).to(torch.float32) * len(batches)
        else:
            losses = torch.zeros(len(losses), device=stage.device, dtype=torch.float32)

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


def gather_pp_params(state_dict: Dict[str, torch.Tensor]):
    if mpu.get_data_parallel_rank() != 0:
        torch.distributed.barrier()
        return state_dict

    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    dtype = list(state_dict.values())[0].dtype

    if pp_rank == 0:
        for i in range(1, pp_size):
            param_shapes = [None]
            torch.distributed.recv_object_list(
                param_shapes,
                group=pp_group,
                group_src=i,
            )

            for param_name, shape in param_shapes[0].items():
                param = torch.empty(shape, dtype=dtype, device="cuda")
                torch.distributed.recv(
                    param,
                    group=pp_group,
                    group_src=i,
                )
                state_dict[param_name] = param.cpu()

    else:
        param_shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
        torch.distributed.send_object_list(
            [param_shapes],
            group=pp_group,
            group_dst=0,
        )

        for param_name in param_shapes:
            torch.distributed.send(
                state_dict[param_name].cuda(),
                group=pp_group,
                dst=0,
            )

    torch.distributed.barrier()
    return state_dict
