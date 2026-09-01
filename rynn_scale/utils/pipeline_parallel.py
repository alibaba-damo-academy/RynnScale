from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor

from .. import parallel_state as mpu
from ..utils import logging

logger = logging.get_logger(__name__)


class PipelineStage(object):
    def __init__(
        self,
        module: FSDPModule,
        group: torch.distributed.ProcessGroup,
        dtype: torch.dtype | None = None,
    ):
        self.module = module
        self.group = group
        # P2P recv buffers must match the sent activation dtype (FSDP mp_policy
        # param_dtype), which differs from module.dtype (fp32 master params).
        self.dtype = dtype if dtype is not None else module.dtype

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

    def set_requires_gradient_sync(self, requires_gradient_sync: bool):
        self.module.set_requires_gradient_sync(requires_gradient_sync)

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
            assert self.fwd_recv_buffer is not None
            self.fwd_recv_buffer.requires_grad_(True)
            self.input_queue.put(self.fwd_recv_buffer)
            model_inputs["inputs_embeds"] = self.fwd_recv_buffer
            # MTP on the last stage re-embeds shifted input_ids, so keep them.
            mtp_enabled = getattr(self.module.config, "mtp_loss_weight", 0) > 0
            if not (self.is_last_stage and mtp_enabled):
                model_inputs.pop("input_ids", None)

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
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, send output: {outputs.last_hidden_state.shape}"
            )

        return loss, send_ops

    @torch.cuda.nvtx.range("backward")
    def backward_one_chunk(self, batch_index: int):
        assert not self.output_queue.empty()
        output = self.output_queue.get()

        logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, backward batch {batch_index + 1}")
        if self.is_last_stage:
            output.backward()
        else:
            assert self.bwd_recv_buffer is not None
            output.backward(self.bwd_recv_buffer)
            self.bwd_recv_buffer = None

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
                dtype=self.dtype,
                device=self.module.device,
            )
            ops.append(dist.P2POp(dist.irecv, self.fwd_recv_buffer, self.prev_stage_rank, self.group))
            logger.debug(
                f"stage {self.stage_index + 1}/{self.num_stages}, receive input: {self.fwd_recv_buffer.shape}"
            )
        return ops

    def get_bwd_recv_ops(self, batch: Dict[str, Any]) -> List[dist.P2POp]:
        ops = []
        if self.next_stage_rank is not None:
            shape = self._shape_inference(batch)
            self.bwd_recv_buffer = torch.empty(
                size=(*shape, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            ops.append(dist.P2POp(dist.irecv, self.bwd_recv_buffer, self.next_stage_rank, self.group))
            logger.debug(f"stage {self.stage_index + 1}/{self.num_stages}, receive grad: {self.bwd_recv_buffer.shape}")
        return ops

    def _has_pp_shared_embedding(self) -> bool:
        if self.num_stages <= 1:
            return False
        config = self.module.config
        tied = getattr(config, "tie_word_embeddings", False)
        mtp_enabled = getattr(config, "mtp_loss_weight", 0) > 0
        return tied or mtp_enabled

    def all_reduce_embedding_grads(self):
        if not self._has_pp_shared_embedding():
            return

        if not (self.is_first_stage or self.is_last_stage):
            return

        embeddings = self.module.get_input_embeddings()
        grad = embeddings.weight.grad
        if grad is None:
            return

        new_grad = grad.full_tensor()
        torch.distributed.all_reduce(new_grad, group=mpu.get_embedding_group())

        embeddings.weight.grad = distribute_tensor(
            new_grad,
            device_mesh=grad.device_mesh,
            placements=grad.placements,
        )

    def get_pp_shared_params(self) -> List[torch.nn.Parameter]:
        if not self._has_pp_shared_embedding():
            return []
        if not self.is_last_stage or self.is_first_stage:
            return []
        embeddings = self.module.get_input_embeddings()
        if embeddings is None or embeddings.weight is None:
            return []
        return [embeddings.weight]


def _batch_isend_irecv(ops: List[dist.P2POp]) -> List[dist.Work]:
    if len(ops) == 0:
        return []
    return dist.batch_isend_irecv(ops)


class BasePipelineSchedule(ABC):
    def __init__(self, stages: List[PipelineStage]):
        assert isinstance(stages, (list, tuple)) and len(stages) > 0
        self.stages = stages

    @abstractmethod
    def _step(self, batches: List[Dict[str, Any]]):
        pass

    def step(self, batches: List[Dict[str, Any]]):
        losses = self._step(batches)
        for stage in self.stages:
            stage.all_reduce_embedding_grads()
        return losses


class ScheduleNoPipelining(BasePipelineSchedule):
    def _step(self, batches: List[Dict[str, Any]]):
        loss_scaling_factor = 1 / len(batches)
        stage = self.stages[0]
        losses = []
        for i in range(len(batches)):
            stage.set_requires_gradient_sync(i == len(batches) - 1)
            loss, _ = stage.forward_one_chunk(
                batches,
                batch_index=i,
                loss_scaling_factor=loss_scaling_factor,
            )
            stage.backward_one_chunk(batch_index=i)
            losses.append(loss)
        return torch.stack(losses).to(torch.float32) * len(batches)


class ScheduleGPipe(BasePipelineSchedule):
    def _step(self, batches: List[Dict[str, Any]]):
        raise NotImplementedError


class Schedule1F1B(BasePipelineSchedule):
    def _step(self, batches: List[Dict[str, Any]]):
        loss_scaling_factor = 1 / len(batches)
        stage = self.stages[0]
        stage.set_requires_gradient_sync(False)
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

            if bwd_mb_index == len(batches) - 1:
                stage.set_requires_gradient_sync(True)

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

            if bwd_mb_index == len(batches) - 1:
                stage.set_requires_gradient_sync(True)

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
