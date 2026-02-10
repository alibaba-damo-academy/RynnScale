import torch

try:
    from deep_ep import Buffer

    Buffer.set_num_sms(24)
except Exception:
    Buffer = None


_buffer = None


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor):
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


class DeepEPDispatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        topk_idx,
        topk_weights,
        num_experts,
        group,
    ):
        buffer = get_buffer(group, get_hidden_bytes(x))

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            previous_event=None,
            async_finish=True,
            allocate_on_comm_stream=False,
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()

        ctx.group = group
        ctx.handle = handle

        return (recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle)

    @staticmethod
    def backward(
        ctx,
        grad_recv_x,
        grad_recv_topk_idx,
        grad_recv_topk_weights,
        grad_tokens_per_expert,
        grad_handle,
    ):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_recv_x))
        handle = ctx.handle

        combined_grad_x, combined_grad_recv_topk_weights, event = buffer.combine(
            grad_recv_x,
            handle,
            topk_weights=grad_recv_topk_weights,
            async_finish=False,
        )

        return combined_grad_x, None, combined_grad_recv_topk_weights, None, None


class DeepEPCombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        handle,
        group,
    ):
        buffer = get_buffer(group, get_hidden_bytes(x))

        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            async_finish=False,
            previous_event=None,
            allocate_on_comm_stream=False,
        )

        ctx.handle = handle
        ctx.group = group

        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_combined_x))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_combined_x,
            handle=ctx.handle,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, None


def deepep_dispatch(
    x,
    topk_idx,
    topk_weights,
    num_experts,
    group,
):
    return DeepEPDispatchFunction.apply(
        x,
        topk_idx,
        topk_weights,
        num_experts,
        group,
    )


def deepep_combine(
    x,
    handle,
    group,
):
    return DeepEPCombineFunction.apply(
        x,
        handle,
        group,
    )
