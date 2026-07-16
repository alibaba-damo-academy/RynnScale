import torch
import torch.nn.functional as F


def cross_entropy_loss(
    hidden_states,
    lm_head,
    position_ids,
    labels,
    num_items_in_batch,
    sequence_splitter,
    loss_reduction_scope,
    loss_implementation,
):
    batch_size = hidden_states.size(0)

    shift_labels = F.pad(labels[..., 1:], (0, 1), value=-100)
    global_mask = shift_labels >= 0
    mask = global_mask[:, sequence_splitter]

    if mask.sum() == 0:
        logits = lm_head(hidden_states[:, :1])
        loss = 0.0 * logits.mean()
        return loss

    hidden_states = hidden_states[mask].contiguous()
    shift_labels = shift_labels[:, sequence_splitter][mask].contiguous()

    if num_items_in_batch is None:
        reduction = "mean"
        denominator = None

    elif loss_reduction_scope == "batch":
        reduction = "sum"
        denominator = num_items_in_batch

    elif loss_reduction_scope == "sequence":
        reduction = "none"

        if batch_size == 1:
            # NOTE: packed sequence
            if position_ids.ndim == 3:
                position_ids = position_ids[0]
            start_indices = torch.nonzero(position_ids[0] == 0)[:, 0]
            end_indices = F.pad(start_indices[1:], (0, 1), value=position_ids.size(1))
            batch_indices = torch.cat(
                [
                    torch.full(
                        (e - s,),
                        fill_value=i,
                        device=position_ids.device,
                        dtype=torch.long,
                    )
                    for i, (s, e) in enumerate(zip(start_indices, end_indices))
                ],
            ).unsqueeze(0)
        else:
            batch_indices = torch.arange(batch_size, device=position_ids.device)
            batch_indices = batch_indices.unsqueeze(1).expand(-1, hidden_states.size(1))

        num_tokens = F.one_hot(batch_indices[global_mask]).sum(dim=0)
        batch_indices = batch_indices[:, sequence_splitter][mask]
        denominator = num_tokens[batch_indices] * num_items_in_batch

    else:
        raise ValueError(f"Unknown reduction scope: {loss_reduction_scope}")

    if loss_implementation == "torch":
        logits = lm_head(hidden_states)
        loss = torch.nn.functional.cross_entropy(
            logits.float(),
            shift_labels,
            reduction=reduction,
        )
    elif loss_implementation == "cce":
        from cut_cross_entropy import linear_cross_entropy

        loss = linear_cross_entropy(
            hidden_states,
            lm_head.weight,
            shift_labels,
            bias=lm_head.bias,
            reduction=reduction,
            accum_e_fp32=True,
            accum_c_fp32=True,
        )
    else:
        raise ValueError(f"Unkown loss implementation: {loss_implementation}")

    if denominator is not None:
        loss = loss / denominator
        if loss.ndim > 0:
            loss = loss.sum()

    return loss
