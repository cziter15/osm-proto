import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .data import mixed_random_batch, random_batch, random_batch_with_mask


def _autocast(device: str, precision: str):
    if device == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _weighted_cross_entropy(logits, targets, vocab_size, weights=None):
    token_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    if weights is None:
        return token_loss.mean()

    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("loss weights contain no active targets")
    return (token_loss * weights).sum() / denominator


@torch.no_grad()
def evaluate(
    model,
    data,
    batch_size,
    context,
    vocab_size=None,
    batches=8,
    device="cpu",
    precision="fp32",
    loss_mask=None,
    boundary_token_id=None,
):
    """Evaluate Q-OSM on random held-out windows."""
    if vocab_size is None:
        vocab_size = model.lm_head.out_features

    was_training = model.training
    model.eval()
    losses = []

    for _ in range(batches):
        if loss_mask is None:
            x, y = random_batch(data, batch_size, context, device, boundary_token_id)
            weights = None
        else:
            x, y, weights = random_batch_with_mask(
                data,
                loss_mask,
                batch_size,
                context,
                device,
                boundary_token_id,
            )

        with _autocast(device, precision):
            logits = model(x)
            loss = _weighted_cross_entropy(logits, y, vocab_size, weights)
        losses.append(loss.item())

    model.train(was_training)
    return sum(losses) / len(losses)


def train(
    model,
    train_data,
    val_data,
    *,
    steps,
    batch_size,
    context,
    lr=2e-3,
    min_lr=1e-4,
    weight_decay=0.01,
    device="cpu",
    eval_every=200,
    resume_state=None,
    checkpoint_callback=None,
    precision="fp32",
    grad_accum_steps=1,
    eval_batch_size=None,
    val_sources=None,
    train_sources=None,
    source_sampling_alpha=1.0,
    selection_metric="combined",
    val_loss_mask=None,
    boundary_token_id=None,
):
    """Train the current QuantumOSMForCausalLM model."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if batch_size <= 0 or context <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch_size, context and grad_accum_steps must be positive")

    model.to(device)
    vocab_size = model.lm_head.out_features

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        fused=(device == "cuda"),
    )

    start_step = 0
    best_val = float("inf")
    best_state = None
    history = []

    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_step = int(resume_state.get("step", 0))
        best_val = float(resume_state.get("best_val", best_val))
        history = list(resume_state.get("history", []))
        best_state = resume_state.get("best_model_state_dict")

        if resume_state.get("selection_metric", "combined") != selection_metric:
            best_val = float("inf")
            best_state = None

    schedule_end_step = (
        int(resume_state.get("schedule_end_step", 0))
        if resume_state
        else 0
    )

    if (
        resume_state
        and "scheduler_state_dict" in resume_state
        and steps <= schedule_end_step
    ):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=min_lr,
        )
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["initial_lr"] = lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, steps - start_step),
            eta_min=min_lr,
        )

    for step in range(start_step + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        for _ in range(grad_accum_steps):
            if train_sources:
                x, y, loss_weights = mixed_random_batch(
                    train_sources,
                    batch_size,
                    context,
                    device,
                    alpha=source_sampling_alpha,
                    boundary_token_id=boundary_token_id,
                )
            else:
                x, y = random_batch(
                    train_data,
                    batch_size,
                    context,
                    device,
                    boundary_token_id,
                )
                loss_weights = None

            with _autocast(device, precision):
                logits = model(x)
                raw_loss = _weighted_cross_entropy(
                    logits,
                    y,
                    vocab_size,
                    loss_weights,
                )
                loss = raw_loss / grad_accum_steps

            loss.backward()
            train_loss += raw_loss.item() / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 and step % eval_every != 0:
            print(
                {
                    "step": step,
                    "steps_total": steps,
                    "train_ce": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                },
                flush=True,
            )

        if step == 1 or step % eval_every == 0 or step == steps:
            eval_bs = eval_batch_size or batch_size
            val = evaluate(
                model,
                val_data,
                eval_bs,
                context,
                vocab_size=vocab_size,
                device=device,
                precision=precision,
                loss_mask=val_loss_mask,
                boundary_token_id=boundary_token_id,
            )

            record = {
                "step": step,
                "train_ce": train_loss,
                "val_ce": val,
                "bits_per_token": val / math.log(2),
            }

            if val_sources:
                record["val_by_source"] = {
                    name: evaluate(
                        model,
                        item["data"] if isinstance(item, dict) else item,
                        eval_bs,
                        context,
                        vocab_size=vocab_size,
                        batches=2,
                        device=device,
                        precision=precision,
                        loss_mask=(
                            item.get("loss_mask")
                            if isinstance(item, dict)
                            else None
                        ),
                        boundary_token_id=boundary_token_id,
                    )
                    for name, item in val_sources.items()
                    if len(item["data"] if isinstance(item, dict) else item)
                    > context + 1
                }
                if record["val_by_source"]:
                    record["macro_val_ce"] = sum(
                        record["val_by_source"].values()
                    ) / len(record["val_by_source"])

            history.append(record)
            print(record, flush=True)

            selection_value = (
                record.get("macro_val_ce", val)
                if selection_metric == "macro"
                else val
            )

            if selection_value < best_val:
                best_val = selection_value
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "step": step,
                        "best_val": best_val,
                        "history": history,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "schedule_end_step": steps,
                        "selection_metric": selection_metric,
                        "best_model_state_dict": best_state,
                    }
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val
