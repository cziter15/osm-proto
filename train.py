import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from .data import mixed_random_batch, random_batch, random_batch_with_mask


def _autocast(device, precision):
    if device == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, data, batch_size, context, vocab_size, batches=8, device="cpu",
             precision="fp32", loss_mask=None):
    model.eval()
    losses = []
    for _ in range(batches):
        if loss_mask is None:
            x, y = random_batch(data, batch_size, context, device)
            weights = None
        else:
            x, y, weights = random_batch_with_mask(data, loss_mask, batch_size, context, device)
        with _autocast(device, precision):
            logits = model(x)
            token_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                         reduction="none").view_as(y)
            loss = token_loss.mean() if weights is None else (token_loss * weights).sum() / weights.sum()
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, *, steps, batch_size, context, lr=2e-3,
          min_lr=1e-4, weight_decay=0.01, device="cpu", eval_every=200,
          resume_state=None, checkpoint_callback=None, precision="fp32",
          grad_accum_steps=1, eval_batch_size=None, val_sources=None,
          train_sources=None, source_sampling_alpha=1.0,
          selection_metric="combined", val_loss_mask=None):
    model.to(device)
    vocab_size = model.lm_head.out_features
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            fused=(device == "cuda"))
    start_step = 0
    best_val = float("inf")
    best_state = None
    history = []

    if resume_state:
        opt.load_state_dict(resume_state["optimizer_state_dict"])
        start_step = int(resume_state.get("step", 0))
        best_val = float(resume_state.get("best_val", best_val))
        history = list(resume_state.get("history", []))
        best_state = resume_state.get("best_model_state_dict")
        if resume_state.get("selection_metric", "combined") != selection_metric:
            best_val = float("inf")
            best_state = None
    schedule_end_step = int(resume_state.get("schedule_end_step", 0)) if resume_state else 0
    if resume_state and "scheduler_state_dict" in resume_state and steps <= schedule_end_step:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=min_lr)
        sched.load_state_dict(resume_state["scheduler_state_dict"])
    else:
        # A longer continuation is a new cosine stage, starting at the requested LR.
        for group in opt.param_groups:
            group["lr"] = lr
            group["initial_lr"] = lr
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, steps - start_step), eta_min=min_lr
        )

    for step in range(start_step + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(grad_accum_steps):
            if train_sources:
                x, y, loss_weights = mixed_random_batch(train_sources, batch_size, context, device,
                                                        alpha=source_sampling_alpha)
            else:
                x, y = random_batch(train_data, batch_size, context, device)
                loss_weights = None
            with _autocast(device, precision):
                logits = model(x)
                token_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                             reduction="none").view_as(y)
                raw_loss = (token_loss.mean() if loss_weights is None else
                            (token_loss * loss_weights).sum() / loss_weights.sum())
                loss = raw_loss / grad_accum_steps
            loss.backward()
            train_loss += raw_loss.item() / grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            eval_bs = eval_batch_size or batch_size
            val = evaluate(model, val_data, eval_bs, context, vocab_size, device=device,
                           precision=precision, loss_mask=val_loss_mask)
            record = {"step": step, "train_ce": train_loss, "val_ce": val,
                      "bits_per_token": val / math.log(2)}
            if val_sources:
                record["val_by_source"] = {
                    name: evaluate(model, item["data"] if isinstance(item, dict) else item,
                                   eval_bs, context, vocab_size, batches=2, device=device,
                                   precision=precision,
                                   loss_mask=item.get("loss_mask") if isinstance(item, dict) else None)
                    for name, item in val_sources.items()
                    if len(item["data"] if isinstance(item, dict) else item) > context + 1
                }
                record["macro_val_ce"] = sum(record["val_by_source"].values()) / len(record["val_by_source"])
            history.append(record)
            print(history[-1])
            selection_value = record.get("macro_val_ce", val) if selection_metric == "macro" else val
            if selection_value < best_val:
                best_val = selection_value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "step": step,
                    "best_val": best_val,
                    "history": history,
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "schedule_end_step": steps,
                    "selection_metric": selection_metric,
                    "best_model_state_dict": best_state,
                })

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val
