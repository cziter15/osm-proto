from pathlib import Path
from typing import Sequence
import torch
import sentencepiece as spm


def train_sentencepiece(input_path: str, prefix: str, vocab_size: int = 2048):
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        user_defined_symbols=["<|document|>", "<|story|>", "<|instruction|>",
                              "<|input|>", "<|response|>", "<|end|>", "<|code|>"],
    )


def load_bpe_dataset(text_paths: str | Sequence[str], model_path: str, split: float = 0.95,
                     return_sources: bool = False, return_loss_masks: bool = False):
    if isinstance(text_paths, str):
        text_paths = [text_paths]
    texts = [Path(path).read_text(encoding="utf-8", errors="replace") for path in text_paths]
    sp = spm.SentencePieceProcessor(model_file=model_path)
    separator = torch.tensor(sp.encode("\n\n<|document|>\n\n", out_type=int), dtype=torch.long)
    train_parts = []
    val_parts = []
    train_mask_parts = []
    val_mask_parts = []
    source_splits = {}
    for path, text in zip(text_paths, texts):
        ids = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
        loss_mask = _response_loss_mask(ids, sp)
        cut = int(split * len(ids))
        train_ids, val_ids = ids[:cut], ids[cut:]
        train_mask, val_mask = loss_mask[:cut], loss_mask[cut:]
        train_parts.extend((train_ids, separator))
        val_parts.extend((val_ids, separator))
        separator_mask = torch.zeros_like(separator, dtype=torch.float32)
        train_mask_parts.extend((train_mask, separator_mask))
        val_mask_parts.extend((val_mask, separator_mask))
        source_splits[Path(path).stem] = {
            "train": train_ids, "val": val_ids,
            "train_loss_mask": train_mask, "val_loss_mask": val_mask,
        }
    text = "\n\n<|document|>\n\n".join(texts)
    train_data = torch.cat(train_parts[:-1])
    val_data = torch.cat(val_parts[:-1])
    train_loss_mask = torch.cat(train_mask_parts[:-1])
    val_loss_mask = torch.cat(val_mask_parts[:-1])
    result = (text, sp, train_data, val_data)
    if return_loss_masks:
        result += (train_loss_mask, val_loss_mask)
    return result + (source_splits,) if return_sources else result


def _response_loss_mask(ids: torch.Tensor, sp) -> torch.Tensor:
    """Use response-only targets when instruction markers occur, else all targets."""
    response_id = sp.piece_to_id("<|response|>")
    end_id = sp.piece_to_id("<|end|>")
    if response_id == sp.unk_id() or end_id == sp.unk_id() or not torch.any(ids == response_id):
        return torch.ones(len(ids), dtype=torch.float32)
    mask = torch.zeros(len(ids), dtype=torch.float32)
    responses = torch.nonzero(ids == response_id, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(ids == end_id, as_tuple=False).flatten().tolist()
    end_index = 0
    for start in responses:
        while end_index < len(ends) and ends[end_index] <= start:
            end_index += 1
        stop = ends[end_index] if end_index < len(ends) else len(ids) - 1
        mask[start + 1:stop + 1] = 1.0
    return mask


def random_batch(data, batch_size: int, context: int, device="cpu"):
    ix = torch.randint(0, len(data) - context - 1, (batch_size,))
    x = torch.stack([data[i:i+context] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+context+1] for i in ix]).to(device)
    return x, y


def random_batch_with_mask(data, loss_mask, batch_size: int, context: int, device="cpu"):
    rows = []
    while len(rows) < batch_size:
        candidates = torch.randint(0, len(data) - context - 1, (max(8, batch_size - len(rows)),))
        for index in candidates.tolist():
            weights = loss_mask[index + 1:index + context + 1]
            if weights.sum() > 0:
                rows.append((data[index:index + context], data[index + 1:index + context + 1], weights))
                if len(rows) == batch_size:
                    break
    x = torch.stack([row[0] for row in rows]).to(device)
    y = torch.stack([row[1] for row in rows]).to(device)
    weights = torch.stack([row[2] for row in rows]).to(device)
    return x, y, weights


def mixed_random_batch(sources: dict, batch_size: int, context: int,
                       device="cpu", alpha: float = 1.0):
    """Sample a batch across sources using p(source) proportional to tokens**alpha."""
    eligible = []
    for name, item in sources.items():
        data = item["data"] if isinstance(item, dict) else item
        if len(data) > context + 1:
            eligible.append((name, item))
    if not eligible:
        raise ValueError("no source is long enough for the requested context")
    weights = torch.tensor([
        len(item["data"] if isinstance(item, dict) else item) ** alpha
        for _, item in eligible
    ], dtype=torch.float64)
    assignments = torch.multinomial(weights, batch_size, replacement=True)
    xs, ys, loss_weights = [], [], []
    for source_index, (_, item) in enumerate(eligible):
        count = int((assignments == source_index).sum())
        if count:
            data = item["data"] if isinstance(item, dict) else item
            mask = item.get("loss_mask") if isinstance(item, dict) else None
            if mask is not None:
                x, y, batch_weights = random_batch_with_mask(data, mask, count, context, device)
            else:
                x, y = random_batch(data, count, context, device)
                batch_weights = torch.ones_like(y, dtype=torch.float32)
            xs.append(x)
            ys.append(y)
            loss_weights.append(batch_weights)
    x, y = torch.cat(xs), torch.cat(ys)
    batch_weights = torch.cat(loss_weights)
    order = torch.randperm(batch_size, device=x.device)
    return x[order], y[order], batch_weights[order]
