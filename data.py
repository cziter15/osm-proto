from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch


_ACTIVE_INDEX_CACHE = {}
_BOUNDARY_INDEX_CACHE = {}
SPECIAL_TOKENS = [
    "<|document|>",
    "<|story|>",
    "<|instruction|>",
    "<|input|>",
    "<|response|>",
    "<|end|>",
    "<|code|>",
]


def train_sentencepiece(input_path: str, prefix: str, vocab_size: int = 2048):
    """Train the BPE tokenizer used by Q-OSM."""
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
        user_defined_symbols=SPECIAL_TOKENS,
    )


def _response_loss_mask(ids: torch.Tensor, sp) -> torch.Tensor:
    """Use response-only targets for instruction data; otherwise use all targets."""
    response_id = sp.piece_to_id("<|response|>")
    end_id = sp.piece_to_id("<|end|>")

    if (
        response_id == sp.unk_id()
        or end_id == sp.unk_id()
        or not torch.any(ids == response_id)
    ):
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


def _encode_file_chunked(path: str, sp, chunk_chars: int = 4_000_000) -> torch.Tensor:
    """Encode a large text file without materializing the full string/list."""
    parts = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(chunk_chars)
            if not chunk:
                break
            encoded = sp.encode(chunk, out_type=int)
            if encoded:
                parts.append(torch.tensor(encoded, dtype=torch.long))
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(parts)


def load_bpe_dataset(
    text_paths: str | Sequence[str],
    model_path: str,
    split: float = 0.95,
    return_sources: bool = False,
    return_loss_masks: bool = False,
):
    """Tokenize one or more corpora and create train/validation splits."""
    if isinstance(text_paths, str):
        text_paths = [text_paths]
    if not text_paths:
        raise ValueError("text_paths cannot be empty")
    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1")

    sp = spm.SentencePieceProcessor(model_file=model_path)
    separator = torch.tensor(
        sp.encode("\n\n<|document|>\n\n", out_type=int),
        dtype=torch.long,
    )
    separator_mask = torch.zeros_like(separator, dtype=torch.float32)

    train_parts = []
    val_parts = []
    train_mask_parts = []
    val_mask_parts = []
    source_splits = {}

    for path in text_paths:
        ids = _encode_file_chunked(path, sp)
        if len(ids) < 2:
            raise ValueError(f"dataset is too short after tokenization: {path}")

        loss_mask = _response_loss_mask(ids, sp)
        cut = int(split * len(ids))
        train_ids, val_ids = ids[:cut], ids[cut:]
        train_mask, val_mask = loss_mask[:cut], loss_mask[cut:]

        train_parts.extend((train_ids, separator))
        val_parts.extend((val_ids, separator))
        train_mask_parts.extend((train_mask, separator_mask))
        val_mask_parts.extend((val_mask, separator_mask))

        source_splits[Path(path).stem] = {
            "train": train_ids,
            "val": val_ids,
            "train_loss_mask": train_mask,
            "val_loss_mask": val_mask,
        }

    # The training loop only needs token tensors; avoid retaining multi-GB text.
    text = ""
    train_data = torch.cat(train_parts[:-1])
    val_data = torch.cat(val_parts[:-1])
    train_loss_mask = torch.cat(train_mask_parts[:-1])
    val_loss_mask = torch.cat(val_mask_parts[:-1])

    result = (text, sp, train_data, val_data)
    if return_loss_masks:
        result += (train_loss_mask, val_loss_mask)
    if return_sources:
        result += (source_splits,)
    return result


def random_batch(
    data: torch.Tensor,
    batch_size: int,
    context: int,
    device: str = "cpu",
    boundary_token_id: int | None = None,
):
    """Sample causal next-token windows."""
    n = len(data) - context - 1
    if n <= 0:
        raise ValueError("data is shorter than context")

    if boundary_token_id is None:
        ix = torch.randint(0, n, (batch_size,), device=data.device)
    else:
        key = (data.data_ptr(), data.numel(), int(boundary_token_id))
        boundaries = _BOUNDARY_INDEX_CACHE.get(key)
        if boundaries is None or boundaries.device != data.device:
            boundaries = torch.nonzero(data == boundary_token_id, as_tuple=False).flatten()
            _BOUNDARY_INDEX_CACHE[key] = boundaries
        ix = torch.randint(0, n, (batch_size,), device=data.device)
        # Rejection sampling avoids allocating a multi-GB valid-start mask.
        for _ in range(12):
            left = torch.searchsorted(boundaries, ix, right=False)
            right = torch.searchsorted(boundaries, ix + context, right=True)
            bad = left < right
            if not bad.any():
                break
            ix[bad] = torch.randint(0, n, (int(bad.sum()),), device=data.device)
        if bad.any():
            raise RuntimeError("could not sample a window without crossing a document boundary")
    offsets = torch.arange(context, device=data.device)
    x = data[ix[:, None] + offsets]
    y = data[ix[:, None] + offsets + 1]

    target_device = torch.device(device)
    if data.device != target_device:
        x = x.to(target_device)
        y = y.to(target_device)

    return x, y


def random_batch_with_mask(
    data: torch.Tensor,
    loss_mask: torch.Tensor,
    batch_size: int,
    context: int,
    device: str = "cpu",
    boundary_token_id: int | None = None,
):
    """Sample windows centered around active loss targets."""
    if len(data) <= context + 1:
        raise ValueError("data is shorter than context")

    if loss_mask.device != data.device:
        loss_mask = loss_mask.to(data.device)

    cache_key = (data.data_ptr(), loss_mask.data_ptr(), loss_mask.numel())
    active = _ACTIVE_INDEX_CACHE.get(cache_key)
    if active is None or active.device != data.device:
        active = torch.nonzero(loss_mask > 0, as_tuple=False).flatten()
        _ACTIVE_INDEX_CACHE[cache_key] = active

    if active.numel() == 0:
        raise ValueError("loss mask contains no active target tokens")

    max_start = len(data) - context - 1
    target = active[torch.randint(0, active.numel(), (batch_size,), device=active.device)]
    starts = (target - context // 2).clamp(min=0, max=max_start)
    starts = torch.where(
        (target <= starts) & (starts < max_start),
        starts + 1,
        starts,
    )
    if boundary_token_id is not None:
        key = (data.data_ptr(), data.numel(), int(boundary_token_id))
        boundaries = _BOUNDARY_INDEX_CACHE.get(key)
        if boundaries is None or boundaries.device != data.device:
            boundaries = torch.nonzero(data == boundary_token_id, as_tuple=False).flatten()
            _BOUNDARY_INDEX_CACHE[key] = boundaries
        for _ in range(12):
            left = torch.searchsorted(boundaries, starts, right=False)
            right = torch.searchsorted(boundaries, starts + context, right=True)
            bad = left < right
            if not bad.any():
                break
            target[bad] = active[torch.randint(0, active.numel(), (int(bad.sum()),), device=active.device)]
            starts[bad] = (target[bad] - context // 2).clamp(min=0, max=max_start)
        if bad.any():
            raise RuntimeError("could not sample a masked window without crossing a document boundary")

    offsets = torch.arange(context, device=data.device)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    weights = loss_mask[starts[:, None] + offsets + 1]

    target_device = torch.device(device)
    if data.device != target_device:
        x = x.to(target_device)
        y = y.to(target_device)
        weights = weights.to(target_device)

    return x, y, weights


def mixed_random_batch(
    sources: dict,
    batch_size: int,
    context: int,
    device: str = "cpu",
    alpha: float = 1.0,
    boundary_token_id: int | None = None,
):
    """Sample across corpora with p(source) proportional to tokens**alpha."""
    eligible = []
    for name, item in sources.items():
        data = item["data"] if isinstance(item, dict) else item
        if len(data) > context + 1:
            eligible.append((name, item))

    if not eligible:
        raise ValueError("no source is long enough for the requested context")

    source_weights = torch.tensor(
        [
            len(item["data"] if isinstance(item, dict) else item) ** alpha
            for _, item in eligible
        ],
        dtype=torch.float64,
    )
    assignments = torch.multinomial(source_weights, batch_size, replacement=True)

    xs = []
    ys = []
    loss_weights = []

    for source_index, (_, item) in enumerate(eligible):
        count = int((assignments == source_index).sum())
        if count == 0:
            continue

        source_data = item["data"] if isinstance(item, dict) else item
        mask = item.get("loss_mask") if isinstance(item, dict) else None

        if mask is None:
            x, y = random_batch(source_data, count, context, device, boundary_token_id)
            batch_weights = torch.ones_like(y, dtype=torch.float32)
        else:
            x, y, batch_weights = random_batch_with_mask(
                source_data,
                mask,
                count,
                context,
                device,
                boundary_token_id,
            )

        xs.append(x)
        ys.append(y)
        loss_weights.append(batch_weights)

    x = torch.cat(xs)
    y = torch.cat(ys)
    batch_weights = torch.cat(loss_weights)
    order = torch.randperm(batch_size, device=x.device)
    return x[order], y[order], batch_weights[order]
