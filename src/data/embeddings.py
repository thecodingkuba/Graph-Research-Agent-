"""SciBERT node features, computed in resumable shards.

Pooling: mean over non-padding token states of the last hidden layer. For a
BERT model that has not been fine-tuned, the [CLS] vector is not trained to
summarize the sequence (it is only trained for next-sentence prediction), so
mean pooling generally gives more useful similarity structure.

Shard k holds rows [k * shard_size, (k + 1) * shard_size) of the final matrix,
so row i always corresponds to node i. A manifest pins the ordered ID list,
model and max length; if any of those change, stale shards are discarded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

MODEL_NAME = "allenai/scibert_scivocab_uncased"
EMBED_DIM = 768


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def paper_text(title: str, abstract: str) -> str:
    """Title + abstract, or title alone if the abstract is missing."""
    title, abstract = (title or "").strip(), (abstract or "").strip()
    if not abstract:
        return title
    sep = " " if title.endswith((".", "?", "!")) else ". "
    return f"{title}{sep}{abstract}"


class SciBertEncoder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None, max_length: int = 512):
        from transformers import AutoModel, AutoTokenizer

        self.device = pick_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state  # [B, T, 768]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled.float().cpu().numpy()


def _manifest(ids: list[str], model_name: str, max_length: int, shard_size: int) -> dict:
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return {"ids_sha256": digest, "num_nodes": len(ids), "model": model_name,
            "max_length": max_length, "shard_size": shard_size, "pooling": "mean", "dim": EMBED_DIM}


def compute_embeddings(
    ids: list[str],
    texts: list[str],
    cache_dir: Path,
    batch_size: int = 32,
    shard_size: int = 1024,
    max_length: int = 512,
    model_name: str = MODEL_NAME,
    device: str | None = None,
    encoder=None,
) -> np.ndarray:
    """Return a float32 [len(ids), 768] matrix, row i = embedding of ids[i]."""
    assert len(ids) == len(texts)
    cache_dir = Path(cache_dir)
    manifest = _manifest(ids, model_name, max_length, shard_size)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        log.warning("Embedding cache does not match current dataset/config; recomputing.")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    num_shards = (len(ids) + shard_size - 1) // shard_size
    todo = [k for k in range(num_shards) if not (cache_dir / f"shard_{k:05d}.npy").exists()]
    if todo:
        encoder = encoder or SciBertEncoder(model_name, device, max_length)
        log.info("Embedding %d/%d shards on %s", len(todo), num_shards, getattr(encoder, "device", "?"))
        for k in tqdm(todo, desc="SciBERT shards"):
            lo, hi = k * shard_size, min((k + 1) * shard_size, len(ids))
            shard_texts = texts[lo:hi]
            # Length-sorted batches reduce padding; `out[sel]` restores node order.
            order = np.argsort([-len(t) for t in shard_texts], kind="stable")
            out = np.empty((hi - lo, EMBED_DIM), dtype=np.float32)
            for b in range(0, len(order), batch_size):
                sel = order[b : b + batch_size]
                out[sel] = encoder.encode([shard_texts[i] for i in sel])
            tmp = cache_dir / f"shard_{k:05d}.tmp.npy"
            np.save(tmp, out)
            tmp.replace(cache_dir / f"shard_{k:05d}.npy")  # atomic: partial shards never count as done

    x = np.concatenate([np.load(cache_dir / f"shard_{k:05d}.npy") for k in range(num_shards)])
    assert x.shape == (len(ids), EMBED_DIM), x.shape
    return x
