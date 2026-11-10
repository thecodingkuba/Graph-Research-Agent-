"""Build, save and load the PyG citation graph plus its aligned metadata.

Files in the processed directory:
  graph.pt          torch_geometric.data.Data (see build_graph for attributes)
  papers.parquet    one row per node, row i == node i (node_idx column)
  edges.parquet     citing/cited node indices and OpenAlex IDs
  id_to_idx.json    OpenAlex ID -> node index
  label_vocab.json  integer label -> name for field / subfield / topic
  dataset_info.json build config and statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

LABEL_COLUMNS = {"field": "primary_field", "subfield": "primary_subfield", "topic": "primary_topic"}


def make_vocab(values: pd.Series) -> list[str]:
    """Sorted list of non-empty names; label = position, -1 = unknown."""
    return sorted({v for v in values if v})


def encode_labels(values: pd.Series, vocab: list[str]) -> torch.Tensor:
    lookup = {name: i for i, name in enumerate(vocab)}
    return torch.tensor([lookup.get(v, -1) for v in values], dtype=torch.long)


def build_graph(papers: pd.DataFrame, edges: np.ndarray, x: np.ndarray) -> tuple[Data, dict]:
    """Attributes:
    x          [N, 768] float32 SciBERT embeddings
    edge_index [2, E]   row 0 = citing paper, row 1 = cited paper (A cites B => A -> B)
    year       [N]      publication year
    edge_year  [E]      year of the citing paper, i.e. when the edge appeared
    field / subfield / topic  [N] integer labels of the OpenAlex primary topic (-1 = missing)
    """
    assert len(papers) == x.shape[0], "one embedding per paper"
    assert (papers["node_idx"].to_numpy() == np.arange(len(papers))).all(), "papers must be in node order"
    edge_index = torch.from_numpy(np.ascontiguousarray(edges.T)).long().reshape(2, -1)
    year = torch.tensor(papers["year"].to_numpy(), dtype=torch.long)
    vocab = {name: make_vocab(papers[col]) for name, col in LABEL_COLUMNS.items()}
    data = Data(
        x=torch.from_numpy(x).float(),
        edge_index=edge_index,
        year=year,
        edge_year=year[edge_index[0]],
        num_nodes=len(papers),
        **{name: encode_labels(papers[col], vocab[name]) for name, col in LABEL_COLUMNS.items()},
    )
    return data, vocab


def save_dataset(out_dir: Path, data: Data, papers: pd.DataFrame, vocab: dict, info: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / "graph.pt")
    papers.to_parquet(out_dir / "papers.parquet", index=False)
    ids = papers["openalex_id"].to_numpy()
    src, dst = data.edge_index.numpy()
    pd.DataFrame({"src_idx": src, "dst_idx": dst, "src_id": ids[src], "dst_id": ids[dst],
                  "edge_year": data.edge_year.numpy()}).to_parquet(out_dir / "edges.parquet", index=False)
    (out_dir / "id_to_idx.json").write_text(json.dumps(dict(zip(ids.tolist(), range(len(ids))))))
    (out_dir / "label_vocab.json").write_text(json.dumps(vocab, indent=1))
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, default=str))


def load_dataset(out_dir: Path) -> tuple[Data, pd.DataFrame]:
    """Entry point for training scripts: returns (PyG Data, papers metadata in node order)."""
    out_dir = Path(out_dir)
    data = torch.load(out_dir / "graph.pt", weights_only=False)
    papers = pd.read_parquet(out_dir / "papers.parquet")
    return data, papers
