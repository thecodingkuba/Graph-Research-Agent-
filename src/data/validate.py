"""Consistency checks and a human-readable summary for a built dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .embeddings import EMBED_DIM
from .graph import LABEL_COLUMNS


class ValidationError(AssertionError):
    pass


def validate(data: Data, papers: pd.DataFrame, vocab: dict | None = None) -> list[str]:
    """Raise ValidationError listing every failed check; return the passed checks."""
    errors, passed = [], []

    def check(ok: bool, msg: str):
        (passed if ok else errors).append(msg)

    n = data.num_nodes
    ei = data.edge_index
    check(len(papers) == n, f"metadata rows ({len(papers)}) == num_nodes ({n})")
    check((papers["node_idx"].to_numpy() == np.arange(len(papers))).all(), "metadata node_idx == 0..N-1")
    check(papers["openalex_id"].is_unique, "OpenAlex IDs unique (no duplicate papers)")
    check(tuple(data.x.shape) == (n, EMBED_DIM), f"x has shape [N, {EMBED_DIM}] (got {tuple(data.x.shape)})")
    check(bool(torch.isfinite(data.x).all()), "embeddings finite")
    check(bool((data.x.norm(dim=1) > 0).all()), "no all-zero embedding rows")
    check(ei.dim() == 2 and ei.size(0) == 2 and ei.dtype == torch.long, "edge_index is LongTensor [2, E]")
    in_range = ei.numel() == 0 or (int(ei.min()) >= 0 and int(ei.max()) < n)
    check(in_range, "edge indices within [0, N) (no edges to papers outside the dataset)")
    check(bool((ei[0] != ei[1]).all()), "no self-loops")
    check(torch.unique(ei, dim=1).size(1) == ei.size(1), "no duplicate edges")

    # Direction + completeness: edge (u, v) <=> paper u lists paper v in referenced_works.
    if in_range and len(papers) == n:
        ids = papers["openalex_id"].to_numpy()
        refs = papers["referenced_works"].tolist()
        id_set = set(ids)
        expected = {(ids[u], v) for u, rs in enumerate(refs) for v in rs if v in id_set and v != ids[u]}
        actual = {(ids[u], ids[v]) for u, v in ei.t().tolist()}
        check(actual <= expected, "every edge u->v has v in u.referenced_works (u cites v)")
        check(expected <= actual, "every in-dataset reference is present as an edge")

    check(torch.equal(data.year, torch.tensor(papers["year"].to_numpy(), dtype=torch.long)),
          "year tensor matches metadata")
    check(ei.numel() == 0 or torch.equal(data.edge_year, data.year[ei[0]]), "edge_year == year of citing paper")
    if vocab is not None:
        for name, col in LABEL_COLUMNS.items():
            decoded = [vocab[name][i] if i >= 0 else "" for i in data[name].tolist()]
            check(decoded == papers[col].tolist(), f"{name} labels decode to metadata '{col}'")

    if errors:
        raise ValidationError("Dataset validation failed:\n  - " + "\n  - ".join(errors))
    return passed


def summarize(data: Data, papers: pd.DataFrame, num_examples: int = 3, seed: int = 0) -> str:
    ei = data.edge_index
    src, dst = ei
    field = data.field
    known = (field[src] >= 0) & (field[dst] >= 0)
    cross_field = int((known & (field[src] != field[dst])).sum())
    cross_sub = int(((data.subfield[src] >= 0) & (data.subfield[dst] >= 0) & (data.subfield[src] != data.subfield[dst])).sum())
    newer = int((data.year[dst] > data.year[src]).sum())
    out_deg = torch.bincount(src, minlength=data.num_nodes)
    in_deg = torch.bincount(dst, minlength=data.num_nodes)
    isolated = int(((out_deg + in_deg) == 0).sum())

    lines = [
        "=" * 70,
        "DATASET SUMMARY",
        "=" * 70,
        f"Papers (nodes):                 {data.num_nodes}",
        f"Citation edges (directed):      {ei.size(1)}",
        f"Embedding dimensionality:       {data.x.size(1)}",
        f"Distinct publication years:     {papers['year'].nunique()} ({papers['year'].min()}-{papers['year'].max()})",
        f"Distinct fields:                {papers['primary_field'].replace('', np.nan).nunique()}",
        f"Distinct subfields:             {papers['primary_subfield'].replace('', np.nan).nunique()}",
        f"Distinct topics:                {papers['primary_topic'].replace('', np.nan).nunique()}",
        f"Papers missing abstract:        {(papers['abstract'] == '').sum()} "
        f"({100 * (papers['abstract'] == '').mean():.1f}%)",
        f"Isolated papers (degree 0):     {isolated}",
        f"Mean out-/in-degree:            {out_deg.float().mean():.2f}",
        f"Cross-field edges:              {cross_field} ({100 * cross_field / max(ei.size(1), 1):.1f}%)",
        f"Cross-subfield edges:           {cross_sub} ({100 * cross_sub / max(ei.size(1), 1):.1f}%)",
        f"Edges citing a newer paper:     {newer} (preprint/date noise, kept)",
        "",
        "Papers per year:  " + ", ".join(f"{y}:{c}" for y, c in papers["year"].value_counts().sort_index().items()),
        "Papers per field: " + ", ".join(f"{k or '<none>'}:{v}" for k, v in papers["primary_field"].value_counts().items()),
        "",
        "EXAMPLE PAPERS",
        "-" * 70,
    ]
    rng = np.random.default_rng(seed)
    candidates = np.where(out_deg.numpy() > 0)[0]
    if len(candidates) == 0:
        candidates = np.arange(data.num_nodes)
    for i in rng.choice(candidates, size=min(num_examples, len(candidates)), replace=False):
        p = papers.iloc[i]
        cited = dst[src == int(i)].tolist()
        lines += [
            f"[node {i}] {p.openalex_id}  ({p.year}, {p.publication_date})  doi: {p.doi or '-'}",
            f"  title:    {p.title[:100]}",
            f"  abstract: {(p.abstract[:150] + '...') if p.abstract else '<missing>'}",
            f"  field:    {p.primary_field} / {p.primary_subfield} / {p.primary_topic}",
            f"  concepts: {', '.join(p.concepts)}",
            f"  x[{i}][:4] = {[round(float(v), 4) for v in data.x[i, :4]]}",
            f"  cites {len(cited)} in-dataset papers, cited by {int(in_deg[i])}",
        ]
        if cited:
            q = papers.iloc[cited[0]]
            lines.append(f"  e.g. -> [node {cited[0]}] {q.openalex_id} ({q.year}) {q.title[:70]}")
        lines.append("")
    return "\n".join(lines)
