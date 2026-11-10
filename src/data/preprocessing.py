"""Turn raw OpenAlex JSONL records into a clean papers table and citation edge list.

Node order is deterministic: papers are sorted by (publication_year, openalex_id)
and `node_idx` is the row number. Every downstream artifact (embeddings, graph
tensors, metadata) uses this order.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def normalize_id(openalex_id: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'."""
    if not openalex_id:
        return None
    return openalex_id.rstrip("/").rsplit("/", 1)[-1].upper()


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", _TAG.sub(" ", text)).strip()


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    positions.sort()
    return clean_text(" ".join(word for _, word in positions))


def _taxonomy_name(topic: dict | None, level: str) -> str:
    if not topic:
        return ""
    node = topic if level == "topic" else topic.get(level)
    return (node or {}).get("display_name", "") or ""


def parse_work(work: dict, max_concepts: int = 5) -> dict | None:
    """Extract the fields we keep from one OpenAlex work record."""
    pid = normalize_id(work.get("id"))
    title = clean_text(work.get("title") or work.get("display_name"))
    year = work.get("publication_year")
    if not pid or not title or year is None:
        return None
    topic = work.get("primary_topic")
    concepts = sorted(work.get("concepts") or [], key=lambda c: -c.get("score", 0))
    doi = work.get("doi") or ""
    return {
        "openalex_id": pid,
        "title": title,
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": int(year),
        "publication_date": work.get("publication_date") or "",
        "doi": doi.replace("https://doi.org/", ""),
        "type": work.get("type") or "",
        "language": work.get("language") or "",
        "primary_topic": _taxonomy_name(topic, "topic"),
        "primary_subfield": _taxonomy_name(topic, "subfield"),
        "primary_field": _taxonomy_name(topic, "field"),
        "primary_domain": _taxonomy_name(topic, "domain"),
        "concepts": [c["display_name"] for c in concepts[:max_concepts] if c.get("display_name")],
        "selection_group": work.get("_group", ""),
        # Present-day count: leakage risk, kept for reference only, never a feature.
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "referenced_works": sorted({r for r in map(normalize_id, work.get("referenced_works") or []) if r}),
    }


def load_raw(files: list[Path]) -> list[dict]:
    works = []
    for path in files:
        with open(path) as f:
            works.extend(json.loads(line) for line in f if line.strip())
    return works


def build_tables(works: list[dict]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (papers, edges, stats).

    papers: one row per unique paper in node order, with `node_idx` column.
    edges:  int64 array [num_edges, 2] of (citing_idx, cited_idx), deduped,
            no self-loops, only between papers in `papers`.
    """
    stats = {"raw_records": len(works)}
    parsed = {}
    dropped = 0
    for work in works:
        row = parse_work(work)
        if row is None:
            dropped += 1
            continue
        # Duplicates (same paper returned by two group queries): keep the first,
        # but union reference lists in case one record is more complete.
        if row["openalex_id"] in parsed:
            prev = parsed[row["openalex_id"]]
            prev["referenced_works"] = sorted(set(prev["referenced_works"]) | set(row["referenced_works"]))
            continue
        parsed[row["openalex_id"]] = row
    stats["dropped_invalid"] = dropped
    stats["duplicates_removed"] = len(works) - dropped - len(parsed)

    papers = pd.DataFrame(list(parsed.values()))
    papers = papers.sort_values(["year", "openalex_id"], kind="stable").reset_index(drop=True)
    papers.insert(0, "node_idx", np.arange(len(papers), dtype=np.int64))
    id_to_idx = dict(zip(papers["openalex_id"], papers["node_idx"]))

    edge_set = set()
    external_refs = 0
    for src, refs in zip(papers["node_idx"], papers["referenced_works"]):
        for ref in refs:
            dst = id_to_idx.get(ref)
            if dst is None:
                external_refs += 1
            elif dst != src:
                edge_set.add((int(src), int(dst)))
    edges = np.array(sorted(edge_set), dtype=np.int64).reshape(-1, 2)

    years = papers["year"].to_numpy()
    stats["external_references_dropped"] = external_refs
    stats["num_papers"] = len(papers)
    stats["num_edges"] = len(edges)
    # Citing a paper with a *later* publication year (preprint/versioning noise).
    stats["edges_cited_newer_than_citing"] = int((years[edges[:, 1]] > years[edges[:, 0]]).sum()) if len(edges) else 0
    return papers, edges, stats
