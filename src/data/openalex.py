"""Collect paper metadata and citation lists from the OpenAlex works API.

Selection strategy: for every (year, group) pair we take the top-N works by
`cited_by_count`, where a group is an OpenAlex filter expression (e.g. a set of
subfields or topics). Raw API records are written untouched to JSONL so that
preprocessing can be re-run without hitting the API again.

Note: ranking by present-day citation counts uses information from after the
publication year. This only affects *which* papers are selected; citation
counts are never used as node features. See README for discussion.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works"
MAX_BASIC_PAGING_RESULTS = 10_000

SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "display_name",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "abstract_inverted_index",
    "referenced_works",
    "primary_topic",
    "concepts",
    "cited_by_count",
]

# Default paper groups. ML = OpenAlex subfields "Artificial Intelligence" (1702)
# and "Computer Vision and Pattern Recognition" (1707). CompBio = computational
# topics inside the life-science subfields, since the subfields themselves
# (e.g. Molecular Biology) are dominated by wet-lab work.
DEFAULT_GROUPS = {
    "ml": "primary_topic.subfield.id:1702|1707",
    "compbio": "primary_topic.id:" + "|".join([
        "T10015",  # Genomics and Phylogenetic Studies
        "T10044",  # Protein Structure and Dynamics
        "T10211",  # Computational Drug Discovery Methods
        "T10621",  # Gene Regulatory Network Analysis
        "T10885",  # Gene expression and cancer classification
        "T10887",  # Bioinformatics and Genomic Networks
        "T11289",  # Single-cell and spatial transcriptomics
        "T12254",  # Machine Learning in Bioinformatics
        "T13937",  # Genetics, Bioinformatics, and Biomedical Research
    ]),
}


@dataclass
class OpenAlexClient:
    mailto: str | None = None
    api_key: str | None = None
    max_retries: int = 6
    backoff: float = 2.0
    timeout: float = 60.0

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENALEX_API_KEY")
        self.session = requests.Session()

    def get(self, params: dict) -> dict:
        params = dict(params)
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(API_URL, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                wait = self.backoff ** attempt
                log.warning("Request error (%s); retrying in %.0fs", e, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff ** attempt
                log.warning("HTTP %d; retrying in %.0fs", resp.status_code, wait)
                time.sleep(wait)
                continue
            # 4xx other than 429: bad request / budget exhausted / auth -> do not retry.
            raise RuntimeError(f"OpenAlex request failed: HTTP {resp.status_code}: {resp.text[:500]}")
        raise RuntimeError(f"OpenAlex request failed after {self.max_retries} retries: {params}")

    def top_cited(self, filter_expr: str, limit: int, per_page: int = 200):
        """Yield up to `limit` works matching `filter_expr`, most cited first.

        Uses basic `page=` paging: OpenAlex requires an API key for `cursor=`
        paging beyond the first page, while basic paging works anonymously
        for the first 10,000 results of a query (far more than one
        (year, group) query needs)."""
        if limit > MAX_BASIC_PAGING_RESULTS:
            raise ValueError(f"per-query limit {limit} exceeds OpenAlex basic paging cap {MAX_BASIC_PAGING_RESULTS}")
        n, page = 0, 1
        while n < limit:
            data = self.get({
                "filter": filter_expr,
                "sort": "cited_by_count:desc",
                "per_page": per_page,
                "page": page,
                "select": ",".join(SELECT_FIELDS),
            })
            results = data.get("results", [])
            for work in results[: limit - n]:
                yield work
                n += 1
            if len(results) < per_page:
                break
            page += 1


def collect(
    raw_dir: Path,
    start_year: int,
    end_year: int,
    per_year: int,
    groups: dict[str, str] = DEFAULT_GROUPS,
    work_types: str = "article|preprint",
    client: OpenAlexClient | None = None,
    force: bool = False,
) -> list[Path]:
    """Download top-cited works for each (year, group). Resumable: each finished
    query writes a `.done` marker and is skipped on the next run."""
    client = client or OpenAlexClient()
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for year in range(start_year, end_year + 1):
        for name, group_filter in groups.items():
            path = raw_dir / f"works_{year}_{name}.jsonl"
            done = path.with_suffix(".done")
            files.append(path)
            filter_expr = f"{group_filter},publication_year:{year},type:{work_types}"
            spec = {"filter": filter_expr, "limit": per_year}
            if done.exists() and not force:
                prev = json.loads(done.read_text())
                if {k: prev.get(k) for k in spec} == spec:
                    log.info("Skipping %s (already downloaded)", path.name)
                    continue
            count = 0
            tmp = path.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for work in client.top_cited(filter_expr, per_year):
                    work["_group"] = name  # provenance of the selection query
                    f.write(json.dumps(work) + "\n")
                    count += 1
            tmp.replace(path)
            done.write_text(json.dumps({**spec, "count": count}))
            log.info("%s: %d works", path.name, count)
    return files
