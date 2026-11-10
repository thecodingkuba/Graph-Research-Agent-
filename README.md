# Graph-Guided Scientific Discovery Agent

CS224W project: *Learning to Explore Academic Citation Networks*. The question: can graph neural
networks help a research agent identify emerging cross-disciplinary connections more efficiently
than text retrieval or classic graph heuristics?

**Status: Milestone 1, the data pipeline.** OpenAlex papers and citations are turned into SciBERT node features and a
PyTorch Geometric graph with aligned metadata. GNNs, link prediction, temporal splits and the agent
come in later milestones.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.10, torch_geometric 2.8.1 and transformers 5.0 on macOS (Apple MPS). CUDA
and CPU are picked automatically when available.

**OpenAlex access.** No credentials are required. OpenAlex now meters usage: anonymous requests
share a free daily budget per IP (≈ $0.10/day; one 200-result list page ≈ $0.0001), and the
default build uses about 60 requests. For heavier use, get a free API key at
<https://openalex.org/settings/api> and run `export OPENALEX_API_KEY=...`. The client picks it up
automatically. Anonymous clients may not use `cursor=` paging past the first page (HTTP 401), so
the pipeline uses basic `page=` paging. That is capped at 10,000 results per query, far more
than any single (year, group) query needs. Requests retry with exponential backoff on 429/5xx (honoring `Retry-After`). Other
4xx errors, such as an exhausted budget, fail loudly instead of retrying.

## Run

```bash
python scripts/build_dataset.py                       # full default build (~8k papers)
python scripts/build_dataset.py --per-year 20 \
    --raw-dir data/raw_small --out-dir data/processed_small   # ~400-paper smoke test
python scripts/build_dataset.py --help                # all options
```

Useful flags:
- `--start-year/--end-year` (default 2015–2024)
- `--per-year` (papers per year per group, default 400)
- `--groups` (JSON `{name: OpenAlex filter}`)
- `--batch-size`, `--max-length`, `--device`
- `--skip-collect` (reuse raw files)
- `--force-collect`
- `--limit` (debug)

Both stages are resumable. Collection writes one JSONL file per (year, group) plus a `.done`
marker. Embeddings are written in 1024-row shards. Re-running the command after an interruption
skips completed work.

Tests: `pytest -q`. The test marked `slow` downloads and runs SciBERT; use `-m "not slow"` to skip it.

## Pipeline

```
OpenAlex API ──► data/raw/works_<year>_<group>.jsonl      src/data/openalex.py
            ──► papers table + citation edge list          src/data/preprocessing.py
            ──► SciBERT embeddings (sharded cache)         src/data/embeddings.py
            ──► PyG Data + metadata in data/processed/     src/data/graph.py
            ──► validation checks + summary                src/data/validate.py
```
`scripts/build_dataset.py` runs all stages.

### 1. Paper selection (`openalex.py`)
For each publication year and each *group*, the pipeline takes the top-N works (type
article/preprint) by citation count. The default groups are:

| group     | OpenAlex filter |
|-----------|-----------------|
| `ml`      | primary-topic subfield ∈ {Artificial Intelligence (1702), Computer Vision & Pattern Recognition (1707)} |
| `compbio` | primary topic ∈ {Genomics & Phylogenetic Studies, Protein Structure & Dynamics, Computational Drug Discovery, Gene Regulatory Network Analysis, Gene expression & cancer classification, Bioinformatics & Genomic Networks, Single-cell & spatial transcriptomics, ML in Bioinformatics, Genetics/Bioinformatics/Biomedical Research} |

Computational biology is selected by *topic*, not subfield, because the life-science subfields
(e.g. Molecular Biology) are dominated by wet-lab work. Taking the top-cited papers per year keeps
the internal citation graph dense and the years balanced; uniform random sampling at this size gives
an almost edgeless graph.

### 2. Preprocessing (`preprocessing.py`)
- OpenAlex IDs are normalized (`https://openalex.org/W123` → `W123`), and duplicates (a paper
  returned by two queries) are merged.
- Abstracts are rebuilt from OpenAlex's `abstract_inverted_index`. Missing values become `""`,
  and papers with no title or year are dropped.
- **Node order:** papers are sorted by `(year, openalex_id)` and `node_idx` is the row number. The
  embeddings, the graph tensors and `papers.parquet` all follow this order.
- **Edges:** if paper A lists B in `referenced_works` and both are in the dataset, the pipeline
  adds A → B (`edge_index[0]` = citing, `edge_index[1]` = cited). References outside the dataset
  are dropped. Self-loops and duplicate edges are removed.

### 3. Node features (`embeddings.py`)
- The model is `allenai/scibert_scivocab_uncased`, used frozen with `torch.inference_mode()`.
- The input text is `title + ". " + abstract`, or the title alone when the abstract is missing. It
  is truncated to 512 tokens.
- **Pooling: mean over non-padding tokens** of the last layer. Without fine-tuning, BERT's
  `[CLS]` vector is only trained for next-sentence prediction, so mean pooling is the more
  reliable sentence representation.
- Inputs are batched and length-sorted within each shard to reduce padding, then put back in node
  order.
- A manifest stores a hash of the ordered ID list, the model and the max length. If the dataset
  changes, stale shards are discarded automatically.

### 4. Graph and outputs (`graph.py`)

`data/processed/`:

| file | contents |
|------|----------|
| `graph.pt` | `Data(x=[N,768], edge_index=[2,E], year=[N], edge_year=[E], field=[N], subfield=[N], topic=[N])` |
| `papers.parquet` | row *i* = node *i*. Columns: `node_idx, openalex_id, title, abstract, year, publication_date, doi, type, language, primary_topic/subfield/field/domain, concepts, selection_group, cited_by_count, referenced_works` |
| `edges.parquet` | `src_idx, dst_idx, src_id, dst_id, edge_year` |
| `id_to_idx.json` | OpenAlex ID → node index |
| `label_vocab.json` | integer label → name for `field`, `subfield` and `topic` (`-1` = missing) |
| `dataset_info.json` | build config, selection filters and preprocessing stats |
| `embeddings/` | shard cache (can be deleted once `graph.pt` exists) |

`edge_year` is the citing paper's publication year, i.e. when the edge appeared. It's there so
snapshots `G_t` can be built later. Load the dataset with:

```python
from src.data.graph import load_dataset
data, papers = load_dataset("data/processed")
```

### 5. Validation (`validate.py`)
After saving, the build reloads the files from disk and checks:
- every node has exactly one finite, non-zero 768-dimensional embedding
- metadata rows match `num_nodes`, `node_idx == 0..N-1`, and IDs are unique
- edge indices are within `[0, N)`, with no self-loops and no duplicate edges
- **direction and completeness:** the edge set equals the set of in-dataset `referenced_works`
  pairs (u → v ⇔ u cites v)
- `year`, `edge_year` and the field/subfield/topic labels match the metadata

Any failure aborts the build. The script then prints a summary with counts, degree statistics,
cross-field edge share and example papers.

## Known limitations and decisions

- **Selection bias / leakage.** Papers are chosen by *present-day* `cited_by_count`. That only
  affects *which* papers are in the dataset; citation counts are never used as features. It does
  bias the dataset toward papers that later became influential, which matters for interpreting
  temporal link-prediction results. `cited_by_count` stays in `papers.parquet` for reference only.
  Don't use it as a feature.
- **Retrospective labels.** OpenAlex topics and fields are assigned by a current classifier, not
  at publication time. SciBERT was pretrained on papers up to 2019. Both are mild sources of
  temporal leakage, as the proposal notes.
- **Citations to "newer" papers.** About 5% of edges point to a paper with a later
  `publication_year`. Inspection shows these are real citations where OpenAlex merged versions and
  dates the work by a later reprint (for example, AlexNet (2012) is dated 2017 via its CACM
  reprint). The edges are kept and counted in the summary. Snapshot code should keep only nodes
  with `year <= t` and edges with `edge_year <= t` whose endpoints both exist.
- **Missing abstracts.** About 17% of papers have no abstract in OpenAlex (publisher
  restrictions). Their embedding uses the title only.
- **Fields.** With the default groups, `primary_field` is mostly two-valued (Computer Science vs.
  Biochemistry/Genetics/Molecular Biology, plus a few others). `subfield` and `topic` give finer
  labels for defining "cross-disciplinary".
- **Scale.** The default build targets about 8k papers, below the proposal's 50–100k. Increase
  `--per-year` to scale up. Cost grows linearly, and an API key is recommended beyond roughly 50k
  papers.

## Next milestone
- Build temporal snapshots and splits (train ≤ 2021, validate 2022, test 2023–24)
- Negative sampling
- GraphSAGE / GAT encoders with an MLP link predictor
- Heuristic baselines (PageRank, common neighbors)
- Cross-field evaluation (P@K, R@K, AP)
- The budgeted research agent
