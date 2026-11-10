"""Build the OpenAlex citation graph dataset end to end.

    python scripts/build_dataset.py                       # default ~8k papers
    python scripts/build_dataset.py --per-year 20 --out-dir data/processed_small

Stages: collect (OpenAlex API -> data/raw) -> preprocess -> SciBERT embeddings
-> PyG graph -> validation + summary. Collection and embedding are resumable;
re-running the command picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import embeddings, graph, openalex, preprocessing, validate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--per-year", type=int, default=400, help="papers per (year, group)")
    p.add_argument("--groups", type=str, default=None,
                   help='JSON dict {name: OpenAlex filter}; default: ML + CompBio (see src/data/openalex.py)')
    p.add_argument("--work-types", default="article|preprint")
    p.add_argument("--mailto", default=None, help="email for OpenAlex polite pool")
    p.add_argument("--skip-collect", action="store_true", help="use files already in --raw-dir")
    p.add_argument("--force-collect", action="store_true", help="re-download even if cached")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--model", default=embeddings.MODEL_NAME)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--limit", type=int, default=None, help="debug: keep only the first N papers")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("build_dataset")
    t0 = time.time()
    groups = json.loads(args.groups) if args.groups else openalex.DEFAULT_GROUPS

    # 1. Collect
    if args.skip_collect:
        files = sorted(args.raw_dir.glob("works_*.jsonl"))
        if not files:
            sys.exit(f"--skip-collect given but no works_*.jsonl in {args.raw_dir}")
    else:
        client = openalex.OpenAlexClient(mailto=args.mailto)
        if not client.api_key:
            log.info("No OPENALEX_API_KEY set; using shared anonymous budget (fine for ~10k papers).")
        files = openalex.collect(args.raw_dir, args.start_year, args.end_year, args.per_year, groups,
                                 args.work_types, client, force=args.force_collect)

    # 2. Preprocess
    works = preprocessing.load_raw(files)
    papers, edges, stats = preprocessing.build_tables(works)
    if args.limit:  # rebuild from a subset so edges are restricted to the kept papers
        keep = set(papers["openalex_id"][: args.limit])
        works = [w for w in works if preprocessing.normalize_id(w.get("id")) in keep]
        papers, edges, stats = preprocessing.build_tables(works)
    log.info("Preprocessing: %s", stats)

    # 3. Embed
    texts = [embeddings.paper_text(t, a) for t, a in zip(papers["title"], papers["abstract"])]
    x = embeddings.compute_embeddings(
        papers["openalex_id"].tolist(), texts, args.out_dir / "embeddings",
        batch_size=args.batch_size, shard_size=args.shard_size, max_length=args.max_length,
        model_name=args.model, device=args.device,
    )

    # 4. Graph
    data, vocab = graph.build_graph(papers, edges, x)
    info = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "groups": groups,
        "raw_files": [f.name for f in files],
        "preprocessing_stats": stats,
        "embedding": {"model": args.model, "pooling": "mean", "max_length": args.max_length,
                      "text": "title + '. ' + abstract (title only if abstract missing)"},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    graph.save_dataset(args.out_dir, data, papers, vocab, info)

    # 5. Validate (re-load from disk to check what training scripts will see)
    data, papers = graph.load_dataset(args.out_dir)
    passed = validate.validate(data, papers, vocab)
    log.info("All %d validation checks passed:\n  - %s", len(passed), "\n  - ".join(passed))
    print(data)
    print(validate.summarize(data, papers))
    log.info("Done in %.1f min. Output: %s", (time.time() - t0) / 60, args.out_dir)


if __name__ == "__main__":
    main()
