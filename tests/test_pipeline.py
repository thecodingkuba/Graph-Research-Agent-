"""Pipeline tests. Unit tests use small hand-written OpenAlex-shaped records;
the `slow` test downloads and runs the real SciBERT model."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import embeddings, graph, preprocessing, validate  # noqa: E402

OA = "https://openalex.org/"


def work(wid, year, refs=(), title="A paper", abstract=None, field="Computer Science", group="ml"):
    return {
        "id": OA + wid,
        "doi": f"https://doi.org/10.1/{wid.lower()}",
        "title": title,
        "display_name": title,
        "publication_year": year,
        "publication_date": f"{year}-01-01",
        "abstract_inverted_index": abstract,
        "referenced_works": [OA + r for r in refs],
        "primary_topic": {"display_name": f"{field} topic", "subfield": {"display_name": f"{field} sub"},
                          "field": {"display_name": field}, "domain": {"display_name": "D"}},
        "concepts": [{"display_name": "c1", "score": 0.2}, {"display_name": "c2", "score": 0.9}],
        "cited_by_count": 5,
        "_group": group,
    }


@pytest.fixture
def works():
    return [
        work("W3", 2020, refs=["W1", "W2", "W2", "W999", "W3"], title="Third",
             abstract={"Hello": [0], "world": [1, 3], "again": [2]}),
        work("W1", 2018, title="First"),
        work("W2", 2019, refs=["W1"], title="Second", field="Biology", group="compbio"),
        work("W2", 2019, refs=["W1"], title="Second", field="Biology", group="ml"),  # duplicate
        work("W4", 2020, refs=["W1"], title=None),  # display_name also None -> dropped
    ]


class FakeEncoder:
    device = "fake"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.stack([np.full(embeddings.EMBED_DIM, len(t), dtype=np.float32) for t in texts])


def test_abstract_reconstruction_and_ids():
    assert preprocessing.reconstruct_abstract({"b": [1], "a": [0], "c": [2]}) == "a b c"
    assert preprocessing.reconstruct_abstract(None) == ""
    assert preprocessing.normalize_id("https://openalex.org/w42") == "W42"
    assert embeddings.paper_text("Title", "") == "Title"
    assert embeddings.paper_text("Title", "Abs") == "Title. Abs"


def test_build_tables(works):
    papers, edges, stats = preprocessing.build_tables(works)
    assert papers["openalex_id"].tolist() == ["W1", "W2", "W3"]  # sorted by (year, id), dedup, W4 dropped
    assert papers["node_idx"].tolist() == [0, 1, 2]
    assert stats["duplicates_removed"] == 1 and stats["dropped_invalid"] == 1
    # W3 cites W1, W2 (dup + external W999 + self-loop removed); W2 cites W1.
    assert edges.tolist() == [[1, 0], [2, 0], [2, 1]]
    assert papers.loc[2, "abstract"] == "Hello world again world"
    assert papers.loc[0, "abstract"] == ""
    assert papers.loc[0, "concepts"] == ["c2", "c1"]
    assert papers.loc[0, "doi"] == "10.1/w1"


def build(works, tmp_path):
    papers, edges, _ = preprocessing.build_tables(works)
    texts = [embeddings.paper_text(t, a) for t, a in zip(papers.title, papers.abstract)]
    x = embeddings.compute_embeddings(papers.openalex_id.tolist(), texts, tmp_path / "emb",
                                      shard_size=2, batch_size=1, encoder=FakeEncoder())
    data, vocab = graph.build_graph(papers, edges, x)
    graph.save_dataset(tmp_path, data, papers, vocab, {})
    return graph.load_dataset(tmp_path) + (vocab, texts)


def test_graph_alignment_and_validation(works, tmp_path):
    data, papers, vocab, texts = build(works, tmp_path)
    validate.validate(data, papers, vocab)
    assert data.x.shape == (3, 768) and data.edge_index.shape == (2, 3)
    # Row i of x was computed from node i's text (FakeEncoder encodes text length).
    assert data.x[:, 0].tolist() == [float(len(t)) for t in texts]
    assert data.year.tolist() == [2018, 2019, 2020]
    assert data.edge_year.tolist() == [2019, 2020, 2020]
    assert [vocab["field"][i] for i in data.field] == ["Computer Science", "Biology", "Computer Science"]
    assert "Cross-field edges:              2" in validate.summarize(data, papers)


@pytest.mark.parametrize("corrupt, expected", [
    (lambda d, p: setattr(d, "edge_index", d.edge_index.flip(0)), "u cites v"),
    (lambda d, p: setattr(d, "edge_index", torch.cat([d.edge_index, d.edge_index[:, :1]], 1)), "duplicate"),
    (lambda d, p: setattr(d, "edge_index", torch.tensor([[0], [7]])), "outside the dataset"),
    (lambda d, p: setattr(d, "x", d.x[:2]), "shape"),
    (lambda d, p: p.iloc[1:], "metadata rows"),
    (lambda d, p: p.__setitem__("year", p["year"].values[::-1]), "year tensor"),
])
def test_validation_catches_corruption(works, tmp_path, corrupt, expected):
    data, papers, vocab, _ = build(works, tmp_path)
    corrupted = corrupt(data, papers)  # mutates in place, or returns a new metadata frame
    papers = papers if corrupted is None else corrupted
    with pytest.raises(validate.ValidationError, match=expected):
        validate.validate(data, papers, vocab)


def test_embedding_resume(tmp_path):
    ids, texts = [f"W{i}" for i in range(5)], ["a" * (i + 1) for i in range(5)]
    enc = FakeEncoder()
    x1 = embeddings.compute_embeddings(ids, texts, tmp_path, shard_size=2, batch_size=2, encoder=enc)
    (tmp_path / "shard_00001.npy").unlink()  # simulate interruption
    enc2 = FakeEncoder()
    x2 = embeddings.compute_embeddings(ids, texts, tmp_path, shard_size=2, batch_size=2, encoder=enc2)
    assert enc2.calls == 1 and np.array_equal(x1, x2)
    # Changing the node set invalidates the cache.
    enc3 = FakeEncoder()
    embeddings.compute_embeddings(ids[::-1], texts[::-1], tmp_path, shard_size=2, batch_size=2, encoder=enc3)
    assert enc3.calls == 3


@pytest.mark.slow
def test_scibert_real():
    enc = embeddings.SciBertEncoder(max_length=64)
    out = enc.encode(["Graph neural networks.", "Protein folding with deep learning. " * 50, "x"])
    assert out.shape == (3, 768) and np.isfinite(out).all()
    assert not np.allclose(out[0], out[1])
