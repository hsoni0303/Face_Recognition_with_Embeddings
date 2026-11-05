import numpy as np

from face_id import Gallery


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_match_returns_closest_name_above_threshold():
    rng = np.random.default_rng(0)
    alice, bob = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    g = Gallery()
    g.add("alice", alice)
    g.add("bob", bob)

    probe = unit(alice + 0.02 * rng.normal(size=512))  # noisy photo of alice
    name, score = g.match(probe)
    assert name == "alice" and score > 0.7


def test_unknown_below_threshold_and_empty_gallery():
    rng = np.random.default_rng(1)
    g = Gallery()
    assert g.match(unit(rng.normal(size=512))) == ("unknown", 0.0)
    g.add("alice", unit(rng.normal(size=512)))
    name, _ = g.match(unit(rng.normal(size=512)))  # random face ~ orthogonal
    assert name == "unknown"


def test_save_load_roundtrip(tmp_path):
    g = Gallery()
    g.add("alice", unit(np.ones(512)))
    path = tmp_path / "gallery.npz"
    g.save(path)
    loaded = Gallery.load(path)
    assert loaded.names == ["alice"]
    np.testing.assert_allclose(loaded.embeddings, g.embeddings)
