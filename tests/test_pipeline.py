"""High-level entry points: `optimize_glb` auto-ratio and the LOD ladder (skip / rank / atomic write)."""
import hashlib
import os

import pytest
from pygltflib import GLTF2

from retopotool import decimate as D
from retopotool import pipeline
from retopotool.pipeline import LodTier, build_lod_ladder, merge_lod_levels, optimize_glb


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tris(path):
    return D.mesh_stats(path)["triangles"]


# ---- optimize_glb -------------------------------------------------------------------------------

def test_optimize_derives_the_ratio_from_the_triangle_budget(small_mesh_glb, tmp_path):
    before = _sha(small_mesh_glb)
    src_tris = _tris(small_mesh_glb)
    stats = optimize_glb(small_mesh_glb, str(tmp_path / "opt.glb"), target_triangles=5_000)
    assert stats["target_ratio"] == pytest.approx(round(5_000 / src_tris, 3))
    assert 4_000 < stats["triangles_after"] <= 5_000 * 1.02
    assert stats["normal_map_baked"] is True
    assert _sha(small_mesh_glb) == before, "the optimizer must never touch its input"


def test_optimize_under_budget_is_a_light_pass(small_mesh_glb, tmp_path):
    stats = optimize_glb(small_mesh_glb, str(tmp_path / "opt.glb"), target_triangles=10_000_000,
                         bake=False)
    assert stats["target_ratio"] == pytest.approx(0.95)
    assert stats["normal_map_baked"] is False


def test_optimize_explicit_ratio_and_guards(small_mesh_glb, small_rig_glb, tmp_path):
    stats = optimize_glb(small_mesh_glb, str(tmp_path / "half.glb"), ratio=0.5, bake=False)
    assert stats["target_ratio"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        optimize_glb(small_mesh_glb, str(tmp_path / "x.glb"), ratio=0.99)
    with pytest.raises(ValueError):
        optimize_glb(small_mesh_glb, small_mesh_glb, ratio=0.5)
    with pytest.raises(ValueError, match="rigged"):
        optimize_glb(small_rig_glb, str(tmp_path / "rig_opt.glb"))


# ---- build_lod_ladder ---------------------------------------------------------------------------

def test_ladder_builds_ranked_rungs_and_skips_oversized_tiers(small_rig_glb, tmp_path):
    tiers = [LodTier("big", 100_000, 512), LodTier("mid", 8_000, 256), LodTier("tiny", 3_000, 256)]
    ladder = build_lod_ladder(small_rig_glb, str(tmp_path), tiers, stem="hero")
    assert ladder["skipped"] == ["big"]
    assert [e["name"] for e in ladder["levels"]] == ["mid", "tiny"]
    assert [e["level"] for e in ladder["levels"]] == [0, 1]
    assert [e["screen_pct"] for e in ladder["levels"]] == [1.0, 0.5]
    for e in ladder["levels"]:
        assert e["file"] == str(tmp_path / f"hero_lod_{e['name']}.glb")
        assert os.path.getsize(e["file"]) == e["size_bytes"]
        assert e["triangles"] <= e["target_triangles"] * 1.02
        assert e["normal_map_baked"] is True
        assert "sliver_frac" in e["quality"] and "boundary_edges" in e["quality"]
        D._assert_skin_intact(e["file"])          # skin survives in the file actually written
        assert GLTF2().load(e["file"]).skins
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp.glb")]


def test_ladder_validates_tiers_before_writing_anything(small_rig_glb, tmp_path):
    bad = [
        [LodTier("../escape", 5_000, 256)],
        [LodTier("Upper", 5_000, 256)],
        [LodTier("ok", 10, 256)],
        [LodTier("ok", 5_000, 300)],
        [LodTier("dup", 5_000, 256), LodTier("dup", 4_000, 256)],
    ]
    for tiers in bad:
        with pytest.raises(ValueError):
            build_lod_ladder(small_rig_glb, str(tmp_path), tiers)
    assert os.listdir(tmp_path) == []


def test_ladder_removes_the_temp_file_when_a_rung_fails(small_rig_glb, tmp_path, monkeypatch):
    def boom(src, tmp, *a, **kw):
        with open(tmp, "wb") as fh:
            fh.write(b"half-written")
        raise ValueError("simulated failure")

    monkeypatch.setattr(pipeline, "decimate_rigged_glb", boom)
    with pytest.raises(ValueError, match="simulated"):
        build_lod_ladder(small_rig_glb, str(tmp_path), [LodTier("low", 3_000, 256)])
    assert os.listdir(tmp_path) == []


def test_merge_lod_levels_merges_by_name_and_reranks():
    existing = [{"name": "high", "triangles": 100}, {"name": "low", "triangles": 10},
                {"triangles": 5}]
    built = [{"name": "mid", "triangles": 50}, {"name": "low", "triangles": 12}]
    merged = merge_lod_levels(existing, built, removed=["high"])
    assert [(e["name"], e["triangles"], e["level"], e["screen_pct"]) for e in merged] == [
        ("mid", 50, 0, 1.0), ("low", 12, 1, 0.5)]


def test_warn_lod_quality_flags_corruption_and_cosmetics():
    assert pipeline.warn_lod_quality("ok", {"sliver_frac": 0.01, "boundary_edges": 0}) == []
    warnings = pipeline.warn_lod_quality("bad", {"sliver_frac": 0.2, "boundary_edges": 3,
                                                 "skin_keyed_on_position": False})
    assert len(warnings) == 3
