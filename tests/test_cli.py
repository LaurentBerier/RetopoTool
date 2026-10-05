"""`retopo` CLI smoke tests: every subcommand runs end to end on a synthetic mesh."""
import json
import os

from retopotool.cli import main


def _run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_stats(capsys, small_mesh_glb):
    code, out, _ = _run(capsys, "stats", small_mesh_glb)
    assert code == 0
    stats = json.loads(out)
    assert stats["triangles"] > 10_000 and stats["has_skin"] is False


def test_optimize_then_measure(capsys, small_mesh_glb, tmp_path):
    dst = str(tmp_path / "opt.glb")
    code, out, _ = _run(capsys, "optimize", small_mesh_glb, dst, "--target-triangles", "5000")
    assert code == 0 and os.path.exists(dst)
    assert json.loads(out)["triangles_after"] <= 5_100

    code, out, _ = _run(capsys, "measure", "fidelity", "--quality", dst)
    assert code == 0 and "topology" in out and "slivers" in out

    code, out, _ = _run(capsys, "measure", "fidelity", small_mesh_glb, dst)
    assert code == 0 and "p2s" in out

    code, out, _ = _run(capsys, "measure", "uv-drift", small_mesh_glb, dst)
    assert code == 0 and "drift texels" in out

    png = str(tmp_path / "view.png")
    code, _, _ = _run(capsys, "measure", "render", dst, png, "--res", "128")
    assert code == 0 and os.path.getsize(png) > 0


def test_lod(capsys, small_rig_glb, tmp_path):
    code, out, _ = _run(capsys, "lod", small_rig_glb, str(tmp_path), "--tier", "low:3000:256",
                        "--no-bake")
    assert code == 0
    ladder = json.loads(out)
    assert [e["name"] for e in ladder["levels"]] == ["low"]
    assert os.path.exists(ladder["levels"][0]["file"])


def test_errors_are_reported_not_raised(capsys, small_rig_glb, tmp_path):
    code, _, err = _run(capsys, "optimize", small_rig_glb, str(tmp_path / "x.glb"))
    assert code == 1 and "rigged" in err
    code, _, err = _run(capsys, "measure", "nope")
    assert code == 2 and "usage" in err
