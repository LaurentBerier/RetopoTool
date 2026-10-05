"""`retopo` command line.

    retopo stats IN.glb
    retopo optimize IN.glb OUT.glb [--ratio R | --target-triangles N] [--profile P] [--no-bake]
    retopo lod IN.glb OUT_DIR [--tier name:triangles|percent%[:texture] ...] [--profile P]
    retopo measure fidelity|uv-drift|render ...

Results are printed to stdout as JSON; logs go to stderr (`-v` for progress).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import numpy as np

_MEASURE = {
    "fidelity": "retopotool.measure.fidelity",
    "uv-drift": "retopotool.measure.uv_drift",
    "render": "retopotool.measure.render",
}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=_jsonable))


def _parse_tier(spec: str):
    """`name:BUDGET[:texture_size]` where BUDGET is a triangle count (30000) or a percentage of
    the source (25%)."""
    from .pipeline import LodTier
    try:
        bits = spec.split(":")
        if len(bits) not in (2, 3):
            raise ValueError
        name, budget = bits[0], bits[1].strip()
        tex = int(bits[2]) if len(bits) == 3 and bits[2] else None
        if budget.endswith("%"):
            return LodTier(name, ratio=float(budget[:-1]) / 100.0, texture_size=tex)
        return LodTier(name, int(budget), tex)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "tier must be name:triangles[:texture_size] or name:percent%[:texture_size] "
            f"(e.g. low:30000:1024, lod2:25%), got {spec!r}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="retopo",
        description="Seam-preserving GLB decimation, normal-map bake and LOD ladders.")
    ap.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="cheap mesh report (triangles, skin, recommended ratio)")
    p.add_argument("input")

    p = sub.add_parser("optimize", help="decimate an UNRIGGED GLB (+ normal-map bake)")
    p.add_argument("input")
    p.add_argument("output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ratio", type=float, help="fraction of triangles to keep (0.05-0.95)")
    g.add_argument("--target-triangles", type=int, help="triangle budget (default 400000)")
    p.add_argument("--no-bake", action="store_true", help="skip the normal-map bake")
    p.add_argument("--profile", choices=("prop", "character"), default="prop",
                   help="density allocation: uniform (prop, default) or head/hands first "
                        "(character, Y-up A/T-pose)")
    p.add_argument("--head-boost", type=float, help="head density boost (1.0 = off)")
    p.add_argument("--hand-boost", type=float, help="hand density boost (1.0 = off)")

    p = sub.add_parser("lod", help="build an LOD ladder from a GLB (skin preserved if rigged)")
    p.add_argument("input")
    p.add_argument("out_dir")
    p.add_argument("--tier", action="append", type=_parse_tier, metavar="NAME:BUDGET[:TEX]",
                   help="repeatable; BUDGET is a triangle count or a percentage (25%%). Default: "
                        "lod1:50%% lod2:25%% lod3:10%% for a static mesh, high:100000:2048 "
                        "medium:60000:1024 low:30000:1024 minimum:15000:512 for a skinned one")
    p.add_argument("--stem", help="output file stem (default: the input's)")
    p.add_argument("--no-bake", action="store_true", help="skip the normal-map bake")
    p.add_argument("--profile", choices=("auto", "prop", "character"), default="auto",
                   help="density allocation (auto = character for a skinned GLB, prop otherwise)")
    p.add_argument("--head-boost", type=float, help="head density boost (1.0 = off)")
    p.add_argument("--hand-boost", type=float, help="hand density boost (1.0 = off)")

    sub.add_parser("measure", help="quality instruments: fidelity | uv-drift | render "
                                   "(see `retopo measure <tool> --help`)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = any(a in ("-v", "--verbose") for a in argv[:1])
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")

    # `measure` forwards everything after the tool name verbatim to that tool's own parser.
    rest = [a for a in argv if a not in ("-v", "--verbose")]
    if rest[:1] == ["measure"]:
        if len(rest) < 2 or rest[1] not in _MEASURE:
            print(f"usage: retopo measure {{{','.join(_MEASURE)}}} ...", file=sys.stderr)
            return 2
        import importlib
        return int(importlib.import_module(_MEASURE[rest[1]]).main(rest[2:]) or 0)

    a = _build_parser().parse_args(argv)
    from . import pipeline
    from .decimate import TARGET_TRIANGLES, mesh_stats
    try:
        if a.cmd == "stats":
            _print_json(mesh_stats(a.input))
        elif a.cmd == "optimize":
            _print_json(pipeline.optimize_glb(
                a.input, a.output, ratio=a.ratio,
                target_triangles=a.target_triangles or TARGET_TRIANGLES,
                bake=not a.no_bake, profile=a.profile,
                head_boost=a.head_boost, hand_boost=a.hand_boost))
        elif a.cmd == "lod":
            _print_json(pipeline.build_lod_ladder(
                a.input, a.out_dir, a.tier or None, stem=a.stem, bake=not a.no_bake,
                profile=a.profile, head_boost=a.head_boost, hand_boost=a.hand_boost))
    except (ValueError, FileNotFoundError) as exc:
        print(f"retopo: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
