#!/usr/bin/env python3
"""Build data/processed/<region>/cells.parquet from WorldPop + boundary, or synthetic fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.grid import load_or_build_cells  # noqa: E402
from sim.region import DEFAULT_REGION, load_region  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()
    region = load_region(args.region)
    df = load_or_build_cells(region, ROOT, write=True)
    out = region.processed_dir(ROOT) / "cells.parquet"
    print(f"Wrote {len(df)} cells to {out}")
    print(df.head().to_string(index=False))
    print(f"population sum: {df['population'].sum():,.0f}")


if __name__ == "__main__":
    main()
