#!/usr/bin/env python3
"""Build data/processed/cells.parquet from WorldPop + boundary, or synthetic fallback."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.config import load_config  # noqa: E402
from sim.grid import load_or_build_cells  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    df = load_or_build_cells(cfg.grid, ROOT, write=True)
    out = ROOT / cfg.grid.cells_path
    print(f"Wrote {len(df)} cells to {out}")
    print(df.head().to_string(index=False))
    print(f"population sum: {df['population'].sum():,.0f}")


if __name__ == "__main__":
    main()
