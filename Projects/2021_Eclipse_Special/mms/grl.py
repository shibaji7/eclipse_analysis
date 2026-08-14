#!/usr/bin/env python3
"""Command-line entry point for the consolidated GRL figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the consolidated GRL-style eclipse figure for the 2021-12-04 event."
    )
    parser.add_argument(
        "--output-dir",
        default="grl_summary",
        help="Directory where the figure will be written. Default: grl_summary",
    )
    parser.add_argument(
        "--start",
        default="2021-12-04T06:00:00",
        help="Start time for the event window. Default: 2021-12-04T06:00:00",
    )
    parser.add_argument(
        "--end",
        default="2021-12-04T10:00:00",
        help="End time for the event window. Default: 2021-12-04T10:00:00",
    )
    parser.add_argument(
        "--toi",
        default="2021-12-04T07:30:00",
        help="Time of interest used for the spacecraft-location schematic. Default: 2021-12-04T07:30:00",
    )
    parser.add_argument(
        "--feature-start",
        default="2021-12-04T07:00:00",
        help="Start of the highlighted feature window. Default: 2021-12-04T07:00:00",
    )
    parser.add_argument(
        "--feature-end",
        default="2021-12-04T07:45:00",
        help="End of the highlighted feature window. Default: 2021-12-04T07:45:00",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Use only local cached data and skip update attempts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Allow running as `python grl.py` from inside `mms/`.
    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from mms.workflow import build_grl_summary_figure

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    build_grl_summary_figure(
        probes=[1],
        time_range=(args.start, args.end),
        time_of_interest=args.toi,
        output_dir=outdir,
        feature_window=(args.feature_start, args.feature_end),
        no_update=args.no_update,
    )
    print(f"Saved GRL figure to {outdir / 'grl_consolidated_figure.png'}")
    print(f"Saved GRL figure to {outdir / 'grl_consolidated_figure.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
