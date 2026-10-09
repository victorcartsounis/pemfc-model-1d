"""Sweep the base case from 1.20 V down to 0.05 V, time it and plot the result.

Usage: python scripts/polarization_curve.py [--step 0.05] [--out results]
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pemfc_model_1d.assembly import build
from pemfc_model_1d.plotting import plot_polarization_curve, plot_profiles
from pemfc_model_1d.solve import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05, help="voltage step [V]")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    start = time.perf_counter()
    build()
    setup = time.perf_counter() - start
    voltages = np.round(np.arange(1.20, 0.05 - 1e-9, -args.step), 4)
    sweep = solve(voltages)
    total = time.perf_counter() - start

    print(f"setup {setup:.2f} s, sweep {total - setup:.2f} s, total {total:.2f} s")
    print("   U [V]   I [A/cm2]   nodes")
    for point in sweep.points:
        current = point.current_density / 1e4
        print(f"{point.voltage:8.3f}  {current:10.4f}  {point.nodes:6d}")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_polarization_curve(sweep).savefig(args.out / "polarization_curve.png", dpi=150)
    plot_profiles(sweep).savefig(args.out / "profiles.png", dpi=150)


if __name__ == "__main__":
    main()
