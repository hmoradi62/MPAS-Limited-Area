#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose and plot the 24 NON_MONOTONIC topology-aware normal rays.

Inputs:
  iran.static.nc
  boundary_ray_topology.csv

Outputs:
  non_monotonic_ray_diagnostics.csv
  non_monotonic_ray_panels.png

The plot is a geographic overview based on MPAS cell centers. It is intended
for topology/mask-pattern diagnosis, not as a metric-accurate map projection.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def to_degrees(values):
    values = np.asarray(values, dtype=float)
    # MPAS lat/lon variables are normally stored in radians.
    if np.nanmax(np.abs(values)) <= 2.0 * np.pi + 0.1:
        return np.degrees(values)
    return values


def parse_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(x) for x in str(value).split(";")]


def parse_masks(value):
    return [int(x) for x in str(value).split("-")]


def main():
    project_root = Path(__file__).resolve().parents[3]

    results_dir = (
        project_root
        / "boundary_analysis"
        / "results"
        / "diagnostics"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        default=str(project_root / "iran.static.nc")
    )
    parser.add_argument(
        "--csv",
        default=str(
            project_root
            / "boundary_analysis"
            / "results"
            / "topology"
            / "v2"
            / "boundary_ray_topology.csv"
        )
    )
    parser.add_argument(
        "--out-prefix",
        default=str(results_dir / "non_monotonic_ray")
    )
    args = parser.parse_args()

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    bad = df[df["status"] == "NON_MONOTONIC"].copy()
    if bad.empty:
        raise SystemExit("No NON_MONOTONIC rows found in the input CSV.")

    with Dataset(args.static) as nc:
        lat = to_degrees(nc.variables["latCell"][:])
        lon = to_degrees(nc.variables["lonCell"][:])
        mask = np.asarray(nc.variables["bdyMaskCell"][:], dtype=int)

    ncell = len(lat)
    diagnostics = []

    for _, row in bad.iterrows():
        cells = parse_ids(row["visited_cells"])
        masks_from_csv = parse_masks(row["mask_sequence"])
        if len(cells) != len(masks_from_csv):
            raise ValueError(
                f"Cell/mask length mismatch at inner_edge={row['inner_edge']}: "
                f"{len(cells)} cells vs {len(masks_from_csv)} masks"
            )
        if any(c < 0 or c >= ncell for c in cells):
            raise ValueError(f"Out-of-range zero-based cell ID at edge {row['inner_edge']}")

        actual_masks = [int(mask[c]) for c in cells]
        if actual_masks != masks_from_csv:
            raise ValueError(
                f"Mask mismatch at inner_edge={row['inner_edge']}. "
                "Check whether the CSV and static file belong to the same mesh."
            )

        decreases = [(i, a, b, cells[i], cells[i+1])
                     for i, (a, b) in enumerate(zip(actual_masks, actual_masks[1:]))
                     if b < a]
        increases = [(i, a, b) for i, (a, b) in enumerate(zip(actual_masks, actual_masks[1:]))
                     if b > a]
        jumps = [(i, a, b) for i, (a, b) in enumerate(zip(actual_masks, actual_masks[1:]))
                 if b - a > 1]

        diagnostics.append({
            "inner_edge": int(row["inner_edge"]),
            "start_cell": int(row["start_cell"]),
            "exit_edge": int(row["topology_exit_edge"]),
            "exit_cell": int(row["topology_exit_cell"]),
            "width_km": float(row["topology_exit_distance_km"]),
            "n_cells": len(cells),
            "n_mask_decreases": len(decreases),
            "n_mask_increases": len(increases),
            "n_mask_jumps_gt1": len(jumps),
            "mask_sequence": "-".join(map(str, actual_masks)),
            "decrease_transitions": "; ".join(
                f"step {i+1}: {a}->{b} (cell {ca}->{cb})"
                for i, a, b, ca, cb in decreases
            ),
            "visited_cells": ";".join(map(str, cells)),
            "visited_edges": row["visited_edges"],
            "origin_lon": float(row["origin_lon"]),
            "origin_lat": float(row["origin_lat"]),
            "exit_lon": float(row["topology_exit_lon"]),
            "exit_lat": float(row["topology_exit_lat"]),
        })

    diag = pd.DataFrame(diagnostics).sort_values("width_km", ascending=False)
    diag.to_csv(f"{args.out_prefix}_diagnostics.csv", index=False)

    # 4 columns x 6 rows, one panel per non-monotonic ray.
    n = len(bad)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    # Broad mesh backdrop. Each panel is zoomed to its own ray.
    for ax, (_, row) in zip(axes, bad.sort_values("topology_exit_distance_km", ascending=False).iterrows()):
        cells = parse_ids(row["visited_cells"])
        path_lon = lon[cells]
        path_lat = lat[cells]

        # Select a local view around the ray, with a modest geographic margin.
        x0 = min(float(row["origin_lon"]), float(row["topology_exit_lon"]), float(np.min(path_lon)))
        x1 = max(float(row["origin_lon"]), float(row["topology_exit_lon"]), float(np.max(path_lon)))
        y0 = min(float(row["origin_lat"]), float(row["topology_exit_lat"]), float(np.min(path_lat)))
        y1 = max(float(row["origin_lat"]), float(row["topology_exit_lat"]), float(np.max(path_lat)))
        dx = max(x1 - x0, 1.0)
        dy = max(y1 - y0, 1.0)
        padx = max(0.12 * dx, 1.0)
        pady = max(0.12 * dy, 1.0)

        # Background cell centers are deliberately faint.
        ax.scatter(lon, lat, s=2, c="lightgray", alpha=0.35, linewidths=0, zorder=1)

        # Highlight the traversed cell centers by their actual mask values.
        sc = ax.scatter(path_lon, path_lat, c=[mask[c] for c in cells],
                        cmap="viridis", vmin=0, vmax=7, s=35,
                        edgecolors="black", linewidths=0.35, zorder=3)

        # Connect cell centers to show the topological walk.
        ax.plot(path_lon, path_lat, color="crimson", linewidth=1.2,
                alpha=0.9, zorder=2)

        # Number a subset of cells to avoid excessive label overlap.
        if len(cells) <= 16:
            label_indices = range(len(cells))
        else:
            label_indices = sorted(set([0, 1, len(cells)//4, len(cells)//2,
                                        3*len(cells)//4, len(cells)-2, len(cells)-1]))
        for j in label_indices:
            ax.annotate(str(cells[j]), (path_lon[j], path_lat[j]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color="black")

        ax.scatter([row["origin_lon"]], [row["origin_lat"]],
                   marker="*", s=100, c="gold", edgecolors="black",
                   linewidths=0.6, zorder=5, label="Ray origin")
        ax.scatter([row["topology_exit_lon"]], [row["topology_exit_lat"]],
                   marker="X", s=65, c="red", edgecolors="black",
                   linewidths=0.5, zorder=5, label="Boundary exit")

        seq = parse_masks(row["mask_sequence"])
        dec = sum(b < a for a, b in zip(seq, seq[1:]))
        ax.set_xlim(x0 - padx, x1 + padx)
        ax.set_ylim(y0 - pady, y1 + pady)
        ax.set_title(
            f"Inner edge {int(row['inner_edge'])} | exit {int(row['topology_exit_edge'])}\n"
            f"{float(row['topology_exit_distance_km']):.1f} km | "
            f"{len(cells)} cells | decreases={dec}",
            fontsize=9
        )
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    for ax in axes[len(bad):]:
        ax.axis("off")

    cbar = fig.colorbar(sc, ax=list(axes[:len(bad)]), shrink=0.78, pad=0.015)
    cbar.set_label("bdyMaskCell")
    fig.suptitle(
        "MPAS NON_MONOTONIC normal-ray paths\n"
        "Gold star: projected ray origin | Red X: physical boundary exit",
        fontsize=14, y=0.995
    )
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(f"{args.out_prefix}_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("NON-MONOTONIC RAY DIAGNOSTICS")
    print("=" * 72)
    print(f"Number of paths: {len(diag)}")
    print(f"Total mask decreases: {diag['n_mask_decreases'].sum()}")
    print(f"Mean width: {diag['width_km'].mean():.2f} km")
    print(f"Median width: {diag['width_km'].median():.2f} km")
    print("\nLongest paths:")
    print(diag[["inner_edge", "exit_edge", "width_km", "n_cells",
                "n_mask_decreases", "mask_sequence"]].head(10).to_string(index=False))
    print(f"\nSaved: {args.out_prefix}_diagnostics.csv")
    print(f"Saved: {args.out_prefix}_panels.png")


if __name__ == "__main__":
    main()
