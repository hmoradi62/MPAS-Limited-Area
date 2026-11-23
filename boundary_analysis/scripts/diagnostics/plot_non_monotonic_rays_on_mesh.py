#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot NON_MONOTONIC MPAS rays over actual MPAS cell polygons colored by bdyMaskCell.

Run from the MPAS-Limited-Area directory:
python plot_non_monotonic_rays_on_mesh.py \
  --static iran.static.nc \
  --csv boundary_ray_topology.csv

Outputs:
  non_monotonic_mesh_panels.png
  non_monotonic_mesh_panels.pdf
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm
from netCDF4 import Dataset
from pyproj import CRS, Transformer


def degrees(values):
    a = np.asarray(values, dtype=float)
    if np.nanmax(np.abs(a)) <= 2.0 * np.pi + 0.1:
        return np.degrees(a)
    return a


def ids(text):
    if pd.isna(text) or not str(text).strip():
        return []
    return [int(v) for v in str(text).split(";")]


def main():
    project_root = Path(__file__).resolve().parents[3]

    diagnostics_dir = (
        project_root
        / "boundary_analysis"
        / "results"
        / "diagnostics"
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--static",
        default=str(project_root / "iran.static.nc")
    )
    ap.add_argument(
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
    ap.add_argument(
        "--output",
        default=str(diagnostics_dir / "non_monotonic_mesh_panels")
    )
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rays = pd.read_csv(args.csv)
    rays = rays[rays.status == "NON_MONOTONIC"].copy()
    if rays.empty:
        raise SystemExit("No NON_MONOTONIC rays found.")

    with Dataset(args.static) as nc:
        latc = degrees(nc.variables["latCell"][:])
        lonc = degrees(nc.variables["lonCell"][:])
        latv = degrees(nc.variables["latVertex"][:])
        lonv = degrees(nc.variables["lonVertex"][:])
        masks = np.asarray(nc.variables["bdyMaskCell"][:], dtype=int)
        voc = np.asarray(nc.variables["verticesOnCell"][:], dtype=int)
        if "nEdgesOnCell" in nc.variables:
            n_edges = np.asarray(nc.variables["nEdgesOnCell"][:], dtype=int)
        else:
            n_edges = np.sum(voc > 0, axis=1)

    # Convert MPAS one-based vertex IDs to zero-based arrays.
    cell_vertex_ids = []
    for i in range(len(latc)):
        nv = int(n_edges[i])
        vids = voc[i, :nv] - 1
        vids = vids[vids >= 0]
        cell_vertex_ids.append(vids)

    ncols = 4
    nrows = math.ceil(len(rays) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4.7*nrows), squeeze=False)
    axs = axs.ravel()

    cmap = plt.get_cmap("viridis", 8)
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)

    for ax, (_, r) in zip(axs, rays.sort_values(
            "topology_exit_distance_km", ascending=False).iterrows()):

        # A local AEQD projection matches the geometry used by the ray tracer.
        lon0, lat0 = float(r.origin_lon), float(r.origin_lat)
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
            "+datum=WGS84 +units=m +no_defs"
        )
        transformer = Transformer.from_crs("EPSG:4326", local, always_xy=True)

        # Project all cell centers and vertices once for this local panel.
        xc, yc = transformer.transform(lonc, latc)
        xv, yv = transformer.transform(lonv, latv)

        origin_x, origin_y = transformer.transform(lon0, lat0)
        exit_x, exit_y = transformer.transform(
            float(r.topology_exit_lon), float(r.topology_exit_lat)
        )

        # Use the ray endpoints to define a padded local view.
        span = max(math.hypot(exit_x-origin_x, exit_y-origin_y), 1.0)
        pad = max(0.12 * span, 80000.0)
        xmin, xmax = min(origin_x, exit_x)-pad, max(origin_x, exit_x)+pad
        ymin, ymax = min(origin_y, exit_y)-pad, max(origin_y, exit_y)+pad

        # Draw actual MPAS cell polygons. The axis limits crop the full mesh.
        polygons, poly_masks = [], []
        for ci, vids in enumerate(cell_vertex_ids):
            if len(vids) < 3:
                continue
            px, py = xv[vids], yv[vids]
            # Cheap bounding-box rejection before creating the polygon.
            if (np.max(px) < xmin or np.min(px) > xmax or
                    np.max(py) < ymin or np.min(py) > ymax):
                continue
            polygons.append(np.column_stack([px, py]))
            poly_masks.append(masks[ci])

        collection = PolyCollection(
            polygons,
            array=np.asarray(poly_masks, dtype=float),
            cmap=cmap,
            norm=norm,
            edgecolors=(0.25, 0.25, 0.25, 0.55),
            linewidths=0.25,
            zorder=1
        )
        ax.add_collection(collection)

        # Straight ray in the same local projected plane.
        ax.plot([origin_x, exit_x], [origin_y, exit_y],
                color="red", linewidth=1.8, zorder=5, label="Normal ray")

        path_cells = ids(r.visited_cells)
        px, py = xc[path_cells], yc[path_cells]
        ax.plot(px, py, color="white", linewidth=0.8,
                linestyle=":", alpha=0.9, zorder=6)
        ax.scatter(px, py, c=masks[path_cells], cmap=cmap, norm=norm,
                   s=24, edgecolors="black", linewidths=0.35, zorder=7)

        # Label only selected cells to keep long paths readable.
        if len(path_cells) <= 18:
            label_idx = range(len(path_cells))
        else:
            label_idx = sorted(set([
                0, 1, len(path_cells)//4, len(path_cells)//2,
                3*len(path_cells)//4, len(path_cells)-2, len(path_cells)-1
            ]))
        for j in label_idx:
            ax.annotate(str(path_cells[j]), (px[j], py[j]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color="black", zorder=8)

        ax.scatter([origin_x], [origin_y], marker="*", s=100,
                   c="gold", edgecolors="black", linewidths=0.6,
                   zorder=9, label="Ray origin")
        ax.scatter([exit_x], [exit_y], marker="X", s=65,
                   c="red", edgecolors="black", linewidths=0.5,
                   zorder=9, label="Boundary exit")

        seq = [int(v) for v in str(r.mask_sequence).split("-")]
        ndec = sum(b < a for a, b in zip(seq, seq[1:]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Inner edge {int(r.inner_edge)} | exit {int(r.topology_exit_edge)}\n"
            f"{float(r.topology_exit_distance_km):.1f} km | "
            f"{len(path_cells)} cells | decreases={ndec}",
            fontsize=9
        )
        ax.set_xlabel("Local east (km)")
        ax.set_ylabel("Local north (km)")
        ax.grid(True, alpha=0.18, linewidth=0.4)

    for ax in axs[len(rays):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axs[:len(rays)]),
                        fraction=0.012, pad=0.012, ticks=range(8))
    cbar.set_label("bdyMaskCell")
    fig.suptitle(
        "MPAS NON_MONOTONIC rays over actual cell polygons\n"
        "Red line: geometric normal ray | Gold star: origin | Red X: physical boundary exit",
        fontsize=14
    )
    # Constrained layout avoids the tight_layout warning caused by the shared colorbar.
    fig.set_layout_engine("constrained")
    fig.savefig(args.output + ".png", dpi=220)
    fig.savefig(args.output + ".pdf")
    plt.close(fig)

    print(f"Rays plotted: {len(rays)}")
    print(f"Saved: {args.output}.png")
    print(f"Saved: {args.output}.pdf")


if __name__ == "__main__":
    main()
