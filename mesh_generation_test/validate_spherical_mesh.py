#!/usr/bin/env python3
"""Validate a spherical MPAS mesh generated through JIGSAW."""

import argparse
import os
import numpy as np
import xarray as xr


def stats(name, values, unit=""):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]

    print(f"\n{name} [{unit}]")
    print(f"  Valid: {finite.size}/{a.size}")
    print(f"  NaN/Inf: {a.size - finite.size}")

    if finite.size:
        q = np.percentile(
            finite, [0, 1, 5, 25, 50, 75, 95, 99, 100]
        )
        labels = [
            "Min", "P01", "P05", "P25", "Median",
            "P75", "P95", "P99", "Max"
        ]
        for label, value in zip(labels, q):
            print(f"  {label:>6}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh",
        nargs="?",
        default="base_mesh_test.nc",
        help="MPAS mesh NetCDF file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.mesh):
        raise FileNotFoundError(args.mesh)

    ds = xr.open_dataset(args.mesh)

    try:
        print("===== FILE =====")
        print("File:", args.mesh)
        print("Size (MB):", round(os.path.getsize(args.mesh) / 2**20, 2))

        print("\n===== DIMENSIONS =====")
        for name, size in ds.sizes.items():
            print(f"{name}: {size}")

        print("\n===== SPHERE =====")
        print("on_a_sphere:", ds.attrs.get("on_a_sphere"))
        radius = float(ds.attrs["sphere_radius"])
        print("sphere_radius (m):", radius)

        if str(ds.attrs.get("on_a_sphere", "")).upper() != "YES":
            raise ValueError("Mesh is not marked as spherical")

        area = ds["areaCell"].values
        dc = ds["dcEdge"].values / 1000.0
        dv = ds["dvEdge"].values / 1000.0

        theoretical_area = 4.0 * np.pi * radius**2
        mesh_area = np.sum(area)
        relative_error = abs(mesh_area / theoretical_area - 1.0)

        print("\n===== AREA CLOSURE =====")
        print(f"Theoretical (km2): {theoretical_area / 1e6:.6f}")
        print(f"Mesh total (km2):  {mesh_area / 1e6:.6f}")
        print(f"Relative error:    {relative_error:.12e}")

        stats("Cell area", area / 1e6, "km2")
        stats("dcEdge", dc, "km")
        stats("dvEdge", dv, "km")

        print("\n===== QUALITY =====")
        for name in [
            "cellQuality",
            "triangleQuality",
            "triangleAngleQuality"
        ]:
            if name in ds:
                values = ds[name].values
                stats(name, values, "dimensionless")

                finite = values[np.isfinite(values)]
                for threshold in [0.3, 0.5, 0.7, 0.9]:
                    count = np.count_nonzero(finite < threshold)
                    print(
                        f"  Below {threshold}: {count} "
                        f"({100 * count / finite.size:.3f}%)"
                    )

        if "obtuseTriangle" in ds:
            obtuse = ds["obtuseTriangle"].values
            print("\nObtuse triangles:", np.count_nonzero(obtuse))

        if "nEdgesOnCell" in ds:
            counts = ds["nEdgesOnCell"].values
            print("\n===== CELL EDGE COUNTS =====")
            unique, number = np.unique(counts, return_counts=True)
            for n, count in zip(unique, number):
                print(f"{n} edges: {count}")

        print("\n===== CONNECTIVITY =====")
        for name in [
            "cellsOnEdge",
            "verticesOnEdge",
            "cellsOnVertex",
            "edgesOnVertex"
        ]:
            if name not in ds:
                continue

            a = ds[name].values
            zeros = np.count_nonzero(a == 0)
            print(f"{name}: zeros={zeros}")

        print("\n===== GRID SPACING METADATA =====")
        if "gridSpacing" in ds:
            print("Attributes:", ds["gridSpacing"].attrs)
            print(
                "Raw min/max:",
                float(ds["gridSpacing"].min()),
                float(ds["gridSpacing"].max())
            )

        print("\n===== VALIDATION SUMMARY =====")
        print("Area closure relative error:", relative_error)
        print("Mesh dimensions:", dict(ds.sizes))
        print("Validation calculations completed.")

    finally:
        ds.close()


if __name__ == "__main__":
    main()
