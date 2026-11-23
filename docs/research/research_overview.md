# MPAS Boundary Research

## 1. Objective

This research investigates the geometry, topology, and effective width
of the lateral boundary zone in the MPAS limited-area mesh.

## 2. Input Data

- `iran.static.nc`
- `boundary_analysis/results/geometric/local_boundary_width.csv`

The NetCDF input is generated separately and is not tracked by Git.

## 3. Research Components

- Geometry analysis
- Boundary-layer analysis
- Local boundary-width calculation
- Topology-aware ray tracing
- Path and intersection diagnostics
- Geometric and network-width comparison
- Boundary topology validation

## 4. Main Results

The local network-width analysis identified 207 inner boundary edges.
All 207 paths were successfully traced.

The calculated widths were:

- Minimum: 13.70 km
- Median: 15.15 km
- Mean: 15.54 km
- Maximum: 18.52 km

## 5. Methodological Notes

Geometric normal-ray results require additional validation.
Large ray distances must not be interpreted as verified boundary widths.

The topology-aware network method and geometric normal method
represent different approaches and must be evaluated separately.

## 6. Reproducibility

Research scripts are located under `boundary_analysis/scripts/`
and `domain_analysis/scripts/`.

Python dependencies are listed in `requirements-research.txt`.
