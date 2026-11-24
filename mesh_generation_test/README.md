# Spherical MPAS Mesh Generation Test

## Purpose
This directory documents a global spherical mesh-generation experiment using JIGSAW through MPAS-Tools. The experiment uses a uniform target cell width of 100 km. It is a technical validation, not an operational variable-resolution atmospheric mesh.

## Software
- MPAS-Tools: 2.2.2
- JIGSAW-Python: 1.1.1
- Python: 3.11
- MPAS mesh converter: MpasMeshConverter.x

## Input configuration
- Earth radius: 6,371,000 m
- Longitude: -180 to 180 degrees
- Latitude: -90 to 90 degrees
- Sampling interval: 1 degree
- Target cell width: 100 km
- JIGSAW mesh dimension: 2

The one-degree sampling defines the cell-width field. It does not constrain output mesh vertices to a one-degree grid.

## Generated mesh
| Quantity | Value |
|---|---:|
| Cells | 59,226 |
| Edges | 177,672 |
| Vertices | 118,448 |
| Maximum edges per cell | 8 |
| Vertex degree | 3 |

## Topological validation
- One connected component
- Zero boundary edges
- Zero non-manifold edges
- Euler characteristic V - E + F = 2
- All raw JIGSAW vertices lie on the sphere

MPAS conversion reported zero incomplete cells and zero obtuse dual triangles.

## Geometric validation
| Metric | Result |
|---|---:|
| Theoretical sphere area | 510,064,471.910 km2 |
| Sum of cell areas | 510,064,471.920 km2 |
| Median dcEdge | 99.776 km |
| Median dvEdge | 57.631 km |
| Median cell area | 8,618.894 km2 |

The MPAS file stores dcEdge and dvEdge in meters. Values above are converted to kilometers.

## Mesh quality
cellQuality is the ratio of the shortest to longest edge of each primal cell.

| Statistic | Value |
|---|---:|
| Minimum | 0.29270 |
| P05 | 0.57881 |
| Median | 0.92915 |
| P95 | 0.98862 |
| Maximum | 0.99983 |

There are 1,021 cells (1.724%) with cellQuality below 0.5. Cell 30883 has a shortest edge of 22.955 km and a longest edge of 78.423 km.

| Dual-triangle metric | Median | Minimum |
|---|---:|---:|
| triangleQuality | 0.98004 | 0.70763 |
| triangleAngleQuality | 0.96718 | 0.55363 |

## Reproducibility
Run from this directory:
```bash
conda run -n mpas_mesh python validate_spherical_mesh.py
```

The script reads the NetCDF file and does not modify the mesh.

## Limitations and next steps
1. Investigate low-quality primal cells and their spatial distribution.
2. Verify gridSpacing meaning and units before using it as a physical distance.
3. Test a spatially varying cell-width field.
4. Evaluate refinement and transition quality.
5. Validate the mesh with MPAS atmospheric initialization and runtime tests.
