# Boundary Geometry Analysis

## Objective

Analyze the geometric structure of the MPAS limited-area boundary
and identify the edges and cells associated with each boundary layer.

## Input Data

- `iran.static.nc`
- `bdyMaskCell`
- `bdyMaskEdge`
- `cellsOnEdge`
- `verticesOnEdge`
- `latCell`, `lonCell`
- `latVertex`, `lonVertex`

## Analysis

The geometry scripts examine:

- Boundary edge locations
- Boundary-layer structure
- Inner boundary edges
- Local boundary-width calculations
- Geometric normal directions and intersections

## Local Network Width

The local network method traces paths across successive boundary
mask layers, from the inner interface toward the outer boundary.

The current analysis identified 207 inner edges. All paths were
successfully traced.

The calculated width statistics are:

- Minimum: 13.70 km
- Median: 15.15 km
- Mean: 15.54 km
- Maximum: 18.52 km

## Geometric Normal Method

Normal-ray calculations are retained as a separate diagnostic method.
Their large calculated distances require further investigation and
must not be treated as validated local boundary widths.

## Output

Results are stored under:

`boundary_analysis/results/geometric/`
