# Boundary Topology Analysis

## Objective

Investigate connectivity and path structure across MPAS boundary
layers using mesh topology.

## Input Data

- `iran.static.nc`
- `bdyMaskCell`
- `bdyMaskEdge`
- `cellsOnEdge`
- `verticesOnEdge`
- Local boundary-width results

## Method

The topology-aware analysis traces paths from inner boundary edges
through successive boundary-mask layers toward the physical boundary.

The analysis checks:

- Cell and edge connectivity
- Boundary-layer mask sequence
- Path continuity
- Shared outer edges
- Non-monotonic paths
- Ray intersections and topology conflicts

## Current Findings

- 207 inner boundary edges were analyzed.
- All 207 network paths were successfully traced.
- 61 unique physical outer edges were reached.
- 46 outer edges were shared by multiple paths.
- Outer edge 123 was used by 15 paths.

## Scripts

Scripts are located in:

`boundary_analysis/scripts/topology/`

Additional diagnostics are located in:

`boundary_analysis/scripts/diagnostics/`

## Outputs

Topology results are stored under:

`boundary_analysis/results/topology/`

The topology-aware method and geometric normal-ray method
must be evaluated separately.
