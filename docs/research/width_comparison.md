# Boundary Width Comparison

## Objective

Compare local network-based boundary widths with geometric
normal-ray estimates.

## Methods

### 1. Local Network Method

Uses mesh connectivity and boundary-mask layers to trace paths
from inner boundary edges to physical outer boundary edges.

### 2. Geometric Normal Method

Uses geometric directions and ray intersections to estimate
distances across the boundary zone.

## Local Network Results

- Number of inner edges: 207
- Successful paths: 207
- Failed paths: 0
- Minimum width: 13.70 km
- Median width: 15.15 km
- Mean width: 15.54 km
- Maximum width: 18.52 km

## Interpretation

The network method produces local widths consistent with the
boundary-layer structure.

The normal-ray method has produced much larger distances in the
current analysis. These values require further geometric and
topological validation.

The two methods should not be treated as interchangeable.

## Output Files

- `boundary_width_comparison.csv`
- `boundary_width_comparison_report.txt`
- `geometric_boundary_width.csv`
- `geometric_boundary_width_report.txt`

Results are stored under:

`boundary_analysis/results/`
