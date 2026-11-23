# Boundary Analysis Validation

## Objective

Validate mesh connectivity, boundary-layer transitions, and
the consistency of calculated boundary paths.

## Validation Checks

- Verify inner boundary edge connectivity.
- Check cell references in `cellsOnEdge`.
- Verify boundary-mask sequences.
- Check path continuity across adjacent cells.
- Identify repeated and shared outer edges.
- Detect non-monotonic paths.
- Inspect geometric ray intersections.

## Current Results

- 207 inner boundary edges were checked.
- All 207 network paths were successfully traced.
- No failed network paths were reported.
- 61 unique physical outer edges were reached.
- 46 outer edges were shared among paths.

## Important Limitation

Successful path tracing does not by itself prove that every
geometric width estimate is physically valid.

Normal-ray distances and non-monotonic paths require separate
inspection before scientific interpretation.

## Scripts

Validation scripts:

`boundary_analysis/scripts/validation/`

Diagnostic scripts:

`boundary_analysis/scripts/diagnostics/`
