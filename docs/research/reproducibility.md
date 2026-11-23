# Research Reproducibility

## Requirements

Python dependencies are listed in:

`requirements-research.txt`

Install them using:

```bash
python -m pip install -r requirements-research.txt


## Input Data

The research requires:

- `iran.static.nc`
- Generated intermediate CSV files where required by individual scripts

The NetCDF input is not tracked by Git because generated scientific
data files are excluded through `.gitignore`.

## Scripts

Research scripts are organized under:

- `boundary_analysis/scripts/`
- `domain_analysis/scripts/`
- `scripts/`

## Execution

Run scripts from the project root unless their command-line
arguments specify otherwise.

Use each script's `--help` option when available to inspect
supported arguments.

## Results

Generated reports, CSV files, and figures are stored under:

- `boundary_analysis/results/`
- `domain_analysis/results/`

## Reproducibility Notes

Record the input mesh version, script revision, Python version,
dependency versions, and execution parameters when producing
new scientific results.

Do not overwrite reference datasets or previously validated
results during experiments.
