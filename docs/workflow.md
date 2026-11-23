# MPAS Limited-Area Project Workflow

## Project Structure

- `limited_area/`: Limited-area mesh and region utilities
- `boundary_analysis/`: Boundary geometry and topology research
- `domain_analysis/`: Mesh domain visualization
- `scripts/`: Data preparation and utility scripts
- `config/`: Runtime configuration files
- `docs/`: Project and research documentation

## Operational Workflow

### 1. GFS Data

Configure the forecast cycle and download parameters in:

`config/gfs.env`

Use the project GFS download script to retrieve input data.

### 2. WPS Preprocessing

Process GFS data using:

- `geogrid.exe`
- `ungrib.exe`
- `metgrid.exe`

### 3. Limited-Area Mesh

Use `create_region` and the region specification file
to generate the limited-area mesh.

### 4. Initial and Boundary Conditions

Run the MPAS initialization workflow to generate initial
conditions and lateral boundary conditions.

### 5. LBC Preparation

Use `scripts/clean_lbc_qv.py` when water-vapor cleaning
is required. Preserve the original LBC files.

### 6. Forecast

Run `atmosphere_model` using the configured MPI processes.

### 7. Research

Run geometry, topology, validation, and comparison scripts
under `boundary_analysis/scripts/`.

Generate domain visualizations using:

`domain_analysis/scripts/plot_mpas_domain.py`

## Documentation

- Model execution: `docs/model_execution.md`
- Research overview: `docs/research/research_overview.md`
- Research methods: `docs/research/`

## Data and Version Control

Large input datasets, NetCDF files, GRIB files, and model run
outputs are excluded from Git.

Preserve validated scientific inputs and outputs separately.
