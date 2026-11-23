# MPAS Limited-Area Model Execution

## Workflow

The model workflow consists of:

1. GFS data download
2. WPS preprocessing
3. Limited-area mesh generation
4. Initial condition generation
5. Lateral boundary condition generation
6. LBC water-vapor cleaning
7. MPAS atmosphere model execution
8. Post-processing and research analysis

## Reference Configuration

- Initial time: 2026-10-31 00:00 UTC
- Forecast duration: 48 hours
- Time step: 90 seconds
- MPI processes: 4
- Forecast steps: 1920
- LBC interval: 3 hours
- History interval: 6 hours
- Diagnostic interval: 3 hours
- Vertical levels: 55

## GFS and WPS

Download GFS forecast data using the project download scripts.

Run the WPS programs in order:

1. `geogrid.exe`
2. `ungrib.exe`
3. `metgrid.exe`

Verify the generated meteorological fields before initialization.

## Mesh and Initialization

Generate the limited-area mesh using `create_region`.

Run the MPAS initialization workflow to generate the initial
conditions and required static and graph files.

## Lateral Boundary Conditions

Generate LBC files using the initialization workflow.

Preserve original LBC files. Cleaned water-vapor files must remain
separate from the original files.

## Atmosphere Model

Run the model with the configured MPI process count.

Example:

`mpirun --oversubscribe -n 4 ./atmosphere_model`

Check the log for errors and critical messages after execution.

## Data Preservation

Do not overwrite validated reference runs, original LBC files,
cleaned LBC datasets, or static mesh files during experiments.

## Important

The configuration above describes the validated reference case.
Test experiments must use separate run directories.
