# MMS workflow

This folder contains the new MMS analysis layer for the 4 Dec 2021 eclipse project.

## What it does

- Loads MMS MEC, FGM, and FPI products through `pyspedas`
- Uses writable local cache directories for `SPACEPY`, `MPLCONFIGDIR`, and MMS data
- Renders:
  - a not-to-scale MMS location schematic
  - a stacked MMS time-series figure aligned with eclipse timing

## Runtime notes

- `pyspedas` is installed in the `eclipse` conda environment.
- `spacepy` needs a writable `SPACEPY` directory. The code sets that to a local cache under this folder.
- AL/SML overlay source is still a project-level decision. The loader helper for SuperMAG is provided, but the final source should match the manuscript-wide choice.

## Suggested entry point

Use `build_mms_figures(...)` from `workflow.py` once the MMS probe list, eclipse obscuration series, and optional AL/SML series are ready.

