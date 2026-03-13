# Repository Architecture

## Overview

The repository currently has two layers:

- An umbrella root containing datasets, standalone scripts, and project-level supporting files.
- A more concentrated `OOP_ver/` subproject that appears to be the main analysis workspace developed near submission time.

The purpose of this document is to describe the current layout without implying that every file is still active.

## Material Classes

- `Keep-active-candidate`: likely current workflow material that should stay available for later consolidation.
- `Archive-historical`: preserved historical snapshots, backups, and versioned copies.
- `Generated-output`: derived artifacts such as plots, result tables, cache files, and export outputs.
- `Review-needed`: items that are still part of the repository but whose long-term role is unclear.

## Top-Level Repo Map

- `README.md`: navigation page for the umbrella project.
- `AGENTS.md`: repository operating instructions.
- `data/`: local dataset area moved into the repository after the original work was created.
- `data/NIC2/`: primary active dataset area for current work.
- `data/_archive/`: archived local datasets, dataset copies, reports, and exploratory data not part of the active workflow.
- `OOP_ver/`: primary analysis workspace.
- `outputs/`: generated output from root-level experiments.
- `__pycache__/`: generated cache artifacts.
- Root-level Python scripts such as `a.py`, `b.py`, `c.py`, `plot.py`, `plot copy.py`, `game.py`, `game_2classes.py`, `recorder.py`, and `testing.py`: `Review-needed` standalone experiments or prototypes pending later review.
- Root-level MATLAB files such as `TFR.m`, `trialfun_edf_annotations_mi.m`, and `edfreader.mlx`: supporting analysis artifacts and utilities pending later review.

## OOP_ver Map

- `data_collection/`: clearest runtime and acquisition area; includes experiment, recorder, replay plotter, and channel-renaming utilities.
- `eeg_processing.py`, `eeg_processing_organized.py`, `ersd_pipeline.py`, `process.py`, `processing.py`: `Keep-active-candidate` pipeline scripts that look like the most current analysis code, without declaring a single canonical file yet.
- `*.ipynb` in `OOP_ver/`: notebook workspace spanning exploratory, iterative, and late-stage submission work.
- `Pipeline_backups/`: `Archive-historical` notebook snapshots and backups.
- `outputs/`: `Generated-output` pipeline outputs.
- `results/`: `Generated-output` results, figures, tables, and large analysis exports.
- `results/figs/trash/`: non-canonical generated output kept for historical reference only.
- `ersd_outputs/`: `Generated-output` ERD/ERS artifacts.
- `Report/`: report-supporting exported tables and related materials.
- `__pycache__/`: generated cache artifacts.

## Current Interpretation

- `OOP_ver/` should be the default place to look first when reconstructing the analysis workflow.
- Repo-local data should be referenced through `data/` paths rather than the old external `D:/.../MSc_Project/Data/` directory.
- `data/NIC2/` is the default active data source.
- `data/_archive/` contains the non-active local data that remains available for provenance and occasional reference.
- Root-level scripts should not be deleted or promoted yet; they need later comparison against the `OOP_ver/` workflow.
- Any future cleanup should use the labels above consistently so that archive, generated, and active-candidate files are not mixed.
