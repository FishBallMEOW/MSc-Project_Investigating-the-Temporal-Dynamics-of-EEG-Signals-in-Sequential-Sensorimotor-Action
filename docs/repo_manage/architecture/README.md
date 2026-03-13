# Repository Architecture

## Overview

The repository now separates active source, working notebooks, generated artifacts, and historical material into named top-level folders. The purpose of this document is to make the current layout easy to navigate without losing the provenance of the September 2025 submission-period files.

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
- `src/`: active Python source tree.
- `src/pipelines/`: active analysis scripts.
- `src/data_collection/`: active runtime and acquisition area, including experiment, recorder, replay plotter, and channel-renaming utilities.
- `notebooks/main/`: the single retained working notebook for active analysis.
- `notebooks/archive/`: historical notebook variants preserved for provenance.
- `archive/`: historical source files, MATLAB files, reports, and pipeline backups.
- `artifacts/`: generated output moved out of source folders.
- `__pycache__/`: generated cache artifacts.
- `archive/legacy_source/root/`: root prototypes and standalone experiments moved out of the active root.
- `archive/matlab/`: archived MATLAB support files.

## Active Source Map

- `src/pipelines/eeg_processing_organized.py`: current canonical analysis entrypoint.
- `src/pipelines/eeg_processing.py`, `src/pipelines/ersd_pipeline.py`, `src/pipelines/process.py`, `src/pipelines/processing.py`: support or review-needed analysis variants retained beside the canonical entrypoint for now.
- `src/data_collection/`: grouped collection/runtime area kept separate from the general pipeline folder.
- `src/data_collection/mapping.json`: channel mapping support file used by the collection utilities.
- `notebooks/main/eeg_analysis_main.ipynb`: the single retained active notebook.

## Archive And Artifact Map

- `archive/legacy_source/root/`: archived root prototypes and unrelated experiments.
- `archive/legacy_source/oop_ver/`: legacy source preserved from the former `OOP_ver/` area.
- `archive/pipeline_backups/`: preserved notebook backup sets.
- `archive/reports/`: report-supporting exports retained for reference.
- `archive/matlab/`: archived MATLAB helpers.
- `artifacts/outputs/`, `artifacts/results/`, `artifacts/ersd_outputs/`, `artifacts/root_outputs/`: generated output separated from active source.

## Current Interpretation

- Repo-local data should be referenced through `data/` paths rather than the old external `D:/.../MSc_Project/Data/` directory.
- `data/NIC2/` is the default active data source.
- `data/_archive/` contains the non-active local data that remains available for provenance and occasional reference.
- Active work should start in `src/` and `notebooks/main/`.
- Historical material remains available but should not be mixed back into active source by default.
- Any future cleanup should use the labels above consistently so that archive, generated, and active-candidate files are not mixed.

