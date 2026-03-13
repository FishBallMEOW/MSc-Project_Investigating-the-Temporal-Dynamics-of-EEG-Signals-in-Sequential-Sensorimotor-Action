# Cleanup Inventory

This document records duplicate, historical, and generated material that should be reviewed in a later cleanup pass. Nothing listed here is being removed in this phase.

## Classification Defaults

- `Archive-historical`: names containing `copy`, `backup`, `trash`, or files inside `archive/` or `notebooks/archive/`, unless the file is clearly generated output.
- `Generated-output`: files inside `artifacts/` or any `__pycache__/` directory.
- `Review-needed`: active-adjacent pipeline variants that are still present in `src/pipelines/` but may later move to archive.
- `Keep-active-candidate`: active source files in `src/` and the single working notebook in `notebooks/main/`.

## Inventory Groups

### Notebook Family: `eeg_process_pipeline*.ipynb`

- `notebooks/archive/eeg_process_pipeline.ipynb`: `Archive-historical`
- `notebooks/archive/eeg_process_pipeline copy.ipynb`: `Archive-historical`
- `notebooks/archive/eeg_process_pipeline copy 2.ipynb`: `Archive-historical`
- `notebooks/archive/eeg_process_pipeline_ersd_added.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline copy.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline_backup.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline_backup_Aug28.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline_Aug29.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline_backup_20250829_221800.ipynb`: `Archive-historical`
- `archive/pipeline_backups/Pipeline_backups/eeg_process_pipeline_updated.ipynb`: `Archive-historical`

### Notebook Family: `EEG_processing_pipeline_ver_1.0*.ipynb`

- `notebooks/main/eeg_analysis_main.ipynb`: `Keep-active-candidate`
- `notebooks/archive/EEG_processing_pipeline_ver_1.0.ipynb`: `Archive-historical`
- `notebooks/archive/EEG_processing_pipeline_ver_1.0 copy.ipynb`: `Archive-historical`
- `notebooks/archive/EEG_processing_pipeline_ver_1.0 copy 2.ipynb`: `Archive-historical`

### Root Duplicate Pair

- `archive/legacy_source/root/plot.py`: `Archive-historical`
- `archive/legacy_source/root/plot copy.py`: `Archive-historical`

### Historical Backup Area

- Everything in `archive/pipeline_backups/`: `Archive-historical`

### Generated Duplicate Area

- `artifacts/results/figs/`: `Generated-output`
- `artifacts/results/figs/trash/`: `Generated-output`, explicitly non-canonical
- Repeated figure concepts across `figs/` and `figs/trash/` should be reviewed together in a later pass rather than deleted by filename pattern alone.

### Active Pipeline Area

- `src/pipelines/eeg_processing_organized.py`: `Keep-active-candidate`
- `src/pipelines/eeg_processing.py`: `Review-needed`
- `src/pipelines/ersd_pipeline.py`: `Review-needed`
- `src/pipelines/process.py`: `Review-needed`
- `src/pipelines/processing.py`: `Review-needed`

### Local Data Area

- `data/NIC2/`: active local dataset area for current work
- `data/_archive/Data_from_dataset/`: retained sample EDF/event dataset copy
- all non-`NIC2` folders under `data/_archive/`: `Archive-historical`

## Later Cleanup Actions

The next cleanup pass should:

- compare notebook families to identify the latest meaningful working versions,
- determine whether the remaining support pipeline scripts should stay in `src/pipelines/` or move to archive,
- consolidate data-path assumptions around `data/NIC2/` and keep non-active local data under `data/_archive/`,
- define a policy for large result files and caches,
- move only after a canonical file list has been agreed.
