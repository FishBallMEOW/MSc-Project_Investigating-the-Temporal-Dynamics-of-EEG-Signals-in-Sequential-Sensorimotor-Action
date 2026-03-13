# Cleanup Inventory

This document records duplicate, historical, and generated material that should be reviewed in a later cleanup pass. Nothing listed here is being removed in this phase.

## Classification Defaults

- `Archive-historical`: names containing `copy`, `backup`, `trash`, or files inside `OOP_ver/Pipeline_backups/`, unless the file is clearly generated output.
- `Generated-output`: files inside `outputs/`, `OOP_ver/outputs/`, `OOP_ver/ersd_outputs/`, `OOP_ver/results/`, or any `__pycache__/` directory.
- `Review-needed`: root-level standalone scripts and utilities whose long-term role is not yet clear.
- `Keep-active-candidate`: non-copy Python pipeline files in `OOP_ver/`.

## Inventory Groups

### Notebook Family: `eeg_process_pipeline*.ipynb`

- `OOP_ver/eeg_process_pipeline.ipynb`: `Keep-active-candidate`
- `OOP_ver/eeg_process_pipeline copy.ipynb`: `Archive-historical`
- `OOP_ver/eeg_process_pipeline copy 2.ipynb`: `Archive-historical`
- `OOP_ver/eeg_process_pipeline_ersd_added.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline copy.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline_backup.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline_backup_Aug28.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline_Aug29.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline_backup_20250829_221800.ipynb`: `Archive-historical`
- `OOP_ver/Pipeline_backups/eeg_process_pipeline_updated.ipynb`: `Archive-historical`

### Notebook Family: `EEG_processing_pipeline_ver_1.0*.ipynb`

- `OOP_ver/EEG_processing_pipeline_ver_1.0.ipynb`: `Keep-active-candidate`
- `OOP_ver/EEG_processing_pipeline_ver_1.0 copy.ipynb`: `Archive-historical`
- `OOP_ver/EEG_processing_pipeline_ver_1.0 copy 2.ipynb`: `Archive-historical`
- `OOP_ver/EEG_processing_pipeline_ver_1.0 copy 3.ipynb`: `Archive-historical`

### Root Duplicate Pair

- `plot.py`: `Review-needed`
- `plot copy.py`: `Archive-historical`

### Historical Backup Area

- Everything in `OOP_ver/Pipeline_backups/`: `Archive-historical`

### Generated Duplicate Area

- `OOP_ver/results/figs/`: `Generated-output`
- `OOP_ver/results/figs/trash/`: `Generated-output`, explicitly non-canonical
- Repeated figure concepts across `figs/` and `figs/trash/` should be reviewed together in a later pass rather than deleted by filename pattern alone.

### Local Data Area

- `data/NIC2/`: active local dataset area for current work
- `data/Data_from_dataset/`: retained sample EDF/event dataset copy
- other folders under `data/`: `Archive-historical` candidates until a later review decides what should remain nearby, move elsewhere, or be documented separately

## Later Cleanup Actions

The next cleanup pass should:

- compare notebook families to identify the latest meaningful working versions,
- determine whether any root-level scripts duplicate `OOP_ver/` behavior,
- consolidate data-path assumptions around `data/NIC2/` and a documented archival policy for the rest of `data/`,
- define a policy for large result files and caches,
- move only after a canonical file list has been agreed.
