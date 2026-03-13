# MSc Project: Temporal Dynamics of EEG Signals in Sequential Sensorimotor Action

This repository is an MSc project workspace and archive for EEG collection, preprocessing, ERD/ERS analysis, plotting, and related prototype scripts created around the September 2025 submission period.

The repository root is the umbrella project. Active code now lives under `src/`, the single retained working notebook lives under `notebooks/main/`, generated outputs are separated under `artifacts/`, and historical material is preserved under `archive/`.

## Start Here

- Documentation entrypoint: `docs/index.md`
- Repository-management docs: `docs/repo_manage/index.md`
- Active analysis code: `src/pipelines/`
- Active collection utilities: `src/data_collection/`
- Main notebook: `notebooks/main/eeg_analysis_main.ipynb`
- Active local dataset area: `data/NIC2/`
- Local archive dataset area: `data/_archive/`

## Directory Guide

- `data/`: local datasets moved into the repository workspace
- `data/NIC2/`: current primary dataset area for active work
- `data/_archive/`: local archival and exploratory datasets not used in the active workflow
- `src/pipelines/`: active analysis scripts, with `eeg_processing_organized.py` as the current canonical entrypoint
- `src/data_collection/`: runtime collection, replay, recording, and channel-renaming utilities
- `notebooks/main/`: the single retained working notebook
- `notebooks/archive/`: historical notebook variants kept for provenance
- `archive/`: legacy source files, MATLAB helpers, reports, and pipeline backups
- `artifacts/`: generated outputs, result tables, and plots kept outside active source
- `docs/repo_manage/`: repo-management docs including plans, status, architecture notes, and handoffs
- `docs/report/`: thesis, abstract, and report PDFs
- `docs/presentation/`: progress and final presentation slide decks
- `docs/figures/`: figure assets and exported reference visuals

## Current Organization Policy

- Dataset paths have been moved from the old external `D:/user/Files_without_backup/MSc_Project/Data/` location into the local `data/` folder.
- Code defaults now point at repo-local data paths instead of the old absolute external path.
- `data/NIC2/` is the only active dataset folder intended to remain trackable.
- `data/_archive/` is kept local and ignored by Git.
- `artifacts/` is treated as generated output and ignored by Git.
- Historical code and notebooks are preserved in `archive/` and `notebooks/archive/` rather than mixed with active source.
