# MSc Project: Temporal Dynamics of EEG Signals in Sequential Sensorimotor Action

This repository is an MSc project workspace and archive for EEG collection, preprocessing, ERD/ERS analysis, plotting, and related prototype scripts created around the September 2025 submission period.

The repository root should be treated as the umbrella project. The primary analysis workspace is `OOP_ver/`, which contains the densest pipeline, notebook, and result history.

## Start Here

- Documentation entrypoint: `docs/index.md`
- Primary analysis workspace: `OOP_ver/`
- Active local dataset area: `data/NIC2/`
- Sample EDF dataset copy: `data/Data_from_dataset/`

## Directory Guide

- `OOP_ver/`: main analysis and notebook workspace
- `data/`: local datasets moved into the repository workspace
- `data/NIC2/`: current primary dataset area for active work
- `data/Data_from_dataset/`: EDF inputs and related dataset sample files
- other folders under `data/`: local archival or exploratory datasets pending later cleanup
- `outputs/`: generated output from root-level experiments
- `OOP_ver/outputs/`: generated pipeline outputs
- `OOP_ver/results/`: generated analysis results, tables, plots, and large artifacts
- `OOP_ver/ersd_outputs/`: generated ERD/ERS output artifacts
- Root-level `.py` files: mixed standalone experiments and prototypes pending later review

## Current Organization Policy

- This repository is being organized documentation-first.
- Dataset paths have been moved from the old external `D:/user/Files_without_backup/MSc_Project/Data/` location into the local `data/` folder.
- Code defaults now point at repo-local data paths instead of the old absolute external path.
- `data/` is treated as local-only and is ignored by Git to avoid accidentally committing the full dataset tree.
- Existing code, notebooks, datasets, and outputs are being mapped before any cleanup or deletion.
- Duplicate and historical versions are tracked in docs as cleanup candidates, not removed in this pass.
