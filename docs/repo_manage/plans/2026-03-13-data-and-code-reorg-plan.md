# Task Plan: data-and-code-reorg-plan

- Date: 2026-03-13
- Owner: Codex
- Objective: Define the full implementation phases needed to cleanly reorganize the local `data/` area and the remaining root + `OOP_ver/` code into a minimal, professional repository layout.
- Why this matters: The repository still mixes active work, local archives, generated artifacts, notebook copies, and prototype scripts. The next implementation pass needs a fixed destination layout before any moves happen.
- In scope:
  - Final target structure for `data/`, root, and `OOP_ver/`
  - Move sequence for active versus archival material
  - Git tracking policy for active data versus local archive data
  - Acceptance criteria for each cleanup phase
- Out of scope:
  - Executing the moves in this document
  - Deleting historical files without review
  - Refactoring analysis code internals beyond path/import fixes needed by file moves
- Inputs / dependencies:
  - Baseline snapshot commit `c5ce4f3`
  - Path/docs cleanup commit `0222e4b`
  - Existing docs scaffold under `docs/`
- Risks:
  - Breaking working analysis paths while moving scripts and notebooks
  - Accidentally committing local archive datasets or generated artifacts
  - Losing provenance by flattening historical notebook variants too aggressively
- Expected handoff artifacts:
  - One implementation-ready reorganization spec
  - A clear target tree for active code, active data, archives, and generated outputs

## Target Structure

The cleanup implementation should converge to this minimal top-level layout:

- `data/`
  - `NIC2/` as the only active tracked dataset area
  - `_archive/` for all other local datasets and dataset copies
- `docs/`
  - keep the current AGENTS-aligned documentation structure
- `apps/`
  - `data_collection/` for runnable collection and replay utilities currently living in `OOP_ver/data_collection/`
- `pipelines/`
  - main Python analysis scripts promoted from `OOP_ver/`
- `notebooks/`
  - `active/` for the few notebook variants that remain useful
  - `archive/` for historical notebook copies and backups
- `matlab/`
  - MATLAB helpers currently at the repository root
- `artifacts/`
  - local generated outputs moved out of mixed source locations and kept ignored by Git
- `archive/`
  - legacy source files that are preserved for provenance but not part of the active workflow

## Phase 1: Data Reorganization

- Keep `data/NIC2/` as the only active dataset folder.
- Move every other current child of `data/` into `data/_archive/`.
- Move `data/NIC2 - Copy/` into `data/_archive/NIC2 - Copy/` without treating it as active.
- Keep `data/Data_from_dataset/`, `data/Data_from_dataset_Schalk_et_al/`, `data/data_miller/`, `data/Report/`, `data/Simulated_data/`, and the timestamped experiment folders under `data/_archive/`.
- Update `.gitignore` so:
  - `data/_archive/` stays ignored
  - `data/NIC2/` becomes trackable
  - no other `data/` subfolder is meant to remain tracked
- Acceptance:
  - `data/` contains only `NIC2/` and `_archive/`
  - `NIC2/` is the only active dataset location referenced in active docs/code defaults
  - `NIC2 - Copy/` is no longer beside the active dataset

## Phase 2: Generated Artifact Separation

- Create `artifacts/` with subfolders for analysis outputs now scattered across:
  - root `outputs/`
  - `OOP_ver/outputs/`
  - `OOP_ver/results/`
  - `OOP_ver/ersd_outputs/`
- Treat `artifacts/` as local/generated and ignored by Git by default.
- Move `__pycache__/` and any cache directories out of consideration entirely; they remain ignored and untracked.
- Acceptance:
  - source directories no longer mix code and bulk generated outputs
  - Git status does not surface new result exports by default

## Phase 3: Canonical Python Workflow Selection

- Identify the active candidates in `OOP_ver/` and reduce them to a small active set before moving files:
  - likely analysis candidates: `eeg_processing_organized.py`, `eeg_processing.py`, `ersd_pipeline.py`, `process.py`, `processing.py`
  - likely app candidate: `data_collection/eeg_replay_plotter.py`
- For each workflow type, choose one canonical destination:
  - data collection / replay
  - preprocessing / loading
  - ERD/ERS analysis
  - plotting / utilities
- Root-level prototype scripts such as `a.py`, `b.py`, `c.py`, `plot.py`, `plot copy.py`, `game.py`, `game_2classes.py`, `recorder.py`, and `testing.py` should be reviewed and either:
  - promoted into active code,
  - moved to `archive/legacy_source/`,
  - or left as MATLAB/support utilities if still needed
- Acceptance:
  - every root `.py` file and every major `OOP_ver/*.py` file is classified
  - the active Python workflow is represented by a small named set of files, not a folder of variants

## Phase 4: Directory Moves for Active Code

- Move `OOP_ver/data_collection/` into `apps/data_collection/`.
- Move the selected active analysis scripts from `OOP_ver/` into `pipelines/`.
- Move MATLAB helpers `TFR.m`, `trialfun_edf_annotations_mi.m`, and `edfreader.mlx` into `matlab/`.
- Move preserved but non-active root/OOP source files into `archive/legacy_source/`.
- Update imports and path handling only as required by these file moves.
- Acceptance:
  - the root no longer contains scattered executable source files
  - active runnable code lives under `apps/`, `pipelines/`, and `matlab/`
  - archived source is clearly separate from active source

## Phase 5: Notebook Reduction and Archival

- Move notebook families out of `OOP_ver/` into `notebooks/`.
- Keep only a small `notebooks/active/` set that directly supports the canonical workflow.
- Move the rest into `notebooks/archive/`, including:
  - `eeg_process_pipeline*`
  - `EEG_processing_pipeline_ver_1.0*`
  - `Pipeline_backups/*`
- Keep notebook filenames stable during the first move; do not rename aggressively until the active subset is confirmed.
- Acceptance:
  - active notebooks are obvious
  - historical notebook copies no longer sit beside active code

## Phase 6: Final Docs and Repo Hygiene

- Update `README.md`, architecture docs, status docs, and handoff docs to match the new structure.
- Add runbooks for:
  - using `data/NIC2/`
  - running the active replay/data collection app
  - running the active analysis pipeline
- Finalize `.gitignore` for:
  - `data/_archive/`
  - `artifacts/`
  - caches and checkpoints
- Acceptance:
  - a new contributor can find the active data, active code, notebooks, and docs without reading historical filenames
  - Git status stays focused on actual source/documentation changes

## Defaults Locked For Implementation

- `data/NIC2/` is the only active dataset folder.
- `data/NIC2 - Copy/` is not active and should move into `data/_archive/`.
- All non-`NIC2` local datasets should move into `data/_archive/`.
- The implementation pass should not delete archive material unless a later explicit review approves that separately.

