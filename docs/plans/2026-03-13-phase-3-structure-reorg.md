# Task Plan: phase-3-structure-reorg

- Date: 2026-03-13
- Owner: Codex
- Objective: Define the implementation-ready repository reorganization that dissolves `OOP_ver/` and redistributes active code, notebooks, artifacts, and legacy material into a clear top-level structure.
- Why this matters: The repository still mixes active analysis code, historical variants, generated outputs, prototype scripts, and archived MATLAB work across the root and `OOP_ver/`. The next implementation pass needs a fixed destination layout before any file moves occur.
- In scope:
  - Final destination structure for the remaining root code and `OOP_ver/`
  - Canonical placement of active pipelines, data collection tools, notebooks, artifacts, and archives
  - File classification defaults for root scripts, `OOP_ver/*.py`, notebooks, MATLAB files, and generated outputs
  - Move sequencing and acceptance criteria for the reorganization pass
- Out of scope:
  - Executing the file moves in this document
  - Deleting legacy files
  - Refactoring pipeline internals beyond path/import fixes required by directory moves
  - Redesigning notebook contents
- Inputs / dependencies:
  - Baseline snapshot commit `c5ce4f3`
  - Path/docs cleanup commit `0222e4b`
  - Phase-2 docs commit `f2ff636`
  - Phase-2 structure commit `2e26ac6`
  - Active NIC2 dataset commit `b9d6031`
  - Existing cleanup docs under `docs/architecture/` and `docs/status/`
- Risks:
  - Moving scripts too early could break hardcoded relative paths
  - Historical notebook variants may contain small but important late-stage differences
  - Root prototypes may still duplicate logic that is not yet fully represented in the selected canonical pipeline
- Validation:
  - Every current root code file and every major `OOP_ver/` file has a destination
  - `OOP_ver/` has no remaining active responsibilities after the move plan
  - The resulting top-level layout clearly separates active source, notebooks, archives, data, docs, and generated artifacts
- Expected handoff artifacts:
  - One implementation-ready move plan
  - One active task note for the future move pass
  - Updated status/handoff docs pointing to the phase-3 structure plan

## Locked Decisions

- Dissolve `OOP_ver/` completely and redistribute its contents into top-level named folders.
- Use `src/` as the active code root.
- Keep data-collection code grouped under `src/data_collection/`, not flattened into a general scripts folder.
- Use `src/pipelines/` for the canonical analysis entrypoint and related workflow modules.
- Treat `OOP_ver/eeg_processing_organized.py` as the canonical pipeline entrypoint for now.
- Keep one active notebook only: rename `OOP_ver/EEG_processing_pipeline_ver_1.0 copy 3.ipynb` to `notebooks/main/eeg_analysis_main.ipynb`.
- Archive all other notebooks.
- Treat root prototypes and unrelated experiments as archive-by-default.
- Archive the MATLAB files rather than keeping an active `matlab/` workspace.

## Target Top-Level Structure

The repository should converge to this layout:

- `data/`
  - `NIC2/` as the only active tracked dataset
  - `_archive/` for local archived datasets
- `docs/`
  - keep the current AGENTS-aligned documentation structure
- `src/`
  - `pipelines/`
  - `data_collection/`
  - `shared/` for small reusable helpers only if needed during the move pass
- `notebooks/`
  - `main/`
  - `archive/`
- `artifacts/`
  - generated outputs moved out of source folders and ignored by Git
- `archive/`
  - `legacy_source/`
  - `matlab/`
  - `reports/`
  - `pipeline_backups/`

## File Classification

### Active source destinations

- `OOP_ver/eeg_processing_organized.py`
  - destination: `src/pipelines/eeg_processing_organized.py`
  - role: canonical pipeline entrypoint for now
- `OOP_ver/eeg_processing.py`
  - destination: `src/pipelines/eeg_processing.py`
  - role: support or review-needed pipeline variant
- `OOP_ver/ersd_pipeline.py`
  - destination: `src/pipelines/ersd_pipeline.py`
  - role: support pipeline focused on ERD/ERS analysis
- `OOP_ver/process.py`
  - destination: `src/pipelines/process.py`
  - role: review-needed support workflow script
- `OOP_ver/processing.py`
  - destination: `src/pipelines/processing.py`
  - role: review-needed support workflow script
- `OOP_ver/data_collection/`
  - destination: `src/data_collection/`
  - role: active runtime/data collection area that remains grouped as its own folder

### Main notebook destination

- `OOP_ver/EEG_processing_pipeline_ver_1.0 copy 3.ipynb`
  - destination: `notebooks/main/eeg_analysis_main.ipynb`
  - role: single active notebook retained for the main analysis workflow

### Archive-by-default destinations

- Root prototypes and experiments:
  - `a.py`
  - `b.py`
  - `c.py`
  - `plot.py`
  - `plot copy.py`
  - `game.py`
  - `game_2classes.py`
  - `recorder.py`
  - `testing.py`
  - destination: `archive/legacy_source/root/`
- OOP legacy experiment:
  - `OOP_ver/a.py`
  - destination: `archive/legacy_source/oop_ver/`
- Notebook families other than the main notebook:
  - `OOP_ver/EEG_processing_pipeline_ver_1.0.ipynb`
  - `OOP_ver/EEG_processing_pipeline_ver_1.0 copy.ipynb`
  - `OOP_ver/EEG_processing_pipeline_ver_1.0 copy 2.ipynb`
  - `OOP_ver/eeg_process_pipeline.ipynb`
  - `OOP_ver/eeg_process_pipeline copy.ipynb`
  - `OOP_ver/eeg_process_pipeline copy 2.ipynb`
  - `OOP_ver/eeg_process_pipeline_ersd_added.ipynb`
  - `OOP_ver/ERSD_after_preproc.ipynb`
  - destination: `notebooks/archive/`
- MATLAB files:
  - `TFR.m`
  - `trialfun_edf_annotations_mi.m`
  - `edfreader.mlx`
  - destination: `archive/matlab/`
- Historical support material:
  - `OOP_ver/Pipeline_backups/`
  - destination: `archive/pipeline_backups/`
  - `OOP_ver/Report/`
  - destination: `archive/reports/`
  - `OOP_ver/README_ersd_pipeline.txt`
  - destination: `archive/legacy_source/oop_ver/README_ersd_pipeline.txt`

### Generated artifact destinations

- `outputs/`
  - destination: `artifacts/root_outputs/`
- `OOP_ver/outputs/`
  - destination: `artifacts/outputs/`
- `OOP_ver/results/`
  - destination: `artifacts/results/`
- `OOP_ver/ersd_outputs/`
  - destination: `artifacts/ersd_outputs/`
- cache directories such as `__pycache__/`
  - remain ignored and out of scope for source moves

## Move Strategy

### Phase 3A: Prepare destination folders

- Create the top-level destination folders:
  - `src/pipelines/`
  - `src/data_collection/`
  - `notebooks/main/`
  - `notebooks/archive/`
  - `artifacts/`
  - `archive/legacy_source/root/`
  - `archive/legacy_source/oop_ver/`
  - `archive/matlab/`
  - `archive/reports/`
  - `archive/pipeline_backups/`
- Update `.gitignore` so `artifacts/` is ignored by default.

### Phase 3B: Move active source first

- Move `OOP_ver/data_collection/` into `src/data_collection/`.
- Move the selected pipeline scripts into `src/pipelines/`.
- Keep filenames stable on the first move unless a path/import fix requires a rename.
- Update imports and path handling only as required by the new locations.

### Phase 3C: Move notebooks

- Rename and move the main notebook to `notebooks/main/eeg_analysis_main.ipynb`.
- Move every other notebook out of `OOP_ver/` into `notebooks/archive/`.
- Keep archived notebook filenames stable to preserve provenance.

### Phase 3D: Move archives and generated outputs

- Move root prototype scripts into `archive/legacy_source/root/`.
- Move `OOP_ver/a.py` and text leftovers into `archive/legacy_source/oop_ver/`.
- Move MATLAB files into `archive/matlab/`.
- Move `OOP_ver/Pipeline_backups/` and `OOP_ver/Report/` into their archive destinations.
- Move generated outputs into `artifacts/`.

### Phase 3E: Remove the now-empty `OOP_ver/` container

- After all active code, notebooks, archives, and artifacts are redistributed, remove the empty `OOP_ver/` directory from the active layout.
- Validate that no docs still describe `OOP_ver/` as an active workspace.

## Acceptance Criteria

- `OOP_ver/` is dissolved and no longer used as an active workspace.
- Active Python code lives under `src/pipelines/` and `src/data_collection/`.
- The only active notebook is `notebooks/main/eeg_analysis_main.ipynb`.
- Root-level prototype scripts are no longer mixed with active source.
- MATLAB files, reports, and pipeline backups are visibly archival.
- Generated outputs live under `artifacts/` instead of mixed source folders.
- `README.md`, architecture docs, status docs, and handoffs reflect the new structure.

## Post-Move Follow-Up

- Reassess whether `src/pipelines/eeg_processing.py`, `src/pipelines/process.py`, and `src/pipelines/processing.py` should remain active support files or be archived in a later pruning pass.
- Add small local README files for `src/pipelines/` and `src/data_collection/`.
- Add runbooks for:
  - running the canonical pipeline
  - running the data collection utilities
  - understanding the notebook/archive split
