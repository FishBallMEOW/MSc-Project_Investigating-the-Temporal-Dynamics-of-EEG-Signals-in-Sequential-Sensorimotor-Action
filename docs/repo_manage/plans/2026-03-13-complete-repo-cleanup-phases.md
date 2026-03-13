# Task Plan: complete-repo-cleanup-phases

- Date: 2026-03-13
- Owner: Codex
- Objective: Define the full set of phases needed to make the repository clean, reproducible, and easier to navigate after the submission-period snapshot.
- Why this matters: The repository still mixes active analysis code, notebook variants, generated outputs, local datasets, and historical prototypes from multiple late-stage iterations.
- In scope:
  - Multi-phase roadmap for repository cleanup
  - Data organization strategy
  - Duplicate notebook/script review strategy
  - Generated artifact and Git tracking strategy
- Out of scope:
  - Executing the later cleanup phases in this document
  - Deleting data or historical files immediately
  - Declaring a canonical analysis workflow before review
- Inputs / dependencies:
  - Baseline snapshot commit `c5ce4f3`
  - Current docs scaffold under `docs/`
  - Repo-local `data/` folder
- Risks:
  - Removing too aggressively and losing submission-period context
  - Breaking analysis paths while consolidating scripts and notebooks
  - Accidentally versioning large local data or result artifacts
- Steps:
  - Phase 1: stabilize paths and documentation
  - Phase 2: inventory active vs historical code and notebooks
  - Phase 3: define archival boundaries for local data and generated outputs
  - Phase 4: consolidate canonical workflows and retire duplicates
  - Phase 5: harden repo hygiene for future work
- Validation:
  - Each phase has explicit deliverables and acceptance checks
  - No phase depends on undocumented assumptions about current files
- Expected handoff artifacts:
  - Cleanup roadmap usable by a later implementer

## Phase Roadmap

### Phase 1: Path Stabilization and Documentation

- Finish replacing hardcoded external data paths with repo-local `data/` references.
- Update docs and README so the moved data layout is explicit.
- Add Git ignore rules for local datasets and cache artifacts.
- Acceptance:
  - no source files still point at `D:/user/Files_without_backup/MSc_Project/Data/`
  - README and architecture docs describe `data/NIC2/` as the current active dataset area

### Phase 2: Workflow Inventory

- Compare root scripts against `OOP_ver/` pipeline files and notebook families.
- Label each script/notebook as active candidate, archive candidate, generated, or unclear.
- Identify likely canonical entrypoints for:
  - EDF analysis
  - CSV replay/data collection
  - ERD/ERS analysis
- Acceptance:
  - every root `.py` file and every `OOP_ver/*.ipynb` file is classified
  - a short candidate list exists for canonical workflows

### Phase 3: Data and Artifact Boundaries

- Decide what stays under `data/` as active local material.
- Decide what parts of `data/` should be treated as archive-only, external-only, or removed from the repo workspace later.
- Separate long-term generated outputs from one-off exports and caches.
- Acceptance:
  - `data/NIC2/` is confirmed as active or replaced by a better-defined active subset
  - non-active data folders have a documented archive rule
  - results/output/cache handling rules are written down

### Phase 4: Canonicalization and Archival Moves

- Move historical notebook copies and backups into clearly archival locations if they are still mixed with active files.
- Rename or annotate canonical scripts and notebooks so the default workflow is obvious.
- Remove or quarantine redundant root-level prototype files only after they are cross-checked against `OOP_ver/`.
- Acceptance:
  - a new contributor can identify the main workflow without reading backup filenames
  - archive-only materials no longer sit beside active candidates without labels

### Phase 5: Repo Hygiene Hardening

- Finalize `.gitignore` for local data, caches, and generated artifacts.
- Add lightweight runbooks for the main workflows.
- Update docs status, handoff, and architecture notes to match the cleaned structure.
- Acceptance:
  - Git status stays focused on real source/doc changes
  - key workflows are documented and reproducible enough for later continuation

