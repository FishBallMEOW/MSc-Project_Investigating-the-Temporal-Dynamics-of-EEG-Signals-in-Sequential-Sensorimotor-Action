# Current Status

## Phase

Post-submission archival cleanup and organization.

## Top Priorities

- Document the current repository structure.
- Treat `OOP_ver/` as the primary analysis workspace.
- Normalize old external data paths to the repo-local `data/` folder.
- Treat `data/NIC2/` as the current active dataset area and `data/_archive/` as the local archive boundary.
- Separate active-looking workflow files from historical snapshots and generated artifacts.
- Prepare for a later reviewed cleanup of duplicate notebooks, backup files, and generated results.

## Active Task

- `light-reorg-baseline`: documentation-first repo mapping and cleanup inventory.
- `data-and-code-reorg-plan`: implementation-ready plan for data archiving and source layout cleanup.
- `data-phase-2-reorg`: completed local data reorganization so only `data/NIC2/` remains active.
- `phase-3-structure-reorg`: planned source-layout reorganization that dissolves `OOP_ver/` into `src/`, `notebooks/`, `archive/`, and `artifacts/`.

## Blockers

- Phase-3 file moves have not yet been executed, so `OOP_ver/` still remains the physical workspace.
- Duplicate notebooks and prototype scripts have not yet been moved into their archive destinations.
- Large result files and cache artifacts still need a tracking/ignore policy.
- Some review-needed pipeline variants still need a later keep-versus-archive decision after the structural move.

## Major Risks

- Deleting or moving files too early could lose context from the September 2025 submission period.
- Multiple notebook families may encode different late-stage decisions that are not obvious from names alone.
- Root-level prototypes may still contain logic not fully reflected in `OOP_ver/`.

## Recent Decisions

- Keep the repository root as the umbrella project.
- Dissolve `OOP_ver/` during the next structure pass rather than keeping it as the long-term workspace.
- Use documentation and classification before any cleanup or movement.
- Preserve commit `c5ce4f3` as the baseline safety snapshot for later pruning.
- Use repo-local `data/` paths instead of the old external `D:/.../MSc_Project/Data/` directory.
- Keep `data/NIC2/` active and move other local datasets under `data/_archive/`.
- Use `src/pipelines/` and `src/data_collection/` as the active Python source layout.
- Keep one active notebook as `notebooks/main/eeg_analysis_main.ipynb` and archive the rest.
- Treat root prototypes, unrelated games, and MATLAB files as archive-by-default.

## Next Milestone

Execute the phase-3 structure reorganization so active code, notebooks, archives, and generated outputs have separate homes and `OOP_ver/` is removed from the active layout.
