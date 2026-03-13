# Current Status

## Phase

Post-submission archival cleanup and organization.

## Top Priorities

- Keep the new `src/`, `notebooks/`, `archive/`, and `artifacts/` layout stable.
- Treat `data/NIC2/` as the current active dataset area and `data/_archive/` as the local archive boundary.
- Review the remaining support pipeline variants under `src/pipelines/`.
- Keep generated outputs and cache artifacts out of version control.
- Prepare a later pruning pass for duplicate notebooks, backup files, and redundant analysis variants.

## Active Task

- `light-reorg-baseline`: documentation-first repo mapping and cleanup inventory.
- `data-and-code-reorg-plan`: implementation-ready plan for data archiving and source layout cleanup.
- `data-phase-2-reorg`: completed local data reorganization so only `data/NIC2/` remains active.
- `phase-3-structure-reorg`: completed source-layout reorganization that dissolved `OOP_ver/` into `src/`, `notebooks/`, `archive/`, and `artifacts/`.

## Blockers

- Remaining support pipeline scripts still need a later keep-versus-archive decision.
- Large local result files still need a final retention policy beyond being placed under `artifacts/`.
- Some historical docs still describe earlier phases and should remain understood as historical state, not current layout.

## Major Risks

- Deleting or moving files too early could lose context from the September 2025 submission period.
- Multiple notebook families may encode different late-stage decisions that are not obvious from names alone.
- Root-level prototypes may still contain logic not fully reflected in `OOP_ver/`.

## Recent Decisions

- Keep the repository root as the umbrella project.
- Dissolve `OOP_ver/` rather than keeping it as the long-term workspace.
- Use documentation and classification before any cleanup or movement.
- Preserve commit `c5ce4f3` as the baseline safety snapshot for later pruning.
- Use repo-local `data/` paths instead of the old external `D:/.../MSc_Project/Data/` directory.
- Keep `data/NIC2/` active and move other local datasets under `data/_archive/`.
- Use `src/pipelines/` and `src/data_collection/` as the active Python source layout.
- Keep one active notebook as `notebooks/main/eeg_analysis_main.ipynb` and archive the rest.
- Treat root prototypes, unrelated games, and MATLAB files as archive-by-default.
- Place generated outputs under `artifacts/` rather than mixing them into active source folders.

## Next Milestone

Finish the post-move cleanup pass: validate the updated paths, decide which support pipeline variants remain active, and finalize the ignore policy for generated artifacts and caches.
