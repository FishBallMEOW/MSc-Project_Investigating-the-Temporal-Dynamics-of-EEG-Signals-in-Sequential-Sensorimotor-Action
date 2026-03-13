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

## Blockers

- Canonical analysis entrypoints are not yet formally selected.
- Duplicate notebooks and prototype scripts have not yet been compared in detail.
- Large result files and cache artifacts still need a tracking/ignore policy.
- The active code layout still needs a later reorganization pass.

## Major Risks

- Deleting or moving files too early could lose context from the September 2025 submission period.
- Multiple notebook families may encode different late-stage decisions that are not obvious from names alone.
- Root-level prototypes may still contain logic not fully reflected in `OOP_ver/`.

## Recent Decisions

- Keep the repository root as the umbrella project.
- Treat `OOP_ver/` as the primary workspace for future orientation.
- Use documentation and classification before any cleanup or movement.
- Preserve commit `c5ce4f3` as the baseline safety snapshot for later pruning.
- Use repo-local `data/` paths instead of the old external `D:/.../MSc_Project/Data/` directory.
- Keep `data/NIC2/` active and move other local datasets under `data/_archive/`.

## Next Milestone

Implement the remaining root and `OOP_ver/` codebase reorganization so active code, notebooks, archives, and generated outputs have separate homes.
