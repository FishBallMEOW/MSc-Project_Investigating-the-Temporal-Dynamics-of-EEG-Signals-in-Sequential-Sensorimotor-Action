# AGENTS.md — Multi-Agent Operating Manual

## 0. Purpose
This file defines a **standard operating system for human contributors and AI agents** working inside a shared repository. It is designed to be reusable across projects, while also giving project-specific guidance through clearly named directories, documents, and handoff rules.

The goal is simple:
- avoid duplicated work
- avoid conflicting edits
- preserve context across sessions
- improve quality and traceability
- let multiple agents work as if they are coordinated members of one engineering organization

This file should be treated as operational policy, not informal advice.

---

## 1. Working Principles
1. **Read before writing.** Never start implementing before understanding the current state.
2. **Plan before editing.** Every non-trivial task needs a local plan and scope.
3. **Leave the repo more legible than you found it.** Update docs, decisions, and handoff notes.
4. **Prefer additive clarity over hidden cleverness.** Code and docs should be easy for the next agent to continue.
5. **Record assumptions explicitly.** Silent assumptions cause agent drift.
6. **Separate facts, decisions, and hypotheses.** Do not present guesses as project truth.
7. **Minimize conflict surfaces.** Claim tasks, announce touched areas, and keep changes scoped.
8. **Treat reproducibility as a product feature.** Code, experiments, and docs must be rerunnable and auditable.
9. **Do not invent repo structure ad hoc.** Follow the documented architecture.
10. **A handoff is part of the work, not optional overhead.**

---

## 2. Default Agent Roles
One agent may play several roles, but the roles should remain conceptually separate.

### 2.1 Planner
- clarifies scope
- reads context
- writes plan
- identifies dependencies and risks

### 2.2 Implementer
- writes code or docs according to plan
- updates tests and examples
- keeps changes localized

### 2.3 Reviewer
- checks design coherence, correctness, risk, and maintainability
- flags policy or architecture violations

### 2.4 Researcher
- gathers external or internal references
- summarizes findings into reusable project docs

### 2.5 QA / Validator
- runs tests, smoke checks, schema validation, and acceptance checks
- logs gaps and failed assumptions

### 2.6 Release / Handoff Coordinator
- prepares final summary
- updates status docs
- records next steps and owner suggestions

---

## 3. Mandatory Start-of-Task SOP
Before making any change, the agent must do the following in order.

### Step 1 — Read project orientation
Read, if present:
- `README.md`
- `docs/index.md`
- `docs/project/vision.md`
- `docs/project/prd.md` or equivalent product doc
- `docs/architecture/README.md`
- `docs/decisions/`
- `docs/status/current.md`
- `docs/handoffs/latest.md`

### Step 2 — Read active execution context
Read, if present:
- `docs/plans/`
- `docs/tasks/active/`
- `docs/worklog/`
- issue tracker references
- recent ADRs or design notes related to the target area

### Step 3 — Check ownership and conflict risk
Check whether someone else is actively working on:
- the same folder
- the same module
- the same decision space
- the same documentation set

If conflict risk exists, narrow scope or update the task note before proceeding.

### Step 4 — Write or update a task plan
For every non-trivial task, create or update a plan document in:
- `docs/plans/YYYY-MM-DD-task-slug.md`

Each plan should include:
- task summary
- objective
- scope in / out
- files likely to change
- assumptions
- dependencies
- implementation outline
- validation approach
- handoff expectations

### Step 5 — Announce the work area
Update or create a task note in:
- `docs/tasks/active/<task-slug>.md`

Include:
- owner / agent name
- date
- modules touched
- status
- blockers

No major work should begin without a visible task note unless the repo is in solo mode.

---

## 4. Repository Navigation Contract
Every project using this guide should aim to keep these documentation locations stable.

### Core docs
- `docs/index.md` — documentation map
- `docs/project/` — vision, PRD, glossary, scope
- `docs/architecture/` — repo architecture, system design, module boundaries
- `docs/decisions/` — ADRs and decision logs
- `docs/plans/` — forward-looking implementation plans
- `docs/tasks/active/` — active tasks and ownership
- `docs/tasks/archive/` — completed tasks
- `docs/status/` — current project status and milestones
- `docs/handoffs/` — latest summaries for the next agent
- `docs/runbooks/` — operational procedures
- `docs/experiments/` — experiment protocols and results summaries
- `docs/specs/` — schema and interface docs

### Code areas
- `apps/` — entrypoint applications
- `services/` — deployable backend services
- `packages/` — shared libraries / SDKs
- `infra/` — deployment and infra-as-code
- `scripts/` — utility scripts
- `tests/` — automated tests
- `examples/` — usage examples and reference configs

If the repo differs, create `docs/index.md` immediately and document the local equivalents.

---

## 5. Planning Standard
A good plan must be:
- scoped
- testable
- dependency-aware
- reversible where possible
- explicit about assumptions

### Plan template
```md
# Task Plan: <slug>
- Date:
- Owner:
- Objective:
- Why this matters:
- In scope:
- Out of scope:
- Inputs / dependencies:
- Files / modules likely touched:
- Risks:
- Steps:
- Validation:
- Expected handoff artifacts:
```

### Planning rules
- Break large work into phases.
- Mark unknowns early.
- Prefer one decisive path over many speculative branches.
- If architecture is unclear, write an ADR before deep implementation.

---

## 6. Documentation Standard
Agents must continuously maintain documents that other agents can trust.

### 6.1 Facts vs decisions vs proposals
Use different documents for different kinds of truth:
- **facts:** current state, implementation behavior, measured outputs
- **decisions:** what was chosen and why
- **proposals:** options under consideration

### 6.2 Required documentation updates after non-trivial work
Update at least one of:
- `docs/handoffs/latest.md`
- `docs/status/current.md`
- related ADR
- module README
- task note

### 6.3 Module readmes
Every significant package or service should contain a local README with:
- purpose
- key entrypoints
- dependencies
- config expectations
- test instructions
- extension points

---

## 7. Decision Logging Standard
Use `docs/decisions/ADR-XXXX-title.md` for decisions that affect architecture, interfaces, storage, security, workflows, or public behavior.

### ADR template
```md
# ADR-XXXX: <title>
- Status: proposed | accepted | superseded | deprecated
- Date:
- Context:
- Decision:
- Alternatives considered:
- Consequences:
- Follow-up actions:
```

Create an ADR when:
- changing repo structure
- defining an interface used by multiple modules
- choosing storage or orchestration technology
- changing schemas
- introducing security or policy constraints
- changing benchmark or evaluation semantics

---

## 8. Implementation Rules

### 8.1 Make small coherent changes
- Avoid mixing unrelated refactors with feature work.
- Avoid large silent renames unless documented.
- Prefer layered commits and traceable diffs.

### 8.2 Respect boundaries
Do not reach across module boundaries casually. If a change requires boundary crossing, check architecture docs and update them if needed.

### 8.3 Keep interfaces explicit
Public interfaces must be documented and type-checked where possible.

### 8.4 Preserve backwards compatibility intentionally
If breaking compatibility:
- document it
- update migration notes
- update examples and tests

### 8.5 Prefer config over hardcoding
Especially for:
- paths
- providers
- model names
- resource settings
- environment endpoints

---

## 9. Testing and Validation Rules
No change is complete without validation proportional to its risk.

### Minimum validation checklist
- schema validation passes
- lint / formatting passes if configured
- relevant unit tests pass
- new behavior has tests or example coverage
- docs updated
- task note updated

### Additional validation for data / model / evaluation work
- run smoke tests on a tiny fixture
- verify version / provenance capture
- verify metrics do not silently change
- compare against at least one known baseline or golden output when feasible

### Additional validation for infra work
- startup test
- health check
- local dev path verified
- failure path or rollback documented

---

## 10. Experiment and Benchmark Discipline
For projects involving ML, evaluation, or training infrastructure, follow these rules.

### 10.1 Every run must have identity
Store or log:
- run id
- dataset/task version
- model version
- evaluator version
- config hash
- seed
- adapter set
- timestamp

### 10.2 Never overwrite results silently
Use append-only or versioned result outputs.

### 10.3 Separate official from exploratory work
- official benchmark runs
- exploratory runs
- sandbox/manual tests

must be distinguishable in naming, storage, and reporting.

### 10.4 Record failed runs too
Failures are project data. Keep them unless there is a privacy or cost reason not to.

---

## 11. Task Ownership and Conflict Avoidance
When multiple agents work in the same repo, use these controls.

### 11.1 Active task files
Each active task should have a note in `docs/tasks/active/` with:
- owner
- scope
- touched areas
- start date
- status

### 11.2 Soft locking
If you plan to touch a major module, add a short note such as:
- `Working in packages/compat and docs/specs/model-spec.md`

This is a coordination signal, not a hard lock.

### 11.3 Handoff before abandonment
If stopping mid-task, update:
- what is done
- what is half-done
- what is risky
- exact next recommended step

### 11.4 Avoid duplicated planning
Before creating a new plan, search whether the task already has one.

---

## 12. Handoff Standard
A handoff must let a new agent continue without rereading the entire repo.

### Handoff location
- `docs/handoffs/YYYY-MM-DD-HHMM-<slug>.md`
- optionally update `docs/handoffs/latest.md` to point to the newest one

### Handoff template
```md
# Handoff: <slug>
- Date:
- Author:
- Summary:
- Completed:
- Files changed:
- Decisions made:
- Validation performed:
- Open issues / risks:
- Exact next steps:
- Suggested owner type:
```

### Handoff quality bar
The next agent should be able to answer all three:
1. What happened?
2. Why was it done this way?
3. What should happen next?

---

## 13. Project Status Tracking
Maintain a concise current status file:
- `docs/status/current.md`

It should contain:
- project phase
- top priorities
- active tasks
- blocked tasks
- major risks
- recent decisions
- next milestone

This is the default landing page for agents after `docs/index.md`.

---

## 14. Runbooks and Operational Knowledge
Repeated operational work should become a runbook.

Examples:
- how to start local stack
- how to register a dataset
- how to add a new model adapter
- how to run smoke tests
- how to release a schema change
- how to import a public benchmark

Store them under `docs/runbooks/`.

---

## 15. Prompting Rules for Coding Agents
When an agent is prompted to work, it should internally or explicitly do the following:
1. Restate the target outcome.
2. Check existing docs and plans.
3. Identify affected modules.
4. Note assumptions and unresolved unknowns.
5. Prefer concrete deliverables over vague brainstorming.
6. Update documentation as part of the output, not as an afterthought.
7. End with a precise handoff or completion note.

### Prompt style guidance
Good project prompts usually specify:
- objective
- constraints
- acceptance criteria
- allowed scope
- file targets
- required validation
- documentation updates required

---

## 16. Quality Bar for Any Deliverable
A deliverable is not complete unless it is:
- correct enough for its current stage
- understandable by another engineer or agent
- testable or at least checkable
- documented in the relevant project locations
- consistent with architecture and prior decisions

---

## 17. Anti-Patterns to Avoid
- editing code before reading context
- writing large undocumented changes
- creating duplicate docs with overlapping truth
- silently changing schemas or interfaces
- mixing exploratory hacks into production paths
- hiding uncertainty
- deleting logs, notes, or failed experiments without policy reason
- assuming the next agent will “figure it out”

---

## 18. Suggested Minimal Project-Specific Additions
Each project should add a small section near the top of this file covering:
- project purpose
- primary stack
- local dev commands
- architecture doc locations
- status doc location
- benchmark / data policy notes
- security constraints

This keeps the file reusable while still actionable in a concrete repo.

---

## 19. Example Local Project Addendum
```md
## Project Addendum
- Product: Multimodal Evaluation Lab
- Primary docs: docs/project/prd.md, docs/architecture/repo-architecture.md
- Current status: docs/status/current.md
- Local stack: uv + python, pnpm + web, docker compose for services
- Critical modules: packages/specs, packages/compat, services/run-service, apps/web
- Sensitive areas: dataset connectors, sandbox execution, evaluator versioning
```

---

## 20. Final Rule
When in doubt, optimize for **clarity, continuity, and verifiable progress**.

A strong agent does not just finish its own task. It makes the next correct task easier.
