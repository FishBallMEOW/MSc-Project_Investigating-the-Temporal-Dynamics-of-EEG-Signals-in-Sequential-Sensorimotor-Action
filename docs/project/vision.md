# Project Vision

## Purpose

This repository preserves the code, notebooks, data references, and generated artifacts for an MSc project investigating the temporal dynamics of EEG signals in sequential sensorimotor action.

## Current Framing

- The repository is both a working project and an archive of late-stage submission-period material.
- Active EEG analysis code now lives under `src/pipelines/`, with `eeg_processing_organized.py` as the current canonical entrypoint.
- Active collection and replay utilities now live under `src/data_collection/`.
- The repository root remains an umbrella space for data, docs, active source, generated artifacts, and preserved historical material.

## Near-Term Goal

The immediate goal is to make the repository legible without changing analysis content:

- separate active source from historical material,
- keep only one obvious main notebook,
- isolate generated outputs from source directories,
- preserve provenance while making the active workflow easy to find.

## Non-Goals For This Pass

- No deletion of historical material
- No deep refactor of the analysis internals
- No attempt to merge all remaining pipeline variants into one codebase yet
