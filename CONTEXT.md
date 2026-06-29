# Flood-RM

Flood-RM is a scientific workflow codebase for stakeholder-readable flood, Wflow,
SFINCS, and power-grid resilience studies.

## Language

**Integration Seam**:
The Stage 1 connection point where Location Configuration, clean cores, notebook skins,
and reconciliation tests meet without adding a new orchestrator.
_Avoid_: orchestration layer, pipeline framework

**Location Front Door**:
The `study_location.py` entry point that resolves a notebook or script into one
Location Definition.
_Avoid_: notebook bootstrapper, global config loader

**Runtime Adapter**:
A small domain-owned module that projects a Location Definition into the runtime paths
and settings consumed by a clean core.
_Avoid_: manager, facade, framework

**Clean Core**:
The smaller v2 implementation that owns domain behavior behind a short interface.
_Avoid_: rewrite, prototype

**Notebook Skin**:
The temporary operational package surface that preserves current notebook imports while
delegating reconciled behavior to a clean core.
_Avoid_: compatibility layer as a permanent architecture

**Location Configuration**:
The stakeholder-authored YAML declaration for one Study Location and its model choices.
_Avoid_: generated run state, output database

**Source Artifact**:
A collected or staged input with a stable path, schema, and provenance record used by
later workflow stages.
_Avoid_: temporary file, cache blob

**Reconciliation Net**:
The tests and fixture comparisons that prove a clean-core slice preserves notebook-visible
schemas, paths, manifests, and scientific behavior before delegation or deletion.
_Avoid_: smoke test when equivalence is required

## Relationships

- A **Location Front Door** produces exactly one **Location Configuration** view for a
  workflow run.
- A **Runtime Adapter** consumes a **Location Configuration** and feeds one **Clean Core**.
- A **Notebook Skin** preserves public notebook imports until a **Reconciliation Net**
  proves the clean-core path.
- A **Source Artifact** can be referenced by multiple **Runtime Adapters** but is owned by
  the domain that creates it.

## Example dialogue

> **Dev:** "Can the Wflow notebook call `wflow_v2` directly once the runtime adapter exists?"
> **Domain expert:** "Not yet. The current `wflow_runs` Notebook Skin stays public until the Reconciliation Net proves domain manifests, handoff discharge, acceptance JSON, and calibration artifacts."

## Flagged ambiguities

- "Bridge" is resolved as **Integration Seam**: the seam is `study_location.py` plus
  per-domain Runtime Adapters and temporary Notebook Skins, not a new orchestration
  package.
