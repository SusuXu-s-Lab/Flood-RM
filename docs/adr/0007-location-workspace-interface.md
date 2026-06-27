# Location Workspace Interface

Accepted.

Flood-RM now uses `locations/<study_location>/` as the stakeholder-facing interface for a Study Location: `config.yaml`, numbered notebooks, stage-oriented data folders, and SFINCS run artifacts live together in the Location Workspace. Reusable behavior remains in `src/`, source-collection helpers live with their owning adapters, cluster helpers live in `cluster` or `slurm`, and documentation/reference material lives in `docs`. We chose a full move with no compatibility shims because stakeholders should inspect `locations/marshfield/` as the reference workspace without needing to understand the old `design_events/` and `sfincs_runs/` workflow folders.

The one deliberate exception is the HydroMT Data Catalog, which remains a separate YAML file inside the Location Data Workspace because HydroMT requires its own schema.
