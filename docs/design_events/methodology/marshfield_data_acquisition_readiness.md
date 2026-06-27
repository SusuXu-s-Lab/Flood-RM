# Marshfield Data Acquisition Readiness

Marshfield is close to full data acquisition. The remaining gates are execution
readiness, not major architecture work.

## Remaining Steps

1. Run the direct AORC SST collector on the configured 100 km SST region until
   it writes non-empty `storm-stats.csv` and `ranked-storms.csv`.
3. Run an ERA5 wave smoke pull for the configured offshore bounding box and
   confirm the NetCDF opens with `swh`, `pp1d`, `mwd`, and `wdw`.
4. Build or reuse `rainfall_members.csv` from the completed direct AORC SST
   output and run the event-catalog build/audit with rainfall pairing enabled.
5. Spatial infiltration is now a configured SFINCS base-model capability for
   the wave-coupled truth set: SSURGO HSG/Ksat rasters feed CN-with-recovery
   maps, and paired NWM `SOILSAT_TOP` members restage event-specific
   `sfincs.seff` when precipitation/hydrology forcing is staged.
6. Run one final acquisition dry run that confirms output roots, disk
   expectations, resume behavior, source manifests, and readiness gates for
   CORA, NWM soil moisture, direct AORC SST rainfall, ERA5 SnapWave Boundary
   Forcing, and Marshfield's streamflow-unavailable exception.

Smoke-readiness sequence:

Use the Marshfield flood notebooks as the canonical runbook. Start with
`locations/marshfield/02_flood/01_region_setup.ipynb` for static inputs, then
run `02_collect_sources.ipynb` for AORC SST, ERA5 waves, NWM, CORA, and source
readiness gates. The notebook calls the shared collection helpers directly
(`build_source_collection_plan`, `run_collect`, and
`write_data_acquisition_readiness`) so there is no separate `design_events`
command-line surface to keep in sync.

For SSURGO-only repair work, call the notebook helper cells in
`01_region_setup.ipynb`; the raw fetch helper remains importable from
`design_events.collect_sources.ssurgo` for scripted experiments, but the
Location Workspace notebooks own the production workflow.

After all gates pass, full Marshfield acquisition can start. The full direct
AORC SST collection should be run as a managed long job after the smoke
collection finishes cleanly. Full `collect_sources` now includes ERA5 wave
collection when `collection.era5_waves` is configured. The Marshfield default
uses Earth Data Hub and reads its URL/key from `code/api-key.txt`. CDS remains
a fallback by setting `collection.era5_waves.provider: cds` and configuring
`cdsapi` plus `~/.cdsapirc`.
