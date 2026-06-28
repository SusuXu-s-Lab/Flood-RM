# HydroMT & HydroMT-SFINCS — Authoritative Syntax Reference

> **Source**: Official HydroMT docs (deltares.github.io/hydromt/stable/) and HydroMT-SFINCS docs
> (deltares.github.io/hydromt_sfincs/latest/) — 17 pages reviewed 2026-04-03.
>
> **CRITICAL**: Use ONLY this syntax when writing data catalog YAML or SFINCS model code.
> Do not extrapolate or invent fields/parameters.

---

## Table of Contents

1. [HydroMT Data Catalog YAML Syntax](#1-hydromt-data-catalog-yaml-syntax)
2. [Data Types & Drivers](#2-data-types--drivers)
3. [URI Patterns & Placeholders](#3-uri-patterns--placeholders)
4. [Driver Options](#4-driver-options)
5. [Data Adapter](#5-data-adapter)
6. [Complete YAML Examples](#6-complete-yaml-examples)
7. [Python Data Catalog API](#7-python-data-catalog-api)
8. [HydroMT-SFINCS v2 API (Component Architecture)](#8-hydromt-sfincs-v2-api-component-architecture)
9. [Build Workflow — Exact Order of Operations](#9-build-workflow--exact-order-of-operations)
10. [Step-by-Step Build API](#10-step-by-step-build-api)
11. [Forcing Components](#11-forcing-components)
12. [Geometry Components](#12-geometry-components)
13. [Config Options](#13-config-options)
14. [Running SFINCS](#14-running-sfincs)
15. [Reading & Plotting Results](#15-reading--plotting-results)
16. [Animations](#16-animations)
17. [v1→v2 Migration Reference](#17-v1v2-migration-reference)
18. [Raster & Vector Utilities](#18-raster--vector-utilities)

---

## 1. HydroMT Data Catalog YAML Syntax

### Top-level structure

```yaml
meta:
  roots:
    - /path/to/data/directory   # base dir for relative URIs

source_name:
  data_type: RasterDataset | GeoDataFrame | GeoDataset | DataFrame
  driver:
    name: <driver_name>
    options:
      <key>: <value>
  uri: <path_or_pattern>
  data_adapter:
    rename:
      old_var: new_var
    unit_mult:
      var: multiplier
    unit_add:
      var: additive
  metadata:
    crs: <EPSG_code_int_or_string>
    category: <string>
    source_version: <string>
    source_url: <url>
    source_license: <url>
```

### v2 catalog format changes (from v1)

| v1 field | v2 field |
|----------|----------|
| `path` | `uri` |
| `filesystem`, `driver_kwargs` | nested under `driver` |
| `unit_add`, `unit_mult`, `rename` (top-level) | nested under `data_adapter` |
| `meta` | `metadata` |
| `crs`, `nodata` (top-level) | nested under `metadata` |

**Upgrade tool**: `hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v`

---

## 2. Data Types & Drivers

| data_type | Python getter | Returns |
|-----------|--------------|---------|
| `RasterDataset` | `catalog.get_rasterdataset(name)` | xarray Dataset/DataArray |
| `GeoDataFrame` | `catalog.get_geodataframe(name)` | geopandas GeoDataFrame |
| `GeoDataset` | `catalog.get_geodataset(name)` | xarray Dataset/DataArray |
| `DataFrame` | `catalog.get_dataframe(name)` | pandas DataFrame |

| Driver | Use for | Formats |
|--------|---------|---------|
| `rasterio` | Single/multi-file rasters | GeoTIFF, COG, VRT |
| `raster_xarray` | NetCDF/zarr rasters | NetCDF, zarr |
| `pyogrio` | Vector files | GPKG, FlatGeoBuf, Shapefile |
| `geodataset_xarray` | Point time-series (NetCDF) | NetCDF, zarr |
| `geodataframe_table` | Tabular with x/y coords → GeoDataFrame | CSV, Excel |
| `pandas` | Pure tabular data | CSV, Excel |
| `vector` | Point locations + CSV time-series | CSV + CSV (two-file) |

---

## 3. URI Patterns & Placeholders

- `{variable}` — expands per variable (e.g., `merit_hydro/{variable}.tif`)
- `{year}`, `{month}` — temporal partitioning
- `*.tif` — wildcard glob
- Paths resolve relative to `meta.roots`

---

## 4. Driver Options

### Rasterio chunking
```yaml
driver:
  name: rasterio
  options:
    chunks:
      x: 6000
      y: 6000
```

### Pandas driver
```yaml
driver:
  name: pandas
  options:
    parse_dates: true
    index_col: 'time'
    header: 0
```

### GeoDataFrame table (CSV with coords)
```yaml
driver:
  name: geodataframe_table
  options:
    x_dim: x_centroid
    y_dim: y_centroid
```

### Raster Tindex (tiled multi-CRS rasters)
```yaml
uri_resolver:
  class: RasterTindexResolver
  options:
    tileindex: location
driver:
  options:
    mosaic: true
    mosaic_kwargs:
      method: nearest
    chunks:
      x: 3000
      y: 3000
```

---

## 5. Data Adapter

```yaml
data_adapter:
  rename:
    original_name: hydromt_name
  unit_mult:
    precip: 1000        # m → mm
    press_msl: 0.01     # Pa → hPa
    kin: 0.000277778    # J/m² → W/m²
  unit_add:
    temp: -273.15        # K → °C
```

**Order**: multiply first, then add.

### Common variable naming conventions

| Variable | HydroMT name | Units |
|----------|-------------|-------|
| Elevation | `elevtn` | m asl |
| Precipitation | `precip` | mm |
| Temperature | `temp` | °C |
| Pressure | `press_msl` | hPa |
| Radiation | `kin`, `kout` | W/m² |

---

## 6. Complete YAML Examples

### Rasterio (multi-file per variable)
```yaml
merit_hydro:
  data_type: RasterDataset
  driver:
    name: rasterio
    options:
      chunks:
        x: 6000
        y: 6000
  data_adapter:
    rename:
      hnd: height_above_nearest_drain
  uri: merit_hydro/{variable}.tif
```

### NetCDF raster with unit conversion
```yaml
era5:
  data_type: RasterDataset
  driver:
    name: raster_xarray
  data_adapter:
    unit_add:
      temp: -273.15
    unit_mult:
      precip: 1000
      press_msl: 0.01
  metadata:
    crs: 4326
  uri: era5.nc
```

### Vector (GeoPackage)
```yaml
rivers_lin:
  data_type: GeoDataFrame
  driver:
    name: pyogrio
  uri: rivers_lin2019_v1.gpkg
```

### GeoDataset (point time-series, NetCDF)
```yaml
gtsm:
  data_type: GeoDataset
  driver:
    name: geodataset_xarray
  metadata:
    crs: 4326
    category: ocean
  uri: gtsmv3_eu_era5.nc
```

### GeoDataset (two-file CSV: stations + data)
```yaml
waterlevels_txt:
  uri: stations.csv
  data_type: GeoDataset
  driver:
    name: vector
  data_adapter:
    rename:
      stations_data: waterlevel
  options:
    data_path: stations_data.csv
  metadata:
    crs: 4326
```

### DataFrame (tabular CSV)
```yaml
example_csv:
  uri: example.csv
  data_type: DataFrame
  driver:
    name: pandas
    options:
      parse_dates: true
      index_col: time
```

### Simple raster (single file)
```yaml
vito:
  uri: vito.tif
  data_type: RasterDataset
  driver:
    name: rasterio
  metadata:
    crs: 4326
```

---

## 7. Python Data Catalog API

### Initialize
```python
import hydromt
catalog = hydromt.DataCatalog(data_libs=["path/to/catalog.yml"])
catalog = hydromt.DataCatalog(data_libs=["artifact_data=v1.0.0"])
```

### Register sources programmatically (v2 syntax)
```python
sf.data_catalog.from_dict({
    "cudem_elv": {
        "uri": str(cudem_tif),
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
    },
})
```

### get_rasterdataset()
```python
catalog.get_rasterdataset(
    name,
    bbox=[minx, miny, maxx, maxy],
    geom=geometry_obj,
    variables=["var1", "var2"],
    single_var_as_array=True,
    time_range=("2010-01-01", "2010-12-31"),
)
```

### get_geodataframe()
```python
catalog.get_geodataframe(name, bbox=bbox, geom=geom, variables=["col1", "col2"])
```

### get_geodataset()
```python
catalog.get_geodataset(name, bbox=bbox, geom=geom, variables=["var"], single_var_as_array=True)
```

### get_dataframe()
```python
catalog.get_dataframe(name)
```

### export_data()
```python
catalog.export_data(
    new_root="output_dir",
    bbox=[12.0, 46.0, 13.0, 46.5],
    time_range=("2010-02-02", "2010-02-15"),
    source_names=["merit_hydro[elevtn,flwdir]", "era5[precip]"],
    metadata={"version": "1"},
)
```

Source selection with variables: `"source_name[var1,var2]"`

---

## 8. HydroMT-SFINCS v2 API (Component Architecture)

v2 uses **component-based architecture** (not inheritance). Each model aspect is a dedicated component class.

### Component mapping

| Component | Class | Data access | Key methods |
|-----------|-------|-------------|-------------|
| `config` | SfincsConfig | `sf.config.data` | `.update()`, `.write()` |
| `grid` | SfincsGrid | `sf.grid.data` | `.create_from_region()`, `.write()` |
| `elevation` | SfincsElevation | `sf.elevation.data` | `.create()` |
| `mask` | SfincsMask | `sf.mask.data` | `.create_active()`, `.create_boundary()` |
| `roughness` | SfincsRoughness | — | `.create()` |
| `subgrid` | SfincsSubgrid | — | `.create()` |
| `water_level` | SfincsWaterLevel | `sf.water_level.data` | `.create()` |
| `discharge_points` | SfincsDischargePoints | — | `.create()`, `.add_point()`, `.create_timeseries()` |
| `precipitation` | SfincsPrecipitation | — | `.create()`, `.create_uniform()` |
| `pressure` | SfincsPressure | — | `.create()` |
| `wind` | SfincsWind | — | `.create()`, `.create_uniform()` |
| `infiltration` | SfincsInfiltration | — | `.create_constant()`, `.create_cn()`, `.create_cn_with_recovery()` |
| `observation_points` | SfincsObservationPoints | `sf.observation_points.data` | `.create()` |
| `cross_sections` | SfincsCrossSections | — | `.create()` |
| `weirs` | SfincsWeirs | — | `.create()` |
| `thin_dams` | SfincsThinDams | — | `.create()` |
| `drainage_structures` | SfincsDrainageStructures | — | `.create()` |
| `storage_volume` | SfincsStorageVolume | — | `.create()` |
| `output` | SfincsOutput | `sf.output.data` | `.read()` |

### Model initialization
```python
from hydromt_sfincs import SfincsModel

sf = SfincsModel(
    root="path/to/model",    # model root directory
    mode="w+",               # r = read, r+ = append, w = write, w+ = overwrite
    write_gis=True,          # write GIS shapefiles/tifs
    data_libs=["catalog.yml"],  # data catalog paths
)
```

### Reading existing model
```python
sf = SfincsModel(root="path/to/model", mode="r")
sf.read()
```

### Changing root for event copies
```python
sf.root.set("new_location", mode="w+")
```

---

## 9. Build Workflow — Exact Order of Operations

**This order is critical. Do not rearrange.**

1. **Initialize** `SfincsModel(root, mode="w+", write_gis=True)`
2. **Register data sources** via `sf.data_catalog.from_dict(...)` or `data_libs=`
3. **Create grid** via `sf.grid.create_from_region(...)`
4. **Add elevation** via `sf.elevation.create(elevation_list=...)`
5. **Create active mask** via `sf.mask.create_active(zmin=...)`
6. **Create boundaries** via `sf.mask.create_boundary(btype=..., zmax=...)`
7. **Add roughness** via `sf.roughness.create(roughness_list=...)`
8. **Create subgrid** via `sf.subgrid.create(...)` — **replaces elevation & roughness**
9. **Add forcing** (water level, discharge, precipitation, etc.)
10. **Update config** via `sf.config.update({...})`
11. **Write** via `sf.write()`

---

## 10. Step-by-Step Build API

### grid.create_from_region()
```python
sf.grid.create_from_region(
    region={"geom": "path/to/region.geojson"},
    res=50,           # cell resolution in meters
    rotated=True,     # rotate grid to minimize inactive cells
    crs="utm",        # auto-detect UTM zone from region
)
```
Updates config: `mmax`, `nmax`, `dx`, `dy`, `x0`, `y0`, `rotation`, `epsg`.

### elevation.create()
```python
elevation_list = [
    {"elevation": "source_name", "zmin": 0.001},  # optional zmin filter
    {"elevation": "gebco"},                         # second source fills gaps
]
sf.elevation.create(elevation_list=elevation_list, buffer_cells=1)
```

### mask.create_active()
```python
sf.mask.create_active(
    zmin=-5,           # cells below this elevation → inactive
    # fill_area=10,   # fill gaps smaller than X km²
    # drop_area=5,    # drop islands smaller than X km²
)
```

### mask.create_boundary()
```python
# Water-level boundary (mask=2): offshore cells
sf.mask.create_boundary(
    btype="waterlevel",
    zmax=-5,              # only cells with elev < zmax
    reset_bounds=True,    # replace existing boundaries
)

# Outflow boundary (mask=3): river/creek outlets
sf.mask.create_boundary(
    btype="outflow",
    include_polygon=gdf,  # GeoDataFrame polygon
    reset_bounds=True,
)
```

### roughness.create()
```python
roughness_list = [
    {"lulc": "worldcover", "reclass_table": "/path/to/mapping.csv"},
]
sf.roughness.create(
    roughness_list=roughness_list,
    manning_land=0.04,    # fallback for unclassified land
    manning_sea=0.02,     # open water
    rgh_lev_land=0,       # min elevation for land roughness
)
```

### subgrid.create()
```python
sf.subgrid.create(
    elevation_list=[{"elevation": "cudem_elv"}],
    roughness_list=[{"lulc": "worldcover", "reclass_table": str(esa_mapping)}],
    nr_subgrid_pixels=6,   # subpixels per model cell (e.g., 90m/6 = 15m effective)
    write_dep_tif=True,    # export merged topo
    write_man_tif=True,    # export merged roughness
)
```

**Warning**: Subgrid tables **replace** elevation and roughness data prepared in earlier steps.

---

## 11. Forcing Components

### Water Level (bzs) — from geodataset
```python
sf.config.update({
    "tref": datetime(2010, 2, 5),
    "tstart": datetime(2010, 2, 5),
    "tstop": datetime(2010, 2, 7),
})
sf.water_level.create(geodataset="gtsmv3_eu_era5")
```

### Water Level (bzs) — manual xarray
```python
bzs = xr.DataArray(
    data_array,                           # shape: (time, n_boundary)
    dims=["time", "index"],
    coords={"time": time_index, "index": boundary_point_ids},
    attrs={"units": "m+MSL"},
)
sf.set_forcing(bzs, name="bzs")
```

### Discharge (dis) — add point + timeseries
```python
sf.discharge_points.add_point(x=321483.2, y=5047503.0, value=1000.0, name="Piave_inflow")

sf.discharge_points.create_timeseries(
    index=[0],
    shape="gaussian",
    offset=0,
    peak=5,
    tpeak=86400,         # seconds
    duration=2 * 86400,  # seconds
    timestep=3600,       # seconds
)
```

### Discharge (dis) — manual xarray
```python
dis = xr.DataArray(
    q_array,                              # shape: (time, n_sources)
    dims=["time", "index"],
    coords={"time": time_index, "index": source_point_ids},
    attrs={"units": "m3/s"},
)
sf.set_forcing(dis, name="dis")
```

### Precipitation — from gridded data
```python
sf.precipitation.create(precip="era5_hourly", aggregate=False, buffer=30e3)
```

### Precipitation (netampr) — manual xarray
```python
netampr = xr.DataArray(
    precip_array,                         # shape: (time, n_cells)
    dims=["time", "index"],
    coords={"time": time_index},
    attrs={"units": "mm/hr"},
)
sf.set_forcing(netampr, name="netampr")
```

### Infiltration — curve number method
```python
sf.infiltration.create_cn("gcn250", antecedent_moisture="avg")
```

### Wind & Pressure (referenced, not fully demonstrated)
```python
sf.wind.create(...)             # gridded wind
sf.wind.create_uniform(...)     # uniform wind
sf.pressure.create(...)         # gridded pressure
```

---

## 12. Geometry Components

### Observation Points
```python
sf.observation_points.create(
    locations="data/obs_points.geojson",
    merge=False,
)
```

### Cross Sections
```python
sf.cross_sections.create(locations="data/cross_sections.geojson", merge=False)
```

### Weirs
```python
sf.weirs.create(
    locations="data/weirfile.geojson",  # LineString geometries
    dz=7.7,                              # elevation offset above model DEM
    merge=False,
)
```

### Thin Dams
```python
sf.thin_dams.create(locations="data/thindams.geojson", merge=False)
```

---

## 13. Config Options

Access: `sf.config.data.model_dump()`

| Category | Keys | Typical values |
|----------|------|---------------|
| **Grid** | `mmax`, `nmax`, `dx`, `dy`, `x0`, `y0`, `rotation`, `epsg` | set by `grid.create_from_region()` |
| **Time** | `tref`, `tstart`, `tstop`, `tspinup` | `"YYYYMMDD HHMMSS"` or datetime |
| **Output** | `dtmapout`, `dthisout`, `outputformat` | 3600.0, 600.0, "net" |
| **Physics** | `alpha`, `theta`, `manning_land`, `manning_sea` | 0.5, 1.0, 0.04, 0.02 |
| **Initial** | `zsini`, `qinf` | 0.0, 0.0 |
| **Threshold** | `huthresh`, `hmin_cfl` | 0.01 |

```python
sf.config.update({
    "tref":   "20000101 000000",
    "tstart": "20000101 000000",
    "tstop":  "20000106 000000",
    "epsg":   26919,
})
```

---

## 14. Running SFINCS

### Windows batch file
```
call "c:\path\to\sfincs.exe" > sfincs_log.txt
```

### Python execution
```python
import os
cur_dir = os.getcwd()
os.chdir(run_path)
os.system("run.bat")
os.chdir(cur_dir)
```

### Docker (Linux/HPC)
```bash
docker run -v /path/to/model:/data sfincs_image
```

### Output files
- `sfincs_map.nc` — spatial map results (h, zs, hmax, zsmax, zb)
- `sfincs_his.nc` — time series at observation points
- `sfincs_log.txt` — execution log

---

## 15. Reading & Plotting Results

### Read results
```python
mod = SfincsModel(root="path/to/model", mode="r")
mod.output.read()
# mod.output.data contains: 'hmax', 'zsmax', 'zb', 'zs', 'h'
```

### Maximum water depth (regular model, no subgrid)
```python
da_hmax = mod.output.data["hmax"].max(dim="timemax")
```

### Maximum water depth (subgrid model — requires downscaling!)
```python
from hydromt_sfincs import utils

da_dep = mod.data_catalog.get_rasterdataset(depfile)
da_zsmax = mod.output.data["zsmax"].max(dim="timemax")

da_hmax = utils.downscale_floodmap(
    zsmax=da_zsmax,
    dep=da_dep,
    hmin=0.05,
    gdf_mask=gdf_osm,                          # optional land mask
    floodmap_fn="path/to/floodmap.tif",        # optional GeoTIFF output
)
```

**CRITICAL**: Regular vs subgrid flood map methods are **NOT interchangeable**.

### Flood masking with GSWO
```python
gswo = mod.data_catalog.get_rasterdataset("gswo", geom=mod.region, buffer=10)
gswo_mask = gswo.raster.reproject_like(mod.grid.data, method="max") <= 5
da_hmax = da_hmax.where(gswo_mask).where(da_hmax > 0.05)
```

### Plot basemap with results
```python
fig, ax = mod.plot_basemap(
    fn_out=None,
    figsize=(8, 6),
    variable=da_hmax,
    plot_bounds=False,
    plot_geoms=False,
    bmap="sat",
    zoomlevel=12,
    vmin=0,
    vmax=2.0,
    cmap=plt.cm.viridis,
    cbar_kwargs={"shrink": 0.6, "anchor": (0, 0)},
)
```

### Export flood map GeoTIFF
```python
hmax.attrs.update(long_name="maximum water depth", unit="m")
utils.write_raster(hmax, "path/to/hmax.tif", compress="LZW")
```

---

## 16. Animations

### Setup
```python
import matplotlib.animation as animation
from IPython.display import HTML

mod = SfincsModel(root="path", mode="r")
mod.output.read()

hmin = 0.05
da_h = mod.output.data["h"].copy()
da_h = da_h.where(da_h > hmin).drop("spatial_ref")
da_h.attrs.update(long_name="flood depth", unit="m")
```

### FuncAnimation
```python
def update_plot(i, da_h, cax_h):
    da_hi = da_h.isel(time=i)
    t = da_hi.time.dt.strftime("%d-%B-%Y %H:%M:%S").item()
    ax.set_title(f"SFINCS water depth {t}")
    cax_h.set_array(da_hi.values.ravel())

ani = animation.FuncAnimation(
    fig,
    update_plot,
    frames=np.arange(0, da_h.time.size, step),
    interval=250,
    fargs=(da_h, cax_h,)
)

# Save
ani.save("output.mp4", fps=4, dpi=200)
# Or display in notebook
HTML(ani.to_html5_video())
```

**Note**: For subgrid models, downscale each timestep before animating.

---

## 17. v1→v2 Migration Reference

### Method name changes

| v1.x | v2 |
|------|-----|
| `setup_grid()` | `grid.create()` |
| `setup_grid_from_region()` | `grid.create_from_region()` |
| `setup_dep()` | `elevation.create()` |
| `setup_mask_active()` | `mask.create_active()` |
| `setup_mask_bounds()` | `mask.create_boundary()` |
| `setup_subgrid()` | `subgrid.create()` |
| `setup_manning_roughness()` | `roughness.create()` |
| `setup_config()` | `config.update()` |
| `setup_waterlevel_forcing()` | `water_level.create()` |
| `setup_waterlevel_bnd_from_mask()` | `water_level.create_boundary_points_from_mask()` |
| `setup_discharge_forcing()` | `discharge_points.create()` |
| `setup_precip_forcing()` | `precipitation.create_uniform()` |
| `setup_precip_forcing_from_grid()` | `precipitation.create()` |
| `setup_pressure_forcing_from_grid()` | `pressure.create()` |
| `setup_wind_forcing()` | `wind.create_uniform()` |
| `setup_wind_forcing_from_grid()` | `wind.create()` |
| `setup_constant_infiltration()` | `infiltration.create_constant()` |
| `setup_cn_infiltration()` | `infiltration.create_cn()` |
| `setup_cn_infiltration_with_ks()` | `infiltration.create_cn_with_recovery()` |
| `setup_observation_points()` | `observation_points.create()` |
| `setup_observation_lines()` | `cross_sections.create()` |
| `setup_structures()` | `weirs.create()` or `thin_dams.create()` |
| `setup_drainage_structures()` | `drainage_structures.create()` |
| `setup_storage_volume()` | `storage_volume.create()` |
| `model.write_<component>()` | `model.<component>.write()` |
| `model.read_<component>()` | `model.<component>.read()` |
| `model.set_<component>()` | `model.<component>.set()` |
| `model.setup_<component>()` | `model.<component>.create()` |

### Argument name changes

| v1.x argument | v2 argument |
|--------------|------------|
| `datasets_dep` | `elevation_list` |
| `datasets_rgh` | `roughness_list` |
| `nr_levels` | `nlevels` |
| `include_mask` | `include_polygon` |
| `exclude_mask` | `exclude_polygon` |
| `structures` | `locations` |

### Data access pattern
```python
# v1: model.<component>
# v2: model.<component>.data
grid_data = model.grid.data          # xarray.Dataset
obs_pts = model.observation_points.data  # GeoDataFrame
wl_data = model.water_level.data     # xarray.Dataset
```

### YAML config format (v2)
```yaml
modeltype: sfincs
global:
  data_libs:
    - path/to/catalog.yml
steps:
  - config.update:
      tref: "20100201 000000"
      tstart: "20100201 000000"
      tstop: "20100202 000000"
  - precipitation.create:
      precip: "era5_hourly"
```

---

## 18. Raster & Vector Utilities

### Raster
- `.raster.mask_nodata()` — replace nodata with NaN
- `.raster.to_mapstack()` — export to folder of GeoTIFFs
- `.raster.reproject_like(target, method="max")` — reproject to match target grid
- `.compute()` — load lazy Dask arrays into memory

### Vector / GeoDataset
- `.vector.crs` — access CRS
- `.vector.geometry` — get geometry objects
- `.vector.bounds` — bounding box array [minx, miny, maxx, maxy]
- `.vector.to_crs(epsg)` — reproject
- `.vector.to_gdf(reducer=func)` — convert to GeoDataFrame
- `.vector.to_netcdf(path, ogr_compliant=True)` — write NetCDF
- `.vector.ogr_compliant(reducer=mean)` — make QGIS-readable
- `.vector.to_wkt()` — convert geometry to WKT strings
- `.vector.to_geom()` — convert to Shapely geometry objects

### GeoDataArray from GeoDataFrame
```python
from hydromt.gis import GeoDataArray
ds = GeoDataArray.from_gdf(gdf, np.arange(gdf.index.size))
```
