from types import SimpleNamespace

from sfincs_runs.build_base import region_notebook as region


def test_plot_domains_dispatches_without_recursing(monkeypatch, tmp_path):
    monkeypatch.setattr(region, "_plot_inland_domains", lambda runtime, domains: "inland")
    monkeypatch.setattr(region, "_plot_coastal_domains", lambda runtime, domains: "coastal")

    coastal_runtime = region.CoastalRegionSetupRuntime(
        location_root=tmp_path,
        location_name="test",
        repo_root=tmp_path,
        config={},
        paths={},
        region_setup=SimpleNamespace(coastal_region_output=tmp_path / "missing.geojson"),
        collect_static_inputs=False,
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=False,
    )

    assert region.plot_domains(object(), object()) == "inland"
    assert region.plot_domains(coastal_runtime, object()) == "coastal"
