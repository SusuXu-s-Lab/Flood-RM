import pandas as pd

from design_events.collect_sources import cora
from design_events.collect_sources.cora import collect_cora


def test_cora_daily_reader_reports_day_level_progress(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    progress_calls = []
    postfixes = []

    def fake_read_day(paths, cora_config, date, node):
        return pd.DataFrame({"time": [date], "value": [float(date.day)]})

    class FakeProgress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix_str(self, value, refresh=False):
            postfixes.append(value)

    def fake_progress(iterable, **kwargs):
        progress_calls.append(kwargs)
        return FakeProgress(iterable, **kwargs)

    monkeypatch.setattr(cora, "read_day", fake_read_day)
    monkeypatch.setattr(cora, "iter_progress", fake_progress)

    frames = cora.read_days_with_progress({}, {}, dates, node=1045612, workers=2)

    assert len(frames) == 3
    assert progress_calls == [
        {
            "total": 3,
            "desc": "CORA daily files",
            "unit": "day",
            "dynamic_ncols": True,
        }
    ]
    assert sorted(postfixes) == ["2020-01-01", "2020-01-02", "2020-01-03"]


def test_cora_raw_daily_cache_is_opt_in(tmp_path):
    paths = {"cache_root": tmp_path / "cache"}
    date = pd.Timestamp("1979-02-01")

    assert cora.daily_cache_path(paths, {}, date) is None
    assert cora.daily_cache_path(paths, {"raw_cache_enabled": False}, date) is None
    assert cora.daily_cache_path(paths, {"raw_cache_enabled": True}, date) == (
        tmp_path / "cache/cora_daily_nc/1979/cora_19790201.nc"
    )


def test_collect_cora_reuses_configured_full_csv_and_writes_manifest(tmp_path):
    output = tmp_path / "cora_mfield_boundary_hourly_msl.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("1979-01-01", "1979-01-03", freq="h"),
            "value": [0.1] * 49,
        }
    ).to_csv(output, index=False)
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "source_artifacts_root": tmp_path / "source_artifacts",
        "waterlevel_csv": output,
    }
    settings = {
        "paths": paths,
        "start": pd.Timestamp("1979-01-01"),
        "end": pd.Timestamp("1979-01-02"),
        "cora": {"reuse_existing": True},
    }

    frame = collect_cora(settings, skip_existing=False, smoke=False)

    assert len(frame) == 49
    manifest = paths["source_artifacts_root"] / "cora_boundary_water_level.json"
    assert manifest.exists()
    assert "cora_mfield_boundary_hourly_msl.csv" in manifest.read_text(encoding="utf-8")
