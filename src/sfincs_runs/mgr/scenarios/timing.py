from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class MissingTimingDescriptorsError(ValueError):
    pass


@dataclass(frozen=True)
class DriverWindow:
    driver: str
    start_offset_hours: float
    peak_offset_hours: float | None
    end_offset_hours: float

    def start_time(self, event_reference_time) -> pd.Timestamp:
        return pd.Timestamp(event_reference_time) + pd.Timedelta(hours=float(self.start_offset_hours))

    def peak_time(self, event_reference_time) -> pd.Timestamp | None:
        if self.peak_offset_hours is None:
            return None
        return pd.Timestamp(event_reference_time) + pd.Timedelta(hours=float(self.peak_offset_hours))

    def end_time(self, event_reference_time) -> pd.Timestamp:
        return pd.Timestamp(event_reference_time) + pd.Timedelta(hours=float(self.end_offset_hours))


@dataclass(frozen=True)
class ForcingSupportWindow:
    event_reference_time: pd.Timestamp
    run_start: pd.Timestamp
    run_stop: pd.Timestamp
    driver_windows: tuple[DriverWindow, ...]
    timing_policy: str

    @property
    def duration_hours(self) -> float:
        return float((self.run_stop - self.run_start) / pd.Timedelta(hours=1))


def plan_forcing_support_window(
    *,
    event_reference_time,
    driver_windows,
    spinup_hours=0,
    drain_down_hours=0,
    min_run_hours=None,
    max_run_hours=None,
    timing_policy="descriptors",
) -> ForcingSupportWindow:
    windows = tuple(driver_windows)
    if not windows:
        raise ValueError("at least one driver window is required")

    reference = pd.Timestamp(event_reference_time)
    raw_start = min(window.start_time(reference) for window in windows)
    raw_stop = max(window.end_time(reference) for window in windows)
    run_start = raw_start - pd.Timedelta(hours=float(spinup_hours))
    run_stop = raw_stop + pd.Timedelta(hours=float(drain_down_hours))
    run_start, run_stop = _apply_duration_bounds(
        run_start,
        run_stop,
        reference=reference,
        min_run_hours=min_run_hours,
        max_run_hours=max_run_hours,
    )
    return ForcingSupportWindow(
        event_reference_time=reference,
        run_start=run_start,
        run_stop=run_stop,
        driver_windows=windows,
        timing_policy=str(timing_policy),
    )


def plan_event_forcing_support_window(
    catalog,
    *,
    model_start_time=None,
    coastal_sample_count=None,
    allow_legacy_inference=False,
    spinup_hours=0,
    drain_down_hours=0,
    min_run_hours=None,
    max_run_hours=None,
) -> ForcingSupportWindow:
    catalog = dict(catalog or {})
    catalog = _normalize_event_catalog_timing(catalog)
    descriptor_windows = _descriptor_driver_windows(catalog)
    if descriptor_windows:
        reference = pd.Timestamp(catalog["event_reference_time"])
        return plan_forcing_support_window(
            event_reference_time=reference,
            driver_windows=descriptor_windows,
            spinup_hours=spinup_hours,
            drain_down_hours=drain_down_hours,
            min_run_hours=min_run_hours,
            max_run_hours=max_run_hours,
            timing_policy="descriptors",
        )

    if not allow_legacy_inference:
        raise MissingTimingDescriptorsError(
            "event timing descriptors are required; pass allow_legacy_inference=True for legacy rows"
        )
    if model_start_time is None or coastal_sample_count is None:
        raise MissingTimingDescriptorsError(
            "legacy timing inference requires model_start_time and coastal_sample_count"
        )

    hours = max(0, int(coastal_sample_count) - 1)
    return plan_forcing_support_window(
        event_reference_time=pd.Timestamp(model_start_time),
        driver_windows=[
            DriverWindow(
                "coastal",
                start_offset_hours=0,
                peak_offset_hours=None,
                end_offset_hours=hours,
            )
        ],
        spinup_hours=spinup_hours,
        drain_down_hours=drain_down_hours,
        min_run_hours=min_run_hours,
        max_run_hours=max_run_hours,
        timing_policy="legacy_inferred",
    )


def _descriptor_driver_windows(catalog) -> tuple[DriverWindow, ...]:
    if "event_reference_time" not in catalog or _is_missing(catalog.get("event_reference_time")):
        return ()
    windows = []
    for driver in ("coastal", "wave", "rainfall", "streamflow"):
        start_key = f"{driver}_start_offset_hours"
        end_key = f"{driver}_end_offset_hours"
        if start_key not in catalog and end_key not in catalog:
            continue
        if _is_missing(catalog.get(start_key)) or _is_missing(catalog.get(end_key)):
            continue
        peak = catalog.get(f"{driver}_peak_offset_hours")
        windows.append(
            DriverWindow(
                driver=driver,
                start_offset_hours=float(catalog[start_key]),
                peak_offset_hours=None if _is_missing(peak) else float(peak),
                end_offset_hours=float(catalog[end_key]),
            )
        )
    return tuple(windows)


def _normalize_event_catalog_timing(catalog) -> dict:
    out = dict(catalog or {})

    if _is_missing(out.get("event_reference_time")):
        reference = out.get("coastal_template_peak_time") or out.get("coastal_analog_peak_time")
        if not _is_missing(reference):
            out["event_reference_time"] = reference

    if not _is_missing(out.get("event_reference_time")):
        if _is_missing(out.get("coastal_start_offset_hours")) and not _is_missing(out.get("coastal_valid_start_hour")):
            out["coastal_start_offset_hours"] = out["coastal_valid_start_hour"]
        if _is_missing(out.get("coastal_peak_offset_hours")):
            out["coastal_peak_offset_hours"] = 0.0
        if _is_missing(out.get("coastal_end_offset_hours")) and not _is_missing(out.get("coastal_valid_end_hour")):
            out["coastal_end_offset_hours"] = out["coastal_valid_end_hour"]

        wave_start = out.get("snapwave_valid_start_time")
        wave_end = out.get("snapwave_valid_end_time")
        if (
            _is_missing(out.get("wave_start_offset_hours"))
            and not _is_missing(wave_start)
            and not _is_missing(wave_end)
        ):
            reference = pd.Timestamp(out["event_reference_time"])
            out["wave_start_offset_hours"] = (pd.Timestamp(wave_start) - reference) / pd.Timedelta(hours=1)
            out["wave_peak_offset_hours"] = 0.0
            out["wave_end_offset_hours"] = (pd.Timestamp(wave_end) - reference) / pd.Timedelta(hours=1)

    return out


def _is_missing(value) -> bool:
    return value is None or bool(pd.isna(value))


def _apply_duration_bounds(
    run_start: pd.Timestamp,
    run_stop: pd.Timestamp,
    *,
    reference: pd.Timestamp,
    min_run_hours,
    max_run_hours,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    duration = (run_stop - run_start) / pd.Timedelta(hours=1)
    if min_run_hours is not None and duration < float(min_run_hours):
        extra = pd.Timedelta(hours=(float(min_run_hours) - float(duration)) / 2)
        run_start = run_start - extra
        run_stop = run_stop + extra
        duration = float(min_run_hours)

    if max_run_hours is not None and duration > float(max_run_hours):
        half = pd.Timedelta(hours=float(max_run_hours) / 2)
        center = min(max(reference, run_start + half), run_stop - half)
        run_start = center - half
        run_stop = center + half
    return run_start, run_stop
