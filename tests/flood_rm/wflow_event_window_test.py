from wflow_runs.replay import configured_event_window_hours, resolve_event_window


def test_wflow_event_window_adds_configured_drain_down():
    config = {"scenario_build": {"timing": {"drain_down_hours": 24}}}

    pre_event_hours, post_event_hours = configured_event_window_hours(config)
    start, end = resolve_event_window(
        "2018-09-14T18:00:00",
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )

    assert pre_event_hours == 48
    assert post_event_hours == 96
    assert str(start) == "2018-09-12 18:00:00"
    assert str(end) == "2018-09-18 18:00:00"


def test_wflow_event_window_explicit_config_wins():
    config = {
        "scenario_build": {"timing": {"drain_down_hours": 24}},
        "wflow": {"event_window": {"pre_event_hours": 36, "post_event_hours": 120}},
    }

    assert configured_event_window_hours(config) == (36, 120)
