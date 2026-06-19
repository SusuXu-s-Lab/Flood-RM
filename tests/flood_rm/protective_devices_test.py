import pandas as pd

from power.resilience import derive_lateral_fuses
from power.audit.synthetic_validation import _fuse_counts_by_feeder


def test_single_lateral_off_three_phase_trunk_gets_one_fuse():
    lines = pd.DataFrame(
        [
            {"line_name": "trunk_a", "feeder_id": "F1", "from_bus": "src", "to_bus": "j1", "phases": 3},
            {"line_name": "trunk_b", "feeder_id": "F1", "from_bus": "j1", "to_bus": "j2", "phases": 3},
            {"line_name": "tap", "feeder_id": "F1", "from_bus": "j1", "to_bus": "lat1", "phases": 1},
        ]
    )

    fuses = derive_lateral_fuses(lines)

    assert len(fuses) == 1
    fuse = fuses.iloc[0]
    assert fuse["line_name"] == "tap"
    assert fuse["head_bus"] == "j1"
    assert fuse["feeder_id"] == "F1"
    assert int(fuse["child_phases"]) == 1
    assert int(fuse["parent_phases"]) == 3


def test_phase_drop_two_to_one_also_generates_a_fuse():
    lines = pd.DataFrame(
        [
            {"line_name": "trunk", "feeder_id": "F1", "from_bus": "src", "to_bus": "j1", "phases": 2},
            {"line_name": "tap", "feeder_id": "F1", "from_bus": "j1", "to_bus": "lat1", "phases": 1},
        ]
    )

    fuses = derive_lateral_fuses(lines)

    assert len(fuses) == 1
    assert fuses.iloc[0]["line_name"] == "tap"
    assert int(fuses.iloc[0]["child_phases"]) == 1
    assert int(fuses.iloc[0]["parent_phases"]) == 2


def test_same_phase_continuation_generates_no_fuse():
    lines = pd.DataFrame(
        [
            {"line_name": "seg_a", "feeder_id": "F1", "from_bus": "src", "to_bus": "j1", "phases": 3},
            {"line_name": "seg_b", "feeder_id": "F1", "from_bus": "j1", "to_bus": "j2", "phases": 3},
            {"line_name": "seg_c", "feeder_id": "F1", "from_bus": "j2", "to_bus": "j3", "phases": 3},
        ]
    )

    assert len(derive_lateral_fuses(lines)) == 0


def test_lateral_continuation_segments_do_not_get_extra_fuses():
    lines = pd.DataFrame(
        [
            {"line_name": "trunk", "feeder_id": "F1", "from_bus": "src", "to_bus": "j1", "phases": 3},
            {"line_name": "tap", "feeder_id": "F1", "from_bus": "j1", "to_bus": "lat1", "phases": 1},
            {"line_name": "lat_cont", "feeder_id": "F1", "from_bus": "lat1", "to_bus": "lat2", "phases": 1},
        ]
    )

    fuses = derive_lateral_fuses(lines)

    assert len(fuses) == 1
    assert fuses.iloc[0]["line_name"] == "tap"


def test_multi_feeder_input_carries_feeder_id_correctly():
    lines = pd.DataFrame(
        [
            {"line_name": "f1_trunk", "feeder_id": "F1", "from_bus": "src1", "to_bus": "j1", "phases": 3},
            {"line_name": "f1_tap", "feeder_id": "F1", "from_bus": "j1", "to_bus": "lat1", "phases": 1},
            {"line_name": "f2_trunk", "feeder_id": "F2", "from_bus": "src2", "to_bus": "j2", "phases": 3},
            {"line_name": "f2_tap", "feeder_id": "F2", "from_bus": "j2", "to_bus": "lat2", "phases": 2},
        ]
    )

    fuses = derive_lateral_fuses(lines).set_index("line_name")

    assert set(fuses.index) == {"f1_tap", "f2_tap"}
    assert fuses.loc["f1_tap", "feeder_id"] == "F1"
    assert fuses.loc["f2_tap", "feeder_id"] == "F2"


def test_fuse_counts_by_feeder_aggregates_parquet_rows():
    rows = [
        {"feeder_id": "F1", "fuse_id": "fuse_1"},
        {"feeder_id": "F1", "fuse_id": "fuse_2"},
        {"feeder_id": "F1", "fuse_id": "fuse_3"},
        {"feeder_id": "F2", "fuse_id": "fuse_4"},
    ]
    counts = _fuse_counts_by_feeder(rows)
    assert counts == {"F1": 3, "F2": 1}


def test_fuse_counts_by_feeder_skips_rows_without_feeder_id():
    rows = [
        {"feeder_id": "F1", "fuse_id": "fuse_1"},
        {"feeder_id": "", "fuse_id": "fuse_2"},
        {"fuse_id": "fuse_3"},
    ]
    counts = _fuse_counts_by_feeder(rows)
    assert counts == {"F1": 1}
