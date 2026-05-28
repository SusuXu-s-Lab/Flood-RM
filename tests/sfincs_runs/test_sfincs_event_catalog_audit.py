from sfincs_runs.scenarios.scenarios import assert_event_catalog_audit


def test_assert_event_catalog_audit_refuses_failed_catalog(tmp_path):
    audit = tmp_path / "catalog/event_catalog_audit.json"
    audit.parent.mkdir()
    audit.write_text('{"passed": false, "issue_count": 2, "issues": []}', encoding="utf-8")

    try:
        assert_event_catalog_audit(tmp_path)
    except RuntimeError as exc:
        assert "Event Catalog audit failed" in str(exc)
        assert "2 issues" in str(exc)
    else:
        raise AssertionError("expected failed audit to stop scenario creation")
