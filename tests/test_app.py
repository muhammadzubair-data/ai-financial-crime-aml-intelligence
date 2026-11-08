"""Smoke-test every dashboard page against the committed demo data."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = sorted((ROOT / "app" / "views").glob("p*.py"))
HAS_DEMO = (ROOT / "data" / "demo" / "manifest.json").exists()


@pytest.mark.skipif(not HAS_DEMO, reason="demo artifacts not built")
@pytest.mark.parametrize("page", VIEWS, ids=[p.stem for p in VIEWS])
def test_page_renders_without_errors(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(page), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.skipif(not HAS_DEMO, reason="demo artifacts not built")
def test_case_page_status_change_is_audited(monkeypatch, tmp_path):
    """Drive the case page: move the top case to 'In review' and check the audit trail."""
    import shutil

    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(ROOT / "app"))
    import common
    live = common.DATA / "cases_live.db"
    backup = tmp_path / "backup.db"
    had = live.exists()
    if had:
        shutil.copy(live, backup)
    shutil.copy(common.DATA / "cases.db", live)
    try:
        at = AppTest.from_file(str(ROOT / "app" / "views" / "p5_cases.py"), default_timeout=180)
        at.run()
        case = common.store().cases(at.selectbox[0].value)
        new_cases = case[case.status == "New"]
        if new_cases.empty:
            pytest.skip("no New cases in the live month")
        cid = new_cases.iloc[0].case_id
        at.session_state["case_id"] = cid
        at.run()
        sb = next(s for s in at.selectbox if s.label == "Move to")
        sb.set_value("In review")
        next(b for b in at.button if b.label == "Update status").click()
        at.run()
        assert not at.exception
        assert common.store().case(cid)["status"] == "In review"
        assert "New -> In review" in " ".join(common.store().audit(cid).details)
    finally:
        if had:
            shutil.copy(backup, live)
        else:
            live.unlink(missing_ok=True)
