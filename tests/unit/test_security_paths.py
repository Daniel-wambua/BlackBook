import pytest

from blackbook.utils.paths import is_within, normalize_rel_path, safe_join


def test_normalize_rejects_traversal():
    with pytest.raises(ValueError):
        normalize_rel_path("../etc/passwd")
    with pytest.raises(ValueError):
        normalize_rel_path("a/../../b")


def test_normalize_rejects_absolute():
    with pytest.raises(ValueError):
        normalize_rel_path("/etc/passwd")


def test_safe_join_stays_within(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    ok = safe_join(base, "sub/file.txt")
    assert is_within(ok, base)
    with pytest.raises(ValueError):
        safe_join(base, "../../escape.txt")


def test_is_within(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    child = base / "a" / "b.txt"
    child.parent.mkdir(parents=True)
    child.write_text("x")
    assert is_within(child, base)
    assert not is_within(tmp_path, base)
