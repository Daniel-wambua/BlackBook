"""Regression tests for runtime and configuration hardening."""

from __future__ import annotations

from blackbook.config import load_config
from blackbook.storage.database import Database


def test_database_migration_is_not_left_locked_for_second_process(tmp_path):
    path = tmp_path / "data.db"
    first = Database(path)
    try:
        second = Database(path)
        second.close()
    finally:
        first.close()


def test_yaml_hex_like_source_name_stays_string(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """sources:\n  - id: \"0xdf\"\n    name: \"0xdf\"\n    type: website\n    authority: trusted\n    url: https://0xdf.gitlab.io/\n""",
        encoding="utf-8",
    )
    settings = load_config(config)
    source = settings.get_source("0xdf")
    assert source is not None
    assert source.name == "0xdf"
