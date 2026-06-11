from pathlib import Path

from app.services.filesystem import display_title_from_path, slugify, strip_upload_prefix


def test_slugify_handles_chinese_and_spaces() -> None:
    assert slugify("My Paper 2026") == "my-paper-2026"
    assert slugify("内部 知识 库") == "内部-知识-库"


def test_strip_upload_prefix_and_display_title_from_path() -> None:
    assert strip_upload_prefix("2c0bcfd3a5994fe0817122c0e2de775b-medical_test_case.md") == "medical_test_case.md"
    assert display_title_from_path(Path("2c0bcfd3a5994fe0817122c0e2de775b-medical_test_case.md")) == "medical_test_case"
    assert display_title_from_path(Path("plain-name.md")) == "plain-name"
