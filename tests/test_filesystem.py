import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from app.services.filesystem import (
    InvalidStoragePathError,
    UploadTooLargeError,
    display_title_from_path,
    project_paths,
    save_upload,
    safe_project_slug,
    slugify,
    strip_upload_prefix,
)


def test_slugify_handles_chinese_and_spaces() -> None:
    assert slugify("My Paper 2026") == "my-paper-2026"
    assert slugify("内部 知识 库") == "内部-知识-库"


def test_strip_upload_prefix_and_display_title_from_path() -> None:
    assert strip_upload_prefix("2c0bcfd3a5994fe0817122c0e2de775b-medical_test_case.md") == "medical_test_case.md"
    assert display_title_from_path(Path("2c0bcfd3a5994fe0817122c0e2de775b-medical_test_case.md")) == "medical_test_case"
    assert display_title_from_path(Path("plain-name.md")) == "plain-name"


def test_project_paths_rejects_path_traversal(tmp_path, monkeypatch) -> None:
    from app.services import filesystem

    monkeypatch.setattr(filesystem.settings, "raw_dir", tmp_path / "raw")

    with pytest.raises(InvalidStoragePathError):
        project_paths("../outside")

    for unsafe_slug in ("bad slug", ".", "...", "C:demo"):
        with pytest.raises(InvalidStoragePathError):
            safe_project_slug(unsafe_slug)

    assert safe_project_slug("demo-项目_1") == "demo-项目_1"


def test_save_upload_sanitizes_filename_and_rejects_traversal(tmp_path, monkeypatch) -> None:
    from app.services import filesystem

    monkeypatch.setattr(filesystem.settings, "raw_dir", tmp_path / "raw")
    monkeypatch.setattr(filesystem.settings, "max_upload_bytes", 1024)
    upload = UploadFile(file=BytesIO(b"content"), filename="../evil.pdf")

    with pytest.raises(InvalidStoragePathError):
        asyncio.run(save_upload("demo", upload))


def test_save_upload_deletes_partial_file_when_upload_exceeds_limit(tmp_path, monkeypatch) -> None:
    from app.services import filesystem

    monkeypatch.setattr(filesystem.settings, "raw_dir", tmp_path / "raw")
    monkeypatch.setattr(filesystem.settings, "max_upload_bytes", 5)
    upload = UploadFile(file=BytesIO(b"123456"), filename="paper.pdf")

    with pytest.raises(UploadTooLargeError):
        asyncio.run(save_upload("demo", upload))

    assert list((tmp_path / "raw" / "demo").glob("*")) == []


def test_save_upload_writes_safe_file_without_partial_suffix(tmp_path, monkeypatch) -> None:
    from app.services import filesystem

    monkeypatch.setattr(filesystem.settings, "raw_dir", tmp_path / "raw")
    monkeypatch.setattr(filesystem.settings, "max_upload_bytes", 1024)
    upload = UploadFile(file=BytesIO(b"content"), filename="paper.pdf")

    saved_path = asyncio.run(save_upload("demo", upload))

    assert saved_path.parent == tmp_path / "raw" / "demo"
    assert saved_path.name.endswith("-paper.pdf")
    assert saved_path.read_bytes() == b"content"
    assert not list(saved_path.parent.glob("*.part"))
