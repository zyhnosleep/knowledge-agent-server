from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityIssue,
    CanonicalQualityReport,
    CanonicalTable,
)


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {exc}")

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory links are unavailable: {result.stderr}")


@pytest.fixture
def source_asset(tmp_path: Path) -> Path:
    source = tmp_path / "source-figure.png"
    source.write_bytes(b"canonical-image-bytes")
    return source


@pytest.fixture
def canonical_document(source_asset: Path) -> CanonicalDocument:
    digest = hashlib.sha256(source_asset.read_bytes()).hexdigest()
    return CanonicalDocument(
        document_id="doc-1",
        source_path="source/paper.pdf",
        source_media_type="application/pdf",
        parser_source="mineru",
        parse_version="canonical-v1-abcd",
        title="Faithful Paper Title",
        abstract="Original abstract from the document.",
        keywords=["retrieval", "provenance"],
        outline=["Introduction", "Results"],
        blocks=[
            CanonicalBlock(
                block_id="later",
                block_type="narrative",
                text="Second source paragraph.",
                section_path=["Results"],
                reading_order=20,
                parser_source="mineru",
            ),
            CanonicalBlock(
                block_id="first",
                block_type="heading",
                text="Introduction",
                section_path=["Introduction"],
                reading_order=1,
                parser_source="mineru",
                metadata={"contextual_prefix": "DO NOT RENDER CONTEXT PREFIX"},
            ),
        ],
        tables=[
            CanonicalTable(
                table_id="table-1",
                caption="Source table caption",
                headers=["Model", "F1"],
                rows=[["SAC-KG", "74.7"]],
                normalized_markdown="| Model | F1 |\n|---|---|\n| SAC-KG | 74.7 |",
            )
        ],
        figures=[
            CanonicalFigure(
                figure_id="figure-1",
                caption="Source figure caption",
                asset_path="assets/figures/figure-1.png",
                generated_summary="DO NOT RENDER GENERATED FIGURE SUMMARY",
                analysis_model="vision-model",
            )
        ],
        formulas=[
            CanonicalFormula(
                formula_id="formula-1",
                latex="x = y + 1",
                description="Source formula description",
                generated_explanation="DO NOT RENDER GENERATED FORMULA EXPLANATION",
                analysis_model="language-model",
            )
        ],
        assets=[
            CanonicalAsset(
                asset_id="asset-1",
                path="assets/figures/figure-1.png",
                media_type="image/png",
                sha256=digest,
                source_path=str(source_asset),
            )
        ],
        source_metadata={"filename": "paper.pdf"},
        parser_metadata={"backend_version": "3.4.1"},
        quality=CanonicalQualityReport(
            accepted=True,
            status="accepted",
            score=0.99,
            issues=[
                CanonicalQualityIssue(
                    code="minor-warning",
                    severity="warning",
                    message="Review one formula",
                    block_ids=["later"],
                    repairable=True,
                )
            ],
        ),
        warnings=["Review one formula"],
        metadata={"language": "en"},
        status="ready",
    )


def test_artifact_store_writes_self_contained_bundle(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)

    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    assert (staging / "canonical.md").read_text("utf-8").startswith("---\n")
    assert (staging / "manifest.json").exists()
    assert (staging / "blocks.jsonl").exists()

    final = store.promote("doc-1", "canonical-v1-abcd")

    assert final.exists()
    assert not staging.exists()


def test_staging_contains_all_required_artifacts(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    assert {path.name for path in staging.iterdir()} == {
        "canonical.md",
        "manifest.json",
        "blocks.jsonl",
        "tables.json",
        "figures.json",
        "formulas.json",
        "assets",
    }
    assert (staging / "assets").is_dir()


def test_bundle_json_round_trips_canonical_models(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    manifest = json.loads((staging / "manifest.json").read_text("utf-8"))
    block_lines = (staging / "blocks.jsonl").read_text("utf-8").splitlines()
    blocks = [CanonicalBlock.model_validate_json(line) for line in block_lines]
    tables = [
        CanonicalTable.model_validate(item)
        for item in json.loads((staging / "tables.json").read_text("utf-8"))
    ]
    figures = [
        CanonicalFigure.model_validate(item)
        for item in json.loads((staging / "figures.json").read_text("utf-8"))
    ]
    formulas = [
        CanonicalFormula.model_validate(item)
        for item in json.loads((staging / "formulas.json").read_text("utf-8"))
    ]

    assert isinstance(manifest, dict)
    assert manifest["document_id"] == "doc-1"
    assert manifest["version"] == "canonical-v1-abcd"
    assert manifest["parser"]["source"] == "mineru"
    assert manifest["source"]["metadata"]["filename"] == "paper.pdf"
    assert manifest["quality"]["accepted"] is True
    assert manifest["warnings"] == ["Review one formula"]
    assert manifest["status"] == "ready"
    assert blocks == canonical_document.blocks
    assert tables == canonical_document.tables
    assert figures == canonical_document.figures
    assert formulas == canonical_document.formulas


def test_asset_is_copied_into_bundle(
    tmp_path: Path,
    source_asset: Path,
    canonical_document: CanonicalDocument,
) -> None:
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    copied = staging / "assets" / "figures" / "figure-1.png"
    manifest = json.loads((staging / "manifest.json").read_text("utf-8"))

    assert copied.read_bytes() == source_asset.read_bytes()
    assert manifest["assets"][0]["path"] == "assets/figures/figure-1.png"
    assert "source_path" not in manifest["assets"][0]


@pytest.mark.parametrize(
    "asset_path",
    [
        "../escape.png",
        "assets/../escape.png",
        "/absolute.png",
        "C:/absolute.png",
        "images/not-under-assets.png",
        "assets\\windows-ambiguous.png",
    ],
)
def test_asset_path_must_be_relative_and_under_assets(
    tmp_path: Path,
    source_asset: Path,
    canonical_document: CanonicalDocument,
    asset_path: str,
) -> None:
    canonical_document.assets[0].path = asset_path
    store = CanonicalArtifactStore(tmp_path / "store")

    with pytest.raises(ValueError, match="asset path"):
        store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    document_root = tmp_path / "store" / "doc-1"
    assert not list(document_root.glob("*.staging-*"))


@pytest.mark.parametrize(
    ("document_id", "version"),
    [
        ("../doc-1", "canonical-v1"),
        ("doc-1", "../canonical-v1"),
        ("doc-1/nested", "canonical-v1"),
        ("doc-1", "v1/nested"),
        ("C:\\outside", "canonical-v1"),
    ],
)
def test_document_and_version_path_traversal_is_rejected(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    document_id: str,
    version: str,
) -> None:
    with pytest.raises(ValueError, match="path component"):
        CanonicalArtifactStore(tmp_path).write_staging(
            document_id,
            version,
            canonical_document,
        )


def test_missing_declared_asset_fails_without_leaving_staging(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.assets[0].source_path = str(tmp_path / "missing.png")
    store = CanonicalArtifactStore(tmp_path / "store")

    with pytest.raises(FileNotFoundError, match="asset source"):
        store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    assert not list((tmp_path / "store" / "doc-1").glob("*.staging-*"))


def test_existing_final_bundle_is_not_overwritten(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    final = tmp_path / "doc-1" / "canonical-v1-abcd"
    final.mkdir()
    marker = final / "existing.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        store.promote("doc-1", "canonical-v1-abcd")

    assert marker.read_text("utf-8") == "preserve me"
    assert staging.exists()


def test_generated_analysis_and_contextual_prefix_are_excluded_from_markdown(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    markdown = (staging / "canonical.md").read_text("utf-8")

    assert "Original abstract from the document." in markdown
    assert markdown.index("Introduction") < markdown.index("Second source paragraph.")
    assert '<a id="block-first"></a>' in markdown
    assert '<a id="block-later"></a>' in markdown
    assert "DO NOT RENDER CONTEXT PREFIX" not in markdown
    assert "DO NOT RENDER GENERATED FIGURE SUMMARY" not in markdown
    assert "DO NOT RENDER GENERATED FORMULA EXPLANATION" not in markdown


def test_promote_rejects_incomplete_bundle(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    (staging / "figures.json").unlink()

    with pytest.raises(ValueError, match="incomplete canonical bundle"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()
    assert not (tmp_path / "doc-1" / "canonical-v1-abcd").exists()


@pytest.mark.parametrize(
    ("artifact_name", "replacement", "expected_message"),
    [
        ("canonical.md", "not front matter\n", "canonical.md must start"),
        (
            "manifest.json",
            '{"document_id":"doc-1","version":"canonical-v1-abcd"}\n',
            "manifest missing required fields",
        ),
    ],
)
def test_promote_rejects_malformed_bundle_metadata(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    artifact_name: str,
    replacement: str,
    expected_message: str,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    (staging / artifact_name).write_text(replacement, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()


def test_promote_rejects_malformed_manifest_field_types(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["quality"] = "not a quality report"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest quality"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()


def test_promote_rejects_symlinked_assets_directory(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    external_assets = tmp_path / "external-assets"
    shutil.copytree(staging / "assets", external_assets)
    shutil.rmtree(staging / "assets")
    asset_link = staging / "assets"
    _create_directory_link(asset_link, external_assets)

    with pytest.raises(ValueError, match="symbolic link"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()
    assert not (tmp_path / "doc-1" / "canonical-v1-abcd").exists()
    if asset_link.is_symlink():
        asset_link.unlink()
    else:
        os.rmdir(asset_link)


def test_promote_requires_exactly_one_staging_directory(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="exactly one staging"):
        store.promote("doc-1", "canonical-v1-abcd")

    store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    second = tmp_path / "doc-1" / "canonical-v1-abcd.staging-manual"
    second.mkdir()

    with pytest.raises(ValueError, match="exactly one staging"):
        store.promote("doc-1", "canonical-v1-abcd")
