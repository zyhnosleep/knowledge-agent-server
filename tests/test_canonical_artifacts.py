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
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityIssue,
    CanonicalQualityReport,
    CanonicalTable,
    SectionNode,
    SourceSpan,
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
        outline=[
            SectionNode(
                title="Introduction",
                level=1,
                block_id="first",
                children=[SectionNode(title="Motivation", level=2)],
            ),
            SectionNode(title="Results", level=1),
        ],
        blocks=[
            CanonicalBlock(
                block_id="later",
                block_type="narrative",
                text="Second source paragraph.",
                section_path=["Results"],
                reading_order=20,
                parser_source="mineru",
                source_spans=[SourceSpan(page_index=3, source_block_id="pdf-20")],
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
                description="Original source description",
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
    assert manifest["canonical_markdown_sha256"] == hashlib.sha256(
        (staging / "canonical.md").read_bytes()
    ).hexdigest()
    assert blocks == canonical_document.blocks
    assert tables == canonical_document.tables
    assert figures == canonical_document.figures
    assert formulas == canonical_document.formulas


def test_promoted_bundle_loads_complete_canonical_document(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    store.promote("doc-1", "canonical-v1-abcd")

    restored = store.load("doc-1", "canonical-v1-abcd")
    expected = canonical_document.model_copy(deep=True)
    for asset in expected.assets:
        asset.source_path = None

    assert restored == expected
    assert restored.outline[0].children[0].title == "Motivation"
    assert restored.blocks[0].source_spans[0].page_index == 3


def test_manifest_document_section_persists_non_transient_fields(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    manifest = json.loads((staging / "manifest.json").read_text("utf-8"))

    assert manifest["document"] == {
        "title": canonical_document.title,
        "abstract": canonical_document.abstract,
        "keywords": canonical_document.keywords,
        "outline": [item.model_dump(mode="json") for item in canonical_document.outline],
        "metadata": canonical_document.metadata,
    }


@pytest.mark.parametrize(
    ("document_id", "version", "field", "conflicting_value"),
    [
        ("doc-1", "canonical-v1-abcd", "document_id", "other-doc"),
        ("doc-1", "canonical-v1-abcd", "parse_version", "other-version"),
    ],
)
def test_write_rejects_document_identity_conflicts(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    document_id: str,
    version: str,
    field: str,
    conflicting_value: str,
) -> None:
    setattr(canonical_document, field, conflicting_value)

    with pytest.raises(ValueError, match="does not match"):
        CanonicalArtifactStore(tmp_path).write_staging(
            document_id,
            version,
            canonical_document,
        )


def test_empty_document_identity_is_filled_by_bundle_identity(tmp_path: Path) -> None:
    document = CanonicalDocument(title="Untitled source")
    store = CanonicalArtifactStore(tmp_path)
    store.write_staging("doc-empty", "canonical-v2", document)
    store.promote("doc-empty", "canonical-v2")

    restored = store.load("doc-empty", "canonical-v2")

    assert restored.document_id == "doc-empty"
    assert restored.parse_version == "canonical-v2"


@pytest.mark.parametrize(
    "absolute_source_path",
    [
        "C:/Users/secret/paper.pdf",
        "/home/secret/paper.pdf",
        "\\Users\\secret\\paper.pdf",
        "C:Users\\secret\\paper.pdf",
    ],
)
def test_manifest_does_not_disclose_absolute_source_directories(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    absolute_source_path: str,
) -> None:
    canonical_document.source_path = absolute_source_path
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    manifest_text = (staging / "manifest.json").read_text("utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["source"]["path"] == "paper.pdf"
    assert "Users" not in manifest_text
    assert "home" not in manifest_text
    assert "secret" not in manifest_text


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


def test_asset_hash_is_computed_when_caller_omits_hash(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    source_path = Path(canonical_document.assets[0].source_path or "")
    expected_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    canonical_document.assets[0].sha256 = None
    store = CanonicalArtifactStore(tmp_path)

    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest = json.loads((staging / "manifest.json").read_text("utf-8"))

    assert manifest["assets"][0]["sha256"] == expected_digest
    store.promote("doc-1", "canonical-v1-abcd")
    restored = store.load("doc-1", "canonical-v1-abcd")
    assert restored.assets[0].sha256 == expected_digest


@pytest.mark.parametrize("provided_hash", ["not-a-hash", "A" * 64])
def test_asset_hash_must_be_lowercase_64_hex_when_provided(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    provided_hash: str,
) -> None:
    canonical_document.assets[0].sha256 = provided_hash

    with pytest.raises(ValueError, match="asset sha256"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


def test_promote_rejects_manifest_asset_without_persisted_hash(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["assets"][0]["sha256"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="asset sha256"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_load_rejects_manifest_asset_without_persisted_hash(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    final = store.promote("doc-1", "canonical-v1-abcd")
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["assets"][0]["sha256"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="asset sha256"):
        store.load("doc-1", "canonical-v1-abcd")


@pytest.mark.parametrize(
    "asset_path",
    [
        "../escape.png",
        "assets/../escape.png",
        "/absolute.png",
        "C:/absolute.png",
        "images/not-under-assets.png",
        "assets\\windows-ambiguous.png",
        "assets/CON",
        "assets/NUL.txt",
        "assets/name.",
        "assets/file:stream",
        "assets/bad?.png",
        "assets/control\x01.png",
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
        ("CON", "canonical-v1"),
        ("doc-1", "NUL.txt"),
        ("doc-1", "name."),
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


def test_asset_path_allows_unicode_and_internal_spaces(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    asset_path = "assets/图 表/figure one.png"
    canonical_document.assets[0].path = asset_path
    canonical_document.figures[0].asset_path = asset_path

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    assert (staging / "assets" / "图 表" / "figure one.png").is_file()
    markdown = (staging / "canonical.md").read_text("utf-8")
    assert (
        "![Source figure caption]"
        "(assets/%E5%9B%BE%20%E8%A1%A8/figure%20one.png)"
    ) in markdown


def test_write_rejects_symlinked_document_root_without_writing_external_target(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store_root = tmp_path / "store"
    external_document_root = tmp_path / "external-document"
    store_root.mkdir()
    external_document_root.mkdir()
    document_root = store_root / "doc-1"
    _create_directory_link(document_root, external_document_root)

    with pytest.raises(ValueError, match="document root"):
        CanonicalArtifactStore(store_root).write_staging(
            "doc-1",
            "canonical-v1-abcd",
            canonical_document,
        )

    assert list(external_document_root.iterdir()) == []
    if document_root.is_symlink():
        document_root.unlink()
    else:
        os.rmdir(document_root)


def test_promote_rejects_symlinked_document_root_without_renaming_external_staging(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    external_store_root = tmp_path / "external-store"
    external_store = CanonicalArtifactStore(external_store_root)
    external_staging = external_store.write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    store_root = tmp_path / "store"
    store_root.mkdir()
    document_root = store_root / "doc-1"
    _create_directory_link(document_root, external_store_root / "doc-1")

    with pytest.raises(ValueError, match="document root"):
        CanonicalArtifactStore(store_root).promote("doc-1", "canonical-v1-abcd")

    assert external_staging.exists()
    assert not (external_store_root / "doc-1" / "canonical-v1-abcd").exists()
    if document_root.is_symlink():
        document_root.unlink()
    else:
        os.rmdir(document_root)


def test_load_rejects_symlinked_document_root_without_reading_external_bundle(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    external_store_root = tmp_path / "external-store"
    external_store = CanonicalArtifactStore(external_store_root)
    external_store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    external_store.promote("doc-1", "canonical-v1-abcd")
    store_root = tmp_path / "store"
    store_root.mkdir()
    document_root = store_root / "doc-1"
    _create_directory_link(document_root, external_store_root / "doc-1")

    with pytest.raises(ValueError, match="document root"):
        CanonicalArtifactStore(store_root).load("doc-1", "canonical-v1-abcd")

    if document_root.is_symlink():
        document_root.unlink()
    else:
        os.rmdir(document_root)


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


def test_generated_blocks_are_excluded_from_canonical_markdown_but_analysis_stays_json(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.blocks.append(
        CanonicalBlock(
            block_id="generated-analysis",
            block_type="narrative",
            text="DO NOT RENDER GENERATED BLOCK ANALYSIS",
            reading_order=30,
            parser_source="vision-model",
            metadata={"generated": True, "provenance": "ai_analysis"},
        )
    )

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    markdown = (staging / "canonical.md").read_text("utf-8")
    blocks = (staging / "blocks.jsonl").read_text("utf-8")
    figures = json.loads((staging / "figures.json").read_text("utf-8"))

    assert "DO NOT RENDER GENERATED BLOCK ANALYSIS" not in markdown
    assert "DO NOT RENDER GENERATED BLOCK ANALYSIS" not in blocks
    assert "DO NOT RENDER GENERATED FIGURE SUMMARY" not in markdown
    assert figures[0]["generated_summary"] == "DO NOT RENDER GENERATED FIGURE SUMMARY"
    assert figures[0]["analysis_model"] == "vision-model"


def test_staged_figure_asset_survives_mineru_temporary_directory_cleanup(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    source_path = Path(canonical_document.assets[0].source_path or "")
    expected = source_path.read_bytes()

    staging = CanonicalArtifactStore(tmp_path / "store").write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    source_path.unlink()

    assert (staging / "assets" / "figures" / "figure-1.png").read_bytes() == expected


def test_markdown_includes_unreferenced_structured_source_evidence(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []
    canonical_document.tables[0].footnotes = ["Source table footnote."]
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    markdown = (staging / "canonical.md").read_text("utf-8")

    assert "Source table caption" in markdown
    assert "| SAC-KG | 74.7 |" in markdown
    assert "Source table footnote." in markdown
    assert "![Source figure caption](assets/figures/figure-1.png)" in markdown
    assert "Original source description" in markdown
    assert "$$\nx = y + 1\n$$" in markdown
    assert "Source formula description" in markdown
    assert "DO NOT RENDER GENERATED FIGURE SUMMARY" not in markdown
    assert "DO NOT RENDER GENERATED FORMULA EXPLANATION" not in markdown


@pytest.mark.parametrize("table_source", ["source_markdown", "generated"])
def test_markdown_uses_faithful_table_fallbacks(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    table_source: str,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    if table_source == "source_markdown":
        table.source_markdown = "| Original | Value |\n|---|---|\n| Row | 1 |"
        expected = "| Original | Value |"
    else:
        table.source_markdown = None
        expected = "| Model | F1 |\n| --- | --- |\n| SAC-KG | 74.7 |"
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )

    assert expected in (staging / "canonical.md").read_text("utf-8")


def test_markdown_keeps_linked_block_text_and_formats_heading_blocks(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.blocks.append(
        CanonicalBlock(
            block_id="table-block",
            block_type="table",
            text="RAW TABLE BLOCK SHOULD BE REPLACED",
            section_path=["Results"],
            reading_order=10,
            parser_source="mineru",
            table_id="table-1",
        )
    )

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1",
        "canonical-v1-abcd",
        canonical_document,
    )
    markdown = (staging / "canonical.md").read_text("utf-8")

    assert "# Introduction" in markdown
    assert "RAW TABLE BLOCK SHOULD BE REPLACED" in markdown
    assert markdown.count("| SAC-KG | 74.7 |") == 1


def test_bundle_rejects_extra_top_level_regular_file(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    (staging / "unexpected.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level bundle inventory"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_bundle_rejects_extra_top_level_directory(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    (staging / "unexpected").mkdir()

    with pytest.raises(ValueError, match="top-level bundle inventory"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_bundle_rejects_extra_top_level_link(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = staging / "unexpected-link"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="top-level bundle inventory"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_bundle_rejects_extra_top_level_directory_link(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    target = tmp_path / "outside-directory"
    target.mkdir()
    _create_directory_link(staging / "unexpected-directory-link", target)

    with pytest.raises(ValueError, match="top-level bundle inventory"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_bundle_metadata_paths_are_sanitized_without_touching_text_or_urls(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    secret = "C:\\secret\\project\\private.pdf"
    canonical_document.metadata = {
        "absolute": secret,
        "posix": "/srv/secret/private.json",
        "drive_relative": "C:relative\\private.txt",
        "relative": "docs/private.txt",
        "url": "https://example.test/secret/private.pdf",
        "nested": [{"path": "/nested/secret.bin"}],
    }
    canonical_document.blocks[0].metadata = {"path": secret}
    canonical_document.source_metadata = {"path": secret}
    canonical_document.parser_metadata = {"path": secret}
    canonical_document.tables[0].metadata = {"path": secret}
    canonical_document.figures[0].metadata = {"path": secret}
    canonical_document.formulas[0].metadata = {"path": secret}
    canonical_document.assets[0].metadata = {"path": secret}
    canonical_document.quality.metadata = {"path": secret}
    canonical_document.quality.issues[0].metadata = {"path": secret}
    canonical_document.blocks[0].source_spans[0].metadata = {"path": secret}

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    persisted = "\n".join(
        path.read_text("utf-8")
        for path in [
            staging / "manifest.json",
            staging / "blocks.jsonl",
            staging / "tables.json",
            staging / "figures.json",
            staging / "formulas.json",
        ]
    )
    assert "secret\\\\project" not in persisted
    assert "/srv/secret" not in persisted
    assert '"absolute": "private.pdf"' in persisted
    assert '"relative": "docs/private.txt"' in persisted
    assert "https://example.test/secret/private.pdf" in persisted
    store = CanonicalArtifactStore(tmp_path)
    store.promote("doc-1", "canonical-v1-abcd")
    loaded = store.load("doc-1", "canonical-v1-abcd")
    assert loaded.metadata["absolute"] == "private.pdf"
    assert loaded.metadata["nested"] == [{"path": "secret.bin"}]
    assert loaded.metadata["relative"] == "docs/private.txt"
    assert loaded.metadata["url"] == "https://example.test/secret/private.pdf"


def test_markdown_keeps_block_text_and_renders_multiple_references_in_order(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.blocks = [
        CanonicalBlock(
            block_id="multi",
            block_type="narrative",
            text="Narrative source text.",
            section_path=["Results"],
            reading_order=1,
            parser_source="mineru",
            table_id="table-1",
            figure_id="figure-1",
            formula_id="formula-1",
        )
    ]
    canonical_document.outline = [SectionNode(title="Results", level=1, block_id="multi")]
    canonical_document.quality.issues[0].block_ids = ["multi"]
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    markdown = (staging / "canonical.md").read_text("utf-8")
    text_index = markdown.index("Narrative source text.")
    assert text_index < markdown.index("Source table caption", text_index)
    assert markdown.index("Source table caption", text_index) < markdown.index(
        "Source figure caption", text_index
    )
    assert markdown.index("Source figure caption", text_index) < markdown.index(
        "x = y + 1", text_index
    )


def test_fallback_table_and_figure_markdown_escape_special_characters(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = ["H|1", "H\\2"]
    table.rows = [["a|b", "line1\r\nline2"]]
    figure = canonical_document.figures[0]
    figure.caption = "alt [caption]\\value\nnext"
    figure.asset_path = "assets/figures/figure%20one.png"
    canonical_document.assets[0].path = figure.asset_path
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    markdown = (staging / "canonical.md").read_text("utf-8")
    assert "H\\|1" in markdown
    assert "H\\\\2" in markdown
    assert "a\\|b" in markdown
    assert "line1<br>line2" in markdown
    assert "![alt \\[caption\\]\\\\value next](assets/figures/figure%2520one.png)" in markdown


def test_fallback_table_rejects_rows_with_wrong_width(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = ["H1", "H2"]
    table.rows = [["only-one"]]
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    with pytest.raises(ValueError, match="table row width"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


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


def test_promote_rejects_canonical_markdown_body_tampering(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    markdown_path = staging / "canonical.md"
    original = markdown_path.read_text("utf-8")
    markdown_path.write_text(
        original + "\nAI-injected interpretation not present in the source.\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="canonical.md sha256 mismatch"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()


def test_load_rejects_canonical_markdown_body_tampering(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    final = store.promote("doc-1", "canonical-v1-abcd")
    markdown_path = final / "canonical.md"
    original = markdown_path.read_text("utf-8")
    markdown_path.write_text(
        original.replace("Second source paragraph.", "AI replacement text."),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="canonical.md sha256 mismatch"):
        store.load("doc-1", "canonical-v1-abcd")


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


def test_promote_rejects_unknown_document_manifest_fields(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["document"]["contextual_prefix"] = "not canonical"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields"):
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


def test_promote_rejects_undeclared_extra_asset(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    (staging / "assets" / "extra.bin").write_bytes(b"undeclared")

    with pytest.raises(ValueError, match="asset inventory"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()


def test_promote_rejects_nested_asset_directory_link(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    external = tmp_path / "external-nested"
    external.mkdir()
    nested_link = staging / "assets" / "figures" / "linked"
    _create_directory_link(nested_link, external)

    with pytest.raises(ValueError, match="symbolic link"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()
    if nested_link.is_symlink():
        nested_link.unlink()
    else:
        os.rmdir(nested_link)


def test_write_rejects_undeclared_figure_asset(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.figures[0].asset_path = "assets/not-declared.png"

    with pytest.raises(ValueError, match="figure asset"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1",
            "canonical-v1-abcd",
            canonical_document,
        )


@pytest.mark.parametrize(
    "duplicate_kind",
    ["block", "table", "figure", "formula", "asset_id", "asset_path"],
)
def test_write_rejects_duplicate_canonical_identifiers(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    duplicate_kind: str,
) -> None:
    if duplicate_kind == "block":
        canonical_document.blocks.append(canonical_document.blocks[0].model_copy())
    elif duplicate_kind == "table":
        canonical_document.tables.append(canonical_document.tables[0].model_copy())
    elif duplicate_kind == "figure":
        canonical_document.figures.append(canonical_document.figures[0].model_copy())
    elif duplicate_kind == "formula":
        canonical_document.formulas.append(canonical_document.formulas[0].model_copy())
    elif duplicate_kind == "asset_id":
        duplicate = canonical_document.assets[0].model_copy()
        duplicate.path = "assets/figures/copy.png"
        canonical_document.assets.append(duplicate)
    else:
        duplicate = canonical_document.assets[0].model_copy()
        duplicate.asset_id = "asset-copy"
        canonical_document.assets.append(duplicate)

    with pytest.raises(ValueError, match="duplicate"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1",
            "canonical-v1-abcd",
            canonical_document,
        )


@pytest.mark.parametrize("reference_field", ["table_id", "figure_id", "formula_id"])
def test_write_rejects_broken_block_references(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    reference_field: str,
) -> None:
    setattr(canonical_document.blocks[0], reference_field, "missing-record")

    with pytest.raises(ValueError, match="block reference"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1",
            "canonical-v1-abcd",
            canonical_document,
        )


@pytest.mark.parametrize(
    "reference_kind",
    ["outline", "figure_nearby", "formula_nearby", "quality_issue"],
)
def test_write_rejects_dangling_canonical_block_references(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    reference_kind: str,
) -> None:
    if reference_kind == "outline":
        canonical_document.outline[0].children[0].block_id = "missing-block"
    elif reference_kind == "figure_nearby":
        canonical_document.figures[0].nearby_block_ids = ["missing-block"]
    elif reference_kind == "formula_nearby":
        canonical_document.formulas[0].nearby_block_ids = ["missing-block"]
    else:
        canonical_document.quality.issues[0].block_ids = ["missing-block"]

    with pytest.raises(ValueError, match="canonical block reference"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1",
            "canonical-v1-abcd",
            canonical_document,
        )


def test_promote_rejects_tampered_outline_block_reference(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["document"]["outline"][0]["block_id"] = "missing-block"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical block reference"):
        store.promote("doc-1", "canonical-v1-abcd")

    assert staging.exists()


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


def test_promote_on_missing_store_root_preserves_empty_staging_contract(
    tmp_path: Path,
) -> None:
    store = CanonicalArtifactStore(tmp_path / "missing-store")

    with pytest.raises(
        ValueError,
        match=r"promotion requires exactly one staging directory; found 0",
    ):
        store.promote("doc-1", "canonical-v1-abcd")

    assert not store.root.exists()


def test_write_staging_is_idempotent_and_reuses_valid_bundle(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)

    first = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    second = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    assert second == first
    assert list((tmp_path / "doc-1").glob("canonical-v1-abcd.staging-*")) == [first]
    assert store.promote("doc-1", "canonical-v1-abcd").is_dir()


def test_write_staging_rejects_reuse_with_different_canonical_input(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    first = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    changed = canonical_document.model_copy(deep=True)
    changed.source_path = "different/source.pdf"
    changed.title = "Different title"
    changed.parser_metadata = {"backend_version": "9.9.9"}
    changed.blocks[0].text = "Different source paragraph."

    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        store.write_staging("doc-1", "canonical-v1-abcd", changed)

    assert list((tmp_path / "doc-1").glob("canonical-v1-abcd.staging-*")) == [first]


def test_write_staging_validates_incoming_records_before_reusing_bundle(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    first = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    broken = canonical_document.model_copy(deep=True)
    broken.blocks[0].table_id = "missing-table"

    with pytest.raises(ValueError, match="broken block reference"):
        store.write_staging("doc-1", "canonical-v1-abcd", broken)

    assert list((tmp_path / "doc-1").glob("canonical-v1-abcd.staging-*")) == [first]


def test_promote_rejects_blocks_tampered_without_updating_input_fingerprint(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    blocks_path = staging / "blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text("utf-8").splitlines()]
    blocks[0]["text"] = "Tampered block text."
    blocks_path.write_text(
        "\n".join(json.dumps(block) for block in blocks) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_promote_rejects_invalid_input_fingerprint_format(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["input_fingerprint"] = "not-a-fingerprint"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="input_fingerprint must be lowercase 64-hex"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_markdown_percent_encodes_literal_percent_in_asset_path(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.figures[0].asset_path = "assets/figure%20one.png"
    canonical_document.assets[0].path = "assets/figure%20one.png"
    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )

    markdown = (staging / "canonical.md").read_text("utf-8")
    assert "](assets/figure%2520one.png)" in markdown
    assert (staging / "assets" / "figure%20one.png").is_file()


@pytest.mark.parametrize("operation", ["promote", "load"])
def test_bundle_rejects_noncanonical_absolute_persisted_source_path(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    operation: str,
) -> None:
    canonical_document.source_path = "C:\\safe\\paper.pdf"
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    bundle = staging
    if operation == "load":
        bundle = store.promote("doc-1", "canonical-v1-abcd")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["source"]["path"] == "paper.pdf"
    manifest["source"]["path"] = "C:\\secret\\paper.pdf"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="persisted canonical input is not canonical"):
        if operation == "promote":
            store.promote("doc-1", "canonical-v1-abcd")
        else:
            store.load("doc-1", "canonical-v1-abcd")


def test_bundle_rejects_noncanonical_absolute_manifest_metadata(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.metadata["path"] = "paper.pdf"
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["document"]["metadata"]["path"] = "C:\\secret\\paper.pdf"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="persisted canonical input is not canonical"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_bundle_rejects_noncanonical_absolute_structured_metadata(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    canonical_document.blocks[0].metadata["path"] = "paper.pdf"
    canonical_document.tables[0].metadata["path"] = "table.json"
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    blocks_path = staging / "blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text("utf-8").splitlines()]
    blocks[0]["metadata"]["path"] = "/secret/paper.pdf"
    blocks_path.write_text(
        "\n".join(json.dumps(block) for block in blocks) + "\n",
        encoding="utf-8",
    )
    tables_path = staging / "tables.json"
    tables = json.loads(tables_path.read_text("utf-8"))
    tables[0]["metadata"]["path"] = "C:\\secret\\table.json"
    tables_path.write_text(json.dumps(tables), encoding="utf-8")

    with pytest.raises(ValueError, match="persisted canonical input is not canonical"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_write_staging_reuses_provided_asset_hash_without_runtime_source(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    first = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    assert canonical_document.assets[0].sha256 is not None
    Path(canonical_document.assets[0].source_path or "").unlink()

    second = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    assert second == first


def test_write_staging_rejects_wrong_provided_hash_without_runtime_source(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    first = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    Path(canonical_document.assets[0].source_path or "").unlink()
    canonical_document.assets[0].sha256 = "0" * 64

    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    assert first.is_dir()


@pytest.mark.parametrize("operation", ["promote", "load"])
def test_bundle_rejects_block_with_omitted_persisted_default(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    operation: str,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    bundle = staging
    if operation == "load":
        bundle = store.promote("doc-1", "canonical-v1-abcd")
    blocks_path = bundle / "blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text("utf-8").splitlines()]
    del blocks[0]["retrievable"]
    blocks_path.write_text(
        "\n".join(json.dumps(block) for block in blocks) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="persisted canonical input is not canonical"):
        if operation == "promote":
            store.promote("doc-1", "canonical-v1-abcd")
        else:
            store.load("doc-1", "canonical-v1-abcd")


@pytest.mark.parametrize(
    ("artifact_name", "field"),
    [("tables.json", "status"), ("figures.json", "warnings")],
)
def test_bundle_rejects_structured_record_with_omitted_persisted_default(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    artifact_name: str,
    field: str,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    artifact_path = staging / artifact_name
    records = json.loads(artifact_path.read_text("utf-8"))
    del records[0][field]
    artifact_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="persisted canonical input is not canonical"):
        store.promote("doc-1", "canonical-v1-abcd")


def test_markdown_renders_cells_only_table_evidence(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = []
    table.rows = []
    table.cells = [
        CanonicalCell(
            text="Model", row_index=0, column_index=0, is_header=True
        ),
        CanonicalCell(
            text="Score", row_index=0, column_index=1, is_header=True
        ),
        CanonicalCell(text="SAC|KG", row_index=1, column_index=0),
        CanonicalCell(text="74.7", row_index=1, column_index=1),
    ]
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    markdown = (staging / "canonical.md").read_text("utf-8")

    assert "| Model | Score |" in markdown
    assert "| SAC\\|KG | 74.7 |" in markdown


def test_markdown_renders_source_html_table_without_unsafe_content(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = []
    table.rows = []
    table.cells = []
    table.source_html = """
        <table>
          <thead><tr><th>Model</th><th>Score</th></tr></thead>
          <tbody><tr><td>SAC-KG<script>unsafe()</script></td><td>74.7</td></tr></tbody>
        </table>
        <style>unsafe-style</style>
    """
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    markdown = (staging / "canonical.md").read_text("utf-8")

    assert "| Model | Score |" in markdown
    assert "| SAC-KG | 74.7 |" in markdown
    assert "unsafe()" not in markdown
    assert "unsafe-style" not in markdown


def test_write_rejects_nonempty_table_without_renderable_rows(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = []
    table.rows = []
    table.cells = []
    table.source_html = "<div>No table rows</div>"
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    with pytest.raises(ValueError, match="no renderable table evidence"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


def test_file_uri_paths_are_sanitized_in_source_and_all_metadata(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    windows_uri = "file:///C:/Users/secret/paper%20one.pdf"
    posix_uri = "file:///home/secret/private%20data.json"
    unc_uri = "file://server/share/private%20figure.png"
    canonical_document.source_path = windows_uri
    canonical_document.metadata = {
        "file": windows_uri,
        "https": "https://example.test/secret/paper.pdf",
        "relative": "docs/paper.pdf",
    }
    canonical_document.parser_metadata = {"file": posix_uri}
    canonical_document.source_metadata = {"file": unc_uri}
    canonical_document.blocks[0].metadata = {"file": windows_uri}
    canonical_document.blocks[0].source_spans[0].metadata = {"file": posix_uri}
    canonical_document.tables[0].metadata = {"file": posix_uri}
    canonical_document.figures[0].metadata = {"file": unc_uri}
    canonical_document.formulas[0].metadata = {"file": windows_uri}
    canonical_document.assets[0].metadata = {"file": unc_uri}
    canonical_document.quality.metadata = {"file": posix_uri}
    canonical_document.quality.issues[0].metadata = {"file": windows_uri}

    staging = CanonicalArtifactStore(tmp_path).write_staging(
        "doc-1", "canonical-v1-abcd", canonical_document
    )
    persisted = "\n".join(
        path.read_text("utf-8")
        for path in [
            staging / "manifest.json",
            staging / "blocks.jsonl",
            staging / "tables.json",
            staging / "figures.json",
            staging / "formulas.json",
        ]
    )
    manifest = json.loads((staging / "manifest.json").read_text("utf-8"))

    assert "file://" not in persisted
    assert "Users" not in persisted
    assert "/home/secret" not in persisted
    assert manifest["source"]["path"] == "paper one.pdf"
    assert "paper one.pdf" in persisted
    assert "private data.json" in persisted
    assert "private figure.png" in persisted
    assert "https://example.test/secret/paper.pdf" in persisted
    assert "docs/paper.pdf" in persisted


@pytest.mark.parametrize("target", ["block_metadata", "figure_ai"])
def test_write_rejects_non_finite_values_added_after_model_validation(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    target: str,
) -> None:
    if target == "block_metadata":
        canonical_document.blocks[0].metadata["bad"] = float("nan")
    else:
        canonical_document.figures[0].ai_axes = {"bad": float("inf")}

    with pytest.raises(ValueError, match="finite JSON"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


@pytest.mark.parametrize(
    "cell",
    [
        CanonicalCell(text="too many rows", row_index=10_000, column_index=0),
        CanonicalCell(text="too many columns", row_index=0, column_index=1_000),
        CanonicalCell(text="too many cells", row_index=1_999, column_index=500),
    ],
)
def test_cells_table_rejects_oversized_dense_matrix(cell: CanonicalCell) -> None:
    with pytest.raises(ValueError, match="table matrix exceeds canonical limits"):
        CanonicalArtifactStore._table_grid_from_cells([cell])


def test_source_html_rejects_oversized_span_before_matrix_allocation(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = []
    table.rows = []
    table.cells = []
    table.source_html = '<table><tr><td rowspan="10001">value</td></tr></table>'
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    with pytest.raises(ValueError, match="table matrix exceeds canonical limits"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


@pytest.mark.parametrize(
    "empty_kind",
    ["empty", "normalized_empty", "source_html_empty", "empty_html", "decorations"],
)
def test_write_rejects_table_without_actual_table_evidence(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
    empty_kind: str,
) -> None:
    table = canonical_document.tables[0]
    table.normalized_markdown = None
    table.source_markdown = None
    table.headers = []
    table.rows = []
    table.cells = []
    table.source_html = None
    table.caption = None
    table.footnotes = []
    if empty_kind == "normalized_empty":
        table.normalized_markdown = ""
    elif empty_kind == "source_html_empty":
        table.source_html = ""
    elif empty_kind == "empty_html":
        table.source_html = "<table></table>"
    elif empty_kind == "decorations":
        table.caption = "Caption is not table evidence"
        table.footnotes = ["Footnote is not table evidence"]
    canonical_document.blocks = []
    canonical_document.outline[0].block_id = None
    canonical_document.quality.issues[0].block_ids = []

    with pytest.raises(ValueError, match="no renderable table evidence"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )


def test_write_staging_rejects_existing_final_without_orphan(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    store = CanonicalArtifactStore(tmp_path)
    store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    final = store.promote("doc-1", "canonical-v1-abcd")

    with pytest.raises(FileExistsError, match="already exists"):
        store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)

    assert final.is_dir()
    assert not list((tmp_path / "doc-1").glob("canonical-v1-abcd.staging-*"))


def test_write_staging_rejects_invalid_existing_staging_without_replacing_it(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    existing = tmp_path / "doc-1" / "canonical-v1-abcd.staging-existing"
    existing.mkdir(parents=True)
    marker = existing / "untrusted.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="existing staging bundle is invalid"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )

    assert marker.read_text("utf-8") == "preserve"
    assert list(existing.parent.glob("canonical-v1-abcd.staging-*")) == [existing]


def test_write_staging_rejects_multiple_existing_staging_directories(
    tmp_path: Path,
    canonical_document: CanonicalDocument,
) -> None:
    document_root = tmp_path / "doc-1"
    first = document_root / "canonical-v1-abcd.staging-first"
    second = document_root / "canonical-v1-abcd.staging-second"
    first.mkdir(parents=True)
    second.mkdir()

    with pytest.raises(ValueError, match="multiple staging directories"):
        CanonicalArtifactStore(tmp_path).write_staging(
            "doc-1", "canonical-v1-abcd", canonical_document
        )

    assert set(document_root.glob("canonical-v1-abcd.staging-*")) == {first, second}
