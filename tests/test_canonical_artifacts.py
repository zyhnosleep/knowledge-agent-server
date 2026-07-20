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


def test_markdown_replaces_linked_structures_and_formats_heading_blocks(
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
    assert "RAW TABLE BLOCK SHOULD BE REPLACED" not in markdown
    assert markdown.count("| SAC-KG | 74.7 |") == 1


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
