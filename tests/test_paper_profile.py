from app.models.records import Document
from app.services.paper_profile import alias_in_text, ensure_paper_profile, paper_profile_retrieval_terms, source_fields_for_document


def test_paper_profile_contains_routing_summary_aliases_and_tables() -> None:
    document = Document(
        id="d1",
        project_id="p1",
        title="OPLS5 Force Field Development and Validation",
        file_name="opls5.pdf",
        sha256="abc",
        raw_path="raw/opls5.pdf",
        raw_text="The OPLS5 force field improves torsion and charge parameters for ligand binding.",
        metadata_json={
            "document_intelligence": {
                "tables": [
                    {
                        "page_label": "12",
                        "markdown": "Table 7: OPLS5 and OPLS4 binding RMSE.\n| Model | RMSE |\n| --- | --- |\n| OPLS5 | 0.61 |",
                    }
                ],
                "figures": [{"page_label": "3", "note": "Figure 1 shows the parameterization workflow."}],
            }
        },
        status="ready",
    )

    profile = ensure_paper_profile(document)

    assert profile["profile_version"]
    assert profile["title"] == "OPLS5 Force Field Development and Validation"
    assert "OPLS5" in profile["aliases"]
    assert "Force" not in profile["aliases"]
    assert "Field" not in profile["aliases"]
    assert "Table 7" in profile["key_terms"]
    assert "Figure 1" in profile["key_terms"]
    assert "OPLS5" in profile["routing_summary"]
    assert "Table 7" in profile["routing_summary"]


def test_paper_profile_refreshes_legacy_profile_without_version() -> None:
    document = Document(
        id="d1",
        project_id="p1",
        title="CHARMM36m protein force field",
        file_name="charmm36m.pdf",
        sha256="abc",
        raw_path="raw/charmm36m.pdf",
        raw_text="CHARMM36m improves IDP sampling.",
        metadata_json={"paper_profile": {"routing_summary": "old"}},
        status="ready",
    )

    profile = ensure_paper_profile(document)

    assert profile["profile_version"]
    assert profile["routing_summary"] != "old"
    assert "CHARMM36m" in profile["aliases"]


def test_paper_profile_retrieval_terms_include_late_scientific_terms() -> None:
    noisy_front_matter = " ".join(f"AuthorName{i} Department University Page" for i in range(300))
    atom_type_noise = " ".join(["N-CX-2C-2C", "C-CX-3C-CT", "CX-2C-CA-CA"] * 80)
    image_path_noise = " ".join(f"images/{i:064x}" for i in range(80))
    document = Document(
        id="d1",
        project_id="p1",
        title="ff14SB parameter update",
        file_name="ff14sb.pdf",
        sha256="abc",
        raw_path="raw/ff14sb.pdf",
        raw_text=(
            noisy_front_matter
            + "\n\n"
            + atom_type_noise
            + "\n\n"
            + image_path_noise
            + "\n\nThe fitting protocol later uses GAlib and QM-MM target data for side-chain torsions. "
            + "A separate section discusses CMAP and a 500 K Boltzmann population fit. "
            + "Mechanism notes mention covalent relaxation, steric clashes, explicit hydrogen, "
            + "charge transfer, 34 organic liquids, and \\chi_1 validation."
        ),
        metadata_json={
            "paper_profile": {
                "profile_version": "paper-profile-v1",
                "routing_summary": "old noisy profile",
                "key_terms": ["Page", "Department", "University"],
            }
        },
        status="ready",
    )

    terms = paper_profile_retrieval_terms(document)

    assert "QM-MM" in terms
    assert "GAlib" in terms
    assert "CMAP" in terms
    assert "500 K" in terms
    assert "fitting" in terms[:32]
    assert "protocol" in terms[:32]
    assert "QM-MM" in terms[:32]
    assert "GAlib" in terms[:32]
    assert "covalent relaxation" in terms
    assert "steric clashes" in terms
    assert "explicit hydrogen" in terms
    assert "charge transfer" in terms
    assert "34 organic liquids" in terms
    assert "\u03c71" in terms
    assert "N-CX-2C-2C" not in terms[:32]
    assert not any(str(term).startswith("images/") for term in terms)


def test_alias_in_text_uses_boundaries_for_near_names() -> None:
    assert alias_in_text("OPLS4", "OPLS4 improves condensed phase validation.")
    assert alias_in_text("OPLS4", "OPLS4/OPLS5 comparison.")
    assert not alias_in_text("OPLS4", "OPLS5 is compared against older methods.")
    assert alias_in_text("CHARMM36m", "The CHARMM36m correction is discussed.")
    assert not alias_in_text("CHARMM36", "The CHARMM36m correction is discussed.")
    assert alias_in_text("ff99SB-disp", "The ff99SB-disp model improves IDP ensembles.")
    assert not alias_in_text("ff99SB", "The ff99SB-disp model improves IDP ensembles.")


def test_force_field_source_identity_uses_canonical_benchmark_slugs() -> None:
    cases = [
        ("charmm36.pdf", "CHARMM36 force field", "sources/charmm36-force-field-refinement-for-proteins"),
        ("charmm36idpsff.pdf", "CHARMM36IDPSFF", "sources/charmm36idpsff"),
        ("charmm36m.pdf", "CHARMM36m protein force field", "sources/charmm36m-force-field"),
        ("ff14sb.pdf", "ff14SB parameter update", "sources/ff14sb"),
        ("ff19sb.pdf", "ff19SB protein backbone parameters", "sources/ff19sb-amino-acid-specific-protein-backbone-parameters"),
        ("ff99sb-disp.pdf", "ff99SB-disp force field", "sources/ff99sb-disp"),
        ("tip4p-d.pdf", "TIP4P-D water model", "sources/ff99sb-disp"),
        ("ff99sb-ildn.pdf", "ff99SB-ILDN force field", "sources/ff99sb-ildn"),
        ("opls-aa.pdf", "OPLS-AA force field development", "sources/opls-aa-force-field-development-and-validation"),
        ("opls4.pdf", "OPLS4 force field development", "sources/opls4-force-field-development-and-validation"),
        ("opls5.pdf", "OPLS5 force field development", "sources/opls5-force-field-development-and-validation"),
    ]

    for file_name, title, expected_slug in cases:
        document = Document(
            id=file_name,
            project_id="p1",
            title=title,
            file_name=file_name,
            sha256=file_name,
            raw_path=f"raw/{file_name}",
            raw_text=f"{title} benchmark paper.",
            metadata_json={"source_slug": f"sources/{file_name.removesuffix('.pdf')}"},
            status="ready",
        )

        profile = ensure_paper_profile(document)
        fields = source_fields_for_document(document)

        assert profile["source_slug"] == expected_slug
        assert fields["page_slug"] == expected_slug


def test_unknown_source_identity_keeps_generic_slug_behavior() -> None:
    document = Document(
        id="unknown",
        project_id="p1",
        title="Novel Simulation Method",
        file_name="novel-method.pdf",
        sha256="abc",
        raw_path="raw/novel-method.pdf",
        raw_text="A paper outside the force-field canonical alias list.",
        metadata_json={},
        status="ready",
    )

    fields = source_fields_for_document(document)

    assert fields["page_slug"] == "sources/novel-simulation-method"


def test_canonical_source_identity_does_not_match_body_mentions_only() -> None:
    document = Document(
        id="comparison",
        project_id="p1",
        title="Independent Comparison Study",
        file_name="comparison-study.pdf",
        sha256="abc",
        raw_path="raw/comparison-study.pdf",
        raw_text=(
            "This paper compares OPLS5 force field development and validation with "
            "other baselines, but it is not the OPLS5 paper itself."
        ),
        metadata_json={},
        status="ready",
    )

    fields = source_fields_for_document(document)

    assert fields["page_slug"] == "sources/independent-comparison-study"


def test_canonical_source_identity_does_not_match_profile_alias_mentions_only() -> None:
    document = Document(
        id="comparison",
        project_id="p1",
        title="Independent Comparison Study",
        file_name="comparison-study.pdf",
        sha256="abc",
        raw_path="raw/comparison-study.pdf",
        raw_text="This is a comparison paper, not the OPLS5 paper itself.",
        metadata_json={
            "paper_profile": {
                "title": "Independent Comparison Study",
                "aliases": ["OPLS5"],
            }
        },
        status="ready",
    )

    fields = source_fields_for_document(document)

    assert fields["page_slug"] == "sources/independent-comparison-study"
