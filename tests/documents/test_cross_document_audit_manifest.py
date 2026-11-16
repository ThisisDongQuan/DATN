import json
import sys
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

from src.documents import cross_document_audit as audit_module
from src.documents.cross_document_audit import (
    AUDIT_STATUSES,
    audit_budget_reconciliation,
    audit_workbook_formulas,
    build_manifest_report,
    build_report,
    main,
    read_xlsx,
    run_configured_checks,
)


def create_docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def write_manifest(path: Path, documents: list[dict], checks=None, **extra) -> Path:
    payload = {
        "mission": {"id": "test-mission", "description": "A test mission"},
        "documents": documents,
        "checks": checks or [],
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def document_config(document_id: str, path: Path, file_type: str = "docx") -> dict:
    return {
        "id": document_id,
        "role": f"role-{document_id}",
        "path": path.name,
        "type": file_type,
    }


def test_manifest_accepts_two_arbitrary_documents_and_scopes_checks(tmp_path):
    alpha = tmp_path / "alpha.docx"
    beta = tmp_path / "beta.docx"
    excluded = tmp_path / "unrelated.docx"
    create_docx(alpha, "The project has 18 months.")
    create_docx(beta, "The project schedule is pending.")
    create_docx(excluded, "The project has 18 months.")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [
            document_config("alpha", alpha),
            document_config("beta", beta),
            document_config("unrelated", excluded),
        ],
        checks=[{
            "id": "duration",
            "label": "Duration appears",
            "patterns": [r"\b18\s*months?\b"],
            "documents": ["alpha", "beta"],
        }],
    )

    report = build_manifest_report(manifest_path)

    assert report["mission"]["id"] == "test-mission"
    assert set(report["document_summary"]) == {"alpha", "beta", "unrelated"}
    check = report["parameter_checks"][0]
    assert [item["document"] for item in check["documents"]] == ["alpha", "beta"]
    assert [item["status"] for item in check["documents"]] == ["found", "not_found"]
    assert check["status"] == "incomplete_coverage"


def test_manifest_accepts_more_than_four_documents(tmp_path):
    documents = []
    for index in range(5):
        path = tmp_path / f"source-{index}.docx"
        create_docx(path, f"Source {index}")
        documents.append(document_config(f"source-{index}", path))

    report = build_manifest_report(
        write_manifest(tmp_path / "mission.json", documents)
    )

    assert len(report["document_summary"]) == 5


def test_manifest_rejects_duplicate_document_ids(tmp_path):
    first = tmp_path / "first.docx"
    second = tmp_path / "second.docx"
    create_docx(first, "one")
    create_docx(second, "two")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("duplicate", first), document_config("duplicate", second)],
    )

    with pytest.raises(ValueError, match="Duplicate document id"):
        build_manifest_report(manifest_path)


def test_manifest_rejects_missing_file(tmp_path):
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("missing", tmp_path / "missing.docx")],
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        build_manifest_report(manifest_path)


def test_manifest_rejects_unsupported_extension(tmp_path):
    unsupported = tmp_path / "source.txt"
    unsupported.write_text("text", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("unsupported", unsupported, "txt")],
    )

    with pytest.raises(ValueError, match="Unsupported document type"):
        build_manifest_report(manifest_path)


def test_manifest_reports_unreadable_document_instead_of_skipping_it(tmp_path):
    unreadable = tmp_path / "broken.docx"
    unreadable.write_bytes(b"not a Word document")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("broken", unreadable)],
    )

    with pytest.raises(RuntimeError, match="Could not read document 'broken'"):
        build_manifest_report(manifest_path)


def test_keyword_result_contains_traceable_evidence(tmp_path):
    source = tmp_path / "source.docx"
    create_docx(source, "The project is expected to last 18 months.")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("requirements", source)],
        checks=[{
            "id": "duration",
            "label": "Duration",
            "patterns": [r"18 months"],
            "documents": ["requirements"],
        }],
    )

    check = build_manifest_report(manifest_path)["parameter_checks"][0]
    evidence = check["documents"][0]["evidence"][0]

    assert evidence["check_id"] == "duration"
    assert evidence["document_id"] == "requirements"
    assert evidence["source_path"] == str(source.resolve())
    assert evidence["location"] == "paragraph 1"
    assert evidence["evidence_text"] == "The project is expected to last 18 months."
    assert evidence["check_method"] == "regex_keyword"
    assert evidence["status"] == "mentioned_unverified"


def test_evidence_keeps_unknown_location_empty(tmp_path):
    checks = {
        "duration": {
            "label": "Duration",
            "patterns": ["18 months"],
            "documents": ["requirements"],
        }
    }

    result = run_configured_checks(
        {"requirements": [{"text": "18 months"}]},
        checks,
        {"requirements": tmp_path / "requirements.docx"},
    )[0]

    assert result["documents"][0]["evidence"][0]["location"] is None


def test_xlsx_keyword_reader_reports_actual_sheet_and_row(tmp_path):
    source = tmp_path / "source.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Parameters"
    sheet["B3"] = "Bandwidth 100 MHz"
    workbook.save(source)
    workbook.close()

    blocks, _ = read_xlsx(source)

    assert blocks == [{
        "document": "source.xlsx",
        "location": "sheet Parameters, row 3",
        "text": "B3=Bandwidth 100 MHz",
    }]


def test_keyword_in_assumption_context_is_not_marked_verified(tmp_path):
    source = tmp_path / "economic-assumption.docx"
    create_docx(source, "The 40% cost reduction is an assumption pending validation.")
    check = build_manifest_report(
        write_manifest(
            tmp_path / "mission.json",
            [document_config("economics", source)],
            checks=[{
                "id": "cost-reduction",
                "label": "Cost reduction",
                "patterns": [r"40\s*%"],
                "documents": ["economics"],
            }],
        )
    )["parameter_checks"][0]

    evidence = check["documents"][0]["evidence"][0]
    assert evidence["status"] == "mentioned_unverified"
    assert "assumption pending validation" in evidence["evidence_text"]
    assert evidence["observed_value"] is None
    assert "semantic context" in evidence["reason"]


def test_absent_keyword_means_missing_evidence_not_requirement_failure(tmp_path):
    source = tmp_path / "requirements.docx"
    create_docx(source, "No duration is stated in this paragraph.")
    check = build_manifest_report(
        write_manifest(
            tmp_path / "mission.json",
            [document_config("requirements", source)],
            checks=[{
                "id": "duration",
                "label": "Duration",
                "patterns": [r"18 months"],
                "documents": ["requirements"],
            }],
        )
    )["parameter_checks"][0]

    assert check["status"] == "not_found_in_any"
    assert check["evidence_status"] == "missing_evidence"
    assert check["documents"][0]["evidence"][0]["status"] == "missing_evidence"
    assert "does not establish" in check["documents"][0]["evidence"][0]["reason"]


def test_different_subject_mentions_are_not_classified_as_contradiction(tmp_path):
    vendor = tmp_path / "vendor.docx"
    prototype = tmp_path / "prototype.docx"
    create_docx(vendor, "Reference radio from vendor: 32T32R.")
    create_docx(prototype, "JEV prototype configuration: 4T4R.")
    report = build_manifest_report(
        write_manifest(
            tmp_path / "mission.json",
            [
                document_config("vendor-reference", vendor),
                document_config("jev-prototype", prototype),
            ],
            checks=[
                {
                    "id": "vendor-radio",
                    "label": "Vendor reference radio",
                    "patterns": [r"32T32R"],
                    "documents": ["vendor-reference"],
                },
                {
                    "id": "jev-radio",
                    "label": "JEV prototype radio",
                    "patterns": [r"4T4R"],
                    "documents": ["jev-prototype"],
                },
            ],
        )
    )

    assert all(
        check["evidence_status"] == "mentioned_unverified"
        for check in report["parameter_checks"]
    )
    assert all(
        evidence["status"] != "contradiction_candidate"
        for check in report["parameter_checks"]
        for document in check["documents"]
        for evidence in document["evidence"]
    )


def test_manifest_can_configure_excel_reconciliation(tmp_path):
    workbook_path = tmp_path / "summary.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Totals"
    sheet["A1"] = 25
    sheet["B1"] = 25
    workbook.save(workbook_path)
    workbook.close()
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("summary", workbook_path, "xlsx")],
        reconciliation={
            "document_id": "summary",
            "sheet": "Totals",
            "cells": ["A1", "B1"],
        },
    )

    report = build_manifest_report(manifest_path)

    assert report["workbooks"]["summary"]["reconciliation"]["status"] == "consistent"


def test_unsupported_excel_formula_is_not_reported_as_workbook_error(tmp_path):
    workbook_path = tmp_path / "unsupported.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Totals"
    sheet["A1"] = '=CONCATENATE("A","B")'
    sheet["A2"] = 10
    workbook.save(workbook_path)
    workbook.close()

    formula_audit = audit_workbook_formulas(workbook_path, "summary")
    formula = formula_audit["formulas"][0]
    reconciliation = audit_budget_reconciliation(
        workbook_path,
        "Totals",
        ("A1", "A2"),
        "summary",
    )

    assert formula["formula"] == '=CONCATENATE("A","B")'
    assert formula["cached_value"] is None
    assert formula["calculated_value"] is None
    assert formula["evaluation_status"] == "unsupported"
    assert formula["cache_status"] == "unverifiable"
    assert formula["evidence"]["status"] == "unsupported"
    assert formula["evidence"]["location"] == "sheet Totals, cell A1"
    assert reconciliation["status"] == "unsupported"
    assert reconciliation["evidence_status"] == "unsupported"
    assert reconciliation["checks"][0]["formula"] == formula["formula"]
    assert reconciliation["checks"][0]["cached_value"] is None
    assert reconciliation["checks"][0]["evaluation_status"] == "unsupported"


def test_zero_values_are_valid_reconciliation_evidence(tmp_path):
    workbook_path = tmp_path / "zeros.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Totals"
    sheet["A1"] = 0
    sheet["A2"] = 0
    workbook.save(workbook_path)
    workbook.close()

    report = audit_budget_reconciliation(
        workbook_path, "Totals", ("A1", "A2"), "zero-test"
    )

    assert report["status"] == "consistent"
    assert report["evidence_status"] == "consistent"
    assert [item["calculated_value"] for item in report["checks"]] == [0, 0]
    assert all(item["evidence"]["status"] == "not_applicable" for item in report["checks"])


def test_cached_zero_matches_recalculated_zero(tmp_path, monkeypatch):
    workbook_path = tmp_path / "zero-formula.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = "=1-1"
    workbook.save(workbook_path)
    workbook.close()

    original_load_workbook = audit_module.load_workbook

    def load_with_zero_formula_cache(path, *args, **kwargs):
        loaded = original_load_workbook(path, *args, **kwargs)
        if kwargs.get("data_only"):
            loaded.active["A1"] = 0
        return loaded

    monkeypatch.setattr(audit_module, "load_workbook", load_with_zero_formula_cache)
    formula = audit_workbook_formulas(workbook_path)["formulas"][0]

    assert formula["formula"] == "=1-1"
    assert formula["cached_value"] == 0
    assert formula["calculated_value"] == 0
    assert formula["evaluation_status"] == "evaluated"
    assert formula["cache_status"] == "matches"
    assert formula["evidence"]["status"] == "consistent"


def test_empty_cached_value_is_not_treated_as_missing_or_matching_zero(
    tmp_path, monkeypatch
):
    workbook_path = tmp_path / "empty-formula-cache.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "=1-1"
    workbook.active["A2"] = 0
    workbook.save(workbook_path)
    workbook.close()

    original_load_workbook = audit_module.load_workbook

    def load_with_empty_formula_cache(path, *args, **kwargs):
        loaded = original_load_workbook(path, *args, **kwargs)
        if kwargs.get("data_only"):
            loaded.active["A1"] = ""
        return loaded

    monkeypatch.setattr(audit_module, "load_workbook", load_with_empty_formula_cache)
    formula_audit = audit_workbook_formulas(workbook_path)
    formula = formula_audit["formulas"][0]
    reconciliation = audit_budget_reconciliation(
        workbook_path, "Sheet", ("A1", "A2")
    )

    assert formula["cached_value"] == ""
    assert formula["calculated_value"] == 0
    assert formula["cache_status"] == "differs"
    assert formula["evidence"]["status"] == "contradiction_candidate"
    assert reconciliation["status"] == "inconsistent"
    assert reconciliation["cache_mismatches"] == [{
        "cell": "A1",
        "cached_value": "",
        "calculated_value": 0,
    }]


def test_formula_evaluation_error_is_distinct_from_unsupported(tmp_path):
    workbook_path = tmp_path / "formula-error.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "=1/0"
    workbook.save(workbook_path)
    workbook.close()

    formula = audit_workbook_formulas(workbook_path)["formulas"][0]

    assert formula["evaluation_status"] == "error"
    assert formula["cache_status"] == "unverifiable"
    assert formula["evidence"]["status"] == "error"
    assert "ZeroDivisionError" in formula["evidence"]["reason"]


def test_unsupported_sum_reference_is_not_reported_as_zero(tmp_path):
    workbook_path = tmp_path / "unsupported-sum.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "=SUM(UnknownRange)"
    workbook.save(workbook_path)
    workbook.close()

    formula = audit_workbook_formulas(workbook_path)["formulas"][0]

    assert formula["calculated_value"] is None
    assert formula["evaluation_status"] == "unsupported"
    assert formula["evidence"]["status"] == "unsupported"


def test_formula_with_blank_input_is_missing_evidence(tmp_path):
    workbook_path = tmp_path / "blank-input.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "=A2+1"
    workbook.save(workbook_path)
    workbook.close()

    formula = audit_workbook_formulas(workbook_path)["formulas"][0]

    assert formula["evaluation_status"] == "missing_evidence"
    assert formula["evidence"]["status"] == "missing_evidence"
    assert "A2" in formula["evidence"]["reason"]


def test_blank_reconciliation_cell_is_not_consistent(tmp_path):
    workbook_path = tmp_path / "blank-cell.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Totals"
    sheet["A1"] = 0
    sheet["A2"] = None
    workbook.save(workbook_path)
    workbook.close()

    report = audit_budget_reconciliation(
        workbook_path, "Totals", ("A1", "A2")
    )

    assert report["status"] == "incomplete"
    assert report["evidence_status"] == "missing_evidence"
    assert report["checks"][1]["evidence"]["status"] == "missing_evidence"


def test_invalid_document_does_not_produce_success_shaped_cli_report(
    tmp_path, monkeypatch, capsys
):
    unreadable = tmp_path / "broken.docx"
    unreadable.write_bytes(b"not a docx")
    manifest_path = write_manifest(
        tmp_path / "mission.json",
        [document_config("broken", unreadable)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["cross_document_audit", "--manifest", str(manifest_path)],
    )

    with pytest.raises(SystemExit, match="Could not read document 'broken'"):
        main()
    assert capsys.readouterr().out == ""


def test_legacy_cli_keeps_legacy_report_schema(tmp_path, monkeypatch, capsys):
    mrd = tmp_path / "mrd.docx"
    prd = tmp_path / "prd.docx"
    ttr = tmp_path / "ttr.docx"
    budget = tmp_path / "budget.xlsx"
    for path in (mrd, prd, ttr):
        create_docx(path, "The project has 18 months.")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Tong hop"
    for cell in ("F29", "H29", "D32"):
        sheet[cell] = 100
    workbook.save(budget)
    workbook.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_document_audit",
            "--mrd", str(mrd),
            "--prd", str(prd),
            "--ttr", str(ttr),
            "--budget", str(budget),
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)

    assert report["mode"] == "read-only; no output files"
    assert set(report["document_summary"]) == {"MRD", "PRD", "TTR", "BUDGET"}
    assert report["workbook"]["budget_reconciliation"]["status"] == "consistent"
    assert "parameter_checks" in report
    assert {
        "engine",
        "mode",
        "inputs",
        "document_summary",
        "budget_total_candidates",
        "workbook",
        "limitations",
    } <= set(report)
    assert "found" in {
        document["status"]
        for check in report["parameter_checks"]
        for document in check["documents"]
    }
    assert report["workbook"]["formula_audit"]["formula_count"] == 0
    assert "consistent" in AUDIT_STATUSES

    api_report = build_report({
        "MRD": mrd,
        "PRD": prd,
        "TTR": ttr,
        "BUDGET": budget,
    })
    assert set(api_report["document_summary"]) == {"MRD", "PRD", "TTR", "BUDGET"}
    assert all(
        isinstance(summary["block_count"], int)
        and isinstance(summary["empty"], bool)
        for summary in api_report["document_summary"].values()
    )
    assert all(
        check["status"] in {
            "present_in_all",
            "incomplete_coverage",
            "not_found_in_any",
        }
        for check in api_report["parameter_checks"]
    )
    assert all(
        document["status"] in {"found", "not_found"}
        for check in api_report["parameter_checks"]
        for document in check["documents"]
    )
