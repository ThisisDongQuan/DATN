import json
from pathlib import Path

import pytest
from docx import Document

from src.documents.evidence_semantic_review import (
    ChatCompletion,
    LocalQwenCompletionClient,
    _select_evidence,
    _select_evidence_with_diagnostics,
    review_pair,
)


def create_docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(path)


def create_manifest(tmp_path: Path, left_text: str, right_text: str) -> Path:
    left = tmp_path / "left.docx"
    right = tmp_path / "right.docx"
    unrelated = tmp_path / "unrelated.docx"
    create_docx(left, left_text)
    create_docx(right, right_text)
    create_docx(unrelated, "Private unrelated content not sent to model.")
    manifest = {
        "mission": {"id": "test-mission", "description": "Test review"},
        "documents": [
            {"id": "mrd", "role": "MRD", "path": left.name, "type": "docx"},
            {"id": "prd", "role": "PRD", "path": right.name, "type": "docx"},
            {
                "id": "other",
                "role": "Unrelated",
                "path": unrelated.name,
                "type": "docx",
            },
        ],
        "checks": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def model_finding(
    *,
    status="contradiction_candidate",
    finding_type="direct_contradiction",
    refs=None,
    comparison_basis=None,
):
    if comparison_basis is None and status == "contradiction_candidate":
        comparison_basis = {
            "subject": "planned prototype",
            "quantity": "duration",
            "unit": "months",
            "timeframe": "baseline schedule",
            "conditions": "accepted scope",
        }
    return {
        "finding_type": finding_type,
        "severity": "medium",
        "status": status,
        "claim": "The project duration differs between the two supplied documents.",
        "evidence_refs": ["E001", "E002"] if refs is None else refs,
        "reasoning_summary": "The cited duration statements differ.",
        "confidence": "unrated",
        "comparison_basis": comparison_basis,
        "incompatibility_explanation": (
            "18 months and 12 months cannot both describe the same planned "
            "prototype duration under the baseline schedule."
            if status == "contradiction_candidate" else None
        ),
    }


def response(findings=None, limitations=None) -> str:
    return json.dumps({
        "findings": [] if findings is None else findings,
        "limitations": ["Model review requires human verification"]
        if limitations is None else limitations,
    })


class MockClient:
    def __init__(self, content=None, finish_reason="stop", error=None):
        self.content = content
        self.finish_reason = finish_reason
        self.error = error
        self.prompt = None

    def complete(self, prompt):
        self.prompt = prompt
        if self.error:
            raise self.error
        return ChatCompletion(
            content=self.content,
            finish_reason=self.finish_reason,
            model="mock-qwen",
        )


def review(tmp_path, client):
    manifest = create_manifest(
        tmp_path,
        "The planned prototype project duration is 18 months under the baseline schedule with the accepted scope.",
        "The planned prototype project duration is 12 months under the baseline schedule with the accepted scope.",
    )
    return review_pair(
        manifest,
        "mrd",
        "prd",
        "Compare the project duration.",
        client,
    )


def test_valid_findings_resolve_evidence_refs_to_source_metadata(tmp_path):
    client = MockClient(response([model_finding()]))

    report = review(tmp_path, client)

    assert report["status"] == "completed"
    assert report["outcome"] == "findings"
    finding = report["findings"][0]
    assert finding["finding_id"] == "F001"
    assert finding["status"] == "contradiction_candidate"
    assert finding["confidence"] == "unrated"
    assert [item["document_id"] for item in finding["evidence"]] == ["mrd", "prd"]
    assert all(Path(item["source_path"]).is_file() for item in finding["evidence"])
    assert [item["location"] for item in finding["evidence"]] == [
        "paragraph 1",
        "paragraph 1",
    ]
    assert "Private unrelated content" not in client.prompt


def test_model_cannot_publish_fabricated_evidence_reference(tmp_path):
    client = MockClient(response([model_finding(refs=["E001", "E999"])]))

    report = review(tmp_path, client)

    assert report["status"] == "completed"
    assert report["outcome"] == "findings_rejected"
    assert report["findings"] == []
    assert "not present in the input" in report["rejected_findings"][0]["reason"]


def test_model_cannot_inject_fabricated_source_location(tmp_path):
    finding = model_finding()
    finding["location"] = "paragraph 999, cell Z999"
    report = review(tmp_path, MockClient(response([finding])))

    assert report["status"] == "error"
    assert report["error"]["code"] == "invalid_schema"
    assert report["findings"] == []


def test_duplicate_evidence_references_are_rejected(tmp_path):
    report = review(
        tmp_path,
        MockClient(response([model_finding(refs=["E001", "E001"])])),
    )

    assert report["status"] == "completed"
    assert report["outcome"] == "findings_rejected"
    assert report["findings"] == []
    assert "Duplicate evidence references" in report["rejected_findings"][0]["reason"]


@pytest.mark.parametrize(
    ("content", "error_code"),
    [
        ("not json", "invalid_json"),
        ('{"findings": [], "limitations": [], "extra": true}', "invalid_schema"),
        ('{"findings": "none", "limitations": []}', "invalid_schema"),
        (
            json.dumps({
                "findings": [{
                    **model_finding(),
                    "unexpected": "field",
                }],
                "limitations": [],
            }),
            "invalid_schema",
        ),
    ],
)
def test_invalid_json_or_schema_is_an_error_not_no_findings(
    tmp_path, content, error_code
):
    report = review(tmp_path, MockClient(content))

    assert report["status"] == "error"
    assert report["outcome"] == "error"
    assert report["findings"] == []
    assert report["error"]["code"] == error_code


@pytest.mark.parametrize(
    ("exception", "error_code"),
    [
        (TimeoutError("timed out"), "timeout"),
        (
            OSError("connection refused"),
            "server_unavailable",
        ),
    ],
)
def test_timeout_or_server_offline_is_not_a_successful_empty_review(
    tmp_path, exception, error_code
):
    report = review(tmp_path, MockClient(error=exception))

    assert report["status"] == "error"
    assert report["outcome"] == "error"
    assert report["findings"] == []
    assert report["error"]["code"] == error_code


def test_response_with_truncated_finish_reason_is_an_error(tmp_path):
    report = review(
        tmp_path,
        MockClient(response([]), finish_reason="length"),
    )

    assert report["status"] == "error"
    assert report["error"]["code"] == "truncated_response"


def test_empty_response_is_an_error(tmp_path):
    report = review(tmp_path, MockClient(""))

    assert report["status"] == "error"
    assert report["error"]["code"] == "empty_response"


def test_contradiction_without_two_document_evidence_is_rejected(tmp_path):
    report = review(tmp_path, MockClient(response([
        model_finding(refs=["E001"]),
    ])))

    assert report["status"] == "completed"
    assert report["outcome"] == "findings_rejected"
    assert report["findings"] == []
    assert "two documents" in report["rejected_findings"][0]["reason"]


def test_assumed_cost_claim_cannot_be_marked_consistent(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "JEV cost reduction 40% is an assumption pending BOM, quotation and TCO.",
        "JEV cost reduction 40% is an assumption pending BOM, quotation and TCO.",
    )
    basis = {
        "subject": "JEV cost reduction",
        "quantity": "40%",
        "unit": "40%",
        "timeframe": "pending BOM",
        "conditions": "quotation and TCO",
    }
    finding = model_finding(
        status="consistent",
        finding_type="risk",
        comparison_basis=basis,
    )
    finding["claim"] = "The 40% cost reduction is verified."

    report = review_pair(
        manifest,
        "mrd",
        "prd",
        "Is the cost reduction verified?",
        MockClient(response([finding])),
    )

    assert report["status"] == "completed"
    assert report["outcome"] == "findings_rejected"
    assert report["findings"] == []
    assert "Consistency requires two-document evidence" in report["rejected_findings"][0]["reason"]


def test_missing_evidence_is_scoped_observation_not_grounded_finding(tmp_path):
    finding = model_finding(
        status="missing_evidence",
        finding_type="unsupported_claim",
        refs=[],
    )
    finding["claim"] = "No supporting value appears in the supplied excerpts."

    report = review(tmp_path, MockClient(response([finding])))

    assert report["status"] == "completed"
    assert report["outcome"] == "missing_evidence"
    assert report["findings"] == []
    assert report["evidence_gaps"][0]["status"] == "missing_evidence"
    assert report["evidence_gaps"][0]["evidence_refs"] == []
    assert "complete source documents" in " ".join(
        report["evidence_gaps"][0]["limitations"]
    )


def test_unsupported_is_available_in_response_schema_and_report(tmp_path):
    from src.documents.evidence_semantic_review import (
        RESPONSE_JSON_SCHEMA,
        RESPONSE_SCHEMA_VERSION,
    )

    allowed_statuses = RESPONSE_JSON_SCHEMA["properties"]["findings"]["items"][
        "properties"
    ]["status"]["enum"]
    finding = model_finding(
        status="unsupported",
        finding_type="unsupported_claim",
        refs=["E001"],
    )
    report = review(tmp_path, MockClient(response([finding])))

    assert RESPONSE_SCHEMA_VERSION == 3
    assert "unsupported" in allowed_statuses
    assert report["findings"][0]["status"] == "unsupported"


def test_successful_review_with_no_findings_is_distinct_from_error(tmp_path):
    report = review(tmp_path, MockClient(response([])))

    assert report["status"] == "completed"
    assert report["outcome"] == "no_findings"
    assert report["error"] is None
    assert report["findings"] == []


def test_consistent_requires_verbatim_comparable_basis_in_both_sources(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "The JEV prototype target is 18 months.",
        "The JEV prototype target is 18 months.",
    )
    basis = {
        "subject": "JEV prototype",
        "quantity": "18 months",
        "unit": "months",
        "timeframe": "18 months",
        "conditions": "target",
    }
    finding = model_finding(
        status="consistent",
        finding_type="direct_contradiction",
        comparison_basis=basis,
    )
    finding["claim"] = "The prototype target is consistent."

    report = review_pair(
        manifest,
        "mrd",
        "prd",
        "Compare the JEV prototype target duration.",
        MockClient(response([finding])),
    )

    assert report["status"] == "completed"
    assert report["outcome"] == "findings"
    assert report["findings"][0]["status"] == "consistent"


def test_evidence_selection_keeps_both_document_contexts_represented():
    evidence = [
        {
            "document_id": document_id,
            "source_path": f"{document_id}.docx",
            "location": f"paragraph {index}",
            "evidence_text": "The project duration follows the baseline schedule.",
        }
        for document_id in ("mrd", "prd")
        for index in range(10)
    ]

    selected = _select_evidence(evidence, "project duration baseline schedule")

    counts = {
        document_id: sum(item["document_id"] == document_id for item in selected)
        for document_id in ("mrd", "prd")
    }
    assert len(selected) == 12
    assert counts == {"mrd": 6, "prd": 6}


def test_evidence_selection_matches_equivalent_duration_phrasings_across_docs(
    tmp_path,
):
    manifest = create_manifest(
        tmp_path,
        "Nhiệm vụ nghiên cứu được triển khai trong 18 tháng; đây là kế hoạch, chưa phải kết quả hoàn thành.",
        "The prototype delivery milestone is scheduled at T0 + 18 months.",
    )
    from src.documents.cross_document_audit import load_manifest
    from src.documents.evidence_semantic_review import _document_evidence

    evidence = _document_evidence(
        load_manifest(manifest), ("mrd", "prd")
    )
    selected, diagnostics = _select_evidence_with_diagnostics(
        evidence, "Compare the project schedule and planned duration of 18 months."
    )

    assert {item["document_id"] for item in selected} == {"mrd", "prd"}
    assert all("18" in item["evidence_text"] for item in selected)
    assert diagnostics["missing_coverage"] == []
    assert any("chưa phải kết quả hoàn thành" in item["evidence_text"] for item in selected)
    json.dumps(diagnostics)


def test_missing_quantity_in_one_document_returns_scoped_evidence_gap(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "The project schedule specifies a planned duration of 18 months.",
        "The product requirements describe hardware interfaces but no schedule duration.",
    )
    client = MockClient(response([]))
    report = review_pair(
        manifest,
        "mrd",
        "prd",
        "Compare project schedule duration of 18 months.",
        client,
    )

    assert report["status"] == "completed"
    assert report["outcome"] == "missing_evidence"
    assert report["findings"] == []
    assert report["evidence_gaps"][0]["document_id"] == "prd"
    assert report["evidence_gaps"][0]["status"] == "missing_evidence"
    assert client.prompt is None


def test_same_number_in_different_contexts_does_not_replace_target_evidence(
    monkeypatch,
):
    from src.documents import evidence_semantic_review as review_module

    monkeypatch.setattr(review_module, "MAX_EVIDENCE_BLOCKS", 2)
    evidence = [
        {
            "document_id": "mrd",
            "source_path": "mrd.docx",
            "location": "paragraph 1",
            "evidence_text": "The project duration is 18 months under the delivery plan.",
        },
        {
            "document_id": "mrd",
            "source_path": "mrd.docx",
            "location": "paragraph 2",
            "evidence_text": "The product warranty is 18 months from acceptance.",
        },
        {
            "document_id": "prd",
            "source_path": "prd.docx",
            "location": "paragraph 1",
            "evidence_text": "The project delivery milestone is at T0 plus 18 months.",
        },
    ]

    selected = _select_evidence(evidence, "Compare project duration: 18 months.")

    assert selected[0]["location"] == "paragraph 1"
    assert not any(
        item["document_id"] == "mrd" and item["location"] == "paragraph 2"
        for item in selected
    )


def test_table_evidence_and_long_paragraph_excerpt_keep_source_metadata(tmp_path):
    left = tmp_path / "left.docx"
    right = tmp_path / "right.docx"
    manifest = create_manifest(tmp_path, "placeholder", "placeholder")
    doc = Document()
    doc.add_paragraph(("Unrelated detail " * 130) + " Project duration is 18 months.")
    table =     doc.add_table(rows=1, cols=1)
    doc.tables[0].cell(0, 0).text = (
        "Project prototype acceptance milestone is scheduled at T0 + 18 months."
    )
    doc.save(left)
    create_docx(right, "The project is planned for 18 months.")
    from src.documents.cross_document_audit import load_manifest
    from src.documents.evidence_semantic_review import _document_evidence

    evidence = _document_evidence(load_manifest(manifest), ("mrd", "prd"))
    selected, diagnostics = _select_evidence_with_diagnostics(
        evidence, "project prototype acceptance milestone schedule 18 months"
    )

    table_evidence = next(
        item for item in selected if item["location"] == "table 1, row 1"
    )
    assert "18 months" in table_evidence["evidence_text"]
    assert len(table_evidence["evidence_text"]) <= 1800
    assert table_evidence["source_path"].endswith("left.docx")
    assert diagnostics["missing_coverage"] == []


def test_no_matching_quantities_returns_missing_evidence_without_model_call(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "The project schedule is under review.",
        "The product interfaces are described.",
    )
    client = MockClient(response([]))
    report = review_pair(
        manifest,
        "mrd",
        "prd",
        "Compare the 18-month project schedule.",
        client,
    )

    assert report["outcome"] == "missing_evidence"
    assert report["findings"] == []
    assert len(report["evidence_gaps"]) == 2
    assert client.prompt is None


def test_evidence_selection_obeys_block_and_character_budgets(monkeypatch):
    from src.documents import evidence_semantic_review as review_module

    monkeypatch.setattr(review_module, "MAX_EVIDENCE_BLOCKS", 4)
    monkeypatch.setattr(review_module, "MAX_EVIDENCE_CHARS", 1700)
    evidence = [
        {
            "document_id": document_id,
            "source_path": f"{document_id}.docx",
            "location": f"paragraph {index}",
            "evidence_text": (
                f"The project duration is 18 months. "
                + ("Supporting detail " * 40)
            ),
        }
        for document_id in ("mrd", "prd")
        for index in range(10)
    ]

    selected, diagnostics = _select_evidence_with_diagnostics(
        evidence, "project duration 18 months"
    )

    assert len(selected) <= 4
    assert sum(
        len(item["document_id"]) + len(item["source_path"])
        + len(item["location"]) + len(item["evidence_text"])
        for item in selected
    ) <= 1700
    assert diagnostics["limits"]["max_evidence_blocks"] == 4
    assert diagnostics["limits"]["max_evidence_chars"] == 1700


def test_different_measurement_dimensions_cannot_be_a_contradiction(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "The BBU channel bandwidth is 100 MHz and propagation distance is 100 m under test configuration for an outdoor scenario.",
        "The BBU channel bandwidth is 100 MHz and propagation distance is 100 m under test configuration for an outdoor scenario.",
    )
    finding = model_finding(
        refs=["E001", "E002"],
        comparison_basis={
            "subject": "BBU",
            "quantity": "bandwidth",
            "unit": "MHz",
            "timeframe": "test configuration",
            "conditions": "outdoor scenario",
        },
    )
    finding["incompatibility_explanation"] = (
        "100 MHz and 100 m cannot both be true."
    )
    report = review_pair(
        manifest,
        "mrd",
        "prd",
        "Compare 100 MHz bandwidth and 100 m propagation distance.",
        MockClient(response([finding])),
    )

    assert report["status"] == "completed"
    assert report["outcome"] == "findings_rejected"
    assert report["findings"] == []
    assert "different values" in report["rejected_findings"][0]["reason"]


def test_contradiction_requires_explicit_explanation_and_different_values(
    tmp_path,
):
    manifest = create_manifest(
        tmp_path,
        "The planned prototype project duration is 18 months under the baseline schedule with the accepted scope.",
        "The planned prototype project duration is 12 months under the baseline schedule with the accepted scope.",
    )
    missing_explanation = model_finding()
    missing_explanation["incompatibility_explanation"] = None
    report = review_pair(
        manifest, "mrd", "prd", "Compare project duration.",
        MockClient(response([missing_explanation])),
    )

    assert report["outcome"] == "findings_rejected"
    assert "did not explain" in report["rejected_findings"][0]["reason"]

    equal_values = model_finding()
    equal_values["evidence_refs"] = ["E001", "E002"]
    equal_dir = tmp_path / "equal"
    equal_dir.mkdir()
    same_manifest = create_manifest(
        equal_dir,
        "The planned prototype project duration is 18 months under the baseline schedule with the accepted scope.",
        "The planned prototype project duration is 18 months under the baseline schedule with the accepted scope.",
    )
    equal_report = review_pair(
        same_manifest, "mrd", "prd", "Compare project duration.",
        MockClient(response([equal_values])),
    )
    assert equal_report["outcome"] == "findings_rejected"
    assert "different values" in equal_report["rejected_findings"][0]["reason"]


def test_contradiction_rejects_different_subjects_with_same_dimension(tmp_path):
    manifest = create_manifest(
        tmp_path,
        "The JEV project duration is 18 months under the delivery plan.",
        "The product warranty duration is 12 months from acceptance.",
    )
    finding = model_finding(
        comparison_basis={
            "subject": "duration",
            "quantity": "duration",
            "unit": "months",
            "timeframe": "duration",
            "conditions": "duration",
        }
    )
    finding["incompatibility_explanation"] = (
        "18 months and 12 months cannot both be true."
    )
    report = review_pair(
        manifest, "mrd", "prd", "Compare the JEV project duration.",
        MockClient(response([finding])),
    )

    assert report["outcome"] == "findings_rejected"
    assert "shared local subject context" in report["rejected_findings"][0]["reason"]


def test_local_completion_adapter_checks_finish_reason_and_json_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "qwen-local",
                "choices": [{
                    "message": {"content": '{"findings": [], "limitations": []}'},
                    "finish_reason": "stop",
                }],
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "src.documents.evidence_semantic_review.urlopen",
        fake_urlopen,
    )
    completion = LocalQwenCompletionClient().complete("review this evidence")
    request_payload = json.loads(captured["request"].data)

    assert completion.finish_reason == "stop"
    assert completion.model == "qwen-local"
    assert request_payload["stream"] is False
    assert request_payload["messages"][1]["content"] == "review this evidence"
    assert request_payload["response_format"]["type"] == "json_schema"
    assert request_payload["response_format"]["json_schema"]["strict"] is True
    limitations_schema = request_payload["response_format"]["json_schema"][
        "schema"
    ]["properties"]["limitations"]
    assert limitations_schema["items"]["type"] == "string"
    refs_schema = request_payload["response_format"]["json_schema"]["schema"][
        "properties"
    ]["findings"]["items"]["properties"]["evidence_refs"]
    assert refs_schema["uniqueItems"] is True
    explanation_schema = request_payload["response_format"]["json_schema"]["schema"][
        "properties"
    ]["findings"]["items"]["properties"]["incompatibility_explanation"]
    assert explanation_schema["anyOf"][1]["type"] == "string"
    finding_type_schema = request_payload["response_format"]["json_schema"][
        "schema"
    ]["properties"]["findings"]["items"]["properties"]["finding_type"]
    assert "scope_gap" not in finding_type_schema["enum"]
    assert captured["timeout"] > 0
