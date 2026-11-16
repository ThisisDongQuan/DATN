# JEV Local AI P3.1 Live Verification Report

**Result: PARTIAL**

Runtime and structured-output requests were observed to work, but the three semantic cases did not all produce correct, adequately supported accepted results. In particular, the model proposed a semantically false contradiction for Case C; the evidence validation gate rejected it.

## 1. Runtime environment

- Backend: llama.cpp HTTP API.
- Model: `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Model path recorded by the server: `Models/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Recorded llama.cpp build: `0.6.0-dev`, build 11559, commit `f2cc7282c`.
- Recorded context length: 32,768 tokens; CPU threads: 4.
- Endpoint: `http://127.0.0.1:8080/v1/chat/completions`.
- Health endpoint was queried during finalization and returned HTTP 200 with `{"status":"ok"}`. No model was downloaded and no runtime configuration was changed.

The existing runtime artifacts are dated 2026-11-13; the health check above was repeated during this finalization.

## 2. Health and smoke tests

The existing `runtime_smoke.json` records an actual HTTP 200 request to the chat endpoint, `finish_reason: stop`, and parsed JSON:

```json
{"status":"ok","message":"short confirmation"}
```

Recorded latency was 1,543.1 ms. `runtime_smoke_retry.json` records the same parsed response with HTTP 200 and 2,040.1 ms latency.

The `structured_output_probe.json` artifact records a strict JSON-schema request, HTTP 200, `finish_reason: stop`, and parsed content `{"findings": [], "limitations": []}`. Its basic schema check passed; latency was 4,728.2 ms.

These are evidence of runtime and response-format success, not semantic correctness.

## 3. Semantic cases

Human verdicts below were assigned after checking the referenced passages in the current repository DOCX documents.

| Case | Expected | Actual | Evidence validity | Human verdict | Latency |
|---|---|---|---|---|---:|
| A — MRD / PRD, 18-month plan | Both describe a planned 18-month task/sample period; this does not prove that the project completed. | The saved primary run ended `findings_rejected`. The raw model proposed a contradiction, but cited two MRD excerpts rather than evidence from both MRD and PRD. Other retries were also rejected; the first run returned `invalid_schema`. | Cited IDs in report artifacts resolve to real catalog records and source locations, but the selected set omitted the relevant PRD duration statements, so the case lacks cross-document coverage. | The documents do describe a planned 18-month period: MRD paragraph 11 and PRD paragraph 203 / table 12, row 9. The inference itself did not establish this because the needed PRD evidence was not sent. No completion claim is supported. | 123,942.5 ms (`case_a_final`); other attempts: 40,181.1–163,360.5 ms |
| B — MRD / TTR, approximately 40% cost reduction | The figure is an assumption/target pending BOM, quotations, and TCO; it is not a verified saving. | `case_b_final` and `case_b_unicode` produced an accepted `unsupported_claim` / `mentioned_unverified` finding. `case_b_retry` was rejected for duplicate references; the first run returned `invalid_schema`. | The accepted findings' references are unique, present in the corresponding catalog, belong to MRD/TTR, and resolve to source passages. | Correct for the accepted runs: the passages explicitly describe 40% as an assumption or target pending verification, not an achieved result. The inconsistent retries mean this is not reproducible across every attempt. | 159,099.0 ms (`case_b_final`); Unicode run: 147,458.2 ms |
| C — PRD / TTR, 100 MHz and 100 m | 100 MHz is channel bandwidth; 100 m is propagation distance/test scenario. Different dimensions do not by themselves establish contradiction. | The final raw model output called them mutually exclusive and proposed `contradiction_candidate`. The report rejected it because the comparison basis was unsupported. Other retries were rejected for insufficient two-document evidence. | References in the final report resolve to real PRD/TTR catalog entries and source locations. No finding was accepted. The final model proposal was semantically wrong despite valid JSON and citations. | The model's contradiction claim is false. PRD table 15, row 2 describes the 100 MHz configuration and a conditional 100 m trial; TTR paragraph 155 identifies 100 m as a scenario condition, not a BBU property. The validation gate correctly prevented publication of the unsupported contradiction. | 152,811.6 ms (`case_c_final`); other attempts: 47,522.7–176,018.1 ms |

### Case evidence and validation details

- Across case artifacts with an evidence catalog, the catalog excerpts and locations were re-read from their source DOCX files; no catalog/source mismatches were found.
- Published Case B references were checked against that case's catalog: all referenced IDs existed, were unique, belonged to the compared pair, and resolved to the same evidence records included in the finding.
- Case A's rejection was appropriate: its candidate relied on evidence from only one document. The principal issue was evidence selection, not a fabricated citation.
- Case B's duplicate-reference retry and invalid-schema initial run were not converted to successful findings.
- Case C demonstrates that valid schema and source citations do not guarantee sound reasoning. Its false contradiction was rejected by the comparison-basis validator and did not enter `findings`.
- No model-authored location was accepted. Locations in reports came from the input catalog.

## 4. Semantic accuracy assessment

The expected source-level conclusions are:

1. **Case A:** The planned task is framed as 18 months in both documents. Neither document excerpt proves actual completion. The live review did not achieve a valid cross-document comparison because the selected PRD excerpts did not include its relevant 18-month statements.
2. **Case B:** The approximately 40% cost figure remains unverified pending BOM, quotations, and TCO. The accepted `mentioned_unverified` runs agree with the source; rejected retries show unstable output.
3. **Case C:** 100 MHz and 100 m refer to different test dimensions. The model's contradiction proposal was semantically wrong, but the gate rejected it.

Overall semantic correctness is therefore **partial**, not passed. The review requires human verification, and the saved attempts show sensitivity to evidence selection and model output variability.

## 5. Final regression and CLI verification

Commands run during finalization:

- `python -m pytest -q tests/documents/test_evidence_semantic_review.py tests/documents/test_cross_document_audit_manifest.py test_cross_document_audit.py tests/test_ingest_documents.py`
  - **49 passed**.
- `python -m compileall -q src/documents tests/documents`
  - Passed.
- `git diff --check`
  - Passed. Git emitted only the existing LF-to-CRLF working-copy warning for `src/documents/cross_document_audit.py`.
- Manifest CLI:
  - `python -m src.documents.cross_document_audit --manifest examples/cross_document_audit_manifest.json`
  - Exit code 0; stdout parsed as JSON.
  - Four real inputs were present and non-empty: MRD (234 blocks), PRD (613), TTR (305), and budget (151). Two configured parameter checks were returned.

## 6. Errors observed and handling

- The initial Case A/B/C attempts (`case_a.json`, `case_b.json`, `case_c.json`) returned `invalid_schema`: model `limitations` contained objects rather than non-empty strings. They are recorded as errors, not empty successful reviews.
- Retry artifacts contain rejected proposals for one-document contradictions or duplicate evidence references. They were not published as findings.
- Case C's final model response was schema-valid but semantically incorrect; the evidence-basis validator rejected it.
- Case A's saved runs did not select the relevant PRD duration excerpts. The current evidence-selection behavior was left unchanged in this finalization because correcting it would alter the semantic-review implementation beyond report closure.
- No source/runtime code was changed as part of this finalization.

## 7. Limitations

- Semantic inference is variable across retries; Case B's correct result did not appear in every attempt.
- Case A demonstrates that relevance selection can omit needed evidence even when the source document contains it.
- Case C demonstrates that the model can make an incorrect semantic claim from valid citations; the gate prevents some unsupported claims but does not prove accepted reasoning is correct.
- Latencies for the three-document-pair reviews were high (about 40–176 seconds across recorded attempts).
- Findings remain model-assisted, scoped to selected excerpts, and require human review.

## 8. Git status and files

This finalization added only this report; no source or test code was edited, and no commit or push was made.

The worktree remains dirty on branch `main`, tracking `origin/main`, with pre-existing changes preserved:

- Modified: `src/core/test_validator.py`, `src/documents/cross_document_audit.py`.
- Untracked prior artifacts/files: `audit_detail.txt`, `examples/cross_document_audit_manifest.json`, `jev_audit_utf8.json`, `legacy_cli_output.json`, `llama.rar`, `manifest_cli_output.json`, the contents of `reports/p3_1_live_verification_20261113/`, `src/documents/AUDIT_MANIFEST.md`, `src/documents/SEMANTIC_REVIEW.md`, `src/documents/cross_document_audit.py.bak`, `src/documents/evidence_semantic_review.py`, `test_cross_document_audit.py`, `test_cross_document_audit.py.bak`, `tests/documents/test_cross_document_audit_manifest.py`, `tests/documents/test_evidence_semantic_review.py`, and `workbook_inspection.txt`.
- Added by this finalization: `reports/P3_1_LIVE_VERIFICATION_REPORT.md`.

No existing worktree files were deleted or overwritten.
