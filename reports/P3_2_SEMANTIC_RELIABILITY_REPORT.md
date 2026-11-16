# P3.2 Semantic Reliability Report

## Result

**P3.2: PARTIAL**

Evidence selection covered both reviewed documents for all three cases, and the semantic validator rejected every unsupported contradiction candidate produced for Cases A and C. Case B consistently remained inconclusive rather than treating the 40% estimate as verified. However, Qwen proposed a false contradiction in all six A/C runs, and the final A/C reports did not provide the expected `consistent`/`unsupported` classification. The safety gate held, but semantic correctness is not yet reliable enough to mark P3.2 PASSED.

## Runtime and model

- Local server: `llama.cpp HTTP API`, `http://127.0.0.1:8080`
- Health check: HTTP endpoint returned `{"status":"ok"}`.
- Model returned in inference metadata: `.\Models\Qwen2.5-3B-Instruct-GGUF\Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Runtime settings: context length 32768, 4 threads; request timeout 300 seconds, temperature 0, maximum output 1400 tokens.
- All nine primary runs returned `finish_reason=stop`; raw outputs parsed as JSON.
- Server was stopped after live verification.

The existing local runtime was started with:

```powershell
.\llama\llama-server.exe --model .\Models\Qwen2.5-3B-Instruct-GGUF\Qwen2.5-3B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8080 --ctx-size 32768 --threads 4
```

The nine-run stability batch used prompt `evidence-review-v2` and response schema version 2. A prompt/schema mismatch found during review was corrected afterward: `unsupported` is now explicitly allowed in the strict response schema and the schema version is 3. The version-3 Case A live probe completed successfully at the HTTP/JSON level but repeated the false contradiction proposal; the validator rejected it. The probe latency was 203825.2 ms. The primary 3×3 results below remain labeled version 2 and were not relabeled as version 3.

## Case results

| Case | Expected | Run 1 | Run 2 | Run 3 | Human verdict |
|---|---|---|---|---|---|
| A — MRD/PRD, planned 18 months | Both sources describe a planned 18-month period; do not infer completion. Final classification should be consistent or unsupported. | `findings_rejected`; false `contradiction_candidate` rejected; 115702.7 ms | `findings_rejected`; false `contradiction_candidate` rejected; 139299.5 ms | `findings_rejected`; false `contradiction_candidate` rejected; 99254.0 ms | Source excerpts show an MRD `T0 + 18 tháng` milestone and a PRD `18 tháng (mẫu + hồ sơ)` planned period. Neither proves the project actually completed. No false finding was accepted, but Qwen's repeated contradiction proposal and the lack of a final consistent/unsupported classification make this case a semantic miss. |
| B — MRD/TTR, approximately 40% savings | Treat 40% as a target/assumption pending BOM, quotations, and TCO; never as a verified result. | Accepted `inconclusive` / `unsupported_claim`; 104307.3 ms | Accepted `inconclusive` / `unsupported_claim`; 110701.9 ms | Accepted `inconclusive` / `unsupported_claim`; 89973.4 ms | Correctly avoids a verified-savings claim. Cited source passages describe the figure as a goal/assumption requiring equivalent-configuration price, BOM, and TCO validation. |
| C — PRD/TTR, 100 MHz and 100 m | Distinguish channel bandwidth from propagation distance; do not infer contradiction. | `findings_rejected`; false `contradiction_candidate` rejected; 169544.7 ms | `findings_rejected`; false `contradiction_candidate` rejected; 144470.4 ms | `findings_rejected`; false `contradiction_candidate` rejected; 86364.4 ms | PRD table 15 row 2 defines 100 MHz channel bandwidth and a 100 m trial condition; TTR paragraph 155 describes 100 m as an outdoor test scenario, not a BBU property. They are different dimensions. No false contradiction was accepted, but Qwen repeated the false proposal in all runs. |

Across each case's three runs, the selected excerpts were balanced at six blocks from each document and reported no missing quantity coverage. Case A selected evidence included MRD table 3 row 2 and PRD table 12 row 9; Case B included MRD paragraphs 133/154 and TTR paragraphs 120/164; Case C included PRD table 15 row 2 and TTR paragraph 155. Source excerpts, selector diagnostics, raw output, finish reason, latency, and validated report are retained in the corresponding run artifacts.

### Stability and validation

- Case A: all 3 raw model outputs proposed a contradiction; all 3 were rejected because the evidence did not establish a source-supported matching subject, quantity, timeframe, and conditions.
- Case B: all 3 outputs used `inconclusive`; all 3 reports accepted that cautious classification. None marked the economic estimate `consistent`.
- Case C: all 3 raw outputs proposed a contradiction; all 3 were rejected because the evidence did not establish comparable measured quantities and conditions.
- Across all nine runs, every raw evidence reference resolved to the supplied evidence catalog and to one of the two reviewed documents. Every accepted finding reference also resolved to an evidence record with a source location. No fabricated location or out-of-pair citation was detected.
- The validator therefore blocked false A/C findings, but raw model semantic accuracy remains inadequate for those cases.

The additional version-3 Case A probe also returned a schema-valid false contradiction proposal, which the validator rejected. This confirms the schema correction did not resolve Qwen's reasoning error.

## Regression and CLI verification

| Check | Result |
|---|---|
| `python -m pytest -q tests/documents/test_evidence_semantic_review.py tests/documents/test_cross_document_audit_manifest.py test_cross_document_audit.py tests/test_ingest_documents.py` | **59 passed** |
| `python -m compileall -q src/documents tests/documents` | **Passed** |
| `git diff --check` | **Passed**; Git reported the existing LF-to-CRLF warning for `src/documents/cross_document_audit.py`. |
| `python -m src.documents.cross_document_audit --manifest examples\cross_document_audit_manifest.json` | **Passed**; stdout parsed as JSON with 4 inputs and 2 parameter checks (`duration-18-months`, `frequency-n78`, both `present_in_all`). |
| Offline schema test | **Passed**; `unsupported` is included in schema version 3 and accepted by the report validator. |

## Changed and generated files

P3.2 source/test changes:

- `src/documents/evidence_semantic_review.py`
- `tests/documents/test_evidence_semantic_review.py`

Requested report:

- `reports/P3_2_SEMANTIC_RELIABILITY_REPORT.md`

Final live-run and CLI artifacts:

- `reports/p3_2_live_runs_verified/case_a_run_1.json`
- `reports/p3_2_live_runs_verified/case_a_run_2.json`
- `reports/p3_2_live_runs_verified/case_a_run_3.json`
- `reports/p3_2_live_runs_verified/case_b_run_1.json`
- `reports/p3_2_live_runs_verified/case_b_run_2.json`
- `reports/p3_2_live_runs_verified/case_b_run_3.json`
- `reports/p3_2_live_runs_verified/case_c_run_1.json`
- `reports/p3_2_live_runs_verified/case_c_run_2.json`
- `reports/p3_2_live_runs_verified/case_c_run_3.json`
- `reports/p3_2_live_runs_verified/case_a_schema3_probe.json`
- `reports/p3_2_live_runs_verified/manifest_cli.json`

Earlier failed-serialization artifacts in `reports/p3_2_live_runs/` were retained and were not used for the 3×3 assessment.

## Limitations and next step

- The model repeatedly generated false contradictions for two cases even though source evidence was present. The validator prevented those findings from being accepted, but it does not make the underlying inference correct.
- Excerpts are capped at 12 blocks, 10000 characters total, and 1800 characters per block. Coverage diagnostics describe selected reader output, not proof about omitted portions of source documents.
- The current source readers do not provide page numbers for DOCX paragraphs. Human review remains necessary.
- Numeric and local-context gates are deliberately conservative; they may reject legitimate text-only contradictions.
- The version-3 schema correction was covered by offline tests and one live probe, not a new 3×3 batch. That probe still failed semantic classification as described above.

The next evidence-based step is to improve the model's comparison prompt and post-review classification for matching planned values and distinct measurement dimensions, then rerun the same unchanged cases. Do not relax the validator or treat rejected proposals as findings.

## Git safety

The branch remained `main` tracking `origin/main`. The worktree had pre-existing uncommitted changes before this P3.2 work; they were retained. No commit, push, reset, checkout, or deletion was performed.
