# Evidence-grounded local semantic review

The semantic review is an optional layer alongside the deterministic audit; it
does not replace the manifest checks, formula audit, or reconciliation.

Run one bounded comparison from the repository root:

```powershell
python -m src.documents.evidence_semantic_review `
  --manifest examples/cross_document_audit_manifest.json `
  --left mrd `
  --right prd `
  --question "Do the MRD and PRD state the same 18-month project duration?"
```

The command resolves both document IDs from the manifest, reads their
extractable DOCX/XLSX text, ranks blocks against the question, and sends at
most 12 blocks / 10,000 characters to the existing local llama.cpp chat
endpoint. It writes one JSON report to stdout and does not create report files.
The model/backend and an explicit completion status are included. No
repository-wide content is sent. Requests use llama.cpp's strict JSON Schema
response format when supported by the local server; application-side schema
and evidence-reference validation remains authoritative.

## Result contract

The report includes `status` (`completed` or `error`), `outcome`
(`findings`, `findings_rejected`, `missing_evidence`, `no_findings`, or
`error`), `model`, `backend`, `evidence_catalog`, `findings`,
`evidence_gaps`, `rejected_findings`, and `limitations`.

Each accepted finding has an assigned `finding_id`, `finding_type`, `severity`,
`status`, `claim`, `evidence_refs`, resolved `evidence` records,
`reasoning_summary`, `confidence`, and `limitations`. A cited reference must
match an evidence ID actually sent to the model. Contradiction candidates must
cite at least two documents. `consistent` additionally requires comparable
subject, quantity, unit, timeframe, and conditions copied from both sources.
Evidence gaps are reported separately from grounded findings and are scoped
only to the excerpt set supplied; they do not prove that the complete source
lacks information or that a requirement was violated. Confidence is
`unrated`, because no calibration method is implemented.

Malformed/truncated JSON, invalid schema, HTTP errors, timeouts, an unavailable
server, and failed document reads produce a report with `status: "error"`;
they are not reported as an empty successful review. Invalid or fabricated
evidence references reject the proposed finding.

## Limits

- The current DOCX reader gives paragraph and table/row locations, not page
  numbers. XLSX locations identify sheet/row or exact sheet/cell where
  available.
- Selection is relevance-ranked and bounded; the model sees only the returned
  `evidence_catalog`, not whole documents. Absence from that excerpt set is not
  evidence of absence from a complete document.
- Reference validation establishes traceability, not the correctness of model
  reasoning. All accepted semantic findings require human review.
- This is a single-pair vertical slice. It does not automatically compare all
  documents or replace the existing deterministic audit.
- Live verification is runtime- and model-dependent; check the local health
  endpoint and run representative reviews before relying on semantic results.
