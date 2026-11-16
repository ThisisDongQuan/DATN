# Cross-document audit manifest

Run the configurable audit from the repository root:

```powershell
python -m src.documents.cross_document_audit --manifest examples/cross_document_audit_manifest.json
```

Manifest document paths are resolved relative to the manifest file's directory,
not the current working directory. Absolute paths are also accepted. Supported
file types are `docx` and `xlsx`; the declared type must match the path
extension. Missing files, invalid manifests, duplicate IDs, unsupported types,
and unreadable documents stop the audit with an explicit error.

## Manifest shape

- `mission`: required `id` and `description`.
- `documents`: non-empty array of documents with unique `id`, `role`, `path`,
  and `type`.
- `checks`: optional array of checks with unique `id`, `label`, a non-empty
  list of Python regular-expression `patterns`, and document IDs to check.
- `reconciliation`: optional Excel reconciliation with `document_id`, `sheet`,
  and `cells`. The sheet defaults to `Tong hop`; cells default to `F29`, `H29`,
  and `D32`.

Each check reports whether a matching string was found in the selected
documents. A match does not establish semantic equivalence, correctness, or a
contradiction. Check results retain the legacy `status` field (`present_in_all`,
`incomplete_coverage`, or `not_found_in_any`) and add `evidence_status` plus
per-document `evidence` records. An evidence record includes `check_id`,
`document_id`, `source_path`, `location`, `evidence_text`, `observed_value`,
`check_method`, `status`, `reason`, and `limitations`. Unknown locations are
`null`; the checker does not infer a page or cell from text.

## Evidence and status interpretation

- `mentioned_unverified`: a regex matched text; context, scope, subject, and
  truth were not assessed.
- `missing_evidence`: no configured pattern was found in the blocks read. This
  is not proof that the requirement is absent or violated.
- `incomplete_coverage`: at least one selected document matched and at least
  one did not.
- `consistent`: selected values agree under the configured deterministic
  comparison only; this does not prove business correctness.
- `contradiction_candidate`: a selected deterministic comparison differs and
  warrants human investigation. Regex matches alone never generate it.
- `unsupported`: the formula evaluator cannot recalculate a supported input
  because the formula is outside its implemented subset.
- `error`: formula evaluation failed (for example, division by zero); this is
  distinct from an unsupported formula.
- `not_applicable` and `error` are reserved for future explicit check results;
  unreadable manifest inputs currently stop the run with an error instead of
  generating a partial success report.
- `evaluated` means a cell was read or recalculated, not that its meaning was
  validated.

For DOCX, available locations identify paragraphs or table/row; page numbers
are not extracted. XLSX text checks identify a sheet and row, while formula
checks identify exact sheet/cell. Keyword checks search only configured
documents and extracted text blocks. Formula auditing reports the worksheets
and formula cells scanned. Reconciliation reports the selected sheet and
cells; it does not inspect the rest of the workbook for consistency.

Formula records preserve the original formula, cached value, recalculated
value, evaluation status, and cache comparison status. A missing cache, an
unsupported formula, a formula evaluation error, and a mismatch are distinct.
The evaluator treats numeric zero as a usable value. Reconciliation does not
mark a comparison consistent if a selected value is blank, non-numeric,
unsupported, or errored. A mismatch is a candidate for investigation, not
proof that the workbook is wrong. Matching totals do not establish that the
wider budget, source data, or economic assumptions are correct; the evaluator
does not replace Excel.

## Legacy compatibility

The legacy `build_report(files)` and CLI retain their report keys,
parameter-check aggregate status (`present_in_all`, `incomplete_coverage`, or
`not_found_in_any`), and per-document status (`found` or `not_found`). Evidence
and explicit limitations are additive. Excel formula entries retain
`formula`, `cached_value`, and `calculated_value` and add evaluation/cache
metadata. Reconciliation retains its prior keys; its `status` now distinguishes
`unsupported` and `error` from `incomplete`, so consumers that branch on this
field should handle those additional values.

## Legacy CLI

The original four-document command remains available:

```powershell
python -m src.documents.cross_document_audit
python -m src.documents.cross_document_audit --mrd PATH --prd PATH --ttr PATH --budget PATH
```

The legacy mode retains its prior report keys and the existing per-document
options. Do not combine `--manifest` with legacy document options.
