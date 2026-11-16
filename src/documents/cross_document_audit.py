from __future__ import annotations

import argparse
import ast
import json
import operator
import re
import sys
from pathlib import Path
from typing import Any

from docx import Document
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries


def _project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "src" / "documents").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def _find_input(root: Path, filename: str) -> Path:
    for folder in ("data/document", "data/documents"):
        candidate = root / folder / filename
        if candidate.exists():
            return candidate
    matches = list(root.glob(f"data/*/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Không tìm thấy file '{filename}' trong data/document hoặc data/documents")


def default_files(root: Path | None = None) -> dict[str, Path]:
    root = root or _project_root()
    return {
        "MRD": _find_input(root, "BM.01.01.MRD.NV_2023.docx"),
        "PRD": _find_input(root, "BM.01.02.PRD.NV_2023.docx"),
        "TTR": _find_input(root, "BM.01.03.TTR.NV_2023.docx"),
        "BUDGET": _find_input(root, "BM.01_Dự toán.xlsx"),
    }


DEFAULT_FILES = {
    "MRD": Path("data/documents/BM.01.01.MRD.NV_2023.docx"),
    "PRD": Path("data/documents/BM.01.02.PRD.NV_2023.docx"),
    "TTR": Path("data/documents/BM.01.03.TTR.NV_2023.docx"),
    "BUDGET": Path("data/documents/BM.01_Dự toán.xlsx"),
}

CHECKS = {
    "duration_months": {"label": "Thời gian thực hiện 18 tháng", "patterns": [r"\b18\s*tháng\b", r"\b18\s*months?\b"], "documents": ["MRD", "PRD", "TTR"]},
    "frequency_n78": {"label": "Băng tần n78 / 3.5 GHz", "patterns": [r"\bn78\b", r"3[.,]5\s*GHz", r"3500\s*MHz"], "documents": ["MRD", "PRD", "TTR"]},
    "bandwidth_100mhz": {"label": "Băng thông 100 MHz", "patterns": [r"\b100\s*MHz\b"], "documents": ["MRD", "PRD", "TTR"]},
    "throughput_1_2gbps": {"label": "Thông lượng tối thiểu 1.2 Gbps", "patterns": [r"1[.,]2\s*Gbps", r"1200\s*Mbps"], "documents": ["MRD", "PRD", "TTR"]},
    "evm_3_2": {"label": "EVM không vượt quá 3.2%", "patterns": [r"3[.,]2\s*%", r"3[.,]2\s*percent"], "documents": ["MRD", "PRD", "TTR"]},
    "runtime_72h": {"label": "Chạy liên tục tối thiểu 72 giờ", "patterns": [r"\b72\s*(?:giờ|h|hours?)\b"], "documents": ["MRD", "PRD", "TTR"]},
    "rf_4t4r": {"label": "Cấu hình nguyên mẫu 4T4R", "patterns": [r"\b4T4R\b"], "documents": ["MRD", "PRD", "TTR"]},
    "rf_32t32r": {"label": "Cấu hình tham chiếu 32T32R", "patterns": [r"\b32T32R\b"], "documents": ["MRD", "PRD", "TTR"]},
    "cost_reduction_40": {"label": "Mục tiêu giảm chi phí 40%", "patterns": [r"\b40\s*%", r"\b40\s*percent\b"], "documents": ["MRD", "PRD", "TTR"]},
}

SUPPORTED_DOCUMENT_TYPES = {"docx", "xlsx"}
DEFAULT_RECONCILIATION_CELLS = ("F29", "H29", "D32")
AUDIT_STATUSES = {
    "consistent",
    "contradiction_candidate",
    "missing_evidence",
    "incomplete_coverage",
    "mentioned_unverified",
    "not_applicable",
    "unsupported",
    "error",
    "evaluated",
}


def _evidence_record(
    *,
    check_id: str,
    document_id: str,
    source_path: Path | None,
    location: str | None,
    check_method: str,
    status: str,
    reason: str,
    limitations: list[str],
    evidence_text: str | None = None,
    observed_value: Any = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "document_id": document_id,
        "source_path": str(source_path.resolve()) if source_path is not None else None,
        "location": location,
        "evidence_text": evidence_text,
        "observed_value": observed_value,
        "check_method": check_method,
        "status": status,
        "reason": reason,
        "limitations": limitations,
    }


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").replace("–", "-").replace("—", "-")).strip()


def read_docx(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    document = Document(path)
    blocks = []
    for index, paragraph in enumerate(document.paragraphs, start=1):
        value = normalize_text(paragraph.text)
        if value:
            blocks.append({"document": path.name, "location": f"paragraph {index}", "text": value})
    for ti, table in enumerate(document.tables, start=1):
        for ri, row in enumerate(table.rows, start=1):
            value = " | ".join(normalize_text(c.text) for c in row.cells if normalize_text(c.text))
            if value:
                blocks.append({"document": path.name, "location": f"table {ti}, row {ri}", "text": value})
    return blocks


def read_xlsx(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    value_wb = load_workbook(path, data_only=True, read_only=False)
    formula_wb = load_workbook(path, data_only=False, read_only=False)
    blocks, formulas = [], []
    try:
        for sheet in value_wb.sheetnames:
            vws, fws = value_wb[sheet], formula_wb[sheet]
            for row_index, row in enumerate(vws.iter_rows(), start=1):
                parts = []
                for cell in row:
                    if cell.value is None:
                        continue
                    parts.append(f"{cell.coordinate}={normalize_text(cell.value)}")
                    formula = fws[cell.coordinate].value
                    if isinstance(formula, str) and formula.startswith("="):
                        formulas.append({"sheet": sheet, "cell": cell.coordinate, "formula": formula, "cached_value": cell.value})
                if parts:
                    blocks.append({
                        "document": path.name,
                        "location": f"sheet {sheet}, row {row_index}",
                        "text": " | ".join(parts),
                    })
        return blocks, {"sheets": list(value_wb.sheetnames), "formulas": formulas}
    finally:
        value_wb.close()
        formula_wb.close()


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def run_parameter_checks(documents: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return run_configured_checks(
        documents,
        CHECKS,
    )


def run_configured_checks(
    documents: dict[str, list[dict[str, str]]],
    checks: dict[str, dict[str, Any]],
    source_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    source_paths = source_paths or {}
    results = []
    for check_id, config in checks.items():
        doc_results = []
        configured_documents = ", ".join(config["documents"])
        for key in config["documents"]:
            matched_blocks = [
                block for block in documents.get(key, [])
                if contains_any(block["text"], config["patterns"])
            ]
            matches = [
                {"location": block.get("location"), "quote": block["text"]}
                for block in matched_blocks
            ]
            status = "mentioned_unverified" if matches else "missing_evidence"
            source_path = source_paths.get(key)
            limitations = [
                "Regex matching confirms text occurrence only; it does not "
                "verify meaning, subject, scope, conditions, or truth.",
                f"This check covers only configured documents: {configured_documents}.",
                "Only text blocks extracted by the selected document reader were searched.",
            ]
            evidence = [
                _evidence_record(
                    check_id=check_id,
                    document_id=key,
                    source_path=source_path,
                    location=block.get("location"),
                    evidence_text=block["text"],
                    check_method="regex_keyword",
                    status=status,
                    reason=(
                        "A configured pattern was found; semantic context "
                        "and claim validity were not evaluated."
                        if matches else
                        "No configured pattern was found in the blocks read; "
                        "this does not establish that the requirement is absent."
                    ),
                    limitations=limitations,
                )
                for block in matched_blocks[:10]
            ]
            if not evidence:
                evidence.append(
                    _evidence_record(
                        check_id=check_id,
                        document_id=key,
                        source_path=source_path,
                        location=None,
                        check_method="regex_keyword",
                        status=status,
                        reason=(
                            "No configured pattern was found in the blocks "
                            "read; this does not establish that the requirement "
                            "is absent."
                        ),
                        limitations=limitations,
                    )
                )
            if len(matched_blocks) > 10:
                evidence[0]["limitations"].append(
                    "Only the first 10 matching blocks are included; match_count "
                    "contains the total number found."
                )
            doc_results.append({
                "document": key,
                "status": "found" if matches else "not_found",
                "evidence_status": status,
                "matches": matches[:10],
                "match_count": len(matches),
                "evidence": evidence,
            })
        found = sum(d["status"] == "found" for d in doc_results)
        status = "present_in_all" if found == len(doc_results) else ("not_found_in_any" if found == 0 else "incomplete_coverage")
        evidence_status = (
            "mentioned_unverified" if found == len(doc_results)
            else "missing_evidence" if found == 0
            else "incomplete_coverage"
        )
        results.append({"check_id": check_id, "label": config["label"], "status": status, "documents": doc_results,
                        "evidence_status": evidence_status,
                        "check_method": "regex_keyword",
                        "reason": "Chỉ kiểm tra sự xuất hiện của chuỗi; không kiểm tra ý nghĩa, phạm vi hoặc tính đúng đắn.",
                        "limitations": [
                            "Không tìm thấy chuỗi không chứng minh yêu cầu bị thiếu.",
                            "Không tự kết luận mâu thuẫn ngữ nghĩa.",
                            f"Chỉ tìm trong các tài liệu đã cấu hình: {configured_documents}.",
                            "Chỉ tìm trong các khối văn bản do reader trích xuất.",
                        ],
                        "interpretation": "Chỉ xác nhận chuỗi xuất hiện; chưa xác nhận ngữ cảnh hoặc ý nghĩa."})
    return results


class FormulaEvaluator:
    """Evaluate a conservative subset of Excel formulas and retain failure status."""

    def __init__(self, formula_wb, value_wb):
        self.formula_wb = formula_wb
        self.value_wb = value_wb
        self._cache: dict[tuple[str, str], Any] = {}
        self._evaluation: dict[tuple[str, str], dict[str, str]] = {}
        self._active: set[tuple[str, str]] = set()

    def cell_value(self, sheet: str, cell: str):
        key = (sheet, cell.replace("$", "").upper())
        if key in self._cache:
            return self._cache[key]
        if key in self._active:
            self._evaluation[key] = {
                "status": "error",
                "reason": "Circular formula reference detected.",
            }
            return None
        self._active.add(key)
        try:
            raw = self.formula_wb[sheet][key[1]].value
            if not (isinstance(raw, str) and raw.startswith("=")):
                value = raw
                status = "evaluated" if raw is not None else "missing_evidence"
                reason = (
                    "Cell value read from the workbook."
                    if raw is not None else
                    "Cell is blank."
                )
            else:
                value = self._eval_formula(raw[1:], sheet)
                status = "evaluated" if value is not None else "missing_evidence"
                reason = (
                    "Formula recalculated by the supported evaluator."
                    if value is not None else
                    "Formula did not produce a usable value."
                )
            self._cache[key] = value
            self._evaluation[key] = {"status": status, "reason": reason}
            return value
        except UnsupportedFormulaError as exc:
            self._cache[key] = None
            self._evaluation[key] = {
                "status": "unsupported",
                "reason": str(exc),
            }
            return None
        except MissingFormulaInputError as exc:
            self._cache[key] = None
            self._evaluation[key] = {
                "status": "missing_evidence",
                "reason": str(exc),
            }
            return None
        except Exception as exc:
            self._cache[key] = None
            self._evaluation[key] = {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            return None
        finally:
            self._active.discard(key)

    def cell_evaluation(self, sheet: str, cell: str) -> dict[str, Any]:
        value = self.cell_value(sheet, cell)
        key = (sheet, cell.replace("$", "").upper())
        evaluation = self._evaluation.get(
            key,
            {
                "status": "error",
                "reason": "Formula evaluation did not produce a status.",
            },
        )
        return {
            "value": value,
            "status": evaluation["status"],
            "reason": evaluation["reason"],
        }

    def _split_args(self, text: str) -> list[str]:
        args, start, depth, in_quote = [], 0, 0, False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "'":
                in_quote = not in_quote
            elif not in_quote:
                if ch == "(": depth += 1
                elif ch == ")": depth -= 1
                elif ch == "," and depth == 0:
                    args.append(text[start:i].strip())
                    start = i + 1
            i += 1
        args.append(text[start:].strip())
        return args

    def _eval_formula(self, formula: str, current_sheet: str):
        return self._eval_expr(formula.strip(), current_sheet)

    def _range_values(self, expr: str, current_sheet: str):
        ref = self._parse_ref(expr.strip(), current_sheet)
        if not ref:
            raise UnsupportedFormulaError(
                f"Unsupported range or cell reference: {expr.strip()}"
            )
        sheet, address = ref
        if ":" not in address:
            val = self.cell_value(sheet, address)
            return [] if val is None else [val]
        min_col, min_row, max_col, max_row = range_boundaries(address)
        out = []
        for row in self.formula_wb[sheet].iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
            for cell in row:
                val = self.cell_value(sheet, cell.coordinate)
                if isinstance(val, (int, float)):
                    out.append(val)
        return out

    def _parse_ref(self, token: str, current_sheet: str):
        token = token.strip()
        m = re.fullmatch(r"(?:(?:'([^']+)'|([A-Za-z0-9_ ]+))!)?\$?([A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?)", token, re.I)
        if not m:
            return None
        sheet = m.group(1) or m.group(2) or current_sheet
        address = m.group(3).replace("$", "").upper()
        if sheet not in self.formula_wb.sheetnames:
            return None
        return sheet, address

    def _eval_expr(self, expr: str, current_sheet: str):
        expr = expr.strip()
        # Resolve supported Excel functions recursively, including functions embedded in arithmetic.
        for _ in range(20):
            m = re.search(r"(?i)\b(IF|SUM|ROUND)\s*\(", expr)
            if not m:
                break
            name = m.group(1).upper()
            open_pos = expr.find("(", m.start())
            depth, close_pos, in_quote = 0, None, False
            for i in range(open_pos, len(expr)):
                ch = expr[i]
                if ch == "'": in_quote = not in_quote
                elif not in_quote:
                    if ch == "(": depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            close_pos = i
                            break
            if close_pos is None:
                raise UnsupportedFormulaError("Unclosed or unsupported function.")
            args = self._split_args(expr[open_pos + 1:close_pos])
            if name == "SUM":
                if len(args) != 1:
                    raise UnsupportedFormulaError(
                        "SUM supports exactly one cell or range argument."
                    )
                result = sum(self._range_values(args[0], current_sheet), 0)
            elif name == "ROUND":
                if len(args) != 2:
                    raise UnsupportedFormulaError("ROUND expects 2 arguments.")
                val = self._eval_expr(args[0], current_sheet)
                digits = int(self._eval_expr(args[1], current_sheet))
                result = round(val, digits)
            else:
                if len(args) != 3:
                    raise UnsupportedFormulaError("IF expects 3 arguments.")
                cond = args[0]
                cm = re.fullmatch(r"(.+?)(<=|>=|<>|=|<|>)(.+)", cond)
                if not cm:
                    raise UnsupportedFormulaError("Unsupported IF condition.")
                left = self._eval_expr(cm.group(1), current_sheet)
                right = self._eval_expr(cm.group(3), current_sheet)
                op = cm.group(2)
                truth = {"=": left == right, "<>": left != right, "<": left < right, ">": left > right, "<=": left <= right, ">=": left >= right}[op]
                result = self._eval_expr(args[1] if truth else args[2], current_sheet)
            expr = expr[:m.start()] + str(result) + expr[close_pos + 1:]
        if re.search(r"(?i)\b(IF|SUM|ROUND)\s*\(", expr):
            raise UnsupportedFormulaError("Supported function nesting limit exceeded.")
        if re.search(r"(?i)\b[A-Z][A-Z0-9_.]*\s*\(", expr):
            raise UnsupportedFormulaError("Excel function is not supported.")

        # Replace cell references, including quoted cross-sheet references, with numeric literals.
        pattern = re.compile(r"(?:'([^']+)'!|([A-Za-z0-9_ ]+)!|)(\$?[A-Z]{1,3}\$?\d+)", re.I)
        def replace(match):
            sheet = match.group(1) or match.group(2) or current_sheet
            address = match.group(3).replace("$", "").upper()
            if sheet not in self.formula_wb.sheetnames:
                return match.group(0)
            value = self.cell_value(sheet, address)
            if value is None or not isinstance(value, (int, float)):
                referenced = self._evaluation.get((sheet, address), {})
                status = referenced.get("status")
                if status == "missing_evidence":
                    raise MissingFormulaInputError(
                        f"Referenced cell {sheet}!{address} has no usable value."
                    )
                if status == "error":
                    raise FormulaEvaluationError(
                        f"Referenced cell {sheet}!{address} could not be evaluated."
                    )
                raise UnsupportedFormulaError(
                    f"Referenced cell {sheet}!{address} is unsupported or non-numeric."
                )
            return str(value)
        expr = pattern.sub(replace, expr)
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        binops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
        def walk(node):
            if isinstance(node, ast.Expression): return walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in binops: return binops[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)): return walk(node.operand) * (1 if isinstance(node.op, ast.UAdd) else -1)
            raise UnsupportedFormulaError("Formula expression is not supported.")
        return walk(tree)


class UnsupportedFormulaError(ValueError):
    """Raised when a valid Excel formula is outside the evaluator's subset."""


class MissingFormulaInputError(ValueError):
    """Raised when a supported formula depends on a blank input cell."""


class FormulaEvaluationError(ValueError):
    """Raised when a formula or a required formula dependency fails."""


def audit_workbook_formulas(
    path: Path,
    document_id: str = "BUDGET",
) -> dict[str, Any]:
    formula_wb = load_workbook(path, data_only=False)
    value_wb = load_workbook(path, data_only=True)
    count, missing, details = 0, [], []
    try:
        evaluator = FormulaEvaluator(formula_wb, value_wb)
        for sheet in formula_wb.sheetnames:
            for row in formula_wb[sheet].iter_rows():
                for cell in row:
                    formula = cell.value
                    if not (isinstance(formula, str) and formula.startswith("=")):
                        continue
                    count += 1
                    cached = value_wb[sheet][cell.coordinate].value
                    evaluation = evaluator.cell_evaluation(sheet, cell.coordinate)
                    calculated = evaluation["value"]
                    evaluation_status = evaluation["status"]
                    if evaluation_status == "unsupported":
                        cache_status = "unverifiable"
                        evidence_status = "unsupported"
                        reason = evaluation["reason"]
                    elif evaluation_status == "error":
                        cache_status = "unverifiable"
                        evidence_status = "error"
                        reason = evaluation["reason"]
                    elif evaluation_status == "missing_evidence":
                        cache_status = "unverifiable"
                        evidence_status = "missing_evidence"
                        reason = evaluation["reason"]
                    elif cached is None:
                        cache_status = "missing"
                        evidence_status = "missing_evidence"
                        reason = "The formula was recalculated, but the workbook has no cached value for comparison."
                    else:
                        cache_matches = (
                            abs(float(cached) - float(calculated)) < 0.01
                            if isinstance(cached, (int, float))
                            and not isinstance(cached, bool)
                            and isinstance(calculated, (int, float))
                            and not isinstance(calculated, bool)
                            else cached == calculated
                        )
                        cache_status = "matches" if cache_matches else "differs"
                        evidence_status = (
                            "consistent" if cache_matches else "contradiction_candidate"
                        )
                        reason = (
                            "Cached and recalculated values match."
                            if cache_matches else
                            "Cached and recalculated values differ; investigate before concluding the workbook is wrong."
                        )
                    item = {
                        "sheet": sheet,
                        "cell": cell.coordinate,
                        "formula": formula,
                        "cached_value": cached,
                        "calculated_value": calculated,
                        "evaluation_status": evaluation_status,
                        "cache_status": cache_status,
                        "evidence": _evidence_record(
                            check_id="excel_formula_audit",
                            document_id=document_id,
                            source_path=path,
                            location=f"sheet {sheet}, cell {cell.coordinate}",
                            evidence_text=formula,
                            observed_value={
                                "cached_value": cached,
                                "calculated_value": calculated,
                            },
                            check_method="conservative_formula_evaluator",
                            status=evidence_status,
                            reason=reason,
                            limitations=[
                                "Only a conservative subset of Excel formulas is supported.",
                                "A value match verifies arithmetic for this cell only, not the business correctness of the workbook.",
                            ],
                        ),
                    }
                    details.append(item)
                    if cached is None:
                        missing.append({"sheet": sheet, "cell": cell.coordinate, "formula": formula})
        return {
            "formula_count": count,
            "missing_cached_value_count": len(missing),
            "missing_cached_values": missing,
            "formulas": details,
            "scope": {
                "sheets_scanned": list(formula_wb.sheetnames),
                "formula_cells_scanned": count,
            },
            "limitation": (
                "Chỉ tự tính tập con công thức trong các ô công thức được đọc "
                "từ worksheet; công thức không hỗ trợ có "
                "evaluation_status='unsupported'. Kết quả không xác minh "
                "tính đúng đắn nghiệp vụ và không thay thế Excel."
            ),
        }
    finally:
        formula_wb.close()
        value_wb.close()


def audit_budget_reconciliation(
    path: Path,
    sheet_name: str = "Tong hop",
    cells: tuple[str, ...] = DEFAULT_RECONCILIATION_CELLS,
    document_id: str = "BUDGET",
) -> dict[str, Any]:
    if len(cells) < 2:
        raise ValueError("Excel reconciliation requires at least two cells")
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    wb_f = load_workbook(path, data_only=False)
    wb_v = load_workbook(path, data_only=True)
    try:
        if sheet_name not in wb_f.sheetnames:
            return {
                "status": "incomplete",
                "evidence_status": "missing_evidence",
                "reason": f"missing_sheet_{sheet_name}",
                "checks": [],
                "limitations": ["The requested sheet was not present; no reconciliation was performed."],
            }
        ev = FormulaEvaluator(wb_f, wb_v)
        ws = wb_f[sheet_name]
        checks = []
        for cell in cells:
            raw = ws[cell].value
            value = ev.cell_value(sheet_name, cell)
            cached = wb_v[sheet_name][cell].value
            is_formula = isinstance(raw, str) and raw.startswith("=")
            evaluation = (
                ev.cell_evaluation(sheet_name, cell)
                if is_formula else {
                    "value": value,
                    "status": "evaluated" if value is not None else "missing_evidence",
                    "reason": (
                        "Cell value read from the workbook."
                        if value is not None else "Cell is blank."
                    ),
                }
            )
            value = evaluation["value"]
            cell_is_numeric = (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            if evaluation["status"] == "unsupported":
                status = "unsupported"
                reason = evaluation["reason"]
            elif evaluation["status"] == "error":
                status = "error"
                reason = evaluation["reason"]
            elif evaluation["status"] == "missing_evidence" or not cell_is_numeric:
                status = "missing_evidence"
                reason = (
                    evaluation["reason"]
                    if evaluation["status"] == "missing_evidence"
                    else "Cell value is not a usable numeric value for reconciliation."
                )
            else:
                status = "evaluated"
                reason = "The cell value was read or recalculated; it is not a business validation."
            if status == "evaluated" and is_formula:
                if cached is None:
                    status = "missing_evidence"
                    reason = "The formula was recalculated, but no cached value is available for comparison."
                elif isinstance(cached, (int, float)) and not isinstance(cached, bool):
                    cache_matches = abs(float(cached) - float(value)) < 0.01
                    status = (
                        "consistent" if cache_matches else "contradiction_candidate"
                    )
                    reason = (
                        "Cached and recalculated values match."
                        if cache_matches else
                        "Cached and recalculated values differ; investigate before concluding the workbook is wrong."
                    )
                else:
                    cache_matches = cached == value
                    status = (
                        "consistent" if cache_matches else "contradiction_candidate"
                    )
                    reason = (
                        "Cached and recalculated values match."
                        if cache_matches else
                        "Cached and recalculated values differ; investigate before concluding the workbook is wrong."
                    )
            elif status == "evaluated":
                status = "not_applicable"
                reason = "Literal cell value read; formula/cache comparison is not applicable."
            checks.append({
                "cell": cell,
                "formula_or_value": raw,
                "formula": raw if is_formula else None,
                "cached_value": cached,
                "calculated_value": value,
                "evaluation_status": evaluation["status"],
                "cache_status": (
                    "not_applicable" if not is_formula else
                    "missing" if cached is None and evaluation["status"] == "evaluated" else
                    "unverifiable" if evaluation["status"] != "evaluated" else
                    "matches" if status == "consistent" else
                    "differs" if status == "contradiction_candidate" else
                    "unverifiable"
                ),
                "evidence": _evidence_record(
                    check_id="excel_reconciliation",
                    document_id=document_id,
                    source_path=path,
                    location=f"sheet {sheet_name}, cell {cell}",
                    evidence_text=raw if is_formula else None,
                    observed_value={
                        "cached_value": cached,
                        "calculated_value": value,
                    },
                    check_method="conservative_formula_evaluator",
                    status=status,
                    reason=reason,
                    limitations=[
                        "Cell reconciliation does not verify the correctness of underlying business assumptions or source data.",
                        "Only a conservative subset of formulas can be recalculated.",
                    ],
                ),
            })
        vals = [x["calculated_value"] for x in checks]
        evaluation_statuses = {x["evaluation_status"] for x in checks}
        unsupported = "unsupported" in evaluation_statuses
        has_error = "error" in evaluation_statuses
        complete = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals
        )
        totals_agree = complete and all(abs(float(v) - float(vals[0])) < 0.01 for v in vals[1:])
        cache_mismatches = []
        for item in checks:
            cached = item["cached_value"]
            calc = item["calculated_value"]
            if item["formula"] is None or cached is None or calc is None:
                continue
            matches = (
                abs(float(cached) - float(calc)) < 0.01
                if isinstance(cached, (int, float))
                and not isinstance(cached, bool)
                and isinstance(calc, (int, float))
                and not isinstance(calc, bool)
                else cached == calc
            )
            if not matches:
                cache_mismatches.append({"cell": item["cell"], "cached_value": cached, "calculated_value": calc})
        if has_error:
            status, evidence_status = "error", "error"
        elif unsupported:
            status, evidence_status = "unsupported", "unsupported"
        elif not complete:
            status, evidence_status = "incomplete", "missing_evidence"
        elif not totals_agree or cache_mismatches:
            status, evidence_status = "inconsistent", "contradiction_candidate"
        else:
            status, evidence_status = "consistent", "consistent"
        return {
            "status": status,
            "evidence_status": evidence_status,
            "checks": checks,
            "cache_mismatches": cache_mismatches,
            "scope": {"sheet": sheet_name, "cells": list(cells)},
            "note": f"Đối chiếu {', '.join(cells)} trên sheet {sheet_name}; đây chỉ là so sánh các giá trị đã chọn, không chứng minh tính đúng đắn của nghiệp vụ, giả định hoặc dữ liệu nguồn.",
            "limitations": [
                "Chỉ hỗ trợ một tập con công thức Excel.",
                f"Chỉ đối chiếu sheet {sheet_name} và các cell đã cấu hình: {', '.join(cells)}.",
                "Các tổng khớp không chứng minh toàn bộ ngân sách hoặc dữ liệu nghiệp vụ là đúng.",
                "Chênh lệch là ứng viên cần kiểm tra, không tự động chứng minh workbook sai.",
            ],
        }
    finally:
        wb_f.close()
        wb_v.close()


def inspect_budget_total(blocks: list[dict[str, str]]) -> dict[str, Any]:
    keywords = ("tổng kinh phí", "tổng cộng", "kinh phí tổng", "tổng dự toán")
    matches = [{"location": b["location"], "text": b["text"]} for b in blocks if any(k in b["text"].casefold() for k in keywords)]
    return {"status": "candidates_found" if matches else "not_found", "candidate_count": len(matches), "candidates": matches[:100],
            "note": "Xác minh đúng dòng tổng và đơn vị trước khi kết luận."}


def build_report(files: dict[str, Path]) -> dict[str, Any]:
    docs = {key: read_docx(files[key]) for key in ("MRD", "PRD", "TTR")}
    budget_blocks, meta = read_xlsx(files["BUDGET"])
    docs["BUDGET"] = budget_blocks
    return {"engine": "JEV Deterministic Cross-Document Audit", "mode": "read-only; no output files",
            "inputs": {key: str(path.resolve()) for key, path in files.items()},
            "document_summary": {key: {"block_count": len(v), "empty": not v} for key, v in docs.items()},
            "parameter_checks": run_configured_checks(
                docs, CHECKS, {key: Path(path) for key, path in files.items()}
            ),
            "budget_total_candidates": inspect_budget_total(budget_blocks),
            "workbook": {"sheets": meta["sheets"], "formula_audit": audit_workbook_formulas(files["BUDGET"], "BUDGET"),
                         "budget_reconciliation": audit_budget_reconciliation(files["BUDGET"], document_id="BUDGET")},
            "limitations": ["Chuỗi xuất hiện không chứng minh hai yêu cầu tương đương.", "Không tìm thấy chuỗi không chứng minh nội dung bị thiếu.",
                            "Không tự kết luận mâu thuẫn ngữ nghĩa.", "Chỉ hỗ trợ một tập con công thức Excel.", "Mọi kết luận ngữ nghĩa cần người kiểm tra."]}


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8-sig") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid manifest JSON at {manifest_path}, line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")

    mission = manifest.get("mission")
    if not isinstance(mission, dict):
        raise ValueError("Manifest field 'mission' must be an object")
    mission_id = _require_nonempty_string(mission.get("id"), "mission.id")
    mission_description = _require_nonempty_string(
        mission.get("description"), "mission.description"
    )

    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("Manifest field 'documents' must be a non-empty array")

    documents = []
    document_ids: set[str] = set()
    for index, raw_document in enumerate(raw_documents, start=1):
        context = f"documents[{index - 1}]"
        if not isinstance(raw_document, dict):
            raise ValueError(f"{context} must be an object")

        document_id = _require_nonempty_string(
            raw_document.get("id"), f"{context}.id"
        )
        if document_id in document_ids:
            raise ValueError(f"Duplicate document id: {document_id}")
        document_ids.add(document_id)

        role = _require_nonempty_string(raw_document.get("role"), f"{context}.role")
        raw_path = _require_nonempty_string(
            raw_document.get("path"), f"{context}.path"
        )
        file_type = _require_nonempty_string(
            raw_document.get("type"), f"{context}.type"
        ).casefold().lstrip(".")
        if file_type not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError(
                f"Unsupported document type '{file_type}' for '{document_id}'; "
                f"supported types: {', '.join(sorted(SUPPORTED_DOCUMENT_TYPES))}"
            )

        document_path = Path(raw_path).expanduser()
        if not document_path.is_absolute():
            document_path = manifest_path.parent / document_path
        document_path = document_path.resolve()
        if document_path.suffix.casefold() != f".{file_type}":
            raise ValueError(
                f"Document '{document_id}' declares type '{file_type}' but "
                f"has extension '{document_path.suffix or '(none)'}'"
            )
        if not document_path.is_file():
            raise FileNotFoundError(
                f"Document '{document_id}' does not exist or is not a file: "
                f"{document_path}"
            )

        documents.append({
            "id": document_id,
            "role": role,
            "path": document_path,
            "type": file_type,
        })

    raw_checks = manifest.get("checks", [])
    if not isinstance(raw_checks, list):
        raise ValueError("Manifest field 'checks' must be an array")

    checks: dict[str, dict[str, Any]] = {}
    for index, raw_check in enumerate(raw_checks, start=1):
        context = f"checks[{index - 1}]"
        if not isinstance(raw_check, dict):
            raise ValueError(f"{context} must be an object")

        check_id = _require_nonempty_string(raw_check.get("id"), f"{context}.id")
        if check_id in checks:
            raise ValueError(f"Duplicate check id: {check_id}")
        label = _require_nonempty_string(
            raw_check.get("label"), f"{context}.label"
        )
        patterns = raw_check.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"{context}.patterns must be a non-empty array")
        for pattern_index, pattern in enumerate(patterns, start=1):
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    f"{context}.patterns[{pattern_index - 1}] must be a "
                    "non-empty string"
                )
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in {context}.patterns[{pattern_index - 1}]: {exc}"
                ) from exc

        applies_to = raw_check.get("documents")
        if not isinstance(applies_to, list) or not applies_to:
            raise ValueError(f"{context}.documents must be a non-empty array")
        if any(not isinstance(document_id, str) for document_id in applies_to):
            raise ValueError(f"{context}.documents must contain document IDs")
        if len(set(applies_to)) != len(applies_to):
            raise ValueError(f"{context}.documents contains duplicate document IDs")
        unknown_ids = sorted(set(applies_to) - document_ids)
        if unknown_ids:
            raise ValueError(
                f"{context}.documents references unknown document IDs: "
                f"{', '.join(unknown_ids)}"
            )

        checks[check_id] = {
            "label": label,
            "patterns": patterns,
            "documents": applies_to,
        }

    reconciliation = manifest.get("reconciliation")
    if reconciliation is not None:
        if not isinstance(reconciliation, dict):
            raise ValueError("Manifest field 'reconciliation' must be an object")
        reconciliation_document_id = _require_nonempty_string(
            reconciliation.get("document_id"), "reconciliation.document_id"
        )
        document_by_id = {document["id"]: document for document in documents}
        if reconciliation_document_id not in document_by_id:
            raise ValueError(
                "reconciliation.document_id references unknown document ID: "
                f"{reconciliation_document_id}"
            )
        if document_by_id[reconciliation_document_id]["type"] != "xlsx":
            raise ValueError(
                "reconciliation.document_id must reference an xlsx document"
            )

        sheet_name = _require_nonempty_string(
            reconciliation.get("sheet", "Tong hop"), "reconciliation.sheet"
        )
        raw_cells = reconciliation.get(
            "cells", list(DEFAULT_RECONCILIATION_CELLS)
        )
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError("reconciliation.cells must be a non-empty array")
        cells = []
        for index, cell in enumerate(raw_cells, start=1):
            if not isinstance(cell, str) or not re.fullmatch(
                r"\$?[A-Z]{1,3}\$?\d+", cell, flags=re.IGNORECASE
            ):
                raise ValueError(
                    f"reconciliation.cells[{index - 1}] must be an Excel cell "
                    "reference such as F29"
                )
            cells.append(cell.replace("$", "").upper())
        if len(set(cells)) != len(cells):
            raise ValueError("reconciliation.cells contains duplicate cells")
        if len(cells) < 2:
            raise ValueError("reconciliation.cells must contain at least two cells")
        reconciliation = {
            "document_id": reconciliation_document_id,
            "sheet": sheet_name,
            "cells": tuple(cells),
        }

    return {
        "path": manifest_path,
        "mission": {"id": mission_id, "description": mission_description},
        "documents": documents,
        "checks": checks,
        "reconciliation": reconciliation,
    }


def build_manifest_report(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    document_blocks: dict[str, list[dict[str, str]]] = {}
    workbook_metadata: dict[str, dict[str, Any]] = {}
    document_summary: dict[str, dict[str, Any]] = {}

    for document in manifest["documents"]:
        document_id = document["id"]
        path = document["path"]
        try:
            if document["type"] == "docx":
                blocks = read_docx(path)
            else:
                blocks, metadata = read_xlsx(path)
                workbook_metadata[document_id] = metadata
        except Exception as exc:
            raise RuntimeError(
                f"Could not read document '{document_id}' ({path}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        document_blocks[document_id] = blocks
        document_summary[document_id] = {
            "role": document["role"],
            "path": str(path),
            "type": document["type"],
            "block_count": len(blocks),
            "empty": not blocks,
        }

    workbook_reports: dict[str, dict[str, Any]] = {}
    reconciliation = manifest["reconciliation"]
    for document in manifest["documents"]:
        if document["type"] != "xlsx":
            continue
        document_id = document["id"]
        path = document["path"]
        metadata = workbook_metadata[document_id]
        workbook_report = {
            "sheets": metadata["sheets"],
            "formula_audit": audit_workbook_formulas(path, document_id),
        }
        if reconciliation and reconciliation["document_id"] == document_id:
            workbook_report["reconciliation"] = audit_budget_reconciliation(
                path,
                sheet_name=reconciliation["sheet"],
                cells=reconciliation["cells"],
                document_id=document_id,
            )
        workbook_reports[document_id] = workbook_report

    return {
        "engine": "JEV Deterministic Cross-Document Audit",
        "mode": "manifest; read-only; no output files",
        "mission": manifest["mission"],
        "inputs": {
            document["id"]: str(document["path"])
            for document in manifest["documents"]
        },
        "document_summary": document_summary,
        "parameter_checks": run_configured_checks(
            document_blocks,
            manifest["checks"],
            {
                document["id"]: document["path"]
                for document in manifest["documents"]
            },
        ),
        "workbooks": workbook_reports,
        "evidence_schema_version": 1,
        "status_taxonomy": sorted(AUDIT_STATUSES),
        "limitations": [
            "Chuỗi xuất hiện không chứng minh hai yêu cầu tương đương.",
            "Không tìm thấy chuỗi không chứng minh nội dung bị thiếu.",
            "Không tự kết luận mâu thuẫn ngữ nghĩa.",
            "Chỉ hỗ trợ một tập con công thức Excel.",
            "Mọi kết luận ngữ nghĩa cần người kiểm tra.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JEV read-only document audit (manifest or legacy mode)."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mrd", type=Path)
    parser.add_argument("--prd", type=Path)
    parser.add_argument("--ttr", type=Path)
    parser.add_argument("--budget", type=Path)
    args = parser.parse_args()

    try:
        legacy_args = (args.mrd, args.prd, args.ttr, args.budget)
        if args.manifest is not None:
            if any(path is not None for path in legacy_args):
                parser.error("--manifest cannot be combined with legacy document options")
            report = build_manifest_report(args.manifest)
        else:
            defaults = default_files() if any(
                path is None for path in legacy_args
            ) else {}
            files = {
                "MRD": args.mrd or defaults["MRD"],
                "PRD": args.prd or defaults["PRD"],
                "TTR": args.ttr or defaults["TTR"],
                "BUDGET": args.budget or defaults["BUDGET"],
            }
            report = build_report(files)
    except Exception as exc:
        raise SystemExit(f"[JEV AUDIT ERROR] {type(exc).__name__}: {exc}") from exc
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
