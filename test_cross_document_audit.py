import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook, load_workbook
from src.documents.cross_document_audit import (
    FormulaEvaluator,
    audit_budget_reconciliation,
    audit_workbook_formulas,
)


class FormulaEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / "sample.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Tong hop"
        ws["A1"] = 100
        ws["A2"] = 200
        ws["A3"] = "=SUM(A1:A2)"
        ws["A4"] = "=ROUND(A3*0.85,0)"
        ws["A5"] = "=A3-A4"
        ws["A6"] = "=SUM('PL3'!H9:H10)"
        pl3 = wb.create_sheet("PL3")
        pl3["H9"] = 300
        pl3["H10"] = 400
        wb.save(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_supported_formula_evaluation(self):
        formulas = load_workbook(self.path, data_only=False)
        values = load_workbook(self.path, data_only=True)
        try:
            evaluator = FormulaEvaluator(formulas, values)
            self.assertEqual(evaluator.cell_value("Tong hop", "A3"), 300)
            self.assertEqual(evaluator.cell_value("Tong hop", "A4"), 255)
            self.assertEqual(evaluator.cell_value("Tong hop", "A5"), 45)
            self.assertEqual(evaluator.cell_value("Tong hop", "A6"), 700)
        finally:
            formulas.close()
            values.close()

    def test_formula_audit_reports_missing_cache_without_crashing(self):
        report = audit_workbook_formulas(self.path)
        self.assertGreaterEqual(report["formula_count"], 4)
        self.assertGreaterEqual(report["missing_cached_value_count"], 1)

    def test_budget_reconciliation_handles_missing_summary_cells(self):
        report = audit_budget_reconciliation(self.path)
        self.assertEqual(report["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
