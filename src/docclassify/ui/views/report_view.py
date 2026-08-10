"""
UI for generating and viewing the monthly report, and viewing/exporting
individual document summaries.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QTextEdit
)
from PySide6.QtCore import QThread, Signal


class ReportWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, batch_id: str):
        super().__init__()
        self.batch_id = batch_id

    def run(self):
        try:
            from docclassify.reports.monthly_report import generate_monthly_report
            report = generate_monthly_report(self.batch_id)
            self.finished_ok.emit(report)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ReportView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Monthly Report</h2>"))

        batch_row = QHBoxLayout()
        self.batch_input = QLineEdit()
        self.batch_input.setPlaceholderText("Batch id, e.g. 2026-08")
        generate_button = QPushButton("Generate Report")
        generate_button.clicked.connect(self._generate)
        batch_row.addWidget(self.batch_input)
        batch_row.addWidget(generate_button)
        layout.addLayout(batch_row)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.worker = None

    def _generate(self):
        batch_id = self.batch_input.text().strip()
        if not batch_id:
            return
        self.output.setPlainText("Generating report...")

        self.worker = ReportWorker(batch_id)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(lambda e: self.output.setPlainText(f"ERROR: {e}"))
        self.worker.start()

    def _on_finished(self, report: dict):
        lines = [f"=== Overall Digest ===", report["overall_digest"], "", "=== Stats ===", str(report["stats"]), ""]
        lines.append("=== Per-Category Summaries ===")
        for category, summary in report["category_summaries"].items():
            lines.append(f"\n[{category}]\n{summary}")
        self.output.setPlainText("\n".join(lines))
