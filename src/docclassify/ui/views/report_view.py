"""
Reports view: the monthly batch digest, and on-demand single-document summaries.

What was wrong before, and what this fixes:
  * The batch id had to be typed by hand ("e.g. 2026-08") with no way to discover
    which batches exist, so a typo silently produced a report over zero documents.
    Batches are now a dropdown read from the database, with document counts.
  * This module's own docstring promised "viewing/exporting individual document
    summaries" and neither existed. reports/doc_summary.py was complete but
    unreachable from anywhere in the UI — it is now the second tab.
  * There was no export, despite monthly_reports.report_path existing in the schema
    for exactly that.
  * A running report replaced self.worker while the old QThread was still alive.
"""
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QProgressBar, QPushButton, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from docclassify.ui.widgets import (
    Card, SectionTitle, ViewHeader, format_duration, muted, row, show_error,
)


class ReportWorker(QThread):
    finished_ok = Signal(dict, float)
    failed = Signal(str, str)

    def __init__(self, batch_id: str):
        super().__init__()
        self.batch_id = batch_id

    def run(self):
        import time
        import traceback
        try:
            from docclassify.reports.monthly_report import generate_monthly_report
            started = time.perf_counter()
            report = generate_monthly_report(self.batch_id)
            self.finished_ok.emit(report, time.perf_counter() - started)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


class SummaryWorker(QThread):
    finished_ok = Signal(str, str, float)   # doc_id, summary, seconds
    failed = Signal(str, str)

    def __init__(self, doc_id: str, force: bool):
        super().__init__()
        self.doc_id = doc_id
        self.force = force

    def run(self):
        import time
        import traceback
        try:
            from docclassify.reports.doc_summary import get_or_generate_summary
            from docclassify.storage import lancedb_store

            started = time.perf_counter()
            # The full text is never persisted, only chunks — reassemble it.
            text = lancedb_store.get_document_text(self.doc_id)
            if not text.strip():
                self.failed.emit(
                    "No indexed text for this document.",
                    "get_document_text() returned nothing, so no chunks are stored "
                    "against this doc_id.",
                )
                return
            summary = get_or_generate_summary(self.doc_id, text, force_regenerate=self.force)
            self.finished_ok.emit(self.doc_id, summary, time.perf_counter() - started)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


class ReportView(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: ReportWorker | None = None
        self.summary_worker: SummaryWorker | None = None
        self._last_report: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addWidget(ViewHeader(
            "Reports",
            "A monthly digest built from per-category summaries, and on-demand "
            "one-page summaries for a single document.",
        ))

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_batch_tab(), "Monthly digest")
        self.tabs.addTab(self._build_document_tab(), "Document summary")
        layout.addWidget(self.tabs, 1)

    # ------------------------------------------------------------- batch digest
    def _build_batch_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)

        controls = Card()
        controls.body.addWidget(SectionTitle("Upload batch"))

        self.batch_combo = QComboBox()
        self.batch_combo.setMinimumWidth(280)

        self.refresh_batches_button = QPushButton("Refresh")
        self.refresh_batches_button.clicked.connect(self.reload_batches)

        self.generate_button = QPushButton("Generate Report")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)

        controls.body.addWidget(row(self.batch_combo, self.refresh_batches_button,
                                     self.generate_button, None))
        controls.body.addWidget(muted(
            "Cost is one LLM call per populated category plus one overall digest, so "
            "this is slow on CPU. The window stays responsive throughout."
        ))
        page_layout.addWidget(controls)

        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 1)
        self.report_progress.setValue(0)
        self.report_progress.setTextVisible(False)
        self.report_progress.setVisible(False)
        page_layout.addWidget(self.report_progress)

        self.report_status = muted("Choose a batch and generate a report.")
        page_layout.addWidget(self.report_status)

        self.report_output = QTextEdit()
        self.report_output.setReadOnly(True)
        self.report_output.setPlaceholderText("The generated digest appears here.")
        page_layout.addWidget(self.report_output, 1)

        self.export_button = QPushButton("Export to text file…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        page_layout.addWidget(row(self.export_button, None))

        return page

    def reload_batches(self):
        from docclassify.storage import sqlite_store

        previous = self.batch_combo.currentData()
        self.batch_combo.clear()
        try:
            batches = sqlite_store.list_upload_batches()
        except Exception as e:  # noqa: BLE001
            self.report_status.setText(f"Could not list batches: {e}")
            return

        for entry in batches:
            label = f"{entry['batch']}   ({entry['document_count']} documents)"
            self.batch_combo.addItem(label, entry["batch"])

        if not batches:
            self.batch_combo.addItem("(no batches — ingest documents first)", None)
            self.generate_button.setEnabled(False)
            self.report_status.setText("Nothing ingested yet.")
            return

        self.generate_button.setEnabled(True)
        if previous is not None:
            index = self.batch_combo.findData(previous)
            if index >= 0:
                self.batch_combo.setCurrentIndex(index)

    def _generate(self):
        batch_id = self.batch_combo.currentData()
        if not batch_id:
            show_error(self, "No batch selected",
                       "Ingest some documents first — a report needs a batch to summarise.")
            return
        if self.worker is not None and self.worker.isRunning():
            self.report_status.setText("A report is already being generated…")
            return

        self.report_output.clear()
        self.export_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.report_progress.setRange(0, 0)   # indeterminate: call count is unknown upfront
        self.report_progress.setVisible(True)
        self.report_status.setText(f"Generating the digest for '{batch_id}'…")

        self.worker = ReportWorker(batch_id)
        self.worker.finished_ok.connect(self._on_report_ready)
        self.worker.failed.connect(self._on_report_failed)
        self.worker.start()

    def _on_report_ready(self, report: dict, seconds: float):
        self._last_report = report
        self.generate_button.setEnabled(True)
        self.report_progress.setVisible(False)
        self.report_progress.setRange(0, 1)
        self.export_button.setEnabled(True)

        stats = report.get("stats", {})
        categories = stats.get("by_category", {}) or {}
        calls = len(categories) + 1
        self.report_status.setText(
            f"Generated in {format_duration(seconds)}  ·  {calls} LLM call(s)  ·  "
            f"{stats.get('total_documents', 0)} documents across {len(categories)} categories"
        )
        self.report_output.setPlainText(self._render(report))

    @staticmethod
    def _render(report: dict) -> str:
        stats = report.get("stats", {})
        lines = [
            f"MONTHLY REPORT — batch {report.get('batch_id', '?')}",
            "=" * 70,
            "",
            "OVERALL DIGEST",
            "-" * 70,
            report.get("overall_digest", "").strip(),
            "",
            "STATISTICS",
            "-" * 70,
            f"Total documents : {stats.get('total_documents', 0)}",
        ]
        for heading, key in (("By category", "by_category"),
                              ("By status", "by_status"),
                              ("By language", "by_language")):
            values = stats.get(key) or {}
            if values:
                lines.append(f"{heading}:")
                for name, count in sorted(values.items(), key=lambda kv: -kv[1]):
                    lines.append(f"    {count:>5}  {name}")
        lines += ["", "PER-CATEGORY SUMMARIES", "-" * 70]
        for category, summary in (report.get("category_summaries") or {}).items():
            lines += [f"[{category}]", (summary or "").strip(), ""]
        return "\n".join(lines)

    def _export(self):
        if not self._last_report:
            return
        default_name = f"report_{self._last_report.get('batch_id', 'batch')}.txt"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export report", default_name, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self._render(self._last_report), encoding="utf-8")
        except OSError as e:
            show_error(self, "Could not write the file", str(e))
            return

        # Record where it went, which is what monthly_reports.report_path is for.
        try:
            from docclassify.storage import sqlite_store
            with sqlite_store.get_connection() as conn:
                conn.execute("UPDATE monthly_reports SET report_path = ? WHERE batch_id = ?",
                              (path, self._last_report.get("batch_id")))
        except Exception as e:  # noqa: BLE001 - the file is written; bookkeeping is secondary
            print(f"[report_view] could not record report_path: {e}")

        self.report_status.setText(f"Exported to {path}")

    def _on_report_failed(self, message: str, detail: str):
        self.generate_button.setEnabled(True)
        self.report_progress.setVisible(False)
        self.report_progress.setRange(0, 1)
        self.report_status.setText("Report generation failed.")
        show_error(self, "Report generation failed", message, detail)

    # ---------------------------------------------------------- document summary
    def _build_document_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(14, 14, 14, 14)
        page_layout.setSpacing(12)

        controls = Card()
        controls.body.addWidget(SectionTitle("Document"))

        self.document_combo = QComboBox()
        self.document_combo.setMinimumWidth(320)

        self.refresh_documents_button = QPushButton("Refresh")
        self.refresh_documents_button.clicked.connect(self.reload_documents)

        self.summarise_button = QPushButton("Summarise")
        self.summarise_button.setObjectName("primary")
        self.summarise_button.clicked.connect(lambda: self._summarise(force=False))

        self.regenerate_button = QPushButton("Regenerate")
        self.regenerate_button.setToolTip("Ignore the cached summary and generate a new one.")
        self.regenerate_button.clicked.connect(lambda: self._summarise(force=True))

        controls.body.addWidget(row(self.document_combo, self.refresh_documents_button,
                                     self.summarise_button, self.regenerate_button, None))
        controls.body.addWidget(muted(
            "Summaries are cached on the document row, so asking again is free unless "
            "you regenerate."
        ))
        page_layout.addWidget(controls)

        self.summary_progress = QProgressBar()
        self.summary_progress.setRange(0, 1)
        self.summary_progress.setTextVisible(False)
        self.summary_progress.setVisible(False)
        page_layout.addWidget(self.summary_progress)

        self.summary_status = muted("Choose a document.")
        page_layout.addWidget(self.summary_status)

        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText("The one-page summary appears here.")
        page_layout.addWidget(self.summary_output, 1)

        return page

    def reload_documents(self):
        from docclassify.storage import sqlite_store

        previous = self.document_combo.currentData()
        self.document_combo.clear()
        try:
            with sqlite_store.get_connection() as conn:
                rows = conn.execute(
                    "SELECT doc_id, filename, title_en, category_path FROM documents "
                    "ORDER BY created_at DESC LIMIT 500"
                ).fetchall()
        except Exception as e:  # noqa: BLE001
            self.summary_status.setText(f"Could not list documents: {e}")
            return

        for record in rows:
            label = record["title_en"] or record["filename"] or record["doc_id"]
            category = record["category_path"] or "unassigned"
            self.document_combo.addItem(f"{label}   —   {category}", record["doc_id"])

        enabled = bool(rows)
        if not rows:
            self.document_combo.addItem("(nothing ingested yet)", None)
            self.summary_status.setText("Nothing ingested yet.")
        self.summarise_button.setEnabled(enabled)
        self.regenerate_button.setEnabled(enabled)

        if previous is not None:
            index = self.document_combo.findData(previous)
            if index >= 0:
                self.document_combo.setCurrentIndex(index)

    def _summarise(self, force: bool):
        doc_id = self.document_combo.currentData()
        if not doc_id:
            show_error(self, "No document selected", "Ingest a document first.")
            return
        if self.summary_worker is not None and self.summary_worker.isRunning():
            self.summary_status.setText("A summary is already being generated…")
            return

        self.summary_output.clear()
        self.summarise_button.setEnabled(False)
        self.regenerate_button.setEnabled(False)
        self.summary_progress.setRange(0, 0)
        self.summary_progress.setVisible(True)
        self.summary_status.setText("Regenerating…" if force else "Generating…")

        self.summary_worker = SummaryWorker(doc_id, force)
        self.summary_worker.finished_ok.connect(self._on_summary_ready)
        self.summary_worker.failed.connect(self._on_summary_failed)
        self.summary_worker.start()

    def _on_summary_ready(self, doc_id: str, summary: str, seconds: float):
        self._reset_summary_controls()
        self.summary_status.setText(f"Generated in {format_duration(seconds)}")
        self.summary_output.setPlainText(summary)

    def _on_summary_failed(self, message: str, detail: str):
        self._reset_summary_controls()
        self.summary_status.setText("Summary failed.")
        show_error(self, "Could not generate the summary", message, detail)

    def _reset_summary_controls(self):
        self.summarise_button.setEnabled(True)
        self.regenerate_button.setEnabled(True)
        self.summary_progress.setVisible(False)
        self.summary_progress.setRange(0, 1)

    # ------------------------------------------------------------------- hooks
    def on_shown(self):
        self.reload_batches()
        self.reload_documents()

    def is_busy(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self.worker, self.summary_worker)
        )

    def stop_work(self):
        for worker in (self.worker, self.summary_worker):
            if worker is not None and worker.isRunning():
                worker.wait(15_000)
