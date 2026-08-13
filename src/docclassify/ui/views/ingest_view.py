"""
Ingest view: run the pipeline over the inbox or an arbitrary folder.

What was wrong before, and what this fixes:
  * The worker emitted a single "starting" message and then nothing until the run
    finished. process_folder printed per-document results to stdout, which a GUI
    never sees, so the window looked frozen for however long ingestion took. It now
    reports every document through a progress callback.
  * There was no progress bar and no way to cancel a run.
  * "N documents processed" counted duplicates as successes and omitted failures
    entirely. Outcomes are now counted separately and failures are listed.
  * data/inbox could not be ingested from the UI at all, even though that is the
    documented monthly workflow this view is supposed to drive.
"""
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHeaderView, QLabel, QProgressBar, QPushButton,
    QRadioButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from docclassify.ui.widgets import (
    Card, SectionTitle, ViewHeader, confirm, format_duration, muted, row,
    show_error, status_colour,
)


class IngestWorker(QThread):
    """Runs process_folder off the GUI thread, reporting each document as it lands."""

    document_done = Signal(int, int, str, str, str)   # done, total, filename, status, detail
    finished_ok = Signal(dict)                        # summary counts
    failed = Signal(str, str)                         # message, detail

    def __init__(self, folder_path: str, upload_batch: str):
        super().__init__()
        self.folder_path = folder_path
        self.upload_batch = upload_batch
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self):
        import time
        import traceback

        from docclassify.pipeline import process_folder

        counts = {"ingested": 0, "duplicate": 0, "failed": 0}
        started = time.perf_counter()

        def on_progress(done, total, path, status, record, error):
            counts[status] = counts.get(status, 0) + 1
            if status == "failed":
                detail = f"{type(error).__name__}: {error}"
            elif status == "duplicate":
                detail = "already ingested (identical content)"
            else:
                category = record.get("category_path") or "(unassigned)"
                confidence = record.get("confidence")
                detail = f"{category}  ·  {record.get('status', '?')}"
                if confidence is not None:
                    detail += f"  ·  {confidence:.3f}"
            self.document_done.emit(done, total, Path(path).name, status, detail)

        try:
            process_folder(
                self.folder_path,
                upload_batch=self.upload_batch,
                on_progress=on_progress,
                should_cancel=self._should_cancel,
            )
        except Exception as e:  # noqa: BLE001 - surfaced to the UI rather than crashing the app
            self.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())
            return

        counts["seconds"] = time.perf_counter() - started
        counts["cancelled"] = self._cancelled
        counts["batch"] = self.upload_batch
        self.finished_ok.emit(counts)


class IngestView(QWidget):
    stats_changed = Signal()

    def __init__(self):
        super().__init__()
        self.worker: IngestWorker | None = None
        self.selected_folder: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addWidget(ViewHeader(
            "Ingest & Classify",
            "Parse, OCR where needed, embed, classify and index documents. Each document "
            "is committed as it completes, so cancelling keeps whatever finished.",
        ))

        # --- source selection -------------------------------------------------
        source_card = Card()
        source_card.body.addWidget(SectionTitle("Source"))

        self.inbox_radio = QRadioButton("Inbox  —  data/inbox")
        self.inbox_radio.setChecked(True)
        self.inbox_radio.toggled.connect(self._on_source_changed)
        source_card.body.addWidget(self.inbox_radio)

        self.folder_radio = QRadioButton("Choose a folder…")
        self.folder_radio.toggled.connect(self._on_source_changed)
        source_card.body.addWidget(self.folder_radio)

        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._pick_folder)
        self.browse_button.setEnabled(False)
        self.folder_label = muted("No folder selected")
        source_card.body.addWidget(row(self.browse_button, self.folder_label, stretch=1))

        layout.addWidget(source_card)

        # --- run controls -----------------------------------------------------
        self.run_button = QPushButton("Start Ingestion")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._start)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)

        self.status_label = QLabel("Ready.")
        layout.addWidget(row(self.run_button, self.cancel_button, self.status_label,
                              stretch=2))

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        # --- per-document results --------------------------------------------
        self.results = QTreeWidget()
        self.results.setHeaderLabels(["#", "Document", "Outcome", "Detail"])
        self.results.setRootIsDecorated(False)
        self.results.setAlternatingRowColors(True)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.results.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.results.setColumnWidth(1, 260)
        layout.addWidget(self.results, 1)

        self._refresh_inbox_label()

    # ------------------------------------------------------------------ source
    def _on_source_changed(self):
        use_folder = self.folder_radio.isChecked()
        self.browse_button.setEnabled(use_folder)
        self.folder_label.setEnabled(use_folder)
        self._refresh_inbox_label()

    def _refresh_inbox_label(self):
        if self.inbox_radio.isChecked():
            count = len(self._files_in(self._inbox_path()))
            self.folder_label.setText(f"{count} document(s) waiting in the inbox"
                                      if count else "Inbox is empty")

    @staticmethod
    def _inbox_path() -> Path:
        from docclassify.config import CONFIG
        return Path(CONFIG["storage"]["inbox_dir"])

    @staticmethod
    def _files_in(folder: Path) -> list[Path]:
        from docclassify.pipeline import DEFAULT_EXTENSIONS
        if not folder.is_dir():
            return []
        return [p for p in folder.rglob("*") if p.suffix.lower() in DEFAULT_EXTENSIONS]

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of documents")
        if folder:
            self.selected_folder = folder
            count = len(self._files_in(Path(folder)))
            self.folder_label.setText(f"{folder}   ({count} document(s))")

    def _resolve_source(self) -> Path | None:
        if self.inbox_radio.isChecked():
            return self._inbox_path()
        if not self.selected_folder:
            show_error(self, "No folder selected",
                       "Choose a folder to ingest, or switch to the inbox.")
            return None
        return Path(self.selected_folder)

    # --------------------------------------------------------------------- run
    def _start(self):
        folder = self._resolve_source()
        if folder is None:
            return
        if not folder.is_dir():
            show_error(self, "Folder not found", f"{folder} does not exist.")
            return

        files = self._files_in(folder)
        if not files:
            show_error(self, "Nothing to ingest",
                       f"No supported documents found under {folder}.",
                       "Supported extensions: .pdf .docx .pptx .html .htm .txt .md")
            return

        batch = datetime.now(timezone.utc).strftime("%Y-%m")
        if not confirm(self, "Start ingestion",
                        f"Ingest {len(files)} document(s) from:\n{folder}\n\n"
                        f"They will be tagged with batch '{batch}'."):
            return

        self.results.clear()
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self.status_label.setText(f"Ingesting 0/{len(files)}…")
        self._set_running(True)

        self.worker = IngestWorker(str(folder), batch)
        self.worker.document_done.connect(self._on_document_done)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling after the current document…")

    def _set_running(self, running: bool):
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.inbox_radio.setEnabled(not running)
        self.folder_radio.setEnabled(not running)
        self.browse_button.setEnabled(not running and self.folder_radio.isChecked())

    # ----------------------------------------------------------------- signals
    def _on_document_done(self, done: int, total: int, filename: str,
                           status: str, detail: str):
        item = QTreeWidgetItem([str(done), filename, status, detail])
        item.setForeground(2, status_colour(status))
        self.results.addTopLevelItem(item)
        self.results.scrollToItem(item)
        self.progress.setValue(done)
        self.status_label.setText(f"Ingesting {done}/{total}…")

    def _on_finished(self, counts: dict):
        self._set_running(False)
        self.progress.setValue(self.progress.maximum())
        verb = "Cancelled" if counts.get("cancelled") else "Finished"
        self.status_label.setText(
            f"{verb} in {format_duration(counts.get('seconds', 0.0))}  ·  "
            f"{counts.get('ingested', 0)} ingested, "
            f"{counts.get('duplicate', 0)} duplicate, "
            f"{counts.get('failed', 0)} failed  ·  batch {counts.get('batch', '?')}"
        )
        self.stats_changed.emit()

    def _on_failed(self, message: str, detail: str):
        self._set_running(False)
        self.status_label.setText("Ingestion failed.")
        show_error(self, "Ingestion failed", message, detail)

    # ------------------------------------------------------------------- hooks
    def on_shown(self):
        """Called by MainWindow when this view becomes visible."""
        if self.worker is None or not self.worker.isRunning():
            self._refresh_inbox_label()

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def stop_work(self):
        """Asks a running ingestion to stop, and waits briefly. Used on app close."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(30_000)
