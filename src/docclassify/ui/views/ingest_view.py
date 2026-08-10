"""
UI for picking a folder, running ingestion + classification against it, and
showing progress. Runs the pipeline on a QThread so the GUI stays responsive
during what can be a long-running, GPU-bound operation.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog
)
from PySide6.QtCore import QThread, Signal


class IngestWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(int)
    failed = Signal(str)

    def __init__(self, folder_path: str):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            from docclassify.pipeline import process_folder
            from datetime import datetime, timezone
            batch = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            self.progress.emit(f"Starting ingestion of {self.folder_path} (batch {batch}) ...")
            results = process_folder(self.folder_path, upload_batch=batch)
            self.finished_ok.emit(len(results))
        except Exception as e:  # noqa: BLE001 - surfaced to the UI rather than crashing the app
            self.failed.emit(str(e))


class IngestView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h2>Ingest & Classify Documents</h2>"))

        folder_row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        pick_button = QPushButton("Choose Folder...")
        pick_button.clicked.connect(self._pick_folder)
        folder_row.addWidget(self.folder_label)
        folder_row.addWidget(pick_button)
        layout.addLayout(folder_row)

        self.run_button = QPushButton("Run Ingestion")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_ingestion)
        layout.addWidget(self.run_button)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.selected_folder = None
        self.worker = None

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select document folder")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(folder)
            self.run_button.setEnabled(True)

    def _run_ingestion(self):
        if not self.selected_folder:
            return
        self.run_button.setEnabled(False)
        self.log.append(f"Ingesting from: {self.selected_folder}")

        self.worker = IngestWorker(self.selected_folder)
        self.worker.progress.connect(self.log.append)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, count: int):
        self.log.append(f"Done. {count} documents processed.")
        self.run_button.setEnabled(True)

    def _on_failed(self, error: str):
        self.log.append(f"ERROR: {error}")
        self.run_button.setEnabled(True)
