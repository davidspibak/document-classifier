"""
Semantic search UI: query box + ranked result list with snippets. Search runs
on a QThread since embedding + ANN search + reranking, while fast, still
involves model inference that shouldn't block the GUI thread.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem
)
from PySide6.QtCore import QThread, Signal, Qt


class SearchWorker(QThread):
    results_ready = Signal(list)
    failed = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        try:
            from docclassify.search.reranker import search
            results = search(self.query)
            self.results_ready.emit(results)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class SearchView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h2>Semantic Search</h2>"))

        query_row = QHBoxLayout()
        self.query_box = QLineEdit()
        self.query_box.setPlaceholderText("Search in any language...")
        self.query_box.returnPressed.connect(self._run_search)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._run_search)
        query_row.addWidget(self.query_box)
        query_row.addWidget(search_button)
        layout.addLayout(query_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_list = QListWidget()
        self.results_list.setWordWrap(True)
        layout.addWidget(self.results_list)

        self.worker = None

    def _run_search(self):
        query = self.query_box.text().strip()
        if not query:
            return
        self.status_label.setText("Searching...")
        self.results_list.clear()

        self.worker = SearchWorker(query)
        self.worker.results_ready.connect(self._on_results)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_results(self, results: list[dict]):
        self.status_label.setText(f"{len(results)} results")
        for r in results:
            snippet = r["chunk_text"][:300].replace("\n", " ")
            score = r.get("rerank_score", 0.0)
            item_text = f"[{score:.3f}] {r['doc_id']} - {snippet}..."
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, r["doc_id"])
            self.results_list.addItem(item)

    def _on_failed(self, error: str):
        self.status_label.setText(f"ERROR: {error}")
