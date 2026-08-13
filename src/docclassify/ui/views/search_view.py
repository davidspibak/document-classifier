"""
Search view: two-stage semantic search with filters and a result detail pane.

What was wrong before, and what this fixes:
  * Results were labelled with the raw doc_id (a UUID), which is unreadable. They now
    show filename and title, resolved from SQLite.
  * The doc_id was stashed on each item with setData and then never read, so clicking
    a result did nothing. Selecting a result now shows the full chunk and the
    document's metadata.
  * search() accepts category and language filters and the UI exposed neither, so a
    documented capability of the search subsystem was unreachable.
  * Starting a second search replaced self.worker while the first QThread was still
    running, which risks "QThread: Destroyed while thread is still running". Searches
    are now serialised.
"""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QLineEdit, QPushButton,
    QSpinBox, QSplitter, QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from docclassify.ui.widgets import (
    Card, SectionTitle, ViewHeader, elide, format_duration, muted, row, show_error,
)

ANY_CATEGORY = "All categories"
ANY_LANGUAGE = "Any language"


class SearchWorker(QThread):
    results_ready = Signal(list, float)   # results, seconds
    failed = Signal(str, str)

    def __init__(self, query: str, top_k: int,
                  category: str | None, language: str | None):
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.category = category
        self.language = language

    def run(self):
        import time
        import traceback
        try:
            from docclassify.config import CONFIG
            from docclassify.search.ann_search import search_candidates
            from docclassify.search.query import embed_query
            from docclassify.search.reranker import rerank

            started = time.perf_counter()
            vector, _language = embed_query(self.query)
            candidates = search_candidates(
                vector,
                top_k=CONFIG["search"]["ann_candidate_count"],
                category_filter=self.category,
                language_filter=self.language,
            )
            results = rerank(self.query, candidates, top_n=self.top_k)
            self.results_ready.emit(results, time.perf_counter() - started)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


class SearchView(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: SearchWorker | None = None
        self._document_cache: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addWidget(ViewHeader(
            "Semantic Search",
            "Chunk-level search across the corpus. Queries and documents share one "
            "vector space, so a query in any language matches documents in any other.",
        ))

        # --- query and filters -------------------------------------------------
        query_card = Card()
        self.query_box = QLineEdit()
        self.query_box.setPlaceholderText("Search in any language…")
        self.query_box.returnPressed.connect(self._run_search)

        self.search_button = QPushButton("Search")
        self.search_button.setObjectName("primary")
        self.search_button.clicked.connect(self._run_search)
        query_card.body.addWidget(row(self.query_box, self.search_button, stretch=0))

        self.category_filter = QComboBox()
        self.category_filter.addItem(ANY_CATEGORY, None)
        self.category_filter.setMinimumWidth(220)

        self.language_filter = QComboBox()
        self.language_filter.addItem(ANY_LANGUAGE, None)
        self.language_filter.setMinimumWidth(140)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 50)
        self.top_k.setValue(10)
        self.top_k.setPrefix("top ")

        query_card.body.addWidget(row(
            muted("Filter:"), self.category_filter, self.language_filter,
            self.top_k, None,
        ))
        layout.addWidget(query_card)

        self.status_label = muted("Enter a query to search.")
        layout.addWidget(self.status_label)

        # --- results and detail ------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.results = QTreeWidget()
        self.results.setHeaderLabels(["Score", "Document", "Category", "Snippet"])
        self.results.setRootIsDecorated(False)
        self.results.setAlternatingRowColors(True)
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results.currentItemChanged.connect(self._on_selection_changed)
        header = self.results.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.results.setColumnWidth(1, 230)
        self.results.setColumnWidth(2, 190)
        splitter.addWidget(self.results)

        detail_card = Card()
        detail_card.body.addWidget(SectionTitle("Matching passage"))
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("Select a result to see the full passage and "
                                       "the document it came from.")
        detail_card.body.addWidget(self.detail)
        splitter.addWidget(detail_card)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------ filters
    def reload_filters(self):
        """
        Repopulates the filter dropdowns from what is actually in the corpus. Offering
        a category that no document carries would just produce empty results.
        """
        from docclassify.storage import sqlite_store

        try:
            with sqlite_store.get_connection() as conn:
                categories = [r[0] for r in conn.execute(
                    "SELECT DISTINCT category_path FROM documents "
                    "WHERE category_path IS NOT NULL AND category_path != '' "
                    "ORDER BY category_path"
                ).fetchall()]
                languages = [r[0] for r in conn.execute(
                    "SELECT DISTINCT language FROM documents "
                    "WHERE language IS NOT NULL AND language != '' ORDER BY language"
                ).fetchall()]
        except Exception:  # noqa: BLE001 - an empty database is not an error here
            categories, languages = [], []

        def repopulate(combo: QComboBox, any_label: str, values: list[str]):
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(any_label, None)
            for value in values:
                combo.addItem(value, value)
            if previous is not None:
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        repopulate(self.category_filter, ANY_CATEGORY, categories)
        repopulate(self.language_filter, ANY_LANGUAGE, languages)

    # --------------------------------------------------------------------- run
    def _run_search(self):
        query = self.query_box.text().strip()
        if not query:
            return
        if self.worker is not None and self.worker.isRunning():
            # Serialise rather than replacing a live QThread, which can crash Qt.
            self.status_label.setText("A search is already running…")
            return

        self.results.clear()
        self.detail.clear()
        self._document_cache.clear()
        self.status_label.setText("Searching…")
        self.search_button.setEnabled(False)

        self.worker = SearchWorker(
            query,
            self.top_k.value(),
            self.category_filter.currentData(),
            self.language_filter.currentData(),
        )
        self.worker.results_ready.connect(self._on_results)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _document_for(self, doc_id: str) -> dict:
        if doc_id not in self._document_cache:
            from docclassify.storage import sqlite_store
            self._document_cache[doc_id] = sqlite_store.get_document(doc_id) or {}
        return self._document_cache[doc_id]

    def _on_results(self, results: list, seconds: float):
        self.search_button.setEnabled(True)
        if not results:
            self.status_label.setText(
                f"No results in {format_duration(seconds)}. "
                "Is anything ingested, and do the filters match?"
            )
            return

        self.status_label.setText(
            f"{len(results)} result(s) in {format_duration(seconds)}"
        )
        for hit in results:
            doc_id = hit.get("doc_id", "")
            document = self._document_for(doc_id)
            label = (document.get("title_en") or document.get("title_zh")
                     or document.get("filename") or doc_id or "(unknown)")
            item = QTreeWidgetItem([
                f"{hit.get('rerank_score', 0.0):.4f}",
                str(label),
                hit.get("category_path") or document.get("category_path") or "(unassigned)",
                elide(hit.get("chunk_text", ""), 200),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, hit)
            item.setToolTip(1, f"{document.get('filename', '')}\ndoc_id: {doc_id}")
            self.results.addTopLevelItem(item)

        self.results.setCurrentItem(self.results.topLevelItem(0))

    def _on_selection_changed(self, current: QTreeWidgetItem, _previous):
        if current is None:
            self.detail.clear()
            return
        hit = current.data(0, Qt.ItemDataRole.UserRole) or {}
        document = self._document_for(hit.get("doc_id", ""))

        lines = [
            f"File:        {document.get('filename') or '(row missing)'}",
            f"Category:    {document.get('category_path') or '(unassigned)'}",
            f"Status:      {document.get('status') or '?'}",
            f"Language:    {hit.get('language') or document.get('language') or '?'}",
            f"Chunk:       #{hit.get('chunk_index', '?')}",
            f"Score:       {hit.get('rerank_score', 0.0):.4f}",
        ]
        title = document.get("title_en") or document.get("title_zh")
        if title:
            lines.insert(1, f"Title:       {title}")
        lines += ["", "--- passage ---", (hit.get("chunk_text") or "").strip()]
        self.detail.setPlainText("\n".join(lines))

    def _on_failed(self, message: str, detail: str):
        self.search_button.setEnabled(True)
        self.status_label.setText("Search failed.")
        show_error(self, "Search failed", message, detail)

    # ------------------------------------------------------------------- hooks
    def on_shown(self):
        self.reload_filters()
        self.query_box.setFocus()

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def stop_work(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(15_000)
