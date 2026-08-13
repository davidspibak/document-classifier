"""
Taxonomy view: browse the fixed taxonomy and work the human-review queue.

What was wrong before, and what this fixes:
  * The assignment dropdown listed every category including parents, so a document
    could be filed under "Economics" rather than one of its fields. It now offers
    leaves by default, with an explicit opt-in to show every node.
  * The tree showed no document counts, so the one thing you actually want from a
    taxonomy browser — how the corpus is distributed — was invisible.
  * The review preview listed metadata but not the document itself. It now shows the
    reassembled text, so a human can judge the classification.
  * There was no way to dismiss a queue item that needed no action.
"""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHeaderView, QListWidget,
    QListWidgetItem, QPushButton, QSplitter, QTabWidget, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from docclassify.ui.widgets import (
    Card, SectionTitle, ViewHeader, confirm, muted, row, show_error,
    status_colour,
)


class DocumentTextWorker(QThread):
    """Reassembling a document from its chunks touches LanceDB, so keep it off the GUI thread."""

    loaded = Signal(str, str)   # doc_id, text
    failed = Signal(str, str)   # doc_id, message

    def __init__(self, doc_id: str):
        super().__init__()
        self.doc_id = doc_id

    def run(self):
        try:
            from docclassify.storage import lancedb_store
            self.loaded.emit(self.doc_id, lancedb_store.get_document_text(self.doc_id))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self.doc_id, f"{type(e).__name__}: {e}")


class TaxonomyView(QWidget):
    stats_changed = Signal()

    def __init__(self):
        super().__init__()
        self._pending_items: list[dict] = []
        self._category_paths: dict[str, str] = {}
        self._leaf_ids: set[str] = set()
        self._text_worker: DocumentTextWorker | None = None
        self._loaded_once = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addWidget(ViewHeader(
            "Taxonomy & Review",
            "The fixed category tree on the left; on the right, documents the classifier "
            "and the LLM tie-breaker could not confidently place.",
        ))

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------------------------------------------------- left: tree
        tree_card = Card()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload_all)
        tree_card.body.addWidget(row(SectionTitle("Categories"), None, self.refresh_button))

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Category", "Docs", "Description"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.setColumnWidth(0, 240)
        tree_card.body.addWidget(self.tree)

        self.tree_summary = muted("")
        tree_card.body.addWidget(self.tree_summary)
        splitter.addWidget(tree_card)

        # -------------------------------------------------------- right: queue
        queue_card = Card()
        queue_card.body.addWidget(SectionTitle("Review Queue"))

        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(150)
        self.queue_list.currentRowChanged.connect(self._on_queue_selection)
        queue_card.body.addWidget(self.queue_list)

        self.detail_tabs = QTabWidget()

        self.why_panel = QTextEdit()
        self.why_panel.setReadOnly(True)
        self.why_panel.setPlaceholderText("Select a queued document.")
        self.detail_tabs.addTab(self.why_panel, "Why flagged")

        self.text_panel = QTextEdit()
        self.text_panel.setReadOnly(True)
        self.text_panel.setPlaceholderText("Document text appears here.")
        self.detail_tabs.addTab(self.text_panel, "Document text")

        queue_card.body.addWidget(self.detail_tabs, 1)

        self.leaves_only = QCheckBox("Assign to leaf categories only")
        self.leaves_only.setChecked(True)
        self.leaves_only.toggled.connect(self._populate_picker)
        queue_card.body.addWidget(self.leaves_only)

        self.category_picker = QComboBox()
        queue_card.body.addWidget(self.category_picker)

        self.assign_button = QPushButton("Assign Category")
        self.assign_button.setObjectName("primary")
        self.assign_button.clicked.connect(self._assign_selected)

        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.setToolTip(
            "Mark this item resolved without changing its category — for a document "
            "the classifier actually placed correctly."
        )
        self.dismiss_button.clicked.connect(self._dismiss_selected)
        queue_card.body.addWidget(row(self.assign_button, self.dismiss_button, None))

        self.queue_status = muted("")
        queue_card.body.addWidget(self.queue_status)
        splitter.addWidget(queue_card)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

    # -------------------------------------------------------------------- load
    def reload_all(self):
        self.reload_taxonomy()
        self.reload_review_queue()
        self._loaded_once = True

    def reload_taxonomy(self):
        from docclassify.storage import sqlite_store
        from docclassify.taxonomy.taxonomy_store import build_category_paths, get_full_tree

        self.tree.clear()
        try:
            nodes = get_full_tree()
            counts = sqlite_store.category_counts()
        except Exception as e:  # noqa: BLE001
            self.tree_summary.setText(f"Could not read the taxonomy: {e}")
            return

        self._category_paths = build_category_paths(nodes)
        parent_ids = {n["parent_id"] for n in nodes if n.get("parent_id")}
        self._leaf_ids = {n["category_id"] for n in nodes
                          if n["category_id"] not in parent_ids}

        # Documents are counted against their exact path, so a parent's total is the
        # sum of everything filed at or beneath it.
        def documents_under(path: str) -> int:
            return sum(n for stored, n in counts.items()
                       if stored == path or stored.startswith(path + "/"))

        items: dict[str, QTreeWidgetItem] = {}
        for node in nodes:
            path = self._category_paths.get(node["category_id"], node["name"])
            item = QTreeWidgetItem([node["name"], str(documents_under(path)),
                                     node.get("description") or ""])
            item.setData(0, Qt.ItemDataRole.UserRole, node["category_id"])
            item.setToolTip(0, path)
            item.setToolTip(2, node.get("description") or "")
            items[node["category_id"]] = item

        for node in nodes:
            item = items[node["category_id"]]
            parent_id = node.get("parent_id")
            if parent_id and parent_id in items:
                items[parent_id].addChild(item)
            else:
                self.tree.addTopLevelItem(item)

        self.tree.expandAll()

        unassigned = counts.get("", 0) + counts.get(None, 0) if counts else 0
        total_documents = sum(counts.values()) if counts else 0
        self.tree_summary.setText(
            f"{len(nodes)} categories · {len(self._leaf_ids)} leaves · "
            f"{total_documents} documents ({unassigned} unassigned)"
            if nodes else
            "No taxonomy yet. Import one with scripts/import_taxonomy.py, or build one "
            "with scripts/build_taxonomy.py."
        )
        self._populate_picker()

    def _populate_picker(self):
        selectable = [
            (category_id, path) for category_id, path in sorted(
                self._category_paths.items(), key=lambda kv: kv[1]
            )
            if (not self.leaves_only.isChecked()) or category_id in self._leaf_ids
        ]

        previous = self.category_picker.currentData()
        self.category_picker.blockSignals(True)
        self.category_picker.clear()
        for category_id, path in selectable:
            self.category_picker.addItem(path, category_id)
        if previous is not None:
            index = self.category_picker.findData(previous)
            if index >= 0:
                self.category_picker.setCurrentIndex(index)
        self.category_picker.blockSignals(False)

        has_options = self.category_picker.count() > 0
        self.assign_button.setEnabled(has_options and bool(self._pending_items))
        if not has_options:
            self.category_picker.addItem("(no categories available)", None)

    def reload_review_queue(self):
        from docclassify.classification.review_queue import list_pending

        self.queue_list.clear()
        self.why_panel.clear()
        self.text_panel.clear()
        try:
            self._pending_items = list_pending()
        except Exception as e:  # noqa: BLE001
            self._pending_items = []
            self.queue_status.setText(f"Could not read the review queue: {e}")
            return

        from docclassify.storage import sqlite_store
        for entry in self._pending_items:
            document = sqlite_store.get_document(entry["doc_id"]) or {}
            label = document.get("filename") or entry["doc_id"]
            item = QListWidgetItem(f"{label}   —   {entry.get('reason') or 'unknown'}")
            item.setForeground(status_colour("needs_review"))
            self.queue_list.addItem(item)

        self.queue_status.setText(
            f"{len(self._pending_items)} document(s) awaiting review"
            if self._pending_items else "Review queue is empty."
        )
        has_items = bool(self._pending_items)
        self.dismiss_button.setEnabled(has_items)
        self.assign_button.setEnabled(has_items and self.category_picker.currentData() is not None)

    # --------------------------------------------------------------- selection
    def _selected_entry(self) -> dict | None:
        row_index = self.queue_list.currentRow()
        if row_index < 0 or row_index >= len(self._pending_items):
            return None
        return self._pending_items[row_index]

    def _on_queue_selection(self, row_index: int):
        entry = self._selected_entry()
        if entry is None:
            self.why_panel.clear()
            self.text_panel.clear()
            return

        from docclassify.metadata.extract import decode_list_field
        from docclassify.storage import sqlite_store

        document = sqlite_store.get_document(entry["doc_id"]) or {}
        confidence = document.get("confidence")
        lines = [
            f"Flagged for:  {entry.get('reason') or 'unknown'}",
            f"File:         {document.get('filename') or '(document row missing)'}",
            f"Language:     {document.get('language') or 'unknown'}",
            f"Best guess:   {document.get('category_path') or '(none)'}"
            + (f"   (confidence {confidence:.3f})" if confidence is not None else ""),
        ]
        title = document.get("title_en") or document.get("title_zh")
        if title:
            lines.append(f"Title:        {title}")
        keywords = decode_list_field(document.get("keywords_en") or document.get("keywords_zh"))
        if keywords:
            lines.append(f"Keywords:     {', '.join(keywords)}")

        candidate_ids = decode_list_field(entry.get("candidate_categories"))
        if candidate_ids:
            lines += ["", "Candidates the classifier considered:"]
            lines += [f"  · {self._category_paths.get(cid, f'(unknown category {cid})')}"
                      for cid in candidate_ids]
        self.why_panel.setPlainText("\n".join(lines))

        # Preselect the classifier's own first candidate — usually the right answer,
        # and it saves scrolling a long dropdown.
        for cid in candidate_ids:
            index = self.category_picker.findData(cid)
            if index >= 0:
                self.category_picker.setCurrentIndex(index)
                break

        self._load_document_text(entry["doc_id"])

    def _load_document_text(self, doc_id: str):
        if self._text_worker is not None and self._text_worker.isRunning():
            self._text_worker.wait(5_000)
        self.text_panel.setPlainText("Loading document text…")
        self._text_worker = DocumentTextWorker(doc_id)
        self._text_worker.loaded.connect(self._on_text_loaded)
        self._text_worker.failed.connect(self._on_text_failed)
        self._text_worker.start()

    def _on_text_loaded(self, doc_id: str, text: str):
        entry = self._selected_entry()
        if entry is None or entry["doc_id"] != doc_id:
            return  # selection moved on while we were loading
        self.text_panel.setPlainText(
            text or "(no text stored for this document — nothing was indexed)"
        )

    def _on_text_failed(self, doc_id: str, message: str):
        entry = self._selected_entry()
        if entry is not None and entry["doc_id"] == doc_id:
            self.text_panel.setPlainText(f"Could not reassemble the document text.\n{message}")

    # ------------------------------------------------------------------ action
    def _assign_selected(self):
        entry = self._selected_entry()
        if entry is None:
            show_error(self, "Nothing selected", "Select a document in the review queue first.")
            return

        category_id = self.category_picker.currentData()
        if category_id is None:
            show_error(self, "No category selected",
                       "There are no categories to assign to. Import a taxonomy first.")
            return

        category_path = self._category_paths.get(category_id)
        if not category_path:
            show_error(self, "Category not found",
                       "That category no longer exists. Refresh and try again.")
            return

        from docclassify.classification.review_queue import resolve
        try:
            resolve(entry["doc_id"], category_path)
        except Exception as e:  # noqa: BLE001
            import traceback
            show_error(self, "Could not assign the category",
                       f"{type(e).__name__}: {e}", traceback.format_exc())
            return

        self.queue_status.setText(f"Assigned to {category_path}.")
        self.reload_review_queue()
        self.reload_taxonomy()
        self.stats_changed.emit()

    def _dismiss_selected(self):
        entry = self._selected_entry()
        if entry is None:
            show_error(self, "Nothing selected", "Select a document in the review queue first.")
            return
        if not confirm(self, "Dismiss review item",
                        "Mark this document reviewed without changing its category?"):
            return

        from docclassify.storage import sqlite_store
        try:
            sqlite_store.resolve_review_item(entry["doc_id"])
        except Exception as e:  # noqa: BLE001
            show_error(self, "Could not dismiss the item", f"{type(e).__name__}: {e}")
            return

        self.queue_status.setText("Dismissed without reclassifying.")
        self.reload_review_queue()
        self.stats_changed.emit()

    # ------------------------------------------------------------------- hooks
    def on_shown(self):
        # Loaded on first display rather than in __init__, so opening the app does not
        # block on database reads for a view the user may never visit.
        self.reload_all()

    def is_busy(self) -> bool:
        return self._text_worker is not None and self._text_worker.isRunning()

    def stop_work(self):
        if self._text_worker is not None and self._text_worker.isRunning():
            self._text_worker.wait(5_000)
