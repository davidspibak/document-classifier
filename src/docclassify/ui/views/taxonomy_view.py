"""
UI for browsing the fixed taxonomy tree and resolving the human-review queue
(documents the embedding classifier and LLM tie-breaker both failed to
confidently place).
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QLabel, QPushButton, QComboBox, QTextEdit
)


class TaxonomyView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        # --- left: taxonomy tree ---
        tree_col = QVBoxLayout()
        tree_col.addWidget(QLabel("<h2>Taxonomy</h2>"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Category", "Description"])
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_all)
        tree_col.addWidget(refresh_button)
        tree_col.addWidget(self.tree)
        layout.addLayout(tree_col, stretch=2)

        # --- right: review queue ---
        review_col = QVBoxLayout()
        review_col.addWidget(QLabel("<h2>Review Queue</h2>"))
        self.review_list = QListWidget()
        self.review_list.currentRowChanged.connect(self._on_review_selected)
        review_col.addWidget(self.review_list)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Select a queued document to see why it needs review.")
        self.preview.setMaximumHeight(220)
        review_col.addWidget(self.preview)

        self.category_picker = QComboBox()
        review_col.addWidget(self.category_picker)
        resolve_button = QPushButton("Assign Selected Category")
        resolve_button.clicked.connect(self._resolve_selected)
        review_col.addWidget(resolve_button)
        layout.addLayout(review_col, stretch=1)

        self._pending_items = []
        self._category_paths: dict[str, str] = {}
        self._refresh_all()

    def _refresh_all(self):
        self.reload_taxonomy()
        self.reload_review_queue()

    def reload_taxonomy(self):
        from docclassify.taxonomy.taxonomy_store import build_category_paths, get_full_tree

        self.tree.clear()
        nodes = get_full_tree()
        self._category_paths = build_category_paths(nodes)
        item_by_id = {}

        # two passes: create all QTreeWidgetItems, then attach children to parents
        for n in nodes:
            item_by_id[n["category_id"]] = QTreeWidgetItem([n["name"], n["description"]])
        for n in nodes:
            item = item_by_id[n["category_id"]]
            parent_id = n["parent_id"]
            if parent_id and parent_id in item_by_id:
                item_by_id[parent_id].addChild(item)
            else:
                self.tree.addTopLevelItem(item)

        self.tree.expandAll()

        # Populate the picker with FULL paths, which is both what the user needs to
        # see to tell two same-named leaves apart and what gets stored on the
        # document. userData carries the id so the path can be re-resolved.
        self.category_picker.clear()
        for n in sorted(nodes, key=lambda n: self._category_paths.get(n["category_id"], n["name"])):
            self.category_picker.addItem(self._category_paths[n["category_id"]], userData=n["category_id"])

    def reload_review_queue(self):
        from docclassify.classification.review_queue import list_pending
        self.review_list.clear()
        self.preview.clear()
        self._pending_items = list_pending()
        for item in self._pending_items:
            self.review_list.addItem(QListWidgetItem(f"{item['doc_id']} - {item['reason']}"))

    def _on_review_selected(self, row: int):
        """Shows what is known about the selected document and why it was flagged."""
        if row < 0 or row >= len(self._pending_items):
            self.preview.clear()
            return

        from docclassify.metadata.extract import decode_list_field
        from docclassify.storage import sqlite_store

        queue_item = self._pending_items[row]
        doc = sqlite_store.get_document(queue_item["doc_id"]) or {}

        lines = [
            f"Flagged for: {queue_item.get('reason') or 'unknown'}",
            f"File:        {doc.get('filename') or '(document row missing)'}",
            f"Language:    {doc.get('language') or 'unknown'}",
            f"Best guess:  {doc.get('category_path') or '(none)'}"
            f"  (confidence {doc.get('confidence') or 0:.3f})",
        ]

        title = doc.get("title_en") or doc.get("title_zh")
        if title:
            lines.append(f"Title:       {title}")
        keywords = decode_list_field(doc.get("keywords_en") or doc.get("keywords_zh"))
        if keywords:
            lines.append(f"Keywords:    {', '.join(keywords)}")

        candidate_ids = decode_list_field(queue_item.get("candidate_categories"))
        if candidate_ids:
            lines.append("")
            lines.append("Candidates the classifier considered:")
            lines.extend(
                f"  - {self._category_paths.get(cid, f'(unknown category {cid})')}"
                for cid in candidate_ids
            )

        self.preview.setPlainText("\n".join(lines))

    def _resolve_selected(self):
        row = self.review_list.currentRow()
        if row < 0 or row >= len(self._pending_items):
            return
        doc_id = self._pending_items[row]["doc_id"]
        category_id = self.category_picker.currentData()
        if category_id is None:
            self.preview.setPlainText("No categories exist yet - build the taxonomy first.")
            return

        # The full path, not the bare name: that's what documents.category_path
        # holds and what search's category filter matches against.
        category_path = self._category_paths.get(category_id) or self.category_picker.currentText()

        from docclassify.classification.review_queue import resolve
        resolve(doc_id, category_path)
        self.reload_review_queue()
