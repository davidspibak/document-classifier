"""
UI for browsing the fixed taxonomy tree and resolving the human-review queue
(documents the embedding classifier and LLM tie-breaker both failed to
confidently place).
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QLabel, QPushButton, QComboBox
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
        refresh_button.clicked.connect(self.reload_taxonomy)
        tree_col.addWidget(refresh_button)
        tree_col.addWidget(self.tree)
        layout.addLayout(tree_col, stretch=2)

        # --- right: review queue ---
        review_col = QVBoxLayout()
        review_col.addWidget(QLabel("<h2>Review Queue</h2>"))
        self.review_list = QListWidget()
        self.review_list.currentRowChanged.connect(self._on_review_selected)
        review_col.addWidget(self.review_list)

        self.category_picker = QComboBox()
        review_col.addWidget(self.category_picker)
        resolve_button = QPushButton("Assign Selected Category")
        resolve_button.clicked.connect(self._resolve_selected)
        review_col.addWidget(resolve_button)
        layout.addLayout(review_col, stretch=1)

        self._pending_items = []
        self.reload_taxonomy()
        self.reload_review_queue()

    def reload_taxonomy(self):
        from docclassify.taxonomy.taxonomy_store import get_full_tree
        self.tree.clear()
        nodes = get_full_tree()
        by_id = {n["category_id"]: n for n in nodes}
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

        # populate the category picker (leaf-ish flat list for manual assignment)
        self.category_picker.clear()
        for n in nodes:
            self.category_picker.addItem(n["name"], userData=n["category_id"])

    def reload_review_queue(self):
        from docclassify.classification.review_queue import list_pending
        self.review_list.clear()
        self._pending_items = list_pending()
        for item in self._pending_items:
            self.review_list.addItem(QListWidgetItem(f"{item['doc_id']} - {item['reason']}"))

    def _on_review_selected(self, row: int):
        pass  # extension point: show document preview/snippet for the selected item

    def _resolve_selected(self):
        row = self.review_list.currentRow()
        if row < 0 or row >= len(self._pending_items):
            return
        doc_id = self._pending_items[row]["doc_id"]
        category_name = self.category_picker.currentText()

        from docclassify.classification.review_queue import resolve
        resolve(doc_id, category_name)
        self.reload_review_queue()
