"""
PySide6 main window: a sidebar for navigation + a QStackedWidget holding the
four views (ingest, search, taxonomy, reports). This is the entry point the
packaged .exe launches.
"""
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QStackedWidget, QListWidgetItem,
)

from docclassify.storage import sqlite_store
from docclassify.ui.views.ingest_view import IngestView
from docclassify.ui.views.search_view import SearchView
from docclassify.ui.views.taxonomy_view import TaxonomyView
from docclassify.ui.views.report_view import ReportView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Auto-Classification & Semantic Search")
        self.resize(1200, 800)

        central = QWidget()
        layout = QHBoxLayout(central)

        # --- sidebar navigation ---
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(180)
        for label in ["Ingest", "Search", "Taxonomy", "Reports"]:
            self.nav_list.addItem(QListWidgetItem(label))
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav_list)

        # --- stacked views ---
        self.stack = QStackedWidget()
        self.ingest_view = IngestView()
        self.search_view = SearchView()
        self.taxonomy_view = TaxonomyView()
        self.report_view = ReportView()
        for view in (self.ingest_view, self.search_view, self.taxonomy_view, self.report_view):
            self.stack.addWidget(view)
        layout.addWidget(self.stack)

        self.setCentralWidget(central)
        self.nav_list.setCurrentRow(0)

    def _on_nav_changed(self, index: int):
        self.stack.setCurrentIndex(index)


def main():
    sqlite_store.init_db()  # ensure tables exist before any view queries them
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
