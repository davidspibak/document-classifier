"""
Main window: sidebar navigation over a stack of four views.

What was wrong before, and what this fixes:
  * No menu bar, status bar, shortcuts or About box — the window offered nothing
    beyond four unlabelled navigation rows.
  * Every view was constructed eagerly, and TaxonomyView read the database inside
    __init__, so startup did database work for views the user might never open. Views
    now load their data the first time they are shown, via an on_shown() hook.
  * Closing the window while ingestion was running left a QThread mid-flight.
    closeEvent now asks running work to stop and confirms with the user first.
  * Styling was ad-hoc <h2> tags inside labels. There is now one stylesheet
    (ui/style.py) and a shared page header widget.
"""
import sys

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QStackedWidget, QStyleFactory, QWidget,
)

from docclassify.storage import sqlite_store
from docclassify.ui.style import STYLESHEET
from docclassify.ui.views.ingest_view import IngestView
from docclassify.ui.views.report_view import ReportView
from docclassify.ui.views.search_view import SearchView
from docclassify.ui.views.taxonomy_view import TaxonomyView

APP_TITLE = "Document Auto-Classification & Semantic Search"

VIEW_SPECS = [
    ("Ingest", "Parse, classify and index documents"),
    ("Search", "Multilingual semantic search"),
    ("Taxonomy", "Browse categories and review queue"),
    ("Reports", "Monthly digest and document summaries"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 840)
        self.setMinimumSize(980, 620)

        self.views = [IngestView(), SearchView(), TaxonomyView(), ReportView()]
        self._shown_once = [False] * len(self.views)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setFixedWidth(210)
        for index, (label, tooltip) in enumerate(VIEW_SPECS):
            item = QListWidgetItem(f"{index + 1}.  {label}")
            item.setToolTip(tooltip)
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav_list)

        self.stack = QStackedWidget()
        for view in self.views:
            self.stack.addWidget(view)
        layout.addWidget(self.stack, 1)

        self.setCentralWidget(central)

        self._build_menus()
        self._build_status_bar()

        # Views that change the corpus ask the status bar to refresh.
        for view in self.views:
            signal = getattr(view, "stats_changed", None)
            if signal is not None:
                signal.connect(self.refresh_stats)

        self.nav_list.setCurrentRow(0)

    # ------------------------------------------------------------------- chrome
    def _build_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        refresh_action = QAction("&Refresh current view", self)
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_action.triggered.connect(self._refresh_current_view)
        file_menu.addAction(refresh_action)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu("&View")
        for index, (label, _tooltip) in enumerate(VIEW_SPECS):
            action = QAction(f"&{label}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{index + 1}"))
            # default=index binds the current value rather than closing over the loop
            # variable, which would make every action open the last view.
            action.triggered.connect(lambda _checked=False, target=index:
                                      self.nav_list.setCurrentRow(target))
            view_menu.addAction(action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_status_bar(self):
        self.stats_label = QLabel("")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.refresh_stats()

    def refresh_stats(self):
        try:
            stats = sqlite_store.corpus_stats()
        except Exception as e:  # noqa: BLE001 - the status bar must never break the app
            self.stats_label.setText(f"database unavailable: {e}")
            return
        self.stats_label.setText(
            f"{stats['documents']} documents   ·   {stats['categories']} categories   ·   "
            f"{stats['batches']} batches   ·   {stats['pending_review']} awaiting review"
        )

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_TITLE}",
            "<b>docclassify</b> v0.1.0<br><br>"
            "Fully offline hierarchical document classification and multilingual "
            "semantic search.<br><br>"
            "Classification: BGE-M3 embedding similarity against a fixed taxonomy, with "
            "Qwen2.5-7B as a constrained tie-breaker.<br>"
            "Search: BGE-M3 recall then bge-reranker-v2-m3 precision reranking.<br><br>"
            "No network access at runtime — verify with "
            "<code>scripts/audit_offline.py</code>.",
        )

    # -------------------------------------------------------------- navigation
    def _on_nav_changed(self, index: int):
        if index < 0 or index >= len(self.views):
            return
        self.stack.setCurrentIndex(index)
        self.statusBar().showMessage(VIEW_SPECS[index][1], 4000)

        view = self.views[index]
        hook = getattr(view, "on_shown", None)
        if hook is None:
            return
        # First display always loads; later displays refresh only when idle, so
        # switching away and back cannot disturb a running job.
        if not self._shown_once[index]:
            self._shown_once[index] = True
            hook()
        elif not self._view_busy(view):
            hook()

    def _refresh_current_view(self):
        index = self.stack.currentIndex()
        view = self.views[index]
        if self._view_busy(view):
            self.statusBar().showMessage("Busy — not refreshing while work is running.", 4000)
            return
        hook = getattr(view, "on_shown", None)
        if hook is not None:
            hook()
        self.refresh_stats()
        self.statusBar().showMessage("Refreshed.", 2000)

    @staticmethod
    def _view_busy(view) -> bool:
        checker = getattr(view, "is_busy", None)
        return bool(checker()) if checker is not None else False

    # ------------------------------------------------------------------ closing
    def closeEvent(self, event):
        busy = [VIEW_SPECS[i][0] for i, view in enumerate(self.views)
                if self._view_busy(view)]
        if busy:
            answer = QMessageBox.question(
                self, "Work in progress",
                f"Still running: {', '.join(busy)}.\n\n"
                "Quit anyway? Documents already processed stay saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        for view in self.views:
            stopper = getattr(view, "stop_work", None)
            if stopper is not None:
                try:
                    stopper()
                except Exception as e:  # noqa: BLE001 - never block a quit
                    print(f"[main_window] error stopping a worker: {e}")
        event.accept()


def main():
    sqlite_store.init_db()  # ensure tables exist before any view queries them

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    # Fusion so the app looks identical across Windows versions instead of inheriting
    # native control metrics that break the stylesheet's spacing.
    if "Fusion" in QStyleFactory.keys():
        app.setStyle(QStyleFactory.create("Fusion"))
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
