"""
Small shared widgets and helpers used by more than one view.

Exists so the four views look and behave the same way: same page header, same
error dialog, same way of turning a status string into a coloured pill.
"""
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from docclassify.ui.style import STATUS_COLOURS, TEXT_MUTED


class ViewHeader(QWidget):
    """Title plus one line explaining what the view is for."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("viewTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("viewSubtitle")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)


class SectionTitle(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setObjectName("sectionTitle")


class Card(QFrame):
    """A bordered panel. Use `body` as the layout to put content into."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 8):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(12, 12, 12, 12)
        self.body.setSpacing(spacing)


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


def row(*widgets, spacing: int = 8, stretch: int | None = None) -> QWidget:
    """
    Horizontal strip of widgets with consistent spacing.

    Pass None in place of a widget to insert a spacer. `stretch` is the index of the
    widget that should absorb leftover width (0 for a search box that grows, for
    instance) — expressed here rather than by reaching into the layout afterwards.
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for index, widget in enumerate(widgets):
        if widget is None:
            layout.addStretch(1)
            continue
        layout.addWidget(widget, 1 if index == stretch else 0)
    return container


def status_colour(status: str | None) -> QColor:
    return QColor(STATUS_COLOURS.get((status or "").strip(), TEXT_MUTED))


def show_error(parent: QWidget, title: str, message: str, detail: str = "") -> None:
    """
    A real dialog rather than text appended to a log. A failure the user needs to act
    on should not be something they have to notice.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setDetailedText(detail)
    box.exec()


def confirm(parent: QWidget, title: str, message: str) -> bool:
    answer = QMessageBox.question(
        parent, title, message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def elide(text: str, limit: int = 160) -> str:
    """Single-line, whitespace-collapsed preview of a chunk of document text."""
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "…"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


__all__ = [
    "Card", "SectionTitle", "ViewHeader", "confirm", "elide", "format_duration",
    "muted", "row", "show_error", "status_colour",
]
