"""
One stylesheet for the whole application.

Kept in a single module rather than scattered across the views so spacing, colour
and typography stay consistent — the default Qt look with ad-hoc <h2> tags in
labels is what made the previous UI feel unfinished.

Deliberately a light theme with one accent colour. Qt's Fusion style is applied in
main_window so the app looks the same on every Windows version rather than
inheriting whatever the native theme does to control heights.
"""

ACCENT = "#1F6FB2"
ACCENT_DARK = "#175A93"
TEXT = "#1F2933"
TEXT_MUTED = "#6B7480"
BORDER = "#D6DBE1"
SURFACE = "#FFFFFF"
CANVAS = "#F4F6F8"
SIDEBAR = "#22303C"
SIDEBAR_TEXT = "#C7D2DC"

# Status colours, used for the classification-status pills in tables.
STATUS_COLOURS = {
    "auto_assigned": "#1E7B4F",
    "llm_assigned": "#8A6100",
    "human_assigned": "#1F6FB2",
    "needs_review": "#B3261E",
    "pending": "#6B7480",
    "duplicate": "#6B7480",
    "failed": "#B3261E",
    "ingested": "#1E7B4F",
}

MONO_FONT = "Consolas, 'Cascadia Mono', 'DejaVu Sans Mono', monospace"

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-size: 10pt;
}}
QMainWindow, QStackedWidget > QWidget {{
    background: {CANVAS};
}}

/* ---------- sidebar navigation ---------- */
QListWidget#navList {{
    background: {SIDEBAR};
    border: none;
    outline: none;
    padding-top: 8px;
}}
QListWidget#navList::item {{
    color: {SIDEBAR_TEXT};
    padding: 11px 16px;
    border: none;
}}
QListWidget#navList::item:selected {{
    background: {ACCENT};
    color: #FFFFFF;
}}
QListWidget#navList::item:hover:!selected {{
    background: #2C3E4E;
}}

/* ---------- headings ---------- */
QLabel#viewTitle {{
    font-size: 16pt;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#viewSubtitle {{
    color: {TEXT_MUTED};
    font-size: 9.5pt;
}}
QLabel#sectionTitle {{
    font-size: 10.5pt;
    font-weight: 600;
    color: {TEXT};
}}
QLabel#muted {{
    color: {TEXT_MUTED};
}}

/* ---------- cards ---------- */
QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ---------- inputs ---------- */
QLineEdit, QComboBox, QSpinBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 20px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    background: #EDEFF2;
    color: {TEXT_MUTED};
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 14px;
    min-height: 20px;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
}}
QPushButton:pressed {{
    background: #E8EEF4;
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    background: #EDEFF2;
    border: 1px solid #E2E6EA;
}}
QPushButton#primary {{
    background: {ACCENT};
    color: #FFFFFF;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: {ACCENT_DARK};
    border: 1px solid {ACCENT_DARK};
}}
QPushButton#primary:disabled {{
    background: #A9BFD3;
    border: 1px solid #A9BFD3;
    color: #F0F4F8;
}}
QPushButton#danger {{
    color: #B3261E;
}}

/* ---------- tables and trees ---------- */
QTreeWidget, QTableWidget, QListWidget, QTextEdit, QPlainTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: #D6E6F4;
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: #EEF1F4;
    color: {TEXT_MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QTreeWidget::item, QTableWidget::item {{
    padding: 4px 2px;
}}

/* ---------- progress ---------- */
QProgressBar {{
    background: #E6EAEE;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

/* ---------- misc ---------- */
QSplitter::handle {{
    background: transparent;
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{
    border: none;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE};
}}
QTabBar::tab {{
    background: transparent;
    padding: 7px 14px;
    color: {TEXT_MUTED};
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}
"""
