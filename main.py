"""
Application entry point. This is what PyInstaller/Nuitka packages into the
final .exe (see build/ for the packaging spec/commands).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docclassify.ui.main_window import main

if __name__ == "__main__":
    main()
