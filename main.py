import sys
from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from uie.ui.theme_manager import ThemeManager
from uie.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("UIE")
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    themes = ThemeManager(root / "themes")
    themes.apply(QSettings("UIE", "UniversalIntelligenceEngine").value("theme", themes.default_id))
    win = MainWindow(themes)
    win.show()
    if len(sys.argv) > 1:
        win.open_dataset(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
