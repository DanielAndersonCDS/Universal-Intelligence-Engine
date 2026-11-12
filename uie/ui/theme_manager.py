from __future__ import annotations
import json
from pathlib import Path
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication


class ThemeManager(QObject):
    """Loads every *.json in the themes folders and renders base.qss with the theme's tokens."""
    changed = Signal(dict)

    def __init__(self, builtin_dir: Path, user_dir: Path | None = None):
        super().__init__()
        self.builtin_dir = Path(builtin_dir)
        self.user_dir = user_dir or Path.home() / ".uie" / "themes"
        self.themes: dict[str, dict] = {}
        self.current: dict = {}
        self.reload()

    def reload(self):
        self.themes.clear()
        for d in (self.builtin_dir, self.user_dir):
            if d.is_dir():
                for f in sorted(d.glob("*.json")):
                    try:
                        t = json.loads(f.read_text(encoding="utf-8"))
                        self.themes[t["id"]] = t
                    except Exception as e:
                        print(f"[themes] skipped {f.name}: {e}")
        if not self.themes:
            raise RuntimeError(f"No themes found in {self.builtin_dir}")

    @property
    def default_id(self) -> str:
        return "bloomberg_terminal" if "bloomberg_terminal" in self.themes else next(iter(self.themes))

    def apply(self, theme_id: str):
        t = self.themes.get(theme_id) or self.themes[self.default_id]
        qss = (self.builtin_dir / "base.qss").read_text(encoding="utf-8")
        size = int(t.get("font", {}).get("size", 10))
        tokens = dict(t["colors"])
        tokens.update(font_family=t.get("font", {}).get("family", "monospace"), font_size=size, font_size_big=size + 4)
        for k, v in tokens.items():
            qss = qss.replace(f"@{k}@", str(v))
        QApplication.instance().setStyleSheet(qss)
        self.current = t
        self.changed.emit(t)
