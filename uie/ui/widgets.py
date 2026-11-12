from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout
from PySide6.QtCore import Qt


class Panel(QFrame):
    """Bordered panel with a Bloomberg-style title bar."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        t = QLabel(title.upper())
        t.setObjectName("PanelTitle")
        outer.addWidget(t)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(8, 8, 8, 8)
        self.body.setSpacing(6)
        outer.addLayout(self.body)


class StatTile(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatTile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(0)
        self.label = QLabel(label.upper())
        self.label.setObjectName("StatLabel")
        self.value = QLabel("--")
        self.value.setObjectName("StatValue")
        lay.addWidget(self.label)
        lay.addWidget(self.value)

    def set_value(self, text: str):
        self.value.setText(text)
