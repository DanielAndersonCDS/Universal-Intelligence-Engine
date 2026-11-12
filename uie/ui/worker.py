import traceback
from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    """Runs fn(progress) off the UI thread. fn returns the result passed to `done`."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str, int)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(lambda msg, pct=0: self.progress.emit(msg, pct)))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
