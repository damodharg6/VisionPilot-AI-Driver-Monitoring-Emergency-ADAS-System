import sys
import threading
import time

from config import BEEP_DURATION, BEEP_FREQ, BEEP_INTERVAL

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlarmSystem:
    def __init__(self):
        self._active = threading.Event()

    def start(self):
        if not self._active.is_set():
            self._active.set()
            threading.Thread(target=self._beep_loop, daemon=True).start()

    def stop(self):
        self._active.clear()

    def _beep_loop(self):
        while self._active.is_set():
            if HAS_WINSOUND:
                winsound.Beep(BEEP_FREQ, BEEP_DURATION)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
            elapsed = 0.0
            while elapsed < BEEP_INTERVAL and self._active.is_set():
                time.sleep(0.05)
                elapsed += 0.05

