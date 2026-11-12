"""serve/test_web_history.py - runs serve/test_web_history.js (the web app's request construction, on the real
app.js under node) when node is available; skips otherwise.  No GPU, no pack, no server.

    python -m unittest serve.test_web_history -v
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("test_web_history.js")


@unittest.skipIf(shutil.which("node") is None and shutil.which("nodejs") is None,
                 "node is not installed (the web app's request construction is checked by test_web_history.js)")
class WebAppHistory(unittest.TestCase):
    def test_the_history_the_app_sends(self):
        node = shutil.which("node") or "nodejs"
        done = subprocess.run([node, str(SCRIPT)], capture_output=True, text=True, timeout=60,
                              cwd=str(ROOT))
        self.assertEqual(done.returncode, 0, f"the web app's request construction changed:\n{done.stdout}{done.stderr}")


if __name__ == "__main__":
    unittest.main()
