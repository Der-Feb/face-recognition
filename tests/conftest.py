"""Pytest fixtures shared by all tests.

Automatically adds the project root to sys.path so that `app` is importable
no matter from which directory pytest is launched.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)