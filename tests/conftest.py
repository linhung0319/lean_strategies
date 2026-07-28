"""Make the fakes importable and register them before any algorithm module
is imported, and put ``algorithms/`` on the path so tests can import the
files under test by module name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "algorithms"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import lean_stubs

lean_stubs.install()
