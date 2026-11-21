"""Test minimal : le paquet s'importe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_import():
    import fuzzy_rules

    assert fuzzy_rules.__version__
