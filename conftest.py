"""Put the repository root on ``sys.path`` for the duration of a test run.

The suite imports ``experiments.clarabel_ball``: the Clarabel cone program is no
longer part of the package -- clarabel is a development dependency -- but it is
still the independent second implementation the shipped method is checked
against, which is worth more inside CI than outside it.

``experiments/`` is deliberately not in ``[tool.hatch.build.targets.wheel]``, so
it is not installed and cannot be imported the way ``cvxball`` is. The usual fix
is ``pythonpath = .`` in ``pytest.ini``, but that file is template-owned (see
``.rhiza/template.lock``) and its comment records that ``pythonpath`` was removed
on purpose. A root ``conftest.py`` is locally owned and reaches the same end:
pytest imports it before collection, so the insertion below happens first.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
