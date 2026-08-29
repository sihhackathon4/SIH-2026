"""Convenient launcher: ``python run.py [args]``.

Forwards all arguments to :func:`sim_env.cli.main`. Equivalent to
``python -m sim_env.cli`` from the ``SIMULATION ENV`` directory.
"""

from __future__ import annotations

import sys

from sim_env.cli import main

if __name__ == "__main__":
    sys.exit(main())
