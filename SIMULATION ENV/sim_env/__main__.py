"""Enables ``python -m sim_env`` from the ``SIMULATION ENV`` directory."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
