"""Run the canonical, audit-safe PSR V1 command line."""

from .cli import main

if __name__ == "__main__":  # pragma: no branch
    raise SystemExit(main())
