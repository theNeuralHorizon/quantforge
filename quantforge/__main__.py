"""Allow `python -m quantforge ...` to invoke the CLI."""
from quantforge.cli import main

raise SystemExit(main())
