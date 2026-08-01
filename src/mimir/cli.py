"""Command-line entry point. Subcommands (run / analyze / audit-judge) arrive in M5."""

from mimir import __version__


def main() -> int:
    print(f"mimir {__version__}")
    return 0
