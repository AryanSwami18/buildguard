"""Executable entry point for BuildGuard AI's documentation extraction CLI."""

from app.docs_cli import main as docs_main


def main(argv: list[str] | None = None) -> int:
    """Run the documentation parser and print its JSON result."""
    return docs_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
