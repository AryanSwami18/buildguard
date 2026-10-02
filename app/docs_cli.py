"""Documentation-only command line entry point for BuildGuard AI."""

import argparse
import json
import sys

from dotenv import load_dotenv

from app.config.settings import Settings
from app.github.client import GitHubClient
from app.intelligence.factory import create_provider
from app.parsers.semantic_parser import parse_semantic_facts
from app.parsers.wiki_parser import parse_wiki
from app.pipeline.build_pipeline import BuildPipeline


def main(argv: list[str] | None = None) -> int:
    """Fetch and fully parse one configured GitHub Wiki module page."""
    arguments = _arguments(argv)
    load_dotenv()
    settings = Settings()
    try:
        page = GitHubClient.from_settings(settings).fetch_module_wiki(arguments.module)
        document = parse_wiki(page.content, page.filename)
        semantic_facts = parse_semantic_facts(page.content)
        result: dict[str, object] = {
            "source": page.model_dump(mode="json", exclude={"content"}),
            "document": document.model_dump(mode="json"),
            "semantic_facts": semantic_facts.model_dump(mode="json"),
        }
        if arguments.with_ai:
            instruction = BuildPipeline(create_provider(settings)).from_github_wiki(page)
            result["extracted_steps"] = instruction.model_dump(mode="json")
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"buildguard-docs: {error}", file=sys.stderr)
        return 1


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and structurally parse a BuildGuard documentation Wiki page."
    )
    parser.add_argument(
        "module",
        nargs="?",
        default="Building-Alfresco",
        help="GitHub Wiki page/module name, without the .md suffix (default: Building-Alfresco)",
    )
    parser.add_argument(
        "--with-ai",
        action="store_true",
        help="also extract one Pydantic-validated BuildInstruction through OpenRouter",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
