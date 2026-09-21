# BuildGuard AI

BuildGuard AI compares Linux on IBM Z build documentation with its Bash
automation. It first retains each source's syntax and context, then delegates
semantic interpretation to an injected LLM provider.

```text
Markdown / Bash
    -> source structure (Markdown blocks / bashlex AST)
    -> source and neighbouring-step context
    -> generic LLM request
    -> Pydantic-validated BuildStep
    -> contextual comparison report
```

## Design

- Bash is parsed by `bashlex`. The parser records nodes, nesting, operators,
  words, exact source text, and character positions. It contains no rules for
  `dnf`, `wget`, `tar`, or any other command.
- Markdown parsing keeps headings, paragraphs, code blocks, locations, and
  document order. It does not split documentation into unrelated sentences.
- `BuildPipeline` accepts an `LLMProvider`, a small protocol with one
  `generate(LLMRequest) -> LLMResponse` method. No provider SDK is imported by
  application code.
- The provider returns structured JSON-like data. Pydantic validates it as a
  shared `BuildStep`, so Wiki and Bash output use the same model.

## Using the core pipeline

Implement a small adapter for the LLM backend you choose. Its `generate`
method should turn `request.prompt`, `request.source`, `request.context`, and
`request.expected_schema` into an `LLMResponse(data=...)`.

```python
from app.intelligence.response import LLMResponse
from app.pipeline.build_pipeline import BuildPipeline


class MyProvider:
    def generate(self, request):
        # Call the selected backend here and return a JSON string.
        return LLMResponse(
            content='{"action": "unknown", "description": "Semantic result from the backend"}'
        )


pipeline = BuildPipeline(MyProvider())
bash_instruction = pipeline.from_bash("mkdir -p /opt/example", "install.sh")
wiki_instruction = pipeline.from_wiki("# Setup\n\nCreate the directory.", "guide.md")
```

The pipeline adds trusted source IDs, order, and source context before
Pydantic validates each returned step. The provider supplies the semantic
fields such as action, description, dependencies, packages, paths, URLs, and
execution context.

## Parse documentation from GitHub Wiki

The default source is the [Building Alfresco Wiki page](https://github.com/AryanSwami18/docs/wiki/Building-Alfresco).
Set `GITHUB_TOKEN` when accessing a private Wiki, or set `GITHUB_REPO` to use a
different documentation repository. Do not commit `.env`.

Fetch a module page and print the complete Markdown structure as JSON:

```bash
python -m app.docs_cli
```

The JSON includes source metadata plus the complete parsed document: original
content, headings, and ordered paragraphs, list steps, and code blocks with
their section and line locations. To additionally ask OpenRouter to turn the
complete Wiki document into one validated `BuildInstruction`, set
`OPENROUTER_API_KEY` and run:

```bash
python -m app.docs_cli "Module name" --with-ai
```

With `--with-ai`, the complete parsed Wiki document is sent in one structured
request and the command prints the resulting `BuildInstruction` under
`extracted_steps`. The same command is available through `python -m app.main`.

After installation, the equivalent command is `buildguard-docs "Module name"`.
This documentation command deliberately does not fetch or compare scripts.

## Tests

```bash
pytest -q
```

The tests cover AST structure, Markdown document context, generic LLM pipeline
requests, Pydantic model validation, and contextual matching.

Persistence and change detection are intentionally not implemented yet.
