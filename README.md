# BuildGuard AI

BuildGuard AI compares Linux on IBM Z build documentation with its Bash
automation. It first retains each source's syntax and context, then delegates
semantic interpretation to an injected LLM provider.

```text
Wiki Markdown -> document structure and source facts -> LLM -> validated BuildSteps
Bash script   -> bashlex AST -> deterministic command semantics -> BuildSteps
              -> contextual comparison report
```

## Design

- Bash syntax is parsed by `bashlex`; a semantic pass then identifies commands,
  packages, repositories, Docker images, paths, and other explicit values.
- Markdown parsing keeps headings, paragraphs, fenced or indented code blocks,
  locations, and document order. It does not split documentation into unrelated
  sentences.
- `BuildPipeline` accepts an `LLMProvider`, a small protocol with one
  `generate(LLMRequest) -> LLMResponse` method. No provider SDK is imported by
  application code.
- The provider returns structured JSON-like data for Wiki extraction. Pydantic
  validates both Wiki and Bash results with the shared `BuildStep` model.

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

## Run the Wiki and Bash pipeline

The default sources are the [Build Instructions Wiki](https://github.com/AryanSwami18/docs/wiki)
and the versioned scripts in `AryanSwami18/scripts`. Set `GITHUB_REPO`,
`SCRIPTS_REPO`, `GITHUB_BRANCH`, or `SCRIPTS_BRANCH` to select other sources.
BuildGuard uses a local Ollama model by default, so Wiki extraction and
comparison do not require an AI API key. Install and pull the recommended model
once:

```bash
ollama pull qwen3.5:4b-q4_K_M
```

Ollama must be running locally. BuildGuard uses
[`qwen3.5:4b-q4_K_M`](https://ollama.com/library/qwen3.5) by default. Set
`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT_SECONDS`,
`OLLAMA_MAX_OUTPUT_TOKENS`, or `OLLAMA_CONTEXT_TOKENS` to change the local
configuration. The default context is 32K tokens and output limit is 16384
tokens. Ollama receives the expected JSON Schema for constrained responses.

To use Gemini instead, set `AI_PROVIDER=gemini` and `GEMINI_API_KEY`. Set
`GEMINI_MODEL` to choose another Gemini model. Set `GITHUB_TOKEN` when the
repositories are private. Keep credentials in `.env`; do not commit it.

Run a module by its Wiki page name, without the `.md` suffix:

```bash
python -m app.main Building-Alfresco
```

The command finds the product version in the Wiki, extracts the Wiki steps with
AI, fetches and parses the corresponding script, then asks AI to compare both
validated instructions. Large Wiki pages are extracted in smaller chunks. The
command prints only mismatches, with source locations and excerpts, to stdout;
matched steps are omitted. Progress and errors go to stderr. The installed
command is also available as `buildguard-docs`.

## Tests

```bash
pytest -q
```

The tests cover AST structure, Markdown document context, generic LLM pipeline
requests, Pydantic model validation, and contextual matching.

Persistence and change detection are intentionally not implemented yet.
