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

Run one package by its name, or pass its explicit Wiki module name:

```bash
python -m app.main Alfresco
# Equivalent:
python -m app.main Building-Alfresco
```

The command finds the product version in the Wiki, extracts the Wiki steps with
AI, fetches and parses the corresponding script, then asks AI to compare both
validated instructions. Large Wiki pages are extracted in smaller chunks. The
report contains only discrepancies, with source locations, source evidence,
categories, and suggested corrections; successful matches are omitted. The
default Markdown report can be changed to JSON or standalone HTML, and saved to
a file:

```bash
python -m app.main Alfresco --format json --output report.json
python -m app.main Alfresco --format html --output report.html
```

Progress and errors go to stderr. The installed command is available as
`buildguard`; `buildguard-docs` remains as a compatibility alias.

## Parsed-source cache and GitHub synchronization

BuildGuard stores validated Wiki and Bash `BuildInstruction` objects in a local
SQLite database at `.buildguard/buildguard.sqlite3`. Set
`BUILDGUARD_DATABASE_PATH` to use another location. On a comparison, it checks
the current Wiki revision and script blob SHA first. If the source is unchanged,
the stored parsed output is reused; if it changed or is not cached, the existing
fetch/parse path runs and updates the cache. Reports are also stored using both
source revisions, parser versions, and the AI model as the cache key. An unchanged
package reuses its saved report without another AI comparison. The Streamlit UI
lists saved reports by package so they can be reopened later.

For proactive updates, configure a webhook secret. The recommended launcher
starts both the FastAPI backend and Streamlit UI together:

```bash
./run_buildguard.sh
```

The API listens on `127.0.0.1:8080` and the UI on `127.0.0.1:8501`. Expose
`POST /github/webhook` on port `8080` to GitHub and use the same
`GITHUB_WEBHOOK_SECRET` in the webhook configuration. Enable the **Push** event
for the scripts repository and the **Wiki** event for the documentation
repository. The backend recognizes changed `Building-*.md` pages and versioned
`.sh` build scripts, then refreshes only the affected package's parsed cache in
the background. `GET /healthz` is available for a service health check.

The Streamlit UI communicates with the backend through `BUILDGUARD_API_URL`.
The API uses `GITHUB_WEBHOOK_SECRET` as an internal API key when it is set, so
the comparison and pull-request endpoints are not openly usable through the
webhook tunnel. Override `BUILDGUARD_API_SECRET` if a separate internal key is
preferred. The old `buildguard-webhook` command remains available for running
only the standalone listener.

## Streamlit interface

Install project dependencies, then launch both services from the project root:

```bash
pip install -e .
./run_buildguard.sh
```

The UI is available at `http://127.0.0.1:8501`.

Select a package from the configured Wiki, run the comparison, and review the
mismatch-only report in the browser. The report also has a Code view that shows
the complete Bash script with display-only annotations for actionable findings.
The download control exports the same report as Markdown, JSON, or standalone
HTML.

When a report contains findings, the UI can generate a complete-script fix
proposal and show its diff. After reviewing the diff, use **Create draft PR** to
commit the change and open a pull request against `SCRIPTS_REPO`. Set
`GITHUB_TOKEN` with permission to create branches and pull requests. By default
the branch is created in `SCRIPTS_REPO`; set `BUILDGUARD_PR_HEAD_REPO` to a
fork such as `your-user/scripts` when the token cannot push to the base repo.
`BUILDGUARD_PR_DRAFT=true` keeps generated pull requests in draft state. The
current script is re-read before the commit, and PR creation is refused if it
changed since the comparison.
For a fine-grained GitHub token, grant **Contents: Read and write** and
**Pull requests: Read and write** on the relevant base repository and fork.

## Tests

```bash
pytest -q
```

The tests cover AST structure, Markdown document context, generic LLM pipeline
requests, Pydantic model validation, and contextual matching.
