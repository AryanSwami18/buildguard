"""Streamlit interface for single-package BuildGuard comparisons."""

from __future__ import annotations

import difflib
from html import escape
import re

import streamlit as st

from app.config.settings import Settings, load_project_environment
from app.backend_client import BuildGuardAPIClient
from app.models.report import ComparisonReport
from app.models.repair import ScriptFixProposal
from app.reporting.code_view import annotations_by_line
from app.reporting.reporter import render_report
from app.time_utils import local_timestamp


st.set_page_config(
    page_title="BuildGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_project_environment()
settings = Settings()
api_client = BuildGuardAPIClient()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0e1117;
            --surface: #161b22;
            --surface-raised: #1c2330;
            --ink: #e6edf3;
            --muted: #8b949e;
            --line: #30363d;
            --brand: #6e8cff;
            --brand-dark: #536fe4;
            --brand-soft: #202942;
            --green: #3fb950;
            --green-soft: #17291f;
            --amber: #d29922;
            --amber-soft: #2c2414;
            --red: #f85149;
            --red-soft: #30171a;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); }
        [data-testid="stHeader"] { background: var(--bg); }
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stAppViewContainer"] label {
            color: var(--ink);
        }
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p { color: var(--muted); }
        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding-top: 4.25rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: #0b0e13;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stSidebar"] label {
            color: var(--ink);
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: var(--muted); }
        [data-testid="stSidebar"] hr { border-color: var(--line); }

        .bg-brand {
            display: flex;
            align-items: center;
            gap: .8rem;
            margin-bottom: 2rem;
        }
        .bg-brand-mark {
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            border-radius: .65rem;
            background: var(--brand);
            color: #0e1117;
            font-size: 1.15rem;
            font-weight: 800;
        }
        .bg-brand-name { color: var(--ink); font-size: 1.25rem; font-weight: 750; line-height: 1.1; }
        .bg-form-note { color: var(--muted); font-size: .84rem; line-height: 1.45; padding-top: .2rem; }
        .bg-form-note strong { color: var(--ink); }
        .bg-result {
            display: flex;
            align-items: flex-start;
            gap: .9rem;
            padding: 1rem 1.15rem;
            border: 1px solid;
            border-radius: .9rem;
            margin: .15rem 0 1.35rem;
        }
        .bg-result.review { background: var(--amber-soft); border-color: #6e4f12; }
        .bg-result.clear { background: var(--green-soft); border-color: #235b3b; }
        .bg-result-icon { color: var(--ink); font-size: 1.35rem; line-height: 1.4; }
        .bg-result-title { color: var(--ink); font-size: 1rem; font-weight: 750; }
        .bg-result-copy { color: var(--muted); font-size: .92rem; line-height: 1.5; margin-top: .18rem; }
        .bg-report-heading { color: var(--ink); font-size: 1.2rem; font-weight: 720; margin: .25rem 0 .15rem; }
        .bg-report-subheading { color: var(--muted); font-size: .84rem; }
        .bg-sidebar-brand { color: var(--ink); font-size: 1.15rem; font-weight: 750; }
        .bg-engine {
            border: 1px solid var(--line);
            border-radius: .75rem;
            background: var(--surface);
            padding: .8rem .9rem;
            margin: .85rem 0 1rem;
        }
        .bg-engine-label { color: var(--muted); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; }
        .bg-engine-value { color: var(--ink); font-size: .9rem; font-weight: 650; margin-top: .3rem; overflow-wrap: anywhere; }

        [data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: .85rem;
            background: var(--surface);
            padding: 1rem 1.15rem 1.15rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, .16);
        }
        [data-testid="stForm"] [data-testid="stTextInput"] input {
            border-radius: .65rem;
            border-color: var(--line);
            background: var(--surface-raised);
            color: var(--ink);
        }
        [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
            border-color: var(--brand);
            box-shadow: 0 0 0 1px var(--brand);
        }
        .stButton > button, [data-testid="stDownloadButton"] > button {
            border-radius: .65rem;
            font-weight: 650;
            background: var(--surface-raised);
            color: var(--ink);
            border-color: var(--line);
        }
        .stButton > button[kind="primary"] {
            background: var(--brand);
            border-color: var(--brand);
        }
        .stButton > button[kind="primary"]:hover { background: var(--brand-dark); border-color: var(--brand-dark); }
        [data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: .85rem;
            background: var(--surface);
            padding: .85rem 1rem;
            box-shadow: 0 5px 16px rgba(0, 0, 0, .13);
        }
        [data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-baseweb="select"] > div { background: var(--surface-raised); border-color: var(--line); color: var(--ink); }
        [data-testid="stTabs"] button { font-weight: 650; }
        [data-testid="stExpander"] { border-color: var(--line); border-radius: .8rem; background: var(--surface); }
        [data-testid="stAlert"] { border-radius: .75rem; background: var(--surface-raised); border-color: var(--line); color: var(--ink); }
        .bg-empty { color: var(--muted); font-size: .9rem; margin-top: 1rem; }
        .bg-code-legend { display: flex; flex-wrap: wrap; gap: .55rem; margin: .7rem 0 1rem; }
        .bg-code-legend-item { border: 1px solid var(--line); border-radius: .55rem; padding: .35rem .55rem; color: var(--muted); font-size: .78rem; }
        .bg-code-legend-item::before { content: ""; display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .4rem; background: currentColor; }
        .bg-code-legend-item.undocumented { color: #e3a52f; }
        .bg-code-legend-item.conflict { color: #ff6b62; }
        .bg-code-legend-item.unused { color: #c084fc; }
        .bg-code-view { overflow: auto; border: 1px solid var(--line); border-radius: .8rem; background: #0a0d12; padding: .8rem 0; max-height: 72vh; }
        .bg-code-row { display: flex; min-width: max-content; line-height: 1.55; font: .82rem/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .bg-code-number { user-select: none; width: 4.2rem; flex: 0 0 4.2rem; padding-right: .9rem; text-align: right; color: #536071; }
        .bg-code-text { padding-right: 1.5rem; white-space: pre; color: #d6deeb; }
        .bg-code-comment { padding: .1rem 0 .1rem 0; background: #141a24; }
        .bg-code-comment .bg-code-text { color: #a9b9d1; font-style: italic; }
        .bg-code-line-undocumented { background: rgba(227, 165, 47, .16); border-left: 3px solid #e3a52f; }
        .bg-code-line-conflict { background: rgba(248, 81, 73, .18); border-left: 3px solid #f85149; }
        .bg-code-line-unused { background: rgba(192, 132, 252, .16); border-left: 3px solid #c084fc; }
        .bg-code-line-other { background: rgba(110, 140, 255, .16); border-left: 3px solid var(--brand); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    st.markdown(
        """
        <div class="bg-brand">
            <div class="bg-brand-mark">✦</div>
            <div>
                <div class="bg-brand-name">BuildGuard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _load_package_options(
    api_url: str,
) -> tuple[str, ...]:
    return tuple(BuildGuardAPIClient(api_url).list_packages())


def _open_saved_report(entry: dict[str, object]) -> None:
    report_data = entry.get("report")
    if not isinstance(report_data, dict):
        st.error("The saved report has an invalid format.")
        return
    try:
        report = ComparisonReport.model_validate(report_data)
    except ValueError as error:
        st.error(f"The saved report could not be opened: {error}")
        return
    if "buildguard_saved_report_background_saved" not in st.session_state:
        st.session_state["buildguard_saved_report_background_saved"] = True
        st.session_state["buildguard_report_before_saved_view"] = st.session_state.get(
            "buildguard_report"
        )
        st.session_state["buildguard_package_before_saved_view"] = st.session_state.get(
            "buildguard_report_package"
        )
    st.session_state["buildguard_report"] = report.model_dump(mode="json")
    st.session_state["buildguard_report_package"] = report.package.strip().casefold()
    st.session_state["buildguard_opened_report_id"] = entry.get("id")


def _saved_report_timestamp(value: object) -> str:
    """Display the UTC database timestamp in the local timezone."""
    if not isinstance(value, str) or not value:
        return "time unavailable"
    return local_timestamp(value)


_inject_styles()
_render_brand_header()

with st.sidebar:
    st.markdown(
        '<div class="bg-sidebar-brand">BuildGuard</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Analysis engine")
    if settings.ai_provider == "ollama":
        engine_name = "Ollama · local"
        engine_model = settings.ollama_model
        engine_note = "Ollama must be running on this machine."
    elif settings.ai_provider == "gemini":
        engine_name = "Gemini · cloud"
        engine_model = settings.gemini_model
        engine_note = "Uses the configured Gemini API connection."
    else:
        engine_name = settings.ai_provider.title()
        engine_model = "Configured provider"
        engine_note = "Uses the configured AI provider."
    st.markdown(
        f'<div class="bg-engine">'
        f'<div class="bg-engine-label">Provider</div>'
        f'<div class="bg-engine-value">{escape(engine_name)}</div>'
        f'<div class="bg-engine-label" style="margin-top:.65rem">Model</div>'
        f'<div class="bg-engine-value">{escape(engine_model)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(engine_note)
    st.caption("GitHub is used to retrieve the Wiki and build script.")

    try:
        saved_report_entries = api_client.list_reports()
    except Exception:
        saved_report_entries = []

    if saved_report_entries:
        st.markdown("---")
        st.markdown("#### Saved reports")
        report_groups: dict[str, list[dict[str, object]]] = {}
        for entry in saved_report_entries:
            package_name = str(entry.get("package") or "Unknown package")
            report_groups.setdefault(package_name, []).append(entry)
        for package_name, entries in report_groups.items():
            with st.expander(f"{package_name} ({len(entries)})", expanded=False):
                for entry in entries:
                    version = str(entry.get("version") or "")
                    findings = entry.get("findings", 0)
                    updated_label = _saved_report_timestamp(entry.get("updated_at"))
                    label = (
                        f"{version or 'Unversioned'} · {findings} finding"
                        f"{'s' if findings != 1 else ''} · {updated_label}"
                    )
                    if st.button(
                        label,
                        key=f"buildguard_open_report_{entry.get('id')}",
                        use_container_width=True,
                    ):
                        _open_saved_report(entry)

package_load_error: str | None = None
try:
    with st.spinner("Loading available packages…"):
        package_options = list(
            _load_package_options(
                api_client.base_url,
            )
        )
except Exception:
    package_options = []
    package_load_error = "The package list could not be loaded from the configured Wiki."

if not package_options:
    package_options = ["Alfresco"]
    package_load_error = package_load_error or "No BuildGuard package pages were found in the configured Wiki."

with st.form("buildguard_comparison"):
    package = st.selectbox(
        "Package",
        options=package_options,
        index=package_options.index("Alfresco") if "Alfresco" in package_options else 0,
        help="Choose a package from the configured BuildGuard Wiki.",
    )
    compare_clicked = st.form_submit_button(
        "Run comparison  →",
        type="primary",
        disabled=not package.strip(),
        use_container_width=True,
    )

if package_load_error:
    st.caption(package_load_error)


def _render_findings_overview(report: ComparisonReport) -> None:
    """Render the reader-first view used by the Streamlit report tab."""
    if not report.discrepancies:
        st.success(
            "Nothing needs attention. The documented build actions are represented "
            "by the script."
        )
        return

    st.subheader("What needs attention")
    st.caption(
        "These are the only actionable findings. Expand an item to see what the "
        "Wiki says, what the script does, and the recommended next step."
    )
    for item in report.discrepancies:
        label = _streamlit_finding_label(item.status.value, item.category.value)
        with st.expander(f"{item.id} · {label}", expanded=True):
            st.markdown(f"**Finding:** {item.title}")
            st.caption(_streamlit_priority(item.status.value))
            st.write(item.explanation)

            evidence_columns = st.columns(2)
            with evidence_columns[0]:
                st.markdown("**What the Wiki says**")
                if item.wiki:
                    _render_evidence_command(item.wiki)
                else:
                    st.info("No matching Wiki action was linked.")
            with evidence_columns[1]:
                st.markdown("**What the script does**")
                if item.script:
                    for evidence in item.script:
                        _render_evidence_command(evidence)
                elif item.status.value == "missing":
                    st.info("No matching Bash action was found.")

            st.info(f"Recommended next step: {item.recommendation}")


@st.dialog("Saved comparison report", width="large")
def _render_saved_report_dialog(report: ComparisonReport) -> None:
    """Show a selected historical report separately from the comparison page."""
    report_title = f"{report.package} {report.version or ''}".strip()
    st.markdown(f"### {escape(report_title)}")
    st.caption("Saved report · viewing this report does not replace the comparison page.")
    _render_result_banner(report)

    overview_tab, report_tab, data_tab, code_tab = st.tabs(
        ("Overview", "Full report", "Structured data", "Code")
    )
    with overview_tab:
        _render_findings_overview(report)
    with report_tab:
        st.markdown(render_report(report, "markdown"))
    with data_tab:
        st.json(report.model_dump(mode="json", exclude_none=True))
    with code_tab:
        _render_code_view(report)

    _render_fix_pr_panel(report)
    if st.button("Close report", key=f"buildguard_close_saved_report_{report.package}_{report.version}"):
        previous_report = st.session_state.pop(
            "buildguard_report_before_saved_view", None
        )
        previous_package = st.session_state.pop(
            "buildguard_package_before_saved_view", None
        )
        st.session_state.pop("buildguard_saved_report_background_saved", None)
        if previous_report is None:
            st.session_state.pop("buildguard_report", None)
        else:
            st.session_state["buildguard_report"] = previous_report
        if previous_package is None:
            st.session_state.pop("buildguard_report_package", None)
        else:
            st.session_state["buildguard_report_package"] = previous_package
        st.session_state.pop("buildguard_opened_report_id", None)
        st.rerun()


def _render_evidence_command(evidence) -> None:
    """Keep source evidence concise: only the relevant line and command."""
    if evidence.line_start is not None:
        end = evidence.line_end or evidence.line_start
        line_label = (
            f"Line {end}"
            if end == evidence.line_start
            else f"Lines {evidence.line_start}–{end}"
        )
    else:
        match = re.search(r"\blines?\s+(\d+)(?:\s*[-–]\s*(\d+))?", evidence.location, re.I)
        line_label = ""
        if match:
            start, end = match.groups()
            line_label = f"Line {start}" if not end or end == start else f"Lines {start}–{end}"
    if line_label:
        st.caption(line_label)
    commands = evidence.commands
    if commands:
        st.code("\n".join(commands), language="bash")


def _render_code_view(report: ComparisonReport) -> None:
    """Render the full Bash source with display-only finding annotations."""
    try:
        script = api_client.script_source(report)
    except Exception as error:
        st.error(f"The complete script could not be loaded: {error}")
        return
    if script is None:
        st.info("No Bash script source is available for this comparison.")
        return

    lines = script.splitlines()
    annotations = annotations_by_line(report, len(lines))
    st.caption(
        "This is the complete script. BuildGuard comments and colors are display-only; "
        "the original script is not modified."
    )
    st.markdown(
        '<div class="bg-code-legend">'
        '<span class="bg-code-legend-item undocumented">Undocumented script action</span>'
        '<span class="bg-code-legend-item conflict">Potential conflict</span>'
        '<span class="bg-code-legend-item unused">Unused configuration</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    rows: list[str] = []
    rendered_comments: set[tuple[int, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        line_annotations = annotations.get(line_number, [])
        for annotation in line_annotations:
            comment_key = (annotation.comment_line, annotation.comment)
            if annotation.comment_line != line_number or comment_key in rendered_comments:
                continue
            rendered_comments.add(comment_key)
            rows.append(
                '<div class="bg-code-row bg-code-comment">'
                '<span class="bg-code-number"> </span>'
                f'<span class="bg-code-text">{escape(annotation.comment)}</span>'
                '</div>'
            )

        classes = " ".join(
            f"bg-code-line-{annotation.css_class}"
            for annotation in {
                item.css_class: item for item in line_annotations
            }.values()
        )
        rows.append(
            f'<div class="bg-code-row {classes}">'
            f'<span class="bg-code-number">{line_number}</span>'
            f'<span class="bg-code-text">{escape(line) or " "}</span>'
            '</div>'
        )

    st.markdown(
        '<div class="bg-code-view">' + "".join(rows) + '</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="2s")
def _render_fix_pr_panel(report: ComparisonReport) -> None:
    """Generate a reviewed script diff and optionally create its GitHub PR."""
    if not report.discrepancies:
        return

    st.divider()
    st.subheader("Propose a script fix")
    st.caption(
        "BuildGuard can use the findings to prepare a minimal script change. "
        "Review the complete diff before creating a GitHub pull request."
    )
    # Keep a proposal tied to the exact report that produced it.  A fresh
    # comparison can keep the same package, version, and script path while
    # changing its findings; using only those source labels would then leave
    # an old proposal displayed for the new report.
    report_key = report.model_dump_json(exclude_none=True)
    if st.session_state.get("buildguard_fix_report_key") != report_key:
        st.session_state.pop("buildguard_fix_proposal", None)
        st.session_state.pop("buildguard_fix_original", None)
        st.session_state.pop("buildguard_fix_result", None)
        st.session_state.pop("buildguard_fix_job_id", None)
        st.session_state.pop("buildguard_fix_error", None)
        st.session_state.pop("buildguard_fix_reviewed", None)
        st.session_state["buildguard_fix_report_key"] = report_key

    job_id = st.session_state.get("buildguard_fix_job_id")
    proposal_data = st.session_state.get("buildguard_fix_proposal")
    button_label = (
        "Generating fix proposal..."
        if isinstance(job_id, str) and job_id
        else "Generate fix proposal"
    )
    if st.button(
        button_label,
        key="buildguard_generate_fix",
        disabled=bool(job_id),
    ):
        try:
            new_job_id = api_client.start_fix_proposal(report)
        except Exception as error:  # Display backend failures without a traceback.
            st.session_state["buildguard_fix_error"] = str(error)
        else:
            st.session_state["buildguard_fix_job_id"] = new_job_id
            st.session_state.pop("buildguard_fix_error", None)
            st.session_state.pop("buildguard_fix_original", None)
            st.session_state.pop("buildguard_fix_proposal", None)
            st.session_state.pop("buildguard_fix_result", None)
            job_id = new_job_id
            proposal_data = None

    error_message = st.session_state.get("buildguard_fix_error")
    if isinstance(error_message, str) and error_message:
        st.error(f"Could not generate a fix proposal: {error_message}")

    if isinstance(job_id, str) and job_id and not proposal_data:
        try:
            status, result, error = api_client.get_fix_proposal(job_id)
        except Exception as error:  # Keep polling failures visible to the user.
            st.error(f"Could not read fix proposal status: {error}")
            return
        if status == "running" or status == "queued":
            st.info(
                "The fix proposal is still being generated. This panel checks "
                "again automatically every two seconds."
            )
            return
        if status == "failed" or status == "missing" or result is None:
            message = error or "The backend returned no fix proposal."
            st.session_state.pop("buildguard_fix_job_id", None)
            st.session_state["buildguard_fix_error"] = message
            st.error(f"Could not generate a fix proposal: {message}")
            return
        original, proposal = result
        st.session_state["buildguard_fix_original"] = original
        st.session_state["buildguard_fix_proposal"] = proposal.model_dump(mode="json")
        st.session_state.pop("buildguard_fix_job_id", None)
        st.session_state.pop("buildguard_fix_error", None)
        proposal_data = st.session_state["buildguard_fix_proposal"]

    original = st.session_state.get("buildguard_fix_original")
    if not proposal_data or not isinstance(original, str):
        return

    proposal = ScriptFixProposal.model_validate(proposal_data)
    st.success("Fix proposal ready for review.")
    st.markdown(f"**Summary:** {proposal.summary}")
    st.markdown(f"**Rationale:** {proposal.rationale}")
    if proposal.affected_finding_ids:
        st.caption(
            "Findings addressed: " + ", ".join(proposal.affected_finding_ids)
        )

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposal.fixed_script.splitlines(keepends=True),
            fromfile=report.script_source.path if report.script_source else "original.sh",
            tofile="BuildGuard proposed fix",
        )
    )
    if not diff:
        st.warning("The proposal does not change the script, so no PR can be created.")
        return
    st.code(diff, language="diff")

    if not settings.github_token:
        st.info(
            "Fix proposal generated. Set GITHUB_TOKEN to enable the Create PR button."
        )
        return
    reviewed = st.checkbox(
        "I reviewed this diff and want BuildGuard to create the pull request.",
        key="buildguard_fix_reviewed",
    )
    if st.button(
        "Create draft PR" if settings.github_pr_draft else "Create PR",
        key="buildguard_create_fix_pr",
        type="primary",
        disabled=not reviewed,
    ):
        try:
            with st.spinner("Creating the GitHub pull request…"):
                result = api_client.create_pull_request(
                    report,
                    original,
                    proposal,
                )
            st.session_state["buildguard_fix_result"] = result
        except Exception as error:
            st.error(f"Could not create the pull request: {error}")

    result = st.session_state.get("buildguard_fix_result")
    if isinstance(result, dict):
        url = result.get("html_url")
        if isinstance(url, str) and url:
            st.success(f"Pull request created: {url}")
        else:
            st.success("Pull request created successfully.")


def _streamlit_finding_label(status: str, category: str | None = None) -> str:
    if status == "conflict" and category == "dependency_inconsistency":
        return "Dependency versions differ"
    return {
        "missing": "Wiki instruction not found in the script",
        "extra": "Script action is not represented in the Wiki",
        "conflict": "Wiki and script specify different values or behavior",
        "unused_configuration": "Configuration is defined but not used",
    }.get(status, "Finding requiring review")


def _streamlit_priority(status: str) -> str:
    return {
        "conflict": "Review needed · confirm which source has the intended value",
        "missing": "Needs review · the Wiki describes an action not found in the script",
        "extra": "Needs review · the script performs an action not represented in the Wiki",
        "unused_configuration": "Cleanup candidate · the value is defined but not referenced",
    }.get(status, "Review this finding against both sources")


def _render_result_banner(report: ComparisonReport) -> None:
    if report.counts.findings:
        tone = "review"
        icon = "!"
        title = "Review needed"
    else:
        tone = "clear"
        icon = "✓"
        title = "No actionable issues found"
    st.markdown(
        f'<div class="bg-result {tone}">'
        f'<div class="bg-result-icon">{icon}</div>'
        f'<div><div class="bg-result-title">{title}</div>'
        f'<div class="bg-result-copy">{escape(report.overview)}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="2s")
def _render_comparison_progress(job_id: str) -> None:
    """Poll a backend comparison and show its overall progress."""
    try:
        status, _progress, progress_percent, progress_log, report, error = api_client.get_comparison(job_id)
    except Exception as error:  # Display backend failures without a traceback.
        st.error(f"Could not read comparison progress: {error}")
        return

    if status == "running":
        st.markdown('<div class="bg-report-heading">Comparison in progress</div>', unsafe_allow_html=True)
        st.progress(
            progress_percent / 100,
            text=f"Estimated overall progress: {progress_percent}%",
        )
        with st.expander("Live terminal", expanded=False):
            st.code("\n".join(progress_log), language="text")
        st.caption(
            "Progress updates automatically while BuildGuard checks the cache, "
            "parses sources, and compares the build actions."
        )
        return

    if status == "failed" or report is None:
        st.session_state.pop("buildguard_comparison_job_id", None)
        st.error(f"Comparison failed: {error or 'The backend returned no report.'}")
        return

    st.session_state["buildguard_report"] = report.model_dump(mode="json")
    st.session_state["buildguard_report_package"] = st.session_state.get(
        "buildguard_comparison_package",
        "",
    )
    st.session_state.pop("buildguard_comparison_job_id", None)
    st.session_state.pop("buildguard_comparison_package", None)
    st.session_state.pop("buildguard_opened_report_id", None)
    st.session_state.pop("buildguard_report_before_saved_view", None)
    st.session_state.pop("buildguard_package_before_saved_view", None)
    st.session_state.pop("buildguard_saved_report_background_saved", None)
    st.rerun()


if compare_clicked:
    st.session_state.pop("buildguard_report", None)
    st.session_state.pop("buildguard_report_package", None)
    st.session_state.pop("buildguard_opened_report_id", None)
    st.session_state.pop("buildguard_report_before_saved_view", None)
    st.session_state.pop("buildguard_package_before_saved_view", None)
    st.session_state.pop("buildguard_saved_report_background_saved", None)
    st.session_state.pop("buildguard_comparison_job_id", None)
    st.session_state["buildguard_comparison_package"] = package.strip().casefold()
    try:
        st.session_state["buildguard_comparison_job_id"] = api_client.start_comparison(package)
    except Exception as error:  # Display backend failures without a traceback.
        st.error(f"Could not start comparison: {error}")


active_comparison_job = st.session_state.get("buildguard_comparison_job_id")
if isinstance(active_comparison_job, str) and active_comparison_job:
    _render_comparison_progress(active_comparison_job)

saved_report = st.session_state.get("buildguard_report")
saved_package = st.session_state.get("buildguard_report_package")
opened_report = st.session_state.get("buildguard_opened_report_id") is not None
if saved_report and opened_report:
    _render_saved_report_dialog(ComparisonReport.model_validate(saved_report))

if saved_report and not opened_report:
    if saved_package != package.strip().casefold() and not opened_report:
        st.info("The displayed report belongs to the previous package input. Run a new comparison to update it.")
    else:
        report = ComparisonReport.model_validate(saved_report)
        _render_result_banner(report)

        st.markdown('<div class="bg-report-heading">At a glance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="bg-report-subheading">A quick view of what matched and what needs attention.</div>',
            unsafe_allow_html=True,
        )
        metric_columns = st.columns(4)
        with metric_columns[0]:
            st.metric("Actionable items", report.counts.findings)
            st.caption("Need a person to review")
        with metric_columns[1]:
            st.metric("Semantic matches", report.counts.semantic_matches)
            st.caption("Wiki actions implemented by the script")
        with metric_columns[2]:
            st.metric("Potential conflicts", report.counts.potential_conflicts)
            st.caption("Wiki and script may disagree")
        with metric_columns[3]:
            st.metric("Unused configuration", report.counts.unused_configuration)
            st.caption("Defined but never referenced")

        with st.expander("How to read the results", expanded=False):
            st.write(
                "One Wiki instruction can be implemented by many Bash commands, "
                "variables, and functions. Those related commands count as one "
                "semantic match. Setup and cleanup are shown as findings only when "
                "they change a user-facing build action."
            )

        report_title = f"{report.package} {report.version or ''}".strip()
        title_column, rerun_column, format_column, download_column = st.columns([1.25, .9, .8, .85])
        with title_column:
            st.markdown(f'<div class="bg-report-heading">{escape(report_title)}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="bg-report-subheading">Comparison output</div>',
                unsafe_allow_html=True,
            )
        with rerun_column:
            if st.button(
                "Compare again",
                key=f"buildguard_rerun_{report.package}_{report.version or 'latest'}",
                disabled=bool(active_comparison_job),
                use_container_width=True,
                help="Run a fresh AI comparison while reusing parsed Wiki and script sources.",
            ):
                st.session_state["buildguard_comparison_package"] = report.package.strip().casefold()
                try:
                    st.session_state["buildguard_comparison_job_id"] = api_client.start_comparison(
                        report.package,
                        force_comparison=True,
                    )
                    st.rerun()
                except Exception as error:  # Display backend failures without a traceback.
                    st.error(f"Could not start a fresh comparison: {error}")
        with format_column:
            format_label = st.selectbox(
                "Download format",
                options=("Markdown", "JSON", "HTML"),
                key="buildguard_download_format",
                label_visibility="collapsed",
            )
        format_name, extension, mime_type = {
            "Markdown": ("markdown", "md", "text/markdown"),
            "JSON": ("json", "json", "application/json"),
            "HTML": ("html", "html", "text/html"),
        }[format_label]
        download_content = render_report(report, format_name)
        file_slug = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            f"{report.package}-{report.version or 'comparison'}",
        ).strip("-").lower()
        with download_column:
            st.download_button(
                "Download report",
                data=download_content,
                file_name=f"buildguard-{file_slug}.{extension}",
                mime=mime_type,
                type="primary",
                use_container_width=True,
            )

        overview_tab, report_tab, data_tab, code_tab = st.tabs(
            ("Overview", "Full report", "Structured data", "Code")
        )
        with overview_tab:
            _render_findings_overview(report)
        with report_tab:
            st.markdown(render_report(report, "markdown"))
        with data_tab:
            st.json(report.model_dump(mode="json", exclude_none=True))
        with code_tab:
            _render_code_view(report)
        _render_fix_pr_panel(report)

if not saved_report and not active_comparison_job:
    st.caption("Enter a package and run a comparison to begin.")
