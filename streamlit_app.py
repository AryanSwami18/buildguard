"""Streamlit interface for single-package BuildGuard comparisons."""

from __future__ import annotations

from html import escape
import re

import streamlit as st
from dotenv import load_dotenv

from app.config.settings import Settings
from app.models.report import ComparisonReport
from app.reporting.reporter import render_report
from app.service import list_available_packages, run_package_comparison


st.set_page_config(
    page_title="BuildGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
settings = Settings()


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
    github_repo: str | None,
    github_branch: str,
    github_token: str | None,
) -> tuple[str, ...]:
    source_settings = Settings(
        github_repo=github_repo,
        github_branch=github_branch,
        github_token=github_token,
    )
    return tuple(list_available_packages(source_settings))


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

package_load_error: str | None = None
try:
    with st.spinner("Loading available packages…"):
        package_options = list(
            _load_package_options(
                settings.github_repo,
                settings.github_branch,
                settings.github_token,
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
            "by the script, and internal implementation details were not counted "
            "as discrepancies."
        )
        return

    st.subheader("What needs attention")
    st.caption(
        "These are the only actionable findings. Expand an item to see what the "
        "Wiki says, what the script does, and the recommended next step."
    )
    for item in report.discrepancies:
        label = _streamlit_finding_label(item.status.value)
        with st.expander(f"{item.id} · {label}", expanded=True):
            st.markdown(f"**Finding:** {item.title}")
            st.caption(_streamlit_priority(item.status.value))
            st.write(item.explanation)

            evidence_columns = st.columns(2)
            with evidence_columns[0]:
                st.markdown("**What the Wiki says**")
                if item.wiki:
                    st.write(item.wiki.description)
                    st.caption(item.wiki.location)
                else:
                    st.info("No matching Wiki action was linked.")
            with evidence_columns[1]:
                st.markdown("**What the script does**")
                if item.script:
                    for evidence in item.script:
                        st.write(evidence.description)
                        st.caption(evidence.location)
                elif item.status.value == "missing":
                    st.info("No matching Bash action was found.")

            st.info(f"Recommended next step: {item.recommendation}")


def _streamlit_finding_label(status: str) -> str:
    return {
        "missing": "Wiki instruction not found in the script",
        "extra": "Script action is not represented in the Wiki",
        "conflict": "Wiki and script prescribe different behavior",
        "unused_configuration": "Configuration is defined but not used",
    }.get(status, "Finding requiring review")


def _streamlit_priority(status: str) -> str:
    return {
        "conflict": "High attention · the two sources may prescribe different behavior",
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


if compare_clicked:
    st.session_state.pop("buildguard_report", None)
    st.session_state.pop("buildguard_report_package", None)
    with st.status("Starting comparison…", expanded=True) as progress_status:
        try:
            report = run_package_comparison(
                package,
                settings,
                progress=progress_status.write,
            )
        except Exception as error:  # Display source/provider failures without a traceback.
            progress_status.update(
                label="Comparison failed",
                state="error",
                expanded=True,
            )
            st.error(str(error))
        else:
            progress_status.update(
                label="Comparison complete",
                state="complete",
                expanded=False,
            )
            st.session_state["buildguard_report"] = report.model_dump(mode="json")
            st.session_state["buildguard_report_package"] = package.strip().casefold()

saved_report = st.session_state.get("buildguard_report")
saved_package = st.session_state.get("buildguard_report_package")
if saved_report:
    if saved_package != package.strip().casefold():
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
                "semantic match. Internal setup and cleanup are ignored unless they "
                "represent a meaningful user-facing build action."
            )

        report_title = f"{report.package} {report.version or ''}".strip()
        title_column, format_column, download_column = st.columns([1.45, .8, .85])
        with title_column:
            st.markdown(f'<div class="bg-report-heading">{escape(report_title)}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="bg-report-subheading">Comparison output</div>',
                unsafe_allow_html=True,
            )
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

        overview_tab, report_tab, data_tab = st.tabs(
            ("Overview", "Full report", "Structured data")
        )
        with overview_tab:
            _render_findings_overview(report)
        with report_tab:
            st.markdown(render_report(report, "markdown"))
        with data_tab:
            st.json(report.model_dump(mode="json", exclude_none=True))

if not saved_report:
    st.caption("Enter a package and run a comparison to begin.")
