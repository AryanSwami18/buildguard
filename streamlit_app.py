"""Streamlit interface for single-package BuildGuard comparisons."""

from __future__ import annotations

import re

import streamlit as st
from dotenv import load_dotenv

from app.config.settings import Settings
from app.models.report import ComparisonReport
from app.reporting.reporter import render_report
from app.service import run_package_comparison


st.set_page_config(
    page_title="BuildGuard AI",
    page_icon="🛡️",
    layout="wide",
)

load_dotenv()
settings = Settings()

st.title("🛡️ BuildGuard AI")
st.write(
    "Compare one package’s Build Instructions with its versioned automation "
    "script. The report lists discrepancies and suggested corrections."
)

with st.sidebar:
    st.header("Configuration")
    st.write(f"**AI provider:** `{settings.ai_provider}`")
    if settings.ai_provider == "ollama":
        st.write(f"**Local model:** `{settings.ollama_model}`")
        st.caption("Ollama must be running on this machine.")
    elif settings.ai_provider == "gemini":
        st.write(f"**Gemini model:** `{settings.gemini_model}`")
    st.caption("GitHub access is used to retrieve the Wiki and build script.")

with st.form("buildguard_comparison"):
    package = st.text_input(
        "Package or Wiki module",
        value="Alfresco",
        placeholder="Alfresco or Building-Alfresco",
        help="Enter one package name or its full Wiki module name.",
    )
    compare_clicked = st.form_submit_button(
        "Compare build instructions",
        type="primary",
        disabled=not package.strip(),
        use_container_width=True,
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
        st.divider()
        if report.counts.findings:
            st.warning(f"{report.status.replace('_', ' ').upper()} · {report.overview}")
        else:
            st.success(f"{report.status.replace('_', ' ').upper()} · {report.overview}")

        metric_columns = st.columns(4)
        metric_columns[0].metric("Findings", report.counts.findings)
        metric_columns[1].metric("Missing Wiki steps", report.counts.missing_steps)
        metric_columns[2].metric("Wiki/script conflicts", report.counts.conflicting_steps)
        metric_columns[3].metric(
            "Undocumented script steps",
            report.counts.undocumented_script_steps,
        )

        st.subheader(f"{report.package} {report.version or ''}".strip())
        format_label = st.selectbox(
            "Download format",
            options=("Markdown", "JSON", "HTML"),
            key="buildguard_download_format",
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
        st.download_button(
            "Download report",
            data=download_content,
            file_name=f"buildguard-{file_slug}.{extension}",
            mime=mime_type,
            type="primary",
            use_container_width=False,
        )

        report_tab, data_tab = st.tabs(("Report", "Structured data"))
        with report_tab:
            st.markdown(render_report(report, "markdown"))
        with data_tab:
            st.json(report.model_dump(mode="json", exclude_none=True))
