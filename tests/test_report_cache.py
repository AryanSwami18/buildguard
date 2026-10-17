from app.models.report import ComparisonCounts, ComparisonReport
from app.storage.repository import ComparisonCacheKey, ParsedSourceRepository


def _key(*, wiki_revision: str = "wiki-1") -> ComparisonCacheKey:
    return ComparisonCacheKey(
        package="Demo",
        version="1.0",
        wiki_repository="example/docs",
        wiki_path="Building-Demo.md",
        wiki_ref="HEAD",
        wiki_revision=wiki_revision,
        script_repository="example/scripts",
        script_path="Demo/1.0/build_demo.sh",
        script_ref="master",
        script_revision="script-1",
        wiki_parser_version="wiki-v1",
        bash_parser_version="bash-v1",
        analysis_version="comparison-v1:ollama:test",
    )


def _report(overview: str = "cached") -> ComparisonReport:
    return ComparisonReport(
        package="Demo",
        version="1.0",
        status="no_mismatches_found",
        overview=overview,
        counts=ComparisonCounts(
            wiki_steps_compared=1,
            script_steps_checked=1,
            missing_steps=0,
            conflicting_steps=0,
            undocumented_script_steps=0,
            out_of_scope_steps=0,
            findings=0,
        ),
    )


def test_comparison_report_cache_requires_matching_source_revisions(tmp_path) -> None:
    repository = ParsedSourceRepository(tmp_path / "buildguard.sqlite3")
    key = _key()

    saved = repository.save_comparison_report(key, _report())
    loaded = repository.get_comparison_report(key)

    assert loaded is not None
    assert loaded.report_id == saved.report_id
    assert loaded.report.overview == "cached"
    assert len(repository.list_comparison_reports()) == 1
    assert repository.get_comparison_report(_key(wiki_revision="wiki-2")) is None


def test_comparison_report_cache_replaces_same_revision_entry(tmp_path) -> None:
    repository = ParsedSourceRepository(tmp_path / "buildguard.sqlite3")
    key = _key()

    first = repository.save_comparison_report(key, _report("first"))
    second = repository.save_comparison_report(key, _report("second"))

    assert first.report_id == second.report_id
    assert repository.get_comparison_report(key).report.overview == "second"
    assert len(repository.list_comparison_reports()) == 1
