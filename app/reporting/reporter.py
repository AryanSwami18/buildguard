from app.models.validation import ValidationResult


def render_report(result: ValidationResult) -> str:
    """Render the small validation model as readable plain text."""
    lines = [result.summary]
    for match in result.matches:
        wiki_id = match.wiki_step.id if match.wiki_step else "-"
        bash_id = match.automation_step.id if match.automation_step else "-"
        lines.append(f"{match.status.value}: wiki={wiki_id}, bash={bash_id}. {match.explanation}")
    return "\n".join(lines)
