"""Extract the product version from a Build Instruction."""

from __future__ import annotations

import re


_VERSION = r"\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]+)?"

_PATTERNS = (
    # Alfresco 25.3.0
    re.compile(
        rf"\b(?P<product>[A-Za-z][A-Za-z0-9_-]*)"
        rf"\s+(?:version\s+)?(?P<version>{_VERSION})\b",
        re.IGNORECASE,
    ),

    # version 25.3.0
    re.compile(
        rf"\bversion\s+(?P<version>{_VERSION})\b",
        re.IGNORECASE,
    ),

    # v25.3.0
    re.compile(
        rf"\bv(?P<version>{_VERSION})\b",
        re.IGNORECASE,
    ),
)


def product_from_module(module_name: str) -> str:
    """Convert a Wiki module name such as Building-Alfresco to Alfresco."""

    value = module_name.strip()

    if value.lower().startswith("building-"):
        value = value[len("building-"):]

    value = value.removesuffix(".md")

    return value


def extract_version(
    content: str,
    product: str,
) -> str | None:
    """Extract the documented product version.

    Product-specific matches are preferred so that unrelated versions such
    as Maven or Java versions are not selected.
    """

    product_pattern = re.compile(
        rf"\b{re.escape(product)}\b"
        rf"(?:\s+version)?\s+"
        rf"(?P<version>{_VERSION})\b",
        re.IGNORECASE,
    )

    match = product_pattern.search(content)

    if match:
        return match.group("version")

    for pattern in _PATTERNS[1:]:
        match = pattern.search(content)

        if match:
            return match.group("version")

    return None
