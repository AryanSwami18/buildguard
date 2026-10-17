"""Resolve package names between the Wiki and automation repositories.

The two repositories are maintained independently, so a Wiki package name is
not always the same as the scripts repository directory name.  Keep those
exceptions in one place and let the GitHub client handle safe case/punctuation
normalization for new packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import PurePosixPath
import re


@dataclass(frozen=True)
class ScriptPackageTarget:
    """Location of the script implementing one canonical Wiki package."""

    directory: str
    filename: str | None = None


@dataclass(frozen=True)
class PackageMapping:
    """A Wiki package and its corresponding scripts-repository location."""

    wiki_package: str
    target: ScriptPackageTarget
    version_prefixes: tuple[str, ...] = ()


_KNOWN_MAPPINGS = (
    PackageMapping("ANTLR-4", ScriptPackageTarget("AntLR")),
    PackageMapping("Apache-Cassandra", ScriptPackageTarget("ApacheCassandra", "build_cassandra.sh")),
    PackageMapping("Apache-Cassandra-3.11", ScriptPackageTarget("ApacheCassandra", "build_cassandra.sh"), ("3.11.",)),
    PackageMapping("Apache-Cassandra-4.x", ScriptPackageTarget("ApacheCassandra", "build_cassandra.sh"), ("4.",)),
    PackageMapping("Apache-Cassandra-5.x", ScriptPackageTarget("ApacheCassandra", "build_cassandra.sh"), ("5.",)),
    PackageMapping("Cassandra", ScriptPackageTarget("ApacheCassandra", "build_cassandra.sh")),
    PackageMapping("Apache-CloudStack", ScriptPackageTarget("CloudStack")),
    PackageMapping("Apache-HTTP-Server", ScriptPackageTarget("ApacheHttpServer")),
    PackageMapping("Apache-Kafka", ScriptPackageTarget("ApacheKafka")),
    PackageMapping("Apache-Spark", ScriptPackageTarget("ApacheSpark")),
    PackageMapping("Calico-3.x", ScriptPackageTarget("Calico"), ("3.",)),
    PackageMapping("Calico-with-Kubernetes", ScriptPackageTarget("Calico")),
    PackageMapping("Cruise-Control", ScriptPackageTarget("CruiseControl")),
    PackageMapping("Kubernetes-Minikube", ScriptPackageTarget("Minikube")),
    PackageMapping("MySQL-8.0.x", ScriptPackageTarget("MySQL"), ("8.0.",)),
    PackageMapping("MySQL-8.x", ScriptPackageTarget("MySQL"), ("8.",)),
    PackageMapping("MySQL-9.x", ScriptPackageTarget("MySQL"), ("9.",)),
    PackageMapping("PostgreSQL-16.x", ScriptPackageTarget("PostgreSQL"), ("16.",)),
    PackageMapping("PostgreSQL-17.x", ScriptPackageTarget("PostgreSQL"), ("17.",)),
    PackageMapping("PostgreSQL-18.x", ScriptPackageTarget("PostgreSQL"), ("18.",)),
    PackageMapping("Python-3.13.x", ScriptPackageTarget("Python3"), ("3.13.",)),
    PackageMapping("Python-3.14.x", ScriptPackageTarget("Python3"), ("3.14.",)),
    PackageMapping("Python-3.x", ScriptPackageTarget("Python3"), ("3.",)),
    PackageMapping("TensorFlow", ScriptPackageTarget("Tensorflow")),
    PackageMapping("TensorFlow-Serving", ScriptPackageTarget("TensorflowServing")),
    PackageMapping("TensorFlow-Transform", ScriptPackageTarget("TensorflowTransform")),
    PackageMapping("Zabbix-agent", ScriptPackageTarget("Zabbix", "build_zabbixagent.sh")),
    PackageMapping("Zabbix-server", ScriptPackageTarget("Zabbix", "build_zabbixserver.sh")),
    PackageMapping("htop", ScriptPackageTarget("Htop")),
    PackageMapping("sysdig", ScriptPackageTarget("Sysdig")),
    PackageMapping(
        "NGINX-Ingress-Controller",
        ScriptPackageTarget("NGINX-ingress-controller"),
    ),
)


def normalize_package_name(value: str) -> str:
    """Normalize only case and separators, preserving meaningful digits."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def canonical_wiki_package(value: str, version: str | None = None) -> str:
    """Return the canonical Wiki package for a user or webhook package name."""
    package = _clean_package_name(value)
    mapping = _mapping_for_wiki_package(package, version)
    if mapping is not None:
        return mapping.wiki_package

    reverse = _reverse_mappings(package, version=version)
    if reverse:
        return reverse[0].wiki_package
    return package


def script_target_for_package(
    wiki_package: str,
    version: str | None = None,
) -> ScriptPackageTarget:
    """Resolve a Wiki package to its scripts-repository directory/file."""
    package = _clean_package_name(wiki_package)
    custom = _custom_mappings().get(normalize_package_name(package))
    if custom is not None:
        return custom.target

    mapping = _mapping_for_wiki_package(package, version)
    if mapping is not None:
        return mapping.target

    # New packages with matching names are resolved by the GitHub client using
    # a case/punctuation-insensitive directory lookup.
    return ScriptPackageTarget(package)


def wiki_package_for_script_path(path: str) -> str:
    """Map a changed script path back to the canonical Wiki package."""
    parts = PurePosixPath(path).parts
    if len(parts) < 2:
        return parts[0] if parts else path

    directory = parts[0]
    version = parts[1]
    filename = parts[-1]
    candidates = [
        mapping
        for mapping in (*_custom_mappings().values(), *_KNOWN_MAPPINGS)
        if _mapping_matches(mapping, directory, filename, version)
    ]
    if candidates:
        candidates.sort(
            key=lambda mapping: (
                len(mapping.target.filename or ""),
                max((len(prefix) for prefix in mapping.version_prefixes), default=0),
            ),
            reverse=True,
        )
        return candidates[0].wiki_package
    return directory


def _clean_package_name(value: str) -> str:
    package = value.strip()
    if package.casefold().startswith("building-"):
        package = package[len("building-"):]
    return package.removesuffix(".md")


def _mapping_for_wiki_package(
    package: str,
    version: str | None,
) -> PackageMapping | None:
    normalized = normalize_package_name(package)
    candidates = [
        mapping
        for mapping in (*_custom_mappings().values(), *_KNOWN_MAPPINGS)
        if normalize_package_name(mapping.wiki_package) == normalized
        and _version_matches(mapping, version)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda mapping: len(mapping.version_prefixes))


def _reverse_mappings(
    directory: str,
    *,
    version: str | None,
) -> list[PackageMapping]:
    return [
        mapping
        for mapping in (*_custom_mappings().values(), *_KNOWN_MAPPINGS)
        if normalize_package_name(mapping.target.directory) == normalize_package_name(directory)
        and _version_matches(mapping, version)
    ]


def _mapping_matches(
    mapping: PackageMapping,
    directory: str,
    filename: str,
    version: str,
) -> bool:
    if normalize_package_name(mapping.target.directory) != normalize_package_name(directory):
        return False
    if mapping.target.filename and mapping.target.filename.casefold() != filename.casefold():
        return False
    return _version_matches(mapping, version)


def _version_matches(mapping: PackageMapping, version: str | None) -> bool:
    return not mapping.version_prefixes or (
        version is not None
        and any(version.startswith(prefix) for prefix in mapping.version_prefixes)
    )


def _custom_mappings() -> dict[str, PackageMapping]:
    """Read optional aliases without requiring a code change for new names.

    Format:
    {"Wiki-Name": {"directory": "ScriptsDir", "filename": "build.sh"}}
    """
    raw = os.getenv("BUILDGUARD_PACKAGE_ALIASES", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    mappings: dict[str, PackageMapping] = {}
    for wiki_package, value in payload.items():
        if not isinstance(wiki_package, str):
            continue
        if isinstance(value, str):
            directory, filename = value, None
        elif isinstance(value, dict) and isinstance(value.get("directory"), str):
            directory = value["directory"]
            filename = value.get("filename")
            if filename is not None and not isinstance(filename, str):
                continue
        else:
            continue
        mapping = PackageMapping(
            wiki_package=_clean_package_name(wiki_package),
            target=ScriptPackageTarget(directory, filename),
        )
        mappings[normalize_package_name(mapping.wiki_package)] = mapping
    return mappings
