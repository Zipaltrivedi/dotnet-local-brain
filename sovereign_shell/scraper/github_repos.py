"""GitHub repos scraper — clones official dotnet/* repositories.

Shallow-clones key .NET repos, walks .cs files, extracts class/method
signatures with doc comments, and feeds through Phi-4 for feature extraction.
"""

from __future__ import annotations

import logging
import re
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import (
    Category,
    CSharpVersion,
    DotNetRecord,
    DotNetVersion,
    ValidationStatus,
)
logger = logging.getLogger(__name__)

# Repos to clone with their category mapping
REPOS: list[dict] = [
    {"url": "https://github.com/dotnet/runtime.git", "name": "runtime", "category": Category.BCL},
    {"url": "https://github.com/dotnet/aspnetcore.git", "name": "aspnetcore", "category": Category.ASPNET},
    {"url": "https://github.com/dotnet/roslyn.git", "name": "roslyn", "category": Category.ROSLYN},
    {"url": "https://github.com/dotnet/efcore.git", "name": "efcore", "category": Category.EF_CORE},
    {"url": "https://github.com/dotnet/maui.git", "name": "maui", "category": Category.MAUI},
    {"url": "https://github.com/dotnet/csharplang.git", "name": "csharplang", "category": Category.LANGUAGE},
]

# Patterns to skip (generated code, tests, build artifacts)
_SKIP_PATTERNS = [
    r"[\\/]obj[\\/]",
    r"[\\/]bin[\\/]",
    r"[\\/]artifacts[\\/]",
    r"\.Designer\.cs$",
    r"\.generated\.cs$",
    r"\.g\.cs$",
    r"AssemblyInfo\.cs$",
    r"GlobalUsings\.cs$",
    r"[\\/]ref[\\/]",        # Reference assemblies
]

_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS))


def _clone_repo(url: str, dest: Path) -> bool:
    """Shallow clone a repo. Returns True if successful."""
    if dest.exists():
        logger.info("Repo already cloned: %s", dest)
        return True

    logger.info("Cloning %s -> %s", url, dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return dest.exists()
    except Exception:
        logger.exception("Failed to clone %s", url)
        return False


def _should_skip(path: Path) -> bool:
    """Check if a file should be skipped."""
    return bool(_SKIP_RE.search(str(path)))


def _extract_cs_summary(content: str, file_path: str) -> str:
    """Extract a meaningful summary from a .cs file for Phi-4 extraction.

    Focuses on: XML doc comments, class/interface/record declarations,
    public method signatures. Truncates to ~2500 chars.
    """
    lines = content.split("\n")
    summary_parts: list[str] = []

    # Include file path context
    summary_parts.append(f"// File: {file_path}")

    # Extract using statements (first 10)
    usings = [l.strip() for l in lines if l.strip().startswith("using ")][:10]
    summary_parts.extend(usings)
    summary_parts.append("")

    # Extract namespace, class/interface/record declarations, and doc comments
    in_doc_comment = False
    doc_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        # XML doc comments
        if stripped.startswith("///"):
            in_doc_comment = True
            doc_buffer.append(stripped)
            continue

        if in_doc_comment:
            in_doc_comment = False
            summary_parts.extend(doc_buffer)
            doc_buffer = []

        # Important declarations
        if any(kw in stripped for kw in [
            "namespace ", "public class ", "public interface ", "public record ",
            "public struct ", "public enum ", "public abstract ",
            "public static ", "public async ", "internal class ",
        ]):
            summary_parts.append(stripped)

        # Public method signatures
        if "public " in stripped and "(" in stripped and ")" in stripped:
            summary_parts.append(stripped)

    result = "\n".join(summary_parts)
    return result[:2500]


def _extract_proposal_summary(content: str) -> str:
    """Extract summary from a csharplang proposal markdown file."""
    # These are markdown files with feature descriptions
    return content[:3000]


def scrape_github_repos(
    config: Optional[SovereignConfig] = None,
    max_files_per_repo: int = 500,
    include_tests: bool = False,
) -> list[DotNetRecord]:
    """Clone repos and extract features from .cs files.

    Parameters
    ----------
    max_files_per_repo : int
        Maximum .cs files to process per repository.
    include_tests : bool
        Whether to include test files.
    """
    cfg = config or get_config()
    repos_dir = cfg.project_root / "data" / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[DotNetRecord] = []

    for repo_info in REPOS:
        repo_name = repo_info["name"]
        repo_dir = repos_dir / repo_name
        category = repo_info["category"]

        # Clone
        if not _clone_repo(repo_info["url"], repo_dir):
            logger.warning("Skipping %s — clone failed", repo_name)
            continue

        # Special handling for csharplang (markdown proposals)
        if repo_name == "csharplang":
            proposals_dir = repo_dir / "proposals"
            if proposals_dir.exists():
                md_files = list(proposals_dir.glob("**/*.md"))[:max_files_per_repo]
                logger.info("Processing %d proposal files from csharplang", len(md_files))

                for md_file in md_files:
                    try:
                        content = md_file.read_text(encoding="utf-8", errors="ignore")
                        if len(content.strip()) < 200:
                            continue

                        summary = _extract_proposal_summary(content)
                        from sovereign_shell.scraper.parser import extract_features_from_text
                        extraction = extract_features_from_text(
                            text=summary,
                            category=Category.LANGUAGE,
                            source_url=f"https://github.com/dotnet/csharplang/blob/main/{md_file.relative_to(repo_dir)}",
                        )
                        for feat in extraction.features:
                            from sovereign_shell.scraper.sentinel import _feature_to_record
                            record = _feature_to_record(feat, Category.LANGUAGE, f"github:csharplang/{md_file.name}")
                            if record:
                                all_records.append(record)
                    except Exception:
                        logger.debug("Failed to process %s", md_file)
            continue

        # Walk .cs files
        cs_files = list(repo_dir.glob("**/*.cs"))
        logger.info("Found %d .cs files in %s", len(cs_files), repo_name)

        # Filter
        filtered = []
        for f in cs_files:
            if _should_skip(f):
                continue
            if not include_tests and ("test" in str(f).lower() or "Test" in f.name):
                continue
            filtered.append(f)

        # Limit
        filtered = filtered[:max_files_per_repo]
        logger.info("Processing %d files from %s (after filtering)", len(filtered), repo_name)

        for i, cs_file in enumerate(filtered):
            if (i + 1) % 100 == 0:
                logger.info("  %s: %d/%d files processed, %d records so far", repo_name, i + 1, len(filtered), len(all_records))

            try:
                content = cs_file.read_text(encoding="utf-8", errors="ignore")
                if len(content.strip()) < 100:
                    continue

                summary = _extract_cs_summary(content, str(cs_file.relative_to(repo_dir)))

                # Create a direct record from the source code
                # instead of Phi-4 extraction for speed
                record = DotNetRecord(
                    id=str(uuid.uuid4()),
                    category=category,
                    csharp_version=CSharpVersion.V12_0,  # Most dotnet/* code is modern
                    dotnet_version=DotNetVersion.NET_9,
                    feature_name=cs_file.stem[:300],
                    description=f"Source from dotnet/{repo_name}: {cs_file.relative_to(repo_dir)}",
                    code_snippet=content[:3000],
                    legacy_equivalent="N/A - official .NET source",
                    nuget_packages=[],
                    source_url=f"https://github.com/dotnet/{repo_name}/blob/main/{cs_file.relative_to(repo_dir)}",
                    validation_status=ValidationStatus.UNTESTED,
                    tags=[repo_name, category.value, "source-code"],
                )
                all_records.append(record)

            except Exception:
                logger.debug("Failed to process %s", cs_file)

    logger.info("GitHub repos: processed %d repos, extracted %d records", len(REPOS), len(all_records))
    return all_records
