"""The Stack v2 scraper — streams C# code from HuggingFace.

Uses bigcode/the-stack-v2-dedup (deduplicated, permissively licensed C# code
from GitHub). Filters for quality, samples strategically, and creates
DotNetRecords from source code files.

Requires: HuggingFace account + accepting the dataset license.
Auth: `huggingface-cli login` before first use.
"""

from __future__ import annotations

import logging
import re
import uuid
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

# Namespace-to-category mapping for auto-detection
_NAMESPACE_CATEGORY: list[tuple[str, Category]] = [
    ("Microsoft.AspNetCore", Category.ASPNET),
    ("Microsoft.Extensions", Category.ASPNET),
    ("Microsoft.EntityFrameworkCore", Category.EF_CORE),
    ("Microsoft.Maui", Category.MAUI),
    ("Xamarin", Category.MAUI),
    ("Microsoft.AspNetCore.SignalR", Category.SIGNALR),
    ("Grpc", Category.GRPC),
    ("Microsoft.ML", Category.MLNET),
    ("Azure", Category.AZURE_SDK),
    ("Microsoft.Build", Category.MSBUILD),
    ("Microsoft.CodeAnalysis", Category.ROSLYN),
    ("NuGet", Category.NUGET),
    ("System.Linq", Category.BCL),
    ("System.Collections", Category.BCL),
    ("System.IO", Category.BCL),
    ("System.Net", Category.BCL),
    ("System.Text.Json", Category.BCL),
    ("System.Threading", Category.BCL),
]


def _detect_category_from_code(content: str) -> Category:
    """Detect category from using statements and namespaces."""
    for pattern, cat in _NAMESPACE_CATEGORY:
        if pattern in content:
            return cat
    return Category.LANGUAGE


def _detect_version_from_code(content: str) -> CSharpVersion:
    """Rough C# version detection from language features used."""
    if "required " in content and "init;" in content:
        return CSharpVersion.V11_0
    if "record struct" in content:
        return CSharpVersion.V10_0
    if " record " in content:
        return CSharpVersion.V9_0
    if "switch {" in content or "switch\n{" in content:
        return CSharpVersion.V8_0
    if "=> " in content and ";" in content:
        return CSharpVersion.V7_0
    if "nameof(" in content or "?." in content:
        return CSharpVersion.V6_0
    if "async " in content:
        return CSharpVersion.V5_0
    return CSharpVersion.V9_0  # Default modern


def _is_quality_file(content: str) -> bool:
    """Check if a file has enough quality signals for training."""
    lines = content.split("\n")
    line_count = len(lines)

    # Too short or too long
    if line_count < 30 or line_count > 500:
        return False

    # Must have at least one class/struct/record/interface
    has_type = any(
        kw in content
        for kw in ["class ", "struct ", "record ", "interface ", "enum "]
    )
    if not has_type:
        return False

    # Should have some doc comments or meaningful comments
    has_docs = "///" in content or "// " in content
    has_public = "public " in content

    return has_docs or has_public


def _extract_feature_name(content: str) -> str:
    """Extract the main type name from C# source."""
    # Look for class/struct/record/interface declarations
    match = re.search(
        r"(?:public|internal|private|protected)\s+(?:abstract\s+|static\s+|sealed\s+|partial\s+)*"
        r"(?:class|struct|record|interface|enum)\s+(\w+)",
        content,
    )
    if match:
        return match.group(1)
    return "Unknown"


def scrape_the_stack(
    config: Optional[SovereignConfig] = None,
    max_records: int = 50000,
    dataset_name: str = "bigcode/the-stack-dedup",
) -> list[DotNetRecord]:
    """Stream C# files from The Stack and create records.

    Parameters
    ----------
    max_records : int
        Maximum records to extract.
    dataset_name : str
        HuggingFace dataset name. Default is v1 dedup which includes
        source content. v2-dedup is metadata-only (no content column).
    """
    from datasets import load_dataset

    cfg = config or get_config()
    logger.info("Streaming C# code from %s...", dataset_name)

    records: list[DotNetRecord] = []
    processed = 0
    skipped = 0

    # Dataset-specific C# partition paths
    _DATA_DIRS = {
        "bigcode/the-stack-dedup": "data/c-sharp",
        "bigcode/the-stack-v2-dedup": "data/C-Sharp",
        "bigcode/starcoderdata": "csharp",
    }
    data_dir = _DATA_DIRS.get(dataset_name)

    try:
        kwargs = {"split": "train", "streaming": True}
        if data_dir:
            kwargs["data_dir"] = data_dir
        ds = load_dataset(dataset_name, **kwargs)
    except Exception:
        try:
            # Fallback: no data_dir filter
            ds = load_dataset(dataset_name, split="train", streaming=True)
        except Exception:
            logger.exception("Failed to load %s. Make sure you're logged in: huggingface-cli login", dataset_name)
            return records

    for row in ds:
        processed += 1

        if processed % 10000 == 0:
            logger.info(
                "The Stack: processed %d files, kept %d, skipped %d",
                processed, len(records), skipped,
            )

        if len(records) >= max_records:
            break

        # Filter for C# (v2 has a lang column)
        lang = row.get("lang", "") or row.get("language", "")
        if lang and lang.lower() not in ("c#", "csharp"):
            continue

        content = row.get("content", "") or row.get("code", "") or ""
        if not content:
            continue

        # Quality filter
        if not _is_quality_file(content):
            skipped += 1
            continue

        category = _detect_category_from_code(content)
        cs_version = _detect_version_from_code(content)
        feature_name = _extract_feature_name(content)

        try:
            record = DotNetRecord(
                id=str(uuid.uuid4()),
                category=category,
                csharp_version=cs_version,
                dotnet_version=DotNetVersion.NET_8,  # Default — most modern OSS code
                feature_name=feature_name[:300],
                description=f"Open-source C# code from The Stack: {feature_name}",
                code_snippet=content[:3000],
                legacy_equivalent="N/A - open source code",
                nuget_packages=[],
                source_url=f"huggingface:{dataset_name}",
                validation_status=ValidationStatus.UNTESTED,
                tags=["the-stack", category.value, "open-source"],
            )
            records.append(record)
        except Exception:
            pass

    logger.info(
        "The Stack: processed %d files, extracted %d records (skipped %d low-quality)",
        processed, len(records), skipped,
    )
    return records
