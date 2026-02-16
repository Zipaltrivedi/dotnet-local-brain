"""HuggingFace instruction datasets scraper.

Streams pre-processed instruction datasets from HuggingFace,
filters for C# related content, and converts to DotNetRecords.

These datasets are already in Q&A / instruction format, so they
can be used both as training records and as direct training export.
"""

from __future__ import annotations

import logging
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

# Keywords that indicate C# content
_CSHARP_KEYWORDS = {
    "c#", "csharp", "dotnet", ".net", "asp.net", "blazor", "entity framework",
    "ef core", "linq", "wpf", "winforms", "xamarin", "maui", "signalr",
    "roslyn", "nuget", "msbuild", "razor", "minimal api",
}


def _is_csharp_content(text: str) -> bool:
    """Check if text is related to C#/.NET."""
    lower = text.lower()
    return any(kw in lower for kw in _CSHARP_KEYWORDS)


def scrape_stack_exchange_instruction(
    config: Optional[SovereignConfig] = None,
    max_records: int = 30000,
) -> list[DotNetRecord]:
    """Stream ArmelR/stack-exchange-instruction and filter for C#.

    This dataset has Q&A pairs already in instruction format.
    """
    from datasets import load_dataset

    cfg = config or get_config()
    logger.info("Streaming ArmelR/stack-exchange-instruction...")

    records: list[DotNetRecord] = []

    try:
        try:
            ds = load_dataset("ArmelR/stack-exchange-instruction", split="train", streaming=True)
        except ValueError:
            # Some datasets only have 'test' split
            ds = load_dataset("ArmelR/stack-exchange-instruction", split="test", streaming=True)
    except Exception:
        logger.exception("Failed to load ArmelR/stack-exchange-instruction")
        return records

    processed = 0
    for row in ds:
        processed += 1
        if processed % 50000 == 0:
            logger.info("Processed %d rows, found %d C# records...", processed, len(records))

        if len(records) >= max_records:
            break

        # Check if it's C# related
        question = row.get("question", "") or ""
        answer = row.get("response", "") or row.get("answer", "") or ""
        title = row.get("title", "") or ""

        combined = f"{title} {question} {answer}"
        if not _is_csharp_content(combined):
            continue

        # Create record
        try:
            record = DotNetRecord(
                id=str(uuid.uuid4()),
                category=Category.LANGUAGE,
                csharp_version=CSharpVersion.V9_0,  # Default
                dotnet_version=DotNetVersion.NET_8,
                feature_name=(title or question[:200])[:300],
                description=question[:500],
                code_snippet=answer[:3000],
                legacy_equivalent="N/A - instruction dataset",
                nuget_packages=[],
                source_url="huggingface:ArmelR/stack-exchange-instruction",
                validation_status=ValidationStatus.UNTESTED,
                tags=["instruction", "stack-exchange"],
            )
            records.append(record)
        except Exception:
            pass

    logger.info("Stack Exchange Instruction: processed %d rows, extracted %d C# records", processed, len(records))
    return records


def scrape_stack_exchange_preferences(
    config: Optional[SovereignConfig] = None,
    max_records: int = 10000,
) -> list[DotNetRecord]:
    """Stream HuggingFaceH4/stack-exchange-preferences for RLHF data."""
    from datasets import load_dataset

    cfg = config or get_config()
    logger.info("Streaming HuggingFaceH4/stack-exchange-preferences...")

    records: list[DotNetRecord] = []

    try:
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
    except Exception:
        logger.exception("Failed to load stack-exchange-preferences")
        return records

    processed = 0
    for row in ds:
        processed += 1
        if processed % 50000 == 0:
            logger.info("Processed %d rows, found %d C# records...", processed, len(records))

        if len(records) >= max_records:
            break

        question = row.get("question", "") or ""
        # Get the best answer
        answers = row.get("answers", []) or []
        if not answers:
            continue

        # Sort by score, pick best
        best = sorted(answers, key=lambda a: a.get("pm_score", 0), reverse=True)[0]
        answer_text = best.get("text", "") or ""

        combined = f"{question} {answer_text}"
        if not _is_csharp_content(combined):
            continue

        try:
            record = DotNetRecord(
                id=str(uuid.uuid4()),
                category=Category.LANGUAGE,
                csharp_version=CSharpVersion.V9_0,
                dotnet_version=DotNetVersion.NET_8,
                feature_name=question[:300],
                description=question[:500],
                code_snippet=answer_text[:3000],
                legacy_equivalent="N/A - preference dataset",
                nuget_packages=[],
                source_url="huggingface:HuggingFaceH4/stack-exchange-preferences",
                validation_status=ValidationStatus.UNTESTED,
                tags=["preferences", "rlhf", "stack-exchange"],
            )
            records.append(record)
        except Exception:
            pass

    logger.info("Stack Exchange Preferences: processed %d rows, extracted %d C# records", processed, len(records))
    return records


def scrape_hf_datasets(
    config: Optional[SovereignConfig] = None,
    max_records: int = 30000,
) -> list[DotNetRecord]:
    """Scrape all HuggingFace instruction datasets."""
    records: list[DotNetRecord] = []

    records.extend(scrape_stack_exchange_instruction(config, max_records))
    remaining = max_records - len(records)
    if remaining > 0:
        records.extend(scrape_stack_exchange_preferences(config, min(remaining, 10000)))

    logger.info("HF Datasets total: %d C# records", len(records))
    return records
