"""StackOverflow scraper via HuggingFace dataset.

Streams the mikex86/stackoverflow-posts dataset, filters for C# tagged
posts with score >= 5, combines Q&A pairs, and feeds them through
Phi-4 extraction or directly exports as instruction pairs.
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

# Tag-to-category mapping for auto-detection
_TAG_CATEGORY_MAP: dict[str, Category] = {
    "asp.net-core": Category.ASPNET,
    "asp.net-mvc": Category.ASPNET,
    "asp.net": Category.ASPNET,
    "blazor": Category.BLAZOR,
    "entity-framework": Category.EF_CORE,
    "entity-framework-core": Category.EF_CORE,
    "ef-core": Category.EF_CORE,
    "maui": Category.MAUI,
    "xamarin": Category.MAUI,
    "signalr": Category.SIGNALR,
    "grpc": Category.GRPC,
    "ml.net": Category.MLNET,
    "azure": Category.AZURE_SDK,
    "msbuild": Category.MSBUILD,
    "roslyn": Category.ROSLYN,
    "nuget": Category.NUGET,
    "linq": Category.BCL,
    "system.text.json": Category.BCL,
    "collections": Category.BCL,
    "async-await": Category.LANGUAGE,
    "generics": Category.LANGUAGE,
    "pattern-matching": Category.LANGUAGE,
    "records": Category.LANGUAGE,
}

# C# version keywords for rough version detection
_VERSION_HINTS: list[tuple[str, CSharpVersion]] = [
    ("c#-14", CSharpVersion.V14_0),
    ("c#-13", CSharpVersion.V13_0),
    ("c#-12", CSharpVersion.V12_0),
    ("c#-11", CSharpVersion.V11_0),
    ("c#-10", CSharpVersion.V10_0),
    ("c#-9", CSharpVersion.V9_0),
    ("c#-8", CSharpVersion.V8_0),
    ("c#-7", CSharpVersion.V7_0),
    ("c#-6", CSharpVersion.V6_0),
    ("c#-5", CSharpVersion.V5_0),
    ("c#-4", CSharpVersion.V4_0),
    ("c#-3", CSharpVersion.V3_0),
    ("c#-2", CSharpVersion.V2_0),
    (".net-8", CSharpVersion.V12_0),
    (".net-7", CSharpVersion.V11_0),
    (".net-6", CSharpVersion.V10_0),
    (".net-5", CSharpVersion.V9_0),
    (".net-core-3", CSharpVersion.V8_0),
]


def _strip_html(html: str) -> str:
    """Simple HTML tag stripping."""
    return re.sub(r"<[^>]+>", "", html).strip()


def _tags_to_str(tags) -> str:
    """Normalize tags to a space-separated string."""
    if isinstance(tags, list):
        return " ".join(str(t).lower() for t in tags)
    if isinstance(tags, str):
        return tags.lower().replace("<", " ").replace(">", " ")
    return ""


def _detect_category(tags: str) -> Category:
    """Detect category from StackOverflow tags."""
    tag_list = tags.split()
    for tag in tag_list:
        if tag in _TAG_CATEGORY_MAP:
            return _TAG_CATEGORY_MAP[tag]
    return Category.LANGUAGE  # Default for general C# questions


def _detect_version(tags: str, title: str) -> CSharpVersion:
    """Rough version detection from tags and title."""
    text = (tags + " " + title).lower()
    for hint, version in _VERSION_HINTS:
        if hint in text:
            return version
    return CSharpVersion.V9_0  # Default to C# 9.0 (modern but safe)


def _detect_dotnet_version(tags: str) -> DotNetVersion:
    """Detect .NET version from tags."""
    text = tags.lower()
    if ".net-8" in text or "net8" in text:
        return DotNetVersion.NET_8
    if ".net-7" in text or "net7" in text:
        return DotNetVersion.NET_7
    if ".net-6" in text or "net6" in text:
        return DotNetVersion.NET_6
    if ".net-5" in text or "net5" in text:
        return DotNetVersion.NET_5
    if ".net-core" in text:
        return DotNetVersion.CORE_3_1
    if ".net-framework" in text:
        return DotNetVersion.FRAMEWORK_4_8
    return DotNetVersion.NET_8  # Default to modern


def _make_record_from_qa(
    title: str,
    question_body: str,
    answer_body: str,
    tags: str,
    score: int,
    post_id: int,
) -> DotNetRecord:
    """Convert a StackOverflow Q&A pair into a DotNetRecord."""
    category = _detect_category(tags)
    cs_version = _detect_version(tags, title)
    dn_version = _detect_dotnet_version(tags)

    # Clean HTML
    q_text = _strip_html(question_body)
    a_text = _strip_html(answer_body)

    # Extract code from answer (look for code blocks)
    code_blocks = re.findall(r"<code>(.*?)</code>", answer_body, re.DOTALL)
    code_snippet = "\n\n".join(_strip_html(c) for c in code_blocks[:3]) if code_blocks else a_text[:500]

    return DotNetRecord(
        id=str(uuid.uuid4()),
        category=category,
        csharp_version=cs_version,
        dotnet_version=dn_version,
        feature_name=title[:300],
        description=q_text[:500],
        code_snippet=code_snippet[:2000] if code_snippet else "// See answer",
        legacy_equivalent="N/A - StackOverflow Q&A",
        nuget_packages=[],
        source_url=f"https://stackoverflow.com/questions/{post_id}",
        validation_status=ValidationStatus.UNTESTED,
        tags=[t for t in tags.split() if t],
    )


def scrape_stackoverflow(
    config: Optional[SovereignConfig] = None,
    max_records: int = 50000,
    min_score: int = 5,
    min_answer_score: int = 3,
) -> list[DotNetRecord]:
    """Stream StackOverflow C# posts from HuggingFace and convert to records.

    Parameters
    ----------
    max_records : int
        Maximum number of records to extract (default 50K to keep manageable).
    min_score : int
        Minimum question score to include.
    min_answer_score : int
        Minimum answer score for non-accepted answers. Accepted answers
        (green tick) are always kept regardless of score.
    """
    from datasets import load_dataset

    cfg = config or get_config()
    logger.info("Streaming StackOverflow posts from HuggingFace...")

    records: list[DotNetRecord] = []
    questions: dict[int, dict] = {}  # post_id → question data
    answers: dict[int, dict] = {}    # parent_id → best answer data

    # Stream the dataset
    ds = load_dataset("mikex86/stackoverflow-posts", split="train", streaming=True)

    processed = 0
    for row in ds:
        processed += 1
        if processed % 100000 == 0:
            logger.info("Processed %d rows, found %d C# records so far...", processed, len(records))

        if len(records) >= max_records:
            break

        # Dataset uses capitalized column names; Tags is a list[str]
        raw_tags = row.get("Tags") or []
        tags = _tags_to_str(raw_tags)

        # Filter: must have C# tag
        if "c#" not in tags:
            continue

        post_type = row.get("PostTypeId", 0)
        score = row.get("Score", 0) or 0

        if post_type == 1:  # Question
            if score >= min_score:
                post_id = row.get("Id", 0)
                questions[post_id] = {
                    "title": row.get("Title", "") or "",
                    "body": row.get("Body", "") or "",
                    "tags": tags,
                    "score": score,
                    "accepted_answer_id": row.get("AcceptedAnswerId"),
                }

        elif post_type == 2:  # Answer
            parent_id = row.get("ParentId", 0)
            if parent_id in questions:
                existing = answers.get(parent_id)
                q = questions[parent_id]
                is_accepted = (row.get("Id") == q.get("accepted_answer_id"))

                # Accept: green-tick answers always, others only if score >= threshold
                if not is_accepted and score < min_answer_score:
                    continue

                # Keep accepted answer, or upgrade to higher-scored answer
                if existing is None or is_accepted or score > existing.get("score", 0):
                    answers[parent_id] = {
                        "body": row.get("Body", "") or "",
                        "score": score,
                        "is_accepted": is_accepted,
                    }

                # If we have both Q and A, create a record
                if parent_id in answers:
                    q = questions[parent_id]
                    a = answers[parent_id]
                    try:
                        record = _make_record_from_qa(
                            title=q["title"],
                            question_body=q["body"],
                            answer_body=a["body"],
                            tags=q["tags"],
                            score=q["score"],
                            post_id=parent_id,
                        )
                        records.append(record)
                    except Exception:
                        logger.debug("Failed to create record from post %d", parent_id)

                    # Clean up to save memory
                    del questions[parent_id]
                    if parent_id in answers:
                        del answers[parent_id]

    logger.info("StackOverflow: processed %d rows, extracted %d C# records", processed, len(records))
    return records
