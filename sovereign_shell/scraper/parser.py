"""Parse raw HTML/markdown into structured DotNetRecords via Phi-4.

Flow: markdown text → chunk → Phi-4 extraction → parse JSON → ExtractedFeature list
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sovereign_shell.inference.phi4 import Phi4Engine
from sovereign_shell.inference.prompts import build_extraction_messages
from sovereign_shell.models.schemas import (
    Category,
    ExtractedFeature,
    ExtractionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = 3000,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks that fit the model context.

    Uses paragraph boundaries where possible to avoid splitting mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at a paragraph boundary
        if end < len(text):
            # Look for double newline near the end of the chunk
            break_point = text.rfind("\n\n", start + max_chars // 2, end)
            if break_point != -1:
                end = break_point + 2  # Include the newlines
            else:
                # Fall back to single newline
                break_point = text.rfind("\n", start + max_chars // 2, end)
                if break_point != -1:
                    end = break_point + 1

        chunks.append(text[start:end].strip())
        start = end - overlap  # Overlap for context continuity

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------

def _parse_extraction_json(raw: str) -> list[dict]:
    """Attempt to parse Phi-4's JSON output, handling common formatting issues."""
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                data = json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                logger.warning("Failed to parse extraction JSON")
                return []
        else:
            return []

    if isinstance(data, dict) and "features" in data:
        return data["features"]
    if isinstance(data, list):
        return data
    return []


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features_from_text(
    text: str,
    category: Category,
    source_url: str,
    engine: Optional[Phi4Engine] = None,
) -> ExtractionResult:
    """Extract structured features from documentation text using Phi-4.

    Chunks the text, sends each chunk to Phi-4 for extraction, and
    merges results into a single ExtractionResult.
    """
    engine = engine or Phi4Engine.get()
    all_features: list[ExtractedFeature] = []

    chunks = chunk_text(text)
    logger.info("Processing %d chunks for %s from %s", len(chunks), category.value, source_url)

    for i, chunk in enumerate(chunks):
        messages = build_extraction_messages(
            content=chunk,
            category=category.value,
            source_url=source_url,
        )

        try:
            raw_output = engine.chat(
                messages=messages,
                max_tokens=2048,
                temperature=0.1,  # Low temp for structured extraction
            )
        except Exception:
            logger.exception("Phi-4 extraction failed on chunk %d", i)
            continue

        raw_features = _parse_extraction_json(raw_output)

        for feat_dict in raw_features:
            try:
                feature = ExtractedFeature(**feat_dict)
                all_features.append(feature)
            except Exception:
                logger.warning("Skipping malformed feature in chunk %d: %s", i, feat_dict.get("feature_name", "?"))

    # Deduplicate by feature_name
    seen: set[str] = set()
    unique: list[ExtractedFeature] = []
    for f in all_features:
        key = f.feature_name.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    logger.info("Extracted %d unique features from %s", len(unique), source_url)
    return ExtractionResult(features=unique)
