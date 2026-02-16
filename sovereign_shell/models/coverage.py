"""Coverage matrix logic — tracks scraping and validation progress.

Reads records from the database (or in-memory list) and computes
per-version and per-category completion stats. Persists to
data/coverage_matrix.json for the CLI dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import (
    Category,
    CoverageEntry,
    CoverageMatrix,
    CSharpVersion,
    DotNetRecord,
    ValidationStatus,
)


def compute_coverage(records: list[DotNetRecord]) -> CoverageMatrix:
    """Compute coverage stats from a list of records."""
    # Per C# version
    by_version: dict[str, CoverageEntry] = {}
    for v in CSharpVersion:
        by_version[v.value] = CoverageEntry(key=v.value)

    # Per category
    by_category: dict[str, CoverageEntry] = {}
    for c in Category:
        by_category[c.value] = CoverageEntry(key=c.value)

    for record in records:
        ver_key = record.csharp_version.value
        cat_key = record.category.value

        # Version stats
        entry = by_version[ver_key]
        entry.total_features += 1
        if record.validation_status == ValidationStatus.COMPILES:
            entry.features_validated += 1
        elif record.validation_status == ValidationStatus.FAILS:
            entry.features_failed += 1

        # Category stats
        entry = by_category[cat_key]
        entry.total_features += 1
        if record.validation_status == ValidationStatus.COMPILES:
            entry.features_validated += 1
        elif record.validation_status == ValidationStatus.FAILS:
            entry.features_failed += 1

    total_records = len(records)
    total_validated = sum(
        1 for r in records if r.validation_status == ValidationStatus.COMPILES
    )

    return CoverageMatrix(
        last_updated=datetime.now(timezone.utc).isoformat(),
        by_version=[e for e in by_version.values() if e.total_features > 0],
        by_category=[e for e in by_category.values() if e.total_features > 0],
        total_records=total_records,
        total_validated=total_validated,
    )


def save_coverage(
    matrix: CoverageMatrix,
    config: Optional[SovereignConfig] = None,
) -> Path:
    """Save coverage matrix to JSON file."""
    cfg = config or get_config()
    cfg.coverage_path.parent.mkdir(parents=True, exist_ok=True)

    data = matrix.model_dump(mode="json")
    cfg.coverage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cfg.coverage_path


def load_coverage(
    config: Optional[SovereignConfig] = None,
) -> CoverageMatrix:
    """Load coverage matrix from JSON file."""
    cfg = config or get_config()
    if not cfg.coverage_path.exists():
        return CoverageMatrix()

    data = json.loads(cfg.coverage_path.read_text(encoding="utf-8"))
    return CoverageMatrix(**data)
