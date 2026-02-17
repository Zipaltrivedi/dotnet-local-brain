"""Sentinel — the BFS web crawler for harvesting .NET documentation.

Uses crawl4ai's AsyncWebCrawler with BFS deep-crawl strategy to
recursively scrape Microsoft Learn docs, then feeds the content
through Phi-4 extraction to produce DotNetRecords.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import (
    Category,
    DotNetRecord,
    ExtractedFeature,
    ValidationStatus,
)
from sovereign_shell.scraper.sources import get_seeds

logger = logging.getLogger(__name__)


def _feature_to_record(
    feature: ExtractedFeature,
    category: Category,
    source_url: str,
) -> Optional[DotNetRecord]:
    """Convert an ExtractedFeature into a full DotNetRecord.

    Returns None if version enums can't be resolved.
    """
    from sovereign_shell.models.schemas import CSharpVersion, DotNetVersion

    # Try to match csharp_version string to enum
    cs_ver = None
    for v in CSharpVersion:
        if v.value == feature.csharp_version:
            cs_ver = v
            break
    if cs_ver is None:
        logger.warning("Unknown C# version '%s' for feature '%s'", feature.csharp_version, feature.feature_name)
        return None

    # Try to match dotnet_version string to enum
    dn_ver = None
    for v in DotNetVersion:
        if v.value == feature.dotnet_version:
            dn_ver = v
            break
    if dn_ver is None:
        logger.warning("Unknown .NET version '%s' for feature '%s'", feature.dotnet_version, feature.feature_name)
        return None

    return DotNetRecord(
        id=str(uuid.uuid4()),
        category=category,
        csharp_version=cs_ver,
        dotnet_version=dn_ver,
        feature_name=feature.feature_name,
        description=feature.description or "No description extracted.",
        code_snippet=feature.code_snippet or "// No code extracted",
        legacy_equivalent=feature.legacy_equivalent or "N/A",
        nuget_packages=feature.nuget_packages,
        source_url=source_url,
        validation_status=ValidationStatus.UNTESTED,
        tags=feature.tags,
    )


def _save_raw_html(
    html_dir: Path,
    category: str,
    url: str,
    content: str,
) -> None:
    """Cache raw markdown content to disk for reproducibility."""
    cat_dir = html_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Create a filename from the URL
    safe_name = url.replace("https://", "").replace("http://", "")
    safe_name = safe_name.replace("/", "_").replace("?", "_")[:150]
    safe_name = safe_name.rstrip("_") + ".md"

    (cat_dir / safe_name).write_text(content, encoding="utf-8")


async def crawl_category(
    category: Category,
    config: Optional[SovereignConfig] = None,
    max_pages: Optional[int] = None,
) -> list[DotNetRecord]:
    """Crawl a single category using BFS deep-crawl and extract features.

    Returns a list of DotNetRecords (unvalidated) extracted from the crawled pages.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

    cfg = config or get_config()
    seeds = get_seeds(category).get(category, [])
    if not seeds:
        logger.warning("No seed URLs for category %s", category.value)
        return []

    page_limit = max_pages or cfg.max_pages_per_category

    # BFS strategy: crawl up to depth=3 from each seed
    deep_strategy = BFSDeepCrawlStrategy(
        max_depth=cfg.crawl_depth,
        include_external=False,
        max_pages=page_limit,
    )

    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_strategy,
        # Only follow links within Microsoft Learn
        # crawl4ai handles this via include_external=False
    )

    all_records: list[DotNetRecord] = []
    pages_fetched = 0

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for seed_url in seeds:
            if pages_fetched >= page_limit:
                break

            logger.info("Crawling seed: %s (category: %s)", seed_url, category.value)

            try:
                results = await crawler.arun(
                    url=seed_url,
                    config=run_config,
                )
            except Exception:
                logger.exception("Crawl failed for %s", seed_url)
                continue

            # crawl4ai returns a list of results for deep crawl
            if not isinstance(results, list):
                results = [results]

            for result in results:
                if pages_fetched >= page_limit:
                    break

                if not result.success:
                    logger.warning("Failed to fetch: %s", result.url)
                    continue

                markdown = result.markdown or ""
                if len(markdown.strip()) < 100:
                    continue  # Skip near-empty pages

                pages_fetched += 1
                url = result.url or seed_url

                # Cache raw content
                _save_raw_html(cfg.raw_html_dir, category.value, url, markdown)

                # Extract features via Phi-4
                from sovereign_shell.scraper.parser import extract_features_from_text
                extraction = extract_features_from_text(
                    text=markdown,
                    category=category,
                    source_url=url,
                )

                for feat in extraction.features:
                    record = _feature_to_record(feat, category, url)
                    if record:
                        all_records.append(record)

    logger.info(
        "Category %s: fetched %d pages, extracted %d records",
        category.value,
        pages_fetched,
        len(all_records),
    )
    return all_records


async def crawl_all(
    config: Optional[SovereignConfig] = None,
    categories: Optional[list[Category]] = None,
) -> list[DotNetRecord]:
    """Crawl all (or specified) categories and return combined records."""
    cfg = config or get_config()
    targets = categories or list(Category)
    all_records: list[DotNetRecord] = []

    for cat in targets:
        records = await crawl_category(cat, cfg)
        all_records.extend(records)

    logger.info("Total: crawled %d categories, extracted %d records", len(targets), len(all_records))
    return all_records


def save_extracted_records(
    records: list[DotNetRecord],
    config: Optional[SovereignConfig] = None,
) -> Path:
    """Save extracted records to a JSON file in the extracted/ directory."""
    cfg = config or get_config()
    cfg.extracted_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = cfg.extracted_dir / f"extraction_{timestamp}.json"

    data = [r.model_dump(mode="json") for r in records]
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    logger.info("Saved %d records to %s", len(records), output_path)
    return output_path
