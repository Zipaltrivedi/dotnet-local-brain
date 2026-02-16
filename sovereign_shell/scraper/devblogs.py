"""DevBlogs scraper — fetches .NET blog articles via sitemap.

Parses the devblogs.microsoft.com/dotnet/ sitemap to get all article URLs,
then scrapes each article with crawl4ai and feeds through Phi-4 extraction.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import Category, DotNetRecord
from sovereign_shell.scraper.parser import extract_features_from_text
from sovereign_shell.scraper.sentinel import _feature_to_record, _save_raw_html

logger = logging.getLogger(__name__)

SITEMAP_URL = "https://devblogs.microsoft.com/dotnet/sitemap.xml"

# URL patterns to category mapping
_URL_CATEGORY_HINTS: list[tuple[str, Category]] = [
    ("blazor", Category.BLAZOR),
    ("aspnet", Category.ASPNET),
    ("asp-net", Category.ASPNET),
    ("entity-framework", Category.EF_CORE),
    ("ef-core", Category.EF_CORE),
    ("maui", Category.MAUI),
    ("signalr", Category.SIGNALR),
    ("grpc", Category.GRPC),
    ("ml-net", Category.MLNET),
    ("machine-learning", Category.MLNET),
    ("azure", Category.AZURE_SDK),
    ("msbuild", Category.MSBUILD),
    ("roslyn", Category.ROSLYN),
    ("nuget", Category.NUGET),
    ("performance", Category.BCL),
    ("runtime", Category.BCL),
    ("csharp", Category.LANGUAGE),
    ("c-sharp", Category.LANGUAGE),
]


def _detect_category_from_url(url: str) -> Category:
    """Guess category from the article URL."""
    url_lower = url.lower()
    for hint, cat in _URL_CATEGORY_HINTS:
        if hint in url_lower:
            return cat
    return Category.LANGUAGE  # Default


def _fetch_sitemap_urls(sitemap_url: str = SITEMAP_URL) -> list[str]:
    """Fetch and parse the sitemap XML to extract article URLs."""
    logger.info("Fetching sitemap: %s", sitemap_url)

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(sitemap_url)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)

    # Handle sitemap index (points to sub-sitemaps)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []

    # Check if this is a sitemap index
    sub_sitemaps = root.findall(".//sm:sitemap/sm:loc", ns)
    if sub_sitemaps:
        # It's a sitemap index — fetch each sub-sitemap
        for sub in sub_sitemaps:
            sub_url = sub.text.strip() if sub.text else ""
            if not sub_url:
                continue
            logger.info("Fetching sub-sitemap: %s", sub_url)
            try:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    resp = client.get(sub_url)
                    resp.raise_for_status()
                sub_root = ET.fromstring(resp.text)
                for loc in sub_root.findall(".//sm:url/sm:loc", ns):
                    if loc.text:
                        urls.append(loc.text.strip())
            except Exception:
                logger.warning("Failed to fetch sub-sitemap: %s", sub_url)
    else:
        # Direct sitemap with URLs
        for loc in root.findall(".//sm:url/sm:loc", ns):
            if loc.text:
                urls.append(loc.text.strip())

    # Filter to only blog post URLs (not category/tag pages)
    blog_urls = [
        u for u in urls
        if "/dotnet/" in u
        and not u.endswith("/dotnet/")
        and "/page/" not in u
        and "/tag/" not in u
        and "/category/" not in u
        and "/author/" not in u
    ]

    logger.info("Found %d blog post URLs in sitemap", len(blog_urls))
    return blog_urls


async def scrape_devblogs(
    config: Optional[SovereignConfig] = None,
    max_articles: int = 0,
    delay: float = 1.0,
) -> list[DotNetRecord]:
    """Scrape DevBlogs articles and extract features.

    Parameters
    ----------
    max_articles : int
        Maximum articles to scrape (0 = all).
    delay : float
        Seconds between requests (courtesy rate limit).
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    cfg = config or get_config()
    urls = _fetch_sitemap_urls()

    if max_articles > 0:
        urls = urls[:max_articles]

    logger.info("Scraping %d DevBlogs articles...", len(urls))

    all_records: list[DotNetRecord] = []
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(urls):
            logger.info("DevBlogs %d/%d: %s", i + 1, len(urls), url)

            try:
                result = await crawler.arun(url=url, config=run_config)
            except Exception:
                logger.warning("Failed to crawl: %s", url)
                continue

            if not result.success:
                logger.warning("Crawl unsuccessful: %s", url)
                continue

            markdown = result.markdown or ""
            if len(markdown.strip()) < 200:
                continue

            # Cache raw content
            _save_raw_html(cfg.raw_html_dir, "devblogs", url, markdown)

            # Detect category
            category = _detect_category_from_url(url)

            # Extract features via Phi-4
            extraction = extract_features_from_text(
                text=markdown,
                category=category,
                source_url=url,
            )

            for feat in extraction.features:
                record = _feature_to_record(feat, category, url)
                if record:
                    all_records.append(record)

            # Courtesy delay
            if delay > 0:
                await asyncio.sleep(delay)

    logger.info("DevBlogs: scraped %d articles, extracted %d records", len(urls), len(all_records))
    return all_records
