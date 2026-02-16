"""Typer CLI for Sovereign Shell.

Commands:
  sovereign status   — Environment health check
  sovereign scrape   — Run Sentinel scraper pipeline
  sovereign coverage — Show coverage matrix
  sovereign chat     — Interactive RAG-powered Q&A
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sovereign_shell.config import get_config

app = typer.Typer(
    name="sovereign",
    help="Sovereign Shell — Local C# Master AI Coding Agent",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# sovereign status
# ---------------------------------------------------------------------------

@app.command()
def status():
    """Environment health check — verify all dependencies are available."""
    cfg = get_config()
    all_ok = True

    def _check(label: str, ok: bool, detail: str = ""):
        nonlocal all_ok
        icon = "[green]OK[/]" if ok else "[red]FAIL[/]"
        if not ok:
            all_ok = False
        msg = f"  [{('green' if ok else 'red')}]{icon}[/] {label}"
        if detail:
            msg += f"  [dim]{detail}[/]"
        console.print(msg)

    console.print("\n[bold]Sovereign Shell — Status[/]\n")

    # Python
    _check("Python", True, sys.version.split()[0])

    # .NET SDKs
    try:
        result = subprocess.run(
            ["dotnet", "--list-sdks"], capture_output=True, text=True, timeout=10
        )
        sdks = result.stdout.strip().split("\n") if result.returncode == 0 else []
        _check(".NET SDKs", len(sdks) > 0, f"{len(sdks)} installed")
        for sdk in sdks:
            console.print(f"      [dim]{sdk.strip()}[/]")
    except FileNotFoundError:
        _check(".NET SDK", False, "dotnet not found")

    # Model file
    model_exists = cfg.model_path.exists()
    size_mb = cfg.model_path.stat().st_size / (1024 * 1024) if model_exists else 0
    _check("Phi-4 GGUF", model_exists, f"{size_mb:.0f} MB" if model_exists else "not found")

    # llama-cpp-python
    try:
        from llama_cpp import Llama  # noqa: F401
        _check("llama-cpp-python", True)
    except ImportError:
        _check("llama-cpp-python", False, "not installed")

    # crawl4ai
    try:
        import crawl4ai  # noqa: F401
        _check("crawl4ai", True)
    except ImportError:
        _check("crawl4ai", False, "not installed")

    # sqlite-vec
    try:
        import sqlite_vec  # noqa: F401
        _check("sqlite-vec", True)
    except ImportError:
        _check("sqlite-vec", False, "not installed")

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                _check("GPU", True, f"{parts[0]} — {parts[2]} / {parts[1]} MB used")
            else:
                _check("GPU", True, result.stdout.strip())
        else:
            _check("GPU", False, "nvidia-smi failed")
    except FileNotFoundError:
        _check("GPU", False, "nvidia-smi not found")

    # Database
    db_exists = cfg.db_path.exists()
    _check("Database", db_exists or True, str(cfg.db_path) + (" (exists)" if db_exists else " (will be created)"))

    console.print()
    if all_ok:
        console.print("[bold green]All checks passed.[/]\n")
    else:
        console.print("[bold yellow]Some checks failed — see above.[/]\n")


# ---------------------------------------------------------------------------
# sovereign scrape
# ---------------------------------------------------------------------------

@app.command()
def scrape(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Single category to scrape (e.g. 'language', 'blazor')"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", "-m", help="Max pages per category"),
):
    """Run the Sentinel scraper pipeline to harvest .NET documentation."""
    from sovereign_shell.models.schemas import Category
    from sovereign_shell.scraper.sentinel import crawl_all, crawl_category, save_extracted_records

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    cfg = get_config()
    cfg.raw_html_dir.mkdir(parents=True, exist_ok=True)

    if category:
        try:
            cat = Category(category.lower())
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/]")
            console.print(f"Valid: {', '.join(c.value for c in Category)}")
            raise typer.Exit(1)

        console.print(f"\n[bold]Scraping category: {cat.value}[/]\n")
        records = asyncio.run(crawl_category(cat, cfg, max_pages))
    else:
        console.print("\n[bold]Scraping all categories[/]\n")
        cats = list(Category) if category is None else None
        records = asyncio.run(crawl_all(cfg, cats))

    if records:
        output = save_extracted_records(records, cfg)
        console.print(f"\n[green]Extracted {len(records)} records → {output}[/]\n")
    else:
        console.print("\n[yellow]No records extracted.[/]\n")


# ---------------------------------------------------------------------------
# sovereign validate
# ---------------------------------------------------------------------------

@app.command()
def validate(
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="JSON file with extracted records to validate"),
):
    """Validate extracted code snippets using dotnet build."""
    import json as json_mod
    from sovereign_shell.models.schemas import DotNetRecord
    from sovereign_shell.validator.dotnet_build import validate_batch

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    cfg = get_config()

    if input_file:
        path = cfg.project_root / input_file
    else:
        # Find the most recent extraction file
        extracted = sorted(cfg.extracted_dir.glob("extraction_*.json"), reverse=True)
        if not extracted:
            console.print("[red]No extraction files found. Run 'sovereign scrape' first.[/]")
            raise typer.Exit(1)
        path = extracted[0]

    console.print(f"\n[bold]Validating records from: {path.name}[/]\n")

    data = json_mod.loads(path.read_text(encoding="utf-8"))
    records = [DotNetRecord(**r) for r in data]

    validated = validate_batch(records, cfg)

    # Save validated results
    output_path = path.with_stem(path.stem + "_validated")
    out_data = [r.model_dump(mode="json") for r in validated]
    output_path.write_text(json_mod.dumps(out_data, indent=2), encoding="utf-8")

    passed = sum(1 for r in validated if r.validation_status.value == "compiles")
    failed = sum(1 for r in validated if r.validation_status.value == "fails")

    console.print(f"\n[green]{passed} compiled[/]  [red]{failed} failed[/]  Total: {len(validated)}")
    console.print(f"Results saved to: {output_path}\n")


# ---------------------------------------------------------------------------
# sovereign coverage
# ---------------------------------------------------------------------------

@app.command()
def coverage():
    """Display the coverage matrix as a Rich table."""
    from sovereign_shell.models.coverage import load_coverage

    matrix = load_coverage()

    if matrix.total_records == 0:
        console.print("\n[yellow]No records yet. Run 'sovereign scrape' first.[/]\n")
        return

    console.print(f"\n[bold]Coverage Matrix[/]  [dim](updated: {matrix.last_updated or 'never'})[/]\n")

    # By C# Version
    table = Table(title="By C# Version")
    table.add_column("Version", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Validated", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Completion", justify="right")

    for entry in sorted(matrix.by_version, key=lambda e: e.key):
        pct = entry.completion_pct
        pct_style = "green" if pct >= 80 else "yellow" if pct >= 40 else "red"
        table.add_row(
            f"C# {entry.key}",
            str(entry.total_features),
            str(entry.features_validated),
            str(entry.features_failed),
            f"[{pct_style}]{pct:.0f}%[/]",
        )

    console.print(table)
    console.print()

    # By Category
    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Validated", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Completion", justify="right")

    for entry in sorted(matrix.by_category, key=lambda e: e.key):
        pct = entry.completion_pct
        pct_style = "green" if pct >= 80 else "yellow" if pct >= 40 else "red"
        table.add_row(
            entry.key,
            str(entry.total_features),
            str(entry.features_validated),
            str(entry.features_failed),
            f"[{pct_style}]{pct:.0f}%[/]",
        )

    console.print(table)

    console.print(
        f"\n[bold]Totals:[/] {matrix.total_records} records, "
        f"{matrix.total_validated} validated "
        f"({matrix.total_validated / matrix.total_records * 100:.0f}% overall)\n"
        if matrix.total_records > 0 else ""
    )


# ---------------------------------------------------------------------------
# sovereign chat
# ---------------------------------------------------------------------------

@app.command()
def chat():
    """Interactive RAG-powered Q&A with the local Phi-4 model."""
    from sovereign_shell.inference.phi4 import Phi4Engine
    from sovereign_shell.inference.prompts import build_chat_messages

    console.print("\n[bold]Sovereign Shell — Chat Mode[/]")
    console.print("[dim]Type 'exit' or 'quit' to leave. Model loading...[/]\n")

    engine = Phi4Engine.get()
    engine.load()
    console.print("[green]Model loaded.[/]\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("[bold cyan]You>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        messages = build_chat_messages(
            user_message=user_input,
            history=history[-6:],  # Last 3 exchanges
        )

        with console.status("[dim]Thinking...[/]"):
            response = engine.chat(messages=messages, max_tokens=2048, temperature=0.7)

        console.print(f"\n[bold green]Sovereign>[/] {response}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

    engine.unload()
    console.print("\n[dim]Session ended. VRAM released.[/]\n")


if __name__ == "__main__":
    app()
