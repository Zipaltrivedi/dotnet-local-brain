# Sovereign Shell - Project Standards

## Project Goal

Build a training dataset spanning ALL C# versions (1.0 through 14.0) and the entire .NET ecosystem,
then fine-tune Phi-4 into a C#-only master model that runs locally on an RTX 5070 Ti (16GB VRAM).

The model will serve as a coding agent via OpenAI-compatible API for use with OpenCode and Claude Code.

## C# Standards

### When generating NEW code (project tooling, tests, etc.)
- Use C# 14 / .NET 10 idioms:
  - Extension members (not static helper classes)
  - `field` keyword for field-backed properties
  - Collection expressions `[...]` over `new List<T> { }`
  - Primary constructors over boilerplate ctor + fields
  - Raw string literals for multi-line strings
  - File-scoped namespaces (one per file)
  - Null-conditional assignment `??=`
  - `Span<T>` implicit conversions over array copies
  - Pattern matching (list patterns, property patterns)
  - Global using directives
- TargetFramework: `net10.0`
- LangVersion: `14.0`
- Nullable: enable
- ImplicitUsings: enable

### When scraping/validating dataset records
- Respect each version's era-appropriate syntax
- C# 2.0 snippets use `List<T>`, not collection expressions
- C# 7.0 snippets use traditional switch, not switch expressions
- Each record's LangVersion must match its claimed csharp_version
- Older versions use traditional namespaces (not file-scoped)

## .NET SDK Validation
- Multiple SDKs installed: .NET 6, 8, 9, 10
- Validate each snippet against its natural TargetFramework
- Framework-specific snippets (Blazor, EF Core) include NuGet package references

## Python Standards
- Python 3.14 (fallback to 3.12 venv if wheel compatibility issues arise)
- Type hints required on all public functions
- Pydantic v2 for all schemas
- async/await for I/O-bound operations (scraping)
- Rich for console output
- Typer for CLI

## VRAM Budget (RTX 5070 Ti - 16GB)
- Phi-4 Q4_K_M inference: ~6GB (model + KV cache at 4096 ctx)
- Never load multiple models simultaneously
- For training (QLoRA): ~14GB (4-bit base + LoRA + optimizer + activations)
- Always check `nvidia-smi` after model load to verify budget

## Architecture
- `sovereign_shell/` - main Python package
- Typer CLI: scrape, chat, coverage, status, export, train, eval, serve
- crawl4ai for web scraping (14 documentation categories)
- llama-cpp-python for local Phi-4 inference (GGUF format)
- sqlite-vec for vector storage
- SQLite-backed knowledge graph for GraphRAG
- All inference is LOCAL - no cloud API calls

## Key Files
- `sovereign_shell/models/schemas.py` - Pydantic models (every module depends on this)
- `sovereign_shell/inference/phi4.py` - Phi-4 engine singleton
- `sovereign_shell/scraper/sentinel.py` - Main scraping pipeline
- `sovereign_shell/validator/dotnet_build.py` - Multi-SDK compilation validation
- `sovereign_shell/memory/vectordb.py` - Vector storage
- `sovereign_shell/memory/graphdb.py` - Knowledge graph
- `sovereign_shell/memory/graph_rag.py` - Hybrid retrieval
