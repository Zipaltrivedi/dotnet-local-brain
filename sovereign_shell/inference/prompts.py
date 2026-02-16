"""Prompt templates for Phi-4 extraction and chat.

Two main prompt types:
1. EXTRACTION — converts raw HTML/markdown into structured JSON features
2. CHAT — interactive Q&A with RAG context injected
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Extraction prompt — used by the scraper to parse docs into DotNetRecords
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are a C#/.NET documentation analyst. Your job is to extract structured \
feature records from documentation text.

Rules:
- Output ONLY valid JSON — no markdown fences, no commentary.
- Each feature must include: feature_name, csharp_version, dotnet_version, \
description, code_snippet, legacy_equivalent.
- If the text does not contain extractable C#/.NET features, return: {"features": []}
- For csharp_version use format "X.0" (e.g. "9.0", "14.0").
- For dotnet_version use full name (e.g. ".NET 8.0", ".NET Framework 4.8").
- code_snippet must be compilable C# — include necessary usings.
- legacy_equivalent should show how this was done in the previous version. \
If this is a v1.0 feature or there is no prior equivalent, write "N/A — original feature".
- nuget_packages: list any required NuGet packages. Empty list if none.
- tags: relevant keywords (e.g. ["async", "linq", "pattern-matching"]).
"""

EXTRACTION_USER = """\
Extract all C#/.NET features from the following documentation text.
Category: {category}
Source URL: {source_url}

---
{content}
---

Return a JSON object with this exact schema:
{{
  "features": [
    {{
      "feature_name": "string",
      "csharp_version": "string",
      "dotnet_version": "string",
      "description": "string",
      "code_snippet": "string",
      "legacy_equivalent": "string",
      "nuget_packages": ["string"],
      "tags": ["string"]
    }}
  ]
}}
"""


def build_extraction_messages(
    content: str,
    category: str,
    source_url: str,
) -> list[dict[str, str]]:
    """Build the message list for a Phi-4 extraction call."""
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": EXTRACTION_USER.format(
                content=content,
                category=category,
                source_url=source_url,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Chat prompt — used in interactive mode with RAG context
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """\
You are Sovereign Shell, a C#/.NET expert assistant running locally on the \
user's machine. You have deep knowledge of every C# version from 1.0 through \
14.0 and the entire .NET ecosystem.

When answering:
- Cite the C# version and .NET version for any feature you mention.
- Show compilable code snippets with necessary using statements.
- When relevant, show how things were done in older versions (migration context).
- Be concise but thorough — prefer code over prose.
- If RAG context is provided below, prioritize it over general knowledge.

{rag_context}"""

CHAT_SYSTEM_NO_RAG = """\
You are Sovereign Shell, a C#/.NET expert assistant running locally on the \
user's machine. You have deep knowledge of every C# version from 1.0 through \
14.0 and the entire .NET ecosystem.

When answering:
- Cite the C# version and .NET version for any feature you mention.
- Show compilable code snippets with necessary using statements.
- When relevant, show how things were done in older versions (migration context).
- Be concise but thorough — prefer code over prose."""


def build_chat_messages(
    user_message: str,
    history: list[dict[str, str]] | None = None,
    rag_context: str = "",
) -> list[dict[str, str]]:
    """Build the message list for an interactive chat call.

    Parameters
    ----------
    user_message:
        The current user question.
    history:
        Previous exchanges as [{"role": "user"|"assistant", "content": ...}, ...].
        Kept short (last 3 exchanges = 6 messages) to fit context window.
    rag_context:
        Formatted context from vector/graph retrieval, injected into system prompt.
    """
    if rag_context:
        system = CHAT_SYSTEM.format(rag_context=rag_context)
    else:
        system = CHAT_SYSTEM_NO_RAG

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Re-extraction prompt — retry after validation failure
# ---------------------------------------------------------------------------

REEXTRACT_USER = """\
The following code snippet failed to compile with this error:

Feature: {feature_name}
Target: {validation_target}
Error:
{error_message}

Original snippet:
```csharp
{code_snippet}
```

Fix the code so it compiles. Return ONLY the corrected C# code, no explanation.
Include all necessary using statements and wrap in a namespace/class if needed.
"""


def build_reextract_messages(
    feature_name: str,
    code_snippet: str,
    error_message: str,
    validation_target: str,
) -> list[dict[str, str]]:
    """Build messages for re-extracting a failed snippet."""
    return [
        {
            "role": "system",
            "content": "You are a C# compiler error fixer. Return ONLY valid, compilable C# code.",
        },
        {
            "role": "user",
            "content": REEXTRACT_USER.format(
                feature_name=feature_name,
                code_snippet=code_snippet,
                error_message=error_message,
                validation_target=validation_target,
            ),
        },
    ]
