"""Multi-SDK dotnet build validator.

Compiles extracted code snippets against the appropriate .NET SDK
to verify they are valid, compilable C#. Each snippet is compiled
against its natural target framework with the correct LangVersion.

Supported SDKs: .NET 6, 8, 9, 10 (must be installed side-by-side).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.models.schemas import (
    Category,
    DotNetRecord,
    DotNetVersion,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version mapping: DotNetVersion → TargetFramework moniker
# ---------------------------------------------------------------------------

_TFM_MAP: dict[DotNetVersion, str] = {
    DotNetVersion.NET_10: "net10.0",
    DotNetVersion.NET_9: "net9.0",
    DotNetVersion.NET_8: "net8.0",
    DotNetVersion.NET_7: "net7.0",
    DotNetVersion.NET_6: "net6.0",
    DotNetVersion.NET_5: "net9.0",       # .NET 5 EOL — validate on 9
    DotNetVersion.CORE_3_1: "net9.0",    # Core 3.1 EOL — validate on 9
    DotNetVersion.CORE_3_0: "net9.0",
    DotNetVersion.CORE_2_1: "net9.0",
    DotNetVersion.CORE_2_0: "net9.0",
    DotNetVersion.CORE_1_0: "net9.0",
    # .NET Framework → compile on net9.0 with matching LangVersion
    DotNetVersion.FRAMEWORK_4_8: "net9.0",
    DotNetVersion.FRAMEWORK_4_7: "net9.0",
    DotNetVersion.FRAMEWORK_4_6: "net9.0",
    DotNetVersion.FRAMEWORK_4_5: "net9.0",
    DotNetVersion.FRAMEWORK_4_0: "net9.0",
    DotNetVersion.FRAMEWORK_3_5: "net9.0",
    DotNetVersion.FRAMEWORK_3_0: "net9.0",
    DotNetVersion.FRAMEWORK_2_0: "net9.0",
    DotNetVersion.FRAMEWORK_1_1: "net9.0",
    DotNetVersion.FRAMEWORK_1_0: "net9.0",
}

# Categories that need web SDK
_WEB_CATEGORIES: set[Category] = {
    Category.ASPNET,
    Category.BLAZOR,
    Category.MINIMAL_APIS,
    Category.SIGNALR,
    Category.GRPC,
}


def _get_tfm(record: DotNetRecord) -> str:
    """Resolve the TargetFramework moniker for a record."""
    return _TFM_MAP.get(record.dotnet_version, "net10.0")


def _get_lang_version(record: DotNetRecord) -> str:
    """Map CSharpVersion enum to LangVersion MSBuild property."""
    return record.csharp_version.value


def _needs_web_sdk(record: DotNetRecord) -> bool:
    """Check if this record requires Microsoft.NET.Sdk.Web."""
    return record.category in _WEB_CATEGORIES


def _generate_csproj(
    record: DotNetRecord,
    tfm: str,
    lang_version: str,
) -> str:
    """Generate a .csproj file for compiling the snippet."""
    sdk = "Microsoft.NET.Sdk.Web" if _needs_web_sdk(record) else "Microsoft.NET.Sdk"

    nuget_refs = ""
    if record.nuget_packages:
        refs = []
        for pkg in record.nuget_packages:
            # Use wildcard version — NuGet will resolve latest compatible
            refs.append(f'    <PackageReference Include="{pkg}" Version="*" />')
        nuget_refs = "\n  <ItemGroup>\n" + "\n".join(refs) + "\n  </ItemGroup>"

    return f"""<Project Sdk="{sdk}">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{tfm}</TargetFramework>
    <LangVersion>{lang_version}</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>{nuget_refs}
</Project>
"""


def _wrap_snippet(record: DotNetRecord, code: str) -> str:
    """Wrap a code snippet to make it compilable if it's a bare fragment.

    If the snippet already contains a namespace or top-level statements,
    return it as-is. Otherwise, wrap in a namespace + class.
    """
    stripped = code.strip()

    # Already has namespace or is a full file
    if "namespace " in stripped or "class " in stripped or "record " in stripped:
        return stripped

    # Check if it looks like top-level statements (has using or direct statements)
    if stripped.startswith("using ") or stripped.startswith("//"):
        return stripped

    # Wrap bare code in a class
    return f"""using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ValidationTest;

public static class Snippet
{{
    public static void Run()
    {{
        {stripped}
    }}
}}
"""


def validate_record(
    record: DotNetRecord,
    config: Optional[SovereignConfig] = None,
    timeout_seconds: int = 60,
) -> DotNetRecord:
    """Validate a single record by compiling its code snippet.

    Returns a copy of the record with validation_status,
    validation_target, and validation_error updated.
    """
    cfg = config or get_config()
    cfg.validation_project_dir.mkdir(parents=True, exist_ok=True)

    tfm = _get_tfm(record)
    lang_version = _get_lang_version(record)
    project_dir = cfg.validation_project_dir / f"val_{uuid.uuid4().hex[:8]}"

    try:
        project_dir.mkdir(parents=True, exist_ok=True)

        # Write .csproj
        csproj_content = _generate_csproj(record, tfm, lang_version)
        (project_dir / "Validation.csproj").write_text(csproj_content, encoding="utf-8")

        # Write the snippet
        wrapped = _wrap_snippet(record, record.code_snippet)
        (project_dir / "Snippet.cs").write_text(wrapped, encoding="utf-8")

        # Run dotnet build
        result = subprocess.run(
            ["dotnet", "build", "--nologo", "-v", "quiet"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return record.model_copy(
                update={
                    "validation_status": ValidationStatus.COMPILES,
                    "validation_target": tfm,
                    "validation_error": None,
                }
            )
        else:
            error = result.stderr.strip() or result.stdout.strip()
            # Truncate very long errors
            if len(error) > 2000:
                error = error[:2000] + "\n... (truncated)"

            return record.model_copy(
                update={
                    "validation_status": ValidationStatus.FAILS,
                    "validation_target": tfm,
                    "validation_error": error,
                }
            )

    except subprocess.TimeoutExpired:
        return record.model_copy(
            update={
                "validation_status": ValidationStatus.FAILS,
                "validation_target": tfm,
                "validation_error": "Build timed out after {timeout_seconds}s",
            }
        )
    except Exception as e:
        return record.model_copy(
            update={
                "validation_status": ValidationStatus.FAILS,
                "validation_target": tfm,
                "validation_error": f"Validation error: {e}",
            }
        )
    finally:
        # Clean up temp project
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)


def validate_batch(
    records: list[DotNetRecord],
    config: Optional[SovereignConfig] = None,
) -> list[DotNetRecord]:
    """Validate a batch of records. Returns updated copies."""
    validated: list[DotNetRecord] = []
    total = len(records)

    for i, record in enumerate(records):
        logger.info(
            "Validating %d/%d: %s (C# %s, %s)",
            i + 1,
            total,
            record.feature_name,
            record.csharp_version.value,
            record.dotnet_version.value,
        )
        result = validate_record(record, config)
        validated.append(result)

        if result.validation_status == ValidationStatus.COMPILES:
            logger.info("  ✓ Compiles on %s", result.validation_target)
        else:
            logger.warning("  ✗ Failed: %s", (result.validation_error or "")[:100])

    passed = sum(1 for r in validated if r.validation_status == ValidationStatus.COMPILES)
    failed = sum(1 for r in validated if r.validation_status == ValidationStatus.FAILS)
    logger.info("Validation complete: %d passed, %d failed out of %d", passed, failed, total)

    return validated
