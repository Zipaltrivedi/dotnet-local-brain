"""Core Pydantic models for the Sovereign Shell training dataset.

Every record represents a feature/API from the .NET ecosystem,
tagged with version info for training a C#-specialized model.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Category(str, Enum):
    """Documentation category for scraping targets."""
    LANGUAGE = "language"
    ASPNET = "aspnet"
    BLAZOR = "blazor"
    EF_CORE = "ef_core"
    MAUI = "maui"
    MINIMAL_APIS = "minimal_apis"
    SIGNALR = "signalr"
    GRPC = "grpc"
    MLNET = "mlnet"
    AZURE_SDK = "azure_sdk"
    MSBUILD = "msbuild"
    ROSLYN = "roslyn"
    NUGET = "nuget"
    BCL = "bcl"


class CSharpVersion(str, Enum):
    """All C# language versions from 1.0 through 14.0."""
    V1_0 = "1.0"
    V1_2 = "1.2"
    V2_0 = "2.0"
    V3_0 = "3.0"
    V4_0 = "4.0"
    V5_0 = "5.0"
    V6_0 = "6.0"
    V7_0 = "7.0"
    V7_1 = "7.1"
    V7_2 = "7.2"
    V7_3 = "7.3"
    V8_0 = "8.0"
    V9_0 = "9.0"
    V10_0 = "10.0"
    V11_0 = "11.0"
    V12_0 = "12.0"
    V13_0 = "13.0"
    V14_0 = "14.0"


class DotNetVersion(str, Enum):
    """All .NET runtime versions."""
    FRAMEWORK_1_0 = ".NET Framework 1.0"
    FRAMEWORK_1_1 = ".NET Framework 1.1"
    FRAMEWORK_2_0 = ".NET Framework 2.0"
    FRAMEWORK_3_0 = ".NET Framework 3.0"
    FRAMEWORK_3_5 = ".NET Framework 3.5"
    FRAMEWORK_4_0 = ".NET Framework 4.0"
    FRAMEWORK_4_5 = ".NET Framework 4.5"
    FRAMEWORK_4_6 = ".NET Framework 4.6"
    FRAMEWORK_4_7 = ".NET Framework 4.7"
    FRAMEWORK_4_8 = ".NET Framework 4.8"
    CORE_1_0 = ".NET Core 1.0"
    CORE_2_0 = ".NET Core 2.0"
    CORE_2_1 = ".NET Core 2.1"
    CORE_3_0 = ".NET Core 3.0"
    CORE_3_1 = ".NET Core 3.1"
    NET_5 = ".NET 5.0"
    NET_6 = ".NET 6.0"
    NET_7 = ".NET 7.0"
    NET_8 = ".NET 8.0"
    NET_9 = ".NET 9.0"
    NET_10 = ".NET 10.0"


class ValidationStatus(str, Enum):
    """Result of dotnet build validation."""
    UNTESTED = "untested"
    COMPILES = "compiles"
    FAILS = "fails"


# ---------------------------------------------------------------------------
# Core record — each instance becomes a training data point
# ---------------------------------------------------------------------------

class DotNetRecord(BaseModel):
    """A single feature/API record from the .NET ecosystem.

    The csharp_version + dotnet_version + legacy_equivalent triple
    is the key training signal: it teaches version migration,
    framework evolution, and API changes.
    """

    # Identity
    id: Optional[str] = Field(default=None, description="UUID, assigned on DB insert")

    # Categorization
    category: Category
    csharp_version: CSharpVersion
    dotnet_version: DotNetVersion
    feature_name: str = Field(..., min_length=1, max_length=300)

    # Content
    description: str = Field(..., description="What this feature does / why it exists")
    code_snippet: str = Field(..., description="Compilable C# code demonstrating the feature")
    legacy_equivalent: str = Field(
        ...,
        description="How this was done in prior versions (cross-version training signal)",
    )
    nuget_packages: list[str] = Field(
        default_factory=list,
        description="Required NuGet packages, e.g. ['Microsoft.EntityFrameworkCore']",
    )

    # Metadata
    source_url: str
    validation_status: ValidationStatus = ValidationStatus.UNTESTED
    validation_target: str = Field(
        default="",
        description="TargetFramework used for validation, e.g. 'net10.0'",
    )
    validation_error: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

class CoverageEntry(BaseModel):
    """Coverage stats for one C# version or category."""
    key: str  # version string or category name
    total_features: int = 0
    features_validated: int = 0
    features_failed: int = 0

    @property
    def completion_pct(self) -> float:
        if self.total_features == 0:
            return 0.0
        return (self.features_validated / self.total_features) * 100.0


class CoverageMatrix(BaseModel):
    """Top-level progress tracking across all versions and categories."""
    last_updated: Optional[str] = None  # ISO 8601
    by_version: list[CoverageEntry] = Field(default_factory=list)
    by_category: list[CoverageEntry] = Field(default_factory=list)
    total_records: int = 0
    total_validated: int = 0


# ---------------------------------------------------------------------------
# Extraction helpers (used by Phi-4 prompts)
# ---------------------------------------------------------------------------

class ExtractedFeature(BaseModel):
    """Single feature extracted from HTML by Phi-4.
    Intermediate format before becoming a full DotNetRecord.
    """
    feature_name: str
    csharp_version: str
    dotnet_version: str
    description: str = ""
    code_snippet: str = ""
    legacy_equivalent: str = ""
    nuget_packages: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Phi-4 extraction output: a list of features from one HTML chunk."""
    features: list[ExtractedFeature] = Field(default_factory=list)
