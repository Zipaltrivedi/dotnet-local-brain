"""Seed URLs for the Sovereign Shell scraping pipeline.

Each category maps to a list of starting URLs that the BFS crawler
will expand from (up to crawl_depth=3 and max_pages_per_category).
"""

from __future__ import annotations

from sovereign_shell.models.schemas import Category

# ---------------------------------------------------------------------------
# Seed URLs per category
# ---------------------------------------------------------------------------

SEED_URLS: dict[Category, list[str]] = {
    # ── C# Language Features (per version) ──────────────────────────────
    Category.LANGUAGE: [
        # What's-new pages per version
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-version-history",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-14",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-13",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-12",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-11",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-10",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-9",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-8",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-7",
        "https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-6",
        # Language reference
        "https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/",
        "https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/",
        "https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/",
    ],

    # ── ASP.NET Core ────────────────────────────────────────────────────
    Category.ASPNET: [
        "https://learn.microsoft.com/en-us/aspnet/core/introduction-to-aspnet-core",
        "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/",
        "https://learn.microsoft.com/en-us/aspnet/core/mvc/overview",
        "https://learn.microsoft.com/en-us/aspnet/core/razor-pages/",
        "https://learn.microsoft.com/en-us/aspnet/core/security/",
        "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/middleware/",
    ],

    # ── Blazor ──────────────────────────────────────────────────────────
    Category.BLAZOR: [
        "https://learn.microsoft.com/en-us/aspnet/core/blazor/",
        "https://learn.microsoft.com/en-us/aspnet/core/blazor/components/",
        "https://learn.microsoft.com/en-us/aspnet/core/blazor/forms/",
        "https://learn.microsoft.com/en-us/aspnet/core/blazor/javascript-interoperability/",
        "https://learn.microsoft.com/en-us/aspnet/core/blazor/state-management",
    ],

    # ── Entity Framework Core ───────────────────────────────────────────
    Category.EF_CORE: [
        "https://learn.microsoft.com/en-us/ef/core/",
        "https://learn.microsoft.com/en-us/ef/core/get-started/overview/first-app",
        "https://learn.microsoft.com/en-us/ef/core/dbcontext-configuration/",
        "https://learn.microsoft.com/en-us/ef/core/modeling/",
        "https://learn.microsoft.com/en-us/ef/core/querying/",
        "https://learn.microsoft.com/en-us/ef/core/saving/",
        "https://learn.microsoft.com/en-us/ef/core/managing-schemas/migrations/",
    ],

    # ── MAUI ────────────────────────────────────────────────────────────
    Category.MAUI: [
        "https://learn.microsoft.com/en-us/dotnet/maui/what-is-maui",
        "https://learn.microsoft.com/en-us/dotnet/maui/fundamentals/",
        "https://learn.microsoft.com/en-us/dotnet/maui/user-interface/",
        "https://learn.microsoft.com/en-us/dotnet/maui/data/",
    ],

    # ── Minimal APIs ───────────────────────────────────────────────────
    Category.MINIMAL_APIS: [
        "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/minimal-apis/overview",
        "https://learn.microsoft.com/en-us/aspnet/core/fundamentals/minimal-apis/",
        "https://learn.microsoft.com/en-us/aspnet/core/tutorials/min-web-api",
    ],

    # ── SignalR ─────────────────────────────────────────────────────────
    Category.SIGNALR: [
        "https://learn.microsoft.com/en-us/aspnet/core/signalr/introduction",
        "https://learn.microsoft.com/en-us/aspnet/core/signalr/hubs",
        "https://learn.microsoft.com/en-us/aspnet/core/signalr/dotnet-client",
        "https://learn.microsoft.com/en-us/aspnet/core/signalr/streaming",
    ],

    # ── gRPC ────────────────────────────────────────────────────────────
    Category.GRPC: [
        "https://learn.microsoft.com/en-us/aspnet/core/grpc/",
        "https://learn.microsoft.com/en-us/aspnet/core/grpc/basics",
        "https://learn.microsoft.com/en-us/aspnet/core/grpc/services",
        "https://learn.microsoft.com/en-us/aspnet/core/grpc/client",
    ],

    # ── ML.NET ──────────────────────────────────────────────────────────
    Category.MLNET: [
        "https://learn.microsoft.com/en-us/dotnet/machine-learning/",
        "https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/",
        "https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/",
    ],

    # ── Azure SDK ───────────────────────────────────────────────────────
    Category.AZURE_SDK: [
        "https://learn.microsoft.com/en-us/dotnet/azure/sdk/azure-sdk-for-dotnet",
        "https://learn.microsoft.com/en-us/dotnet/azure/",
        "https://learn.microsoft.com/en-us/azure/developer/dotnet/",
    ],

    # ── MSBuild ─────────────────────────────────────────────────────────
    Category.MSBUILD: [
        "https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild",
        "https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-concepts",
        "https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties",
        "https://learn.microsoft.com/en-us/visualstudio/msbuild/msbuild-targets",
    ],

    # ── Roslyn ──────────────────────────────────────────────────────────
    Category.ROSLYN: [
        "https://learn.microsoft.com/en-us/dotnet/csharp/roslyn-sdk/",
        "https://learn.microsoft.com/en-us/dotnet/csharp/roslyn-sdk/tutorials/how-to-write-csharp-analyzer-code-fix",
        "https://learn.microsoft.com/en-us/dotnet/csharp/roslyn-sdk/source-generators-overview",
    ],

    # ── NuGet ───────────────────────────────────────────────────────────
    Category.NUGET: [
        "https://learn.microsoft.com/en-us/nuget/what-is-nuget",
        "https://learn.microsoft.com/en-us/nuget/quickstart/create-and-publish-a-package-using-the-dotnet-cli",
        "https://learn.microsoft.com/en-us/nuget/create-packages/",
    ],

    # ── BCL / Runtime Libraries ─────────────────────────────────────────
    Category.BCL: [
        "https://learn.microsoft.com/en-us/dotnet/fundamentals/runtime-libraries/",
        "https://learn.microsoft.com/en-us/dotnet/standard/collections/",
        "https://learn.microsoft.com/en-us/dotnet/standard/io/",
        "https://learn.microsoft.com/en-us/dotnet/standard/serialization/system-text-json/overview",
        "https://learn.microsoft.com/en-us/dotnet/fundamentals/networking/overview",
    ],
}


# ---------------------------------------------------------------------------
# Supplemental sources (DevBlogs, GitHub repos)
# ---------------------------------------------------------------------------

SUPPLEMENTAL_URLS: list[str] = [
    "https://devblogs.microsoft.com/dotnet/announcing-csharp-14/",
    "https://devblogs.microsoft.com/dotnet/announcing-dotnet-10/",
    "https://devblogs.microsoft.com/dotnet/announcing-dotnet-9/",
    "https://devblogs.microsoft.com/dotnet/announcing-csharp-13/",
    "https://devblogs.microsoft.com/dotnet/announcing-csharp-12/",
    "https://devblogs.microsoft.com/dotnet/performance-improvements-in-net-9/",
]


def get_seeds(category: Category | None = None) -> dict[Category, list[str]]:
    """Return seed URLs, optionally filtered to a single category."""
    if category is not None:
        return {category: SEED_URLS.get(category, [])}
    return dict(SEED_URLS)
