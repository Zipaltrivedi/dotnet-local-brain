"""Auto-populate the knowledge graph from DotNetRecords.

After scraping, this module:
1. Creates version nodes (C# 1.0-14.0, .NET Framework → .NET 10)
2. Creates framework/category nodes (Blazor, EF Core, etc.)
3. Creates feature nodes from every record
4. Builds edges from record metadata:
   - csharp_version → INTRODUCED_IN
   - dotnet_version → REQUIRES_RUNTIME
   - category → PART_OF
   - nuget_packages → USES_PACKAGE
   - legacy_equivalent → REPLACES (if parseable)
"""

from __future__ import annotations

import logging
from typing import Optional

from sovereign_shell.memory.graphdb import GraphDB, GraphEdge, GraphNode
from sovereign_shell.models.schemas import (
    Category,
    CSharpVersion,
    DotNetRecord,
    DotNetVersion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical node IDs — deterministic so we don't create duplicates
# ---------------------------------------------------------------------------

def _version_node_id(version: CSharpVersion) -> str:
    return f"csharp:{version.value}"

def _runtime_node_id(version: DotNetVersion) -> str:
    return f"dotnet:{version.value}"

def _category_node_id(category: Category) -> str:
    return f"framework:{category.value}"

def _feature_node_id(record: DotNetRecord) -> str:
    return f"feature:{record.id}"

def _package_node_id(package: str) -> str:
    return f"nuget:{package.lower()}"


# ---------------------------------------------------------------------------
# Seed nodes — versions and frameworks
# ---------------------------------------------------------------------------

def _create_version_nodes() -> list[GraphNode]:
    """Create nodes for all C# language versions."""
    nodes = []
    for v in CSharpVersion:
        nodes.append(GraphNode(
            id=_version_node_id(v),
            node_type="csharp_version",
            name=f"C# {v.value}",
        ))
    return nodes


def _create_runtime_nodes() -> list[GraphNode]:
    """Create nodes for all .NET runtime versions."""
    nodes = []
    for v in DotNetVersion:
        nodes.append(GraphNode(
            id=_runtime_node_id(v),
            node_type="dotnet_version",
            name=v.value,
        ))
    return nodes


def _create_framework_nodes() -> list[GraphNode]:
    """Create nodes for each documentation category / framework."""
    # Friendly names for categories
    names = {
        Category.LANGUAGE: "C# Language",
        Category.ASPNET: "ASP.NET Core",
        Category.BLAZOR: "Blazor",
        Category.EF_CORE: "Entity Framework Core",
        Category.MAUI: ".NET MAUI",
        Category.MINIMAL_APIS: "Minimal APIs",
        Category.SIGNALR: "SignalR",
        Category.GRPC: "gRPC",
        Category.MLNET: "ML.NET",
        Category.AZURE_SDK: "Azure SDK",
        Category.MSBUILD: "MSBuild",
        Category.ROSLYN: "Roslyn",
        Category.NUGET: "NuGet",
        Category.BCL: "Base Class Library",
    }
    nodes = []
    for cat in Category:
        nodes.append(GraphNode(
            id=_category_node_id(cat),
            node_type="framework",
            name=names.get(cat, cat.value),
        ))
    return nodes


# Known framework dependency edges
_FRAMEWORK_DEPS: list[tuple[Category, Category]] = [
    (Category.BLAZOR, Category.ASPNET),
    (Category.MINIMAL_APIS, Category.ASPNET),
    (Category.SIGNALR, Category.ASPNET),
    (Category.GRPC, Category.ASPNET),
    (Category.EF_CORE, Category.BCL),
]

# C# version evolution chain
_VERSION_EVOLUTION: list[tuple[CSharpVersion, CSharpVersion]] = [
    (CSharpVersion.V1_0, CSharpVersion.V1_2),
    (CSharpVersion.V1_2, CSharpVersion.V2_0),
    (CSharpVersion.V2_0, CSharpVersion.V3_0),
    (CSharpVersion.V3_0, CSharpVersion.V4_0),
    (CSharpVersion.V4_0, CSharpVersion.V5_0),
    (CSharpVersion.V5_0, CSharpVersion.V6_0),
    (CSharpVersion.V6_0, CSharpVersion.V7_0),
    (CSharpVersion.V7_0, CSharpVersion.V7_1),
    (CSharpVersion.V7_1, CSharpVersion.V7_2),
    (CSharpVersion.V7_2, CSharpVersion.V7_3),
    (CSharpVersion.V7_3, CSharpVersion.V8_0),
    (CSharpVersion.V8_0, CSharpVersion.V9_0),
    (CSharpVersion.V9_0, CSharpVersion.V10_0),
    (CSharpVersion.V10_0, CSharpVersion.V11_0),
    (CSharpVersion.V11_0, CSharpVersion.V12_0),
    (CSharpVersion.V12_0, CSharpVersion.V13_0),
    (CSharpVersion.V13_0, CSharpVersion.V14_0),
]


def seed_graph(graph: GraphDB) -> dict:
    """Populate the graph with foundational nodes and structural edges.

    Call this once before adding feature records. Idempotent.
    Returns counts of nodes and edges created.
    """
    nodes_added = 0
    edges_added = 0

    # Version nodes
    all_nodes = _create_version_nodes() + _create_runtime_nodes() + _create_framework_nodes()
    nodes_added += graph.add_nodes_batch(all_nodes)

    # Version evolution chain: C# 1.0 → 1.2 → 2.0 → ... → 14.0
    for old_ver, new_ver in _VERSION_EVOLUTION:
        graph.add_edge(GraphEdge(
            id=None,
            source_id=_version_node_id(old_ver),
            target_id=_version_node_id(new_ver),
            relation="EVOLVED_INTO",
            weight=0.8,
        ))
        edges_added += 1

    # Framework dependencies
    for child, parent in _FRAMEWORK_DEPS:
        graph.add_edge(GraphEdge(
            id=None,
            source_id=_category_node_id(child),
            target_id=_category_node_id(parent),
            relation="DEPENDS_ON",
            weight=0.9,
        ))
        edges_added += 1

    logger.info("Graph seeded: %d nodes, %d edges", nodes_added, edges_added)
    return {"nodes_added": nodes_added, "edges_added": edges_added}


# ---------------------------------------------------------------------------
# Build graph from records
# ---------------------------------------------------------------------------

def add_record_to_graph(record: DotNetRecord, graph: GraphDB) -> int:
    """Add a single record to the graph as a feature node with edges.

    Returns the number of edges created.
    """
    feature_id = _feature_node_id(record)
    edges_created = 0

    # Create the feature node
    graph.add_node(GraphNode(
        id=feature_id,
        node_type="feature",
        name=record.feature_name,
        properties={
            "category": record.category.value,
            "csharp_version": record.csharp_version.value,
            "dotnet_version": record.dotnet_version.value,
            "validation_status": record.validation_status.value,
        },
    ))

    # Edge: feature → INTRODUCED_IN → csharp_version
    graph.add_edge(GraphEdge(
        id=None,
        source_id=feature_id,
        target_id=_version_node_id(record.csharp_version),
        relation="INTRODUCED_IN",
    ))
    edges_created += 1

    # Edge: feature → REQUIRES_RUNTIME → dotnet_version
    graph.add_edge(GraphEdge(
        id=None,
        source_id=feature_id,
        target_id=_runtime_node_id(record.dotnet_version),
        relation="REQUIRES_RUNTIME",
    ))
    edges_created += 1

    # Edge: feature → PART_OF → framework/category
    graph.add_edge(GraphEdge(
        id=None,
        source_id=feature_id,
        target_id=_category_node_id(record.category),
        relation="PART_OF",
    ))
    edges_created += 1

    # Edge: feature → USES_PACKAGE → nuget_package (for each package)
    for pkg in record.nuget_packages:
        pkg_id = _package_node_id(pkg)
        # Create the package node if it doesn't exist
        graph.add_node(GraphNode(
            id=pkg_id,
            node_type="nuget_package",
            name=pkg,
        ))
        graph.add_edge(GraphEdge(
            id=None,
            source_id=feature_id,
            target_id=pkg_id,
            relation="USES_PACKAGE",
        ))
        edges_created += 1

    return edges_created


def build_graph_from_records(
    records: list[DotNetRecord],
    graph: Optional[GraphDB] = None,
) -> dict:
    """Build the full knowledge graph from a list of records.

    Seeds the graph with version/framework nodes first, then adds
    all records as feature nodes with edges.
    """
    graph = graph or GraphDB()

    # Step 1: Seed foundational nodes
    seed_result = seed_graph(graph)

    # Step 2: Add each record (batch mode — commit every 500 records)
    graph._batch_mode = True
    total_edges = 0
    for i, record in enumerate(records):
        edges = add_record_to_graph(record, graph)
        total_edges += edges
        if (i + 1) % 5000 == 0:
            graph.conn.commit()
        if (i + 1) % 5000 == 0:
            logger.info("Added %d/%d records to graph", i + 1, len(records))
    graph.conn.commit()
    graph._batch_mode = False

    result = {
        "seed_nodes": seed_result["nodes_added"],
        "seed_edges": seed_result["edges_added"],
        "feature_nodes": len(records),
        "feature_edges": total_edges,
    }
    logger.info(
        "Graph built: %d seed nodes, %d feature nodes, %d total edges",
        result["seed_nodes"], result["feature_nodes"],
        result["seed_edges"] + result["feature_edges"],
    )
    return result
