"""SQLite-backed knowledge graph.

Stores typed nodes and edges representing relationships between
C# features, versions, frameworks, namespaces, and NuGet packages.
Supports N-hop traversal for GraphRAG context expansion.

No external graph database needed — everything in SQLite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    node_type: str  # "feature", "csharp_version", "dotnet_version", "framework", "namespace", "nuget_package", "concept"
    name: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: Optional[int]
    source_id: str
    target_id: str
    relation: str  # INTRODUCED_IN, REQUIRES_RUNTIME, REPLACES, EVOLVED_INTO, DEPENDS_ON, PART_OF, USES_PACKAGE, RELATED_TO, NAMESPACE_OF
    weight: float = 1.0
    properties: dict = field(default_factory=dict)


@dataclass
class TraversalResult:
    """Result of a graph traversal — nodes and edges found."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}


# ---------------------------------------------------------------------------
# Relation types
# ---------------------------------------------------------------------------

RELATION_TYPES = {
    "INTRODUCED_IN",      # feature → csharp_version
    "REQUIRES_RUNTIME",   # feature → dotnet_version
    "REPLACES",           # new_feature → old_feature
    "EVOLVED_INTO",       # old_feature → new_feature
    "DEPENDS_ON",         # framework → framework
    "PART_OF",            # feature → framework
    "USES_PACKAGE",       # feature → nuget_package
    "RELATED_TO",         # feature → feature (conceptual)
    "NAMESPACE_OF",       # namespace → feature
}

NODE_TYPES = {
    "feature",
    "csharp_version",
    "dotnet_version",
    "framework",
    "namespace",
    "nuget_package",
    "concept",
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON graph_nodes(name);

CREATE TABLE IF NOT EXISTS graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES graph_nodes(id),
    target_id TEXT NOT NULL REFERENCES graph_nodes(id),
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    properties TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON graph_edges(relation);
"""


class GraphDB:
    """SQLite-backed knowledge graph with typed nodes and edges."""

    def __init__(self, config: Optional[SovereignConfig] = None) -> None:
        self.config = config or get_config()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._batch_mode: bool = False
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.config.db_path), timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(_GRAPH_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ----- Node operations -----

    def add_node(self, node: GraphNode) -> None:
        """Insert or update a node."""
        self.conn.execute(
            "INSERT OR REPLACE INTO graph_nodes (id, node_type, name, properties) VALUES (?, ?, ?, ?)",
            (node.id, node.node_type, node.name, json.dumps(node.properties)),
        )
        if not self._batch_mode:
            self.conn.commit()

    def add_nodes_batch(self, nodes: list[GraphNode]) -> int:
        """Insert multiple nodes. Returns count added."""
        count = 0
        for node in nodes:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO graph_nodes (id, node_type, name, properties) VALUES (?, ?, ?, ?)",
                    (node.id, node.node_type, node.name, json.dumps(node.properties)),
                )
                count += 1
            except Exception:
                logger.warning("Failed to add node: %s", node.id)
        self.conn.commit()
        return count

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        row = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        rows = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE node_type = ?", (node_type,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def search_nodes(self, name_pattern: str, node_type: Optional[str] = None) -> list[GraphNode]:
        """Search nodes by name (LIKE pattern)."""
        query = "SELECT * FROM graph_nodes WHERE name LIKE ?"
        params: list = [f"%{name_pattern}%"]
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ----- Edge operations -----

    def add_edge(self, edge: GraphEdge) -> None:
        """Insert an edge (allows duplicates with different properties)."""
        # Check if this exact edge already exists
        existing = self.conn.execute(
            "SELECT id FROM graph_edges WHERE source_id = ? AND target_id = ? AND relation = ?",
            (edge.source_id, edge.target_id, edge.relation),
        ).fetchone()
        if existing:
            return  # Don't duplicate

        self.conn.execute(
            "INSERT INTO graph_edges (source_id, target_id, relation, weight, properties) VALUES (?, ?, ?, ?, ?)",
            (edge.source_id, edge.target_id, edge.relation, edge.weight, json.dumps(edge.properties)),
        )
        if not self._batch_mode:
            self.conn.commit()

    def add_edges_batch(self, edges: list[GraphEdge]) -> int:
        """Insert multiple edges. Returns count added."""
        count = 0
        for edge in edges:
            try:
                self.add_edge(edge)
                count += 1
            except Exception:
                logger.warning("Failed to add edge: %s -[%s]-> %s", edge.source_id, edge.relation, edge.target_id)
        return count

    def get_outgoing(self, node_id: str, relation: Optional[str] = None) -> list[GraphEdge]:
        """Get all edges originating from a node."""
        if relation:
            rows = self.conn.execute(
                "SELECT * FROM graph_edges WHERE source_id = ? AND relation = ?",
                (node_id, relation),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM graph_edges WHERE source_id = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_incoming(self, node_id: str, relation: Optional[str] = None) -> list[GraphEdge]:
        """Get all edges pointing to a node."""
        if relation:
            rows = self.conn.execute(
                "SELECT * FROM graph_edges WHERE target_id = ? AND relation = ?",
                (node_id, relation),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM graph_edges WHERE target_id = ?", (node_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_neighbors(self, node_id: str, max_neighbors: int = 10) -> list[tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes (both directions) with their connecting edges."""
        results: list[tuple[GraphNode, GraphEdge]] = []

        # Outgoing
        for edge in self.get_outgoing(node_id):
            node = self.get_node(edge.target_id)
            if node:
                results.append((node, edge))

        # Incoming
        for edge in self.get_incoming(node_id):
            node = self.get_node(edge.source_id)
            if node:
                results.append((node, edge))

        # Sort by edge weight descending, limit
        results.sort(key=lambda x: x[1].weight, reverse=True)
        return results[:max_neighbors]

    # ----- Traversal -----

    def expand_neighbors(
        self,
        node_ids: list[str],
        max_hops: int = 2,
        max_neighbors_per_hop: int = 5,
    ) -> TraversalResult:
        """Multi-hop graph traversal from a set of starting nodes.

        Expands outward from the starting nodes, collecting nodes and edges
        at each hop level. Used by GraphRAG to enrich vector search results.
        """
        visited: set[str] = set()
        all_nodes: list[GraphNode] = []
        all_edges: list[GraphEdge] = []

        # Add starting nodes
        current_frontier = set(node_ids)
        for nid in node_ids:
            node = self.get_node(nid)
            if node:
                all_nodes.append(node)
                visited.add(nid)

        for hop in range(max_hops):
            next_frontier: set[str] = set()

            for nid in current_frontier:
                neighbors = self.get_neighbors(nid, max_neighbors=max_neighbors_per_hop)

                for neighbor_node, edge in neighbors:
                    all_edges.append(edge)

                    if neighbor_node.id not in visited:
                        visited.add(neighbor_node.id)
                        all_nodes.append(neighbor_node)
                        next_frontier.add(neighbor_node.id)

            current_frontier = next_frontier
            if not current_frontier:
                break  # No more nodes to expand

        return TraversalResult(nodes=all_nodes, edges=all_edges)

    # ----- Stats -----

    def stats(self) -> dict:
        """Return graph statistics."""
        node_count = self.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]

        types = {}
        for row in self.conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM graph_nodes GROUP BY node_type"
        ).fetchall():
            types[row["node_type"]] = row["cnt"]

        relations = {}
        for row in self.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM graph_edges GROUP BY relation"
        ).fetchall():
            relations[row["relation"]] = row["cnt"]

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_types": types,
            "relation_types": relations,
        }

    # ----- Helpers -----

    def _row_to_node(self, row: sqlite3.Row) -> GraphNode:
        return GraphNode(
            id=row["id"],
            node_type=row["node_type"],
            name=row["name"],
            properties=json.loads(row["properties"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=row["relation"],
            weight=row["weight"],
            properties=json.loads(row["properties"]),
        )
