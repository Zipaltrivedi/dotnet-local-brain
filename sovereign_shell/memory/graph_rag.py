"""GraphRAG — Hybrid vector + graph retrieval.

Combines two retrieval strategies:
1. Vector similarity: find top-K records closest to the query embedding
2. Graph traversal: expand from those records via 2-hop graph walks
   to find related features, version context, and framework connections

The merged result gives the LLM richer context than vector-only search.

Example: query "How do records work in C# 14?"
- Vector search finds: C# 14 record features
- Graph expansion finds: C# 9 records (origin), C# 10 record structs,
  pattern matching (RELATED_TO), and .NET version requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sovereign_shell.config import SovereignConfig, get_config
from sovereign_shell.memory.embeddings import embed_text
from sovereign_shell.memory.graphdb import GraphDB, GraphEdge, GraphNode
from sovereign_shell.memory.vectordb import VectorDB
from sovereign_shell.models.schemas import DotNetRecord

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """A single retrieval result with its source and relevance info."""
    record: DotNetRecord
    score: float  # Lower = more similar (distance)
    source: str   # "vector" or "graph"
    related_edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class RAGContext:
    """Complete retrieval context ready for prompt injection."""
    results: list[RAGResult] = field(default_factory=list)
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)

    def format_for_prompt(self, max_chars: int = 3000) -> str:
        """Format the retrieval results as text for the system prompt.

        Includes record details and relationship annotations.
        """
        if not self.results:
            return ""

        parts = ["## Retrieved Context\n"]
        char_count = 0

        for i, result in enumerate(self.results):
            r = result.record
            section = (
                f"### [{i+1}] {r.feature_name} "
                f"(C# {r.csharp_version.value}, {r.dotnet_version.value})\n"
                f"Category: {r.category.value}\n"
                f"{r.description}\n"
            )

            # Add code snippet (truncated if long)
            code = r.code_snippet
            if code and code != "// No code extracted":
                if len(code) > 500:
                    code = code[:500] + "\n// ... (truncated)"
                section += f"```csharp\n{code}\n```\n"

            # Add legacy context
            if r.legacy_equivalent and r.legacy_equivalent != "N/A":
                section += f"Previously: {r.legacy_equivalent}\n"

            # Add relationship annotations from graph
            if result.related_edges:
                rels = []
                for edge in result.related_edges[:3]:  # Limit annotations
                    rels.append(f"  - {edge.relation}: {edge.target_id}")
                section += "Relationships:\n" + "\n".join(rels) + "\n"

            section += "\n"

            if char_count + len(section) > max_chars:
                break
            parts.append(section)
            char_count += len(section)

        # Add version context from graph traversal
        version_nodes = [n for n in self.graph_nodes if n.node_type == "csharp_version"]
        if version_nodes:
            versions = ", ".join(n.name for n in version_nodes[:5])
            parts.append(f"Related versions: {versions}\n")

        return "".join(parts)


class GraphRAGRetriever:
    """Hybrid retrieval engine combining vector search + graph traversal."""

    def __init__(
        self,
        vectordb: Optional[VectorDB] = None,
        graphdb: Optional[GraphDB] = None,
        config: Optional[SovereignConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self.vectordb = vectordb or VectorDB(self.config)
        self.graphdb = graphdb or GraphDB(self.config)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        graph_hops: int = 2,
        graph_neighbors: int = 5,
    ) -> RAGContext:
        """Hybrid retrieval: vector similarity + graph expansion.

        Steps:
        1. Embed the query
        2. Vector search for top-K similar records
        3. For each result, traverse the graph 2 hops
        4. Merge and deduplicate
        5. Return formatted context
        """
        # Step 1: Embed query
        try:
            query_vec = embed_text(query, self.config)
        except Exception:
            logger.exception("Failed to embed query")
            return RAGContext()

        # Step 2: Vector similarity search
        vector_results = self.vectordb.search_similar(query_vec, top_k=top_k)

        if not vector_results:
            logger.info("No vector results for query")
            return RAGContext()

        # Build RAGResults from vector search
        rag_results: list[RAGResult] = []
        feature_node_ids: list[str] = []

        for record, distance in vector_results:
            rag_results.append(RAGResult(
                record=record,
                score=distance,
                source="vector",
            ))
            feature_node_ids.append(f"feature:{record.id}")

        # Step 3: Graph expansion from vector results
        traversal = self.graphdb.expand_neighbors(
            node_ids=feature_node_ids,
            max_hops=graph_hops,
            max_neighbors_per_hop=graph_neighbors,
        )

        # Annotate vector results with their graph edges
        for result in rag_results:
            fid = f"feature:{result.record.id}"
            result.related_edges = [
                e for e in traversal.edges
                if e.source_id == fid or e.target_id == fid
            ]

        # Step 4: Find additional records from graph that weren't in vector results
        seen_ids = {r.record.id for r in rag_results}
        for node in traversal.nodes:
            if node.node_type == "feature" and node.id.startswith("feature:"):
                record_id = node.id[len("feature:"):]
                if record_id not in seen_ids:
                    record = self.vectordb.get_by_id(record_id)
                    if record:
                        rag_results.append(RAGResult(
                            record=record,
                            score=999.0,  # High score = less relevant than vector results
                            source="graph",
                            related_edges=[
                                e for e in traversal.edges
                                if e.source_id == node.id or e.target_id == node.id
                            ],
                        ))
                        seen_ids.add(record_id)

        # Step 5: Sort — vector results first (lower distance), then graph results
        rag_results.sort(key=lambda r: (0 if r.source == "vector" else 1, r.score))

        return RAGContext(
            results=rag_results,
            graph_nodes=traversal.nodes,
            graph_edges=traversal.edges,
        )

    def close(self) -> None:
        """Release resources."""
        self.vectordb.close()
        self.graphdb.close()
