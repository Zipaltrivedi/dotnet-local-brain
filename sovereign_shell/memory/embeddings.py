"""Embedding generation for DotNetRecords.

Uses Phi-4 via llama-cpp-python to generate embeddings, then
mean-pools the per-token vectors and reduces to the configured
dimension (default 128) via simple truncation + L2 normalization.

The embedding text combines feature_name + description + code_snippet
to create a rich representation of each record.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from llama_cpp import Llama

from sovereign_shell.config import SovereignConfig, get_config

logger = logging.getLogger(__name__)

# Module-level embedding model (separate from chat model to avoid reloading)
_embed_llm: Optional[Llama] = None


def _get_embed_model(config: Optional[SovereignConfig] = None) -> Llama:
    """Get or create the embedding model instance."""
    global _embed_llm
    if _embed_llm is None:
        cfg = config or get_config()
        _embed_llm = Llama(
            model_path=str(cfg.model_path),
            n_ctx=512,  # Small context for embeddings — saves VRAM
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            n_threads=cfg.n_threads,
            embedding=True,
            verbose=False,
        )
    return _embed_llm


def unload_embed_model() -> None:
    """Release the embedding model from VRAM."""
    global _embed_llm
    if _embed_llm is not None:
        del _embed_llm
        _embed_llm = None


def _mean_pool(token_embeddings: list[list[float]]) -> list[float]:
    """Mean-pool per-token embeddings into a single vector."""
    if not token_embeddings:
        return []
    if not isinstance(token_embeddings[0], list):
        # Already a single vector
        return token_embeddings

    dim = len(token_embeddings[0])
    pooled = [0.0] * dim
    for token_vec in token_embeddings:
        for i in range(dim):
            pooled[i] += token_vec[i]
    n = len(token_embeddings)
    return [v / n for v in pooled]


def _normalize_l2(vec: list[float]) -> list[float]:
    """L2-normalize a vector."""
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude == 0:
        return vec
    return [v / magnitude for v in vec]


def _truncate_to_dim(vec: list[float], target_dim: int) -> list[float]:
    """Truncate or pad vector to target dimension."""
    if len(vec) >= target_dim:
        return vec[:target_dim]
    # Pad with zeros if somehow shorter
    return vec + [0.0] * (target_dim - len(vec))


def embed_text(
    text: str,
    config: Optional[SovereignConfig] = None,
) -> list[float]:
    """Generate a normalized embedding vector for text.

    Returns a vector of dimension config.embedding_dim (default 128).
    """
    cfg = config or get_config()
    model = _get_embed_model(cfg)

    # Truncate input to avoid context overflow
    truncated = text[:1500]

    result = model.create_embedding(truncated)
    raw = result["data"][0]["embedding"]

    # Handle per-token vs single vector
    if isinstance(raw[0], list):
        pooled = _mean_pool(raw)
    else:
        pooled = raw

    # Truncate to target dimension and normalize
    truncated_vec = _truncate_to_dim(pooled, cfg.embedding_dim)
    return _normalize_l2(truncated_vec)


def embed_record_text(
    feature_name: str,
    description: str,
    code_snippet: str,
    csharp_version: str = "",
    category: str = "",
) -> str:
    """Build the text representation of a record for embedding.

    Combines key fields into a single string that captures the
    semantic meaning of the record.
    """
    parts = []
    if category:
        parts.append(f"[{category}]")
    if csharp_version:
        parts.append(f"C# {csharp_version}:")
    parts.append(feature_name)
    if description:
        parts.append(f"- {description}")
    if code_snippet and code_snippet != "// No code extracted":
        # Include first ~500 chars of code for semantic signal
        parts.append(f"Code: {code_snippet[:500]}")
    return " ".join(parts)
