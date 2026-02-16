"""Central configuration for Sovereign Shell."""

import os
from pathlib import Path
from pydantic import BaseModel


class SovereignConfig(BaseModel):
    """All paths, model parameters, and budget settings."""

    # Paths
    project_root: Path = Path(r"C:\C#LocalModel")
    secrets_dir: Path = Path(r"C:\C#LocalModel\secrets")
    model_path: Path = Path(r"C:\C#LocalModel\phi-4-Q4_K_M.gguf")
    db_path: Path = Path(r"C:\C#LocalModel\data\sovereign_shell.db")
    coverage_path: Path = Path(r"C:\C#LocalModel\data\coverage_matrix.json")
    raw_html_dir: Path = Path(r"C:\C#LocalModel\data\raw_html")
    extracted_dir: Path = Path(r"C:\C#LocalModel\data\extracted")
    training_dir: Path = Path(r"C:\C#LocalModel\data\training")
    validation_project_dir: Path = Path(r"C:\C#LocalModel\validation_projects")

    # Model parameters
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # Full GPU offload
    n_batch: int = 512
    n_threads: int = 8
    temperature: float = 0.1  # Low temp for extraction accuracy

    # Scraper parameters
    crawl_depth: int = 3
    max_pages_per_category: int = 50

    # Embedding
    embedding_dim: int = 128

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000


def get_config() -> SovereignConfig:
    """Return the default configuration."""
    cfg = SovereignConfig()
    _load_secrets(cfg)
    return cfg


def _load_secrets(cfg: SovereignConfig) -> None:
    """Load API tokens from secrets/ into environment variables."""
    hf_token_file = cfg.secrets_dir / "hf_token.txt"
    if hf_token_file.exists() and not os.environ.get("HF_TOKEN"):
        token = hf_token_file.read_text().strip()
        if token:
            os.environ["HF_TOKEN"] = token
