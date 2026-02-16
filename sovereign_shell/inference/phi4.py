"""Phi-4 inference engine using llama-cpp-python.

Singleton wrapper that manages VRAM lifecycle:
- load()  → loads GGUF into GPU
- unload() → releases VRAM
- complete() → text completion (low temp for extraction)
- chat()  → chat completion (higher temp for interactive)
"""

from __future__ import annotations

import os
from typing import Optional

from llama_cpp import Llama

from sovereign_shell.config import SovereignConfig, get_config

# Ensure CUDA DLLs are discoverable on Windows
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.isdir(_cuda_bin):
    os.add_dll_directory(_cuda_bin)
    os.environ["PATH"] = _cuda_bin + ";" + os.environ.get("PATH", "")


class Phi4Engine:
    """Singleton wrapper around llama-cpp-python for Phi-4 GGUF inference."""

    _instance: Optional[Phi4Engine] = None

    def __init__(self, config: SovereignConfig) -> None:
        self.config = config
        self._llm: Optional[Llama] = None

    @classmethod
    def get(cls, config: Optional[SovereignConfig] = None) -> Phi4Engine:
        """Return the singleton engine, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(config or get_config())
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful for tests)."""
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """Load model into VRAM. Idempotent."""
        if self._llm is not None:
            return
        self._llm = Llama(
            model_path=str(self.config.model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            verbose=False,
        )

    def unload(self) -> None:
        """Release VRAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Run text completion. Returns generated text only."""
        self.load()
        temp = temperature if temperature is not None else self.config.temperature
        result = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop=stop,
        )
        return result["choices"][0]["text"]

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Chat completion for interactive Q&A."""
        self.load()
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"]
