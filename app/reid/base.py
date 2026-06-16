"""ReID adapter base class + dtype conventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ReIDConfig:
    name: str
    embedding_dim: int
    device: str = "gpu"
    use_fp16: bool = True
    batch_size: int = 16
    qdrant_collection: str = ""
    qdrant_distance: str = "cosine"
    input_size: tuple[int, int] = (256, 128)  # (W, H)


class ReIDAdapter(ABC):
    """All adapters expose a single API: extract a 1D float32 vector per crop."""

    config: ReIDConfig

    @abstractmethod
    def load(self) -> None:
        """Load weights, move to device, switch to FP16 if requested."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass to allocate GPU memory / cuDNN heuristics."""

    @abstractmethod
    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Input: list of BGR crops (H, W, 3). Output: (N, embedding_dim) float32.

        All output vectors are L2-normalized to unit length.
        """

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def qdrant_collection(self) -> str:
        return self.config.qdrant_collection
