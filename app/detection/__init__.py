"""Detection package — Paddle PP-Human production path."""

from .pphuman_pipeline import (
    Detection,
    PPHumanDetectorAdapter,
    PPHumanPipelineSubprocessManager,
)

__all__ = [
    "Detection",
    "PPHumanDetectorAdapter",
    "PPHumanPipelineSubprocessManager",
]
