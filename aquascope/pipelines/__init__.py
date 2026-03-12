"""Model implementation pipelines — auto-build approved methodologies."""

from aquascope.pipelines.model_builder import (
    PIPELINE_REGISTRY,
    PipelineResult,
    list_available_pipelines,
    run_pipeline,
)

__all__ = [
    "PIPELINE_REGISTRY",
    "PipelineResult",
    "list_available_pipelines",
    "run_pipeline",
]
