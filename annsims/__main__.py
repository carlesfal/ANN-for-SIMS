"""Allow ``python -m annsims`` to run the pipeline with default config."""

from .config import PipelineConfig
from .main import run_pipeline

if __name__ == "__main__":
    run_pipeline(PipelineConfig())
