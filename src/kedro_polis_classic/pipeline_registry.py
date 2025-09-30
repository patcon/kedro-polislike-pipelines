from kedro.pipeline import Pipeline
from .pipelines.experimental import pipeline as experiment_pipeline
from .pipelines.config import load_pipelines_config


def register_pipelines() -> dict[str, Pipeline]:
    # Load pipeline configurations
    pipelines_config = load_pipelines_config()
    experimental_pipeline_names = list(pipelines_config.keys())

    pipelines = {}

    # Add shorthand name for original polis pipeline.
    # This now includes preprocessing with namespace
    pipelines["polis_classic"] = experiment_pipeline.create_pipeline(
        "mean_pca_bestkmeans"
    )

    # Add experimental pipelines using iteration
    # Each pipeline now includes preprocessing with namespace
    for name in experimental_pipeline_names:
        pipelines[name] = experiment_pipeline.create_pipeline(name)

    return pipelines
