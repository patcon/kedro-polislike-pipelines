from kedro.pipeline import Pipeline
from .pipelines.polis import pipeline as polis_pipeline
from .pipelines.experimental import pipeline as experiment_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    # List of experimental pipeline names
    experimental_pipeline_names = [
        "polis_classic",
        "mean_pacmap_kmeans",
        "mean_localmap_kmeans",
        "knn_pacmap_kmeans",
        "knn_localmap_kmeans",
    ]

    # Create base pipelines
    pipelines = {
        "polis": polis_pipeline.create_pipeline(),
        "__default__": polis_pipeline.create_pipeline(),
    }

    # Add experimental pipelines using iteration
    for name in experimental_pipeline_names:
        pipelines[name] = experiment_pipeline.create_pipeline(name)

    return pipelines
