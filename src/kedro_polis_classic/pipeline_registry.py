from kedro.pipeline import Pipeline
from .pipelines.polis import pipeline as polis_pipeline
from .pipelines.experimental import pipeline as experiment_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    # List of experimental pipeline names
    experimental_pipeline_names = [
        "mean_pca_kmeans",
        "mean_pacmap_kmeans",
        "mean_pacmap_hdbscan",
        "mean_localmap_kmeans",
        "mean_localmap_hdbscan",
        "knn_pacmap_kmeans",
        "knn5d_pacmap_hdbscan",
        "knn_localmap_kmeans",
        "knn_localmap_hdbscan",
    ]

    # Create base pipelines
    pipelines = {
        "polis": polis_pipeline.create_pipeline(),
        "polis_classic": experiment_pipeline.create_pipeline("mean_pca_kmeans"),
        "__default__": polis_pipeline.create_pipeline(),
    }

    # Add experimental pipelines using iteration
    for name in experimental_pipeline_names:
        pipelines[name] = experiment_pipeline.create_pipeline(name)

    return pipelines
