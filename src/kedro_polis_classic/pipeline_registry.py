from kedro.pipeline import Pipeline
from .pipelines.polis import pipeline as polis_pipeline
from .pipelines.experimental import pipeline as experiment_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "polis": polis_pipeline.create_pipeline(),
        "polis_classic_dummy": experiment_pipeline.create_pipeline("polis_classic_dummy"),
        "knn_pacmap_kmeans": experiment_pipeline.create_pipeline("knn_pacmap_kmeans"),
        "__default__": polis_pipeline.create_pipeline(),
    }
