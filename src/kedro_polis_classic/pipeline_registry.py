from kedro.pipeline import Pipeline
from kedro_polis_classic.pipelines.polis import pipeline as polis_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "polis": polis_pipeline.create_pipeline(),
        "__default__": polis_pipeline.create_pipeline(),
    }
