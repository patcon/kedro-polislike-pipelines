from kedro.pipeline import Pipeline
from .pipelines.default import pipeline as default_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {}

    # Default pipeline is the branching pipeline that creates a DAG structure
    # This pipeline runs preprocessing once, each imputer once, each reducer once per imputer,
    # and clusterers on each imputer-reducer combination output
    try:
        pipelines["__default__"] = default_pipeline.create_branching_pipeline()
    except ValueError as e:
        # If branching_pipeline config is not found, skip it
        print(f"Warning: Skipping default pipeline - {e}")

    return pipelines
