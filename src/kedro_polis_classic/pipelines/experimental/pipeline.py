from kedro.pipeline import Pipeline, node
from .nodes import run_pipeline_node


def create_pipeline(params_key="pipeline_experiment") -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_pipeline_node,
                inputs=["features_train", f"params:{params_key}"],
                outputs="trained_pipeline_experiment",
                name="run_experiment_pipeline",
            )
        ]
    )
