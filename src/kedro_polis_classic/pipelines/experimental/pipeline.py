from kedro.pipeline import Pipeline, node
from .nodes import run_component_node


def create_pipeline(pipeline_key="polis_classic_dummy") -> Pipeline:
    step_names = ["imputer", "reducer", "scaler", "clusterer"]
    nodes = []

    prev_output = "features_train"
    for step in step_names:
        # use lambda with default argument to fix step_name
        nodes.append(
            node(
                func=lambda X, params, step_name=step: run_component_node(
                    X, params, step_name
                ),
                inputs=[prev_output, f"params:pipelines.{pipeline_key}"],
                outputs=f"{step}_output",
                name=f"{step}_node",
            )
        )
        prev_output = f"{step}_output"

    return Pipeline(nodes)
