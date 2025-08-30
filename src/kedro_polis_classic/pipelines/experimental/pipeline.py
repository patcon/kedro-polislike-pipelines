from kedro.pipeline import Pipeline, node
from .nodes import run_component_node


def create_pipeline(pipeline_key="polis_classic_dummy") -> Pipeline:
    step_names = ["imputer", "reducer", "scaler", "clusterer"]
    nodes = []

    prev_output = "features_train"
    for step in step_names:
        # Each step gets its own node
        # input: X + full params dict
        nodes.append(
            node(
                func=run_component_node,
                inputs=[prev_output, f"params:pipelines.{pipeline_key}", step],
                outputs=f"{step}_output",
                name=f"{step}_node",
            )
        )
        prev_output = f"{step}_output"

    return Pipeline(nodes)
