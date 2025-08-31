from kedro.pipeline import Pipeline, node
from .nodes import (
    run_component_node,
    load_polis_data,
    split_raw_data,
    dedup_votes,
    make_raw_vote_matrix,
    create_labels_dataframe,
)


def create_pipeline(pipeline_key="polis_classic_dummy") -> Pipeline:
    nodes = []

    # Data loading nodes
    nodes.extend(
        [
            node(
                func=load_polis_data,
                inputs="params:report_id",
                outputs="raw_data",
                name="load_polis_data",
            ),
            node(
                func=split_raw_data,
                inputs="raw_data",
                outputs=["raw_votes", "raw_comments"],
                name="split_raw_data",
            ),
            node(
                func=dedup_votes,
                inputs="raw_votes",
                outputs="deduped_votes",
                name="dedup_votes",
            ),
            node(
                func=make_raw_vote_matrix,
                inputs="deduped_votes",
                outputs="raw_vote_matrix",
                name="make_raw_vote_matrix",
            ),
        ]
    )

    # Component processing nodes
    step_names = ["imputer", "reducer", "scaler", "clusterer"]
    prev_output = "raw_vote_matrix"  # Use vote matrix as input to components

    for step in step_names:
        # use lambda with default argument to fix step_name
        nodes.append(
            node(
                func=lambda X, params, step_name=step: run_component_node(
                    X, params, step_name
                ),
                inputs=[prev_output, f"params:pipelines.{pipeline_key}.{step}"],
                outputs=f"{step}_output",
                name=f"{step}_node",
            )
        )
        prev_output = f"{step}_output"

    # Add labels processing node
    nodes.append(
        node(
            func=create_labels_dataframe,
            inputs=["clusterer_output", "raw_vote_matrix"],
            outputs="labels_dataframe",
            name="create_labels_dataframe",
        )
    )

    return Pipeline(nodes)
