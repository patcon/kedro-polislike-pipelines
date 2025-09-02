from kedro.pipeline import Pipeline, node
from .nodes import (
    run_component_node,
    load_polis_data,
    split_raw_data,
    dedup_votes,
    make_raw_vote_matrix,
    create_labels_dataframe,
    create_scatter_plot,
)


def _extract_input_parameters(params_dict: dict) -> list[str]:
    """
    Extract catalog item names from parameters that start with 'input:'.

    Args:
        params_dict: Dictionary of parameters that may contain 'input:' values

    Returns:
        List of catalog item names referenced by 'input:' parameters.
        Returns an empty list if no 'input:' parameters are found.

    Example:
        If params_dict = {"name": "SparsityAwareScaler", "X_sparse": "input:raw_vote_matrix"}
        Returns ["raw_vote_matrix"]
    """
    input_catalog_items = []
    for key, value in params_dict.items():
        if key != "name" and isinstance(value, str) and value.startswith("input:"):
            catalog_item_name = value[6:]  # Remove "input:" prefix
            input_catalog_items.append(catalog_item_name)
    return input_catalog_items


def _load_pipeline_params(pipeline_key: str) -> dict:
    """Load pipeline parameters from the configuration file."""
    import yaml
    from pathlib import Path

    try:
        config_path = Path("conf/base/parameters_experimental.yml")
        if not config_path.exists():
            return {}
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        return params.get("pipelines", {}).get(pipeline_key, {})
    except Exception:
        return {}


def create_pipeline(pipeline_key) -> Pipeline:
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

    # Load pipeline parameters to determine required catalog inputs
    pipeline_params = _load_pipeline_params(pipeline_key)

    for step in step_names:
        # Check for input: parameters and build catalog inputs list
        step_params = pipeline_params.get(step, {})
        required_catalog_inputs = _extract_input_parameters(step_params)

        # Build inputs list - start with the basic inputs, then add catalog inputs (empty list if none)
        inputs = [prev_output, f"params:pipelines.{pipeline_key}.{step}"]
        inputs.extend(required_catalog_inputs)

        # Create generic estimator wrapper for all steps
        def create_estimator_wrapper(step_name, required_inputs):
            def estimator_wrapper(*args):
                X, params = args[0], args[1]
                # Map remaining args to catalog input names
                catalog_kwargs = {name: args[i + 2] for i, name in enumerate(required_inputs) if i + 2 < len(args)}
                return run_component_node(X, params, step_name, **catalog_kwargs)
            return estimator_wrapper

        nodes.append(
            node(
                func=create_estimator_wrapper(step, required_catalog_inputs),
                inputs=inputs,
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

    # Add scatter plot visualization node
    nodes.append(
        node(
            func=create_scatter_plot,
            inputs=[
                "scaler_output",
                "clusterer_output",
                "params:visualization.flip_x",
                "params:visualization.flip_y",
            ],
            outputs=f"{pipeline_key}.scatter_plot",
            name="create_scatter_plot",
        )
    )

    return Pipeline(nodes)
