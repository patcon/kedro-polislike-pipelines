from kedro.pipeline import Pipeline, node
from .nodes import (
    run_component_node,
    create_labels_dataframe,
    create_scatter_plot,
    create_scatter_plot_by_participant_id,
    create_scatter_plot_by_vote_proportions,
    save_scatter_plot_image,
)
from ..config import load_pipelines_config
from ..preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline


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


def create_pipeline(pipeline_key) -> Pipeline:
    """
    Create an experimental pipeline that includes preprocessing and experimental processing.
    
    This pipeline combines the preprocessing pipeline (with namespace) and the experimental
    processing nodes into a single pipeline that can be run independently.
    
    Args:
        pipeline_key: The key identifying which pipeline configuration to use
        
    Returns:
        Pipeline: A Kedro pipeline containing both preprocessing and experimental nodes
    """
    # Load pipeline parameters
    pipelines_config = load_pipelines_config()
    pipeline_params = pipelines_config.get(pipeline_key, {})

    # Create the preprocessing pipeline with namespace
    preprocessing_pipeline = Pipeline(
        create_preprocessing_pipeline(),
        namespace="preprocessing",
        prefix_datasets_with_namespace=False,
        parameters={
            "params:polis_id",  # Keep polis_id parameter without namespace
            "params:min_votes_threshold",  # Keep min_votes_threshold parameter without namespace
        },
        outputs={
            "masked_vote_matrix",  # Keep masked_vote_matrix output without namespace
            "participant_mask",    # Keep participant_mask output without namespace
            "statement_mask",      # Keep statement_mask output without namespace
            "raw_vote_matrix",     # Keep raw_vote_matrix output without namespace
            "raw_comments",        # Keep raw_comments output without namespace
        }
    )

    nodes = []

    # Component processing nodes
    step_names = ["imputer", "reducer", "scaler", "filter", "clusterer"]
    prev_output = "masked_vote_matrix"  # Use masked vote matrix as input to components

    for step in step_names:
        # All steps are now explicitly defined in the YAML config
        step_params = pipeline_params[step]

        # Check for input: parameters and build catalog inputs list
        required_catalog_inputs = _extract_input_parameters(step_params)

        # Build inputs list - start with the basic inputs, then add catalog inputs (empty list if none)
        inputs = [prev_output, f"params:pipelines.{pipeline_key}.{step}"]
        inputs.extend(required_catalog_inputs)

        # Create generic estimator wrapper for all steps
        def create_estimator_wrapper(step_name, required_inputs):
            def estimator_wrapper(*args):
                X, params = args[0], args[1]
                # Map remaining args to catalog input names
                catalog_kwargs = {
                    name: args[i + 2]
                    for i, name in enumerate(required_inputs)
                    if i + 2 < len(args)
                }
                return run_component_node(X, params, step_name, **catalog_kwargs)

            return estimator_wrapper

        nodes.append(
            node(
                func=create_estimator_wrapper(step, required_catalog_inputs),
                inputs=inputs,
                outputs=f"{pipeline_key}__{step}_output",
                name=f"{step}_node",
            )
        )
        prev_output = f"{pipeline_key}__{step}_output"

    # Add scatter plot visualization nodes
    # Always use filter_output since we now guarantee a filter step exists

    # Original scatter plot colored by cluster
    nodes.append(
        node(
            func=create_scatter_plot,
            inputs=[
                f"{pipeline_key}__filter_output",
                f"{pipeline_key}__clusterer_output",
                "params:visualization.flip_x",
                "params:visualization.flip_y",
            ],
            outputs=f"{pipeline_key}__scatter_plot",
            name="create_scatter_plot",
        )
    )

    # Scatter plot colored by participant ID
    nodes.append(
        node(
            func=create_scatter_plot_by_participant_id,
            inputs=[
                f"{pipeline_key}__filter_output",
                "participant_mask",
                "params:visualization.flip_x",
                "params:visualization.flip_y",
            ],
            outputs=f"{pipeline_key}__scatter_plot_by_participant_id",
            name="create_scatter_plot_by_participant_id",
        )
    )

    # Add scatter plot image saving nodes
    def create_image_saver_wrapper(pipeline_name, plot_suffix=""):
        def image_saver_wrapper(scatter_plot):
            filename_suffix = f"_{plot_suffix}" if plot_suffix else ""
            return save_scatter_plot_image(
                scatter_plot, f"{pipeline_name}{filename_suffix}"
            )

        return image_saver_wrapper

    # Save original cluster plot
    nodes.append(
        node(
            func=create_image_saver_wrapper(pipeline_key, "cluster"),
            inputs=f"{pipeline_key}__scatter_plot",
            outputs=f"{pipeline_key}__scatter_plot_image_path",
            name="save_scatter_plot_image",
        )
    )

    # Save participant ID plot
    nodes.append(
        node(
            func=create_image_saver_wrapper(pipeline_key, "participant_id"),
            inputs=f"{pipeline_key}__scatter_plot_by_participant_id",
            outputs=f"{pipeline_key}__scatter_plot_by_participant_id_image_path",
            name="save_scatter_plot_by_participant_id_image",
        )
    )

    # Combine preprocessing pipeline with experimental nodes
    return preprocessing_pipeline + Pipeline(nodes)
