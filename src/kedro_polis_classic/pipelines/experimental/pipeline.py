from kedro.pipeline import Pipeline, node
from .nodes import (
    run_component_node,
    create_labels_dataframe,
    create_scatter_plot,
    create_scatter_plot_by_participant_id,
    create_scatter_plot_by_vote_proportions,
    save_scatter_plot_image,
    create_votes_dataframe,
    save_projections_json,
    save_statements_json,
    save_meta_json,
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
            "params:polis_url",  # Keep polis_url parameter without namespace
            "params:base_url",  # Keep base_url parameter without namespace
            "params:import_dir",  # Keep import_dir parameter without namespace
            "params:min_votes_threshold",  # Keep min_votes_threshold parameter without namespace
        },
        outputs={
            "masked_vote_matrix",  # Keep masked_vote_matrix output without namespace
            "participant_mask",  # Keep participant_mask output without namespace
            "statement_mask",  # Keep statement_mask output without namespace
            "raw_vote_matrix",  # Keep raw_vote_matrix output without namespace
            "raw_comments",  # Keep raw_comments output without namespace
        },
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
                "participant_mask",
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

    # Add Red-Dwarf minimal dataset generation nodes
    # These generate the votes.parquet, projections.json, statements.json, and meta.json files

    # Generate votes dataframe for parquet storage
    nodes.append(
        node(
            func=create_votes_dataframe,
            inputs=[
                "raw_vote_matrix",
                "participant_mask",
            ],
            outputs=f"{pipeline_key}__votes_parquet",
            name="create_votes_dataframe",
        )
    )

    # Generate projections JSON
    nodes.append(
        node(
            func=save_projections_json,
            inputs=[
                f"{pipeline_key}__filter_output",
                "participant_mask",
            ],
            outputs=f"{pipeline_key}__projections_json",
            name="save_projections_json",
        )
    )

    # Generate statements JSON
    nodes.append(
        node(
            func=save_statements_json,
            inputs="raw_comments",
            outputs=f"{pipeline_key}__statements_json",
            name="save_statements_json",
        )
    )

    # Generate metadata JSON
    nodes.append(
        node(
            func=save_meta_json,
            inputs=[
                "params:polis_url",
                f"params:pipelines.{pipeline_key}.reducer",
            ],
            outputs=f"{pipeline_key}__meta_json",
            name="save_meta_json",
        )
    )

    # Combine preprocessing pipeline with experimental nodes
    return preprocessing_pipeline + Pipeline(nodes)


def create_branching_pipeline() -> Pipeline:
    """
    Create a branching DAG pipeline where preprocessing runs once, each reducer runs once,
    and then multiple clusterers run on each reducer's output.

    This approach eliminates redundant computation by sharing intermediate results
    across multiple pipeline branches.

    Returns:
        Pipeline: A Kedro pipeline containing the branching DAG structure
    """
    from kedro.config import OmegaConfigLoader

    # Load branching pipeline configuration
    config_loader = OmegaConfigLoader(
        conf_source="conf", base_env="base", default_run_env="local"
    )
    params = config_loader["parameters"]
    config = params.get("branching_pipeline", {})

    if not config:
        raise ValueError("No branching_pipeline configuration found in parameters")

    # Create the preprocessing pipeline with namespace
    preprocessing_pipeline = Pipeline(
        create_preprocessing_pipeline(),
        namespace="preprocessing",
        prefix_datasets_with_namespace=False,
        parameters={
            "params:polis_url",
            "params:base_url",
            "params:import_dir",
            "params:min_votes_threshold",
        },
        outputs={
            "masked_vote_matrix",
            "participant_mask",
            "statement_mask",
            "raw_vote_matrix",
            "raw_comments",
        },
    )

    nodes = []

    # Get configuration sections
    shared_stages = config.get("shared_stages", {})
    imputers = config.get("imputers", [])
    reducers = config.get("reducers", [])
    clusterers = config.get("clusterers", [])
    enabled_combinations = config.get("enabled_combinations", None)

    # If enabled_combinations is specified, filter to only those combinations
    if enabled_combinations:
        valid_combinations = set()
        for combo in enabled_combinations:
            # Support both 2-tuple (reducer, clusterer) and 3-tuple (imputer, reducer, clusterer)
            if len(combo) == 2:
                # Legacy format: assume first imputer for backward compatibility
                imputer_name = imputers[0]["name"] if imputers else "SimpleImputer"
                valid_combinations.add((imputer_name, combo["reducer"], combo["clusterer"]))
            else:
                valid_combinations.add((combo["imputer"], combo["reducer"], combo["clusterer"]))
    else:
        # Build all combinations
        valid_combinations = None

    # Create imputer-reducer-clusterer combinations
    for i, imputer_config in enumerate(imputers):
        imputer_name = imputer_config["name"]
        imputer_estimator = imputer_config["estimator"]
        imputer_key = f"imputer_{imputer_name.lower()}"

        # Check if any combinations are enabled for this imputer
        if valid_combinations:
            imputer_has_enabled_combos = any(
                combo[0] == imputer_estimator for combo in valid_combinations
            )
            if not imputer_has_enabled_combos:
                continue

        # Create imputer node
        required_catalog_inputs = _extract_input_parameters(imputer_config)
        inputs = ["masked_vote_matrix"]
        inputs.extend(required_catalog_inputs)

        def create_imputer_wrapper(imputer_cfg, required_inputs):
            def imputer_wrapper(*args):
                X = args[0]
                catalog_kwargs = {
                    name: args[i + 1]
                    for i, name in enumerate(required_inputs)
                    if i + 1 < len(args)
                }
                return run_component_node(X, imputer_cfg, "imputer", **catalog_kwargs)
            return imputer_wrapper

        nodes.append(node(
            func=create_imputer_wrapper(imputer_config, required_catalog_inputs),
            inputs=inputs,
            outputs=f"{imputer_key}__output",
            name=f"{imputer_key}_node",
            tags=["imputer", imputer_name.lower()]
        ))

        # For each imputer, create reducer branches
        for j, reducer_config in enumerate(reducers):
            reducer_name = reducer_config["name"]
            reducer_estimator = reducer_config["estimator"]
            imputer_reducer_key = f"{imputer_name.lower()}_{reducer_name.lower()}"

            # Check if any clusterer combinations are enabled for this imputer-reducer pair
            if valid_combinations:
                pair_has_enabled_combos = any(
                    combo[0] == imputer_estimator and combo[1] == reducer_estimator
                    for combo in valid_combinations
                )
                if not pair_has_enabled_combos:
                    continue

            # Create reducer node - exclude parameters starting with underscore (special config keys)
            reducer_config_clean = {k: v for k, v in reducer_config.items() if not k.startswith("_")}
            required_catalog_inputs = _extract_input_parameters(reducer_config_clean)
            inputs = [f"{imputer_key}__output"]
            inputs.extend(required_catalog_inputs)

            def create_reducer_wrapper(reducer_cfg, required_inputs):
                def reducer_wrapper(*args):
                    X = args[0]
                    catalog_kwargs = {
                        name: args[i + 1]
                        for i, name in enumerate(required_inputs)
                        if i + 1 < len(args)
                    }
                    return run_component_node(X, reducer_cfg, "reducer", **catalog_kwargs)
                return reducer_wrapper

            nodes.append(node(
                func=create_reducer_wrapper(reducer_config_clean, required_catalog_inputs),
                inputs=inputs,
                outputs=f"{imputer_reducer_key}__reducer_output",
                name=f"{imputer_reducer_key}_reducer_node",
                tags=["reducer", imputer_name.lower(), reducer_name.lower()]
            ))

            # For each imputer-reducer pair, branch to enabled clusterers
            for k, clusterer_config in enumerate(clusterers):
                clusterer_name = clusterer_config["name"]
                clusterer_estimator = clusterer_config["estimator"]
                combination_key = f"{imputer_name.lower()}_{reducer_name.lower()}_{clusterer_name.lower()}"

                # Skip if this combination is not enabled
                if valid_combinations and (imputer_estimator, reducer_estimator, clusterer_estimator) not in valid_combinations:
                    continue

                combination_tags = [
                    combination_key,
                    imputer_name.lower(),
                    reducer_name.lower(),
                    clusterer_name.lower(),
                    "combination"
                ]

                # Scaler node - now get scaler config from the reducer config
                scaler_config = reducer_config.get("_scaler", {})
                if scaler_config:
                    required_catalog_inputs = _extract_input_parameters(scaler_config)
                    inputs = [f"{imputer_reducer_key}__reducer_output"]
                    inputs.extend(required_catalog_inputs)

                    def create_scaler_wrapper(scaler_cfg, required_inputs):
                        def scaler_wrapper(*args):
                            X = args[0]
                            catalog_kwargs = {
                                name: args[i + 1]
                                for i, name in enumerate(required_inputs)
                                if i + 1 < len(args)
                            }
                            return run_component_node(X, scaler_cfg, "scaler", **catalog_kwargs)
                        return scaler_wrapper

                    nodes.append(node(
                        func=create_scaler_wrapper(scaler_config, required_catalog_inputs),
                        inputs=inputs,
                        outputs=f"{combination_key}__scaler_output",
                        name=f"{combination_key}_scaler_node",
                        tags=combination_tags + ["scaler"]
                    ))
                    scaler_output = f"{combination_key}__scaler_output"
                else:
                    scaler_output = f"{imputer_reducer_key}__reducer_output"

                # Filter node
                filter_config = shared_stages.get("filter", {})
                if filter_config:
                    required_catalog_inputs = _extract_input_parameters(filter_config)
                    inputs = [scaler_output]
                    inputs.extend(required_catalog_inputs)

                    def create_filter_wrapper(filter_cfg, required_inputs):
                        def filter_wrapper(*args):
                            X = args[0]
                            catalog_kwargs = {
                                name: args[i + 1]
                                for i, name in enumerate(required_inputs)
                                if i + 1 < len(args)
                            }
                            return run_component_node(X, filter_cfg, "filter", **catalog_kwargs)
                        return filter_wrapper

                    nodes.append(node(
                        func=create_filter_wrapper(filter_config, required_catalog_inputs),
                        inputs=inputs,
                        outputs=f"{combination_key}__filter_output",
                        name=f"{combination_key}_filter_node",
                        tags=combination_tags + ["filter"]
                    ))
                    filter_output = f"{combination_key}__filter_output"
                else:
                    filter_output = scaler_output

                # Clusterer node
                required_catalog_inputs = _extract_input_parameters(clusterer_config)
                inputs = [filter_output]
                inputs.extend(required_catalog_inputs)

                def create_clusterer_wrapper(clusterer_cfg, required_inputs):
                    def clusterer_wrapper(*args):
                        X = args[0]
                        catalog_kwargs = {
                            name: args[i + 1]
                            for i, name in enumerate(required_inputs)
                            if i + 1 < len(args)
                        }
                        return run_component_node(X, clusterer_cfg, "clusterer", **catalog_kwargs)
                    return clusterer_wrapper

                nodes.append(node(
                    func=create_clusterer_wrapper(clusterer_config, required_catalog_inputs),
                    inputs=inputs,
                    outputs=f"{combination_key}__clusterer_output",
                    name=f"{combination_key}_clusterer_node",
                    tags=combination_tags + ["clusterer"]
                ))

                # Add visualization nodes for this combination
                _add_visualization_nodes(nodes, combination_key, filter_output, combination_tags)

                # Add Red-Dwarf dataset generation nodes
                _add_dataset_generation_nodes(nodes, combination_key, filter_output, combination_tags)

    # Add single shared statements JSON node (same for all combinations)
    nodes.append(node(
        func=save_statements_json,
        inputs="raw_comments",
        outputs="statements_json",
        name="save_statements_json",
        tags=["shared", "dataset"]
    ))

    # Add single shared votes parquet node (same for all combinations)
    nodes.append(node(
        func=create_votes_dataframe,
        inputs=[
            "raw_vote_matrix",
            "participant_mask",
        ],
        outputs="votes_parquet",
        name="create_votes_dataframe",
        tags=["shared", "dataset"]
    ))

    return preprocessing_pipeline + Pipeline(nodes)


def _add_visualization_nodes(nodes: list, combination_key: str, filter_output: str, tags: list):
    """Add visualization nodes for a specific reducer-clusterer combination."""

    # Original scatter plot colored by cluster
    nodes.append(node(
        func=create_scatter_plot,
        inputs=[
            filter_output,
            f"{combination_key}__clusterer_output",
            "participant_mask",
            "params:visualization.flip_x",
            "params:visualization.flip_y",
        ],
        outputs=f"{combination_key}__scatter_plot",
        name=f"{combination_key}_create_scatter_plot",
        tags=tags + ["visualization"]
    ))

    # Scatter plot colored by participant ID
    nodes.append(node(
        func=create_scatter_plot_by_participant_id,
        inputs=[
            filter_output,
            "participant_mask",
            "params:visualization.flip_x",
            "params:visualization.flip_y",
        ],
        outputs=f"{combination_key}__scatter_plot_by_participant_id",
        name=f"{combination_key}_create_scatter_plot_by_participant_id",
        tags=tags + ["visualization"]
    ))

    # Only add plot image saving nodes if generate_plots is True
    if generate_plots:
        # Save scatter plot images
        def create_image_saver_wrapper(pipeline_name, plot_suffix=""):
            def image_saver_wrapper(scatter_plot):
                filename_suffix = f"_{plot_suffix}" if plot_suffix else ""
                return save_scatter_plot_image(
                    scatter_plot, f"{pipeline_name}{filename_suffix}"
                )
            return image_saver_wrapper

        # Save original cluster plot
        nodes.append(node(
            func=create_image_saver_wrapper(combination_key, "cluster"),
            inputs=f"{combination_key}__scatter_plot",
            outputs=f"{combination_key}__scatter_plot_image_path",
            name=f"{combination_key}_save_scatter_plot_image",
            tags=tags + ["visualization", "save"]
        ))

        # Save participant ID plot
        nodes.append(node(
            func=create_image_saver_wrapper(combination_key, "participant_id"),
            inputs=f"{combination_key}__scatter_plot_by_participant_id",
            outputs=f"{combination_key}__scatter_plot_by_participant_id_image_path",
            name=f"{combination_key}_save_scatter_plot_by_participant_id_image",
            tags=tags + ["visualization", "save"]
        ))


def _add_dataset_generation_nodes(nodes: list, combination_key: str, filter_output: str, tags: list):
    """Add Red-Dwarf dataset generation nodes for a specific reducer-clusterer combination."""

    # Note: votes_parquet is now shared across all combinations since it only depends on
    # raw_vote_matrix and participant_mask, which are the same for all combinations

    # Generate projections JSON
    nodes.append(node(
        func=save_projections_json,
        inputs=[
            filter_output,
            "participant_mask",
        ],
        outputs=f"{combination_key}__projections_json",
        name=f"{combination_key}_save_projections_json",
        tags=tags + ["dataset"]
    ))

    # Note: statements_json is now shared across all combinations since it's just raw comments
    # No need to create combination-specific statements JSON nodes

    # Generate metadata JSON - we need to find the right reducer config
    # For new format: imputer_reducer_clusterer, reducer is the middle part
    # For old format: reducer_clusterer, reducer is the first part
    parts = combination_key.split('_')
    if len(parts) == 3:
        # New format: imputer_reducer_clusterer
        reducer_name = parts[1]
    else:
        # Old format: reducer_clusterer (backward compatibility)
        reducer_name = parts[0]

    # Create a wrapper that finds the right reducer config
    def create_meta_wrapper(combo_key):
        def meta_wrapper(polis_url, all_reducers):
            # Find the reducer config for this combination
            parts = combo_key.split('_')
            if len(parts) == 3:
                # New format: imputer_reducer_clusterer
                reducer_name = parts[1]
            else:
                # Old format: reducer_clusterer
                reducer_name = parts[0]

            reducer_config = None
            for reducer in all_reducers:
                if reducer["name"].lower() == reducer_name:
                    reducer_config = reducer
                    break
            return save_meta_json(polis_url, reducer_config)
        return meta_wrapper

    nodes.append(node(
        func=create_meta_wrapper(combination_key),
        inputs=[
            "params:polis_url",
            f"params:branching_pipeline.reducers",
        ],
        outputs=f"{combination_key}__meta_json",
        name=f"{combination_key}_save_meta_json",
        tags=tags + ["dataset"]
    ))


def create_estimator_wrapper(step_name, required_inputs):
    """Create a wrapper function for estimator nodes that handles catalog inputs."""
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
