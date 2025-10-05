from kedro.pipeline import Pipeline, node
from .nodes import (
    filter_votes_for_islands,
    aggregate_participant_islands,
    load_island_geojson,
    create_island_shapes,
    assign_participant_coordinates,
    create_participant_geojson,
    save_participant_geojson,
    create_geographic_scatter_plot,
)
from ..preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the geographic projection pipeline.

    This pipeline takes votes data and creates geographic projections by:
    1. Filtering votes for island-related statements
    2. Aggregating participant preferences by island
    3. Loading island geographic data
    4. Assigning random coordinates within preferred islands
    5. Creating and saving participant GeoJSON

    Returns:
        Pipeline: A Kedro pipeline for geographic projection
    """
    # Create the preprocessing pipeline to get deduped_votes
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
            "deduped_votes",
            "raw_comments",
        },
    )

    # Geographic processing nodes
    geographic_nodes = Pipeline(
        [
            node(
                func=filter_votes_for_islands,
                inputs="deduped_votes",
                outputs="filtered_island_votes",
                name="filter_votes_for_islands",
            ),
            node(
                func=aggregate_participant_islands,
                inputs="filtered_island_votes",
                outputs="participant_islands",
                name="aggregate_participant_islands",
            ),
            node(
                func=load_island_geojson,
                inputs="params:geographic.geojson_path",
                outputs="islands_geojson",
                name="load_island_geojson",
            ),
            node(
                func=create_island_shapes,
                inputs="islands_geojson",
                outputs="island_shapes",
                name="create_island_shapes",
            ),
            node(
                func=assign_participant_coordinates,
                inputs=["participant_islands", "island_shapes"],
                outputs="participant_features",
                name="assign_participant_coordinates",
            ),
            node(
                func=create_participant_geojson,
                inputs="participant_features",
                outputs="participant_geojson",
                name="create_participant_geojson",
            ),
            node(
                func=save_participant_geojson,
                inputs=["participant_geojson", "params:geographic.output_path"],
                outputs="participant_geojson_path",
                name="save_participant_geojson",
            ),
            node(
                func=create_geographic_scatter_plot,
                inputs="participant_geojson",
                outputs="geographic__scatter_plot",
                name="create_scatter_plot",
            ),
        ]
    )

    # Combine preprocessing pipeline with geographic nodes
    return preprocessing_pipeline + geographic_nodes
