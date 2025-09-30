from kedro.pipeline import Pipeline, node
from ..experimental.nodes import (
    load_polis_data,
    split_raw_data,
    dedup_votes,
    make_raw_vote_matrix,
    make_participant_mask,
    make_statement_mask,
    make_masked_vote_matrix,
)


def create_pipeline() -> Pipeline:
    """
    Create the preprocessing pipeline that runs up to masked_vote_matrix.

    This pipeline handles all the data loading and preprocessing steps that
    are common to all experimental pipelines, producing the masked_vote_matrix
    and related outputs that other pipelines can use as inputs.

    Returns:
        Pipeline: A Kedro pipeline containing all preprocessing nodes
    """
    nodes = [
        # Data loading nodes
        node(
            func=load_polis_data,
            inputs=[
                "params:base_url",
                "params:polis_url",
                "params:import_dir",
            ],
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
        # Preprocessing nodes
        node(
            func=make_participant_mask,
            inputs=["raw_vote_matrix", "params:min_votes_threshold"],
            outputs="participant_mask",
            name="make_participant_mask",
        ),
        node(
            func=make_statement_mask,
            inputs=["raw_comments"],
            outputs="statement_mask",
            name="make_statement_mask",
        ),
        node(
            func=make_masked_vote_matrix,
            inputs=["raw_vote_matrix", "statement_mask"],
            outputs="masked_vote_matrix",
            name="make_masked_vote_matrix",
        ),
    ]

    return Pipeline(nodes)
