from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(split_raw_data, inputs="raw_data", outputs=["raw_votes", "raw_comments"], name="split_raw_data"),
        node(dedup_votes, inputs="raw_votes", outputs="deduped_votes", name="dedup_votes"),
        node(make_raw_vote_matrix, inputs="deduped_votes", outputs="raw_vote_matrix", name="make_raw_matrix"),
        node(filter_participants, inputs=["raw_vote_matrix", "params:min_votes_threshold"], outputs="participant_filter_mask", name="filter_participants"),
        node(filter_statements, inputs="raw_comments", outputs="statement_filter_mask", name="filter_statements"),
        #node(apply_mask, inputs=["raw_vote_matrix", "params:mask"], outputs="masked_vote_matrix", name="mask_matrix"),
        #node(run_pca, inputs="masked_vote_matrix", outputs="participant_projections", name="run_pca"),
        #node(cluster_kmeans, inputs="participant_projections", outputs="labels", name="kmeans_cluster"),
    ])
