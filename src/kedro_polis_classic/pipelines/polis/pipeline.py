from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(load_polis_data, inputs="params:report_id", outputs="raw_data", name="load_polis_data"),
        node(split_raw_data, inputs="raw_data", outputs=["raw_votes", "raw_comments"], name="split_raw_data"),
        node(dedup_votes, inputs="raw_votes", outputs="deduped_votes", name="dedup_votes"),
        node(make_raw_vote_matrix, inputs="deduped_votes", outputs="raw_vote_matrix", name="make_raw_matrix"),
        node(filter_participants, inputs=["raw_vote_matrix", "params:min_votes_threshold"], outputs="participant_filter_mask", name="filter_participants"),
        node(filter_statements, inputs="raw_comments", outputs="statement_filter_mask", name="filter_statements"),
        node(create_filtered_vote_matrix, inputs=["raw_vote_matrix", "participant_filter_mask", "statement_filter_mask"], outputs="filtered_vote_matrix", name="create_filtered_matrix"),
        node(create_vote_heatmap, inputs="filtered_vote_matrix", outputs="vote_heatmap_fig", name="create_heatmap"),
        node(save_heatmap_html, inputs="vote_heatmap_fig", outputs="heatmap_filepath", name="save_heatmap"),
        #node(run_pca, inputs="filtered_vote_matrix", outputs="participant_projections", name="run_pca"),
        #node(cluster_kmeans, inputs="participant_projections", outputs="labels", name="kmeans_cluster"),
    ])
