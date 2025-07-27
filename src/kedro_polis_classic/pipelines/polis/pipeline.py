from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as n

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(n.load_polis_data, inputs="params:report_id", outputs="raw_data", name="load_polis_data"),
        node(n.split_raw_data, inputs="raw_data", outputs=["raw_votes", "raw_comments"], name="split_raw_data"),
        node(n.dedup_votes, inputs="raw_votes", outputs="deduped_votes", name="dedup_votes"),
        node(n.make_raw_vote_matrix, inputs="deduped_votes", outputs="raw_vote_matrix", name="make_raw_matrix"),
        node(n.make_participant_mask, inputs=["raw_vote_matrix", "params:min_votes_threshold"], outputs="participant_filter_mask", name="make_participant_mask"),
        node(n.make_statement_mask, inputs=["raw_comments", "params:strict_moderation"], outputs="statement_filter_mask", name="make_statement_mask"),
        node(n.create_filtered_vote_matrix, inputs=["raw_vote_matrix", "participant_filter_mask", "statement_filter_mask"], outputs="filtered_vote_matrix", name="create_filtered_matrix"),
        node(n.create_vote_heatmap, inputs="filtered_vote_matrix", outputs="vote_heatmap_fig", name="create_heatmap"),
        node(n.save_heatmap_html, inputs="vote_heatmap_fig", outputs="heatmap_filepath", name="save_heatmap"),
        #node(n.run_pca, inputs="filtered_vote_matrix", outputs="participant_projections", name="run_pca"),
        #node(n.cluster_kmeans, inputs="participant_projections", outputs="labels", name="kmeans_cluster"),
    ])
