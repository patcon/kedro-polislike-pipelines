from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as n


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                n.load_polis_data,
                inputs="params:polis_id",
                outputs="raw_data",
                name="load_polis_data",
            ),
            node(
                n.split_raw_data,
                inputs="raw_data",
                outputs=["raw_votes", "raw_comments"],
                name="split_raw_data",
            ),
            node(
                n.dedup_votes,
                inputs="raw_votes",
                outputs="deduped_votes",
                name="dedup_votes",
            ),
            node(
                n.make_raw_vote_matrix,
                inputs="deduped_votes",
                outputs="raw_vote_matrix",
                name="make_raw_matrix",
            ),
            node(
                n.create_participants_meta,
                inputs=["raw_vote_matrix", "raw_comments"],
                outputs="raw_participants_meta",
                name="create_participants_meta",
            ),
            node(
                n.make_participant_mask,
                inputs=["raw_vote_matrix", "params:min_votes_threshold"],
                outputs="participant_filter_mask",
                name="make_participant_mask",
            ),
            node(
                n.make_statement_mask,
                inputs=["raw_comments", "params:strict_moderation"],
                outputs="statement_filter_mask",
                name="make_statement_mask",
            ),
            node(
                n.make_masked_vote_matrix,
                inputs=["raw_vote_matrix", "statement_filter_mask"],
                outputs="masked_vote_matrix",
                name="create_masked_matrix",
            ),
            node(
                n.create_vote_heatmap,
                inputs=[
                    "raw_vote_matrix",
                    "participant_filter_mask",
                    "statement_filter_mask",
                ],
                outputs="vote_heatmap_fig",
                name="create_heatmap",
            ),
            # PCA Subpipeline
            node(
                n.mean_impute_vote_matrix,
                inputs="masked_vote_matrix",
                outputs="imputed_vote_matrix",
                name="mean_impute_matrix",
            ),
            node(
                n.reduce_with_pca,
                inputs=["imputed_vote_matrix", "params:pca.n_components"],
                outputs="pca_components",
                name="reduce_with_pca",
            ),
            node(
                n.apply_sparsity_aware_scaler,
                inputs=["pca_components", "masked_vote_matrix"],
                outputs="scaled_pca_components",
                name="apply_sparsity_scaler",
            ),
            node(
                n.create_pca_scatter_plots,
                inputs=[
                    "scaled_pca_components",
                    "raw_participants_meta",
                    "params:pca.flip_x",
                    "params:pca.flip_y",
                ],
                outputs=["pca_scatter_fig_pids", "pca_scatter_fig_vote_counts"],
                name="create_pca_plots",
            ),
            # Polismath JSON reporting
            node(
                n.generate_polismath_json,
                inputs=["raw_vote_matrix", "raw_comments", "participant_filter_mask"],
                outputs="polismath_json",
                name="generate_polismath_json",
            ),
            # node(n.cluster_kmeans, inputs="participant_projections", outputs="labels", name="kmeans_cluster"),
        ]
    )
