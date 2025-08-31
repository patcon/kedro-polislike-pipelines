from ..builder import build_pipeline_from_params
import copy
import pandas as pd
from kedro_polis_classic.datasets.polis_api import PolisAPIDataset


def run_component_node(X, params, step_name):
    """
    Runs a single pipeline component.
    X: input features
    params: full nested pipeline parameters dict
    step_name: which step to build (imputer/reducer/scaler/clusterer)
    """
    # copy to avoid mutating params
    step_config = params
    pipeline = build_pipeline_from_params({step_name: step_config})

    # For clusterer, use fit_predict to get labels instead of fit_transform
    if step_name == "clusterer":
        return pipeline.fit_predict(X)
    else:
        return pipeline.fit_transform(X)


# Minimal data loader nodes from original polis pipeline


def load_polis_data(report_id: str):
    """Load raw data from Polis API"""
    dataset = PolisAPIDataset(report_id=report_id)
    return dataset.load()


def split_raw_data(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw data into votes and comments"""
    return raw_data["votes"], raw_data["comments"]


def dedup_votes(raw_votes: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate votes, keeping the most recent"""
    # Sort so newest votes are last
    votes_sorted = raw_votes.sort_values("timestamp")

    # Drop duplicates, keeping the most recent
    deduped_votes = votes_sorted.drop_duplicates(
        subset=["voter-id", "comment-id"], keep="last"
    )

    return deduped_votes


def make_raw_vote_matrix(deduped_votes: pd.DataFrame) -> pd.DataFrame:
    """Create vote matrix from deduplicated votes"""
    matrix = deduped_votes.pivot(index="voter-id", columns="comment-id", values="vote")

    # Convert vote values to integers (handles NaN values properly)
    matrix = matrix.astype("Int64")

    return matrix


def create_labels_dataframe(
    clusterer_output, raw_vote_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Create labels dataframe from clusterer output and raw vote matrix.

    Args:
        clusterer_output: Output from the clusterer (numpy array of labels)
        raw_vote_matrix: Raw vote matrix with participant IDs as index

    Returns:
        DataFrame with participant_id and label columns
    """
    import numpy as np

    # Convert clusterer output to numpy array if it isn't already
    if not isinstance(clusterer_output, np.ndarray):
        labels = np.array(clusterer_output)
    else:
        labels = clusterer_output

    # Flatten the labels array to ensure it's 1-dimensional
    labels = labels.flatten()

    # Get participant IDs from the raw vote matrix index
    participant_ids = raw_vote_matrix.index.tolist()

    # Ensure the lengths match
    if len(labels) != len(participant_ids):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(participant_ids)} participants"
        )

    # Create the labels dataframe
    labels_df = pd.DataFrame({"participant_id": participant_ids, "label": labels})

    return labels_df
