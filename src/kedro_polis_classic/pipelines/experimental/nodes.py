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
    step_config = copy.deepcopy(params[step_name])
    pipeline = build_pipeline_from_params({step_name: step_config})

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
