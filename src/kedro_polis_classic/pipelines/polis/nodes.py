import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def split_raw_data(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return raw_data["votes"], raw_data["comments"]

def make_raw_vote_matrix(votes: pd.DataFrame) -> pd.DataFrame:
    # 1. Sort so newest votes are last
    votes_sorted = votes.sort_values("timestamp")

    # 2. Drop duplicates, keeping the most recent
    deduped_votes = votes_sorted.drop_duplicates(
        subset=["voter-id", "comment-id"], keep="last"
    )

    # 3. Pivot to wide matrix format
    matrix = deduped_votes.pivot(
        index="voter-id",
        columns="comment-id",
        values="vote"
    ).fillna(0)

    return matrix

def apply_mask(matrix: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return matrix.loc[:, mask]

def run_pca(matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix)
    return pd.DataFrame(components, index=matrix.index, columns=["x", "y"])

def filter_participants(matrix: pd.DataFrame, min_votes: int = 5) -> pd.Series:
    return matrix.abs().sum(axis=1) >= min_votes

def cluster_kmeans(matrix: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(matrix)
    return pd.Series(kmeans.labels_, index=matrix.index)
