import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from kedro_polis_classic.datasets.polis_api import PolisAPIDataset

# Helpers

import functools
import inspect

def process_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Ensure x is a pandas Series (e.g., extract first column of a single-column DataFrame)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame")
        return x.iloc[:, 0]
    elif isinstance(x, pd.Series):
        return x
    else:
        raise TypeError("Expected Series or single-column DataFrame")


def ensure_series(argname: str):
    """Decorator to apply `process_series` to a named argument."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind args and kwargs to named parameters
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if argname not in bound.arguments:
                raise ValueError(f"Argument '{argname}' not found when calling {func.__name__}")

            bound.arguments[argname] = process_series(bound.arguments[argname])
            return func(*bound.args, **bound.kwargs)

        return wrapper
    return decorator

@ensure_series('statement_mask')
def apply_statement_filter(matrix: pd.DataFrame, statement_mask: pd.Series) -> pd.DataFrame:
    """Filter out moderated statements from the vote matrix"""
    # Filter to only statements that are True in the mask
    unfiltered_statement_ids = statement_mask.loc[statement_mask].index
    # Convert to strings as more universal type.
    # NOTE: Are the any circumstances where `matrix` might still have numeric column names?
    return matrix.loc[:, unfiltered_statement_ids.astype(str)]

@ensure_series('participant_mask')
def apply_participant_filter(matrix: pd.DataFrame, participant_mask: pd.Series) -> pd.DataFrame:
    """Filter out participants who don't meet the minimum vote threshold"""
    # Filter to only participants that are True in the mask
    unfiltered_participant_ids = participant_mask.loc[participant_mask].index
    return matrix.loc[unfiltered_participant_ids, :]

# Nodes

def load_polis_data(report_id: str):
    dataset = PolisAPIDataset(report_id=report_id)
    return dataset.load()

def split_raw_data(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return raw_data["votes"], raw_data["comments"]

def dedup_votes(raw_votes: pd.DataFrame) -> pd.DataFrame:
    # 1. Sort so newest votes are last
    votes_sorted = raw_votes.sort_values("timestamp")

    # 2. Drop duplicates, keeping the most recent
    deduped_votes = votes_sorted.drop_duplicates(
        subset=["voter-id", "comment-id"], keep="last"
    )

    return deduped_votes

def make_raw_vote_matrix(deduped_votes: pd.DataFrame) -> pd.DataFrame:
    matrix = deduped_votes.pivot(
        index="voter-id",
        columns="comment-id",
        values="vote"
    )

    return matrix

def make_participant_mask(matrix: pd.DataFrame, min_votes: int = 7) -> pd.Series:
    mask = matrix.count(axis="columns") >= min_votes

    mask.index.name = "voter-id"
    mask.name = "participant-in" # sample-in
    return mask

def make_statement_mask(comments: pd.DataFrame, strict_moderation: bool = True) -> pd.Series:
    """Return a mask for unmoderated statements.

    If `strict_moderation=True`, only keep comments explicitly moderated in (`moderated=1`).
    If `strict_moderation=False`, allow unmoderated (`moderated=0`).
    """
    threshold = 1 if strict_moderation else 0
    mask = comments["moderated"] >= threshold

    mask.name = "statement-in" # feature-in
    return mask

def create_filtered_vote_matrix(
    raw_vote_matrix: pd.DataFrame,
    participant_mask: pd.Series,
    statement_mask: pd.Series
) -> pd.DataFrame:
    """Apply both participant and statement filters to create the final filtered matrix"""
    # First filter statements (columns)
    raw_vote_matrix = apply_statement_filter(raw_vote_matrix, statement_mask)

    # Then filter participants (rows)
    raw_vote_matrix = apply_participant_filter(raw_vote_matrix, participant_mask)

    return raw_vote_matrix

def run_pca(matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(matrix)
    return pd.DataFrame(components, index=matrix.index, columns=["x", "y"])

def cluster_kmeans(matrix: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(matrix)
    return pd.Series(kmeans.labels_, index=matrix.index)

def create_vote_heatmap(filtered_matrix: pd.DataFrame) -> go.Figure:
    """
    Create a plotly heatmap of the filtered vote matrix with custom color scheme:
    - Red (-1) for disagree
    - White (0) for neutral/pass
    - Green (+1) for agree
    - Pale yellow for missing votes (NaN)
    """

    # Create a copy of the matrix for display
    display_matrix = filtered_matrix.copy()
    display_matrix.sort_index(inplace=True, ascending=False)

    # Create custom colorscale (matching Polis website)
    # NaN's are handled below as background color.
    polisColorScale = [
        [0.0, '#e74c3c'],    # Red for -1
        [0.5, '#e6e6e6'],    # White for 0
        [1.0, '#2ecc71']     # Green for +1
    ]

    # Same colorscale used in CompDem analysis notebooks
    # See: https://github.com/compdemocracy/analysis/blob/acc27dca89a37f8690e32dbd40aa8bc5ebfa851c/notebooks/jupyter/american-assembly-bg-analysis.heatmap.v0.6.ipynb
    analysisColorScale = px.colors.diverging.RdYlBu

    # Create the base heatmap
    fig = go.Figure(data=go.Heatmap(
        z=display_matrix.values,
        x=[f"{col}" for col in display_matrix.columns],
        y=[f"{idx}" for idx in display_matrix.index],
        colorscale=analysisColorScale,
        zmin=-1,
        zmid=0,
        zmax=1,
        hoverongaps=False,
        hovertemplate='Participant: %{y}<br>Statement: %{x}<br>Vote: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Vote",
            tickvals=[-1, 0, 1],
            ticktext=["Disagree", "Pass", "Agree"]
        )
    ))

    # Update layout
    fig.update_layout(
        title="Polis Vote Matrix Heatmap (Filtered)",
        xaxis_title="Statements",
        yaxis_title="Participants",
        width=max(800, len(display_matrix.columns) * 20),
        height=max(600, len(display_matrix.index) * 15),
        # xaxis=dict(tickangle=45),
        font=dict(size=10),
        plot_bgcolor='white',
    )

    return fig

def save_heatmap_html(fig: go.Figure, filepath: str = "data/08_reporting/vote_heatmap.html"):
    """Save the plotly figure as an HTML file"""
    fig.write_html(filepath)
    return filepath
