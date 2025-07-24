import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

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

def filter_participants(matrix: pd.DataFrame, min_votes: int = 7) -> pd.Series:
    mask = matrix.count(axis="columns") >= min_votes
    mask.index.name = "voter-id"
    mask.name = "meets-threshold"
    return mask

def filter_statements(comments: pd.DataFrame) -> pd.Series:
    """Create a mask to filter out moderated statements (moderated = -1)"""
    # Create a mask indexed by comment-id (which matches the vote matrix columns)
    mask = comments["moderated"] > -1
    # Set the index to comment-id so it aligns with vote matrix columns
    mask.index = comments["comment-id"]
    mask.name = "mod-in"
    return mask

def apply_statement_filter(matrix: pd.DataFrame, statement_mask: pd.Series) -> pd.DataFrame:
    """Filter out moderated statements from the vote matrix"""
    # Convert DataFrame to Series if needed (when loaded from CSV)
    if isinstance(statement_mask, pd.DataFrame):
        statement_mask = statement_mask.set_index(statement_mask.columns[0]).iloc[:, 0]

    matrix.set_index("voter-id", inplace=True)
    # Only keep comment IDs that exist in both the matrix and the mask
    string_statement_ids = [str(i) for i in statement_mask.index]
    common_comment_ids = matrix.columns.intersection(string_statement_ids)
    return matrix.loc[:, common_comment_ids]

def apply_participant_filter(matrix: pd.DataFrame, participant_mask: pd.Series) -> pd.DataFrame:
    """Filter out participants who don't meet the minimum vote threshold"""
    # Convert DataFrame to Series if needed (when loaded from CSV)
    if isinstance(participant_mask, pd.DataFrame):
        # Set the first column as index and take the second column as the series
        participant_mask = participant_mask.set_index(participant_mask.columns[0]).iloc[:, 0]

    # Filter to only participants that are True in the mask
    # Use .loc with boolean indexing on rows
    return matrix.loc[participant_mask]

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
    display_matrix.set_index("voter-id", inplace=True)
    display_matrix.sort_index(inplace=True, ascending=False)

    # Create custom colorscale (matching Polis website)
    # NaN's are handled below as background color.
    colorscale = [
        [0.0, '#e74c3c'],    # Red for -1
        [0.5, '#e6e6e6'],    # White for 0
        [1.0, '#2ecc71']     # Green for +1
    ]

    # Create the base heatmap
    fig = go.Figure(data=go.Heatmap(
        z=display_matrix.values,
        x=[f"Statement {col}" for col in display_matrix.columns],
        y=[f"Participant {idx}" for idx in display_matrix.index],
        colorscale=colorscale,
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
        # Show missing votes as white
        plot_bgcolor='white',
    )

    return fig

def save_heatmap_html(fig: go.Figure, filepath: str = "data/08_reporting/vote_heatmap.html"):
    """Save the plotly figure as an HTML file"""
    fig.write_html(filepath)
    return filepath
