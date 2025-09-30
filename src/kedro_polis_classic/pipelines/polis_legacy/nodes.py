import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from kedro_polis_classic.datasets.polis_api import PolisAPIDataset
from .utils import ensure_series
from reddwarf.sklearn.transformers import SparsityAwareScaler
from reddwarf.utils.statements import process_statements

# Helpers


@ensure_series("statement_mask")
def _apply_statement_filter(
    matrix: pd.DataFrame, statement_mask: pd.Series, filter_type: str = "fill_zero"
) -> pd.DataFrame:
    """Filter out moderated statements from the vote matrix

    Args:
        matrix: Vote matrix with statements as columns
        statement_mask: Boolean mask indicating which statements to keep
        filter_type: Strategy for handling filtered statements:
            - "fill_zero": Keep all columns but fill filtered statements with 0 (default)
            - "drop": Remove columns for filtered statements
    """
    # Convert statement IDs to strings as more universal type
    # NOTE: Are there any circumstances where `matrix` might still have numeric column names?
    statement_mask = statement_mask.copy()
    statement_mask.index = statement_mask.index.astype(str)

    if filter_type == "drop":
        # Filter to only statements that are True in the mask
        unfiltered_statement_ids = statement_mask.loc[statement_mask].index
        return matrix.loc[:, unfiltered_statement_ids]

    elif filter_type == "fill_zero":
        # Create a copy to avoid modifying the original
        result = matrix.copy()
        # Get statement IDs that should be filtered out (False in mask)
        filtered_statement_ids = statement_mask.loc[~statement_mask].index
        # Fill filtered columns with 0, only for columns that exist in the matrix
        existing_filtered_cols = [
            col for col in filtered_statement_ids if col in result.columns
        ]
        if existing_filtered_cols:
            result.loc[:, existing_filtered_cols] = 0
        return result

    else:
        raise ValueError(
            f"Invalid filter_type '{filter_type}'. Must be 'drop' or 'fill_zero'."
        )


@ensure_series("participant_mask")
def _apply_participant_filter(
    matrix: pd.DataFrame, participant_mask: pd.Series
) -> pd.DataFrame:
    """Filter out participants who don't meet the minimum vote threshold"""
    # Filter to only participants that are True in the mask
    unfiltered_participant_ids = participant_mask.loc[participant_mask].index
    return matrix.loc[unfiltered_participant_ids, :]


def _create_filtered_vote_matrix(
    raw_vote_matrix: pd.DataFrame,
    participant_mask: Optional[pd.Series] = None,
    statement_mask: Optional[pd.Series] = None,
    filter_type: str = "fill_zero",
) -> pd.DataFrame:
    filtered_matrix = raw_vote_matrix.copy()
    # First filter statements (columns)
    if statement_mask is not None:
        filtered_matrix = _apply_statement_filter(
            filtered_matrix, statement_mask, filter_type=filter_type
        )

    # Then filter participants (rows)
    if participant_mask is not None:
        filtered_matrix = _apply_participant_filter(filtered_matrix, participant_mask)

    # Ensure vote values are integers
    filtered_matrix = filtered_matrix.astype("Int64")

    return filtered_matrix


def _create_scatter_plot(
    data: pd.DataFrame,
    flip_x: bool,
    flip_y: bool,
    colorbar_title: str,
    color_values: pd.Series,
    title: str,
) -> go.Figure:
    """
    Helper function to create a 2D or 3D scatter plot based on the number of columns in the data.

    Args:
        data: DataFrame with data for plotting
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1
        colorbar_title: Title for the colorbar
        color_values: Data for the marker color
        title: Title for the plot

    Returns:
        A Plotly figure (2D or 3D scatter plot)
    """

    # Get column names
    x_col = data.columns[0]
    y_col = data.columns[1]

    # Apply flipping if requested
    x_data = data[x_col] * (-1 if flip_x else 1)
    y_data = data[y_col] * (-1 if flip_y else 1)

    # Check for 2D or 3D plot based on column count
    if len(data.columns) == 3:
        # 3D scatter plot
        z_col = data.columns[2]
        fig = go.Figure(
            data=go.Scatter3d(
                x=x_data,
                y=y_data,
                z=data[z_col],
                mode="markers",
                marker=dict(
                    size=6,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=colorbar_title),
                ),
                text=[f"Participant {idx}" for idx in range(len(data))],
                hovertemplate=f"%{{text}}<br>{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<br>{z_col}: %{{z:.3f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"{x_col.upper()} Component",
                yaxis_title=f"{y_col.upper()} Component",
                zaxis_title=f"{z_col.upper()} Component",
            ),
            width=800,
            height=600,
        )

    elif len(data.columns) == 2:
        # 2D scatter plot
        fig = go.Figure(
            data=go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=colorbar_title),
                ),
                text=[f"Participant {idx}" for idx in range(len(data))],
                hovertemplate=f"%{{text}}<br>{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=f"{x_col.upper()} Component",
            yaxis_title=f"{y_col.upper()} Component",
            width=800,
            height=600,
            plot_bgcolor="white",
        )

    else:
        raise ValueError("Data must have exactly 2 or 3 columns for 2D or 3D plots.")

    return fig


# Nodes


def load_polis_data(polis_id: str):
    dataset = PolisAPIDataset(polis_id=polis_id)
    return dataset.load()


def split_raw_data(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return raw_data["votes"], raw_data["comments"]


def dedup_votes(raw_votes: pd.DataFrame) -> pd.DataFrame:
    # 1. Sort so newest votes are last
    votes_sorted = raw_votes.sort_values("modified")

    # 2. Drop duplicates, keeping the most recent
    deduped_votes = votes_sorted.drop_duplicates(
        subset=["participant_id", "statement_id"], keep="last"
    )

    return deduped_votes


def make_raw_vote_matrix(deduped_votes: pd.DataFrame) -> pd.DataFrame:
    matrix = deduped_votes.pivot(
        index="participant_id", columns="statement_id", values="vote"
    )

    # Convert vote values to integers (handles NaN values properly)
    matrix = matrix.astype("Int64")

    return matrix


def make_participant_mask(matrix: pd.DataFrame, min_votes: int = 7) -> pd.Series:
    mask = matrix.count(axis="columns") >= min_votes

    mask.index.name = "participant_id"
    mask.name = "participant-in"  # sample-in
    return mask


def make_statement_mask(
    comments: pd.DataFrame, strict_moderation: bool = True
) -> pd.Series:
    """Return a mask for unmoderated statements.

    If `strict_moderation=True`, only keep comments explicitly moderated in (`moderated=1`).
    If `strict_moderation=False`, allow unmoderated (`moderated=0`).
    """
    threshold = 1 if strict_moderation else 0
    mask = comments["moderated"] >= threshold

    mask.name = "statement-in"  # feature-in
    return mask


def make_masked_vote_matrix(
    raw_vote_matrix: pd.DataFrame,
    statement_mask: pd.Series,
) -> pd.DataFrame:
    """Apply both participant and statement filters to create the final filtered matrix"""
    return _create_filtered_vote_matrix(
        raw_vote_matrix=raw_vote_matrix,
        statement_mask=statement_mask,
        filter_type="fill_zero",
    )


# def cluster_kmeans(matrix: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
#     kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(matrix)
#     return pd.Series(kmeans.labels_, index=matrix.index)

# PCA Subpipeline Nodes


def mean_impute_vote_matrix(filtered_vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Perform mean imputation on the filtered vote matrix using sklearn's SimpleImputer.
    Replace NaN values with the mean of each statement (column).
    """
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(filtered_vote_matrix)

    return pd.DataFrame(
        imputed_data,
        index=filtered_vote_matrix.index,
        columns=filtered_vote_matrix.columns,
    )


def reduce_with_pca(
    imputed_vote_matrix: pd.DataFrame, n_components: int = 2
) -> pd.DataFrame:
    """
    Apply PCA dimensionality reduction to the imputed vote matrix.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(imputed_vote_matrix)

    # Create column names based on number of components
    DIMENSION_COLS = ["x", "y", "z"]
    if n_components <= 3:
        column_names = DIMENSION_COLS[:n_components]
    else:
        column_names = [f"PC{i + 1}" for i in range(n_components)]

    return pd.DataFrame(
        components, index=imputed_vote_matrix.index, columns=pd.Index(column_names)
    )


def apply_sparsity_aware_scaler(
    participant_projections: pd.DataFrame, filtered_vote_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply SparsityAwareScaler to the projections, scaling based on sparse matrix.
    """
    scaler = SparsityAwareScaler(X_sparse=filtered_vote_matrix.values)
    scaled_data = scaler.fit_transform(participant_projections.values)

    return pd.DataFrame(
        scaled_data,
        index=participant_projections.index,
        columns=participant_projections.columns,
    )


def create_pca_scatter_plots(
    pca_components: pd.DataFrame,
    participant_meta: pd.DataFrame,
    flip_x: bool = False,
    flip_y: bool = False,
) -> tuple[go.Figure, go.Figure]:
    """
    Create a scatter plot of the PCA components for visualization.
    Supports 2D and 3D projections.

    Args:
        pca_components: DataFrame with PCA components
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1
    """
    pid_plot = _create_scatter_plot(
        data=pca_components,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Participant ID",
        color_values=pd.Series(pca_components.index.astype(int)),
        title="Scaled Participant Projections",
    )

    vote_count_plot = _create_scatter_plot(
        data=pca_components,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Vote Count",
        color_values=participant_meta.loc[:, "n-votes"],
        title="Scaled Participant Projections",
    )

    return pid_plot, vote_count_plot


def create_participants_meta(
    raw_vote_matrix: pd.DataFrame, raw_comments: pd.DataFrame
) -> pd.DataFrame:
    """
    Create participant metadata with voting statistics.

    Args:
        raw_vote_matrix: DataFrame with participants as rows and statements as columns
        raw_comments: DataFrame with comment data to count n-comments

    Returns:
        DataFrame with columns: n-comments, n-votes, n-agree, n-disagree, n-pass
    """
    participants_meta = pd.DataFrame(index=raw_vote_matrix.index)
    participants_meta.index.name = "participant_id"

    # Count total votes (non-NaN values)
    participants_meta["n-votes"] = raw_vote_matrix.count(axis=1)

    # Count agrees (1.0 values)
    participants_meta["n-agree"] = (raw_vote_matrix == 1.0).sum(axis=1)

    # Count disagrees (-1.0 values)
    participants_meta["n-disagree"] = (raw_vote_matrix == -1.0).sum(axis=1)

    # Count passes/neutral (0.0 values)
    participants_meta["n-pass"] = (raw_vote_matrix == 0.0).sum(axis=1)

    # Count comments authored by each participant
    # raw_comments has participant_id
    if "participant_id" in raw_comments.columns:
        comment_counts = raw_comments["participant_id"].value_counts()
        participants_meta["n-comments"] = (
            participants_meta.index.to_series()
            .map(comment_counts)
            .fillna(0)
            .astype(int)
        )
    else:
        # If no participant_id column, set to 0 for all participants
        participants_meta["n-comments"] = 0

    # Reorder columns for better readability
    participants_meta = participants_meta[
        ["n-comments", "n-votes", "n-agree", "n-disagree", "n-pass"]
    ]

    return participants_meta


def create_vote_heatmap(
    raw_vote_matrix: pd.DataFrame,
    participant_mask: pd.Series,
    statement_mask: pd.Series,
) -> go.Figure:
    """
    Create a plotly heatmap of the filtered vote matrix with custom color scheme:
    - Red (-1) for disagree
    - White (0) for neutral/pass
    - Green (+1) for agree
    - Pale yellow for missing votes (NaN)
    """
    # Create a copy of the matrix for display
    display_matrix = _create_filtered_vote_matrix(
        raw_vote_matrix=raw_vote_matrix,
        statement_mask=statement_mask,
        participant_mask=participant_mask,
        filter_type="drop",
    )

    display_matrix.sort_index(inplace=True, ascending=False)

    # Create custom colorscale (matching Polis website)
    # NaN's are handled below as background color.
    polisColorScale = [  # noqa: F841
        [0.0, "#e74c3c"],  # Red for -1
        [0.5, "#e6e6e6"],  # White for 0
        [1.0, "#2ecc71"],  # Green for +1
    ]

    # Same colorscale used in CompDem analysis notebooks
    # See: https://github.com/compdemocracy/analysis/blob/acc27dca89a37f8690e32dbd40aa8bc5ebfa851c/notebooks/jupyter/american-assembly-bg-analysis.heatmap.v0.6.ipynb
    analysisColorScale = px.colors.diverging.RdYlBu

    # Create the base heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix.values,
            x=[f"{col}" for col in display_matrix.columns],
            y=[f"{idx}" for idx in display_matrix.index],
            colorscale=analysisColorScale,
            zmin=-1,
            zmid=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate="Participant: %{y}<br>Statement: %{x}<br>Vote: %{z}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Vote",
                tickvals=[-1, 0, 1],
                ticktext=["Disagree", "Pass", "Agree"],
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title="Polis Vote Matrix Heatmap (Filtered)",
        xaxis_title="Statements",
        yaxis_title="Participants",
        width=max(800, len(display_matrix.columns) * 20),
        height=max(600, len(display_matrix.index) * 15),
        # xaxis=dict(tickangle=45),
        font=dict(size=10),
        plot_bgcolor="white",
    )

    return fig


@ensure_series("participant_mask")
def generate_polismath_json(
    raw_vote_matrix: pd.DataFrame,
    raw_comments: pd.DataFrame,
    participant_mask: pd.Series,
) -> dict:
    """
    Generate polismath JSON structure with the required keys.

    Args:
        raw_vote_matrix: The raw vote matrix
        raw_comments: Raw comments data

    Returns:
        Dictionary with polismath JSON structure
    """
    # Get basic counts
    n_participants = len(raw_vote_matrix.index)
    n_comments = len(raw_comments.index)

    # Remap columns into expected format and process.
    col_map = {
        "statement_id": "statement_id",
        "is_meta": "is_meta",
    }
    remapped_comments = (
        raw_comments.reset_index()
        .rename(columns=col_map, errors="raise")
        .to_dict(orient="records")
    )
    _, mod_in_statement_ids, mod_out_statement_ids, meta_statement_ids = (
        process_statements(statement_data=remapped_comments)
    )

    participants_in_conv = participant_mask.loc[participant_mask].index.to_list()

    # Calculate user vote counts (non-NaN votes per participant)
    user_vote_counts = {}
    for voter_id in raw_vote_matrix.index:
        vote_count = raw_vote_matrix.loc[
            voter_id
        ].count()  # count() excludes NaN values
        user_vote_counts[voter_id] = int(vote_count)

    # Create the polismath JSON structure
    polismath_data = {
        "comment-priorities": {},
        "user-vote-counts": user_vote_counts,
        "meta-tids": meta_statement_ids,
        "pca": {},
        "group-clusters": {},
        "n": n_participants,
        "consensus": {"agree": [], "disagree": []},
        "n-cmts": n_comments,
        "repness": {},
        "group-aware-consensus": {},
        "mod-in": mod_in_statement_ids,
        "votes-base": {},
        "base-clusters": {"id": [], "members": [], "x": [], "y": [], "count": []},
        "mod-out": mod_out_statement_ids,
        "group-votes": {},
        "lastModTimestamp": None,
        "in-conv": participants_in_conv,
        "tids": [],
        "lastVoteTimestamp": 0,
        "math_tick": 0,
    }

    return polismath_data
