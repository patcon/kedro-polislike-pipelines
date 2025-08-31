from ..builder import build_pipeline_from_params
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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


def _create_scatter_plot(
    data: pd.DataFrame,
    flip_x: bool,
    flip_y: bool,
    colorbar_title: str,
    color_values: pd.Series,
    title: str,
    use_categorical_colors: bool = False,
) -> go.Figure:
    """
    Simplified helper function to create a 2D or 3D scatter plot using plotly express.

    Args:
        data: DataFrame with data for plotting
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1
        colorbar_title: Title for the colorbar
        color_values: Data for the marker color
        title: Title for the plot
        use_categorical_colors: If True, use categorical color scale (good for clusters)

    Returns:
        A Plotly figure (2D or 3D scatter plot)
    """
    import numpy as np

    # Create a copy of the data to avoid modifying the original
    plot_data = data.copy()

    # Get column names
    x_col = plot_data.columns[0]
    y_col = plot_data.columns[1]

    # Apply flipping if requested
    if flip_x:
        plot_data[x_col] = plot_data[x_col] * -1
    if flip_y:
        plot_data[y_col] = plot_data[y_col] * -1

    # Add color values to the dataframe for plotly express
    plot_data[colorbar_title] = color_values

    # Add participant labels for hover
    plot_data["Participant"] = [f"Participant {idx}" for idx in range(len(plot_data))]

    # Check for 2D or 3D plot based on column count
    if len(data.columns) == 3:
        # 3D scatter plot
        z_col = plot_data.columns[2]

        if use_categorical_colors:
            # Use discrete colors for categorical data
            fig = px.scatter_3d(
                plot_data,
                x=x_col,
                y=y_col,
                z=z_col,
                color=colorbar_title,
                hover_name="Participant",
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
        else:
            # Use continuous color scale
            fig = px.scatter_3d(
                plot_data,
                x=x_col,
                y=y_col,
                z=z_col,
                color=colorbar_title,
                hover_name="Participant",
                title=title,
                color_continuous_scale="Viridis",
            )

        # Update axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{str(x_col).upper()} Component",
                yaxis_title=f"{str(y_col).upper()} Component",
                zaxis_title=f"{str(z_col).upper()} Component",
            ),
            width=800,
            height=600,
        )

    elif len(data.columns) == 2:
        # 2D scatter plot
        if use_categorical_colors:
            # Use discrete colors for categorical data
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=colorbar_title,
                hover_name="Participant",
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
        else:
            # Use continuous color scale
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=colorbar_title,
                hover_name="Participant",
                title=title,
                color_continuous_scale="Viridis",
            )

        # Update axis labels and layout
        fig.update_layout(
            xaxis_title=f"{str(x_col).upper()} Component",
            yaxis_title=f"{str(y_col).upper()} Component",
            width=800,
            height=600,
            plot_bgcolor="white",
        )

    else:
        raise ValueError("Data must have exactly 2 or 3 columns for 2D or 3D plots.")

    # Update marker size for better visibility
    fig.update_traces(marker=dict(size=8 if len(data.columns) == 2 else 6))

    return fig


def create_scaler_scatter_plot(
    scaler_output,  # Can be numpy array or DataFrame
    clusterer_output,  # Cluster labels
    flip_x: bool = False,
    flip_y: bool = False,
) -> go.Figure:
    """
    Create a scatter plot of the scaler output for visualization.
    Supports 2D and 3D projections.
    Adapted from polis pipeline create_pca_scatter_plots node.

    Args:
        scaler_output: Numpy array or DataFrame with scaled components from the experimental pipeline
        clusterer_output: Cluster labels for coloring the points
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1

    Returns:
        Plotly figure showing the scatter plot
    """
    import numpy as np

    # Convert numpy array to DataFrame if needed
    if isinstance(scaler_output, np.ndarray):
        # Create generic column names based on dimensions
        n_components = scaler_output.shape[1] if len(scaler_output.shape) > 1 else 1
        if n_components <= 3:
            column_names = ["x", "y", "z"][:n_components]
        else:
            column_names = [f"PC{i + 1}" for i in range(n_components)]

        # Create DataFrame with generic participant IDs
        data = pd.DataFrame(
            scaler_output,
            index=range(len(scaler_output)),
            columns=pd.Index(column_names),
        )
    else:
        # Already a DataFrame
        data = scaler_output

    # Convert cluster labels to pandas Series of strings for categorical coloring
    if isinstance(clusterer_output, np.ndarray):
        cluster_labels = pd.Series(clusterer_output.flatten()).astype(str)
    else:
        cluster_labels = pd.Series(clusterer_output).astype(str)

    # Create scatter plot colored by cluster labels
    scatter_plot = _create_scatter_plot(
        data=data,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Cluster",
        color_values=cluster_labels,
        title="Experimental Pipeline: Scaled Participant Projections (Colored by Cluster)",
        use_categorical_colors=True,
    )

    return scatter_plot
