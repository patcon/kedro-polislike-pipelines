from ..builder import build_pipeline_from_params
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from kedro_polis_classic.datasets.polis_api import PolisAPIDataset
from ..polis_legacy.utils import ensure_series
import logging

logger = logging.getLogger(__name__)


def run_component_node(X, params, step_name, **catalog_inputs):
    """
    Runs a single pipeline component.
    X: input features
    params: full nested pipeline parameters dict
    step_name: which step to build (imputer/reducer/scaler/clusterer)
    **catalog_inputs: catalog items for parameters that start with 'input:'
    """
    # copy to avoid mutating params
    step_config = params.copy()

    # Process input: parameters by replacing them with actual catalog data
    processed_config = _process_input_parameters(step_config, catalog_inputs)

    pipeline = build_pipeline_from_params({step_name: processed_config})

    # For clusterer, use fit_predict to get labels instead of fit_transform
    if step_name == "clusterer":
        return pipeline.fit_predict(X)
    else:
        return pipeline.fit_transform(X)


def _process_input_parameters(config: dict, catalog_inputs: dict) -> dict:
    """
    Process configuration parameters, replacing 'input:' values with catalog data.

    Args:
        config: Configuration dictionary that may contain 'input:' values
        catalog_inputs: Dictionary mapping catalog item names to their data

    Returns:
        Processed configuration with 'input:' values replaced by catalog data
    """
    processed_config = {}

    for key, value in config.items():
        if isinstance(value, str) and value.startswith("input:"):
            # Extract the catalog item name (everything after "input:")
            catalog_item_name = value[6:]  # Remove "input:" prefix

            if catalog_item_name in catalog_inputs:
                processed_config[key] = catalog_inputs[catalog_item_name]
            else:
                raise ValueError(
                    f"Catalog item '{catalog_item_name}' not found in inputs for parameter '{key}'"
                )
        else:
            processed_config[key] = value

    return processed_config


# Minimal data loader nodes from original polis pipeline


def load_polis_data(
    base_url: str | None = None,
    polis_url: str | None = None,
    import_dir: str | None = None,
):
    """Load raw data from Polis API or local directory"""
    dataset = PolisAPIDataset(
        base_url=base_url, polis_url=polis_url, import_dir=import_dir
    )
    return dataset.load()


def split_raw_data(raw_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw data into votes and comments"""
    return raw_data["votes"], raw_data["comments"]


def dedup_votes(raw_votes: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate votes, keeping the most recent"""
    # Sort so newest votes are last
    votes_sorted = raw_votes.sort_values("modified")

    # Drop duplicates, keeping the most recent
    deduped_votes = votes_sorted.drop_duplicates(
        subset=["participant_id", "statement_id"], keep="last"
    )

    return deduped_votes


def make_raw_vote_matrix(deduped_votes: pd.DataFrame) -> pd.DataFrame:
    """Create vote matrix from deduplicated votes"""
    matrix = deduped_votes.pivot(
        index="participant_id", columns="statement_id", values="vote"
    )

    # Convert vote values to integers (handles NaN values properly)
    matrix = matrix.astype("Int64")

    return matrix


# Preprocessing nodes from polis pipeline


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


def make_participant_mask(matrix: pd.DataFrame, min_votes: int = 7) -> pd.Series:
    """Create a mask for participants who meet the minimum vote threshold"""
    mask = matrix.count(axis="columns") >= min_votes

    mask.index.name = "participant_id"
    mask.name = "participant-in"
    return mask


def make_statement_mask(
    comments: pd.DataFrame,
    strict_moderation: bool = False,
    mask_out_is_meta: bool = True,
) -> pd.Series:
    """Return a mask for unmoderated statements.

    If `strict_moderation=True`, only keep comments explicitly moderated in (`moderated=1`).
    If `strict_moderation=False`, extend statements to include unmoderated (`moderated=0`).

    By default, upstream Polis has this behavior:
    - acts like strict moderation is disabled all the time for filtering the vote matrix,
      so only explicitly moderated statements are ever masked out (This is perhaps an oversight.)
    - `is_meta` statements are masked out.
    """
    in_threshold = 1 if strict_moderation else 0

    if mask_out_is_meta:
        mask = (comments["moderated"] >= in_threshold) & (comments["is_meta"] == False)
    else:
        mask = comments["moderated"] >= in_threshold

    mask.name = "statement-in"
    return mask


def make_masked_vote_matrix(
    raw_vote_matrix: pd.DataFrame,
    statement_mask: pd.Series,
) -> pd.DataFrame:
    """Apply statement filter to create the masked vote matrix using fill_zero strategy"""
    return _apply_statement_filter(
        matrix=raw_vote_matrix,
        statement_mask=statement_mask,
        filter_type="fill_zero",
    )


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
    category_orders: dict | None = None,
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
    color_continuous_scale = "YlOrRd"
    color_discrete_scale = px.colors.qualitative.Set1

    # Add participant labels for hover
    plot_data["Participant"] = [f"Participant {idx}" for idx in plot_data.index]

    # Check for 2D or 3D plot based on column count
    if len(data.columns) == 3:
        # 3D scatter plot
        z_col = plot_data.columns[2]

        if use_categorical_colors:
            # Use discrete colors for categorical data
            kwargs = {
                "data_frame": plot_data,
                "x": x_col,
                "y": y_col,
                "z": z_col,
                "color": colorbar_title,
                "hover_name": "Participant",
                "title": title,
                "color_discrete_sequence": color_discrete_scale,
            }
            if category_orders is not None:
                kwargs["category_orders"] = category_orders

            fig = px.scatter_3d(**kwargs)
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
                color_continuous_scale=color_continuous_scale,
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
            kwargs = {
                "data_frame": plot_data,
                "x": x_col,
                "y": y_col,
                "color": colorbar_title,
                "hover_name": "Participant",
                "title": title,
                "color_discrete_sequence": color_discrete_scale,
            }
            if category_orders is not None:
                kwargs["category_orders"] = category_orders

            fig = px.scatter(**kwargs)
        else:
            # Use continuous color scale
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=colorbar_title,
                hover_name="Participant",
                title=title,
                color_continuous_scale=color_continuous_scale,
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


@ensure_series("participant_mask")
def create_scatter_plot(
    filter_output,  # Can be numpy array or DataFrame - filtered data from SampleMaskFilter
    clusterer_output,  # Cluster labels
    participant_mask: pd.Series,  # Mask indicating which participants are included
    flip_x: bool = False,
    flip_y: bool = False,
) -> go.Figure:
    """
    Create a scatter plot of the output for visualization.
    Supports 2D and 3D projections.
    Adapted from polis pipeline create_pca_scatter_plots node.

    Args:
        filter_output: Numpy array or DataFrame with filtered components from the experimental pipeline
        clusterer_output: Cluster labels for coloring the points
        participant_mask: Boolean mask indicating which participants are included in the filtered data
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1

    Returns:
        Plotly figure showing the scatter plot
    """
    import numpy as np

    # Get the participant IDs that are included (where mask is True)
    included_participant_ids = participant_mask.index[participant_mask]

    # Convert numpy array to DataFrame if needed
    if isinstance(filter_output, np.ndarray):
        # Create generic column names based on dimensions
        n_components = filter_output.shape[1] if len(filter_output.shape) > 1 else 1
        if n_components <= 3:
            column_names = ["x", "y", "z"][:n_components]
        else:
            column_names = [f"PC{i + 1}" for i in range(n_components)]

        # Create DataFrame with actual participant IDs as index
        data = pd.DataFrame(
            filter_output,
            index=included_participant_ids,  # type: ignore
            columns=pd.Index(column_names),
        )
    else:
        # Already a DataFrame, but ensure it has the correct index
        data = filter_output.copy()
        data.index = included_participant_ids

    # Convert cluster labels to pandas Series of strings for categorical coloring
    # Make sure the cluster labels have the same index as the data DataFrame
    if isinstance(clusterer_output, np.ndarray):
        cluster_labels = pd.Series(clusterer_output.flatten(), index=data.index)
    else:
        cluster_labels = pd.Series(clusterer_output, index=data.index)

    # Sort unique cluster labels numerically to ensure proper legend ordering
    unique_labels = sorted(cluster_labels.unique())

    # Always include -1 (noise/outliers) in category order for consistent coloring
    if -1 not in unique_labels:
        unique_labels = [-1] + unique_labels

    cluster_labels = cluster_labels.astype(str)

    # Create category orders for plotly express
    category_orders = {"Cluster": [str(label) for label in unique_labels]}

    # Create scatter plot colored by cluster labels
    scatter_plot = _create_scatter_plot(
        data=data,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Cluster",
        color_values=cluster_labels,
        title="Experimental Pipeline: Participant Projections (Colored by Cluster)",
        use_categorical_colors=True,
        category_orders=category_orders,
    )

    return scatter_plot


@ensure_series("participant_mask")
def create_scatter_plot_by_participant_id(
    filter_output,  # Can be numpy array or DataFrame - filtered data from SampleMaskFilter
    participant_mask: pd.Series,  # Mask indicating which participants are included
    flip_x: bool = False,
    flip_y: bool = False,
) -> go.Figure:
    """
    Create a scatter plot colored by participant ID using viridis color scheme.
    Supports 2D and 3D projections.

    Args:
        filter_output: Numpy array or DataFrame with filtered components from the experimental pipeline
        raw_vote_matrix: Raw vote matrix (not used in this function but kept for consistent interface)
        participant_mask: Boolean mask indicating which participants are included in the filtered data
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1

    Returns:
        Plotly figure showing the scatter plot colored by participant ID
    """
    import numpy as np

    # Get the participant IDs that are included (where mask is True)
    included_participant_ids = participant_mask.index[participant_mask]

    # Convert numpy array to DataFrame if needed
    if isinstance(filter_output, np.ndarray):
        # Create generic column names based on dimensions
        n_components = filter_output.shape[1] if len(filter_output.shape) > 1 else 1
        if n_components <= 3:
            column_names = ["x", "y", "z"][:n_components]
        else:
            column_names = [f"PC{i + 1}" for i in range(n_components)]

        # Create DataFrame with actual participant IDs as index
        data = pd.DataFrame(
            filter_output,
            index=included_participant_ids,  # type: ignore
            columns=pd.Index(column_names),
        )
    else:
        # Already a DataFrame, but ensure it has the correct index
        data = filter_output.copy()
        data.index = included_participant_ids

    # Get participant IDs as numeric values for continuous color scale
    participant_ids = pd.Series(data.index, index=data.index)

    # Create scatter plot colored by participant ID
    scatter_plot = _create_scatter_plot(
        data=data,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Participant ID",
        color_values=participant_ids,
        title="Experimental Pipeline: Participant Projections (Colored by Participant ID)",
        use_categorical_colors=False,  # Use continuous viridis color scale
    )

    return scatter_plot


@ensure_series("participant_mask")
def create_scatter_plot_by_vote_proportions(
    filter_output,  # Can be numpy array or DataFrame - filtered data from SampleMaskFilter
    raw_vote_matrix: pd.DataFrame,  # Raw vote matrix to calculate vote counts from
    participant_mask: pd.Series,  # Mask indicating which participants are included
    flip_x: bool = False,
    flip_y: bool = False,
) -> go.Figure:
    """
    Create a scatter plot colored by total number of votes cast by each participant.
    Supports 2D and 3D projections.

    Args:
        filter_output: Numpy array or DataFrame with filtered components from the experimental pipeline
        raw_vote_matrix: Raw vote matrix to calculate vote counts from
        participant_mask: Boolean mask indicating which participants are included in the filtered data
        flip_x: If True, flip the x-axis by multiplying by -1
        flip_y: If True, flip the y-axis by multiplying by -1

    Returns:
        Plotly figure showing the scatter plot colored by total vote counts
    """
    import numpy as np

    # Get the participant IDs that are included (where mask is True)
    included_participant_ids = participant_mask.index[participant_mask]

    # Convert numpy array to DataFrame if needed
    if isinstance(filter_output, np.ndarray):
        # Create generic column names based on dimensions
        n_components = filter_output.shape[1] if len(filter_output.shape) > 1 else 1
        if n_components <= 3:
            column_names = ["x", "y", "z"][:n_components]
        else:
            column_names = [f"PC{i + 1}" for i in range(n_components)]

        # Create DataFrame with actual participant IDs as index
        data = pd.DataFrame(
            filter_output,
            index=included_participant_ids,  # type: ignore
            columns=pd.Index(column_names),
        )
    else:
        # Already a DataFrame, but ensure it has the correct index
        data = filter_output.copy()
        data.index = included_participant_ids

    # Calculate total number of votes cast by each included participant
    # Vote values: 1 = agree, -1 = disagree, 0 = pass, NaN = no vote
    # Count all non-NaN values (any vote cast) for the included participants only
    total_votes_cast = raw_vote_matrix.loc[included_participant_ids].count(axis=1)

    # Create scatter plot colored by total vote count
    scatter_plot = _create_scatter_plot(
        data=data,
        flip_x=flip_x,
        flip_y=flip_y,
        colorbar_title="Total Votes Cast",
        color_values=total_votes_cast,
        title="Experimental Pipeline: Participant Projections (Colored by Total Votes Cast)",
        use_categorical_colors=False,  # Use continuous viridis color scale
    )

    return scatter_plot


def save_scatter_plot_image(
    scatter_plot: go.Figure,
    pipeline_name: str,
) -> str:
    """
    Save scatter plot as an image file with ISO timestamp prefix.

    Args:
        scatter_plot: Plotly figure to save
        pipeline_name: Name of the pipeline for the filename

    Returns:
        The filepath where the image was saved
    """
    from datetime import datetime
    import os

    # Create ISO timestamp prefix (YYYY-MM-DD-HH-MM format)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create filename with timestamp prefix
    filename = f"{timestamp}_{pipeline_name}_scatter_plot.png"

    # Create the directory path
    output_dir = f"data/{pipeline_name}/08_reporting"
    os.makedirs(output_dir, exist_ok=True)

    # Full filepath
    filepath = os.path.join(output_dir, filename)

    # Save the plot as PNG image
    scatter_plot.write_image(filepath, width=800, height=600, scale=2)

    logger.info(f"Scatter plot image saved to: {filepath}")
    return filepath


@ensure_series("participant_mask")
def create_votes_dataframe(
    raw_vote_matrix: pd.DataFrame,
    participant_mask: pd.Series,
) -> pd.DataFrame:
    """
    Create votes dataframe in Red-Dwarf specification format for SQLite storage.

    Args:
        raw_vote_matrix: Raw vote matrix with participant_id as index and comment_id as columns
        participant_mask: Boolean mask indicating which participants to include

    Returns:
        DataFrame with columns: participant_id, comment_id, vote
    """
    # Get the participant IDs that are included (where mask is True)
    included_participant_ids = participant_mask.index[participant_mask]
    df = raw_vote_matrix.loc[included_participant_ids].copy()

    # Reset index and rename the index column to participant_id
    # The index column name will be the original index name (e.g., "participant_id")
    df = df.reset_index()
    index_col_name = df.columns[0]  # Get the actual name of the index column
    df = df.rename(columns={index_col_name: "participant_id"})

    # Melt to long format
    long_df = df.melt(
        id_vars="participant_id", var_name="comment_id", value_name="vote"
    )
    long_df = long_df.dropna(subset=["vote"]).astype(
        {"participant_id": str, "comment_id": str, "vote": int}
    )

    logger.info(f"Votes dataframe created with {len(long_df)} vote records")
    return long_df


@ensure_series("participant_mask")
def save_projections_json(
    filter_output,  # Filtered data from the experimental pipeline
    participant_mask: pd.Series,
) -> list:
    """
    Save dimensionality-reduced projections as JSON according to Red-Dwarf specification.
    Returns data in format for Kedro JSON dataset.

    Args:
        filter_output: Numpy array or DataFrame with filtered components from the experimental pipeline
        participant_mask: Boolean mask indicating which participants are included

    Returns:
        List in format [[participant_id, [x, y]], ...] for Kedro JSON dataset
    """
    import numpy as np

    # Get the participant IDs that are included (where mask is True)
    included_participant_ids = list(participant_mask.index[participant_mask])  # type: ignore

    # Convert numpy array to proper format if needed
    if isinstance(filter_output, np.ndarray):
        X_clustered = filter_output
    else:
        # If it's a DataFrame, get the values
        X_clustered = filter_output.values

    # Ensure we have 2D coordinates (take first 2 dimensions if more)
    if X_clustered.shape[1] > 2:
        X_clustered = X_clustered[:, :2]

    # Create the format: [[participant_id, [x, y]], ...]
    X_with_ids = []
    for i, participant_id in enumerate(included_participant_ids):
        coords = X_clustered[i].tolist()
        X_with_ids.append([str(participant_id), coords])

    logger.info(f"Projections data prepared with {len(X_with_ids)} participants")
    return X_with_ids


def save_statements_json(raw_comments: pd.DataFrame) -> list:
    """
    Save statements (comments) as JSON - just dump the raw comments data.
    Returns data in format for Kedro JSON dataset.

    Args:
        raw_comments: Raw comments DataFrame

    Returns:
        Dictionary representation of the raw comments DataFrame
    """
    # Convert DataFrame to dictionary format that preserves all original data
    # Replace NaN values with None (which becomes null in JSON)
    statements_dict = (
        raw_comments.reset_index()
        .replace({pd.NA: None, float("nan"): None})
        .to_dict(orient="records")
    )

    logger.info(f"Statements data prepared with {len(statements_dict)} comments")
    return statements_dict


def save_meta_json(
    polis_url: str | None = None, reducer_params: dict | None = None
) -> dict:
    """
    Save dataset metadata as JSON according to Red-Dwarf specification.
    Returns data in format for Kedro JSON dataset.

    Args:
        polis_url: Polis URL (optional)
        reducer_params: Parameters for the reducer step to extract n_neighbors (optional)

    Returns:
        Dictionary with metadata for Kedro JSON dataset
    """
    # Extract polis_id from URL if provided
    polis_id = None
    if polis_url:
        from kedro_polis_classic.datasets.polis_api import _parse_polis_url

        try:
            _, polis_id = _parse_polis_url(polis_url)
        except ValueError:
            polis_id = None

    # Create metadata
    meta = {
        "about_url": None,
        "conversation_url": f"https://pol.is/{polis_id}" if polis_id else None,
        "report_url": f"https://pol.is/report/{polis_id}" if polis_id else None,
        "last_vote": None,  # Could be extracted from vote timestamps if needed
        "n_neighbors": reducer_params.get("n_neighbors", 10) if reducer_params else 10,
    }

    logger.info(
        f"Metadata prepared for polis_url: {polis_url}, extracted polis_id: {polis_id}"
    )
    return meta
