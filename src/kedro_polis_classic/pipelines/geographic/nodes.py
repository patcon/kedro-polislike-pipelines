import pandas as pd
import json
import random
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Dict, List, Any
import plotly.graph_objects as go
from ..polis_legacy.utils import ensure_series


def filter_votes_for_islands(votes: pd.DataFrame) -> pd.DataFrame:
    """
    Filter votes to only include island-related statements with vote==1.

    Args:
        votes: Raw votes DataFrame

    Returns:
        Filtered votes DataFrame
    """
    # Map statement IDs to islands
    statement_to_island = {
        64: "Orcas Island",
        65: "Lopez Island",
        66: "San Juan Island",
        67: "Shaw Island",
    }
    island_statements = list(statement_to_island.keys())

    # Filter votes to relevant statements and vote==1
    filtered_votes = votes[votes["vote"] == 1].copy()
    filtered_votes = filtered_votes[
        filtered_votes["statement_id"].isin(island_statements)
    ]

    return filtered_votes


@ensure_series("participant_mask")
def aggregate_participant_islands(
    filtered_votes: pd.DataFrame, participant_mask: pd.Series
) -> Dict[int, List[str]]:
    """
    Aggregate participants to their preferred islands, including those with no island votes.

    Args:
        filtered_votes: Filtered votes DataFrame
        participant_mask: Boolean mask indicating which participants are included

    Returns:
        Dictionary mapping participant_id to list of island names (or ["Other"] for no votes)
    """
    statement_to_island = {
        64: "Orcas Island",
        65: "Lopez Island",
        66: "San Juan Island",
        67: "Shaw Island",
    }

    # Get participants who voted for islands
    participant_islands = (
        filtered_votes.groupby("participant_id")["statement_id"]
        .apply(lambda x: [statement_to_island[i] for i in x])
        .to_dict()
    )

    # Add participants who didn't vote for any islands to "Other"
    # Only include participants who are in the participant_mask (meet minimum vote threshold)
    included_participant_ids = participant_mask.index[participant_mask].tolist()
    for pid in included_participant_ids:
        if pid not in participant_islands:
            participant_islands[pid] = ["Other"]

    return participant_islands


def load_island_geojson(geojson_path: str) -> Dict[str, Any]:
    """
    Load island GeoJSON data from file.

    Args:
        geojson_path: Path to the GeoJSON file

    Returns:
        GeoJSON data as dictionary
    """
    with open(geojson_path, "r") as f:
        islands_geojson = json.load(f)

    return islands_geojson


def create_island_shapes(islands_geojson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map island names to shapely polygon objects.

    Args:
        islands_geojson: GeoJSON data

    Returns:
        Dictionary mapping island names to shapely polygons
    """
    island_shapes = {}

    for feature in islands_geojson["features"]:
        name = feature["properties"].get("name")
        geom = feature["geometry"]

        if geom["type"] == "Polygon":
            island_shapes[name] = Polygon(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            island_shapes[name] = MultiPolygon(
                [Polygon(p[0]) for p in geom["coordinates"]]
            )

    return island_shapes


def random_point_in_polygon(polygon: Polygon) -> List[float]:
    """
    Generate a random point within a polygon.

    Args:
        polygon: Shapely polygon object

    Returns:
        List containing [x, y] coordinates
    """
    minx, miny, maxx, maxy = polygon.bounds

    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            # Optional: tiny jitter to make it clear data is synthetic
            return [p.x, p.y]


def assign_participant_coordinates(
    participant_islands: Dict[int, List[str]], island_shapes: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign random coordinates to participants based on their island preferences.
    Participants with no island votes are placed in an "Other" circular region.

    Args:
        participant_islands: Dictionary mapping participant_id to list of island names
        island_shapes: Dictionary mapping island names to shapely polygons

    Returns:
        List of GeoJSON features for participants
    """
    participant_features = []

    # Island size ranking (largest to smallest)
    # San Juan > Orcas > Lopez > Shaw
    island_size_priority = {
        "San Juan Island": 4,  # Largest
        "Orcas Island": 3,
        "Lopez Island": 2,
        "Shaw Island": 1,  # Smallest (highest priority)
        "Other": 0,  # Special case
    }

    for pid, islands in participant_islands.items():
        # Choose the smallest island if multiple votes (favor niche/rare choices)
        if len(islands) > 1:
            # Sort by size priority (smallest first)
            islands_sorted = sorted(
                islands, key=lambda x: island_size_priority.get(x, 0)
            )
            island_name = islands_sorted[0]  # Pick the smallest
        else:
            island_name = islands[0]

        if island_name == "Other":
            # Create "Other" region - circular area to the right of the islands
            # San Juan Islands are roughly around -123.0 longitude, so place "Other" at -122.5
            center_x, center_y = -122.5, 48.5
            radius = 0.1  # degrees

            # Generate random point in circle
            import math

            angle = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.uniform(0, 1))
            coords = [center_x + r * math.cos(angle), center_y + r * math.sin(angle)]

            feature = {
                "type": "Feature",
                "properties": {"participant_id": pid, "island": island_name},
                "geometry": {"type": "Point", "coordinates": coords},
            }
            participant_features.append(feature)
        else:
            poly = island_shapes.get(island_name)
            if poly:
                coords = random_point_in_polygon(poly)
                feature = {
                    "type": "Feature",
                    "properties": {"participant_id": pid, "island": island_name},
                    "geometry": {"type": "Point", "coordinates": coords},
                }
                participant_features.append(feature)

    return participant_features


def create_participant_geojson(
    participant_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create GeoJSON FeatureCollection from participant features.

    Args:
        participant_features: List of GeoJSON features

    Returns:
        GeoJSON FeatureCollection
    """
    participant_geojson = {
        "type": "FeatureCollection",
        "features": participant_features,
    }

    print(f"Generated {len(participant_features)} participant points.")
    return participant_geojson


def save_participant_geojson(
    participant_geojson: Dict[str, Any], output_path: str
) -> str:
    """
    Save participant GeoJSON to file.

    Args:
        participant_geojson: GeoJSON FeatureCollection
        output_path: Path where to save the file

    Returns:
        Path where the file was saved
    """
    with open(output_path, "w") as f:
        json.dump(participant_geojson, f, indent=2)

    print(f"Participant GeoJSON saved to: {output_path}")
    return output_path


def create_geographic_scatter_plot(participant_geojson: Dict[str, Any]) -> go.Figure:
    """
    Create a plotly scatter plot from geographic coordinates using the existing scatter plot helper.

    Args:
        participant_geojson: GeoJSON FeatureCollection with participant points

    Returns:
        Plotly figure showing participants on a geographic scatter plot
    """
    # Import the existing scatter plot helper from experimental nodes
    from ..experimental.nodes import _create_scatter_plot

    # Extract coordinates and metadata from GeoJSON
    coords_data = []
    participant_ids = []
    islands = []

    for feature in participant_geojson["features"]:
        coords = feature["geometry"]["coordinates"]
        coords_data.append([coords[0], coords[1]])  # [longitude, latitude]
        participant_ids.append(feature["properties"]["participant_id"])
        islands.append(feature["properties"]["island"])

    # Create DataFrame with geographic coordinates and proper participant ID index
    import pandas as pd

    data = pd.DataFrame(coords_data, columns=["longitude", "latitude"])
    # Ensure participant IDs are set as the index for proper hover tooltips
    data.index = pd.Index(participant_ids, name="participant_id")

    # Convert islands to pandas Series for color values with matching index
    island_labels = pd.Series(islands, index=data.index, name="Island")

    # Sort unique island labels for consistent coloring (put "Other" last)
    unique_islands = sorted(
        [island for island in island_labels.unique() if island != "Other"]
    )
    if "Other" in island_labels.unique():
        unique_islands.append("Other")
    category_orders = {"Island": unique_islands}

    # Use the existing scatter plot helper
    scatter_plot = _create_scatter_plot(
        data=data,
        flip_x=False,
        flip_y=False,
        colorbar_title="Island",
        color_values=island_labels,
        title="Geographic Projection: Participant Locations by Island Preference",
        use_categorical_colors=True,
        category_orders=category_orders,
    )

    print(f"Created geographic scatter plot with {len(coords_data)} participant points")
    return scatter_plot
