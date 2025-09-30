import os
import re
import pandas as pd
from kedro.io import AbstractDataset
from urllib.parse import urlparse


def _parse_polis_url(polis_url: str) -> tuple[str, str]:
    """
    Parse a polis URL to extract base_url and polis_id.

    Args:
        polis_url: URL in format "https://polis.example.com/{polis_convo_id}"
                  or "https://polis.example.com/report/{polis_report_id}"

    Returns:
        tuple: (base_url, polis_id)

    Raises:
        ValueError: If URL format is invalid
    """
    parsed = urlparse(polis_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Remove leading/trailing slashes and split path
    path_parts = parsed.path.strip("/").split("/")

    if not path_parts or not path_parts[-1]:
        raise ValueError(
            f"Invalid polis URL format: {polis_url}. Expected format: 'https://polis.example.com/{{polis_id}}' or 'https://polis.example.com/report/{{polis_id}}'"
        )

    # Check if it's a report URL
    if len(path_parts) >= 2 and path_parts[-2] == "report":
        polis_id = path_parts[-1]
        # Ensure report IDs start with 'r'
        if not polis_id.startswith("r"):
            polis_id = f"r{polis_id}"
    else:
        # Direct conversation URL
        polis_id = path_parts[-1]
        # Ensure conversation IDs don't start with 'r'
        if polis_id.startswith("r"):
            raise ValueError(
                f"Conversation URL should not contain report ID: {polis_url}"
            )

    return base_url, polis_id


class PolisAPIDataset(AbstractDataset):
    def __init__(
        self,
        polis_id: str | None = None,
        base_url: str | None = None,
        polis_url: str | None = None,
        repair_is_meta_column: bool = True,
        import_dir: str | None = None,
    ):
        # Handle polis_url parameter - it takes precedence over polis_id and base_url
        if polis_url:
            if polis_id or base_url:
                raise ValueError(
                    "Cannot specify both polis_url and polis_id/base_url parameters"
                )
            self.base_url, self.polis_id = _parse_polis_url(polis_url)
        else:
            self.polis_id = polis_id
            self.base_url = base_url if base_url else "https://pol.is"

        self.polis_url = polis_url
        self.repair_is_meta_column = repair_is_meta_column
        self.import_dir = import_dir

        # Initialize report_id and conversation_id
        self.report_id = None
        self.conversation_id = None

        # If import_dir is provided, prioritize it over polis_id
        if import_dir:
            # We'll determine conversation_id from the loaded data
            pass
        elif self.polis_id:
            # Determine if polis_id is a report_id or conversation_id
            if self.polis_id.startswith("r"):
                self.report_id = self.polis_id
            elif self.polis_id[0].isdigit():
                self.conversation_id = self.polis_id
            else:
                raise ValueError(
                    "polis_id must start with 'r' (for report_id) or a digit (for conversation_id)"
                )
        else:
            raise ValueError(
                "Either polis_id, polis_url, or import_dir must be provided"
            )

    def load(self) -> dict[str, pd.DataFrame]:
        """Load data using the appropriate method based on provided parameters."""
        if self.import_dir:
            return self.load_from_directory()
        elif self.report_id:
            return self.load_from_csv()
        elif self.conversation_id:
            return self.load_from_api()
        else:
            raise ValueError("No valid parameters provided for loading data")

    def load_from_csv(self) -> dict[str, pd.DataFrame]:
        """Load data from CSV endpoints using report_id with reddwarf Loader."""
        if not self.report_id:
            raise ValueError("report_id is required for loading from CSV")

        from reddwarf.data_loader import Loader
        import ssl
        import requests

        # Try loading with current base_url, fallback to HTTP if HTTPS fails
        base_url = self.base_url

        try:
            # Use Loader with csv_export data source
            loader = Loader(
                polis_id=self.report_id,
                data_source="csv_export",
                polis_instance_url=base_url,
            )

            # Convert the list data to DataFrames
            comments = pd.DataFrame(loader.comments_data)
            votes = pd.DataFrame(loader.votes_data)

        except (ssl.SSLError, requests.exceptions.SSLError) as e:
            if base_url.startswith("https://"):
                print(f"SSL error with HTTPS, trying HTTP: {e}")
                http_base_url = base_url.replace("https://", "http://")

                loader = Loader(
                    polis_id=self.report_id,
                    data_source="csv_export",
                    polis_instance_url=http_base_url,
                )

                # Convert the list data to DataFrames
                comments = pd.DataFrame(loader.comments_data)
                votes = pd.DataFrame(loader.votes_data)
            else:
                raise

        # No column renaming - use original column names
        return {"comments": comments, "votes": votes}

    def load_from_api(self) -> dict[str, pd.DataFrame]:
        """Load data using the reddwarf data loader with conversation_id."""
        if not self.conversation_id:
            raise ValueError("conversation_id is required for loading from API")

        from reddwarf.data_loader import Loader
        import ssl
        import requests

        # Try loading with current base_url, fallback to HTTP if HTTPS fails
        base_url = self.base_url

        try:
            loader = Loader(
                conversation_id=self.conversation_id, polis_instance_url=base_url
            )

            # Convert the list data to DataFrames
            comments = pd.DataFrame(loader.comments_data)
            votes = pd.DataFrame(loader.votes_data)

        except (ssl.SSLError, requests.exceptions.SSLError) as e:
            if base_url.startswith("https://"):
                print(f"SSL error with HTTPS, trying HTTP: {e}")
                http_base_url = base_url.replace("https://", "http://")

                loader = Loader(
                    conversation_id=self.conversation_id,
                    polis_instance_url=http_base_url,
                )

                # Convert the list data to DataFrames
                comments = pd.DataFrame(loader.comments_data)
                votes = pd.DataFrame(loader.votes_data)
            else:
                raise

        # No column renaming - use original column names
        return {"comments": comments, "votes": votes}

    def load_from_directory(self) -> dict[str, pd.DataFrame]:
        """Load data from local JSON files in the specified directory."""
        if not self.import_dir:
            raise ValueError("import_dir is required for loading from directory")

        from reddwarf.data_loader import Loader

        # Construct file paths
        filepaths = [
            os.path.join(self.import_dir, "comments.json"),
            os.path.join(self.import_dir, "votes.json"),
            os.path.join(self.import_dir, "math-pca2.json"),
            os.path.join(self.import_dir, "conversation.json"),
        ]

        # Use Loader with filepaths
        loader = Loader(filepaths=filepaths)

        # Set conversation_id from loaded conversation data
        conversation_id = loader.conversation_data["conversation_id"]
        loader.conversation_id = conversation_id

        # Update our instance variables to make conversation_id available in catalog
        self.conversation_id = conversation_id
        self.polis_id = conversation_id

        # Store the polis_id as an environment variable for potential use in catalog resolvers
        os.environ["KEDRO_POLIS_ID"] = conversation_id

        # Convert the list data to DataFrames
        comments = pd.DataFrame(loader.comments_data)
        votes = pd.DataFrame(loader.votes_data)

        return {"comments": comments, "votes": votes}

    def save(self, data: dict[str, pd.DataFrame]) -> None:
        raise NotImplementedError("Saving to Polis API is not supported.")

    def _describe(self) -> dict:
        return {
            "polis_id": self.polis_id,
            "polis_url": self.polis_url,
            "report_id": self.report_id,
            "conversation_id": self.conversation_id,
            "base_url": self.base_url,
            "repair_is_meta_column": self.repair_is_meta_column,
            "import_dir": self.import_dir,
        }
