import os
import pandas as pd
from kedro.io import AbstractDataset


class PolisAPIDataset(AbstractDataset):
    def __init__(
        self,
        polis_id: str | None = None,
        base_url: str | None = None,
        repair_is_meta_column: bool = True,
        import_dir: str | None = None,
    ):
        self.polis_id = polis_id
        self.base_url = base_url if base_url else "https://pol.is"
        self.repair_is_meta_column = repair_is_meta_column
        self.import_dir = import_dir

        # Initialize report_id and conversation_id
        self.report_id = None
        self.conversation_id = None

        # If import_dir is provided, prioritize it over polis_id
        if import_dir:
            # We'll determine conversation_id from the loaded data
            pass
        elif polis_id:
            # Determine if polis_id is a report_id or conversation_id
            if polis_id.startswith("r"):
                self.report_id = polis_id
            elif polis_id[0].isdigit():
                self.conversation_id = polis_id
            else:
                raise ValueError(
                    "polis_id must start with 'r' (for report_id) or a digit (for conversation_id)"
                )
        else:
            raise ValueError("Either polis_id or import_dir must be provided")

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

        # Use Loader with csv_export data source
        loader = Loader(
            polis_id=self.report_id,
            data_source="csv_export",
            polis_instance_url=self.base_url,
        )

        # Convert the list data to DataFrames
        comments = pd.DataFrame(loader.comments_data)
        votes = pd.DataFrame(loader.votes_data)

        # No column renaming - use original column names

        return {"comments": comments, "votes": votes}

    def load_from_api(self) -> dict[str, pd.DataFrame]:
        """Load data using the reddwarf data loader with conversation_id."""
        if not self.conversation_id:
            raise ValueError("conversation_id is required for loading from API")

        from reddwarf.data_loader import Loader

        loader = Loader(
            conversation_id=self.conversation_id, polis_instance_url=self.base_url
        )

        # Convert the list data to DataFrames
        comments = pd.DataFrame(loader.comments_data)
        votes = pd.DataFrame(loader.votes_data)

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
        loader.conversation_id = loader.conversation_data["conversation_id"]

        # Convert the list data to DataFrames
        comments = pd.DataFrame(loader.comments_data)
        votes = pd.DataFrame(loader.votes_data)

        return {"comments": comments, "votes": votes}

    def save(self, data: dict[str, pd.DataFrame]) -> None:
        raise NotImplementedError("Saving to Polis API is not supported.")

    def _describe(self) -> dict:
        return {
            "polis_id": self.polis_id,
            "report_id": self.report_id,
            "conversation_id": self.conversation_id,
            "base_url": self.base_url,
            "repair_is_meta_column": self.repair_is_meta_column,
            "import_dir": self.import_dir,
        }
