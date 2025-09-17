import pandas as pd
from kedro.io import AbstractDataset


class PolisAPIDataset(AbstractDataset):
    def __init__(
        self,
        polis_id: str,
        base_url: str | None = None,
        repair_is_meta_column: bool = True,
    ):
        self.polis_id = polis_id
        self.base_url = base_url if base_url else "https://pol.is"
        self.repair_is_meta_column = repair_is_meta_column

        # Determine if polis_id is a report_id or conversation_id
        if polis_id.startswith("r"):
            self.report_id = polis_id
            self.conversation_id = None
        elif polis_id[0].isdigit():
            self.conversation_id = polis_id
            self.report_id = None
        else:
            raise ValueError(
                "polis_id must start with 'r' (for report_id) or a digit (for conversation_id)"
            )

    def load(self) -> dict[str, pd.DataFrame]:
        """Load data using the appropriate method based on provided parameters."""
        if self.report_id:
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

        # Rename columns to expected format (same as load_from_api)
        votes = votes.rename(
            columns={
                "modified": "timestamp",
                "participant_id": "voter-id",
                "statement_id": "comment-id",
            }
        )

        comments = comments.rename(
            columns={
                "statement_id": "comment-id",
                "is_meta": "is-meta",
            }
        )

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

        # Rename files to expected format
        votes = votes.rename(
            columns={
                "modified": "timestamp",
                "participant_id": "voter-id",
                "statement_id": "comment-id",
            }
        )

        comments = comments.rename(
            columns={
                "statement_id": "comment-id",
                "is_meta": "is-meta",
            }
        )

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
        }
