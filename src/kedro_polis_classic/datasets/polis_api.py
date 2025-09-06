import pandas as pd
import requests
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
        """Load data from CSV endpoints using report_id."""
        if not self.report_id:
            raise ValueError("report_id is required for loading from CSV")

        export_base = f"{self.base_url}/api/v3/reportExport/{self.report_id}"
        comments_url = f"{export_base}/comments.csv"
        votes_url = f"{export_base}/votes.csv"

        comments = pd.read_csv(comments_url)
        votes = pd.read_csv(votes_url)

        # Check if 'is-meta' column exists, and if not, fetch the conversation_id
        if "is-meta" not in comments.columns and self.repair_is_meta_column:
            conversation_id = self._get_conversation_id()
            comments = self._add_is_meta_column(comments, conversation_id)

        return {"comments": comments, "votes": votes}

    def load_from_api(self) -> dict[str, pd.DataFrame]:
        """Load data using the reddwarf data loader with conversation_id."""
        if not self.conversation_id:
            raise ValueError("conversation_id is required for loading from API")

        from reddwarf.data_loader import Loader

        loader = Loader(conversation_id=self.conversation_id)

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

    def _get_conversation_id(self) -> str:
        """
        Fetches the conversation_id using the report_id from the reports API endpoint.
        """
        report_url = f"{self.base_url}/api/v3/reports?report_id={self.report_id}"
        response = requests.get(report_url)
        response.raise_for_status()  # Ensure a valid response

        # Extract the conversation_id from the response JSON
        data = response.json()
        conversation_id = data[0].get("conversation_id")
        if not conversation_id:
            raise ValueError(f"conversation_id not found for report {self.report_id}")
        return conversation_id

    def _add_is_meta_column(
        self, comments: pd.DataFrame, conversation_id: str
    ) -> pd.DataFrame:
        """
        Adds an 'is-meta' column to the comments DataFrame by fetching comment data from the API.
        """
        comments_api_url = f"{self.base_url}/api/v3/comments?conversation_id={conversation_id}&moderation=true&include_voting_patterns=true"
        response = requests.get(comments_api_url)
        response.raise_for_status()

        # Extract the comments data from the API response
        comments_data = response.json()

        # Create a dictionary to map 'tid' to 'is_meta'
        is_meta_dict = {comment["tid"]: comment["is_meta"] for comment in comments_data}

        # Map the 'is-meta' values to the comments DataFrame based on 'comment-id' (or 'tid')
        comments["is-meta"] = comments["comment-id"].map(lambda tid: is_meta_dict[tid])

        return comments
