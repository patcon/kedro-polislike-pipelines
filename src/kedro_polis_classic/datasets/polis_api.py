import pandas as pd
from kedro.io import AbstractDataset
from typing import Dict

class PolisAPIDataset(AbstractDataset):
    def __init__(self, report_id: str):
        self.report_id = report_id

    def load(self) -> Dict[str, pd.DataFrame]:
        base = f"https://pol.is/api/v3/reportExport/{self.report_id}"
        comments_url = f"{base}/comments.csv"
        votes_url = f"{base}/votes.csv"

        comments = pd.read_csv(comments_url)
        votes = pd.read_csv(votes_url)

        return {"comments": comments, "votes": votes}

    def save(self, data: Dict[str, pd.DataFrame]) -> None:
        raise NotImplementedError("Saving to Polis API is not supported.")

    def _describe(self) -> dict:
        return {"report_id": self.report_id}
