import pandas as pd
from kedro.io import AbstractDataset

class PolisAPIDataset(AbstractDataset):
    def __init__(self, report_id: str, base_url: str | None = None):
        self.report_id = report_id
        self.base_url = base_url if base_url else "https://pol.is"

    def load(self) -> dict[str, pd.DataFrame]:
        export_base = f"${self.base_url}/api/v3/reportExport/{self.report_id}"
        comments_url = f"{export_base}/comments.csv"
        votes_url = f"{export_base}/votes.csv"

        comments = pd.read_csv(comments_url)
        votes = pd.read_csv(votes_url)

        return {"comments": comments, "votes": votes}

    def save(self, data: dict[str, pd.DataFrame]) -> None:
        raise NotImplementedError("Saving to Polis API is not supported.")

    def _describe(self) -> dict:
        return {"report_id": self.report_id}
