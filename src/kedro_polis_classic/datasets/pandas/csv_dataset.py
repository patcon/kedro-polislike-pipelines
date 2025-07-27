from kedro_datasets.pandas.csv_dataset import CSVDataset, TablePreview
import pandas as pd

class CustomCSVDataset(CSVDataset):
    def preview(self, nrows: int = 5) -> TablePreview:
        """
        Generate a preview of the dataset with a specified number of rows.

        This is customed beyond pandas.CSVDataset in several ways:
        - Defaults to 10-row previews instead of 5, without setting
          `metadata.kedro-viz.preview_args.nrows` for each individual catalog item.
        - Adds option to set `metadata.index_in_preview` to include index column in previews.
        - Resolves bug where boolean columns weren't rendering in preview.

        Args:
            nrows: The number of rows to include in the preview. Defaults to 10.

        Returns:
            dict: A dictionary containing the data in a split format.
        """
        # Create a copy so it doesn't contaminate the original dataset
        dataset_copy = self._copy()
        dataset_copy._load_args["nrows"] = nrows  # type: ignore[attr-defined]
        data = dataset_copy.load()

        # Default behavior: don't include index unless explicitly requested
        index_in_preview = False
        if self.metadata:
            index_in_preview = self.metadata.get("index_in_preview", False)

        # Only add index if it's not a boring RangeIndex
        if index_in_preview and not isinstance(data.index, pd.RangeIndex):
            data = data.reset_index()
            # rename empty index column for clarity
            if data.columns[0] in (None, "index"):
                data.rename(columns={data.columns[0]: "__index__"}, inplace=True)

        # Convert all boolean columns to strings
        # See: https://github.com/kedro-org/kedro-viz/issues/2456
        bool_cols = data.select_dtypes(include="bool").columns
        data[bool_cols] = data[bool_cols].astype(str)

        return data.to_dict(orient="split")