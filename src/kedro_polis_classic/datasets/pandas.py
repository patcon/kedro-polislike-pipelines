from kedro_datasets.pandas.csv_dataset import CSVDataset, TablePreview
from copy import deepcopy
import pandas as pd


class CustomCSVDataset(CSVDataset):
    def __init__(self, *args, **kwargs):
        """
        A drop-in replacement for `kedro_datasets.pandas.CSVDataset` with optional preview
        enhancements and shorthand index configuration.

        This subclass supports two custom `metadata` flags:

        - `metadata.save_load_index`: When `True`, injects default index behavior
          equivalent to setting the following in the catalog entry:
              load_args:
                index_col: 0
              save_args:
                index: true

        - `metadata.index_in_preview`: When `True`, includes the index in kedro-viz
          previews (unless it's a default RangeIndex). The index column is renamed
          to `"__index__"` if unnamed.

        All other behavior is identical to the upstream `CSVDataset`. For full details,
        see the `kedro_datasets.pandas.CSVDataset` documentation.
        """
        metadata = deepcopy(kwargs.get("metadata", {}))
        load_args = deepcopy(kwargs.get("load_args", {})) or {}
        save_args = deepcopy(kwargs.get("save_args", {})) or {}

        if metadata.get("save_load_index", False):
            load_args.setdefault("index_col", 0)
            save_args.setdefault("index", True)
            kwargs["load_args"] = load_args
            kwargs["save_args"] = save_args

        super().__init__(*args, **kwargs)

    def preview(self, nrows: int = 10) -> TablePreview:
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
