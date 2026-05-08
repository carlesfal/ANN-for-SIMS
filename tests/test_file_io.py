from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.utils.file_io import load_tabular_data, save_dataframe_csv


class TestFileIO(unittest.TestCase):
    def test_csv_round_trip(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            save_dataframe_csv(df, path)
            loaded = load_tabular_data(path)

        self.assertEqual(list(loaded.columns), ["a", "b"])
        self.assertEqual(loaded.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
