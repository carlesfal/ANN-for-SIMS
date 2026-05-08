from __future__ import annotations

import unittest

import pandas as pd

from app.utils.preprocessing import normalize_dataframe, split_xy


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})

    def test_min_max_normalization(self) -> None:
        result = normalize_dataframe(self.df, ["x"], "min-max")
        self.assertAlmostEqual(float(result["x"].min()), 0.0)
        self.assertAlmostEqual(float(result["x"].max()), 1.0)

    def test_z_score_normalization(self) -> None:
        result = normalize_dataframe(self.df, ["x"], "z-score")
        self.assertAlmostEqual(float(result["x"].mean()), 0.0, places=7)

    def test_split_ratios(self) -> None:
        splits = split_xy(self.df, ["x"], "y", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        total = len(splits["x_train"]) + len(splits["x_val"]) + len(splits["x_test"])
        self.assertEqual(total, len(self.df))


if __name__ == "__main__":
    unittest.main()
