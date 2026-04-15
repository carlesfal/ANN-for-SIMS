"""
Unit tests for backend/python/utils.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add the python directory to sys.path so utils can be imported directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    average_metrics,
    compute_metrics,
    create_scalers,
    fit_scale,
    inverse_scale_y,
    load_data,
    transform_scale,
)


# ─── compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y)
        assert m['rmse'] == pytest.approx(0.0)
        assert m['mae'] == pytest.approx(0.0)
        assert m['r2'] == pytest.approx(1.0)
        assert m['mape'] == pytest.approx(0.0)

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        m = compute_metrics(y_true, y_pred)
        assert m['rmse'] == pytest.approx(0.5)
        assert m['mae'] == pytest.approx(0.5)
        # MAPE: each point is 0.5/true → [50%, 25%, 16.67%, 12.5%] → mean≈26.04%
        assert m['mape'] == pytest.approx(26.0417, abs=0.1)

    def test_r2_negative_for_bad_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        m = compute_metrics(y_true, y_pred)
        assert m['r2'] < 0

    def test_mape_nan_when_all_targets_near_zero(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert math.isnan(m['mape'])

    def test_accepts_lists(self):
        m = compute_metrics([1.0, 2.0], [1.0, 2.0])
        assert m['rmse'] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        m = compute_metrics([1.0], [1.0])
        assert set(m.keys()) == {'rmse', 'mae', 'r2', 'mape'}


# ─── average_metrics ──────────────────────────────────────────────────────────

class TestAverageMetrics:
    def test_empty_list(self):
        assert average_metrics([]) == {}

    def test_single_dict(self):
        m = {'rmse': 0.5, 'mae': 0.3, 'r2': 0.9, 'mape': 10.0}
        result = average_metrics([m])
        assert result['rmse'] == pytest.approx(0.5)
        assert result['mae'] == pytest.approx(0.3)

    def test_average_of_two(self):
        m1 = {'rmse': 0.4, 'mae': 0.2, 'r2': 0.8, 'mape': 8.0}
        m2 = {'rmse': 0.6, 'mae': 0.4, 'r2': 0.9, 'mape': 12.0}
        result = average_metrics([m1, m2])
        assert result['rmse'] == pytest.approx(0.5)
        assert result['mae'] == pytest.approx(0.3)
        assert result['r2'] == pytest.approx(0.85)
        assert result['mape'] == pytest.approx(10.0)

    def test_std_keys_present(self):
        m1 = {'rmse': 0.4, 'mae': 0.2, 'r2': 0.8, 'mape': 8.0}
        m2 = {'rmse': 0.6, 'mae': 0.4, 'r2': 0.9, 'mape': 12.0}
        result = average_metrics([m1, m2])
        assert 'rmse_std' in result
        assert 'mae_std' in result

    def test_ignores_nan_values(self):
        m1 = {'rmse': 0.5, 'mape': float('nan')}
        m2 = {'rmse': 0.3, 'mape': float('nan')}
        result = average_metrics([m1, m2])
        assert result['rmse'] == pytest.approx(0.4)
        assert math.isnan(result['mape'])


# ─── create_scalers ───────────────────────────────────────────────────────────

class TestCreateScalers:
    def test_returns_two_scalers(self):
        scaler_X, scaler_y = create_scalers()
        assert scaler_X is not None
        assert scaler_y is not None

    def test_scalers_are_independent(self):
        scaler_X, scaler_y = create_scalers()
        X = np.array([[0.0], [1.0]])
        y = np.array([[10.0], [20.0]])
        scaler_X.fit(X)
        scaler_y.fit(y)
        # Transforming X should not affect y scaler and vice versa
        assert scaler_X.data_max_[0] == pytest.approx(1.0)
        assert scaler_y.data_max_[0] == pytest.approx(20.0)


# ─── fit_scale / transform_scale / inverse_scale_y ────────────────────────────

class TestScaling:
    def setup_method(self):
        self.scaler_X, self.scaler_y = create_scalers()
        self.X_train = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        self.y_train = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

    def test_fit_scale_output_in_0_1(self):
        X_s, y_s = fit_scale(self.scaler_X, self.scaler_y, self.X_train, self.y_train)
        assert X_s.min() == pytest.approx(0.0)
        assert X_s.max() == pytest.approx(1.0)
        assert y_s.min() == pytest.approx(0.0)
        assert y_s.max() == pytest.approx(1.0)

    def test_transform_scale_uses_fitted_scaler(self):
        fit_scale(self.scaler_X, self.scaler_y, self.X_train, self.y_train)
        X_test = np.array([[0.0, 0.0]], dtype=np.float32)
        X_s = transform_scale(self.scaler_X, self.scaler_y, X_test)
        # Min value should map to 0
        assert X_s[0, 0] == pytest.approx(0.0)

    def test_inverse_scale_y_recovers_original(self):
        _, y_s = fit_scale(self.scaler_X, self.scaler_y, self.X_train, self.y_train)
        y_inv = inverse_scale_y(self.scaler_y, y_s)
        np.testing.assert_allclose(y_inv, self.y_train.flatten(), rtol=1e-5)


# ─── load_data ────────────────────────────────────────────────────────────────

class TestLoadData:
    def _make_csv(self, df, suffix='.csv'):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def _make_excel(self, df):
        tmp = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        tmp.close()
        df.to_excel(tmp.name, index=False)
        return tmp.name

    def test_loads_csv_correctly(self):
        df = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6], 'target': [7, 8, 9]})
        path = self._make_csv(df)
        try:
            X, y, cols = load_data(path, 'target')
            assert X.shape == (3, 2)
            assert y.shape == (3, 1)
            assert 'feat1' in cols and 'feat2' in cols
        finally:
            os.unlink(path)

    def test_loads_excel_correctly(self):
        openpyxl = pytest.importorskip('openpyxl', reason='openpyxl not installed')
        df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0], 'y': [5.0, 6.0]})
        path = self._make_excel(df)
        try:
            X, y, cols = load_data(path, 'y')
            assert X.shape == (2, 2)
            assert y.shape == (2, 1)
        finally:
            os.unlink(path)

    def test_explicit_feature_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'target': [7, 8]})
        path = self._make_csv(df)
        try:
            X, y, cols = load_data(path, 'target', feature_columns=['a', 'c'])
            assert X.shape == (2, 2)
            assert cols == ['a', 'c']
        finally:
            os.unlink(path)

    def test_explicit_feature_columns_as_comma_string(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'target': [7, 8]})
        path = self._make_csv(df)
        try:
            X, y, cols = load_data(path, 'target', feature_columns='a, b')
            assert cols == ['a', 'b']
        finally:
            os.unlink(path)

    def test_raises_for_missing_target(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        path = self._make_csv(df)
        try:
            with pytest.raises(ValueError, match="Target column"):
                load_data(path, 'no_such_column')
        finally:
            os.unlink(path)

    def test_raises_for_missing_feature_columns(self):
        df = pd.DataFrame({'a': [1, 2], 'target': [3, 4]})
        path = self._make_csv(df)
        try:
            with pytest.raises(ValueError, match="Feature columns not found"):
                load_data(path, 'target', feature_columns=['a', 'missing'])
        finally:
            os.unlink(path)

    def test_drops_na_rows(self):
        df = pd.DataFrame({'feat': [1.0, None, 3.0], 'target': [4.0, 5.0, 6.0]})
        path = self._make_csv(df)
        try:
            X, y, _ = load_data(path, 'target')
            assert X.shape[0] == 2  # one NaN row dropped
        finally:
            os.unlink(path)

    def test_returns_float32_arrays(self):
        df = pd.DataFrame({'feat': [1, 2, 3], 'target': [4, 5, 6]})
        path = self._make_csv(df)
        try:
            X, y, _ = load_data(path, 'target')
            assert X.dtype == np.float32
            assert y.dtype == np.float32
        finally:
            os.unlink(path)
