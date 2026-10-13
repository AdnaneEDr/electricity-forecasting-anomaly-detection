import io
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from energy.anomalies import Autoencoder, events, score_detectors, windows
from energy.data import causal_fill, partitions, validate_csv
from energy.evaluation import anomaly_metrics, inject
from energy.forecast import features, fit_ridge, metrics, supervised
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def series():
    t = pd.date_range("2020-01-01", periods=1200, freq="h")
    y = 2 + np.sin(np.arange(1200) * np.pi / 12)
    return y, t


def test_targets_stay_in_partition(series):
    y, t = series
    a, b = partitions(len(y))
    x, targets = supervised(y, t, a)
    assert len(x) == len(targets)
    np.testing.assert_equal(targets[-1], y[168 + 3 * (len(x) - 1):168 + 3 * (len(x) - 1) + 24])
    assert 168 + 3 * (len(x) - 1) + 24 <= a < b < len(y)


def test_features_future_invariance(series):
    y, t = series
    changed = y.copy()
    changed[500:] = 999
    np.testing.assert_array_equal(features(y, t, 500), features(changed, t, 500))


def test_imputation_causal():
    np.testing.assert_equal(causal_fill([1, np.nan, 3]), [1, 1, 3])
    with pytest.raises(ValueError):
        causal_fill([np.nan, 1])


def test_trailing_window():
    np.testing.assert_equal(windows(np.arange(50))[-1], np.arange(2, 50))


def test_metrics():
    assert metrics([1, 3], [2, 1], 2) == {"MAE": 1.5, "RMSE": np.sqrt(2.5), "MASE": .75}


def test_scaler_train_and_reload(series):
    y, t = series
    model = fit_ridge(y, t, 600, 100)
    train_x, _ = supervised(y, t, 600)
    np.testing.assert_allclose(model[0].mean_, train_x.mean(axis=0))
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(model.predict(features(y, t, 800)[None]),
                               loaded.predict(features(y, t, 800)[None]))


def test_alerts_future_invariance(series):
    y, t = series
    torch.manual_seed(42)
    torch.set_num_threads(2)
    bundle = {"center": np.zeros(24), "spread": np.ones(24),
              "scaler": StandardScaler().fit(y[:500, None]),
              "state_dict": Autoencoder().state_dict()}
    original = score_detectors(y, t, bundle)
    changed = y.copy()
    changed[800:] += 10
    updated = score_detectors(changed, t, bundle)
    prefix = score_detectors(y[:800], t[:800], bundle)
    for name in original:
        np.testing.assert_allclose(original[name][:800], updated[name][:800], equal_nan=True)
        np.testing.assert_allclose(original[name][:800], prefix[name], equal_nan=True, atol=1e-6)
        np.testing.assert_array_equal(original[name][:800] > 1, prefix[name] > 1)


def test_events():
    result = events([False, True, True, False, True], np.arange(5))
    assert list(result.duration_h) == [2, 1]


def test_injection_is_separate_and_event_delay(series):
    y, _ = series
    original = y.copy()
    _, labels, specs = inject(y, 800, 44)
    np.testing.assert_equal(y, original)
    result = anomaly_metrics(labels, labels.astype(float), .5, specs, 800)
    assert result["F1"] == 1 and result["mean_delay_h_detected"] == 0


def test_csv_validation(series):
    y, t = series
    df = pd.DataFrame({"timestamp": t[:200], "power_kw": y[:200]})
    validated = validate_csv(io.StringIO(df.to_csv(index=False)))
    assert len(validated) == 200
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="dupliqués"):
        validate_csv(io.StringIO(df.to_csv(index=False)))


def test_csv_rejects_units_and_frequency(series):
    y, t = series
    df = pd.DataFrame({"timestamp": t[:200], "power_kw": y[:200]})
    df.loc[2, "power_kw"] = -1
    with pytest.raises(ValueError, match="négative"):
        validate_csv(io.StringIO(df.to_csv(index=False)))
    df.loc[2, "power_kw"] = 1
    df = df.drop(index=3)
    with pytest.raises(ValueError, match="horaire"):
        validate_csv(io.StringIO(df.to_csv(index=False)))
