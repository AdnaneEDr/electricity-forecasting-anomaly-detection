import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def features(y, times, origin):
    """Origin is the first unseen target index; all values precede origin."""
    past = np.asarray(y[origin - 168:origin])
    t = pd.Timestamp(times[origin])
    return np.r_[past[-1], past[-24:], past[-168:-144],
                 past[-24:].mean(), past[-24:].std(), past.mean(),
                 np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24),
                 np.sin(2 * np.pi * t.dayofweek / 7), np.cos(2 * np.pi * t.dayofweek / 7)]


def supervised(y, times, end, horizon=24):
    origins = np.arange(168, end - horizon + 1, 3)
    return (np.stack([features(y, times, o) for o in origins]),
            np.stack([y[o:o + horizon] for o in origins]))


def fit_ridge(y, times, end, alpha):
    x, target = supervised(y, times, end)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, target)
    return model


def fit_ets(y, origin, ets_days=28):
    past = y[max(0, origin - ets_days * 24):origin]
    return ExponentialSmoothing(past, trend=None, seasonal="add", seasonal_periods=24,
                                initialization_method="estimated").fit(optimized=True)


def predict_all(y, times, origin, ridge, ets_days=28):
    ets = fit_ets(y, origin, ets_days)
    return {"Naïve": np.repeat(y[origin - 1], 24),
            "Saisonnière 24h": y[origin - 24:origin].copy(),
            "ETS": np.asarray(ets.forecast(24)),
            "Ridge direct": ridge.predict(features(y, times, origin)[None])[0]}


def metrics(actual, predicted, scale):
    error = np.asarray(actual) - np.asarray(predicted)
    return {"MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "MASE": float(np.abs(error).mean() / scale)}
