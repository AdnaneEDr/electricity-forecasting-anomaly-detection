"""Acquisition, apprentissage, évaluation et sauvegarde : python run.py --mode quick."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from statsmodels.tsa.seasonal import STL

from energy.anomalies import events, fit_detectors, score_detectors
from energy.data import acquire, causal_fill, partitions
from energy.evaluation import anomaly_metrics, inject
from energy.forecast import features, fit_ets, fit_ridge, metrics, predict_all


def main(mode="quick"):
    root = Path(__file__).parent
    config = json.loads((root / "config.json").read_text())
    if config["horizon"] != 24 or config["window"] != 48:
        raise ValueError("Cette version prend en charge horizon=24 et window=48 uniquement.")
    np.random.seed(config["seed"])
    frame, metadata = acquire(root / "data")
    if mode == "quick":
        # A complete year retains weekly and yearly variation.
        frame = frame.loc[frame.timestamp.between("2009-01-01", "2009-12-31 23:00:00")].copy()
    else:
        frame = frame.loc[frame.power_kw.first_valid_index():].copy()
    frame = frame.reset_index(drop=True)
    frame["imputed"] = frame.power_kw.isna()
    frame["power_kw"] = causal_fill(frame.power_kw)
    y = frame.power_kw.to_numpy()
    times = pd.DatetimeIndex(frame.timestamp)
    a, b = partitions(len(y))
    out = root / "artifacts" / mode
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "series.csv", index=False)
    exploration = {"distribution_kw": frame.power_kw.describe().to_dict(),
                   "autocorrelations": {str(lag): frame.power_kw.autocorr(lag)
                                         for lag in [1, 24, 168]},
                   "missing_by_month": frame.assign(month=frame.timestamp.dt.strftime("%Y-%m"))
                   .groupby("month").imputed.sum().astype(int).to_dict()}
    (out / "exploration.json").write_text(json.dumps(exploration, indent=2))
    scale = float(np.abs(y[24:a] - y[:a - 24]).mean())
    val_origins = np.unique(np.linspace(a, b - 24, config["validation_origins_quick"] if mode == "quick" else 40).astype(int))
    tuning = []
    # Two expanding training folds, both before reserved test.
    half = len(val_origins) // 2
    folds = [(a, val_origins[:half]), (int((a + b) / 2), val_origins[half:])]
    for alpha in [1., 100., 1000.]:
        errors = []
        for end, origins in folds:
            model = fit_ridge(y, times, end, alpha)
            errors.extend([np.abs(y[o:o + 24] - model.predict(features(y, times, o)[None])[0]).mean()
                           for o in origins if o >= end])
        tuning.append({"alpha": alpha, "validation_MAE": float(np.mean(errors))})
    alpha = min(tuning, key=lambda t: t["validation_MAE"])["alpha"]
    # Statistical model's history length selected on validation only.
    provisional = fit_ridge(y, times, a, alpha)
    ets_tuning = []
    for days in [14, 28]:
        errors = [np.abs(y[o:o + 24] - predict_all(y, times, o, provisional, days)["ETS"]).mean()
                  for o in val_origins[::2]]
        ets_tuning.append({"days": days, "validation_MAE": float(np.mean(errors))})
    ets_days = min(ets_tuning, key=lambda t: t["validation_MAE"])["days"]
    # Keep scaler and ridge fitted to TRAIN for consistent comparison and strict isolation.
    ridge = fit_ridge(y, times, a, alpha)
    test_origins = np.unique(np.linspace(b, len(y) - 24,
                                        config["test_origins_quick"] if mode == "quick" else 100).astype(int))
    records = []
    for o in test_origins:
        for name, pred in predict_all(y, times, o, ridge, ets_days).items():
            for h in range(24):
                records.append({"origin": str(times[o]), "timestamp": str(times[o + h]),
                                "horizon": h + 1, "model": name,
                                "actual": y[o + h], "prediction": pred[h],
                                "imputed_target": bool(frame.imputed.iloc[o + h])})
    predictions = pd.DataFrame(records)
    # Scores use observed targets only; causal imputations remain available as context.
    measured = predictions.loc[~predictions.imputed_target].copy()
    forecast_metrics = [{"model": name, **metrics(g.actual, g.prediction, scale)}
                        for name, g in measured.groupby("model")]
    pd.DataFrame(forecast_metrics).to_csv(out / "forecast_metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    horizon = [{"model": name, "horizon": h, **metrics(g.actual, g.prediction, scale)}
               for (name, h), g in measured.groupby(["model", "horizon"])]
    pd.DataFrame(horizon).to_csv(out / "horizon_metrics.csv", index=False)
    measured["hour"] = pd.to_datetime(measured.timestamp).dt.hour
    measured["weekend"] = pd.to_datetime(measured.timestamp).dt.dayofweek >= 5
    period = [{"model": name, "weekend": bool(weekend), **metrics(g.actual, g.prediction, scale)}
              for (name, weekend), g in measured.groupby(["model", "weekend"])]
    pd.DataFrame(period).to_csv(out / "period_metrics.csv", index=False)
    detectors = fit_detectors(y, times, a, b,
                              config["epochs_quick"] if mode == "quick" else config["epochs_full"])
    pd.DataFrame({"epoch": np.arange(1, len(detectors["loss"]) + 1),
                  "train_MSE": detectors["loss"]}).to_csv(out / "training_loss.csv", index=False)
    scores = score_detectors(y, times, detectors)
    alert_records = []
    for name, score in scores.items():
        threshold = detectors["thresholds"][name]
        ev = events(score[b:] > threshold, times[b:])
        ev.to_csv(out / f"events_{name}.csv", index=False)
        alert_records.extend([{"timestamp": str(times[i]), "method": name,
                               "score": score[i], "threshold": threshold,
                               "alert": bool(score[i] > threshold),
                               "imputed": bool(frame.imputed.iloc[i])}
                              for i in range(b, len(y))])
    pd.DataFrame(alert_records).to_csv(out / "alerts.csv", index=False)
    synthetic = []
    specs_all = {}
    # Calibration scenarios are diagnostic only; clean-validation q99 thresholds stay frozen.
    for split, start, end, seed in [("calibration", a, b, 43), ("test", b, len(y), 44)]:
        altered, labels, specs = inject(y[:end], start, seed)
        specs_all[split] = specs
        injected_scores = score_detectors(altered, times[:end], detectors)
        pd.DataFrame({"timestamp": times[:end], "power_kw": altered, "label": labels}).to_csv(
            out / f"injected_{split}.csv", index=False)
        for name, score in injected_scores.items():
            synthetic.append({"split": split, "method": name,
                              **anomaly_metrics(labels, score, detectors["thresholds"][name],
                                                specs, start)})
    pd.DataFrame(synthetic).to_csv(out / "synthetic_metrics.csv", index=False)
    bundle = {"ridge": ridge, "detectors": detectors, "ets_days": ets_days,
              "ets_at_test_start": fit_ets(y, b, ets_days),
              "unit": "kW", "frequency": "h", "timezone": "naive-local",
              "train_end": str(times[a - 1]), "validation_end": str(times[b - 1]),
              "seed": 42, "mode": mode, "source_sha256": metadata["sha256"]}
    with (out / "models.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    torch.save(detectors["state_dict"], out / "autoencoder.pt")
    model_metadata = {key: bundle[key] for key in ["unit", "frequency", "timezone", "train_end",
                                                  "validation_end", "seed", "mode", "source_sha256"]}
    model_metadata.update({"ridge_alpha": alpha, "ets_days": ets_days,
                           "thresholds": detectors["thresholds"], "window": 48, "horizon": 24})
    (out / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    report = {"mode": mode, "n": len(y), "train": a, "validation": b - a,
              "test": len(y) - b, "imputed": int(frame.imputed.sum()),
              "MASE_train_denominator": scale, "ridge_tuning": tuning,
              "ets_tuning": ets_tuning, "thresholds": detectors["thresholds"],
              "injections": specs_all, "forecast": forecast_metrics, "synthetic": synthetic}
    (out / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    px.line(frame, x="timestamp", y="power_kw", title="Puissance moyenne horaire (kW)").write_html(out / "series.html")
    px.line(pd.DataFrame(horizon), x="horizon", y="MAE", color="model",
            title="MAE par horizon — test observé").write_html(out / "horizons.html")
    px.bar(pd.DataFrame(synthetic), x="method", y="F1", color="split",
           barmode="group", title="Anomalies injectées uniquement").write_html(out / "synthetic.html")
    decomposition = STL(y[:a], period=24, robust=True).fit()
    pd.DataFrame({"timestamp": times[:a], "trend": decomposition.trend,
                  "seasonal": decomposition.seasonal, "residual": decomposition.resid}).to_csv(
        out / "stl_train.csv", index=False)
    # Actual reload verification.
    with (out / "models.pkl").open("rb") as f:
        loaded = pickle.load(f)
    np.testing.assert_allclose(ridge.predict(features(y, times, b)[None]),
                               loaded["ridge"].predict(features(y, times, b)[None]))
    np.testing.assert_allclose(bundle["ets_at_test_start"].forecast(24),
                               loaded["ets_at_test_start"].forecast(24))
    weights = torch.load(out / "autoencoder.pt", weights_only=True)
    for key, value in weights.items():
        torch.testing.assert_close(value, loaded["detectors"]["state_dict"][key])
    changed = y.copy()
    changed[b + 100:] += 100
    changed_scores = score_detectors(changed, times, loaded["detectors"])
    prefix_scores = score_detectors(y[:b + 100], times[:b + 100], loaded["detectors"])
    for name in scores:
        np.testing.assert_allclose(scores[name][:b + 100], changed_scores[name][:b + 100],
                                   equal_nan=True, atol=1e-6)
        np.testing.assert_allclose(scores[name][:b + 100], prefix_scores[name], equal_nan=True, atol=1e-6)
    (out / "verification.json").write_text(json.dumps({"reload": True, "future_invariance": True,
                                                       "ets_reload": True, "torch_weights_reload": True,
                                                       "prefix_replay": True}))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    main(parser.parse_args().mode)
