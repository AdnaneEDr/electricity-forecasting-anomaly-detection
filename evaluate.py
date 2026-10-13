"""Réévaluer les artefacts figés, sans entraînement ni sélection de seuil."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from energy.anomalies import score_detectors
from energy.evaluation import anomaly_metrics
from energy.forecast import metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    folder = Path(__file__).parent / "artifacts" / args.mode
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    with (folder / "models.pkl").open("rb") as f:
        bundle = pickle.load(f)
    predictions = pd.read_csv(folder / "predictions.csv")
    predictions = predictions.loc[~predictions.imputed_target]
    rows = [{"model": name, **metrics(g.actual, g.prediction, run["MASE_train_denominator"])}
            for name, g in predictions.groupby("model")]
    result = pd.DataFrame(rows)
    original = pd.read_csv(folder / "forecast_metrics.csv")
    np.testing.assert_allclose(result[["MAE", "RMSE", "MASE"]], original[["MAE", "RMSE", "MASE"]])
    print(result.to_string(index=False))
    cases = []
    for split in ["calibration", "test"]:
        frame = pd.read_csv(folder / f"injected_{split}.csv", parse_dates=["timestamp"])
        start = run["train"] if split == "calibration" else run["train"] + run["validation"]
        scores = score_detectors(frame.power_kw.to_numpy(), frame.timestamp, bundle["detectors"])
        for name, score in scores.items():
            result = anomaly_metrics(frame.label.to_numpy(), score,
                                     bundle["detectors"]["thresholds"][name],
                                     run["injections"][split], start)
            expected = pd.read_csv(folder / "synthetic_metrics.csv")
            expected = expected.loc[(expected.split == split) & (expected.method == name)].iloc[0]
            for metric in ["precision", "recall", "F1", "PR_AUC_AP", "event_recall"]:
                np.testing.assert_allclose(result[metric], expected[metric], atol=1e-6)
            for spec in run["injections"][split]:
                pos, duration = spec["position"], spec["duration_h"]
                hit = np.flatnonzero(score[pos:pos + duration] > bundle["detectors"]["thresholds"][name])
                cases.append({"split": split, "method": name, "kind": spec["kind"],
                              "detected": bool(len(hit)), "delay_h": int(hit[0]) if len(hit) else None})
            print(split, name, result)
    print("Réévaluation identique aux métriques sauvegardées.")
    pd.DataFrame(cases).to_csv(folder / "injection_cases.csv", index=False)
