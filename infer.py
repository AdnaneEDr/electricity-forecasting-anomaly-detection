import argparse
import pickle
from pathlib import Path

import pandas as pd

from energy.anomalies import score_detectors
from energy.data import validate_csv
from energy.forecast import predict_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--model", default="artifacts/quick/models.pkl")
    p.add_argument("--output", default="inference")
    args = p.parse_args()
    df = validate_csv(args.csv)
    if len(df) < 168:
        raise ValueError("168 observations nécessaires pour toutes les prévisions")
    # Load only trusted, locally generated pickle files.
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    y = df.power_kw.to_numpy()
    times = pd.DatetimeIndex(df.timestamp)
    future = pd.date_range(times[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    predictions = predict_all(y, times.append(future), len(y), model["ridge"], model["ets_days"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": future, **predictions}).to_csv(out / "forecasts.csv", index=False)
    scores = score_detectors(y, times, model["detectors"])
    for name, score in scores.items():
        pd.DataFrame({"timestamp": times, "score": score,
                      "alert": score > model["detectors"]["thresholds"][name]}).to_csv(
            out / f"alerts_{name}.csv", index=False)


if __name__ == "__main__":
    main()
