"""Acquisition/préparation seules, sans apprentissage."""
import argparse
from pathlib import Path

from energy.data import acquire, causal_fill


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "data"
    frame, metadata = acquire(root)
    if not args.acquire_only:
        frame = frame.loc[frame.power_kw.first_valid_index():].copy()
        frame["imputed"] = frame.power_kw.isna()
        frame["power_kw"] = causal_fill(frame.power_kw)
        frame.to_csv(root / "prepared_hourly.csv", index=False)
    print(metadata)
