import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def acquire(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "uci.zip"
    if not archive.exists():
        response = requests.get(URL, timeout=180)
        response.raise_for_status()
        archive.write_bytes(response.content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        raw = pd.read_csv(io.BytesIO(z.read("household_power_consumption.txt")),
                          sep=";", usecols=["Date", "Time", "Global_active_power"],
                          na_values=["?"], dtype={"Global_active_power": float})
    timestamps = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M:%S")
    if timestamps.isna().any() or np.isinf(raw.Global_active_power).any():
        raise ValueError("Schéma UCI invalide : dates manquantes ou puissance infinie")
    if (raw.Global_active_power.dropna() < 0).any():
        raise ValueError("Puissance UCI négative incompatible avec le schéma")
    series = pd.Series(raw.Global_active_power.to_numpy(), index=timestamps).sort_index()
    duplicates = int(series.index.duplicated().sum())
    series = series.groupby(level=0).mean()
    hourly = series.resample("h").agg(["mean", "count"])
    # At least 45 observed minutes required; imputation occurs downstream causally.
    y = hourly["mean"].where(hourly["count"] >= 45)
    frame = pd.DataFrame({"timestamp": y.index, "power_kw": y.values,
                          "observed_minutes": hourly["count"].values})
    frame.to_csv(root / "hourly.csv", index=False)
    metadata = {"url": URL, "sha256": digest, "license": "CC BY 4.0",
                "raw_rows": len(raw), "raw_missing": int(raw.Global_active_power.isna().sum()),
                "duplicates": duplicates, "hourly_missing": int(y.isna().sum()),
                "minute_gaps": int((series.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum()),
                "start": str(y.index.min()), "end": str(y.index.max()),
                "timezone": "Non précisé par UCI; horodatages locaux naïfs conservés",
                "unit": "kW, puissance moyenne horaire", "anomaly_labels": False}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return frame, metadata


def causal_fill(values):
    """No backward filling: leading missing observations are rejected."""
    s = pd.Series(values, dtype=float)
    if s.empty or pd.isna(s.iloc[0]):
        raise ValueError("La première observation doit être disponible.")
    return s.ffill().to_numpy()


def validate_csv(source):
    df = pd.read_csv(source)
    if not {"timestamp", "power_kw"}.issubset(df.columns):
        raise ValueError("Colonnes requises : timestamp, power_kw (kW).")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="raise")
    if df.timestamp.isna().any() or df.timestamp.dt.tz is not None:
        raise ValueError("Dates manquantes ou fuseau incompatible : dates locales sans fuseau.")
    if df.timestamp.duplicated().any():
        raise ValueError("Horodatages dupliqués : import refusé.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["power_kw"] = pd.to_numeric(df.power_kw, errors="raise")
    if np.isinf(df.power_kw).any() or (df.power_kw.dropna() < 0).any():
        raise ValueError("Puissance négative ou infinie incompatible.")
    if len(df) < 49 or not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Au moins 49 lignes à fréquence strictement horaire requises.")
    df["imputed"] = df.power_kw.isna()
    df["power_kw"] = causal_fill(df.power_kw)
    return df


def partitions(n):
    return int(n * .6), int(n * .8)
