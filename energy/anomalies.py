import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, window=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window, 24), nn.ReLU(), nn.Linear(24, 8),
                                 nn.ReLU(), nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, window))

    def forward(self, x):
        return self.net(x)


def windows(y, width=48):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(y), width).copy()


def fit_detectors(y, times, train_end, val_end, epochs=12, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    residual = y[24:train_end] - y[:train_end - 24]
    hours = pd.DatetimeIndex(times[24:train_end]).hour
    center = np.array([np.median(residual[hours == h]) for h in range(24)])
    spread = np.array([max(1.4826 * np.median(np.abs(residual[hours == h] - center[h])),
                           .05) for h in range(24)])
    scaler = StandardScaler().fit(y[:train_end, None])
    z = scaler.transform(y[:, None]).ravel()
    model = Autoencoder()
    x = torch.tensor(windows(z[:train_end]), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x, batch_size=256, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.
        for batch in loader:
            optimizer.zero_grad()
            loss = (model(batch) - batch).square().mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(x))
    bundle = {"center": center, "spread": spread, "scaler": scaler,
              "state_dict": model.state_dict(), "loss": history}
    scores = score_detectors(y[:val_end], times[:val_end], bundle)
    bundle["thresholds"] = {name: float(np.quantile(s[train_end:val_end], .99))
                            for name, s in scores.items()}
    return bundle


def score_detectors(y, times, bundle):
    y = np.asarray(y)
    robust = np.full(len(y), np.nan)
    hours = pd.DatetimeIndex(times).hour.to_numpy()[24:]
    robust[24:] = np.abs(y[24:] - y[:-24] - bundle["center"][hours]) / bundle["spread"][hours]
    model = Autoencoder()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    z = bundle["scaler"].transform(y[:, None]).ravel()
    x = torch.tensor(windows(z), dtype=torch.float32)
    with torch.no_grad():
        # Entire trailing-window MSE assigned only to the final timestamp.
        errors = torch.cat([(model(b) - b).square().mean(dim=1)
                            for b in x.split(1024)]).numpy()
    auto = np.full(len(y), np.nan)
    auto[47:] = errors
    return {"Robuste": robust, "Autoencodeur": auto}


def events(flags, times):
    flags = np.asarray(flags, dtype=bool)
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
    return pd.DataFrame([{"start": str(times[a]), "end": str(times[b]),
                          "duration_h": int(b - a + 1)} for a, b in zip(starts, ends)],
                        columns=["start", "end", "duration_h"])
