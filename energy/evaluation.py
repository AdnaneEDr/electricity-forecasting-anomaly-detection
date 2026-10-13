import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def inject(y, start, seed):
    """Distinct copy; five disjoint events, positions and amplitudes logged."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    labels = np.zeros(len(y), dtype=bool)
    span = len(y) - start
    specs = []
    for i, kind in enumerate(["pic", "chute", "niveau", "derive", "constant"]):
        pos = start + int((i + .4) * span / 5) + int(rng.integers(0, 12))
        duration = [3, 8, 24, 36, 24][i]
        amplitude = 3 * float(np.std(y[:start]))
        if kind == "pic":
            out[pos:pos + duration] += amplitude
        elif kind == "chute":
            out[pos:pos + duration] *= .05
        elif kind == "niveau":
            out[pos:pos + duration] += amplitude * .7
        elif kind == "derive":
            out[pos:pos + duration] += np.linspace(0, amplitude, duration)
        else:
            out[pos:pos + duration] = y[pos - 1]
        labels[pos:pos + duration] = True
        specs.append({"kind": kind, "position": pos, "duration_h": duration,
                      "amplitude_kw": amplitude, "seed": seed})
    return out, labels, specs


def anomaly_metrics(labels, score, threshold, specs, start):
    flags = score > threshold
    p, r, f, _ = precision_recall_fscore_support(labels[start:], flags[start:],
                                                average="binary", zero_division=0)
    delays = []
    for spec in specs:
        a, d = spec["position"], spec["duration_h"]
        hit = np.flatnonzero(flags[a:a + d])
        if len(hit):
            delays.append(int(hit[0]))
    false = int(np.sum(flags[start:] & ~labels[start:]))
    return {"precision": float(p), "recall": float(r), "F1": float(f),
            "PR_AUC_AP": float(average_precision_score(labels[start:], score[start:])),
            "event_recall": len(delays) / len(specs),
            "mean_delay_h_detected": float(np.mean(delays)) if delays else None,
            "false_positive_hours": false, "events_detected": len(delays)}
