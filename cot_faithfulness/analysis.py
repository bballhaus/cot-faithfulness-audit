import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def two_proportion_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else float("nan")
    pval = math.erfc(abs(z) / math.sqrt(2))
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z,
            "p_value": pval, "n1": n1, "n2": n2}


def bootstrap_diff(a, b, n_boot=10000, seed=0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
    return {"diff": float(a.mean() - b.mean()),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d < 0).mean(), (d > 0).mean()))}


def bootstrap_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(v[rng.integers(0, len(v), len(v))])
    return {
        "point": float(statistic(v)),
        "lo": float(np.percentile(boots, 100 * alpha / 2)),
        "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }


def plot_layer_heatmap(probs, target_names, title=None, ax=None, vmin=0.0, vmax=1.0):
    arr = probs.numpy() if hasattr(probs, "numpy") else np.asarray(probs)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        arr,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        xticklabels=target_names,
        yticklabels=False,
        ax=ax,
        cbar_kws={"label": "probability"},
    )
    ax.set_xlabel("class")
    ax.set_ylabel("layer (deepest at bottom)")
    if title:
        ax.set_title(title)
    return ax


def plot_position_layer_heatmap(probs_2d, title=None, ax=None, vmin=0.0, vmax=1.0):
    arr = probs_2d.numpy() if hasattr(probs_2d, "numpy") else np.asarray(probs_2d)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(arr, cmap="viridis", vmin=vmin, vmax=vmax, ax=ax,
                cbar_kws={"label": "p(correct)"})
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    if title:
        ax.set_title(title)
    return ax


def commitment_summary(records):
    df = pd.DataFrame(records)
    out = {
        "n": len(df),
        "mean_commitment_layer": df["commitment_layer"].replace(-1, np.nan).mean(),
        "median_commitment_layer": df["commitment_layer"].replace(-1, np.nan).median(),
        "frac_committed": (df["commitment_layer"] >= 0).mean(),
        "accuracy": df["correct"].mean(),
    }
    if "correct" in df.columns:
        for label, group in df.groupby("correct"):
            out[f"mean_layer_correct={label}"] = group["commitment_layer"].replace(-1, np.nan).mean()
    return out


def plot_commitment_histogram(records, n_layers, ax=None):
    df = pd.DataFrame(records)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df[df["commitment_layer"] >= 0], x="commitment_layer",
                 hue="correct", multiple="stack", bins=n_layers, ax=ax)
    ax.set_xlabel("commitment layer (τ=0.8)")
    ax.set_ylabel("count")
    ax.set_title("Layer at which p(answer) > 0.8")
    return ax


def patching_summary(records, with_ci=False):
    df = pd.DataFrame(records)
    out = {
        "n": len(df),
        "flip_rate": df["flipped"].mean(),
        "mean_logit_drop": df["logit_diff_drop"].mean(),
        "accuracy_clean": df["correct_clean"].mean() if "correct_clean" in df else None,
        "accuracy_corrupted": df["correct_corrupted"].mean() if "correct_corrupted" in df else None,
    }
    if "patch_recovers_clean" in df:
        out["patch_recovery_rate"] = df["patch_recovers_clean"].mean()
    if with_ci:
        out["flip_rate_ci"] = bootstrap_ci(df["flipped"].astype(float))
        out["mean_logit_drop_ci"] = bootstrap_ci(df["logit_diff_drop"])
        if "patch_recovers_clean" in df:
            out["patch_recovery_rate_ci"] = bootstrap_ci(df["patch_recovers_clean"].astype(float))
    return out


def commitment_summary_ci(records):
    df = pd.DataFrame(records)
    cl = df["commitment_layer"].replace(-1, np.nan)
    out = commitment_summary(records)
    out["mean_commitment_layer_ci"] = bootstrap_ci(cl)
    out["frac_committed_ci"] = bootstrap_ci((df["commitment_layer"] >= 0).astype(float))
    out["accuracy_ci"] = bootstrap_ci(df["correct"].astype(float))
    return out


def faithfulness_accuracy(records):
    # Proposal eval metric 3: are causally-influential (flipped-under-corruption)
    # CoTs also more accurate? Positive r => genuine reasoning tracks correctness.
    df = pd.DataFrame(records)
    fl = df["flipped"].astype(bool)
    acc = df["correct_clean"].astype(float)
    n1, n0 = int(fl.sum()), int((~fl).sum())
    out = {
        "n": int(len(df)),
        "acc_when_flipped": float(acc[fl].mean()) if n1 else float("nan"),
        "acc_when_not_flipped": float(acc[~fl].mean()) if n0 else float("nan"),
        "n_flipped": n1,
        "n_not_flipped": n0,
    }
    if n1 and n0:
        out["point_biserial_r"] = float(np.corrcoef(fl.astype(float), acc)[0, 1])
        out["two_proportion_z"] = two_proportion_z(
            int(acc[fl].sum()), n1, int(acc[~fl].sum()), n0)
    return out


def threshold_sweep(all_probs, labels, taus=(0.5, 0.7, 0.8, 0.9)):
    arr = np.asarray(all_probs)
    labels = np.asarray(labels)
    n, n_layers, _ = arr.shape
    rows = []
    for tau in taus:
        commit = np.full(n, -1)
        for i in range(n):
            hits = np.where(arr[i, :, labels[i]] > tau)[0]
            if len(hits):
                commit[i] = hits[0]
        valid = commit[commit >= 0].astype(float)
        rows.append({
            "tau": tau,
            "frac_committed": float((commit >= 0).mean()),
            "mean_layer": float(valid.mean()) if len(valid) else float("nan"),
            "median_layer": float(np.median(valid)) if len(valid) else float("nan"),
            "mean_depth_frac": float(valid.mean() / (n_layers - 1)) if len(valid) else float("nan"),
        })
    return pd.DataFrame(rows)


def qualitative_extremes(records, k=25, text_key="cot_text"):
    df = pd.DataFrame(records)
    committed = df[df["commitment_layer"] >= 0].sort_values("commitment_layer")
    cols = [c for c in ["idx", "commitment_layer", "correct", "pred", "label_text", text_key]
            if c in committed.columns]
    early = committed.head(k)[cols].assign(group="early")
    late = committed.tail(k)[cols].assign(group="late")
    return pd.concat([early, late], ignore_index=True)


def plot_commitment_compare(records_no_cot, records_with_cot, n_layers, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    for recs, name in [(records_no_cot, "no-CoT"), (records_with_cot, "with-CoT")]:
        cl = pd.DataFrame(recs)["commitment_layer"]
        cl = cl[cl >= 0]
        sns.histplot(cl, bins=n_layers, stat="density", element="step",
                     fill=False, label=name, ax=ax)
    ax.axvline(0, color="grey", alpha=0.3)
    ax.set_xlabel("commitment layer (τ=0.8)")
    ax.set_ylabel("density")
    ax.set_title("Commitment depth: no-CoT vs with-CoT")
    ax.legend()
    return ax
