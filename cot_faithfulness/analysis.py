import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def patching_summary(records):
    df = pd.DataFrame(records)
    return {
        "n": len(df),
        "flip_rate": df["flipped"].mean(),
        "mean_logit_drop": df["logit_diff_drop"].mean(),
        "accuracy_clean": df["correct_clean"].mean() if "correct_clean" in df else None,
        "accuracy_corrupted": df["correct_corrupted"].mean() if "correct_corrupted" in df else None,
    }
