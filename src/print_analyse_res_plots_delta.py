import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

_score1 = "langid"
_score2 = "bleu"
_score3 = "comet"

# ---------------------------
# Parsing
# ---------------------------
def parse_results(df, language):
    records = []
    for sae in df.columns:
        if sae in ["layer", "prompt"]:
            continue

        for _, row in df.iterrows():
            langid, bleu, comet = row[sae].split("/")
            records.append({
                "language": language,
                "sae": "-".join(sae.replace("_512_2100000000","").replace("-SAES","").split("-")[:2]),
                "layer": int(row["layer"]),
                _score1: float(langid),
                _score2: float(bleu),
                _score3: float(comet),
            })
    return pd.DataFrame(records)


def load_all_results(input_dir, model, type_):
    all_results = []
    for fname in os.listdir(input_dir):
        if not fname.endswith(".tsv"):
            continue
        if not fname.startswith(model):
            continue
        if f"_{type_}." not in fname:
            continue

        language = fname.split(".")[-2]
        df = pd.read_csv(os.path.join(input_dir, fname), sep="\t")
        all_results.append(parse_results(df, language))

    if not all_results:
        raise RuntimeError("No valid TSV files found.")

    return pd.concat(all_results, ignore_index=True)


# ---------------------------
# Baseline logic
# ---------------------------
def detect_scope_baselines(df):
    baselines = sorted(
        sae for sae in df["sae"].unique()
        if "scope" in sae.lower()
    )
    if not baselines:
        raise ValueError("No SAE containing 'scope' found for baseline.")
    return baselines



def add_delta_from_scope(df, metric):
    baselines = detect_scope_baselines(df)

    baseline_df = (
        df[df["sae"].isin(baselines)]
        .groupby(["language", "layer"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"{metric}_baseline"})
    )

    df = df.merge(
        baseline_df,
        on=["language", "layer"],
        how="left"
    )

    df[f"{metric}_delta"] = df[metric] - df[f"{metric}_baseline"]

    # Drop baseline column immediately to avoid collisions
    df = df.drop(columns=[f"{metric}_baseline"])

    return df


# ---------------------------
# Plots
# ---------------------------
def save_delta_heatmap(df, metric, out_dir):
    pivot = (
        df.groupby(["sae", "layer"])[f"{metric}_delta"]
        .mean()
        .reset_index()
        .pivot(index="sae", columns="layer", values=f"{metric}_delta")
    )

    vmax = pivot.abs().max().max()

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax
    )
    plt.title(f"{metric.upper()} Δ vs SCOPE Baseline")
    plt.xlabel("Layer")
    plt.ylabel("SAE Variant")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"heatmap_{metric}_delta_scope.png"),
        dpi=200
    )
    plt.close()


def save_layer_trend(df, metric, out_dir):
    layer_avg = (
        df.groupby("layer")[f"{metric}_delta"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 4))
    # plt.plot(
    #     layer_avg["layer"],
    #     layer_avg[f"{metric}_delta"],
    #     marker="o"
    # )
    sns.pointplot(
        data=layer_avg,
        x="layer",
        y=f"{metric}_delta",
        errorbar=None,
        capsize=0.1
    )
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.title(f"Average {metric.upper()} Δ vs Layer",fontsize=22)
    plt.xlabel("")
    plt.xticks(fontsize=22)
    plt.ylabel(f"")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"layer_trend_{metric}_delta.png"),
        dpi=200
    )
    plt.close()


def save_sae_pointplot(df, metric, out_dir):
    plt.figure(figsize=(10, 5))
    sns.pointplot(
        data=df,
        x="sae",
        y=f"{metric}_delta",
        errorbar="ci",
        capsize=0.1
    )
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xticks(rotation=15,fontsize=16)
    plt.title(f"{metric.upper()} Δ vs SCOPE (Mean ± CI)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"sae_point_{metric}_delta.png"),
        dpi=200
    )
    plt.close()


def save_sae_pointplot_max(df, metric, out_dir):
    layer_avg = (
        df.groupby(["sae", "layer"])[f"{metric}_delta"]
        .mean()
        .reset_index()
    )

    max_per_sae = (
        layer_avg
        .loc[layer_avg.groupby("sae")[f"{metric}_delta"].idxmax()]
        .reset_index(drop=True)
    )

    plt.figure(figsize=(6, 3.8))
    ax = sns.pointplot(
        data=max_per_sae,
        x="sae",
        y=f"{metric}_delta",
        errorbar=None
    )
    y_range = (
        max_per_sae[f"{metric}_delta"].max()
        - max_per_sae[f"{metric}_delta"].min()
    )

    for i, row in max_per_sae.iterrows():
        y_offset = 0.15 * y_range
        if row[f"{metric}_delta"] == 0 :
            va = "bottom" 
        else:
            y_offset *= -1
            va = "top"
        ax.text(
            i,
            row[f"{metric}_delta"]+y_offset,
            f"L{row['layer']}",
            ha="center",
            va=va,
            fontsize=16
        )

    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xticks(rotation=15,fontsize=16)
    plt.title(f"{metric.upper()} Δ vs SCOPE (Max, Layer Selected)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"sae_point_{metric}_delta_max.png"),
        dpi=200
    )
    plt.close()


def save_language_variance(df, metric, out_dir):
    lang_var = (
        df.groupby("language")[f"{metric}_delta"]
        .std()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 4))
    lang_var.plot(kind="bar")
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.title(f"{metric.upper()} Δ Variance Across Languages")
    plt.ylabel("Std Dev")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"language_variance_{metric}_delta.png"),
        dpi=200
    )
    plt.close()




# ---------------------------
# per-lang
# ---------------------------

def save_language_layer_sae_heatmaps(df, metric, out_dir, max_cols=4):
    """
    Language × Layer heatmaps faceted by SAE.
    - SCOPE baseline SAEs excluded
    - Dedicated GridSpec column for colorbar (never overlaps)
    """

    import numpy as np
    from matplotlib.gridspec import GridSpec

    # -------------------------------------------------
    # Exclude SCOPE baselines
    # -------------------------------------------------
    scope_saes = detect_scope_baselines(df)
    plot_df = df[~df["sae"].isin(scope_saes)]

    saes = sorted(plot_df["sae"].unique())
    languages = sorted(plot_df["language"].unique())
    layers = sorted(plot_df["layer"].unique())

    if not saes:
        raise ValueError("No non-SCOPE SAEs left to plot.")

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    n_sae = len(saes)
    n_cols = min(max_cols, n_sae)
    n_rows = int(np.ceil(n_sae / n_cols))

    height_per_language = 0.35
    fig_height = max(3.5, height_per_language * len(languages) * n_rows)
    fig_width = 4.5 * n_cols + 0.8   # extra space for colorbar

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(
        n_rows,
        n_cols + 1,                 # last column = colorbar
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.25
    )

    # -------------------------------------------------
    # Global color scale
    # -------------------------------------------------
    vmax = plot_df[f"{metric}_delta"].abs().max()

    axes = []

    # -------------------------------------------------
    # Plot per SAE
    # -------------------------------------------------
    for idx, sae in enumerate(saes):
        r = idx // n_cols
        c = idx % n_cols

        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        sub = plot_df[plot_df["sae"] == sae]

        pivot = (
            sub
            .groupby(["language", "layer"])[f"{metric}_delta"]
            .mean()
            .reset_index()
            .pivot(index="language", columns="layer", values=f"{metric}_delta")
            .reindex(index=languages, columns=layers)
        )

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            cbar=False,
            linewidths=0.3
        )

        ax.set_title(sae, fontsize=14)
        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14, rotation=0)

    # -------------------------------------------------
    # Colorbar (dedicated column)
    # -------------------------------------------------
    cax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(-vmax, vmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=f"Δ {metric}")

    # -------------------------------------------------
    # Title
    # -------------------------------------------------
    fig.suptitle(
        f"{metric.upper()} Δ vs SCOPE",
        # "(Language × Layer, Faceted by SAE)",
        fontsize=14,
        y=1.0
    )

    plt.savefig(
        os.path.join(out_dir, f"language_layer_sae_{metric}_delta.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()




# ---------------------------
# Best configs
# ---------------------------
def save_best_configs(df, out_dir):
    rows = []
    for metric in [_score1, _score2, _score3]:
        idx = df[f"{metric}_delta"].idxmax()
        row = df.loc[idx]
        rows.append({
            "metric": metric,
            "language": row["language"],
            "sae": row["sae"],
            "layer": row["layer"],
            "delta": row[f"{metric}_delta"],
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "best_delta_configs.csv"),
        index=False
    )


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    global _score1
    global _score2
    global _score3

    if args.task =="cross_sum":
        _score1 = "LangId"
        _score2 = "RougeL"
        _score3 = "LaSE"
    else:
        _score1 = "LangId"
        _score2 = "SpBLEU"
        _score3 = "COMET"

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all_results(
        args.input_dir,
        args.model,
        args.type
    )

    for metric in [_score1, _score2, _score3]:
        df = add_delta_from_scope(df, metric)

    for metric in [_score1, _score2, _score3]:
        save_delta_heatmap(df, metric, args.output_dir)
        save_layer_trend(df, metric, args.output_dir)
        save_sae_pointplot(df, metric, args.output_dir)
        save_sae_pointplot_max(df, metric, args.output_dir)
        save_language_variance(df, metric, args.output_dir)


        save_language_layer_sae_heatmaps(
            df,
            metric,
            args.output_dir
        )



    # save_best_configs(df, args.output_dir)


if __name__ == "__main__":
    main()
