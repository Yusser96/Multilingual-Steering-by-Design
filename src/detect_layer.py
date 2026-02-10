import torch #
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np



# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/gemma-scope-9b-pt-res-canonical/16k/sae_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/Yusser/LFB-SAES-gemma-2-9b/8x/sae_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/gemma-scope-9b-pt-res-canonical/16k/model_resid_post_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/Yusser/LFB-SAES-gemma-2-9b/8x/sae_activation_vectors_diffmean"


# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/meta-llama/Llama-3.1-8B/llama_scope_lxr_8x/8x/sae_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/meta-llama/Llama-3.1-8B/Yusser/LFB-SAES-Llama-3.1-8B/8x/sae_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/Yusser/LFB-SAES-gemma-2-9b/8x/model_resid_post_activation_vectors_diffmean"


# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/google/gemma-2-9b/Yusser/EN-SAES-gemma-2-9b_512_2100000000/16k/model_resid_post_activation_vectors_diffmean"

# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/meta-llama/Llama-3.1-8B/Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000/8x/model_resid_post_activation_vectors_diffmean"
path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/meta-llama/Llama-3.1-8B/Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000/8x/sae_activation_vectors_diffmean"
# path = "/netscratch/ghussin/2025/Multilingual_By_Design/vectors/meta-llama/Llama-3.1-8B/Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000/8x/sae_activation_vectors_diffmean"

all_svectors = torch.load(path,weights_only=False)

n_layers = len(all_svectors)

print("n_layers: ",n_layers)

languages = ["bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"]

alpha = 100.0

# all_svectors = torch.load("/netscratch/ghussin/2025/SemEval2026/f_flores200_dataset_low_res/identify/aya-expanse-8b/aya8b/myapproach_resid_vectors_diffmean",weights_only=False)
# languages = ["en","de","ru","zh"]

def plot_corr(svectors,save_path=None):
    # Convert list to NumPy array
    data = np.array(list(svectors.values()))

    # If each vector is a row, use rowvar=True (default); otherwise, transpose
    correlation_matrix = np.corrcoef(data)


    # ======================= #
    return correlation_matrix

    #print(correlation_matrix)

    # Plot heatmap with labels
    plt.figure(figsize=(16, 15))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=svectors.keys(),
        yticklabels=svectors.keys()
    )
    plt.title("Correlation Matrix")
    plt.xlabel("Vectors")
    plt.ylabel("Vectors")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs("/".join(save_path.split("/")[:-1]),exist_ok=True)
        plt.savefig(save_path)
    #plt.show()

    return correlation_matrix


all_correlation_matrix = []
for layer_idx,svectors in enumerate(all_svectors):
    print("layer",layer_idx)
    if svectors is not None:
        new_svectors = {}
        for lang in languages:
            new_svectors[lang] = svectors[lang] * alpha
        correlation_matrix = plot_corr(new_svectors, save_path=f"llama_scope/{layer_idx}.png")
        all_correlation_matrix.append(correlation_matrix)
    else:
        all_correlation_matrix.append(None)



import numpy as np
from sklearn.metrics import silhouette_score




# 4) First principal component variance ratio
def multilinguality_first_pc_ratio(corr: np.ndarray, eps: float = 1e-12) -> float:
    """
    Fraction of variance explained by the top eigenvalue of corr.
    Higher = stronger shared (language-agnostic) direction => stronger multilingual factor.
    """
    corr = np.asarray(corr)
    assert corr.ndim == 2 and corr.shape[0] == corr.shape[1], "corr must be square"
    corr_sym = (corr + corr.T) / 2.0
    evals = np.linalg.eigvalsh(corr_sym)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    total = evals.sum() + eps
    return float(evals.max() / total)



def separability_anti_shared_component(corr: np.ndarray):
    """
    1 - (variance explained by the first eigenvalue).
    Higher = less dominated by a single shared factor (better separability).
    """
    C = (np.asarray(corr) + np.asarray(corr).T) / 2.0
    evals = np.linalg.eigvalsh(C)
    evals = np.clip(evals, 0.0, None)
    if np.isclose(evals.sum(), 0.0):
        return np.nan
    
    return float(1.0 - (evals.max() / evals.sum()))


from collections import defaultdict
res = defaultdict(list)
for layer_idx,correlation_matrix in enumerate(all_correlation_matrix):
    print("layer",layer_idx)
    try:
        if correlation_matrix is not None and layer_idx < n_layers-1:


            score = multilinguality_first_pc_ratio(correlation_matrix)
            print("multilinguality_first_pc_ratio:",score)
            res["first_pc_ratio"].append(score)


            score = separability_anti_shared_component(correlation_matrix)
            print("separability_anti_shared_component:",score)
            res["separability_anti_shared_component"].append(score)

            print("-"*30)
    except:
        res["first_pc_ratio"].append(1)
        res["separability_anti_shared_component"].append(0)








import numpy as np
import pandas as pd
from scipy.stats import zscore

def pick_layers(metrics, top_k=4):
    # metrics: dict with keys 'inter_intra_ratio', 'first_pc_ratio', 'spectrum_entropy'
    df = pd.DataFrame(metrics)
    df['layer'] = np.arange(len(df))  # keep track of layer ids

    Z = pd.DataFrame({
        'first_pc_ratio':    zscore(df['first_pc_ratio']),
        'separability_anti_shared_component': zscore(df['separability_anti_shared_component']),
    })
    composite = Z['separability_anti_shared_component']+ Z['first_pc_ratio'] #+ Z['inter_intra_ratio'] + Z['first_pc_ratio'] + Z['cross_language_alignment'] - Z['spectrum_entropy']
    ranks = composite.rank(ascending=False, method='first').astype(int)
    df_out = df.copy()
    df_out['composite'] = composite
    df_out['rank'] = ranks
    return df_out.sort_values('rank').head(top_k), df_out

temp, df_out = pick_layers(res, top_k=6)

print(temp)



import numpy as np

def crossing_indices(y1, y2):
    """
    Return fractional indices i* where y1 and y2 intersect between samples i and i+1.
    If they are exactly equal at a sample, the integer index is returned.
    Handles flat 'equal' plateaus by returning the midpoint index of the plateau
    when there's a sign change across it.

    Parameters
    ----------
    y1, y2 : array-like of shape (n,)

    Returns
    -------
    idx : np.ndarray of shape (k,)
        Fractional indices where the crossing occurs. (Use idx[0] for the first.)
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if y1.shape != y2.shape or y1.ndim != 1 or len(y1) < 2:
        raise ValueError("y1 and y2 must be 1D arrays of the same length (>=2)")

    d = y1 - y2
    out = []
    n = len(d)
    i = 0
    while i < n - 1:
        if np.isnan(d[i]) or np.isnan(d[i+1]):
            i += 1
            continue

        # plateau of exact equality
        if d[i] == 0 and d[i+1] == 0:
            j = i + 1
            while j < n and d[j] == 0:
                j += 1
            # if signs differ across the plateau, count one crossing at the plateau midpoint
            if i > 0 and j < n and d[i-1] * d[j] < 0:
                out.append((i + j - 1) / 2)  # midpoint index of the equal region
            i = j
            continue

        # exact touch at sample i
        if d[i] == 0:
            out.append(float(i))

        # sign change between i and i+1 -> crossing inside segment
        elif d[i] * d[i+1] < 0:
            t = -d[i] / (d[i+1] - d[i])  # fraction in (0,1)
            out.append(i + t)

        i += 1

    return np.array(out)



idx = crossing_indices(y1=df_out.separability_anti_shared_component, y2=df_out.first_pc_ratio)
print("intersection:",idx)

# ax = df_out.first_pc_ratio.plot.line()

# df_out.separability_anti_shared_component.plot.line()


# ax.legend(
#     labels=["Multilinguality", "Separability"],
# #     loc="lower center",   # bottom center inside
# #     bbox_to_anchor=(0.5, 0.001)  # tweak vertical position
# )

# ax.figure.savefig(f'sae_layer_selection2.png')


import numpy as np
import matplotlib.pyplot as plt

def plot_layer_selection_mechanism(
    df,
    save_path,
    multilinguality_col="first_pc_ratio",
    separability_col="separability_anti_shared_component",
    figsize=(6, 4),
    dpi=300,
):
    """
    Plot layerwise multilinguality and separability curves with their
    intersection highlighted as the predicted steering depth.

    Args:
        df (pd.DataFrame): DataFrame indexed by layer, containing
            multilinguality and separability columns.
        save_path (str): Path to save the figure.
        multilinguality_col (str): Column for shared cross-lingual alignment.
        separability_col (str): Column for language separability.
        figsize (tuple): Figure size.
        dpi (int): Resolution for saved figure.
    """

    # Compute intersection layer (min absolute difference)
    intersection_layer = np.argmin(
        np.abs(df[multilinguality_col] - df[separability_col])
    )

    print("intersection_layer:",intersection_layer)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves
    df[multilinguality_col].plot(
        ax=ax,
        linewidth=2,
        label="Multilinguality (shared alignment)",
    )

    df[separability_col].plot(
        ax=ax,
        linewidth=2,
        label="Separability (language-specific)",
    )

    # Highlight intersection
    ax.axvline(
        intersection_layer,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.text(
        intersection_layer,
        ax.get_ylim()[1] * 0.98,
        "Intersection\n(predicted steering depth)",
        ha="right",
        va="top",
        fontsize=9,
    )

    # Regime annotations
    # ax.text(
    #     0.02,
    #     0.05,
    #     "Alignment-dominated",
    #     transform=ax.transAxes,
    #     fontsize=9,
    # )

    # ax.text(
    #     0.68,
    #     0.05,
    #     "Separability-dominated",
    #     transform=ax.transAxes,
    #     fontsize=9,
    # )

    # Labels and styling
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized score")
    ax.legend(frameon=False)
    ax.set_title("Layerwise balance of multilinguality and separability")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)



model_name = path.split("vectors/")[1].split("/")[1]
location = path.split("vectors/")[1].split("/")[-1].replace("_activation_vectors_diffmean","")
plot_layer_selection_mechanism(df_out,f'layer_selection_{model_name}_{location}.png')