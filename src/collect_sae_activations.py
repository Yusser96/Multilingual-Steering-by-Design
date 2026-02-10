#!/usr/bin/env python
import os
import json
import argparse
import multiprocessing as mp
from collections import defaultdict
from functools import partial  # kept in case you want to extend hooks later

import torch
from torch import Tensor
from tqdm import tqdm

from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from transformer_lens.hook_points import HookPoint


def load_json_dim_dataset(path: str, dim: str | None = None, n: int | None = None) -> list[str]:
    """
    Very simple loader matching your JSON format:

    {
        "good": ["sentence 1", "sentence 2", ...],
        "bad": [...],
        ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dim is None:
        raise ValueError("dim must be provided to select a key from the JSON dataset.")

    texts = data[dim]

    if n is not None:
        texts = texts[-n:]
    return texts




from typing import Dict
from huggingface_hub import HfApi, list_repo_files
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


def get_resid_post_saes_for_release(release: str, width: str) -> Dict[str, str]:
    """
    Look up a release in the PretrainedSAELookup and return
    {sae_id: hook_or_path} for all SAEs whose id or path contains `width`.

    If the release does not exist locally in sae_lens' directory,
    automatically check Hugging Face Hub for a repo with that name.
    """

    directory = get_pretrained_saes_directory()
    target = None

    # -------------------------
    # 1. CHECK LOCAL DIRECTORY
    # -------------------------
    if hasattr(directory, "values"):
        for data in directory.values():
            data_release = getattr(data, "release", None)
            if data_release == release:
                target = data
                break

    if target is not None:
        # --- Extract saes_map ---
        saes_map = getattr(target, "saes_map", None)
        if saes_map is None and isinstance(target, dict):
            saes_map = target.get("saes_map")

        if saes_map is None:
            raise RuntimeError(f"Found release '{release}' but no saes_map exists.")

        # --- Filter ---
        return {
            sae_id: path
            for sae_id, path in saes_map.items()
            if width in sae_id or width in path
        }

    # --------------------------------------------------------
    # 2. NOT FOUND LOCALLY → TRY HuggingFace repo named `release`
    # --------------------------------------------------------
    print(f"Local directory does not contain release '{release}'.")
    print(f"Checking HuggingFace Hub for repo '{release}'...")

    api = HfApi()
    try:
        files = list_repo_files(release)
    except Exception as e:
        raise ValueError(
            f"Release '{release}' not found locally or on HuggingFace.\n"
            f"Error from HuggingFace: {e}"
        )

    # --------------------------------------------------------
    # 3. PARSE FILES → Extract SAE IDs by expected naming pattern
    # --------------------------------------------------------
    # Example HF repo structure:
    #   layer_15/config.json
    #   layer_15/sae_weights.safetensors
    #   layer_22/config.json
    #
    # We interpret each top-level folder as an SAE ID.
    sae_ids = sorted({f.split("/")[0] for f in files if "/" in f})

    
    if not sae_ids:
        raise ValueError(
            f"Found HF repo '{release}', but no SAEs matching width '{width}'. "
            f"Available SAEs: {sae_ids}"
        )

    # Return mapping like the local one:
    #   { "layer_15": "Yusser/LFB-SAES-Llama-3.1-8B/layer_15"  
    return {
        sae_id: sae_id
        for sae_id in sae_ids
        if sae_id not in ['.gitattributes', 'README.md']
    }


# def get_resid_post_saes_for_release(release: str, width: str) -> dict[str, str]:
#     """
#     Look up a release in the PretrainedSAELookup and return
#     {sae_id: hook_or_path} for all SAEs whose id or path contains `width`.
#     """
#     directory = get_pretrained_saes_directory()

#     target = None
#     # directory is a PretrainedSAELookup or dict-like object
#     if hasattr(directory, "values"):
#         for data in directory.values():
#             # data is a ReleaseEntry-like object with .release, .saes_map
#             data_release = getattr(data, "release", None)
#             if data_release == release:
#                 target = data
#                 break

#     if target is None:
#         raise ValueError(
#             f"No release named {release!r} found in SAE directory. "
#             f"Available releases: {[getattr(d, 'release', None) for d in directory.values()]}"
#         )

#     # Get the saes_map (either attribute or dict key, depending on version)
#     saes_map = getattr(target, "saes_map", None)
#     if saes_map is None and isinstance(target, dict):
#         saes_map = target["saes_map"]

#     if saes_map is None:
#         raise RuntimeError(
#             f"Could not find saes_map for release {release!r}; target={target}"
#         )

#     # Filter for SAEs matching the requested width
#     resid_saes = {
#         sae_id: path_or_hook
#         for sae_id, path_or_hook in saes_map.items()
#         if width in sae_id or width in path_or_hook
#     }

#     return resid_saes


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Collect Gemma-3 SAE and resid_post activations with sae_lens"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="HF / TransformerLens model name compatible with HookedSAETransformer",
    )
    parser.add_argument(
        "--sae_release",
        type=str,
        required=True,
        help="SAE Lens release name for this model (e.g. 'gemma-3-4b-res-myrelease')",
    )
    parser.add_argument(
        "--sae_width",
        type=str,
        required=True,
        help="SAE Lens release width for this model (e.g. '16k')",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--dim",
        type=str,
        default=None,
        help="Dimension key in JSON (if dataset is a dict of dim-keys)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save activation stats (will be created if missing)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of examples to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size in number of prompts",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    texts = load_json_dim_dataset(args.dataset_path, dim=args.dim, n=args.max_samples)
    texts = [t["prompt"] for t in texts]
    print(f"Loaded {len(texts)} samples from {args.dataset_path}")

    # Pick device + dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    torch.set_grad_enabled(False)

    # Load model
    print(f"Loading HookedSAETransformer model {args.model_name} on {device}...")
    model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
        args.model_name,
        device=device,
        torch_dtype=dtype,
    )
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"Model has {n_layers} layers, d_model={d_model}")

    # Discover all SAEs in this release matching sae_width
    resid_saes_map = get_resid_post_saes_for_release(args.sae_release, args.sae_width)
    print(f"Found {len(resid_saes_map)} SAEs in release {args.sae_release}")
    for sae_id, path_or_hook in resid_saes_map.items():
        print(f"  SAE id={sae_id!r} -> {path_or_hook!r}")

    ##################################################################
    # 1) Load all SAEs once and group them by (layer, hook_name)
    ##################################################################
    print("Loading SAEs...")
    saes_by_layer_and_hook: dict[tuple[int, str], list[SAE]] = defaultdict(list)
    d_sae = None

    for sae_id, _ in resid_saes_map.items():
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=args.sae_release,
            sae_id=sae_id,
            # device=device,
        )
        sae = sae.to(device)

        hook_name = cfg_dict["hook_name"]          # e.g. "blocks.0.hook_resid_post"
        hook_layer = cfg_dict["hook_layer"]        # integer layer index

        print(
            f"Loaded SAE {sae_id} for hook {hook_name}, "
            f"layer={hook_layer}, d_sae={sae.cfg.d_sae}"
        )

        saes_by_layer_and_hook[(hook_layer, hook_name)].append(sae)

        if d_sae is None:
            d_sae = sae.cfg.d_sae
        else:
            if d_sae != sae.cfg.d_sae:
                raise ValueError(
                    f"Mixed d_sae sizes not supported in this script "
                    f"(got {d_sae} and {sae.cfg.d_sae})."
                )

    if d_sae is None:
        raise RuntimeError("No SAEs loaded; aborting.")

    ##################################################################
    # 2) Initialise accumulators
    ##################################################################
    model_resid_post_over_zero = torch.zeros(
        n_layers, d_model, dtype=torch.float32, device=device
    )
    model_resid_post_over_zero_binary = torch.zeros(
        n_layers, d_model, dtype=torch.int64, device=device
    )

    sae_over_zero = torch.zeros(
        n_layers, d_sae, dtype=torch.float32, device=device
    )
    sae_over_zero_binary = torch.zeros(
        n_layers, d_sae, dtype=torch.int64, device=device
    )

    total_tokens = 0  # number of (batch, seq) positions processed

    ##################################################################
    # 3) Run the model once per batch, then feed resid_post to SAEs
    ##################################################################

    # The above comprehension is a bit awkward; a clearer one:
    # needed_hook_names = list({hook_name for (_, hook_name) in saes_by_layer_and_hook.keys()})

    needed_hook_names = [f"blocks.{layer_idx}.hook_resid_post" for layer_idx in range(n_layers)]

    def batches(iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i: i + batch_size]

    print("Collecting activations with cached resid_post...")
    for batch_texts in tqdm(
        list(batches(texts, args.batch_size)),
        total=(len(texts) + args.batch_size - 1) // args.batch_size,
    ):
        # Tokenize
        toks = model.to_tokens(
            batch_texts,
            prepend_bos= False #model.cfg.default_prepend_bos,
        ).to(device)

        pad_mask = (toks != model.tokenizer.pad_token_id).long()
        pad_mask_expanded = pad_mask.unsqueeze(-1) 

        # Run model once, cache only the hooks we care about
        with torch.no_grad():
            _, cache = model.run_with_cache(
                toks,
                stop_at_layer=n_layers,
                names_filter=lambda name: name in needed_hook_names,
            )

        # batch_size, seq_len = toks.shape
        #print(toks.shape)
        # num_positions = batch_size * seq_len
        total_tokens += int(pad_mask.sum())


        for layer_idx in range(n_layers):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"

            resid: torch.Tensor = cache[hook_name]  # (batch, seq, d_model)
            
            tmp_resid = resid * pad_mask_expanded

            # Update model resid stats for this layer
            model_resid_post_over_zero[layer_idx] += tmp_resid.sum(dim=(0, 1))
            model_resid_post_over_zero_binary[layer_idx] += (tmp_resid > 0).sum(dim=(0, 1))

        # For every (layer, hook_name) where we have at least one SAE:
        for (layer_idx, hook_name), sae_list in saes_by_layer_and_hook.items():
            resid: torch.Tensor = cache[hook_name]  # (batch, seq, d_model)
            
            # tmp_resid = resid * pad_mask_expanded
            # Update model resid stats for this layer
            # model_resid_post_over_zero[layer_idx] += tmp_resid.sum(dim=(0, 1))
            # model_resid_post_over_zero_binary[layer_idx] += (tmp_resid > 0).sum(dim=(0, 1))

            # Run each SAE on this layer's resid_post
            for sae in sae_list:
                # SAE.encode expects resid_post with shape (batch, seq, d_model)
                sae_acts: torch.Tensor = sae.encode(resid)  # (batch, seq, d_sae)

                tmp_sae_acts = sae_acts * pad_mask_expanded

                sae_over_zero[layer_idx] += tmp_sae_acts.sum(dim=(0, 1))
                sae_over_zero_binary[layer_idx] += (tmp_sae_acts > 0).sum(dim=(0, 1))

        # Free cache for this batch
        del cache
        torch.cuda.empty_cache()

    ##################################################################
    # 4) Save stats
    ##################################################################
    output_path = os.path.join(args.output_dir, args.model_name, args.sae_release, args.sae_width)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving activation stats to {output_path}")

    model_resid_post_over_zero_output = dict(
        n=total_tokens,
        over_zero=model_resid_post_over_zero.to("cpu"),
    )
    model_resid_post_over_zero_binary_output = dict(
        n=total_tokens,
        over_zero=model_resid_post_over_zero_binary.to("cpu"),
    )

    sae_over_zero_output = dict(
        n=total_tokens,
        over_zero=sae_over_zero.to("cpu"),
    )
    sae_over_zero_binary_output = dict(
        n=total_tokens,
        over_zero=sae_over_zero_binary.to("cpu"),
    )

    torch.save(
        model_resid_post_over_zero_output,
        f"{output_path}/model_resid_post_activation.{args.dim}",
    )
    torch.save(
        model_resid_post_over_zero_binary_output,
        f"{output_path}/model_resid_post_activation_binary.{args.dim}",
    )

    torch.save(
        sae_over_zero_output,
        f"{output_path}/sae_activation.{args.dim}",
    )
    torch.save(
        sae_over_zero_binary_output,
        f"{output_path}/sae_activation_binary.{args.dim}",
    )
    print("Done.")


if __name__ == "__main__":
    main()
