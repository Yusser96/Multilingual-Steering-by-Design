#!/usr/bin/env python

import os
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing as mp

import argparse

import torch
from torch import Tensor

from sae_lens import SAE
from vllm import LLM, SamplingParams
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

import json
from vllm_hooks import apply_steervec_intervention, restore_original_forwards_saes


from tasks.flores import (get_prompts, init, compute_comet_score, lang_to_code,
detect_language, compute_bleu_score, post_process_text, post_process_text)



def main():
    parser = argparse.ArgumentParser(
        description="Gemma-3 SAE steering using resid_post and SAE latent space"
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
    # parser.add_argument(
    #     "--dim",
    #     type=str,
    #     default=None,
    #     help="Dimension key in JSON (if dataset is a dict of dim-keys)",
    # )
    parser.add_argument(
        "--dims",
        nargs='+',
        default=None,
        required=True,
        help="Dimensions keys",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to activation_stats.pt for target corpus",
    )
    parser.add_argument(
        "--layers",
        nargs='+',
        type=int,
        default=14,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling for resid_post steering vector",
    )
    parser.add_argument(
        "--use_sae", 
        action='store_true', 
        help="if used it will steer in sparse space"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="flores, belebele",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save activation stats (will be created if missing)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    torch.set_grad_enabled(False)

    # Load model
    print(f"Loading model {args.model_name}...")
    # model: HookedSAETransformer = HookedSAETransformer.from_pretrained(
    #     args.model_name,
    #     device=device,
    #     torch_dtype=dtype,
    # )


    langid_model, comet_model = init(device)

    
    
    

    # Load stats and build steering vectors


    for source_lang in args.dims:
        print("-"*20)
        print("source_lang:",source_lang)
        print("-"*20)

        def steer_exp(steer_lang):
            for layer in args.layers:

                out_path = os.path.join(args.output_dir, str(args.alpha), args.model_name, str(layer), args.sae_release, args.sae_width)
                output_path = os.path.join(out_path,"eval",args.task_name)
                os.makedirs(output_path, exist_ok=True)

                output_path = os.path.join(output_path,f"{source_lang}-{steer_lang}.json")

                # if os.path.exists(output_path):
                #     print(f"file exists: {output_path}")
                #     continue

                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # if "avg_comet" in data["mod"]:
                    #     continue

                    if "avg_comet_success" in data["base"]:
                        continue

                for k in ["mod", "base"]:
                    res = data[k]

                    sources = [r["source"] for r in res["results"]]
                    steer_ref = [r["reference_translation"] for r in res["results"]]
                    preds = [r["model_translation_post"] for r in res["results"]]
                    avg_comet = compute_comet_score(comet_model, sources, steer_ref , preds ,batch_size=args.batch_size,device=device)

                    data[k]["avg_comet"] = avg_comet



                    sources = [r["source"] for r in res["results"] if r["force_success"]]
                    steer_ref = [r["reference_translation"] for r in res["results"] if r["force_success"]]
                    preds = [r["model_translation_post"] for r in res["results"] if r["force_success"]]
                    if preds:
                        avg_comet_success = compute_comet_score(comet_model, sources, steer_ref , preds ,batch_size=args.batch_size,device=device)

                        data[k]["avg_comet_success"] = avg_comet_success

                        bleu_scores = [r["bleu_score"] for r in res["results"] if r["force_success"]]

                        data[k]["avg_bleu_success"] = sum(bleu_scores)/len(bleu_scores)
                    else:
                        data[k]["avg_comet_success"] = 0
                        data[k]["avg_bleu_success"] = 0


                    


                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        
        # steer_lang = source_lang
        # steer_exp(steer_lang)

        for steer_lang in args.dims:
            # if source_lang == steer_lang:
            #     continue
                
            steer_exp(steer_lang)

    


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()
