#!/usr/bin/env python

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA_VLLM_V1"

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


from tasks.cross_sum import (get_prompts, init, lang_to_code,
detect_language, compute_rouge_score, post_process_text)




def run_task(model, langid_model, rouge_model, task_name, batch_size, source_lang, steer_lang, cache_tag="base",use_beam_search=False):

    prompts = get_prompts(("en",source_lang))

    # print("prompts",len(prompts))

    tokenizer = model.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id
    

    if use_beam_search:
        sampling_params = SamplingParams(
            temperature=1e-6, 
            repetition_penalty=1.1, 
            max_tokens=128,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
            # stop=["\n\n", "Summary", "Summarize", "Summary:", "article","\n"],
            skip_special_tokens=True,

            n=5,                # number of candidates (≈ beams)
            logprobs=1          # enables scoring
        )
    else:
        sampling_params = SamplingParams(
            temperature=0, 
            repetition_penalty=1.1, 
            max_tokens=128,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
            # stop=["\n\n", "Translation", "Translate", "Source:", "Target:","\n"],
            skip_special_tokens=True
        )

    outputs = model.generate([{"prompt":p} for p,r,s in prompts], sampling_params)
    pred = responses = [output.outputs[0].text.strip() for output in outputs]

    # print("pred",len(pred))

    steer_ref = [r for p,r,s in get_prompts(("en",steer_lang))]

    # print("steer_ref",len(steer_ref))

    sources = [s for p,r,s in prompts]

    # print("sources",len(sources))

    results = []
    for i, (pred, ref,src) in enumerate(zip(pred, steer_ref,sources)):
        pred_post = post_process_text(pred)

        detected_lang = detect_language(langid_model,pred_post)
        # print(detected_lang)

        results.append({
            "id": i,
            "prompt":prompts[i][0],
            "source":src,
            "reference_translation": ref,
            "model_translation": pred,
            "model_translation_post": pred_post,
            # "rouge_score": rouge_score,
            # "comet_score":comet_score,
            "force_success": detected_lang == lang_to_code(steer_lang),
        })

    # rouge_scores = [res["rouge_score"] for res in results]
    # avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    langid_scores = [res["force_success"] for res in results]
    langid_acc = sum(langid_scores) / len(langid_scores) if langid_scores else 0.0


    avg_rouge_scores = compute_rouge_score(rouge_model, steer_ref, [item["model_translation_post"] for item in results])


    


    # avg_comet = compute_comet_score(comet_model, sources, steer_ref , [item["model_translation_post"] for item in results],batch_size=batch_size,device=device)


    eval_res = {
        # "avg_comet":avg_comet,
        "langid_acc":langid_acc,
        # "avg_rouge":avg_rouge,
        "results":results
        }
    eval_res.update(avg_rouge_scores)
    return eval_res



def load_steer_vec(vectors_path, dim, layer, use_sae):

    file_name = "model_resid_post_activation"
    if use_sae:
        file_name="sae_activation" 
    
    all_svectors = torch.load(f'{vectors_path}/{file_name}_vectors_diffmean', weights_only=False)

    return torch.Tensor(all_svectors[layer][dim]).to(device)



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
        "--use_beam_search", 
        action='store_true', 
        help="if used it will use beam decoding"
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
        default="float32",
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
    model = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True, dtype=dtype)


    langid_model, comet_model = init(device)

    

    # Load stats and build steering vectors
    vectors_path = os.path.join(args.dataset_path, args.model_name, args.sae_release, args.sae_width)


    for source_lang in args.dims:
        print("source_lang:",source_lang)
        try:
            base_res = run_task(model, langid_model, comet_model, args.task_name, args.batch_size, source_lang, source_lang, cache_tag="base",use_beam_search=args.use_beam_search)
        except Exception as e:
            print(f"*** {source_lang}: {e}")
            continue
            

        def steer_exp(steer_lang):
            for layer in args.layers:

                out_path = os.path.join(args.output_dir, str(args.alpha), args.model_name, str(layer), args.sae_release, args.sae_width)
                output_path = os.path.join(out_path,"eval",args.task_name)
                os.makedirs(output_path, exist_ok=True)

                output_path = os.path.join(output_path,f"{source_lang}-{steer_lang}.json")

                if os.path.exists(output_path):
                    print(f"file exists: {output_path}")
                    continue

                if args.use_sae:
                    # Discover all SAEs in this release matching sae_width
                    resid_saes_map = get_resid_post_saes_for_release(args.sae_release, args.sae_width)
                    print(f"Found {len(resid_saes_map)} SAEs in release {args.sae_release}")
                    for sae_id, path_or_hook in resid_saes_map.items():
                        #print(f"  SAE id={sae_id!r} -> {path_or_hook!r}")
                        if f"layer_{layer}".lower() in sae_id.lower() or f"L{layer}".lower() in sae_id.lower() or f"blocks.{layer}".lower() in sae_id.lower():
                            print(f"  SAE id={sae_id!r} -> {path_or_hook!r}")
                            sae, cfg_dict, sparsity = SAE.from_pretrained(
                                release=args.sae_release,
                                sae_id=sae_id,
                                # device=device,
                            )
                            sae = sae.to(device)
                            break
                


                steer_vec = load_steer_vec(vectors_path, steer_lang, layer, args.use_sae)

                # Register steering hooks
                if args.use_sae:
                    original_forwards = apply_steervec_intervention(args.model_name, model, layer, steer_vec, alpha=args.alpha, sae=sae)
                else:
                    original_forwards = apply_steervec_intervention(args.model_name, model, layer, steer_vec, alpha=args.alpha, sae=None)
                
                
                # run_inference(model, [args.prompt],args.max_new_tokens)
                # run_inference_mcq(model, [args.prompt])
                cache_tag = f"{source_lang}_{steer_lang}_L{layer}_alpha{args.alpha}"
                res = run_task(model, langid_model, comet_model, args.task_name, args.batch_size, source_lang, steer_lang, cache_tag=cache_tag,use_beam_search=args.use_beam_search)
                

                # Clean up hooks / SAEs (optional in a one-off script but good practice)
                restore_original_forwards_saes(model, original_forwards)



                

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"mod":res, "base":base_res}, f, indent=4, ensure_ascii=False)

        
        # steer_lang = source_lang
        # steer_exp(steer_lang)

        for steer_lang in args.dims:
            # if source_lang == steer_lang:
            #     continue
                
            steer_exp(steer_lang)

    


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
