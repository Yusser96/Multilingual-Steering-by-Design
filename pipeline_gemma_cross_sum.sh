#!/bin/sh

export PYTHONPATH=/home/Multilingual_By_Design:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd /home/Multilingual_By_Design

export HF_TOKEN="<HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0





#python3 src/list_saes.py

MODEL_ID="google/gemma-2-9b" # "google/gemma-2-2b"


# languages=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no")
languages=("es" "ru" "ar" "hi" "tr")

SAE_ID=$1
SAE_WIDTH="16k"

# SAE_ID="gemma-scope-9b-pt-res-canonical"
# SAE_ID="Yusser/EN-SAES-gemma-2-9b_512_2100000000" #"Yusser/EN-SAES-gemma-2-9b"
# SAE_ID="Yusser/MULTI21-SAES-gemma-2-9b_512_2100000000" #"Yusser/LFB-SAES-gemma-2-9b"

# SAE_WIDTH="8x"

echo $SAE_ID

python3 src/vllm_cross_sum.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "cross_sum-all_langs2-saes" \
  --dims "${languages[@]}" \
  --layers 6 14 23 32 40 \
  --alpha 100.0 \
  --batch_size 16 \
  --max_new_tokens 1 \
  --task_name "cross_sum" \
  --use_sae



python3 src/ src/eval_lase.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "cross_sum-all_langs2-saes" \
  --dims "${languages[@]}" \
  --layers 6 14 23 32 40 \
  --alpha 100.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "cross_sum" \
  --use_sae


python3 src/vllm_cross_sum.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "cross_sum-all_langs2" \
  --dims "${languages[@]}" \
  --layers 6 14 23 32 40 \
  --alpha 100.0 \
  --batch_size 16 \
  --max_new_tokens 1 \
  --task_name "cross_sum"

  python3 src/ src/eval_lase.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "cross_sum-all_langs2" \
  --dims "${languages[@]}" \
  --layers 6 14 23 32 40 \
  --alpha 100.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "cross_sum"

