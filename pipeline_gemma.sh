#!/bin/bash

export PYTHONPATH=/home/Multilingual_By_Design:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd /home/Multilingual_By_Design

export HF_TOKEN="<HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0




#python3 src/list_saes.py

MODEL_ID="google/gemma-2-9b"

# SAE_ID="gemma-scope-9b-pt-res-canonical"
# SAE_WIDTH="16k"

# SAE_ID="Yusser/EN-SAES-gemma-2-9b_512_2100000000" 
# SAE_ID="Yusser/MULTI21-SAES-gemma-2-9b_512_2100000000" 
# SAE_WIDTH="16k"


SAE_ID=$1
SAE_WIDTH="16k"

echo $SAE_ID



# 
languages=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no" "en")

# # DIM="bad"
echo "${languages[@]}"

for DIM in "${languages[@]}"
do
    echo $DIM
    python3 src/collect_sae_activations.py \
    --model_name $MODEL_ID \
    --sae_release $SAE_ID \
    --sae_width $SAE_WIDTH \
    --batch_size 4 \
    --dataset_path "data/flores200_dataset_low_res.json" \
    --dim $DIM \
    --output_dir activations \
    --max_samples 1000
done



# # languages=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no" "en")


python3 src/create_steer_vector.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "activations" \
  --dims "${languages[@]}" \
  --output_dir vectors

