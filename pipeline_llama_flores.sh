#!/bin/sh

export PYTHONPATH=/home/Multilingual_By_Design:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd /home/Multilingual_By_Design

export HF_TOKEN="<HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0




#python3 src/list_saes.py

MODEL_ID="meta-llama/Llama-3.1-8B" 

languages=("bo" "mt" "it" "es" "de" "ja" "ar" "zh" "af" "nl" "fr" "pt" "ru" "ko" "hi" "tr" "pl" "sv" "da" "no")

SAE_ID=$1

# SAE_ID="llama_scope_lxr_8x" #"llama_scope_lxr_8x" 
# SAE_ID="Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000"  #"Yusser/LFB-SAES-Llama-3.1-8B" #"llama_scope_lxr_8x" 
# SAE_ID="Yusser/EN-SAES-Llama-3.1-8B_512_2100000000" 

SAE_WIDTH="8x"

echo $SAE_ID

python3 src/vllm_flores.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "flores-all_langs2-saes" \
  --dims "${languages[@]}" \
  --layers 6 13 15 19 25 \
  --alpha 5.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "flores" \
  --use_sae 


python3 src/eval_comet.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "flores-all_langs2-saes" \
  --dims "${languages[@]}" \
  --layers 6 13 15 19 25 \
  --alpha 5.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "flores" \
  --use_sae 


python3 src/vllm_flores.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "flores-all_langs2" \
  --dims "${languages[@]}" \
  --layers 6 13 15 19 25 \
  --alpha 5.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "flores"


python3 src/eval_comet.py \
  --model_name $MODEL_ID \
  --sae_release $SAE_ID \
  --sae_width $SAE_WIDTH \
  --dataset_path "vectors" \
  --output_dir "flores-all_langs2" \
  --dims "${languages[@]}" \
  --layers 6 13 15 19 25 \
  --alpha 5.0 \
  --batch_size 128 \
  --max_new_tokens 1 \
  --task_name "flores"
