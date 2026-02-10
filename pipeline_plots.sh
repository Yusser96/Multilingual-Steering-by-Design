#!/bin/bash

export PYTHONPATH=/home/Multilingual_By_Design:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd /home/Multilingual_By_Design

export HF_TOKEN="<HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

task=$1
#task=flores
# task=cross_sum

# script=print_analyse_res_plots.py
script=print_analyse_res_plots_delta.py


# python3 src/$script \
#                                                         --task $task \
#                                                         --input_dir latex2/$task/per_tgt_lang \
#                                                         --output_dir plots2/$task/gemma-2-9b/per_tgt_lang/all \
#                                                         --model gemma-2-9b \
#                                                         --type all

# python3 src/$script \
#                                                         --task $task \
#                                                         --input_dir latex2/$task/per_lang \
#                                                         --output_dir plots2/$task/gemma-2-9b/per_lang/all \
#                                                         --model gemma-2-9b \
#                                                         --type all

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_tgt_lang \
                                                        --output_dir plots2/$task/gemma-2-9b/per_tgt_lang/diff \
                                                        --model gemma-2-9b \
                                                        --type diff


python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_tgt_lang \
                                                        --output_dir plots2/$task/gemma-2-9b/per_tgt_lang/same \
                                                        --model gemma-2-9b \
                                                        --type same

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_lang \
                                                        --output_dir plots2/$task/gemma-2-9b/per_lang/diff \
                                                        --model gemma-2-9b \
                                                        --type diff

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_lang \
                                                        --output_dir plots2/$task/gemma-2-9b/per_lang/same \
                                                        --model gemma-2-9b \
                                                        --type same







# python3 src/$script \
#                                                         --task $task \
#                                                         --input_dir latex2/$task/per_tgt_lang \
#                                                         --output_dir plots2/$task/Llama-3.1-8B/per_tgt_lang/all \
#                                                         --model Llama-3.1-8B \
#                                                         --type all

# python3 src/$script \
#                                                         --task $task \
#                                                         --input_dir latex2/$task/per_lang \
#                                                         --output_dir plots2/$task/Llama-3.1-8B/per_lang/all \
#                                                         --model Llama-3.1-8B \
#                                                         --type all

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_tgt_lang \
                                                        --output_dir plots2/$task/Llama-3.1-8B/per_tgt_lang/diff \
                                                        --model Llama-3.1-8B \
                                                        --type diff


python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_tgt_lang \
                                                        --output_dir plots2/$task/Llama-3.1-8B/per_tgt_lang/same \
                                                        --model Llama-3.1-8B \
                                                        --type same

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_lang \
                                                        --output_dir plots2/$task/Llama-3.1-8B/per_lang/diff \
                                                        --model Llama-3.1-8B \
                                                        --type diff

python3 src/$script \
                                                        --task $task \
                                                        --input_dir latex2/$task/per_lang \
                                                        --output_dir plots2/$task/Llama-3.1-8B/per_lang/same \
                                                        --model Llama-3.1-8B \
                                                        --type same


