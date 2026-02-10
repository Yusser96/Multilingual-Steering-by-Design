#!/bin/sh

export PYTHONPATH=/home/Multilingual_By_Design:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#,1,2,3,4,5

cd /home/Multilingual_By_Design

export HF_TOKEN="<HF_TOKEN>"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"




#python3 src/list_saes.py


##### get parallel summarization data from hugginface for evaluation
python3 src/create_parallel_data_cross_sum.py

##### get flores data from hugginface for steering vectors
python3 src/get_flores_data.py


##### create steering vectors
bash pipeline_llama.sh "llama_scope_lxr_8x"
bash pipeline_llama.sh "Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000"
bash pipeline_llama.sh "Yusser/EN-SAES-Llama-3.1-8B_512_2100000000" 

bash pipeline_gemma.sh "gemma-scope-9b-pt-res-canonical"
bash pipeline_gemma.sh "Yusser/EN-SAES-gemma-2-9b_512_2100000000" 
bash pipeline_gemma.sh "Yusser/MULTI21-SAES-gemma-2-9b_512_2100000000"

##### generate intersection plot to select layers based on difffernt level of seprability
python3 src/detect_layer.py

##### run eval
bash pipeline_llama_flores.sh "llama_scope_lxr_8x"
bash pipeline_llama_flores.sh "Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000"
bash pipeline_llama_flores.sh "Yusser/EN-SAES-Llama-3.1-8B_512_2100000000" 

bash pipeline_llama_cross_sum.sh "llama_scope_lxr_8x"
bash pipeline_llama_cross_sum.sh "Yusser/MULTI21-SAES-Llama-3.1-8B_512_2100000000"
bash pipeline_llama_cross_sum.sh "Yusser/EN-SAES-Llama-3.1-8B_512_2100000000" 


bash pipeline_gemma_flores.sh "gemma-scope-9b-pt-res-canonical"
bash pipeline_gemma_flores.sh "Yusser/EN-SAES-gemma-2-9b_512_2100000000" 
bash pipeline_gemma_flores.sh "Yusser/MULTI21-SAES-gemma-2-9b_512_2100000000"

bash pipeline_gemma_cross_sum.sh "gemma-scope-9b-pt-res-canonical"
bash pipeline_gemma_cross_sum.sh "Yusser/EN-SAES-gemma-2-9b_512_2100000000" 
bash pipeline_gemma_cross_sum.sh "Yusser/MULTI21-SAES-gemma-2-9b_512_2100000000"


##### aggregate results as tsv for reporting and plotting
python3 src/print_res_flores.py "meta-llama/Llama-3.1-8B" "5.0"
python3 src/print_res_flores.py "google/gemma-2-9b" "100.0"

python3 src/print_res_cross_sum.py "meta-llama/Llama-3.1-8B" "5.0"
python3 src/print_res_cross_sum.py "google/gemma-2-9b" "100.0"

#### generate plots
bash pipeline_plots.sh "cross_sum"
bash pipeline_plots.sh "flores"

