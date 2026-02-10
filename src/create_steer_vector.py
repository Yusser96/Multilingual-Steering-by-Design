import torch
import torch.nn.functional as F
import argparse
import numpy as np

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
    "--dims",
    nargs='+',
    default=None,
    required=True,
    help="Dimensions keys",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Where to save activation stats (will be created if missing)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to JSON dataset file",
)


args = parser.parse_args()


import sys



id2dim = {i:dim for i,dim in enumerate(args.dims)}
dim2id = {dim:i for i,dim in enumerate(args.dims)}

import os

input_path = os.path.join(args.dataset_path, args.model_name, args.sae_release, args.sae_width)



def get_data(file_name="model_resid_post_activation"):
    n, over_zero = [], []
    for dim in args.dims:
        data = torch.load(f"{input_path}/{file_name}.{dim}") 

        n.append(data['n'])
        over_zero.append(data['over_zero'])

    n = torch.tensor(n)

    print("N:",n)
    print("over_zero[0]:",over_zero[0])


    over_zero = torch.stack(over_zero, dim=-1)

    over_zero = torch.nan_to_num(over_zero, nan=0.0, posinf=0.0, neginf=0.0)


    num_layers, intermediate_size, dim_num = over_zero.size()
    print(num_layers, intermediate_size, dim_num)

    return n, over_zero



def activation(file_name="model_resid_post_activation"):

    n, over_zero = get_data(file_name)

    # Compute average activation probabilities per dim
    activation_probs = over_zero / n

    n_sum = n.sum() - n

    dim_sum = over_zero.sum(dim=2).unsqueeze(-1) - over_zero

    activation_sum_probs = dim_sum / n_sum

    diff_mean = activation_probs - activation_sum_probs

    norms = torch.norm(diff_mean, dim=1, keepdim=True)
    diff_mean_normalized = diff_mean / norms.clamp(min=1e-12)  # avoid div by zero

    all_svectors = []
    for layer_index in range(0, diff_mean_normalized.shape[0]):
        print("layer:",layer_index)
        svectors = diff_mean_normalized[layer_index, :, :].T
        #print(svectors.shape)
        d_size = svectors.shape[1]
        svectors = svectors.cpu().numpy()

        svectors = {id2dim[i]:svectors[i] for i in range(svectors.shape[0])}
        all_svectors.append(svectors)


    out_path = os.path.join(args.output_dir, args.model_name, args.sae_release, args.sae_width)
    os.makedirs(out_path,exist_ok=True)
    
    torch.save(all_svectors, f'{out_path}/{file_name}_vectors_diffmean')



activation(file_name="model_resid_post_activation")
activation(file_name="sae_activation")