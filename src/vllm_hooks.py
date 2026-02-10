import os

import torch as t
import numpy as np
import torch
import torch.nn.functional as F
from types import MethodType
from collections import defaultdict
import gc
#from sparsify import Sae





def steer_resid (residual, steer_vec, sae=None):

    if sae is not None:

        sae_features = sae.encode(residual)
        first_reconstruct = sae.decode(sae_features)
        delta = residual - first_reconstruct
        steer_vec = steer_vec.to(residual.dtype) #.unsqueeze(0)
        sae_features += steer_vec
        sae_reconstruct = sae.decode(sae_features)
        sae_reconstruct += delta
        residual[:, :] = sae_reconstruct.to(residual.dtype)

    else:
        steer_vec = steer_vec.to(residual.dtype) #.unsqueeze(0)
        residual[:, :]  += steer_vec
    
    return residual

def llama_resid_factory(steer_vec, sae=None):
    def new_block_forward(self,
        positions,
        hidden_states,
        residual
        ):
        # 1. Call the original transformer block forward to get its output

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states) # resid None
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual) # resid not None
       
        # hidden_states, residual  = original_forward(positions, hidden_states, residual)
        # resid_post, _ = self.input_layernorm(
        #         hidden_states, residual)

        residual = steer_resid(residual, steer_vec, sae=sae)

        #hidden_states = residual.to(residual.dtype)


        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)


        return hidden_states, residual
    return new_block_forward




def gemma_resid_factory(steer_vec, sae=None):
    def new_block_forward(self,
        positions,
        hidden_states,
        residual
        ):
        # 1. Call the original transformer block forward to get its output
       
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states) # resid None

        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual) # resid not None

        residual = steer_resid(residual, steer_vec, sae=sae)

            
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        
        return hidden_states, residual
    return new_block_forward


def aya_resid_factory(steer_vec, sae=None):
    print("*"*20,"layer:",layer_idx)
    def new_block_forward(self,
        positions,
        hidden_states,
        residual= None,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states, residual = self.input_layernorm(hidden_states, residual)



        residual = steer_resid(residual, steer_vec, sae=sae)
        


        hidden_states_attention = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states_mlp = self.mlp(hidden_states)
        # Add everything together
        hidden_states = residual + hidden_states_attention + hidden_states_mlp

        return hidden_states, residual
    return new_block_forward




def apply_steervec_intervention(model_id, model, layer_idx, steer_vec, alpha=1, sae=None):
    
    #resid_factory = None
    if "llama" in model_id.lower() or "mistral" in model_id.lower() or "qwen" in model_id.lower(): 
        resid_factory = llama_resid_factory
    elif "gemma" in model_id.lower(): 
        resid_factory = gemma_resid_factory
    elif "aya" in model_id.lower(): 
        resid_factory = aya_resid_factory



    # Store original forward methods to restore later
    original_forwards = {}


    steer_vec = torch.Tensor(steer_vec) * alpha

                
    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]

    # Store original forward method
    original_forwards[layer_idx] = obj.forward
    
    # Apply new forward method
    obj.forward = MethodType(resid_factory(steer_vec, sae=sae), obj)
    
    return original_forwards



def restore_original_forwards_saes(model, original_forwards):
    """Restore original forward methods"""
    for layer_idx in original_forwards:
        original_forward = original_forwards[layer_idx]
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[layer_idx]
        obj.forward = original_forward

