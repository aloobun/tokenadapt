# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script contains helper functions for calculating initial embeddings
# using local and global heuristics.


import torch
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
import faiss
from typing import Optional,Tuple,Dict


def calculate_global_embedding(
    query_token_str: str,
    full_token_embeds_cache: dict,
    faiss_index: faiss.Index,
    old_tokenizer: AutoTokenizer,
    index_to_token: dict,
    old_vocab: dict,
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    k: int,
    temperature: float,
    threshold: float,
    data_type: torch.dtype,
    device: str 
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embedding based on the Global heuristic.

    Select Top-K -> Apply Threshold -> Softmax Weights -> Weighted Sum of Neighbor Embeddings
    
    Args:
        query_token_str: The string representation of the new token (decoded).
        full_token_embeds_cache: Cache mapping token strings to their external embeddings.
        faiss_index: Pre-built FAISS index of old vocabulary external embeddings.
        old_tokenizer: The old tokenizer.
        index_to_token: Mapping from FAISS index ID to old vocabulary token string.
        old_vocab: Original vocabulary mapping (token -> ID).
        original_input_embeddings: The input embedding matrix of the original model.
        original_output_embeddings: The output embedding matrix (if untied and exists), else None.
        k: Number of neighbors to find.
        temperature: Temperature for softmax weighting of similarities.
        threshold: Similarity threshold for considering neighbors.
        data_type: Torch data type for calculations.
        device: Device for torch tensor operations.

    Returns:
        A tuple (embedding_input, embedding_output):
        - embedding_input: Calculated embedding for the input layer (CPU tensor), or None.
        - embedding_output: Calculated embedding for the output layer (CPU tensor), or None if original_output_embeddings was None or calculation failed.

        
    """

    query_token_str = query_token_str
    if query_token_str not in full_token_embeds_cache:
        return None, None

    try:
        query_embedding_list = full_token_embeds_cache[query_token_str]
        query_embedding = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
    except Exception as e:
        print(f"Warning: Error preparing query vector for '{query_token_str}' in global heuristic: {e}")
        return None, None

    try:
        distances, indices = faiss_index.search(query_embedding, k)
        distances = distances.squeeze(0)
        indices = indices.squeeze(0)

        valid_neighbor_orig_ids = []
        valid_similarities = []

        for sim, idx in zip(distances, indices):
            if idx == -1: continue
            neighbor_token = index_to_token.get(idx)
            if neighbor_token is None: continue

            neighbor_orig_id = old_vocab.get(neighbor_token)
            if neighbor_orig_id is not None and (0 <= neighbor_orig_id < original_input_embeddings.shape[0]):
                 valid_neighbor_orig_ids.append(neighbor_orig_id)
                 valid_similarities.append(sim)

        if not valid_neighbor_orig_ids:
            return None, None


        similarities_tensor = torch.tensor(valid_similarities, dtype=data_type, device=device)
        threshold_mask = similarities_tensor >= threshold
        weights_input = similarities_tensor / temperature
        weights_input = torch.where(threshold_mask, weights_input, torch.tensor(float('-inf'), device=device, dtype=data_type))
        weights = F.softmax(weights_input, dim=0)

        if torch.isinf(weights).all():
             # print(f"Warning: All similarities below threshold {threshold} for '{query_token_str}' in global heuristic.")
             return None, None
        
        weights_unsqueezed = weights.unsqueeze(1)


        neighbor_input_embeds = original_input_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
        global_embedding_input = (weights_unsqueezed * neighbor_input_embeds).sum(dim=0).cpu()

        global_embedding_output = None
        if original_output_embeddings is not None:
            neighbor_output_embeds = original_output_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
            global_embedding_output = (weights_unsqueezed * neighbor_output_embeds).sum(dim=0).cpu()
        else:
            global_embedding_output = None



        return global_embedding_input, global_embedding_output

    except Exception as e:
        print(f"Warning: Error during FAISS search/processing for '{query_token_str}': {e}")
        return None, None


def calculate_local_embedding(
    token_str: str, 
    new_token_id: int,
    new_tokenizer: AutoTokenizer,
    old_tokenizer: AutoTokenizer,
    full_token_embeds_cache: dict,
    subtoken_embeds_cache: dict,   
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    temperature: float,
    threshold: float,
    data_type: torch.dtype,
    device: str 
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embedding based on the Local Subword Composition heuristic,
    applying a similarity threshold and using the combined weighting scheme:
    (thresholded_sim_weight + avg_weight + len_weight) / temperature -> softmax

    Args:
        token_str: The unique token string from the new vocabulary.
        new_token_id: The ID of the token in the new vocabulary.
        new_tokenizer: The new tokenizer.
        old_tokenizer: The old tokenizer.
        full_token_embeds_cache: Cache mapping decoded new token strings to external embeddings.
        subtoken_embeds_cache: Cache mapping old subtoken strings to external embeddings.
        original_input_embeddings: The input embedding matrix of the original model.
        original_output_embeddings: The output embedding matrix (if untied and exists), else None.        
        temperature: Temperature for softmax weighting.
        threshold: Similarity threshold for considering subword similarity weights.
        data_type: Torch data type for calculations.
        device: Device for torch tensor operations.

    Returns:
        A tuple (embedding_input, embedding_output):
        - embedding_input: Calculated embedding for the input layer (CPU tensor), or None.
        - embedding_output: Calculated embedding for the output layer (CPU tensor), or None if original_output_embeddings was None or calculation failed.
    """
    threshold = threshold * 0.70  # Generally thresold will be higher for knn ; we 70% of the threshold for subword ; 
                                  # so if 0.7 is threshold for knn, 0.49 is threshold for subword

    full_token_decoded = new_tokenizer.decode([new_token_id])

    if full_token_decoded not in full_token_embeds_cache:
        return None, None

    full_embed_ext = torch.tensor(full_token_embeds_cache[full_token_decoded], dtype=data_type, device=device)
    old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
    if not old_ids:
        return None, None

    valid_subtoken_embeds_ext = []
    valid_subtoken_strs = []
    valid_old_ids_for_input = [] 


    for oid in old_ids:
        if 0 <= oid < original_input_embeddings.shape[0]: 
            subtoken_str = old_tokenizer.decode([oid])
            if subtoken_str in subtoken_embeds_cache:
                valid_subtoken_embeds_ext.append(torch.tensor(subtoken_embeds_cache[subtoken_str], dtype=data_type, device=device))
                valid_subtoken_strs.append(subtoken_str)
                valid_old_ids_for_input.append(oid)

    if not valid_subtoken_embeds_ext:
        return None, None


    num_subtokens = len(valid_subtoken_strs)
    sub_embeds_ext_tensor = torch.stack(valid_subtoken_embeds_ext)

    similarities = F.cosine_similarity(full_embed_ext.unsqueeze(0), sub_embeds_ext_tensor, dim=1)

    threshold_mask = similarities >= threshold
    sim_weight = similarities * threshold_mask.float()  # threshold for only sim weights
    avg_weight = torch.ones_like(similarities) / num_subtokens

    try:
         len_full = len(full_token_decoded)
         if len_full == 0: raise ValueError("Zero length token")
         len_weight = torch.tensor([len(s) / len_full for s in valid_subtoken_strs], dtype=data_type, device=device)
         len_weight = torch.clamp(len_weight, max=1.0) # Clamp length weight
    except Exception:
         len_weight = torch.zeros_like(similarities)
    combined_score = sim_weight + avg_weight + len_weight 
    final_weights = F.softmax(combined_score / (temperature), dim=0) 
    if not torch.any(final_weights > 0) or torch.isnan(final_weights).any():
        print(f"Warning: All final weights zero/NaN for '{full_token_decoded}'. Using equal weights.")
        final_weights = torch.ones_like(similarities) / num_subtokens

    final_weights_unsqueezed = final_weights.unsqueeze(1)



    old_embeds_orig_input = original_input_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
    local_embedding_input = (final_weights_unsqueezed * old_embeds_orig_input).sum(dim=0).cpu()

    local_embedding_output = None
    if original_output_embeddings is not None:
        old_embeds_orig_output = original_output_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
        local_embedding_output = (final_weights_unsqueezed * old_embeds_orig_output).sum(dim=0).cpu()
    else:
        local_embedding_output = None

    return local_embedding_input, local_embedding_output

