# heuristics.py
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
from typing import Optional,Tuple


def calculate_global_embedding(
    query_token_str: str,
    full_token_embeds_cache: dict,
    faiss_index: faiss.Index,
    # old_tokenizer: AutoTokenizer, # Removed - Not used in global heuristic logic
    index_to_token: dict,
    old_vocab: dict,
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    k: int,
    # temperature: float, # Removed - No longer used for global proportional weighting
    similarity_threshold: float,      # NEW: General threshold for neighbors
    min_confidence_threshold: float,  # NEW: Min similarity for the BEST neighbor
    data_type: torch.dtype,
    device: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embedding based on the Global heuristic using similarity thresholding,
    a minimum confidence check, and proportional weighting.

    Retrieves up to 'k' nearest neighbors. Filters them based on 'similarity_threshold'.
    Additionally checks if the *best* neighbor's similarity meets 'min_confidence_threshold'.
    If both checks pass, the embeddings of the valid neighbors (passing similarity_threshold)
    are averaged, weighted proportionally to their cosine similarity scores.

    Args:
        query_token_str: The string representation of the new token (decoded).
        full_token_embeds_cache: Cache mapping token strings to their external embeddings.
        faiss_index: Pre-built FAISS index of old vocabulary external embeddings.
        index_to_token: Mapping from FAISS index ID to old vocabulary token string.
        old_vocab: Original vocabulary mapping (token -> ID).
        original_input_embeddings: The input embedding matrix of the original model.
        original_output_embeddings: The output embedding matrix (if untied and exists), else None.
        k: Initial number of neighbors to retrieve from FAISS search.
        similarity_threshold: Minimum cosine similarity for *any* neighbor to be considered.
        min_confidence_threshold: Minimum cosine similarity required for the *top-ranked*
                                 neighbor for the global result to be considered valid.
        data_type: Torch data type for calculations.
        device: Device for torch tensor operations.

    Returns:
        A tuple (embedding_input, embedding_output):
        - embedding_input: Calculated embedding for the input layer (CPU tensor), or None.
        - embedding_output: Calculated embedding for the output layer (CPU tensor), or None
                          if original_output_embeddings was None or calculation failed.
    """
    # Ensure query token exists in cache
    # query_token_str = query_token_str # This line was redundant
    if query_token_str not in full_token_embeds_cache:
        # print(f"Debug: Query token '{query_token_str}' not in full_token_embeds_cache.")
        return None, None

    try:
        # Prepare query vector
        query_embedding_list = full_token_embeds_cache[query_token_str]
        query_embedding = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding) # Normalize for cosine similarity via IP
    except Exception as e:
        print(f"Warning: Error preparing query vector for '{query_token_str}' in global heuristic: {e}")
        return None, None

    try:
        # --- FAISS Search ---
        distances, indices = faiss_index.search(query_embedding, k)
        # Ensure results are 1D arrays
        distances = distances.squeeze(0) if distances.ndim > 1 else distances
        indices = indices.squeeze(0) if indices.ndim > 1 else indices

        # Handle case where search returns fewer than k results or empty results
        if distances.size == 0 or indices.size == 0:
             # print(f"Debug: FAISS search returned no results for '{query_token_str}'.")
             return None, None

        # --- Filtering Stage 1 & Identify Best ---
        valid_neighbor_orig_ids = []
        valid_similarities = []
        highest_similarity = -1.0 # Track best similarity among those passing stage 1

        for sim, idx in zip(distances, indices):
            # Basic validation of FAISS index
            if not isinstance(idx, (int, np.integer)) or idx < 0:
                 continue

            # Filter 1: General Similarity Threshold
            if sim < similarity_threshold:
                continue # Skip neighbors below the general threshold

            # Get neighbor token string and original ID
            neighbor_token = index_to_token.get(int(idx))
            if neighbor_token is None: continue # Skip if token mapping is missing

            neighbor_orig_id = old_vocab.get(neighbor_token)

            # Check if neighbor exists and is valid in original embeddings
            if neighbor_orig_id is not None and (0 <= neighbor_orig_id < original_input_embeddings.shape[0]):
                 valid_neighbor_orig_ids.append(neighbor_orig_id)
                 valid_similarities.append(sim)
                 # Update highest similarity *among valid neighbors passing threshold*
                 if sim > highest_similarity:
                      highest_similarity = sim
            # else: # Debugging info if needed
                 # print(f"Debug: Neighbor token '{neighbor_token}' (ID: {neighbor_orig_id}) invalid/OOB for input embeddings.")

        # --- Filtering Stage 2: Minimum Confidence Check ---
        if not valid_neighbor_orig_ids or highest_similarity < min_confidence_threshold:
            # if not valid_neighbor_orig_ids:
            #     print(f"Debug: No neighbors passed similarity_threshold {similarity_threshold} for '{query_token_str}'.")
            # elif highest_similarity < min_confidence_threshold:
            #     print(f"Debug: Best neighbor sim {highest_similarity:.4f} < min_confidence_threshold {min_confidence_threshold} for '{query_token_str}'. Rejecting global.")
            return None, None # Fail global heuristic if confidence check fails

        # --- Weight Calculation (Proportional) ---
        # Only proceed if Stage 2 passed
        similarities_tensor = torch.tensor(valid_similarities, dtype=data_type, device=device)
        weights_sum = similarities_tensor.sum()

        if weights_sum > 1e-9: # Avoid division by zero
            weights = similarities_tensor / weights_sum
        else:
            # Fallback for edge cases (should be rare if mct > 0)
            print(f"Warning: Sum of similarities near zero for '{query_token_str}' after thresholding. Using equal weights.")
            weights = torch.ones_like(similarities_tensor) / len(similarities_tensor)

        weights_unsqueezed = weights.unsqueeze(1)

        # --- Embedding Combination ---
        # Calculate for Input Embeddings
        neighbor_input_embeds = original_input_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
        global_embedding_input = (weights_unsqueezed * neighbor_input_embeds).sum(dim=0).cpu()

        # Calculate for Output Embeddings (if applicable)
        global_embedding_output = None
        if original_output_embeddings is not None:
            # Check validity of IDs for the output matrix *before* indexing
            valid_ids_mask_output = [0 <= idx < original_output_embeddings.shape[0] for idx in valid_neighbor_orig_ids]

            if all(valid_ids_mask_output):
                # All IDs valid for input are also valid for output
                neighbor_output_embeds = original_output_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
                global_embedding_output = (weights_unsqueezed * neighbor_output_embeds).sum(dim=0).cpu()
            else:
                # Handle mismatch: Recalculate using only jointly valid IDs
                print(f"Warning: Mismatch in valid neighbor IDs between input and output embeddings for '{query_token_str}'. Recalculating output weights.")
                jointly_valid_indices = [i for i, is_valid in enumerate(valid_ids_mask_output) if is_valid]

                if jointly_valid_indices:
                    joint_valid_ids = [valid_neighbor_orig_ids[i] for i in jointly_valid_indices]
                    joint_similarities = similarities_tensor[jointly_valid_indices] # Use tensor slice

                    joint_weights_sum = joint_similarities.sum()
                    if joint_weights_sum > 1e-9:
                        joint_weights = joint_similarities / joint_weights_sum
                    else:
                        joint_weights = torch.ones_like(joint_similarities) / len(joint_similarities)

                    joint_weights_unsqueezed = joint_weights.unsqueeze(1)
                    joint_neighbor_output_embeds = original_output_embeddings[joint_valid_ids].to(device=device, dtype=data_type)
                    global_embedding_output = (joint_weights_unsqueezed * joint_neighbor_output_embeds).sum(dim=0).cpu()
                else:
                    print(f"Warning: No common valid neighbors found for output embedding calculation for '{query_token_str}'. Setting output to None.")
                    global_embedding_output = None
        # else: Tied model, output is None by default

        return global_embedding_input, global_embedding_output

    except Exception as e:
        print(f"ERROR: Unhandled exception during global embedding calculation for '{query_token_str}': {repr(e)}")
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
    temperature: float, # Temperature IS used here
    data_type: torch.dtype,
    device: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Calculates embedding based on the Local Subword Composition heuristic.
    (Remains unchanged from the version you provided - uses temperature and softmax)
    """
    try:
        full_token_decoded = new_tokenizer.decode([new_token_id], skip_special_tokens=True) # Robust decode
        if not full_token_decoded: # Handle empty string case
             # print(f"Debug: Decoded token for ID {new_token_id} is empty.")
             return None, None
        if full_token_decoded not in full_token_embeds_cache:
            # print(f"Debug: Decoded token '{full_token_decoded}' not in full_token_embeds_cache for local heuristic.")
            return None, None

        full_embed_ext = torch.tensor(full_token_embeds_cache[full_token_decoded], dtype=data_type, device=device)
        # Encode the full token string using the OLD tokenizer
        old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
        if not old_ids:
            # print(f"Debug: Old tokenizer produced no IDs for '{full_token_decoded}'.")
            return None, None

        valid_subtoken_embeds_ext = []
        valid_subtoken_strs = []
        valid_old_ids_for_input = []

        # Iterate through OLD IDs
        for oid in old_ids:
            # Check validity *before* decoding or cache access
            if oid is None or not (0 <= oid < original_input_embeddings.shape[0]):
                continue

            # Decode the OLD ID to get the subtoken string
            subtoken_str = old_tokenizer.decode([oid], skip_special_tokens=True)
            if not subtoken_str: continue # Skip empty subtokens

            if subtoken_str in subtoken_embeds_cache:
                try:
                    valid_subtoken_embeds_ext.append(torch.tensor(subtoken_embeds_cache[subtoken_str], dtype=data_type, device=device))
                    valid_subtoken_strs.append(subtoken_str)
                    valid_old_ids_for_input.append(oid)
                except Exception as e:
                    print(f"Warning: Error processing subtoken '{subtoken_str}' (ID: {oid}) in local heuristic: {e}")
                    continue

        if not valid_subtoken_embeds_ext:
            # print(f"Debug: No valid subtokens found with embeddings for '{full_token_decoded}'.")
            return None, None

        # --- Weighting Logic (Original Local Heuristic) ---
        sub_embeds_ext_tensor = torch.stack(valid_subtoken_embeds_ext)

        # Ensure full_embed_ext is correctly shaped
        if full_embed_ext.ndim == 1: full_embed_ext = full_embed_ext.unsqueeze(0)

        similarities = F.cosine_similarity(full_embed_ext, sub_embeds_ext_tensor, dim=1)
        # Weight based on similarity (softmax)
        weights1 = F.softmax(similarities, dim=0)

        # Weight based on length normalization
        len_full = len(full_token_decoded) # Already checked > 0
        try:
            raw_lens = [len(s) for s in valid_subtoken_strs]
            len_norm = torch.tensor([max(0.0, l / len_full) for l in raw_lens], dtype=data_type, device=device)
        except ZeroDivisionError: # Should not happen due to check above, but safe
             print(f"Warning: ZeroDivisionError during length norm for '{full_token_decoded}'. Skipping norm.")
             len_norm = torch.zeros_like(weights1)
        except Exception as e: # Catch other potential errors
             print(f"Warning: Error calculating length norm for '{full_token_decoded}': {e}. Skipping norm.")
             len_norm = torch.zeros_like(weights1)

        # Combine weights and apply temperature
        combined_weights = (weights1 + len_norm) / 2.0
        final_weights = F.softmax(combined_weights / (temperature + 1e-9), dim=0) # Add epsilon for safety
        final_weights_unsqueezed = final_weights.unsqueeze(1)

        # --- Embedding Combination ---
        # Input Embedding
        old_embeds_orig_input = original_input_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
        local_embedding_input = (final_weights_unsqueezed * old_embeds_orig_input).sum(dim=0).cpu()

        # Output Embedding (if applicable)
        local_embedding_output = None
        if original_output_embeddings is not None:
            # Ensure subtoken IDs are valid for output matrix
            valid_ids_mask_output_local = [0 <= idx < original_output_embeddings.shape[0] for idx in valid_old_ids_for_input]
            if all(valid_ids_mask_output_local):
                old_embeds_orig_output = original_output_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
                local_embedding_output = (final_weights_unsqueezed * old_embeds_orig_output).sum(dim=0).cpu()
            else:
                 print(f"Warning: Mismatch in valid subtoken IDs for output embedding in local heuristic on '{token_str}'. Output set to None.")
                 local_embedding_output = None
        # else: Tied model, output is None

        return local_embedding_input, local_embedding_output

    except Exception as e:
        print(f"ERROR: Unhandled exception during local embedding calculation for '{token_str}': {repr(e)}")
        return None, None
