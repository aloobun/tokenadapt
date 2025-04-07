# heuristics.py add return_details flag
import torch
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
import faiss
from typing import Optional, Tuple, Dict, Any

# --- Helper Function for Softmax ---
def _softmax_with_temperature(similarities: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(similarities / temperature, dim=0)

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
    data_type: torch.dtype,
    device: str,
    return_details: bool = False # ADDED FLAG
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]: # MODIFIED RETURN

    # ... (keep existing code for query_embedding preparation) ...
    query_token_str = query_token_str # Ensure it's the decoded string
    if query_token_str not in full_token_embeds_cache:
        return None, None, None

    try:
        query_embedding_list = full_token_embeds_cache[query_token_str]
        query_embedding = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
    except Exception as e:
        print(f"Warning: Error preparing query vector for '{query_token_str}' in global heuristic: {e}")
        return None, None, None

    details = None
    global_embedding_input = None
    global_embedding_output = None

    try:
        distances, indices = faiss_index.search(query_embedding, k)
        distances = distances.squeeze(0)
        indices = indices.squeeze(0)

        valid_neighbor_orig_ids = []
        valid_similarities = []
        neighbor_tokens = [] # Store neighbor tokens for details

        for sim, idx in zip(distances, indices):
            if idx == -1: continue
            neighbor_token = index_to_token.get(idx)
            if neighbor_token is None: continue

            neighbor_orig_id = old_vocab.get(neighbor_token)
            if neighbor_orig_id is not None and (0 <= neighbor_orig_id < original_input_embeddings.shape[0]):
                 valid_neighbor_orig_ids.append(neighbor_orig_id)
                 valid_similarities.append(sim)
                 neighbor_tokens.append(neighbor_token) # Store for details

        if not valid_neighbor_orig_ids:
            return None, None, None

        similarities_tensor = torch.tensor(valid_similarities, dtype=data_type, device=device)
        weights = _softmax_with_temperature(similarities_tensor, temperature)
        weights_unsqueezed = weights.unsqueeze(1)

        neighbor_input_embeds = original_input_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
        global_embedding_input = (weights_unsqueezed * neighbor_input_embeds).sum(dim=0).cpu()

        if original_output_embeddings is not None:
            # Check if IDs are valid for output embeddings too
            valid_indices_for_output = [i for i, oid in enumerate(valid_neighbor_orig_ids) if 0 <= oid < original_output_embeddings.shape[0]]
            if len(valid_indices_for_output) == len(valid_neighbor_orig_ids): # Simplified check
                 neighbor_output_embeds = original_output_embeddings[valid_neighbor_orig_ids].to(device=device, dtype=data_type)
                 global_embedding_output = (weights_unsqueezed * neighbor_output_embeds).sum(dim=0).cpu()
            # else: Handle case where some neighbors are not in output vocab? Or assume they are.

        if return_details:
            details = {
                'type': 'global',
                'contributor_ids': valid_neighbor_orig_ids,
                'contributor_tokens': neighbor_tokens,
                'similarities': valid_similarities,
                'weights': weights.cpu().tolist()
            }

        return global_embedding_input, global_embedding_output, details # MODIFIED RETURN

    except Exception as e:
        print(f"Warning: Error during FAISS search/processing for '{query_token_str}': {e}")
        return None, None, None


def calculate_local_embedding(
    token_str: str, # This is the raw token string from new vocab
    new_token_id: int,
    new_tokenizer: AutoTokenizer,
    old_tokenizer: AutoTokenizer,
    full_token_embeds_cache: dict,
    subtoken_embeds_cache: dict,
    original_input_embeddings: torch.Tensor,
    original_output_embeddings: Optional[torch.Tensor],
    temperature: float,
    data_type: torch.dtype,
    device: str,
    return_details: bool = False # ADDED FLAG
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]]]: # MODIFIED RETURN

    details = None
    local_embedding_input = None
    local_embedding_output = None

    full_token_decoded = new_tokenizer.decode([new_token_id]) # Use decoded string for cache lookup

    if full_token_decoded not in full_token_embeds_cache:
        # print(f"'{full_token_decoded}' not in full_token_embeds_cache") # Debug print
        return None, None, None

    try:
        full_embed_ext = torch.tensor(full_token_embeds_cache[full_token_decoded], dtype=data_type, device=device)
        old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
        if not old_ids:
            return None, None, None

        valid_subtoken_embeds_ext = []
        valid_subtoken_strs = []
        valid_old_ids_for_input = [] # Use this list for indices

        for oid in old_ids:
             # Check validity for input embeddings
            if 0 <= oid < original_input_embeddings.shape[0]:
                subtoken_str = old_tokenizer.decode([oid])
                if subtoken_str in subtoken_embeds_cache:
                    valid_subtoken_embeds_ext.append(torch.tensor(subtoken_embeds_cache[subtoken_str], dtype=data_type, device=device))
                    valid_subtoken_strs.append(subtoken_str)
                    valid_old_ids_for_input.append(oid) # Store the valid ID
                # else: print(f"Subtoken '{subtoken_str}' not in cache") # Debug print
            # else: print(f"Old ID {oid} out of bounds for input embeds") # Debug print


        if not valid_subtoken_embeds_ext:
            # print("No valid subtokens found.") # Debug print
            return None, None, None

        sub_embeds_ext_tensor = torch.stack(valid_subtoken_embeds_ext)

        # --- Calculate Similarities (Cosine) ---
        # Normalize for cosine similarity calculation
        full_embed_ext_norm = F.normalize(full_embed_ext, p=2, dim=0)
        sub_embeds_ext_tensor_norm = F.normalize(sub_embeds_ext_tensor, p=2, dim=1)
        similarities = torch.mv(sub_embeds_ext_tensor_norm, full_embed_ext_norm) # More efficient dot product

        # --- Length Normalization (Optional but in original code) ---
        try:
             len_full = len(full_token_decoded)
             if len_full == 0: raise ValueError("Zero length token")
             len_norm = torch.tensor([len(s) / len_full for s in valid_subtoken_strs], dtype=data_type, device=device)
        except (ValueError, ZeroDivisionError) as e:
             print(f"Warning: Error calculating length norm for '{full_token_decoded}': {e}. Skipping length norm.")
             len_norm = torch.zeros_like(similarities) # Fallback to zeros

        # --- Combine Weights ---
        # Original code averages similarity and length norm before softmax
        combined_scores = (similarities + len_norm) / 2.0
        final_weights = _softmax_with_temperature(combined_scores, temperature)
        final_weights_unsqueezed = final_weights.unsqueeze(1)

        # --- Weighted Sum for Input Embeddings ---
        old_embeds_orig_input = original_input_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
        local_embedding_input = (final_weights_unsqueezed * old_embeds_orig_input).sum(dim=0).cpu()

        # --- Weighted Sum for Output Embeddings (if untied) ---
        if original_output_embeddings is not None:
            # Check if IDs are valid for output embeddings too
            valid_indices_for_output = [i for i, oid in enumerate(valid_old_ids_for_input) if 0 <= oid < original_output_embeddings.shape[0]]
            if len(valid_indices_for_output) == len(valid_old_ids_for_input): # Simplified check
                old_embeds_orig_output = original_output_embeddings[valid_old_ids_for_input].to(device=device, dtype=data_type)
                local_embedding_output = (final_weights_unsqueezed * old_embeds_orig_output).sum(dim=0).cpu()
            # else: Handle case?

        if return_details:
            details = {
                'type': 'local',
                'contributor_ids': valid_old_ids_for_input,
                'contributor_tokens': valid_subtoken_strs,
                'similarities': similarities.cpu().tolist(), # Raw similarities before length norm
                'combined_scores': combined_scores.cpu().tolist(),
                'weights': final_weights.cpu().tolist()
            }

        return local_embedding_input, local_embedding_output, details # MODIFIED RETURN

    except Exception as e:
        import traceback
        print(f"ERROR calculating local embedding for '{token_str}' (ID: {new_token_id}, Decoded: '{full_token_decoded}'): {e}")
        traceback.print_exc()
        return None, None, None
