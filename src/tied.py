# tied.py
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import math
from heuristics import calculate_local_embedding, calculate_global_embedding
import faiss
from typing import Optional

def transplant_tied_embeddings(
    model, new_tokenizer: AutoTokenizer, shared_vocab: list, unique_tokens: set,
    full_token_embeds_cache: dict, subtoken_embeds_cache: dict, old_vocab: dict,
    new_vocab: dict, old_tokenizer: AutoTokenizer, data_type: torch.dtype,
    temperature: float, # For local heuristic
    pad_to_multiple_of: int,
    faiss_index: Optional[faiss.Index], index_to_token: Optional[dict], k: int, global_weight: float,
    similarity_threshold: float,       # NEW
    min_confidence_threshold: float    # NEW
    ) -> None:
    """
    Transplants embeddings for a model with tied input/output embeddings.
    Uses heuristic helpers with updated global logic.
    """
    calc_temperature = temperature # Keep for local heuristic

    try:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, 'device') and model.device.type != 'meta' and model.device.type == 'cpu':
            calc_device = model.device # Use CPU if model is explicitly on CPU
    except AttributeError:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for heuristic calculations: {calc_device}")

    with torch.no_grad():
        input_embedding_layer = model.get_input_embeddings()
        if input_embedding_layer is None or not hasattr(input_embedding_layer, 'weight'):
            print("Error: Could not retrieve input embedding weights.")
            return

        original_input_embeddings = input_embedding_layer.weight.clone()
        original_output_embeddings = None # Tied
        embed_dim = original_input_embeddings.shape[1]

        new_vocab_size = len(new_tokenizer)
        if pad_to_multiple_of <= 0: pad_to_multiple_of = 8
        padded_size = math.ceil(new_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
        padded_size = max(padded_size, new_vocab_size)

        new_embeddings = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')
        mean = original_input_embeddings.mean().item()
        std = original_input_embeddings.std().item()
        if std < 1e-4: std = 0.02
        new_embeddings.normal_(mean=mean, std=std)
        print(f"Initialized new embedding matrix with size {padded_size}x{embed_dim}")

        # Copy shared tokens
        copied_count = 0
        for token in tqdm(shared_vocab, desc="Copying shared token embeddings (Tied)"):
            old_id = old_vocab.get(token)
            new_id = new_vocab.get(token)
            if old_id is not None and new_id is not None and \
               (0 <= old_id < original_input_embeddings.shape[0]) and \
               (0 <= new_id < new_embeddings.shape[0]):
                 new_embeddings[new_id] = original_input_embeddings[old_id].to(device='cpu', dtype=data_type)
                 copied_count += 1
        print(f"Copied {copied_count}/{len(shared_vocab)} shared token embeddings.")

        # Initialize unique tokens
        local_success, global_success, combined_success, random_init_count = 0, 0, 0, 0
        local_weight = 1.0 - global_weight
        use_global = global_weight > 0 and faiss_index is not None and index_to_token is not None
        use_local = local_weight > 0

        print(f"Initializing unique tokens (Tied).")
        print(f" Global heuristic: enabled={use_global}, weight={global_weight:.2f}, K={k}, sim_thresh={similarity_threshold:.2f}, conf_thresh={min_confidence_threshold:.2f}")
        print(f" Local heuristic: enabled={use_local}, weight={local_weight:.2f}, temp={calc_temperature:.2f}")

        unique_tokens_list = sorted(list(unique_tokens))
        for token_str in tqdm(unique_tokens_list, desc="Initializing unique tokens (Tied Hybrid)"):
            new_id = new_vocab.get(token_str)
            if new_id is None or not (0 <= new_id < new_embeddings.shape[0]): continue

            e_local_in, e_global_in = None, None
            local_calc_done, global_calc_done = False, False

            # Calculate local
            if use_local:
                try:
                    e_local_in, _ = calculate_local_embedding( # Output is None for tied
                        token_str, new_id, new_tokenizer, old_tokenizer,
                        full_token_embeds_cache, subtoken_embeds_cache,
                        original_input_embeddings, original_output_embeddings, # Pass None for output
                        calc_temperature, data_type, calc_device # Use temperature here
                    )
                    local_calc_done = True
                    if e_local_in is not None: local_success += 1
                except Exception as e: print(f"Error local calc for '{token_str}': {repr(e)}")

            # Calculate global
            if use_global:
                try:
                    full_token_decoded = new_tokenizer.decode([new_id], skip_special_tokens=True)
                    if isinstance(full_token_decoded, str) and len(full_token_decoded) > 0:
                        e_global_in, _ = calculate_global_embedding( # Output is None for tied
                            full_token_decoded, full_token_embeds_cache, faiss_index,
                            index_to_token, old_vocab,
                            original_input_embeddings, original_output_embeddings, # Pass None for output
                            k,
                            # temperature, # Not passed to new global calc
                            similarity_threshold,       # Pass NEW threshold
                            min_confidence_threshold,   # Pass NEW threshold
                            data_type, calc_device
                        )
                        global_calc_done = True
                        if e_global_in is not None: global_success += 1
                except Exception as e: print(f"Error global calc for '{token_str}': {repr(e)}")

            # Combine
            final_embedding = None
            if e_local_in is not None and e_global_in is not None:
                final_embedding = (local_weight * e_local_in + global_weight * e_global_in).to(dtype=data_type)
                combined_success += 1
            elif e_local_in is not None and use_local: final_embedding = e_local_in.to(dtype=data_type)
            elif e_global_in is not None and use_global: final_embedding = e_global_in.to(dtype=data_type)

            # Assign
            if final_embedding is not None:
                new_embeddings[new_id] = final_embedding.cpu()
            elif local_calc_done or global_calc_done: # Count as random only if heuristics were attempted
                random_init_count += 1

        print(f"Initialization results for {len(unique_tokens_list)} unique tokens:")
        print(f"  - Local heuristic succeeded: {local_success}")
        print(f"  - Global heuristic succeeded: {global_success} (passing confidence threshold)")
        print(f"  - Combined successfully (weighted): {combined_success}")
        print(f"  - Remained randomly initialized (after attempts): {random_init_count}")

        # Resize and copy
        print("Resizing model token embeddings...")
        for param in model.parameters(): param.requires_grad = False
        try:
             model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        except Exception as e: # Add manual check on error
             print(f"Error during resize: {e}")
             current_size = model.get_input_embeddings().weight.shape[0]
             if current_size != padded_size:
                 print(f"FATAL: Embedding size mismatch after resize attempt ({current_size} vs {padded_size}). Stopping.")
                 return
             else: print("Manual check ok.")
        print(f"Model embedding size after resize: {model.get_input_embeddings().weight.shape}")

        target_layer = model.get_input_embeddings()
        target_device, target_dtype = target_layer.weight.device, target_layer.weight.dtype
        print(f"Target device/dtype: {target_device}, {target_dtype}")

        new_embeddings_tensor = new_embeddings.to(device=target_device, dtype=target_dtype)

        if new_embeddings_tensor.shape == target_layer.weight.shape:
             target_layer.weight.copy_(new_embeddings_tensor)
             print("Copied initialized embeddings to model.")
        else:
             print(f"FATAL: Final shape mismatch. Expected {target_layer.weight.shape}, got {new_embeddings_tensor.shape}.")
             return

        # Tie weights and verify
        try: model.tie_weights(); print("Model weights tied.")
        except Exception as e: print(f"Warning: Failed to tie weights: {e}")
        # (Verification logic remains the same as latest version)
        if model.get_output_embeddings() is not None:
             if model.get_input_embeddings().weight is model.get_output_embeddings().weight: print("Verification: Weights tied (same object).")
             elif torch.equal(model.get_input_embeddings().weight.data, model.get_output_embeddings().weight.data): print("Verification: Weights tied (identical data).")
             else: print("Warning: Verification failed - Weights seem different.")
        elif getattr(model.config, "tie_word_embeddings", False): print("Verification: Output layer None, config tied=True.")
        else: print("Verification: Output layer None.")

        print("Tied embedding update complete.")
