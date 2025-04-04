# untied.py
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

def transplant_untied_embeddings(
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
    Transplants embeddings for a model with untied input/output embeddings.
    Uses heuristic helpers with updated global logic.
    """
    calc_temperature = temperature # Keep for local heuristic

    try:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, 'device') and model.device.type != 'meta' and model.device.type == 'cpu':
            calc_device = model.device
    except AttributeError:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for heuristic calculations: {calc_device}")

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if output_layer is None or not hasattr(output_layer, 'weight'):
        print("Error: Output embeddings layer missing or has no weights.")
        return
    if input_layer is None or not hasattr(input_layer, 'weight'):
        print("Error: Input embeddings layer missing or has no weights.")
        return

    with torch.no_grad():
        original_input_embeddings = input_layer.weight.clone()
        original_output_embeddings = output_layer.weight.clone() # Clone output
        embed_dim = original_input_embeddings.shape[1]
        if original_output_embeddings.shape[1] != embed_dim:
             print(f"Error: Input ({embed_dim}) / Output ({original_output_embeddings.shape[1]}) dimensions mismatch.")
             return

        new_vocab_size = len(new_tokenizer)
        if pad_to_multiple_of <= 0: pad_to_multiple_of = 8
        padded_size = math.ceil(new_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
        padded_size = max(padded_size, new_vocab_size)

        # Initialize input/output separately
        new_input_embeds = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')
        new_output_embeds = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')
        in_mean, in_std = original_input_embeddings.mean().item(), original_input_embeddings.std().item()
        out_mean, out_std = original_output_embeddings.mean().item(), original_output_embeddings.std().item()
        if in_std < 1e-4: in_std = 0.02
        if out_std < 1e-4: out_std = 0.02
        new_input_embeds.normal_(mean=in_mean, std=in_std)
        new_output_embeds.normal_(mean=out_mean, std=out_std)
        print(f"Initialized new input ({padded_size}x{embed_dim})")
        print(f"Initialized new output ({padded_size}x{embed_dim})")

        # Copy shared tokens
        copied_in_count, copied_out_count = 0, 0
        for token in tqdm(shared_vocab, desc="Copying shared token embeddings (Untied)"):
            old_id = old_vocab.get(token)
            new_id = new_vocab.get(token)
            if old_id is not None and new_id is not None:
                if (0 <= old_id < original_input_embeddings.shape[0]) and (0 <= new_id < new_input_embeds.shape[0]):
                     new_input_embeds[new_id] = original_input_embeddings[old_id].to(device='cpu', dtype=data_type)
                     copied_in_count += 1
                if (0 <= old_id < original_output_embeddings.shape[0]) and (0 <= new_id < new_output_embeds.shape[0]):
                     new_output_embeds[new_id] = original_output_embeddings[old_id].to(device='cpu', dtype=data_type)
                     copied_out_count += 1
        print(f"Copied {copied_in_count}/{len(shared_vocab)} shared input embeddings.")
        print(f"Copied {copied_out_count}/{len(shared_vocab)} shared output embeddings.")

        # Initialize unique tokens
        local_success_in, local_success_out = 0, 0
        global_success_in, global_success_out = 0, 0
        combined_success_in, combined_success_out = 0, 0
        random_init_count_in, random_init_count_out = 0, 0

        local_weight = 1.0 - global_weight
        use_global = global_weight > 0 and faiss_index is not None and index_to_token is not None
        use_local = local_weight > 0

        print(f"Initializing unique tokens (Untied).")
        print(f" Global heuristic: enabled={use_global}, weight={global_weight:.2f}, K={k}, sim_thresh={similarity_threshold:.2f}, conf_thresh={min_confidence_threshold:.2f}")
        print(f" Local heuristic: enabled={use_local}, weight={local_weight:.2f}, temp={calc_temperature:.2f}")


        unique_tokens_list = sorted(list(unique_tokens))
        for token_str in tqdm(unique_tokens_list, desc="Initializing unique tokens (Untied Hybrid)"):
            new_id = new_vocab.get(token_str)
            if new_id is None or not (0 <= new_id < new_input_embeds.shape[0]): continue

            e_local_in, e_local_out = None, None
            e_global_in, e_global_out = None, None
            local_calc_done, global_calc_done = False, False

            # Calculate local
            if use_local:
                try:
                    e_local_in, e_local_out = calculate_local_embedding(
                        token_str, new_id, new_tokenizer, old_tokenizer,
                        full_token_embeds_cache, subtoken_embeds_cache,
                        original_input_embeddings, original_output_embeddings, # Pass both
                        calc_temperature, data_type, calc_device # Use temperature
                    )
                    local_calc_done = True
                    if e_local_in is not None: local_success_in += 1
                    if e_local_out is not None: local_success_out += 1
                except Exception as e: print(f"Error local calc for '{token_str}': {repr(e)}")

            # Calculate global
            if use_global:
                try:
                    full_token_decoded = new_tokenizer.decode([new_id], skip_special_tokens=True)
                    if isinstance(full_token_decoded, str) and len(full_token_decoded) > 0:
                        e_global_in, e_global_out = calculate_global_embedding( # Pass both original embeddings
                            full_token_decoded, full_token_embeds_cache, faiss_index,
                            index_to_token, old_vocab,
                            original_input_embeddings, original_output_embeddings, # Pass both
                            k,
                            # temperature, # Not passed to new global calc
                            similarity_threshold,       # Pass NEW threshold
                            min_confidence_threshold,   # Pass NEW threshold
                            data_type, calc_device
                        )
                        global_calc_done = True
                        if e_global_in is not None: global_success_in += 1
                        if e_global_out is not None: global_success_out += 1
                except Exception as e: print(f"Error global calc for '{token_str}': {repr(e)}")

            # Combine for Input
            final_embedding_in = None
            if e_local_in is not None and e_global_in is not None:
                final_embedding_in = (local_weight * e_local_in + global_weight * e_global_in).to(dtype=data_type)
                combined_success_in += 1
            elif e_local_in is not None and use_local: final_embedding_in = e_local_in.to(dtype=data_type)
            elif e_global_in is not None and use_global: final_embedding_in = e_global_in.to(dtype=data_type)

            if final_embedding_in is not None:
                new_input_embeds[new_id] = final_embedding_in.cpu()
            elif local_calc_done or global_calc_done: random_init_count_in += 1

            # Combine for Output
            final_embedding_out = None
            if e_local_out is not None and e_global_out is not None:
                final_embedding_out = (local_weight * e_local_out + global_weight * e_global_out).to(dtype=data_type)
                combined_success_out += 1
            elif e_local_out is not None and use_local: final_embedding_out = e_local_out.to(dtype=data_type)
            elif e_global_out is not None and use_global: final_embedding_out = e_global_out.to(dtype=data_type)

            if final_embedding_out is not None:
                new_output_embeds[new_id] = final_embedding_out.cpu()
            elif local_calc_done or global_calc_done: random_init_count_out += 1

        print(f"Untied initialization results for {len(unique_tokens_list)} unique tokens:")
        print(f" >>> Input Embeddings:")
        print(f"     - Local heuristic success: {local_success_in}")
        print(f"     - Global heuristic success: {global_success_in} (passing confidence threshold)")
        print(f"     - Combined success (weighted): {combined_success_in}")
        print(f"     - Randomly initialized (after attempts): {random_init_count_in}")
        print(f" >>> Output Embeddings:")
        print(f"     - Local heuristic success: {local_success_out}")
        print(f"     - Global heuristic success: {global_success_out} (passing confidence threshold)")
        print(f"     - Combined success (weighted): {combined_success_out}")
        print(f"     - Randomly initialized (after attempts): {random_init_count_out}")

        # Resize and copy
        print("Resizing model token embeddings (Untied)...")
        for param in model.parameters(): param.requires_grad = False
        try:
            model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        except Exception as e: # Add manual check
            print(f"Error during resize: {e}")
            current_in = model.get_input_embeddings().weight.shape[0]
            current_out = model.get_output_embeddings().weight.shape[0]
            if current_in != padded_size or current_out != padded_size:
                print(f"FATAL: Size mismatch after resize attempt ({current_in}/{current_out} vs {padded_size}). Stopping.")
                return
            else: print("Manual check ok.")
        print(f"Embedding sizes after resize: Input {model.get_input_embeddings().weight.shape}, Output {model.get_output_embeddings().weight.shape}")

        target_input_layer = model.get_input_embeddings()
        target_output_layer = model.get_output_embeddings()
        target_device_in, target_dtype_in = target_input_layer.weight.device, target_input_layer.weight.dtype
        target_device_out, target_dtype_out = target_output_layer.weight.device, target_output_layer.weight.dtype
        print(f"Target device/dtype: Input ({target_device_in}, {target_dtype_in}), Output ({target_device_out}, {target_dtype_out})")

        new_input_tensor = new_input_embeds.to(target_device_in, dtype=target_dtype_in)
        new_output_tensor = new_output_embeds.to(target_device_out, dtype=target_dtype_out)

        # Copy Input
        if new_input_tensor.shape == target_input_layer.weight.shape:
             target_input_layer.weight.copy_(new_input_tensor)
             print("Copied initialized embeddings to input layer.")
        else:
             print(f"FATAL: Final shape mismatch for input. Expected {target_input_layer.weight.shape}, got {new_input_tensor.shape}.")
             return

        # Copy Output
        if new_output_tensor.shape == target_output_layer.weight.shape:
             target_output_layer.weight.copy_(new_output_tensor)
             print("Copied initialized embeddings to output layer.")
        else:
             print(f"FATAL: Final shape mismatch for output. Expected {target_output_layer.weight.shape}, got {new_output_tensor.shape}.")
             return # Avoid inconsistent state

        print("Untied embedding update complete.")
