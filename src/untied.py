
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
from typing import Optional , Dict, Set, List

def transplant_untied_embeddings(
    model, new_tokenizer: AutoTokenizer, shared_tokens_map: Dict[int, int], unique_tokens: set,
    full_token_embeds_cache: dict, subtoken_embeds_cache: dict, old_vocab: dict,
    new_vocab: dict, old_tokenizer: AutoTokenizer, data_type: torch.dtype,
    temperature: float, pad_to_multiple_of: int,
    faiss_index: Optional[faiss.Index], index_to_token: Optional[dict], k: int, global_weight: float,
    threshold: float
    ) -> None:
    """
    Transplants embeddings for a model with untied input/output embeddings.
    Uses heuristic helpers, calculating weights once and applying to both layers.
    """

    eps = 1e-5
    calc_temperature = temperature + eps

    try:
        calc_device = model.device if model.device.type != 'meta' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except AttributeError:
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for heuristic calculations: {calc_device}")

    output_layer = model.get_output_embeddings()
    if output_layer is None:
        print("Error: Cannot perform untied transplantation because model.get_output_embeddings() is None.")
        return

    with torch.no_grad():
        original_input_embeddings = model.get_input_embeddings().weight.clone()
        original_output_embeddings = output_layer.weight.clone()
        embed_dim = original_input_embeddings.shape[1]

        new_vocab_size = len(new_tokenizer)
        padded_size = math.ceil(new_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
        new_input_embeds = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')
        new_output_embeds = torch.empty(padded_size, embed_dim, dtype=data_type, device='cpu')

        in_mean, in_std = original_input_embeddings.mean().item(), original_input_embeddings.std().item()
        out_mean, out_std = original_output_embeddings.mean().item(), original_output_embeddings.std().item()
        new_input_embeds.normal_(mean=in_mean, std=in_std)
        new_output_embeds.normal_(mean=out_mean, std=out_std)
        print(f"Initialized new input/output embedding matrices with size {padded_size}x{embed_dim}")


        copied_in_count, copied_out_count = 0, 0
        for new_id, old_id in tqdm(shared_tokens_map.items(), desc="Copying shared token embeddings (Untied)"):
            if not (0 <= new_id < new_input_embeds.shape[0]): continue
            if 0 <= old_id < original_input_embeddings.shape[0]:
                 new_input_embeds[new_id] = original_input_embeddings[old_id].to(device='cpu', dtype=data_type)
                 copied_in_count += 1

            if 0 <= old_id < original_output_embeddings.shape[0]:
                 new_output_embeds[new_id] = original_output_embeddings[old_id].to(device='cpu', dtype=data_type)
                 copied_out_count += 1

        print(f"Copied {copied_in_count}/{len(shared_tokens_map)} shared input embeddings.")
        print(f"Copied {copied_out_count}/{len(shared_tokens_map)} shared output embeddings.")


        local_success_in, local_success_out = 0, 0
        global_success_in, global_success_out = 0, 0
        combined_success_in, combined_success_out = 0, 0
        random_init_count_in, random_init_count_out = 0, 0

        local_weight = 1.0 - global_weight
        use_global = global_weight > 0 and faiss_index is not None
        use_local = local_weight > 0

        print(f"Initializing unique tokens (Untied). Global heuristic enabled: {use_global} (weight={global_weight:.2f}), Local heuristic enabled: {use_local} (weight={local_weight:.2f})")

        for token_str in tqdm(unique_tokens, desc="Initializing unique tokens (Untied Hybrid)"):
            new_id = new_vocab.get(token_str)
            if new_id is None: continue

            e_local_in, e_local_out = None, None
            e_global_in, e_global_out = None, None

            
            if use_local:
                e_local_in, e_local_out = calculate_local_embedding(
                    token_str=token_str, new_token_id=new_id,
                    new_tokenizer=new_tokenizer, old_tokenizer=old_tokenizer,
                    full_token_embeds_cache=full_token_embeds_cache, subtoken_embeds_cache=subtoken_embeds_cache,
                    original_input_embeddings=original_input_embeddings,
                    original_output_embeddings=original_output_embeddings, 
                    temperature=calc_temperature, threshold=threshold, 
                    data_type=data_type, device=calc_device
                )
                if e_local_in is not None: local_success_in += 1
                if e_local_out is not None: local_success_out += 1 

            
            if use_global:
                 try:
                     full_token_decoded = new_tokenizer.decode([new_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                     if full_token_decoded:
                         e_global_in, e_global_out = calculate_global_embedding(
                             query_token_str=full_token_decoded,
                             full_token_embeds_cache=full_token_embeds_cache, faiss_index=faiss_index,
                             old_tokenizer=old_tokenizer, index_to_token=index_to_token, old_vocab=old_vocab,
                             original_input_embeddings=original_input_embeddings,
                             original_output_embeddings=original_output_embeddings,
                             k=k, temperature=calc_temperature, threshold=threshold, 
                             data_type=data_type, device=calc_device
                         )
                         if e_global_in is not None: global_success_in += 1
                         if e_global_out is not None: global_success_out += 1

                 except Exception as e:
                     # print(f"Warning: Error during global calculation setup for '{token_str}': {e}")
                     pass

            
            final_embedding_in = None
            if e_local_in is not None and e_global_in is not None:
                final_embedding_in = (local_weight * e_local_in + global_weight * e_global_in).to(dtype=data_type)
                combined_success_in += 1
            elif e_local_in is not None: final_embedding_in = e_local_in.to(dtype=data_type)
            elif e_global_in is not None: final_embedding_in = e_global_in.to(dtype=data_type)

            if final_embedding_in is not None:
                new_input_embeds[new_id] = final_embedding_in.cpu()
            else:
                random_init_count_in += 1

            
            final_embedding_out = None
            if e_local_out is not None and e_global_out is not None:
                final_embedding_out = (local_weight * e_local_out + global_weight * e_global_out).to(dtype=data_type)
                combined_success_out += 1
            elif e_local_out is not None: final_embedding_out = e_local_out.to(dtype=data_type)
            elif e_global_out is not None: final_embedding_out = e_global_out.to(dtype=data_type)

            if final_embedding_out is not None:
                new_output_embeds[new_id] = final_embedding_out.cpu()
            else:
                random_init_count_out += 1


        print(f"Untied initialization complete for {len(unique_tokens)} unique tokens:")
        print(f">  Input Embeddings:")
        print(f"    - Local heuristic succeeded for: {local_success_in}")
        print(f"    - Global heuristic succeeded for: {global_success_in}")
        print(f"    - Combined successfully (both ran & succeeded): {combined_success_in}")
        print(f"    - Remained randomly initialized: {random_init_count_in}")
        print(f">  Output Embeddings:")
        print(f"    - Local heuristic succeeded for: {local_success_out}")
        print(f"    - Global heuristic succeeded for: {global_success_out}")
        print(f"    - Combined successfully (both ran & succeeded): {combined_success_out}")
        print(f"    - Remained randomly initialized: {random_init_count_out}")


        
        print("Resizing model token embeddings (Untied)...")
        for param in model.parameters(): param.requires_grad = False
        model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=pad_to_multiple_of)
        print(f"Model embedding size after resize: Input {model.get_input_embeddings().weight.shape}, Output {model.get_output_embeddings().weight.shape}")

        target_device_in = model.get_input_embeddings().weight.device
        target_device_out = model.get_output_embeddings().weight.device
        target_dtype_in = model.get_input_embeddings().weight.dtype
        target_dtype_out = model.get_output_embeddings().weight.dtype

        new_input_tensor = new_input_embeds.to(target_device_in, dtype=target_dtype_in)
        new_output_tensor = new_output_embeds.to(target_device_out, dtype=target_dtype_out)

        if new_input_tensor.shape == model.get_input_embeddings().weight.shape:
             model.get_input_embeddings().weight.copy_(new_input_tensor)
        else:
             print(f"Error: Shape mismatch for input embeddings. Expected {model.get_input_embeddings().weight.shape}, got {new_input_tensor.shape}.")

        if new_output_tensor.shape == model.get_output_embeddings().weight.shape:
             model.get_output_embeddings().weight.copy_(new_output_tensor)
        else:
             print(f"Error: Shape mismatch for output embeddings. Expected {model.get_output_embeddings().weight.shape}, got {new_output_tensor.shape}.")

        print("Untied embedding update complete.")