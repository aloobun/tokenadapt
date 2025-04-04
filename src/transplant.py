# transplant.py
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.
# It orchestrates the transplantation process, determining whether embeddings are tied or untied,
# and calls the appropriate transplantation function from tied.py or untied.py.
# It also handles caching of embeddings for full tokens and subtokens using cache.py.


import torch
import faiss
import numpy as np
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
from tqdm.auto import tqdm
from tied import transplant_tied_embeddings
from untied import transplant_untied_embeddings
from cache import load_cache, save_cache, cache_embeddings


# build_faiss_index remains the same as latest version previously provided
def build_faiss_index(embeddings_dict: dict, embed_dim: int):
    """
    Builds a FAISS IndexFlatIP for efficient cosine similarity search.
    """
    print("Building FAISS index for old vocabulary embeddings...")
    start_time = time.time()
    token_list = []
    embedding_matrix_list = []
    for token, embed_list in tqdm(embeddings_dict.items(), desc="Preparing vectors for FAISS"):
        try:
            embed_np = np.array(embed_list, dtype=np.float32)
            if embed_np.shape == (embed_dim,):
                token_list.append(token)
                embedding_matrix_list.append(embed_np)
            else:
                print(f"Warning: Skipping token '{token}' during FAISS build due to unexpected embedding shape {embed_np.shape}. Expected ({embed_dim},)")
        except Exception as e:
            print(f"Warning: Skipping token '{token}' during FAISS build due to error during conversion: {e}")
    if not embedding_matrix_list:
        print("Error: No valid embeddings found to build FAISS index.")
        return None, None
    embedding_matrix = np.vstack(embedding_matrix_list)
    print(f"Prepared {embedding_matrix.shape[0]} vectors for indexing.")
    print("Normalizing vectors (L2 norm) for cosine similarity with IndexFlatIP...")
    faiss.normalize_L2(embedding_matrix)
    print(f"Creating FAISS IndexFlatIP with dimension {embed_dim}...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embedding_matrix)
    index_to_token = {i: token for i, token in enumerate(token_list)}
    end_time = time.time()
    print(f"FAISS index built successfully with {index.ntotal} vectors in {end_time - start_time:.2f} seconds.")
    return index, index_to_token


def main(args):
    """Main function to execute the tokenizer transplantation process."""
    # --------------- Setup ------------------
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    print(f"Data type selected: {args.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")
    print(f"FAISS operations will primarily use CPU.")

    # --------------- Loading Models and Tokenizers ---------------
    print("Loading pre-trained model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cpu", token=args.hf_token
    )

    old_generation_config = None
    if hasattr(model, "generation_config"):
        try:
            old_generation_config = model.generation_config
            print("Stored original generation config.")
        except Exception as e:
            print(f"Warning: Could not retrieve original generation config: {e}")

    print("Loading tokenizers...")
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_path, token=args.hf_token)

    print("Loading embedding model...")
    embed_model = AutoModel.from_pretrained(args.embedding_model_path, trust_remote_code=True).to(device)
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path, trust_remote_code=True)

    # --------------- Setting Up Global Heuristic -----------------
    try:
        embed_dim_external = embed_model.config.hidden_size
        print(f"External embedding model dimension (for FAISS): {embed_dim_external}")
    except AttributeError:
        print("Warning: Could not automatically determine embedding dimension from embed_model.config.hidden_size.")
        embed_dim_external = None

    # --------------- Transplant Start Phase 1: Caching -------------------------
    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    shared_vocab = list(set(new_vocab.keys()) & set(old_vocab.keys()))
    unique_tokens = set(new_vocab.keys()) - set(shared_vocab)
    print(f"Shared tokens: {len(shared_vocab)}")
    print(f"Unique tokens to initialize: {len(unique_tokens)}")

    embed_model_name = args.embedding_model_path.replace('/', '_') # Safe filename
    cache_file = f"cache_{embed_model_name}.json"
    cache = load_cache(cache_file)

    # Cache embeddings for the unique *new* tokens
    full_tokens_to_cache = [new_tokenizer.decode([new_vocab[token_str]], skip_special_tokens=True)
                            for token_str in unique_tokens if token_str in new_vocab]
    full_tokens_to_cache = [t for t in full_tokens_to_cache if isinstance(t, str) and len(t) > 0]
    cache = cache_embeddings(embed_model, embed_tokenizer, list(set(full_tokens_to_cache)), device,
                                                    cache, batch_size=args.batch_size)
    full_token_embeds_cache = {token: cache[token] for token in full_tokens_to_cache if token in cache}
    print(f"Cached/loaded embeddings for {len(full_token_embeds_cache)} unique new tokens.")

    # Cache embeddings for potential *old* subtokens
    subtokens_to_cache = set()
    for token_str in tqdm(unique_tokens, desc="Gathering potential subtokens"):
        if token_str not in new_vocab: continue
        full_token_decoded = new_tokenizer.decode([new_vocab[token_str]], skip_special_tokens=True)
        if not isinstance(full_token_decoded, str) or len(full_token_decoded) == 0: continue
        try:
            old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
            subtokens_to_cache.update(old_tokenizer.decode([oid], skip_special_tokens=True) for oid in old_ids)
        except Exception as e:
            print(f"Warning: Could not encode/decode token '{token_str}' with old tokenizer: {e}")
    subtokens_to_cache = {t for t in subtokens_to_cache if isinstance(t, str) and len(t) > 0}
    cache = cache_embeddings(embed_model, embed_tokenizer, list(subtokens_to_cache), device,
                                                            cache, batch_size=args.batch_size)
    subtoken_embeds_cache = {token: cache[token] for token in subtokens_to_cache if token in cache}
    print(f"Cached/loaded embeddings for {len(subtoken_embeds_cache)} potential subtokens.")

    # Cache embeddings for the *entire old* vocabulary (needed for FAISS)
    # *** CORRECTED LOGIC HERE ***
    old_vocab_token_strings = [t for t in old_vocab.keys() if isinstance(t, str) and len(t) > 0]
    old_vocab_tokens_to_cache = [token for token in old_vocab_token_strings if token not in cache]
    cache = cache_embeddings(embed_model, embed_tokenizer, old_vocab_tokens_to_cache, device,
                                                             cache, batch_size=args.batch_size)
    # Build dict for FAISS using token strings as keys
    old_vocab_embeds_for_index = {token: cache[token] for token in old_vocab_token_strings if token in cache}
    print(f"Cached/loaded embeddings for {len(old_vocab_embeds_for_index)} old vocabulary tokens (for FAISS).")

    # Infer embedding dimension if needed
    if embed_dim_external is None:
        print("Attempting to infer external embedding dimension from cached data...")
        # (Inference logic remains the same as latest version provided previously)
        check_caches = [old_vocab_embeds_for_index, full_token_embeds_cache, subtoken_embeds_cache]
        found_dim = False
        for cache_dict in check_caches:
            if not cache_dict: continue
            try:
                first_available_embedding = next(iter(cache_dict.values()), None)
                if first_available_embedding and isinstance(first_available_embedding, list) and len(first_available_embedding) > 0:
                    embed_dim_external = len(first_available_embedding)
                    print(f"Inferred external embedding dimension from cache: {embed_dim_external}")
                    found_dim = True
                    break
            except Exception as e: print(f"Warning: Error checking cache for dimension: {e}")
        if not found_dim and cache:
             try:
                 first_available_embedding = next(iter(cache.values()), None)
                 if first_available_embedding and isinstance(first_available_embedding, list) and len(first_available_embedding) > 0:
                      embed_dim_external = len(first_available_embedding)
                      print(f"Inferred external embedding dimension from master cache: {embed_dim_external}")
                      found_dim = True
             except Exception as e: print(f"Warning: Error inferring dimension from master cache: {e}")
        if not found_dim:
             print("Error: Cannot determine external embedding dimension. Global heuristic disabled.")
             args.weight = 0.0
             faiss_index = None
             index_to_token = None

    # Save cache
    save_cache(cache_file, cache)

    # Build FAISS index
    faiss_index, index_to_token = None, None
    if embed_dim_external is not None and old_vocab_embeds_for_index:
         faiss_index, index_to_token = build_faiss_index(old_vocab_embeds_for_index, embed_dim_external)
    else:
         print("Skipping FAISS index build: Dimension unknown or no old embeddings.")

    if faiss_index is None and args.weight > 0.0:
        print("Warning: FAISS index could not be built. Disabling Global heuristic (setting global weight to 0.0).")
        args.weight = 0.0
        index_to_token = None

    # Determine if tied
    tied = getattr(model.config, "tie_word_embeddings", False)
    if not tied: # Robust check
        try:
            input_embeds = model.get_input_embeddings()
            output_embeds = model.get_output_embeddings()
            if output_embeds is None: tied = True
            elif input_embeds is not None and hasattr(input_embeds, 'weight') and hasattr(output_embeds, 'weight'):
                 tied = input_embeds.weight is output_embeds.weight
                 if not tied and torch.equal(input_embeds.weight.data, output_embeds.weight.data):
                      print("Warning: Input/output weights separate but identical. Treating as tied.")
                      tied = True
        except AttributeError: pass # Keep config value if check fails
    print(f"Tied embeddings detected: {tied}")

    # --------------- Transplant Phase 2 -------------------------
    transplant_kwargs = {
        "model": model,
        "new_tokenizer": new_tokenizer,
        "shared_vocab": shared_vocab,
        "unique_tokens": unique_tokens,
        "full_token_embeds_cache": full_token_embeds_cache,
        "subtoken_embeds_cache": subtoken_embeds_cache,
        "old_vocab": old_vocab,
        "new_vocab": new_vocab,
        "old_tokenizer": old_tokenizer, # Pass old_tokenizer here
        "data_type": dtype,
        "temperature": args.temperature, # For local heuristic
        "pad_to_multiple_of": args.multiple_of,
        "faiss_index": faiss_index,
        "index_to_token": index_to_token,
        "k": args.top_k,
        "global_weight": args.weight,
        "similarity_threshold": args.similarity_threshold,          # *** NEW ***
        "min_confidence_threshold": args.min_confidence_threshold   # *** NEW ***
    }

    global_heuristic_enabled = args.weight > 0 and faiss_index is not None
    print(f"Proceeding with transplantation (Tied: {tied}).")
    print(f" Global heuristic: enabled={global_heuristic_enabled}, weight={args.weight:.2f}, K={args.top_k}, sim_thresh={args.similarity_threshold:.2f}, conf_thresh={args.min_confidence_threshold:.2f}")
    print(f" Local heuristic: enabled={args.weight < 1.0}, weight={1.0-args.weight:.2f}, temp={args.temperature:.2f}")


    if tied:
        transplant_tied_embeddings(**transplant_kwargs)
    else:
        if model.get_output_embeddings() is None or model.get_input_embeddings() is None:
             print("Error: Model detected as untied, but input/output embeddings layer missing.")
             return
        else:
             transplant_untied_embeddings(**transplant_kwargs)

    # ------------- Clean-Up & Saving -----------------------
    print("Finalizing model configuration...")
    try: # (Configuration logic remains the same as latest version)
        eos_id = getattr(new_tokenizer, "eos_token_id", None)
        bos_id = getattr(new_tokenizer, "bos_token_id", None)
        pad_id = getattr(new_tokenizer, "pad_token_id", None)
        if pad_id is None: pad_id = eos_id
        if pad_id is None and hasattr(new_tokenizer, 'unk_token_id'): pad_id = new_tokenizer.unk_token_id
        config_updates = {}
        if pad_id is not None: config_updates["pad_token_id"] = pad_id
        if eos_id is not None: config_updates["eos_token_id"] = eos_id
        if bos_id is not None: config_updates["bos_token_id"] = bos_id
        config_updates["vocab_size"] = len(new_tokenizer)
        print(f"Updating model config with: {config_updates}")
        model.config.update(config_updates)
        if old_generation_config is not None:
            try:
                gen_config_dict = old_generation_config.to_dict()
                if "pad_token_id" in config_updates: gen_config_dict["pad_token_id"] = config_updates["pad_token_id"]
                if "eos_token_id" in config_updates: gen_config_dict["eos_token_id"] = config_updates["eos_token_id"]
                if "bos_token_id" in config_updates: gen_config_dict["bos_token_id"] = config_updates["bos_token_id"]
                model.generation_config = GenerationConfig.from_dict(gen_config_dict)
                print("Updated existing generation config.")
            except Exception as e:
                print(f"Warning: Failed to update existing generation config: {e}.")
                model.generation_config = GenerationConfig.from_model_config(model.config)
                print("Created new generation config from model config.")
        else:
            model.generation_config = GenerationConfig.from_model_config(model.config)
            print("Created new generation config from model config.")
        if hasattr(old_tokenizer, "chat_template") and old_tokenizer.chat_template and hasattr(new_tokenizer, "chat_template"):
            try: new_tokenizer.chat_template = old_tokenizer.chat_template; print("Copied chat template.")
            except Exception as e: print(f"Warning: Could not copy chat template: {e}")
    except Exception as e: print(f"Warning: Post-transplant configuration update failed: {e}")

    print(f"Saving model and tokenizer to Hugging Face hub: {args.new_model_name}...")
    try:
        model.to('cpu')
        model.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
        new_tokenizer.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
        print("Transplantation, configuration, and upload completed successfully!")
    except Exception as e:
        print(f"Error during Hugging Face Hub upload: {e}")

#  ------------- End ------------------

if __name__ == "__main__":
    # Range validator class (same as before)
    class Range(object):
        def __init__(self, start, end): self.start, self.end = start, end
        def __eq__(self, other):
            try: return self.start <= float(other) <= self.end
            except ValueError: return False
        def __repr__(self): return f"[{self.start}, {self.end}]"

    parser = argparse.ArgumentParser(description="Tokenizer Transplantation Tool")
    # --- Arguments ---
    parser.add_argument("-model", "--model_path", required=True, help="Original model path/ID")
    parser.add_argument("-tk", "--new_tokenizer_path", required=True, help="New tokenizer path/ID")
    parser.add_argument("-embed", "--embedding_model_path", default="nomic-ai/nomic-embed-text-v2-moe", help="External embedding model path/ID (default: nomic-ai/nomic-embed-text-v2-moe)")
    parser.add_argument("-repo", "--new_model_name", required=True, help="HF repo name for the new model")
    parser.add_argument("-auth", "--hf_token", required=True, help="HF auth token")
    parser.add_argument("-temp", "--temperature", default=0.3, type=float, choices=[Range(0.01, 5.0)], help="Temperature for local heuristic softmax (default: 0.3)")
    parser.add_argument("-pad","--multiple_of" , default = 128, type=int, help="Pad vocab size to multiple of this value (default: 128)")
    parser.add_argument("-d", "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation data type (default: bf16)")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="Batch size for embedding extraction (default: 16)")
    parser.add_argument("-k","--top_k", default=10, type=int, help="Initial K neighbors for global heuristic (default: 10)")
    parser.add_argument("-w", "--weight", default=0.3, type=float, choices=[Range(0.0, 1.0)], help="Global heuristic weight (default: 0.3)")
    # *** UPDATED/NEW ARGUMENTS for Global Heuristic ***
    parser.add_argument("-st", "--similarity_threshold", default=0.6, type=float, choices=[Range(0.0, 1.0)], help="General similarity threshold for global neighbors (default: 0.6)")
    parser.add_argument("-mct", "--min_confidence_threshold", default=0.4, type=float, choices=[Range(0.0, 1.0)], help="Minimum confidence threshold (for best neighbor) in global heuristic (default: 0.4)")
    # *** REMOVED OLD "-limit" / "--threshold" argument ***

    args = parser.parse_args()

    # Validation: Ensure min_confidence_threshold is not greater than similarity_threshold
    if args.min_confidence_threshold > args.similarity_threshold:
        print(f"Warning: min_confidence_threshold ({args.min_confidence_threshold}) > similarity_threshold ({args.similarity_threshold}). Setting min_confidence_threshold = similarity_threshold.")
        args.min_confidence_threshold = args.similarity_threshold

    main(args)
