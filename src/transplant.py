# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.
# It orchestrates the transplantation process, determining whether embeddings are tied or untied,
# and calls the appropriate transplantation function from tied.py or untied.py.
# It also handles caching of embeddings for full tokens and subtokens using cache.py.


import torch ; import faiss ; import numpy as np ; import time ; import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from tied import transplant_tied_embeddings
from untied import transplant_untied_embeddings
from cache import load_cache, save_cache, cache_embeddings



def build_faiss_index(embeddings_dict: dict, embed_dim: int):
    """
    Builds a FAISS IndexFlatIP for efficient cosine similarity search.

    Args:
        embeddings_dict: Dictionary mapping token strings to their embedding lists.
        embed_dim: The dimensionality of the embeddings.

    Returns:
        A tuple containing:
        - faiss.Index | None: The built FAISS index, or None if failed.
        - dict | None: A mapping from FAISS index ID back to the original token string, or None.
    """
    print("Building FAISS index for old vocabulary embeddings...")
    start_time = time.time()

    token_list = []
    embedding_matrix_list = []

    for token, embed_list in tqdm(embeddings_dict.items(), desc="Preparing vectors for FAISS"):
        try:
            # fp32
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
    num_vectors = embedding_matrix.shape[0]
    print(f"Prepared {num_vectors} vectors for indexing.")

    
    print("Normalizing vectors (L2 norm)...")
    faiss.normalize_L2(embedding_matrix) 

    
    print(f"Creating FAISS IndexFlatIP with dimension {embed_dim}...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embedding_matrix) 

    # Map FAISS index ID back to original token string
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
    
    old_generation_config = model.generation_config

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

    # --------------- Transplant Start Phase 1 -------------------------

    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()

    decoded_lower_old_map = {}
    print("Building map of decoded, lowercase old vocabulary...")
    for old_token_str, old_token_id in tqdm(old_vocab.items(), desc="Decoding old vocab"):
        try:
            decoded_str = old_tokenizer.decode([old_token_id],
                                            skip_special_tokens=False,
                                            clean_up_tokenization_spaces=True).strip() # Strip whitespace to better normalize "_(spm) or Ġ(bpe)"

            if decoded_str: 
                decoded_lower = decoded_str.lower()
                # Store the mapping. If collisions occur (e.g., "Apple" and "apple" in old vocab),
                # this will keep the ID of the *last* one encountered. This is usually acceptable.
                decoded_lower_old_map[decoded_lower] = old_token_id
        except Exception as e:
            # print(f"Warning: Could not decode old token: '{old_token_str}' (ID: {old_token_id}). Error: {e}")
            pass 

    print(f"Built map with {len(decoded_lower_old_map)} unique decoded lowercase old tokens.")

    shared_tokens_map = {}  # Maps new_token_id -> old_token_id for copying
    unique_tokens = set() 

    exact_matches = 0
    case_insensitive_matches = 0
    decode_failures_new = 0

    print("Mapping new vocabulary tokens...")
    for new_token_str, new_token_id in tqdm(new_vocab.items(), desc="Mapping new tokens"):
        matched = False
        if new_token_str in old_vocab:
            shared_tokens_map[new_token_id] = old_vocab[new_token_str]
            exact_matches += 1
            matched = True
        else:
            try:
                decoded_new = new_tokenizer.decode([new_token_id],
                                                skip_special_tokens=False,
                                                clean_up_tokenization_spaces=True).strip()

                if decoded_new:
                    decoded_lower_new = decoded_new.lower()
                    if decoded_lower_new in decoded_lower_old_map:
                        shared_tokens_map[new_token_id] = decoded_lower_old_map[decoded_lower_new]
                        case_insensitive_matches += 1
                        matched = True

            except Exception as e:
                # print(f"Warning: Could not decode new token: '{new_token_str}' (ID: {new_token_id}). Error: {e}")
                decode_failures_new += 1

        if not matched:
            unique_tokens.add(new_token_str)

    total_shared = exact_matches + case_insensitive_matches
    print(f"Token mapping complete:")
    print(f"  - Exact raw string matches (copied): {exact_matches}")
    print(f"  - Decoded case-insensitive matches (copied): {case_insensitive_matches}")
    print(f"  - Total shared/copied tokens: {total_shared}")
    print(f"  - Unique tokens needing initialization: {len(unique_tokens)}")
    if decode_failures_new > 0:
        print(f"  - New tokens failed to decode (treated as unique): {decode_failures_new}")


    # --------------- Caching Embeddings -----------------

    full_tokens_to_cache = []
    print("Gathering decoded strings for unique tokens needing embeddings...")
    for token_str in tqdm(unique_tokens, desc="Decoding unique tokens"):
        new_token_id = new_vocab.get(token_str)
        if new_token_id is None: continue
        try:
            decoded = new_tokenizer.decode([new_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            if decoded or isinstance(decoded, str):
                full_tokens_to_cache.append(decoded)
            # else: 
            #     print(f"Warning: Decoding unique token '{token_str}' (ID: {new_token_id}) resulted in non-string. Skipping cache.")

        except Exception as e:
            print(f"Warning: Could not decode unique token string '{token_str}' (ID: {new_token_id}) for caching: {e}")
            
    full_tokens_to_cache = list(set(full_tokens_to_cache))



    
    embed_model_name = args.embedding_model_path.split("/")[-1]
    cache_file = f"cache_{embed_model_name}.json"

    
    cache = load_cache(cache_file)


    # --------------- Full-Token Caching -----------------

    print(f"Caching external embeddings for {len(full_tokens_to_cache)} unique decoded token strings...")
    cache = cache_embeddings(embed_model, embed_tokenizer, full_tokens_to_cache, device,
                                                    cache, batch_size=args.batch_size)
    full_token_embeds_cache = {token: cache[token] for token in full_tokens_to_cache if token in cache}
    print(f"Cached embeddings obtained for {len(full_token_embeds_cache)} unique decoded strings.")


    # --------------- Subtoken Caching -----------------

    subtokens_to_cache_decoded = set()
    print("Gathering potential old subtokens for local heuristic...")
    for token_str in tqdm(unique_tokens, desc="Gathering potential subtokens"):
        new_token_id = new_vocab.get(token_str)
        if new_token_id is None: continue
        try:
            full_token_decoded = new_tokenizer.decode([new_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            if full_token_decoded or isinstance(full_token_decoded, str):
                old_ids = old_tokenizer.encode(full_token_decoded, add_special_tokens=False)
                for oid in old_ids:
                    try:
                        subtoken_decoded = old_tokenizer.decode([oid], skip_special_tokens=False, clean_up_tokenization_spaces=True)
                        if subtoken_decoded or isinstance(subtoken_decoded, str):
                            subtokens_to_cache_decoded.add(subtoken_decoded)
                    except Exception as e:
                        # print(f"Warning: Could not decode old subtoken ID {oid}: {e}")
                        pass
            # else: 
            #     print(f"Warning: Skipping subtokens for unique token '{token_str}' as it didn't decode.")

        except Exception as e:
            print(f"Warning: Error processing unique token '{token_str}' for subtokens: {e}")

    
    print(f"Caching external embeddings for {len(subtokens_to_cache_decoded)} potential old subtoken strings...")
    cache = cache_embeddings(embed_model, embed_tokenizer, list(subtokens_to_cache_decoded), device,
                                                            cache, batch_size=args.batch_size)
    
    subtoken_embeds_cache = {token: cache[token] for token in subtokens_to_cache_decoded if token in cache}
    print(f"Cached embeddings obtained for {len(subtoken_embeds_cache)} subtoken strings.")

    # --------------- Old Vocab Caching for FAISS -----------------

    old_vocab_tokens_to_cache = [old_tokenizer.decode([oid]) for oid in old_vocab.values() if old_tokenizer.decode([oid]) not in cache]
    cache = cache_embeddings(embed_model, embed_tokenizer, old_vocab_tokens_to_cache, device,
                                                             cache, batch_size=args.batch_size)
    

    old_vocab_embeds_for_index = {token:cache[old_tokenizer.decode([oid])] for token,oid in old_vocab.items() if old_tokenizer.decode([oid]) in cache}



    if embed_dim_external is None:
        print("Attempting to infer external embedding dimension from cached data...")
        try:
            first_available_embedding = next(iter(old_vocab_embeds_for_index.values()), None)
            if first_available_embedding:
                embed_dim_external = len(first_available_embedding)
                print(f"Inferred external embedding dimension from cache: {embed_dim_external}")
            else:
                first_available_embedding = next(iter(full_token_embeds_cache.values()), None)
                if first_available_embedding:
                     embed_dim_external = len(first_available_embedding)
                     print(f"Inferred external embedding dimension from cache: {embed_dim_external}")
                else:
                     first_available_embedding = next(iter(subtoken_embeds_cache.values()), None)
                     if first_available_embedding:
                          embed_dim_external = len(first_available_embedding)
                          print(f"Inferred external embedding dimension from cache: {embed_dim_external}")
                     else:
                          print("Error: Cannot determine external embedding dimension. No embeddings found in cache.")
                          return
        except Exception as e:
            print(f"Error inferring embedding dimension from cache: {e}")
            return 

    
    save_cache(cache_file, cache)



    faiss_index, index_to_token = build_faiss_index(old_vocab_embeds_for_index, embed_dim_external)

    if faiss_index is None:
        print("Warning: FAISS index could not be built. Disabling Global heuristic (setting global weight to 0.0).")
        args.weight = 0.0 
        index_to_token = None
    

    tied = getattr(model.config, "tie_word_embeddings", False)
    if not tied:
        input_embeds = model.get_input_embeddings().weight
        output_embeds = model.get_output_embeddings()
        tied = output_embeds is None or input_embeds is output_embeds.weight
    print(f"Tied embeddings detected: {tied}")

    # --------------- Transplant Phase 2 -------------------------

    transplant_kwargs = {
    "model": model,
    "new_tokenizer": new_tokenizer,
    "shared_tokens_map": shared_tokens_map, 
    "unique_tokens": unique_tokens, 
    "full_token_embeds_cache": full_token_embeds_cache, 
    "subtoken_embeds_cache": subtoken_embeds_cache, 
    "old_vocab": old_vocab, 
    "new_vocab": new_vocab, 
    "old_tokenizer": old_tokenizer,
    "data_type": dtype,
    "temperature": args.temperature,
    "pad_to_multiple_of": args.multiple_of,
    "faiss_index": faiss_index, 
    "index_to_token": index_to_token, 
    "k": args.top_k,
    "global_weight": args.weight,
    "threshold": args.threshold
}

    print(f"Proceeding with transplantation (Is Tied :-> {tied}). global weight: {args.weight:.2f}, K: {args.top_k}")


    if tied:
        transplant_tied_embeddings(**transplant_kwargs) 
    else:
        if model.get_output_embeddings() is None:
             print("Error: Model detected as untied, but get_output_embeddings() returned None. Cannot proceed.")
             return 
        else:
             transplant_untied_embeddings(**transplant_kwargs)

    # ------------- Clean-Up -----------------------
    print("Finalizing model configuration...")
    try:

        eos_id = getattr(new_tokenizer, "eos_token_id", None)
        bos_id = getattr(new_tokenizer, "bos_token_id", None)
        pad_id = getattr(new_tokenizer, "pad_token_id", None)

        if pad_id is None: pad_id = eos_id
        if eos_id is None: eos_id = bos_id
        if bos_id is None: bos_id = eos_id
        if pad_id is None: pad_id = bos_id

        config_updates = {}
        if pad_id is not None: config_updates["pad_token_id"] = pad_id
        if eos_id is not None: config_updates["eos_token_id"] = eos_id
        if bos_id is not None: config_updates["bos_token_id"] = bos_id

        if hasattr(old_tokenizer, "chat_template") and old_tokenizer.chat_template:
             if hasattr(new_tokenizer,"chat_template"):
                  new_tokenizer.chat_template = old_tokenizer.chat_template
                  print("Copied chat template from old tokenizer.")

        if config_updates:
             print(f"Updating model config with: {config_updates}")
             model.config.update(config_updates)
             if old_generation_config:
                 model.generation_config = old_generation_config
                 model.generation_config.update(config_updates)
                 print("Updated generation config.")
             else:
                 from transformers import GenerationConfig
                 model.generation_config = GenerationConfig(**config_updates)
                 print("Created new generation config.")
        else:
             print("Warning: Could not determine pad, eos, or bos tokens. Skipping config update.")
        

    except Exception as e:
        print(f"Config update failed: {e}")



    
    print(f"Saving to Hugging Face as {args.new_model_name}...")
    try:
        model.to('cpu')
        model.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
        new_tokenizer.push_to_hub(args.new_model_name, private=False, token=args.hf_token)
        print("Transplantation completed!")
    except Exception as e:
        print(f"Error during Hugging Face Hub upload: {e}")

#  ------------- End ------------------ 

if __name__ == "__main__":
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end
        def __eq__(self, other):
            return self.start <= other <= self.end
    
    parser = argparse.ArgumentParser(description="Tokenizer Transplantation ")
    parser.add_argument(
        "-model", "--model_path", required=True, help="Path to the original model"
    )
    parser.add_argument(
        "-tk", "--new_tokenizer_path", required=True, help="Path to the new tokenizer"
    )
    parser.add_argument(
        "-embed", "--embedding_model_path", default="nomic-ai/nomic-embed-text-v2-moe",
        help="Path to embedding model; defaults to nomic-ai/nomic-embed-text-v2-moe"
    )
    parser.add_argument(
        "-repo", "--new_model_name", required=True, help="HF's Repo name for the new model"
    )
    parser.add_argument(
        "-auth", "--hf_token", required=True, help="Hugging Face authentication token"
    )
    parser.add_argument(
        "-temp", "--temperature", default=0.3, 
        help="Temprature for more expresive weighting 0.3 is default more than this is more bland ; less than this is more expressive", 
        type=float,choices=[Range(0.0 , 5.0)]
    )
    parser.add_argument(
        "-pad","--multiple_of" , default = 128,
        help="When Resizing model ; will resize to a multiple of earlier papers proved padding to power of 2 helps in throughput; default is 128",
        type=int
    )
    parser.add_argument(
        "-d", "--dtype", default="fp32", choices=["bf16", "fp16", "fp32"],
        help="Model and Processing data type, default : fp32"
    )
    parser.add_argument(
        "-bs", "--batch_size", default=16, type=int,
        help="Batch size for embedding extraction, default: 16"
    )
    parser.add_argument(
        "-k","--top_k", default=3, type=int,
        help="Top K for global heuristic, default: 3"
    )
    parser.add_argument(
        "-w", "--weight", default=0.3, type=float,
        help="Weight for global heuristic ; default: 0.3; local heuristic is 1 - global heuristic", 
        choices=[Range(0.0 , 1.0)]
    )
    parser.add_argument(
        "-limit", "--threshold", default=0.6, type=float
        , help="Threshold for cosine similarity, default: 0.6"
    )

    args = parser.parse_args()
    main(args)