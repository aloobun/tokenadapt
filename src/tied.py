
# coding: utf-8
# Copyright IsNoobGrammer and aloobun, 2025
#
# This script is part of the Tokenizer Transplantation Tool.


import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm.auto import tqdm

def transplant_tied_embeddings(model, new_tokenizer: AutoTokenizer, shared_vocab: list, unique_tokens: set, 
                              full_token_embeds: dict, subtoken_embeds: dict, old_vocab: dict, 
                              new_vocab: dict, old_tokenizer: AutoTokenizer, data_type: torch.dtype) -> None:
    """Transplants embeddings for a model with tied input/output embeddings.
    For Tied word embeddings, the input and output embeddings are the same matrix.

    Args:
        model: Model to be updated with new embeddings. (AutoModelForCausalLM)
        new_tokenizer: Tokenizer with the new vocabulary. (AutoTokenizer) 
        shared_vocab: Tokens common to old and new vocabularies. (list)
        unique_tokens: Tokens unique to the new vocabulary. (list)
        full_token_embeds: Cached embeddings for new full tokens. (dict of str -> torch.Tensor)
        subtoken_embeds: Cached embeddings for subtokens. (dict of str -> torch.Tensor)
        old_vocab: Original vocabulary mapping (token -> ID).
        new_vocab: New vocabulary mapping (token -> ID).
        old_tokenizer: Original tokenizer. (AutoTokenizer)
        data_type: Torch data type for embeddings (e.g., torch.bfloat16).
    """
    with torch.no_grad():
        
        embed_dim = model.get_input_embeddings().weight.shape[1]
        new_embeddings = torch.rand(len(new_tokenizer), embed_dim, dtype=data_type, device="cpu")
        new_embeddings.normal_(
            mean=model.get_input_embeddings().weight.mean().item(),
            std=model.get_input_embeddings().weight.std().item()
        )

        for token in tqdm(shared_vocab, desc="Copying shared token embeddings"):
            old_id = old_vocab[token]
            new_id = new_vocab[token]
            new_embeddings[new_id] = model.get_input_embeddings().weight[old_id].clone()

       
        success_count = 0
        for token_str in tqdm(unique_tokens, desc="Initializing unique tokens"):
            new_id = new_vocab[token_str]
            full_token = new_tokenizer.decode([new_id])
            if full_token not in full_token_embeds:
                continue
            full_embed = torch.tensor(full_token_embeds[full_token],dtype=torch.bf16)
            old_ids = old_tokenizer.encode(full_token, add_special_tokens=False)
            sub_embeds = [torch.tensor(subtoken_embeds[old_tokenizer.decode([oid])],dtype=torch.bf16)
                          for oid in old_ids if old_tokenizer.decode([oid]) in subtoken_embeds]
            if not sub_embeds:
                continue
            sub_embeds = torch.stack(sub_embeds)
            similarities = F.cosine_similarity(full_embed.unsqueeze(0), sub_embeds, dim=1)
            weights = F.softmax(similarities, dim=0)
            old_embeds = torch.stack([model.get_input_embeddings().weight[oid] for oid in old_ids])
            new_embeddings[new_id] = (weights.unsqueeze(1) * old_embeds).sum(dim=0)
            success_count += 1

        print(f"Transplanted {success_count}/{len(unique_tokens)} new tokens successfully. "
              f"{len(unique_tokens) - success_count} tokens remain randomly initialized.")

        
        for param in model.parameters():
            param.requires_grad = False
        model.resize_token_embeddings(len(new_tokenizer))
        model.get_input_embeddings().weight.copy_(new_embeddings)
        if model.get_output_embeddings() is not None:
            model.get_output_embeddings().weight.copy_(new_embeddings)

