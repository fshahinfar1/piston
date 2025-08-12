from typing import *
import os
import time

import torch
from transformers.generation.utils import DynamicCache

from entities import Request, Replica


# DEV_GPU_ = [torch.device(f'cuda:{i}') for i in range(3)]
DEV_GPU_ = [torch.device('cuda:0') for i in range(3)]
DEV_CPU = torch.device('cpu')


def cache_save_to_disk(cache: DynamicCache, file_name: str) -> None:
    cache_dict = {
        "key_cache": [layer.keys.cpu() for layer in cache.layers],
        "value_cache": [layer.values.cpu() for layer in cache.layers],
    }
    torch.save(cache_dict, file_name)


def cache_load_from_disk(file_name: str) -> DynamicCache:
    loaded: Dict[str, List[torch.Tensor]] = torch.load(file_name,
                                                    map_location=DEV_CPU)
    cache = DynamicCache()
    num_layers = len(loaded['key_cache'])
    cache.append_new_layers(layer_idx=num_layers-1)
    for i in range(num_layers):
        cache.layers[i].keys = loaded['key_cache'][i]
        cache.layers[i].values = loaded['value_cache'][i]
    return cache


def do_one_iteration(req, replica):
    print('*' * 6)
    hidden_state = req.next_token_ids

    for stage in replica.stages:
        cache = req.stage_cache.get(stage.stage_index)

        # bring the input/hidden state to device
        hidden_state = hidden_state.to(stage.device, non_blocking=True)

        # TODO: can we do something here to use the time?

        # wait for data transfer to finish
        torch.cuda.synchronize()

        hidden_state, next_cache = stage.forward(hidden_state, use_cache=True,
                past_key_values=cache)

        req.stage_cache[stage.stage_index] = next_cache

    logits = replica.lm_head(hidden_state)
    logits = logits.float()
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    return next_token


def do_prefill(req, replica):
    count_stages = len(replica.stages)

    # tokenize
    inputs = replica.tokenizer(req.prompt, return_tensors='pt')
    req.next_token_ids = inputs['input_ids']
    req.generated = torch.cat([req.generated, req.next_token_ids], dim=-1)

    next_token = do_one_iteration(req, replica)
    next_token = next_token.cpu()

    req.generated = torch.cat([req.generated, next_token], dim=-1)
    req.next_token_ids = next_token


def do_decode(req, replica, max_iter=32):
    for _ in range(max_iter):
        next_token = do_one_iteration(req, replica)
        # TODO: maybe avoid moving to CPU? or copy to CPU and other GPU async
        next_token = next_token.cpu()
        req.generated = torch.cat([req.generated, next_token], dim=-1)
        req.next_token_ids = next_token

        if next_token.item() == replica.tokenizer.eos_token_id:
            break


def main():
    req = Request(prompt='write 2 page essay on importance of sunshine')

    model_name = 'microsoft/Phi-3.5-mini-instruct'
    replica = Replica(model_name, num_stages=1, device_list=DEV_GPU_)

    do_prefill(req, replica)

    do_decode(req, replica, max_iter=32)


    final_text = replica.tokenizer.decode(req.generated[0],
            skip_special_tokens=True)
    print(final_text)


if __name__ == '__main__':
    main()

