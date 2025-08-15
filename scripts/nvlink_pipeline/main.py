from typing import *
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch

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
    hidden_state = req.next_token_ids

    with torch.no_grad():
        for stage in replica.stages:
            cache = req.stage_cache[0]

            # bring the input/hidden state to device
            hidden_state = hidden_state.to(stage.device, non_blocking=True)

            # TODO: can we do something here to use the time?

            # wait for data transfer to finish
            torch.cuda.synchronize()

            out = stage.forward(hidden_state, use_cache=True,
                    past_key_values=cache)

            hidden_state = out.last_hidden_state

    logits = replica.lm_head(out.last_hidden_state)
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

        torch.cuda.empty_cache()


def main():
    req = Request(prompt='write 2 page essay on importance of sunshine')

    model_name = 'microsoft/Phi-3.5-mini-instruct'
    replica = Replica(model_name, num_stages=2, device_list=DEV_GPU_)

    for stage in replica.stages:
        req.stage_cache[stage.stage_index] = DynamicCache(device=stage.device)
    do_prefill(req, replica)

    do_decode(req, replica, max_iter=2048)

    final_text = replica.tokenizer.decode(req.generated[0],
            skip_special_tokens=True)
    print(final_text)


def test():
    """
    just to test what a normal execution of model produces. Usuful to see how
    much memory is consumed.
    """
    model_name = 'microsoft/Phi-3.5-mini-instruct'
    prompt = 'write 2 page essay on importance of sunshine'
    model = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16, device_map='cuda:0',
            local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model.to('cuda:0')

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda:0')

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values
        next_token_logits = output.logits[:, -1, :]

    generated = input_ids
    next_token_logits = next_token_logits

    cache = DynamicCache.from_legacy_cache(past_key_values)
    past_key_values = cache

    for _ in range(256):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=-1)
        with torch.no_grad():
            output = model(input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True)
            past_key_values = output.past_key_values
            next_token_logits = output.logits[:, -1, :]

    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(final_text)


if __name__ == '__main__':
    # test()
    main()

