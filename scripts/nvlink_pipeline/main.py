from typing import *
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch

from entities import Request, Replica, ExecutionStatistics

MAX_LENGTH = 2048
NUM_DEVICES = 1
NUM_STAGES = 3
DEV_GPU_ = [torch.device(f'cuda:{i % NUM_DEVICES}') for i in range(NUM_STAGES)]
DEV_CPU = torch.device('cpu')


def do_prefill(req, replica):
    # tokenize
    inputs = replica.tokenizer(req.prompt, return_tensors='pt')
    req.next_token_ids = inputs['input_ids']
    # req.generated = torch.cat([req.generated, req.next_token_ids], dim=-1)
    req.generated.append(req.next_token_ids)

    next_token = replica.do_one_iteration(req)
    next_token = next_token.cpu()

    # req.generated = torch.cat([req.generated, next_token], dim=-1)
    req.next_token_ids = next_token
    req.generated.append(req.next_token_ids)


def do_decode(req, replica, stat, max_iter=32):
    for _ in range(max_iter):
        next_token = replica.do_one_iteration(req, stat)

        # TODO: maybe avoid moving to CPU? or copy to CPU and other GPU async
        # next_token = next_token.cpu()
        # req.generated = torch.cat([req.generated, next_token], dim=-1)
        req.generated.append(next_token)

        req.next_token_ids = next_token

        if next_token.item() == replica.tokenizer.eos_token_id:
            break
        
        # torch.cuda.empty_cache()


def stats(lst: List[float]):
    if not lst:
        return "No data"
    S = sorted(lst)
    mean = sum(S) / len(S)
    std = (sum((x - mean) ** 2 for x in S) / len(S)) ** 0.5
    mid = S[len(S) // 2] if len(S) % 2 == 1 else (S[len(S) // 2 - 1] + S[len(S) // 2]) / 2
    return mean, std, mid, S[-1], len(S)


def report_statistics(stat: ExecutionStatistics):
    num_stages = len(stat.stage_exec_times)
    print("Execution Statistics:")
    for stage_index, exec_times in stat.stage_exec_times.items():
        # mid and std and avg of list
        mean, std, mid, _max, count = stats(exec_times)
        print(f"Stage {stage_index} execution times:")
        print(f"\t\tmean={mean:.4f}, std={std:.4f}, mid={mid:.4f}, max={_max:.4f}, count={count}")
    for stage_index, transfer_times in stat.hidden_state_transfer_times.items():
        mean, std, mid, _max, count = stats(transfer_times)
        print(f"hidden state transfer from stage {(stage_index - 1) % num_stages} to stage {stage_index}:")
        print(f"\t\tmean={mean:.4f}, std={std:.4f}, mid={mid:.4f}, max={_max:.4f}, count={count}")


def load_pile_of_request(pile_size):
    prompts = []
    with open('prompts.txt', 'r') as f:
        prompts = list(f.readlines())
    
    num_prompts = len(prompts)
    repeat = pile_size // num_prompts
    rest = pile_size - (repeat * num_prompts)

    requests = []
    for prompt in (prompts * repeat) + prompts[:rest]:
        req = Request(prompt=prompt)
        requests.append(req)
    
    return requests


def get_batch_size(replica):
    total_gpu_mem = 0
    for gpu_index in range(NUM_DEVICES):
        t = torch.cuda.get_device_properties(gpu_index).total_memory * 0.9
        r = torch.cuda.memory_reserved(gpu_index)
        a = torch.cuda.memory_allocated(gpu_index)
        print('GPU', gpu_index, 'total:', t, 'allocated', a)
        total_gpu_mem += t - a
    
    max_kv_size = replica.get_max_kv_cache_size(MAX_LENGTH)
    num_req = int(total_gpu_mem // max_kv_size)
    return num_req


def print_output(req):
    # Actually generate the text
    # generated = torch.cat([t.to(DEV_CPU) for t in req.generated], dim=-1)
    # final_text = replica.tokenizer.decode(generated[0], skip_special_tokens=True)
    # print(final_text)
    return


# def batch_inputs(inputs_list, pad_token_id):
#     # Assume inputs_list = list of tokenized input tensors, e.g. [tensor([1,2,3]), tensor([4,5])]
#     pad_token_id = replica.tokenizer.pad_token_id
#     max_len = max(x.shape[0] for x in inputs_list)

#     # Pad each tensor and create attention mask
#     input_ids = []
#     attention_masks = []
#     for ids in inputs_list:
#         pad_len = max_len - ids.shape
#         padded = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
#         mask = torch.cat([torch.ones(ids.shape), torch.zeros(pad_len)])
#         input_ids.append(padded)
#         attention_masks.append(mask)

#     batch_input_ids = torch.stack(input_ids)       # shape: [batch_size, max_seq_len]
#     batch_attention_mask = torch.stack(attention_masks)  # shape: [batch_size, max_seq_len]


def main():
    pile_size = 54
    requests = load_pile_of_request(pile_size)

    num_stages = 2
    model_name = 'microsoft/Phi-3.5-mini-instruct'
    replica = Replica(model_name, num_stages=num_stages, device_list=DEV_GPU_)

    batch_size = get_batch_size(replica)
    print('Batch size is:', batch_size)

    run_queue = []
    # Do prefill of all request in advance
    for i, req in enumerate(requests):
        do_prefill(req, replica)
        count = len(req.cache.layers)
        dev_map = [DEV_CPU] * count
        req.move_to(dev_map, non_blocking=True)
        run_queue.append(req)

    stat = ExecutionStatistics(len(replica.stages))

    while run_queue:
        # dequeue a batch of requests
        batch_requests = run_queue[:batch_size]
        del run_queue[:batch_size]

        # move batch of requests to GPUs
        for req in batch_requests:
            count = len(req.cache.layers)
            r = count // num_stages
            dev_map = [DEV_GPU_[i // r] for i in range(count)]
            req.move_to(dev_map)

        start_time = time.time()
        for req in batch_requests:
            do_decode(req, replica, stat, max_iter=MAX_LENGTH)
            print_output(req)
        end_time = time.time()
        print(f"Decoding execution time: {end_time - start_time:.2f} seconds")
        #  report_statistics(stat)

        # free memory of requests
        for req in batch_requests:
            req.cache = DynamicCache()
            req.generated = []
            req.next_token = None
        torch.cuda.empty_cache()

    

if __name__ == '__main__':
    # test()
    main()
