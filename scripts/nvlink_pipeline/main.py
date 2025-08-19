from typing import *
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch

from entities import Request, Replica, ExecutionStatistics
from prefill_decode import do_prefill
from simple_pipeline import SimplePipeline

from constants import *


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


def main():
    pile_size = 8
    num_stages = 2
    model_name = 'microsoft/Phi-3.5-mini-instruct'

    # TODO: for experimenting reasons I have limited the KV-Cache size to 3 GB
    available_memory = 3*GB

    mode = 'simple'

    if mode == 'simple':
        pipeline = SimplePipeline(model_name, num_stages, DEV_GPU_,
            max_length=MAX_LENGTH, available_memory=available_memory)
    elif mode == 'swapping':
        # pipeline = SwappingPipeline(model_name, num_stages, DEV_GPU_, batch_size)
        pass
    else:
        raise RuntimeError('Unexpected value for experiment mode')

    print('Batch size is:', pipeline.batch_size)

    for req in load_pile_of_request(pile_size):
        pipeline.add_request(req)
    
    # pipeline.prepare_run_queue()
    pipeline.prepare_run_queue_batched()
    
    start_time = time.time()
    pipeline.process_requests()
    end_time = time.time()
    print(f"Time to process {pile_size} requests: {end_time - start_time:.2f} seconds")
    

if __name__ == '__main__':
    # test()
    main()
