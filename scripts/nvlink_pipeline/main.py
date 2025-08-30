from typing import *
import time
import sys

from core.entities import Request
from core.simple_pipeline import SimplePipeline
from core.swapping_pipeline import SwappingPipeline

from constants import *


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
    num_stages = 2
    model_name = 'microsoft/Phi-3.5-mini-instruct'

    print('Running experiment with pipeline mode:', MODE)
    if MODE == 'simple':
        pipeline = SimplePipeline(model_name, num_stages, DEV_GPU_,
            max_length=MAX_LENGTH, available_memory=AVAILABLE_MEMORY)
    elif MODE == 'swapping':
        pipeline = SwappingPipeline(DEV_GPU_[2], model_name, num_stages,
                                    DEV_GPU_, MAX_LENGTH, AVAILABLE_MEMORY)
    else:
        raise RuntimeError('Unexpected value for experiment mode')

    print(f'Processing a pile of {PILE_SIZE} requests')
    print('Batch size is:', pipeline.batch_size)

    for req in load_pile_of_request(PILE_SIZE):
        pipeline.add_request(req)
    
    # pipeline.prepare_run_queue()
    pipeline.prepare_run_queue_batched()
    
    start_time = time.time()
    pipeline.process_requests()
    end_time = time.time()
    print(f"Time to process {PILE_SIZE} requests: {end_time - start_time:.2f} seconds")
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
