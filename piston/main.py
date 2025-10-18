from typing import *
import os
import time
import sys

import argparse

from piston.core.entity import Request
from piston.core.pipeline import SimplePipeline, SwappingPipeline, OverCommitedSingleStagePipeline
from piston.constants import *


def load_pile_of_request(pile_size):
    prompts = []
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'prompts.txt'), 'r') as f:
        prompts = [l for l in f.readlines() if not l.startswith('#')]
    
    num_prompts = len(prompts)
    repeat = pile_size // num_prompts
    rest = pile_size - (repeat * num_prompts)

    requests = []
    for prompt in (prompts * repeat) + prompts[:rest]:
        req = Request(prompt=prompt)
        requests.append(req)
    
    return requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='number of request in a batch')
    parser.add_argument('--num-requests', type=int, default=32, help='total number of requests to process')
    parser.add_argument('--pipeline', type=str, default='simple', help='which type of pipeline use for requests processing (simple, swapping)')
    parser.add_argument('--iters', type=int, default=1024, help='number of tokens to generate')
    parser.add_argument('--num-stages', type=int, default=2, help='number pipeline stages')
    parser.add_argument('--verbose', help='print the inference resposes', action='store_true')

    args = parser.parse_args()
    return args


def main():
    torch.multiprocessing.set_start_method('spawn')

    model_name = 'microsoft/Phi-3.5-mini-instruct'
    # model_name = '/leonardo_work/EUHPC_D17_077/fshahinf/dequantized/gpt-oss-20b-bf16'

    args = parse_args()
    MODE = args.pipeline
    BATCH_SIZE = args.batch
    MAX_LENGTH = args.iters
    PILE_SIZE = args.num_requests
    VERBOSE=args.verbose
    num_stages = args.num_stages

    assert num_stages == 1 or num_stages == 2

    print('Running experiment with pipeline mode:', MODE, 'Number of stages:', num_stages)
    if MODE == 'simple':
        pipeline = SimplePipeline(model_name, num_stages, DEV_GPU_,
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE, do_print=VERBOSE)
    elif MODE == 'swapping':
        if num_stages == 1:
            pipeline = OverCommitedSingleStagePipeline(SPARE_MEMORY, model_name,
                            num_stages, DEV_GPU_, MAX_LENGTH, BATCH_SIZE, do_print=VERBOSE)
        else:
            devices = DEV_GPU_[:num_stages]
            spare_memory = DEV_GPU_[num_stages:]
            pipeline = SwappingPipeline(spare_memory, model_name, num_stages,
                                        devices, max_length=MAX_LENGTH,
                                        batch_size=BATCH_SIZE, do_print=VERBOSE)
    else:
        raise RuntimeError('Unexpected value for experiment mode')

    print(f'Processing a pile of {PILE_SIZE} requests')
    print('Batch size is:', pipeline.batch_size)

    for req in load_pile_of_request(PILE_SIZE):
        pipeline.add_request(req)
    
    try:
        # pipeline.prepare_run_queue()
        pipeline.prepare_run_queue_batched()
    except Exception as e:
        pipeline.close()
        raise(e)

    print('Prefill finished ---------')

    start_time = time.time()

    try:
        pipeline.process_requests()
    finally:
        pipeline.close()

    end_time = time.time()
    print(f"Time to process {PILE_SIZE} requests: {end_time - start_time:.2f} seconds")
 

if __name__ == '__main__':
    main()
