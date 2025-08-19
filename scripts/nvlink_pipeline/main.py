from typing import *
import time

from core.entities import Request
from core.simple_pipeline import SimplePipeline

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

    mode = 'simple'

    if mode == 'simple':
        pipeline = SimplePipeline(model_name, num_stages, DEV_GPU_,
            max_length=MAX_LENGTH, available_memory=AVAILABLE_MEMORY)
    elif mode == 'swapping':
        # pipeline = SwappingPipeline(model_name, num_stages, DEV_GPU_, batch_size)
        pass
    else:
        raise RuntimeError('Unexpected value for experiment mode')

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
    # test()
    main()
