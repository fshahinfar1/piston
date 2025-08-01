from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch
import time

import os

from cache_helper import *
from tests import *
from req import Req, move_model_to

import cupy as cp

COMPRESSION_ALGORITHMS = ["ANS", "LZ4", "Cascaded", "GDeflate"]

def main() -> None:
    with open('./prompts.txt', 'r') as f:
        all_prompts = f.readlines()
    reqs = [Req(p) for p in all_prompts]

    # If we are not going to run the decode phase move the model to CPU to
    # free some memory
    move_model_to('cpu')

    # assert_compression_decompression_works(reqs[0])
    # return

    cp_pool = cp.get_default_memory_pool()

    for r in reqs:
        print('\n', '-----', r.prompt)
        torch.cuda.empty_cache()
        cp_pool.free_all_blocks()

        # Print the Models response to prompts
        # r.auto_regression(max_lenght=max_context_len, decode=True)

        sample_index = 2
        res = measure_kv_cache_moving_time(r)
        m: List[Tuple[int,int]] = res['measurements']
        print('KV Cache size:', res['bytes'] / 1000000, 'MB')
        print(m[sample_index][0] * 1000, '    ', m[sample_index][1] * 1000)
        print()

        res = measure_compressed_kv_cache_moving_time(r)
        print('***', res['bytes'] / 1000000, 'MB')
        print('original cache: ', res['pointers'])
        for a in COMPRESSION_ALGORITHMS:
            print(a, ':', res[a]['bytes'] / 1000000, 'MB', '    ',
                    'Compression Time:', res[a]['compression_time'] * 1000, 'ms')
            print(res[a]['measurements'][sample_index][0] * 1000, '    ',
                    res[a]['measurements'][sample_index][1] * 1000)
            print('  pointers:', res[a]['pointers'])
            print()



if __name__ == '__main__':
    main()

