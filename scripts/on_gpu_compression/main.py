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

# "Gridfour",
COMPRESSION_ALGORITHMS = ["ANS", "LZ4", "Cascaded", "GDeflate"]

def main2():
    mil = 1000000
    tho = 1000

    with open('./prompts.txt', 'r') as f:
        all_prompts = f.readlines()
    r = Req(all_prompts[2])
    move_model_to('cpu')
    t = move_compressed_cache_defraged(r)

    print(t['pointers'])
    print('-' * 20)
    print(t['ANS']['pointers'])


    print('size:', t['bytes'] / mil, 'comp:', t['ANS']['bytes'] /mil)

    print('ANS')
    for m in t['ANS']['measurements']:
        print(m[0] * tho, '     ', m[1] * tho)


def main() -> None:
    with open('./prompts.txt', 'r') as f:
        all_prompts = f.readlines()
    # all_prompts = all_prompts[2:3]
    reqs = [Req(p) for p in all_prompts]

    # If we are not going to run the decode phase move the model to CPU to
    # free some memory
    move_model_to('cpu')

    # assert_compression_decompression_works(reqs[0])
    # return

    # reqs[2].auto_regression(max_lenght=2**13, decode=True)
    # return

    # for r in reqs:
    #     report_compression_percent(r, 'Gridfour')

    for r in reqs:
        print('\n', '-----', r.prompt)
        torch.cuda.empty_cache()

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
        print()

        for a in COMPRESSION_ALGORITHMS:
            s = res[a]
            print(a, ':', s['bytes'] / 1000000, 'MB', '    ',
                    'Compression Time:', s['compression_time'] * 1000, 'ms')
            # for m in s['measurements'][-10:]:
            #     print(m[0] * 1000, '    ', m[1] * 1000)
            print(s['measurements'][sample_index][0] * 1000, '    ', res[a]['measurements'][sample_index][1] * 1000)
            print('  pointers:', s['pointers'])
            print()


if __name__ == '__main__':
    main()
    # main2()

