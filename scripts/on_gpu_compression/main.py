from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch
import time

import os

from cache_helper import *
from tests import *
from req import Req

COMPRESSION_ALGORITHMS = ["ANS", "LZ4", "Cascaded", "GDeflate"]

def main() -> None:
    with open('./prompts.txt', 'r') as f:
        all_prompts = f.readlines()
    reqs = [Req(p) for p in all_prompts]

    assert_compression_decompression_works(reqs[-2])
    return

    for r in reqs:
        print('\n', '-----', r.prompt)
        torch.cuda.empty_cache()

        # Print the Models response to prompts
        # r.auto_regression(max_lenght=max_context_len, decode=True)

        sample_index = 1
        res = measure_kv_cache_moving_time(r)
        m: List[Tuple[int,int]] = res['measurements']
        print('KV Cache size:', res['bytes'] / 1000000, 'MB')
        print(m[sample_index][0] * 1000, '    ', m[sample_index][1] * 1000)
        print()

        res = measure_compressed_kv_cache_moving_time(r)
        print('vvv', res['bytes'] / 1000000, 'MB')
        for a in COMPRESSION_ALGORITHMS:
            print(a, ':', res[a]['bytes'] / 1000000, 'MB', '    ',
                    'Compression Time:', res[a]['compression_time'] * 1000, 'ms')
            print(res[a]['measurements'][sample_index][0] * 1000, '    ',
                    res[a]['measurements'][sample_index][1] * 1000)
            print()


if __name__ == '__main__':
    main()

