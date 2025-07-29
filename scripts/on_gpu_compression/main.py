from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch
import time

import numpy as np
import cupy as cp
from nvidia import nvcomp


DEV_GPU='cuda:0'
DEV_CPU='cpu'

model_name: str = 'openai-community/gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=DEV_GPU, local_files_only=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model.to('cuda:0')
# help(model)


def cache_move(cache: DynamicCache, dev: str) -> None:
    """
    move tensors of a cache from GPU to CPU
    """
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i].to(dev)
        cache.value_cache[i] = cache.value_cache[i].to(dev)


def cache_compress(cache: DynamicCache, comp_algo: str) -> List[nvcomp.Array]:
    """
    Compress tensors of a cache
    """
    # TODO: figure if I need to set datatype: , dtype="<f2"   # "<f2" for float16
    tensors = cache.key_cache + cache.value_cache
    codec = nvcomp.Codec(algorithm=comp_algo)
    compressed_tensors = []
    for tensor in tensors:
        arr = nvcomp.as_array(tensor)
        compressed = codec.encode(arr)
        compressed_tensors.append(compressed)
    return compressed_tensors


def cache_total_size(cache: DynamicCache) -> int:
    """
    memory usage of a KV Cache tensors
    """
    total_bytes = 0
    for tensor in cache.key_cache + cache.value_cache:
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.element_size() * tensor.numel()
        else:
            print('something is not tensor')
    return total_bytes


class Req:
    # Model
    M = model
    # Tokenizer
    T = tokenizer

    def __init__(self, prompt:str):
        self.prompt = prompt
        self.inputs = self.T(self.prompt, return_tensors='pt')
        self.inputs.to(DEV_GPU)

        # Prefill: run prompt through model, get initial logits and hidden
        # states (past_key_values)
        with torch.no_grad():
            output = self.M(input_ids=self.inputs['input_ids'], use_cache=True)
            self.past_key_values = output.past_key_values
            self.next_token_logits = output.logits[:, -1, :]

        self.move_to(DEV_CPU)

    def move_to(self, dev):
        self.inputs = self.inputs.to(dev)
        self.next_token_logits = self.next_token_logits.to(dev)
        # print('layers:', len(self.past_key_values))
        for kv_layer in self.past_key_values:
            for t in kv_layer:
                t.to(dev)

        # for i in range(len(self.cache.key_cache)):
        #     self.cache.key_cache[i] = self.cache.key_cache[i].to(dev)
        #     self.cache.value_cache[i] = self.cache.value_cache[i].to(dev)

    def auto_regression(self, decode=False, max_lenght=32):
        # Decode: generate step by step using cached states

        generated = self.inputs['input_ids'][:].to(DEV_GPU)

        next_token_logits = self.next_token_logits.to(DEV_GPU)

        cache = DynamicCache.from_legacy_cache(self.past_key_values)
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].to(DEV_GPU)
            cache.value_cache[i] = cache.value_cache[i].to(DEV_GPU)
        past_key_values = cache

        for _ in range(max_lenght):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # print(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=-1)
            with torch.no_grad():
                output = model(input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True)
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
            # TODO: check if end of sentence then stop

        # Decode output tokens
        if decode:
            final_text = tokenizer.decode(generated[0],
                    skip_special_tokens=True)
            print(final_text)

        return past_key_values

    def measure_kv_cache_moving_time(self) -> Dict[str, Any]:
        """
        Do decode procedure to create a full kv cache. Then move it
        between CPU and GPU
        report size and transmission time
        """
        cache = self.auto_regression(max_lenght=128)
        repeat = 40
        cache_size = cache_total_size(cache)
        # we repeat the measurement. each entry is one instance. the entry is a
        # tuple of the form (to-cpu-time, to-gpu-time).
        time_measurements = []

        for _ in range(repeat):
            # to CPU
            start = time.perf_counter()
            cache_move(cache, DEV_CPU)
            end = time.perf_counter()
            to_cpu_time = end - start

            # to GPU
            start = time.perf_counter()
            cache_move(cache, DEV_GPU)
            end = time.perf_counter()
            to_gpu_time = end - start

            time_measurements.append((to_cpu_time, to_gpu_time))

        return { 'bytes': cache_size, 'measurements': time_measurements, }

    def measure_compressed_kv_cache_moving_time(self) -> Dict[str, Any]:
        cache = self.auto_regression(max_lenght=128)
        repeat = 40
        algos = ["ANS", "LZ4", "Cascaded", "GDeflate"]

        cache_size = cache_total_size(cache)
        result: Dict[str, Any] = {
                'bytes': cache_size,
                }
        for algo in algos:
            comp_tensors = cache_compress(cache, algo)
            sz = 0
            for arr in comp_tensors:
                sz += arr.size * arr.item_size

            measurements = []
            for _ in range(repeat):
                start = time.perf_counter()
                for arr in comp_tensors:
                    arr.cpu()
                end = time.perf_counter()
                to_cpu_time = end - start

                start = time.perf_counter()
                for arr in comp_tensors:
                    arr.cuda()
                end = time.perf_counter()
                to_gpu_time = end - start
                measurements.append((to_cpu_time, to_gpu_time))

            result[algo] = {
                    'bytes': sz,
                    'measurements': measurements,
                    }

        return result


def main() -> None:
    with open('./prompts.txt', 'r') as f:
        all_prompts = f.readlines()
    # print(all_prompts)

    reqs = [Req(p) for p in all_prompts]

    for r in reqs:
        print(r.prompt)
        # r.auto_regression(decode=True)
        res = r.measure_kv_cache_moving_time()

        m: List[Tuple[int,int]] = res['measurements']
        print(res['bytes'])
        print(m[5][0] * 1000, '    ', m[5][1] * 1000)

        res = r.measure_compressed_kv_cache_moving_time()
        for a in ["ANS", "LZ4", "Cascaded", "GDeflate"]:
            print(a, ':', res[a]['bytes'], 'B')
            print(res[a]['measurements'][5][0] * 1000,
                    res[a]['measurements'][5][1] * 1000)


    # for i in range(1000):
    #     print(i, '---'*10)
    #     for r in reqs:
    #         r.auto_regression()


if __name__ == '__main__':
    main()

