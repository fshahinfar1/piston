from typing import *

import numpy as np
import cupy as cp
from nvidia import nvcomp

import torch
from transformers.generation.utils import DynamicCache


DEV_CPU='cpu'


def cache_save_to_disk(cache: DynamicCache, file_name: str) -> None:
    cache_dict = {
        "key_cache": [t.cpu() for t in cache.key_cache],
        "value_cache": [t.cpu() for t in cache.value_cache],
    }
    torch.save(cache_dict, file_name)


def cache_load_from_disk(file_name: str) -> DynamicCache:
    loaded: Dict[str, List[torch.Tensor]] = torch.load(file_name, map_location=DEV_CPU)
    cache = DynamicCache()
    cache.key_cache = [t for t in loaded["key_cache"]]
    cache.value_cache = [t for t in loaded["value_cache"]]
    return cache


def cache_move(cache: DynamicCache, dev: str) -> None:
    """
    move tensors of a cache from GPU to CPU
    """
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i].to(dev)
        cache.value_cache[i] = cache.value_cache[i].to(dev)


def cache_decompress(list_comp_arr: List[nvcomp.Array], num_tokens, comp_algo) -> DynamicCache:
    """
    """
    codec = nvcomp.Codec(algorithm=comp_algo)
    cache = DynamicCache()

    for i, arr in enumerate(list_comp_arr):
        arr2 = codec.decode(arr)
        cp_arr = cp.asarray(arr2)
        tensor = torch.as_tensor(cp_arr)
        if i < num_tokens:
            cache.key_cache.append(tensor)
        else:
            cache.value_cache.append(tensor)
    return cache


def cache_compress(cache: DynamicCache, comp_algo: str) -> List[nvcomp.Array]:
    """
    Compress tensors of a cache
    """
    # TODO: figure if I need to set datatype: , dtype="<f2"   # "<f2" for float16
    tensors = cache.key_cache + cache.value_cache
    codec = nvcomp.Codec(algorithm=comp_algo)
    compressed_tensors = []
    for tensor in tensors:
        arr = nvcomp.from_dlpack(tensor)
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

