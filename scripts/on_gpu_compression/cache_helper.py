from typing import *

import numpy as np
import cupy as cp
from nvidia import nvcomp

import torch
from transformers.generation.utils import DynamicCache

# NOTE: Warning: I have put together gridfour, and it is not stable
from gridfour import CompressedData, compress_float16, decompress_float16


DEV_CPU = 'cpu'
DEV_GPU = 'cuda:0'


def cache_gridfour_decompress(list_comp: List[CompressedData], num_layers: int) -> DynamicCache:
    # NOTE: gridfour code currently runs on CPU
    cache = DynamicCache()
    cache.append_new_layers(layer_idx=num_layers - 1)
    for i, comp in enumerate(list_comp):
        assert comp.dtype == 'float16'
        ndarr = decompress_float16(comp)
        tensor = torch.from_numpy(ndarr)
        tensor = tensor.to(DEV_GPU)

        # Figure out which layer it is
        index = i // 2
        # Figure out if it's as key or value
        if i % 2 == 0:
            cache.layers[index].keys = tensor
        else:
            cache.layers[index].values = tensor
    return cache


def cache_gridfour_compress(cache: DynamicCache) -> List[CompressedData]:
    # NOTE: gridfour code currently runs on CPU
    list_comp = []
    tensors = cache_get_all_tensors(cache)
    for tensor in tensors:
        assert tensor.dtype == torch.float16
        # NOTE: compress_float16 expects a linear array
        tensor = tensor.flatten()
        ndarr = tensor.to(DEV_CPU).numpy()
        comp = compress_float16(ndarr)
        list_comp.append(comp)
    return list_comp


def cache_get_all_tensors(cache: DynamicCache) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for layer in cache.layers:
        tensors.append(layer.keys)
        tensors.append(layer.values)
    return tensors



def cache_save_to_disk(cache: DynamicCache, file_name: str) -> None:
    cache_dict = {
        "key_cache": [layer.keys.cpu() for layer in cache.layers],
        "value_cache": [layer.values.cpu() for layer in cache.layers],
    }
    torch.save(cache_dict, file_name)


def cache_load_from_disk(file_name: str) -> DynamicCache:
    loaded: Dict[str, List[torch.Tensor]] = torch.load(file_name, map_location=DEV_CPU)
    cache = DynamicCache()
    num_layers = len(loaded['key_cache'])
    cache.append_new_layers(layer_idx=num_layers-1)
    for i in range(num_layers):
        cache.layers[i].keys = loaded['key_cache'][i]
        cache.layers[i].values = loaded['value_cache'][i]
    return cache


def cache_move(cache: DynamicCache, dev: str) -> None:
    """
    move tensors of a cache from GPU to CPU
    """
    num_layers = len(cache.layers)
    for i in range(num_layers):
        cache.layers[i].keys = cache.layers[i].keys.to(dev)
        cache.layers[i].values = cache.layers[i].values.to(dev)


def cache_decompress(list_comp_arr: List[nvcomp.Array|CompressedData],
        num_layers: int, comp_algo: str) -> DynamicCache:
    """
    NOTE: Assumption is that key/value tensors are fp16
    """
    if comp_algo == 'Gridfour':
        return cache_gridfour_decompress(list_comp_arr, num_layers)

    codec = nvcomp.Codec(algorithm=comp_algo)
    cache = DynamicCache()

    cache.append_new_layers(layer_idx=num_layers-1)

    # mempool = cp.get_default_memory_pool()

    for i, arr in enumerate(list_comp_arr):
        # arr = arr.cuda()
        decomp = codec.decode(arr)# .cuda()
        cp_arr = cp.from_dlpack(decomp.to_dlpack()).view(cp.float16)
        tensor = torch.tensor(cp_arr) # make a copy
        del cp_arr
        del decomp
        # mempool.free_all_blocks()

        # Figure out which layer it is
        index = i // 2
        # Figure out if it's as key or value
        if i % 2 == 0:
            cache.layers[index].keys = tensor
        else:
            cache.layers[index].values = tensor


    return cache


def cache_compress(cache: DynamicCache, comp_algo: str) -> List[nvcomp.Array]:
    """
    Compress tensors of a cache
    
    This will overwrite the cache tensors, so the cache object will be invalid
    """
    if comp_algo == 'Gridfour':
        return cache_gridfour_compress(cache)

    tensors = cache_get_all_tensors(cache)
    codec = nvcomp.Codec(algorithm=comp_algo)
    list_comp_arr = []
    for tensor in tensors:
        assert tensor.dtype == torch.float16, 'compression/decompression assumes and expects tensors of type fp16'
        arr = cp.from_dlpack(tensor).view(cp.uint8)
        comp = codec.encode(nvcomp.from_dlpack(arr))
        list_comp_arr.append(comp)

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    return list_comp_arr


def cache_total_size(cache: DynamicCache) -> int:
    """
    memory usage of a KV Cache tensors
    """
    tensors: List[torch.Tensor] = []
    for layer in cache.layers:
        tensors.append(layer.keys)
        tensors.append(layer.values)

    total_bytes = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.element_size() * tensor.numel()
        else:
            raise RuntimeError('something is not tensor')
    return total_bytes

