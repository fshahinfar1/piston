from typing import *
import torch
from core.entities import Replica


def _get_available_memory(available_memory: Optional[int]) -> int:
    if available_memory is None:
        total_gpu_mem = 0
        for gpu_index in range(NUM_DEVICES):
            t = torch.cuda.get_device_properties(gpu_index).total_memory * 0.9
            r = torch.cuda.memory_reserved(gpu_index)
            a = torch.cuda.memory_allocated(gpu_index)
            print('GPU', gpu_index, 'total:', t, 'allocated', a)
            total_gpu_mem += t - a
    else:
        total_gpu_mem = available_memory
    return total_gpu_mem

def get_batch_size(replica, max_length, available_memory: Optional[int]=None) -> int:
    total_gpu_mem = _get_available_memory(available_memory) 
    max_kv_size = replica.get_max_kv_cache_size(max_length)
    num_req = int(total_gpu_mem // max_kv_size)
    # print(total_gpu_mem, max_kv_size, num_req)
    return num_req

def get_max_num_tokens(replica: Replica, available_memory: Optional[int]=None) -> int:
    total_gpu_mem = _get_available_memory(available_memory)
    token_size = replica.get_kv_cache_token_size()
    max_num_tokens  = int(total_gpu_mem // token_size)
    return max_num_tokens
