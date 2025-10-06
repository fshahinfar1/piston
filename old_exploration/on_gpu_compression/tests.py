from typing import *
import time
import os

import torch
import cupy as cp
from nvidia import nvcomp

from cache_helper import *
from req import Req


MAX_CONTENT_LEN: int = 2 ** 13
DEV_GPU ='cuda:0'
DEV_CPU ='cpu'
# "Gridfour", 
COMPRESSION_ALGORITHMS = ["ANS", "LZ4", "Cascaded", "GDeflate"]
CACHE_DIR='/mnt/farbod'
REPEAT = 5


def measure_kv_cache_moving_time(req: Req) -> Dict[str, Any]:
    """
    Do decode procedure to create a full kv cache. Then move it
    between CPU and GPU
    report size and transmission time
    """
    cache = get_cache(req)
    cache_size = cache_total_size(cache)
    # we repeat the measurement. each entry is one instance. the entry is a
    # tuple of the form (to-cpu-time, to-gpu-time).
    time_measurements = _do_cpu_gpu_movement_exp(cache)

    return { 'bytes': cache_size, 'measurements': time_measurements, }


def measure_compressed_kv_cache_moving_time(req: Req) -> Dict[str, Any]:
    cache = get_cache(req)

    cache_size = cache_total_size(cache)
    result: Dict[str, Any] = { 'bytes': cache_size, }

    ptrs = []
    for t in cache_get_all_tensors(cache):
        ptrs.append(t.data_ptr())
    result['pointers'] = ptrs


    for algo in COMPRESSION_ALGORITHMS:
        # TODO: if we decompress the cache, do we get the same tensors (check correctness)
        start = time.perf_counter()
        comp_tensors = cache_compress(cache, algo)
        end = time.perf_counter()
        comp_time = end - start

        sz = 0
        for arr in comp_tensors:
            if isinstance(arr, CompressedData):
                sz += len(arr.data)
            else:
                sz += arr.size * arr.item_size

        if algo == 'Gridfour':
            # GPU is not implemented yet for girdfour, skip it
            result[algo] = {
                    'bytes': sz,
                    'measurements': [(0, 0)] * REPEAT,
                    'compression_time': 0,
                    'pointers': [],
                    }
            continue

        using_nv_arr = False

        if using_nv_arr:
            measurements = _do_cpu_gpu_movement_exp(comp_tensors)
            ptrs = [arr.__cuda_array_interface__['data'][0] for arr in comp_tensors]
        else:
            # convert nvcomp to tensors
            # copy everything into a torch tensor in a different memory location
            comp_tensors = [torch.tensor(arr) for arr in comp_tensors]

            measurements = _do_cpu_gpu_movement_exp(comp_tensors)

            ptrs = [t.data_ptr() for t in comp_tensors]

        result[algo] = {
                'bytes': sz,
                'measurements': measurements,
                'compression_time': comp_time,
                'pointers': ptrs,
                }
    return result


def move_compressed_cache_defraged(req: Req) -> None:
    cache = get_cache(req)
    cache_size = cache_total_size(cache)

    result: Dict[str, Any] = { 'bytes': cache_size, }
    ptrs = []
    for t in cache_get_all_tensors(cache):
        ptrs.append(t.data_ptr())
    result['pointers'] = ptrs

    algo = 'ANS'
    comp_tensors = cache_compress(cache, algo)
    sz = 0
    cp_arrs = []
    for arr in comp_tensors:
        tmp = torch.tensor(arr.cuda())
        cp_arrs.append(tmp)
        sz = tmp.numel() * tmp.element_size()
    del comp_tensors

    measurements = _do_cpu_gpu_movement_exp(cp_arrs)

    ptrs = [arr.data_ptr() for arr in cp_arrs]

    result[algo] = {
            'bytes': sz,
            'measurements': measurements,
            'compression_time': 0,
            'pointers': ptrs,
            }
    return result


    # # defrag compressed buffers
    # count_arr = len(comp_tensors)
    # num_elemnts = 0
    # for arr in comp_tensors:
    #     num_elemnts += arr.size
    #     assert arr.dtype == cp.int8, 'type is: ' + str(comp_tensors[0].dtype)
    # print(num_elemnts)
    # contiguous_memory = cp.zeros(num_elemnts, dtype=cp.int8)
    # new_arrs = []
    # prev = 0
    # for nv_arr in comp_tensors:
    #     arr = cp.array(nv_arr, copy=False)
    #     assert arr.dtype == cp.int8, 'type is: ' + str(comp_tensors[0].dtype)
    #     next = prev + arr.size
    #     # print(prev, next)
    #     new_arr = contiguous_memory[prev:next]
    #     assert new_arr.size == arr.size, f'{new_arr.size} vs {arr.size}'
    #     cp.copyto(new_arr, arr)
    #     new_nv_arr = nvcomp.as_array(new_arr)
    #     new_arrs.append(new_nv_arr)
    #     prev = next

    # use the defragged values in the next measurements
    # comp_tensors = new_arrs



def report_compression_percent(req: Req, algo: str) -> None:
    cache = get_cache(req)
    cache_size = cache_total_size(cache)
    comp_list = cache_compress(cache, algo)
    comp_size = 0
    for obj in comp_list:
        comp_size += len(obj.data)
    print(algo, ':    ', cache_size, '-->', comp_size)


def assert_compression_decompression_works(req: Req) -> None:
    """
    Make sure when the code for compression decompression is actually working
    correctly.
    """
    print('Computing the kv-cache ...')
    gold_cache = get_cache(req)
    num_layers = len(gold_cache.layers)
    cache_move(gold_cache, DEV_CPU) # we have memory limits

    print('Start compression test ...')
    for algo in COMPRESSION_ALGORITHMS:
        print(algo, '---')

        cache = get_cache(req)
        comp_tensors = cache_compress(cache, algo)
        del cache # this cache is now compressed and its tensors are mangled
        torch.cuda.empty_cache()

        # get compressed size
        comp_sz = 0
        for arr in comp_tensors:
            if isinstance(arr, CompressedData):
                comp_sz += len(arr.data)
            else:
                comp_sz += arr.size * arr.item_size

        decomp_cache = cache_decompress(comp_tensors, num_layers, algo)
        del comp_tensors # we don't need the pointers to arrays, they are mangled
        torch.cuda.empty_cache()
        cache_move(decomp_cache, DEV_CPU) # we have memory limits

        orig_tensors = cache_get_all_tensors(gold_cache)
        decomp_tensors = cache_get_all_tensors(decomp_cache)
        assert len(orig_tensors) == len(decomp_tensors), 'number of tensors must match'

        orig_sz = cache_total_size(gold_cache)
        print('    ', orig_sz, '-->', comp_sz)

        for q, (t1, t2) in enumerate(zip(orig_tensors, decomp_tensors)):
            # Make sure the tensor shape is preserved
            # print('original:', t1.shape)
            # print('decomp:', t2.shape)
            # if t1.shape != t2.shape:
            #     print('Warning: the shapes does not match (maybe the decompressed array is flattened but we should fix this)')

            tmp1 = t1.flatten()
            tmp2 = t2.flatten()
            assert len(tmp1.shape) == 1 and tmp1.shape == tmp2.shape
            assert tmp1.dtype == tmp2.dtype

            if not torch.all(tmp1.eq(tmp2)):
                print('@', q)
                raise RuntimeError('test failed')

        print('   *', algo, 'Passed compression test')


def get_cache(req: Req)-> DynamicCache:
    """
    Load KV cache from disk or compute it
    """
    file_name = f'{CACHE_DIR}/kv_cache_{req.prompt_id}.pt'
    if os.path.isfile(file_name):
        cache = cache_load_from_disk(file_name)
        cache_move(cache, DEV_GPU)
    else:
        cache = req.auto_regression(max_lenght=MAX_CONTENT_LEN)
        cache_save_to_disk(cache, file_name)
    return cache


def _warm_up_comp_arr(comp_arr: List[nvcomp.Array]):
    for arr in comp_arr:
        t = torch.from_dlpack(arr.to_dlpack())

        # print(t.numel(), arr.size)
        # print(t.element_size(), arr.item_size)

        # do something with the tensor
        t += 1
        t -= 1


def _do_cpu_gpu_movement_exp(lst):
    measurements = []

    if isinstance(lst, DynamicCache):
        cache = lst
        for _ in range(REPEAT):

            # to CPU
            torch.cuda.synchronize()
            start = time.perf_counter()
            cache_move(cache, DEV_CPU, non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_cpu_time = end - start

            torch.cuda.empty_cache()

            # to GPU
            torch.cuda.synchronize()
            start = time.perf_counter()
            cache_move(cache, DEV_GPU, non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_gpu_time = end - start

            torch.cuda.empty_cache()

            measurements.append((to_cpu_time, to_gpu_time))
        return measurements

    assert isinstance(lst, list)
    num_arr = len(lst) 
    if num_arr == 0:
        return []

    if isinstance(lst[0], nvcomp.Array):
        # TODO: does cpu() and cuda() always move the data or only once?
        for _ in range(REPEAT):
            # sort the vecrots by pointer address to see if the order matters
            # comp_tensors.sort(key=lambda x: x.__cuda_array_interface__['data'][0])
            # _warm_up_comp_arr(comp_tensors)

            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(num_arr):
                lst[i] = lst[i].cpu()
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_cpu_time = end - start

            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(num_arr):
                lst[i] = lst[i].cuda()
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_gpu_time = end - start

            measurements.append((to_cpu_time, to_gpu_time))

            torch.cuda.empty_cache()
    else:
        assert isinstance(lst[0], torch.Tensor)
        for _ in range(REPEAT):

            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(num_arr):
                lst[i] = lst[i].to(DEV_CPU, non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_cpu_time = end - start

            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(num_arr):
                lst[i] = lst[i].to(DEV_GPU, non_blocking=True)
            torch.cuda.synchronize()
            end = time.perf_counter()
            to_gpu_time = end - start

            measurements.append((to_cpu_time, to_gpu_time))

            torch.cuda.empty_cache()
    
    return measurements

