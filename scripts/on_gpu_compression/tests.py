from typing import *
import torch
from cache_helper import *
import time
import os
import os
from req import Req

MAX_CONTENT_LEN: int = 2 ** 13
DEV_GPU ='cuda:0'
DEV_CPU ='cpu'
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
    time_measurements = []

    for _ in range(REPEAT):
        # to CPU
        start = time.perf_counter()
        cache_move(cache, DEV_CPU)
        end = time.perf_counter()
        to_cpu_time = end - start

        torch.cuda.empty_cache()

        # to GPU
        start = time.perf_counter()
        cache_move(cache, DEV_GPU)
        end = time.perf_counter()
        to_gpu_time = end - start

        torch.cuda.empty_cache()

        time_measurements.append((to_cpu_time, to_gpu_time))

    return { 'bytes': cache_size, 'measurements': time_measurements, }


def measure_compressed_kv_cache_moving_time(req) -> Dict[str, Any]:
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
            sz += arr.size * arr.item_size

        measurements = []
        # TODO: does cpu() and cuda() always move the data or only once?
        num_tensors = len(comp_tensors)
        for _ in range(REPEAT):
            start = time.perf_counter()
            for i in range(num_tensors):
                comp_tensors[i] = comp_tensors[i].cpu()
            end = time.perf_counter()
            to_cpu_time = end - start

            torch.cuda.empty_cache()

            start = time.perf_counter()
            for i in range(num_tensors):
                comp_tensors[i] = comp_tensors[i].cuda()
            end = time.perf_counter()
            to_gpu_time = end - start

            measurements.append((to_cpu_time, to_gpu_time))

            torch.cuda.empty_cache()


        ptrs = []
        for arr in comp_tensors:
            t = torch.from_dlpack(arr)
            ptrs.append(t.data_ptr())

        result[algo] = {
                'bytes': sz,
                'measurements': measurements,
                'compression_time': comp_time,
                'pointers': ptrs,
                }
    return result


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

