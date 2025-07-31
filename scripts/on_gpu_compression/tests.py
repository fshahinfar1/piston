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


def measure_kv_cache_moving_time(req: Req) -> Dict[str, Any]:
    """
    Do decode procedure to create a full kv cache. Then move it
    between CPU and GPU
    report size and transmission time
    """
    cache = get_cache(req)
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
    repeat = 40

    cache_size = cache_total_size(cache)
    result: Dict[str, Any] = { 'bytes': cache_size, }
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
        for _ in range(repeat):
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

        result[algo] = {
                'bytes': sz,
                'measurements': measurements,
                'compression_time': comp_time,
                }
    return result


def assert_compression_decompression_works(req: Req) -> None:
    """
    Make sure when the code for compression decompression is actually working
    correctly.
    """
    from nvidia import nvcomp
    import cupy as cp
    print('Computing the kv-cache ...')

    # cache = get_cache(req)


    codec = nvcomp.Codec(algorithm='LZ4')
    # tensor = cache.key_cache[0]
    # tensor = tensor.flatten()

    # arr = cp.asarray(tensor, dtype=cp.float16)
    # for i in range(tensor.numel()):
    #     assert tensor[i] == arr[i]


    arr = cp.random.rand(4000).astype(cp.float16)
    comp = codec.encode(nvcomp.as_array(arr))

    print('orig:', arr.size * arr.itemsize)
    print('comp:', comp.size * comp.item_size)

    decomp = decomp = codec.decode(comp)
    t2 = cp.array(decomp, dtype=cp.float16)

    print('orig shape:', arr.size)
    print('decomp shape:', t2.size)

    print(t2.shape)
    print(t2[0])
    print(arr[0])


    # print('Start compression test ...')
    # for algo in COMPRESSION_ALGORITHMS:
    #     comp_tensors = cache_compress(cache, algo)
    #     decomp_cache = cache_decompress(comp_tensors, len(cache.key_cache), algo)
    #     for t1, t2 in zip(cache.key_cache + cache.value_cache,
    #             decomp_cache.key_cache + decomp_cache.value_cache):
    #         # Make sure the tensor shape is preserved
    #         print('original:', t1.shape)
    #         print('decomp:', t2.shape)
    #         if t1.shape != t2.shape:
    #             print('Warning: the shapes does not match (maybe flattened)')

    #         tmp1 = t1.flatten()
    #         tmp2 = t2.flatten()
    #         assert len(tmp1.shape) == 1 and tmp1.shape == tmp2.shape

    #         for a,b in zip(tmp1, tmp2):
    #             print(a,b)
    #             assert a == b, f'The {algo} compression/decompression did not work correctly'
    #     print('  --', algo, 'Passed compression test')


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

