import argparse
import torch
import time

GB = 1024 ** 3

def stats(lst):
    if not lst:
        return 0, 0, 0, 0, 0
    S = sorted(lst)
    mean = sum(S) / len(S)
    std = (sum((x - mean) ** 2 for x in S) / len(S)) ** 0.5
    mid = S[len(S) // 2] if len(S) % 2 == 1 else (S[len(S) // 2 - 1] + S[len(S) // 2]) / 2
    return mean, std, mid, S[-1], len(S)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int, help='GBs of data to move to each device')
    parser.add_argument('gpus', type=int, help='number of GPUs in the NVLink domain')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    devs = [torch.device(f'cuda:{i}') for i in range(args.gpus)]

    assume_kv = True
    sub_tensors = 1
    if assume_kv:
        # send two separate tensors
        sub_tensors = 2


    tensor_size = args.size * GB / sub_tensors

    numel = int(tensor_size // 2) # sending float16 which is 2 bytes

    main_dev = devs[0]
    other_devs = devs[1:]
    print(f'Moving {args.size} GB from {main_dev} to each of {other_devs} concurrently')

    buffers = []

    main_tensor = [torch.rand(numel, dtype=torch.float16, device=main_dev) for _ in range(sub_tensors)]
    for d in other_devs:
        b = [torch.empty(numel, dtype=torch.float16, device=d) for _ in range(sub_tensors)]
        buffers.append(b)
    
    # time spans are in seconds
    time_measurements = []
    repeat = 40

    # create a stream for each target destinateion on the main devices
    streams = [torch.cuda.Stream(device=main_dev, priority=0) for _ in other_devs]

    for _ in range(repeat):
        torch.cuda.synchronize()

        # ------
        start = time.time()

        src = main_tensor
        for stream, dst in zip(streams, buffers):
            with torch.cuda.stream(stream):
                # s and d are sub tensors and buffers :)
                for s, d in zip(src, dst):
                    d.copy_(s, non_blocking=True)
        
        torch.cuda.synchronize()

        duration = time.time() - start
        time_measurements.append(duration)

        print(duration * 1000, 'ms')

    # Report measurements
    mean, std, mid, _max, count = stats([t * 1000 for t in time_measurements])
    print(f'Mean: {mean:.3f} ms, Std: {std:.3f}, Median: {mid:.3f} ms , Max: {_max:.3f} ms, Count: {count}')

    throughput = [(args.gpus - 1) * (sub_tensors * tensor_size / GB) / (t) for t in time_measurements] # GB/s
    mean, std, mid, _max, count = stats(throughput)
    print(f'Mean: {mean:.3f} GB/s, Std: {std:.3f}, Median: {mid:.3f} GB/s , Max: {_max:.3f} GB/s, Count: {count}')

if __name__ == '__main__':
    main()
