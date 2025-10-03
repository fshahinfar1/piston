from typing import *
import argparse
import torch
import time

GB = 1024 ** 3
NUM_DEVICES = 2
DEV_GPU_ = [torch.device(f'cuda:{i % NUM_DEVICES}') for i in range(NUM_DEVICES)]
DEV_CPU_ = torch.device('cpu')

def stats(lst: List[float]) -> Tuple[float, float, float, float, float]:
    if not lst:
        return 0, 0, 0, 0, 0
    S = sorted(lst)
    mean = sum(S) / len(S)
    std = (sum((x - mean) ** 2 for x in S) / len(S)) ** 0.5
    mid = S[len(S) // 2] if len(S) % 2 == 1 else (S[len(S) // 2 - 1] + S[len(S) // 2]) / 2
    return mean, std, mid, S[-1], len(S)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int, help='How many GB of data to move?')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    main_dev = DEV_GPU_[0]
    other_dev = DEV_CPU_
    # other_dev = DEV_GPU_[1]

    print('Moving:', args.size, 'GB', f'tensor back and forth {main_dev} <--> {other_dev}')

    tensor_size = args.size * GB
    assert tensor_size < 64 * GB * 0.98
    numel = int(tensor_size // 2) # float16 is 2 bytes

    # Reserve memory on both devices
    torch.empty(numel, dtype=torch.float16, device=main_dev)
    torch.empty(numel, dtype=torch.float16, device=other_dev)

    # Create a random tensor
    large_tensor = torch.rand(numel, dtype=torch.float16, device=main_dev)

    # Move data repeatedly across two device and measure things
    time_measurements = []
    repeat = 40
    target = 1

    devs = [main_dev, other_dev]

    for _ in range(repeat):
        torch.cuda.synchronize()

        # -------------------------------------- 
        start = time.time()

        large_tensor = large_tensor.to(devs[target])
        torch.cuda.synchronize()

        end = time.time()
        # -------------------------------------- 

        duration = (end - start) * 1000
        time_measurements.append(duration)

        # do not free the reserved area
        # torch.cuda.memory.empty_cache()

        target = 0 if target == 1 else 1
        print(duration, 'ms')

    # Report measurements
    mean, std, mid, _max, count = stats(time_measurements)
    print(f'Mean: {mean:.3f} ms, Std: {std:.3f}, Median: {mid:.3f} ms , Max: {_max:.3f} ms, Count: {count}')

    throughput = [(tensor_size / GB) / (t / 1000) for t in time_measurements] # GB/s
    mean, std, mid, _max, count = stats(throughput)
    print(f'Mean: {mean:.3f} GB/s, Std: {std:.3f}, Median: {mid:.3f} GB/s , Max: {_max:.3f} GB/s, Count: {count}')


if __name__ == '__main__':
    main()
