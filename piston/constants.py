import torch

GB = 1024 ** 3
MAX_LENGTH = 1024
NUM_DEVICES = 4
NUM_STAGES = 4
DEV_GPU_ = [torch.device(f'cuda:{i % NUM_DEVICES}') for i in range(NUM_STAGES)]
DEV_CPU = torch.device('cpu')

LOCAL_FILE_ONLY=True

# Use multiprocessing
MPROC_ENABLED=True
PIPE_SIZE =  128 * 1024 * 1024