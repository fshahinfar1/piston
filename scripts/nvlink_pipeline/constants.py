import torch

GB = 1024 ** 3
MAX_LENGTH = 2048
NUM_DEVICES = 1
NUM_STAGES = 3
DEV_GPU_ = [torch.device(f'cuda:{i % NUM_DEVICES}') for i in range(NUM_STAGES)]
DEV_CPU = torch.device('cpu')

LOCAL_FILE_ONLY=True