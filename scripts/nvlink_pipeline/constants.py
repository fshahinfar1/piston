import torch

GB = 1024 ** 3
MAX_LENGTH = 1024
NUM_DEVICES = 3
NUM_STAGES = 3
DEV_GPU_ = [torch.device(f'cuda:{i % NUM_DEVICES}') for i in range(NUM_STAGES)]
DEV_CPU = torch.device('cpu')

# TODO: for experimenting reasons I have limited the KV-Cache size to 3 GB
# AVAILABLE_MEMORY = 3 * GB
# AVAILABLE_MEMORY = 50*GB

# BATCH_SIZE = 3

# Total number of requests to process
# PILE_SIZE = BATCH_SIZE * 8

LOCAL_FILE_ONLY=True

# MODE = 'simple'
# MODE = 'swapping'
