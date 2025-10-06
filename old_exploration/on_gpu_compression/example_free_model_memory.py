from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch
from time import sleep


DEV_GPU='cuda:0'
DEV_CPU='cpu'


def report_gpu_memory() -> None:
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('total:', t, 'free:', f)


model_name: str = 'microsoft/Phi-3.5-mini-instruct'
model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16,
        device_map=DEV_GPU,
        local_files_only=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name,
        local_files_only=True)
model.to(DEV_GPU)
# help(model)

report_gpu_memory()

print('moving to cpu ...')
model.to(DEV_CPU)
torch.cuda.empty_cache()
report_gpu_memory()
sleep(20)

print('deleting model ...')
del model
del tokenizer
torch.cuda.empty_cache()
report_gpu_memory()

sleep(20)
print('Done')
