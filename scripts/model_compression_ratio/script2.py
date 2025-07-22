from transformers import AutoModelForCausalLM, BitsAndBytesConfig, QuantoConfig
import torch
import os
import shutil
import multiprocessing
import subprocess

from transformers import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_name = "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF"

def run_new():
    # Load the original model
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Load on CPU
            load_in_8bit=True  # Enable 8-bit quantization
            )

    return model


def run(quantization: int):
    if quantization == 16:
        # fp16
        model = AutoModelForCausalLM.from_pretrained(model_name,
                torch_dtype=torch.float16, device_map='cpu')
        print('model size (fp16):', model.get_memory_footprint())
        model.save_pretrained("model_files_fp16")

        os.chdir('model_files_fp16')
    if quantization == 8:
        model = run_new()
        # fp8
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     bnb_8bit_compute_dtype=torch.bfloat16, # or torch.float16 depending on your hardware
        # )
        # quantization_config = QuantoConfig(weights="int8")
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='cpu')
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu')

        # model = AutoModel.from_pretrained(model_name)
        # qmodel = torch.quantization.quantize_dynamic(
        #         model,
        #         {torch.nn.Linear},  # Quantize linear layers
        #         dtype=torch.qint8   # INT8 quantization
        #         )
        print('model size (fp8):', model.get_memory_footprint())
        # qmodel.save_pretrained("model_files_fp8")
        model.save_pretrained("model_files_fp8")

        os.chdir('model_files_fp8')
    if quantization == 4:
        raise Exception('err')
        # fp4
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16 depending on your hardware
        # )
        # quantization_config = QuantoConfig(weights="int4")
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #         quantization_config=quantization_config, device_map='cpu')
        print('model size (fp4):', model.get_memory_footprint())
        model.save_pretrained("model_files_fp4")

        os.chdir('model_files_fp4')
    # run the compression
    subprocess.run('pigz -k *.safetensors', shell=True, check=True)
    os.chdir('../')


# Fork process for each case
os.chdir('/home/farbod')
run(8)
# with multiprocessing.Pool(3) as pool:
#     pool.map(run, [16,8,4])
