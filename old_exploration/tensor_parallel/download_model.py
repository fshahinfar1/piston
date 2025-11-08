#! python3

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/Phi-3.5-mini-instruct"

w = os.environ['WORK']
u = os.environ['USER']
cache_dir = w + '/' + u + '/' + "models/phi3.5"
print('Cache dir:', cache_dir)

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

print(f"Model and tokenizer cached in {cache_dir}")
