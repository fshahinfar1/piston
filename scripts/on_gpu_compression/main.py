from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch


DEV_GPU='cuda:0'
DEV_CPU='cpu'


class Req:
    # Model
    M = None
    # Tokenizer
    T = None

    def __init__(self, prompt:str):
        self.prompt = prompt
        self.inputs = self.T(self.prompt, return_tensors='pt')
        self.inputs.to(DEV_GPU)

        # Prefill: run prompt through model, get initial logits and hidden
        # states (past_key_values)
        with torch.no_grad():
            output = self.M(input_ids=self.inputs['input_ids'], use_cache=True)
            self.past_key_values = output.past_key_values
            self.next_token_logits = output.logits[:, -1, :]

        self.move_to(DEV_CPU)

    def move_to(self, dev):
        self.inputs = self.inputs.to(dev)
        self.next_token_logits = self.next_token_logits.to(dev)
        # print('layers:', len(self.past_key_values))
        for kv_layer in self.past_key_values:
            for t in kv_layer:
                t.to(dev)

        # for i in range(len(self.cache.key_cache)):
        #     self.cache.key_cache[i] = self.cache.key_cache[i].to(dev)
        #     self.cache.value_cache[i] = self.cache.value_cache[i].to(dev)

    def auto_regression(self, decode=False):
        # self.move_to(DEV_GPU)

        # Decode: generate step by step using cached states
        generated = self.inputs['input_ids'][:].to(DEV_GPU)

        next_token_logits = self.next_token_logits.to(DEV_GPU)

        cache = DynamicCache.from_legacy_cache(self.past_key_values)
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].to(DEV_GPU)
            cache.value_cache[i] = cache.value_cache[i].to(DEV_GPU)
        past_key_values = cache

        max_lenght = 32
        for _ in range(max_lenght):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # print(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=-1)
            with torch.no_grad():
                output = model(input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True)
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
            # TODO: check if end of sentence then stop

        # Decode output tokens
        if decode:
            final_text = tokenizer.decode(generated[0],
                    skip_special_tokens=True)
            print(final_text)

        # self.move_to(DEV_CPU)


model_name: str = 'openai-community/gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=DEV_GPU, local_files_only=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model.to('cuda:0')
Req.M = model
Req.T = tokenizer

# help(model)

with open('./prompts.txt', 'r') as f:
    all_prompts = f.readlines()
# print(all_prompts)

reqs = [Req(p) for p in all_prompts]

# for r in reqs:
#     r.auto_regression(decode=True)

for i in range(1000):
    print(i, '---'*10)
    for r in reqs:
        r.auto_regression()

