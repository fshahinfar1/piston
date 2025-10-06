from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch
from cache_helper import *

DEV_GPU='cuda:0'
DEV_CPU='cpu'
COMPRESSION_ALGORITHMS = ["ANS", "LZ4", "Cascaded", "GDeflate"]

# 'openai-community/gpt2'
model_name: str = 'microsoft/Phi-3.5-mini-instruct'
model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16, device_map=DEV_GPU, local_files_only=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model.to(DEV_GPU)


def move_model_to(dev: str) -> None:
    global model
    model = model.to(dev)
    torch.cuda.empty_cache()


class Req:
    counter = 0

    def __init__(self, prompt:str):
        self.prompt = prompt
        self.inputs = tokenizer(self.prompt, return_tensors='pt')
        self.inputs.to(DEV_GPU)

        self.prompt_id = Req.counter
        Req.counter += 1

        # Prefill: run prompt through model, get initial logits and hidden
        # states (past_key_values)
        with torch.no_grad():
            output = model(input_ids=self.inputs['input_ids'], use_cache=True)
            self.past_key_values = output.past_key_values
            self.next_token_logits = output.logits[:, -1, :]

        self.move_to(DEV_CPU)

    def move_to(self, dev:str) -> None:
        self.inputs = self.inputs.to(dev)
        self.next_token_logits = self.next_token_logits.to(dev)
        # print('layers:', len(self.past_key_values))
        for kv_layer in self.past_key_values:
            for t in kv_layer:
                t.to(dev)

    def auto_regression(self, decode=False, max_lenght=32) -> DynamicCache:
        # Decode: generate step by step using cached states

        # Make a copy of tensors we use in decode phase so that we
        # can repeat this multiple times without mutating the values
        # we computed during prfill
        generated = self.inputs['input_ids'][:].to(DEV_GPU)

        next_token_logits = self.next_token_logits.to(DEV_GPU)

        cache = DynamicCache.from_legacy_cache(self.past_key_values)
        cache_move(cache, DEV_GPU)
        past_key_values = cache

        for _ in range(max_lenght):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, next_token], dim=-1)
            with torch.no_grad():
                output = model(input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True)
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]

        # Decode output tokens
        if decode:
            final_text = tokenizer.decode(generated[0],
                    skip_special_tokens=True)
            print(final_text)

        return past_key_values

