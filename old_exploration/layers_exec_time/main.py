from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def main() -> None:
    # model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_name: str =  "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            load_in_8bit=True  # Enable 8-bit quantization
            )

    prompt: str = "Explain what a Mixture of Experts is in less than 100 words."

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # output = pipe(prompt)
    # print(output[0]["generated_text"])

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Prefill: get initial model state and past_key_values (the KV cache)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        generated_ids = input_ids

    # Decode: repeatedly generate one token at a time using the cache
    for _ in range(20):  # Number of tokens to generate
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # Feed only the last token, and provide past_key_values for efficient decoding
        outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

    # Decode the full sequence
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(result)


if __name__ == '__main__':
    main()

