from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
import torch


def test():
    """
    just to test what a normal execution of model produces. Usuful to see how
    much memory is consumed.
    """
    model_name = 'microsoft/Phi-3.5-mini-instruct'
    prompt = 'write 2 page essay on importance of sunshine'
    model = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16, device_map='cuda:0',
            local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model.to('cuda:0')

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to('cuda:0')

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
        past_key_values = output.past_key_values
        next_token_logits = output.logits[:, -1, :]

    generated = input_ids
    next_token_logits = next_token_logits

    cache = DynamicCache.from_legacy_cache(past_key_values)
    past_key_values = cache

    for _ in range(256):
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

    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(final_text)


if __name__ == '__main__':
    test()
