from typing import *
from .entities import Request, Replica


def do_prefill(req, replica):
    # tokenize
    inputs = replica.tokenizer(req.prompt, return_tensors='pt')
    req.next_token_ids = inputs['input_ids']
    # req.generated = torch.cat([req.generated, req.next_token_ids], dim=-1)
    req.generated.append(req.next_token_ids)

    next_token = replica.do_one_iteration(req)
    next_token = next_token.cpu()

    # req.generated = torch.cat([req.generated, next_token], dim=-1)
    req.next_token_ids = next_token
    req.generated.append(req.next_token_ids)


def do_decode(req, replica, stat, max_iter=32) -> None:
    for _ in range(max_iter):
        next_token = replica.do_one_iteration(req, stat)

        # TODO: maybe avoid moving to CPU? or copy to CPU and other GPU async
        # next_token = next_token.cpu()
        # req.generated = torch.cat([req.generated, next_token], dim=-1)
        req.generated.append(next_token)

        req.next_token_ids = next_token

        # TODO: what should I do when having a batched request?
        # if next_token.item() == replica.tokenizer.eos_token_id:
        #     break
        
        # torch.cuda.empty_cache()

def print_output(req):
    # Actually generate the text
    # generated = torch.cat([t.to(DEV_CPU) for t in req.generated], dim=-1)
    # final_text = replica.tokenizer.decode(generated[0], skip_special_tokens=True)
    # print(final_text)
    return

def do_batch_prefill(requests: List[Request], replica: Replica) -> Request:
    import sys
    # tokenize
    prompts = [req.prompt for req in requests]
    inputs = replica.tokenizer(prompts, return_tensors='pt', padding=True)

    # print(inputs)
    # sys.exit(1)

    # Create a request to represent the batched request
    req = Request('')
    req.next_token_ids = inputs['input_ids']
    req.attention_mask = inputs['attention_mask']
    req.generated.append(req.next_token_ids)

    next_token = replica.do_one_iteration(req)
    next_token = next_token.cpu()

    req.next_token_ids = next_token
    req.generated.append(req.next_token_ids)

    # print(len(requests))
    # for r in requests:
    #     print(r.prompt)
    # print('Req', req.id, 'size:', req.bytes())

    return req