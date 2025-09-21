from typing import *
import time
import torch
from transformers.generation.utils import DynamicCache

from core.entity import Request
from core.entities import Replica
from core.statistics import ExecutionStatistics
from core.prefill_decode import do_prefill, do_batch_prefill
from utils.memman import get_batch_size, get_max_num_tokens
from constants import *


class SimplePipeline:
    def __init__(self, model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int=1, do_print=False):
        self.num_stages = num_stages
        self.device_list = device_list
        self.replica = Replica(model_name, num_stages=num_stages, device_list=device_list)
        # self.available_memory = available_memory
        self.max_length = max_length
        # self.batch_size = get_batch_size(self.replica, max_length, self.available_memory)
        # self.max_token_limit = get_max_num_tokens(self.replica, self.available_memory)
        self.batch_size = batch_size

        self.rx_queue: List[Request] = []
        self.run_queue: List[Request] = [] # Requests can actually be batched Rquests

        self.do_print = do_print

    def close(self) -> None:
        return

    def print_output(self, req: Request) -> None:
        # Actually generate the text
        if self.do_print:
            generated = torch.cat([t.to(DEV_CPU) for t in req.generated], dim=-1)
            final_text = self.replica.tokenizer.batch_decode(generated,
                                                    skip_special_tokens=True)
            print(final_text)

    def add_request(self, req):
        self.rx_queue.append(req)

    def prepare_run_queue(self) -> None:
        # Do prefill of all request in advance
        for i, req in enumerate(self.rx_queue):
            do_prefill(req, self.replica)
            count = len(req.cache.layers)
            dev_map = [DEV_CPU] * count
            req.move_to(dev_map, non_blocking=True)
            self.run_queue.append(req)
        self.rx_queue.clear()
        torch.cuda.synchronize()

    def prepare_run_queue_batched(self) -> None:
        batch_size = self.batch_size

        while self.rx_queue:
            # dequeue a batch of requests
            batch_requests = self.rx_queue[:batch_size]
            del self.rx_queue[:batch_size]

            # NOTE: do prefill on a list of Request to have the inputs and
            # embedding padded correctly for batch processing
            batched_req = do_batch_prefill(batch_requests, self.replica)
            count = len(batched_req.cache.layers)
            dev_map = [DEV_CPU] * count
            batched_req.move_to(dev_map, non_blocking=True)
            self.run_queue.append(batched_req)
        torch.cuda.synchronize()

    def _do_process_reqeust(self, req: Request) -> None:
        # stat = ExecutionStatistics(self.num_stages)
        stat = None
        for _ in range(self.max_length):
            next_token = self.replica.do_one_iteration(req, stat)
            req.generated.append(next_token)
            req.next_token_ids = next_token

        self.print_output(req)
        # stat.report()
        tmp = req.cache.layers[0].keys.shape
        print('Req', req.id, 'size:', req.cache_size_bytes(), 'number of tokens:', tmp[2], f'(shape: {tmp})')
        req.free()

    def process_requests(self)-> None:
        if not self.run_queue:
            return

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]

        while self.run_queue:
            req = self.run_queue.pop()
            # move batch of requests to GPUs
            req.move_to(dev_map, non_blocking=True)
            self._do_process_reqeust(req)

