from typing import *
import time
import torch
from entities import Request, Replica, ExecutionStatistics
from prefill_decode import do_prefill, do_batch_prefill, do_decode, get_batch_size, print_output
from transformers.generation.utils import DynamicCache

from constants import *


class SimplePipeline:
    def __init__(self, model_name: str, num_stages: int, device_list: List,
                    max_length=32, available_memory: int| None =None):
        self.num_stages = num_stages
        self.device_list = device_list
        self.replica = Replica(model_name, num_stages=num_stages, device_list=device_list)
        self.available_memory = available_memory
        self.max_length = max_length
        self.batch_size = get_batch_size(self.replica, max_length, self.available_memory)

        self.rx_queue: List[Request] = []
        self.run_queue: List[Request] = [] # Requests can actually be batched Rquests
    
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
    
    def process_requests(self)-> None:
        if not self.run_queue:
            return

        stat = ExecutionStatistics(self.num_stages)

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]
        
        while self.run_queue:
            req = self.run_queue.pop()
            # move batch of requests to GPUs
            req.move_to(dev_map, non_blocking=True)
            torch.cuda.synchronize()

            do_decode(req, self.replica, stat, max_iter=self.max_length)
            print_output(req)

            #  report_statistics(stat)

            # free memory of requests
            req.cache = DynamicCache()
            req.generated = []
            req.next_token_ids = None
            # torch.cuda.empty_cache()