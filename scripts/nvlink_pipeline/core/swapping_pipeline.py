from typing import *
import torch
from transformers.generation.utils import DynamicCache

from .simple_pipeline import SimplePipeline
from .prefill_decode import print_output

from utils.worker import Worker, Promise

class SwappingPipeline(SimplePipeline):
    def __init__(self, spare_memory_device, model_name: str, num_stages: int, device_list: List,
                    max_length=32, available_memory: int = 64):
        
        assert num_stages == 2, 'The implementation assumes there are two stages'

        super().__init__(model_name, num_stages, device_list, max_length, available_memory)
        # Device on which we hold temporary data
        self.spare_memory_device = spare_memory_device

        # Allocate and cache memory on spare device
        numel = int(available_memory // 2)
        torch.empty(numel, dtype=torch.float16, device=self.spare_memory_device)

        # For falling back to simple pipeline
        self._do_simple_pipeline_req_processing = super()._do_process_reqeust

        self._count_worker = 4
        self._next_worker = 0
        self._workers = [Worker() for _ in range(self._count_worker)]
    
    def _free_req(self, req):
        req.cache = DynamicCache()
        req.generated = []
        req.next_token_ids = None
    
    def _finish_req_iter(self, req) -> None:
        logits = self.replica.lm_head(req.next_token_ids)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        req.next_token_ids = next_token
        req.generated.append(next_token)
    
    def _do_swapping_decode_on_stage(self, req1, req2, stage_index, finish) -> None:
        """
        ATTENTION: The order of calling this function on requests and
                   stages matter and its your responsibility to get it
                   right!
        
        ATTENTION: call with torch.no_grad

        Process req1 on stage[stage_index]. Move the generated hidden value to
        next stage.  Meanwhile move the KV-Cache of req1 on stage_index to spare
        memory while loading the KV-cache req2 to stage_index
        """
        replica = self.replica
        stage = self.replica.stages[stage_index]

        next_stage_index = (stage_index + 1) % len(self.replica.stages)
        next_stage = self.replica.stages[next_stage_index]

        cache = req1.cache
        hidden_state = req1.next_token_ids

        # assert(str(hidden_state.device) == str(stage.device))

        with torch.Stream(device=stage.device) as s_cuda:
            out = stage.forward(hidden_state,
                                attention_mask=req1.attention_mask,
                                use_cache=True,
                                past_key_values=cache)
        
            # Move the activation values to the next stage
            hidden_state = out.last_hidden_state

            if finish:
                # At this point an iteration of req1 is done
                req1.next_token_ids = hidden_state
                # print('Finish at stage:', stage_index,
                #       'Hidden state is on:', str(req1.next_token_ids.device),
                #       'other is on:', str(self.replica.lm_head.weight.device))
                self._finish_req_iter(req1)
                req1.next_token_ids = req1.next_token_ids.to(next_stage.device, non_blocking=True)
            else:
                req1.next_token_ids = hidden_state.to(next_stage.device, non_blocking=True)
        

        # TODO: move KV-Cache layer by layer
        # Move the KV-Cache layers from this stage to spare memory
        num_layers = len(req1.cache.layers)
        per_layer =  int(num_layers // 2)
        dev_map = [None] * num_layers

        f = stage_index * per_layer
        t = f + per_layer
        dev_map[f:t]  = [self.spare_memory_device, ] * per_layer

        # Also move KV cache to the spare device in parallel
        with torch.Stream(device=stage.device) as s_cuda:
            req1.move_to(dev_map, non_blocking=True)

        # Move the KV-Cache layers from the spare memory to this stage
        # This step must happen after 1) forward pass & 2) moving KV to spare memory
        # Otherwise we must run out of memory when we are fully utilizing the GPUs
        dev_map = [None] * num_layers
        dev_map[f:t] = [stage.device, ] * per_layer
        with torch.Stream(device=stage.device) as s_cuda:
            req2.move_to(dev_map, non_blocking=True)

    def swapping_decode_on_stage(self, req1, req2, stage_index, finish) -> Promise:
        worker = self._workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % self._count_worker
        promise = worker.add_task(SwappingPipeline._do_swapping_decode_on_stage, self, req1, req2,
                                                    stage_index, finish)
        return promise
    
    def process_requests(self)-> None:
        if not self.run_queue:
            return

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]

        dev_spare_mem = [self.spare_memory_device] * count

        stage_zero_dev = self.replica.stages[0].device
        
        while self.run_queue:

            # Check if there is only one request/batch of request left then fall
            # back to simple pipeline
            if len(self.run_queue) == 1:
                req = self.run_queue.pop()
                req.move_to(dev_map, non_blocking=True)
                req.next_token_ids = req.next_token_ids.to(stage_zero_dev)
                self._do_simple_pipeline_req_processing(req)
                continue

            # Work with two request and swap them in and out
            req1 = self.run_queue.pop()
            req2 = self.run_queue.pop()

            # Make sure initial token of both requests are on the right device
            req1.next_token_ids = req1.next_token_ids.to(stage_zero_dev)
            req2.next_token_ids = req2.next_token_ids.to(stage_zero_dev)

            # Move batch of requests to GPUs
            req1.move_to(dev_map, non_blocking=True)
            req2.move_to(dev_spare_mem, non_blocking=True)
            # torch.cuda.synchronize()

            # stat = ExecutionStatistics(self.num_stages)
            with torch.no_grad():
                # Start by running processing on stage 0 and passing the hidden-state to stage 1
                # while loading the req2 to stage 0
                self._do_swapping_decode_on_stage(req1, req2, 0, finish=False)
                # torch.cuda.synchronize()

                # TODO: Improve the code to support multiple stages not just two
                for _ in range(self.max_length):
                    # Calculate req2 on stage 0 and initiate swapping back to req1
                    x = self.swapping_decode_on_stage(req2, req1, 0, finish=False)
                    # Also involve the second stage in processing
                    y = self.swapping_decode_on_stage(req1, req2, 1, finish=True)
                    # torch.cuda.synchronize()

                    x.wait()
                    y.wait()

                    # Complete a swapping cycle
                    x = self.swapping_decode_on_stage(req1, req2, 0, finish=False) # TODO: on the last iteration we should not do this
                    y = self.swapping_decode_on_stage(req2, req1, 1, finish=True)
                    # torch.cuda.synchronize()

                    x.wait()
                    y.wait()

            print_output(req1)
            print_output(req2)

            # stat.report()

            # free memory of requests
            self._free_req(req1)
            self._free_req(req2)
            # torch.cuda.empty_cache()

        print('done')
        for worker in self._workers:
            worker.die()
