from typing import *
import torch
from transformers.generation.utils import DynamicCache

from piston.core.pipeline.simple_pipeline import SimplePipeline
from piston.core.prefill_decode import print_output
from piston.utils.worker import Worker, Promise


class SwappingPipeline_CUDA_GRAPH(SimplePipeline):
    def __init__(self, spare_memory_device, model_name: str, num_stages: int,
                    device_list: List, max_length=32, batch_size: int = 1):

        assert num_stages == 2, 'The implementation assumes there are two stages'

        super().__init__(model_name, num_stages, device_list, max_length, batch_size)
        # Device on which we hold temporary data
        self.spare_memory_device = spare_memory_device

        # Allocate and cache memory on spare device
        # numel = int(available_memory // 2)
        # torch.empty(numel, dtype=torch.float16, device=self.spare_memory_device)

        # For falling back to simple pipeline
        self._do_simple_pipeline_req_processing = super()._do_process_reqeust

        # For knowing which request we are operating when a layer-finish event is observed
        self._active_request: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None,] * len(self.replica.stages)

    def _register_for_layer_completion(self):
        # Register layer completion callbacks for moving KV-cache layer by layer
        for index, stage in enumerate(self.replica.stages):
            stage.register('layer_finish', self._move_kv_cache_layer, index)
    
    def _unregister_from_layer_completion(self):
        for index, stage in enumerate(self.replica.stages):
            stage.unregister('layer_finish', self._move_kv_cache_layer)

    def _finish_req_iter(self, req) -> None:
        logits = self.replica.lm_head(req.next_token_ids)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        req.next_token_ids = next_token
        req.generated.append(next_token)

    def _do_decode_on_stage(self, req1, stage_index, finish) -> None:
        """
        Calculate MLP feed forward part of one iteration and move the
        hidden_state to the next device of stage.

        This function is delegated to workers to run concurrent with
        data-movement.
        """
        stage = self.replica.stages[stage_index]

        next_stage_index = (stage_index + 1) % len(self.replica.stages)
        next_stage = self.replica.stages[next_stage_index]

        out = stage.forward(req1.next_token_ids,
                            attention_mask=req1.attention_mask,
                            use_cache=True,
                            past_key_values=req1.cache)

        # Move the activation values to the next stage
        req1.next_token_ids = out.last_hidden_state
        # print('Req', req1.id, 'size:', req1.bytes())

        if finish:
            # At this point an iteration of req1 is done
            self._finish_req_iter(req1)

        req1.next_token_ids = req1.next_token_ids.to(next_stage.device, non_blocking=True)
    
    def _do_move_kv_cache_layer(self, stage_index, layer_index):
        """
        This function should run in a kv-cache worker thread. So the blocking
        move should be fine.
        """
        tmp = self._active_request[stage_index]
        assert tmp is not None
        req1, req2 = tmp
        assert req1 is not None
        assert req2 is not None

        stage = self.replica.stages[stage_index]

        # move this layer of KV cache to the spare device in parallel
        dev_map = [None] * len(req1.cache.layers)
        dev_map[layer_index] = self.spare_memory_device
        req1.move_to(dev_map, non_blocking=True)

        # bring the layer for the other KV cache to this device
        dev_map = [None] * len(req2.cache.layers)
        dev_map[layer_index] = stage.device
        req2.move_to(dev_map, non_blocking=True)

    def _move_kv_cache_layer(self, stage_index, layer_index):
        """
        This is called when forward pass finishes processing a layer.
        Start moving that layers KV-cache to spare memory

        TODO: start loading the KV-cache for that layer of other requests

        stage: user provided state when registering the callback
        layer_index: index of layer that got finished
        """
        w = self._kv_cache_workers[stage_index]
        p = w.add_task(self._do_move_kv_cache_layer, stage_index, layer_index)
        self._in_flight_copies[stage_index].append(p)
 
    def swapping_decode_on_stage(self, req1, req2, stage_index, finish) -> Promise:
        """
        ATTENTION: The order of calling this function on requests and
                   stages matter and its your responsibility to get it
                   right!

        ATTENTION: call with torch.no_grad

        Process req1 on stage[stage_index]. Move the generated hidden value to
        next stage.  Meanwhile move the KV-Cache of req1 on stage_index to spare
        memory while loading the KV-cache req2 to stage_index
        """
        stage = self.replica.stages[stage_index]

        # mark the active request
        self._active_request[stage_index] = (req1, req2)

        worker = self._workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % self._count_worker

        # Run forward pass in a worker thread
        fn = self._do_decode_on_stage
        promise = worker.add_task(fn, req1, stage_index, finish)

        # NOTE: When the worker finishes processing a layer _move_kv_cache_layer
        # is called and that layers KV cache is moved
 
        def local_load_other_req():
            # wait until req1 calculation is over
            promise.wait()

            # wait until req1 kv cache is copied
            for p in self._in_flight_copies[stage_index]:
                p.wait()
            self._in_flight_copies[stage_index].clear()
            # torch.cuda.synchronize(stage.device)

            # Move the KV-Cache layers from the spare memory to this stage
            # This step must happen after 1) forward pass & 2) moving KV to
            # spare memory. Otherwise we must run out of memory when we are
            # fully utilizing the GPUs
            # num_layers = len(req2.cache.layers)
            # per_layer =  int(num_layers // 2)
            # f = stage.first_layer_index 
            # t = f + per_layer
            # dev_map = [None] * num_layers
            # dev_map[f:t] = [stage.device, ] * per_layer
            # req2.move_to(dev_map, non_blocking=True)

        # NOTE: Add the task to the same worker that we assigned MLP feed
        # forward.  because we are waiting for its completion anyway.
        promise2 = worker.add_task(local_load_other_req)

        return promise2

    def process_requests(self)-> None:
        assert len(self.rx_queue) == 0
        if not self.run_queue:
            return

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]

        dev_spare_mem = [self.spare_memory_device] * count

        stage_zero_dev = self.replica.stages[0].device

        # start listening to layer completion events
        self._register_for_layer_completion()

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

            # TODO: maybe open two streams
            # Move batch of requests to GPUs
            req1.move_to(dev_map, non_blocking=True)
            req2.move_to(dev_spare_mem, non_blocking=True)
            torch.cuda.synchronize()

            # stat = ExecutionStatistics(self.num_stages)

            # Start by running processing on stage 0 and passing the hidden-state to stage 1
            # while loading the req2 to stage 0
            self.swapping_decode_on_stage(req1, req2, 0, finish=False).wait()
            # torch.cuda.synchronize()

            # TODO: Improve the code to support multiple stages not just two
            for _ in range(self.max_length):
                # Calculate req2 on stage 0 and initiate swapping back to req1
                x = self.swapping_decode_on_stage(req2, req1, 0, finish=False)
                # Also involve the second stage in processing
                y = self.swapping_decode_on_stage(req1, req2, 1, finish=True)

                x.wait()
                y.wait()
                # torch.cuda.synchronize()

                # Complete a swapping cycle
                x = self.swapping_decode_on_stage(req1, req2, 0, finish=False) # TODO: on the last iteration we should not do this
                y = self.swapping_decode_on_stage(req2, req1, 1, finish=True)

                x.wait()
                y.wait()
                # torch.cuda.synchronize()

            print_output(req1)
            print_output(req2)

            # stat.report()

            # free memory of requests
            print('Req', req1.id, 'size:', req1.bytes())
            print('Req', req2.id, 'size:', req2.bytes())

            req1.free()
            req2.free()
            # torch.cuda.empty_cache()
        
        self._unregister_from_layer_completion()

