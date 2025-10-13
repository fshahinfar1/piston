from typing import *
import torch
from transformers.generation.utils import DynamicCache

from piston.core.pipeline.simple_pipeline import SimplePipeline
from piston.utils.worker import Worker, Promise


class SwappingPipeline(SimplePipeline):
    def __init__(self, spare_memory_devices: List[torch.device],
                 model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int = 64, do_print=False):

        assert num_stages == 2, 'The implementation assumes there are two stages'

        super().__init__(model_name, num_stages, device_list, max_length,
                            batch_size, do_print=do_print)
        # Device on which we hold temporary data
        self.spare_memory_devices = spare_memory_devices

        # For falling back to simple pipeline
        self._do_simple_pipeline_req_processing = super()._do_process_reqeust

        # For computing forward pass in background
        self._count_worker = 2
        self._next_worker = 0
        self._workers = [Worker() for _ in range(self._count_worker)]

        # For moving kv caches
        self._in_flight_copies: Dict[int, List[Promise]] = {i: [] for i in range(num_stages)}

        self._active_request: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None,] * len(self.replica.stages)

        self._streams = [[torch.cuda.Stream(stage.device, priority=0) for i in range(5)]
                            for stage in self.replica.stages]
        
        # self._mproc = True
        # if self._mproc:
        #     self.pipe, other_end = multiprocessing.Pipe()
        #     self.proc = multiprocessing.Process(target=self._main, args=(other_end,))

    # def _main(self, pipe):
    #     """
    #     Main function of stage if its running in multiprocessing mode
    #     """
    #     while True:
    #         if not pipe.poll():
    #             continue
    #         req = pipe.recv()
    #         self.forward(req)
    #         pipe.send(req)
    
    # def mproc_forward(self, req):
    #     self.pipe.send(req)
    
    # def mproc_block_for_response(self):
    #     new_req = self.pipe.recv()
    #     return new_req
        
    def __del__(self):
        self.close()

    def close(self) -> None:
        """
        Close pipeline and release resources
        """
        for w in self._workers:
            w.die()

        self._unregister_from_layer_completion()

    def _select_spare_memory_device(self, stage_index: int) -> torch.device:
        index = stage_index % len(self.spare_memory_devices)
        dev = self.spare_memory_devices[index]
        return dev

    def _register_for_layer_completion(self):
        # Register layer completion callbacks for moving KV-cache layer by layer
        for index, stage in enumerate(self.replica.stages):
            stage.register('layer_finish', self._move_kv_cache_layer, index)

    def _unregister_from_layer_completion(self):
        for index, stage in enumerate(self.replica.stages):
            stage.unregister('layer_finish', self._move_kv_cache_layer)

    def _finish_req_iter(self, req) -> None:
        logits = self.replica.lm_head(req.hidden_states)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        req.next_token_ids = next_token
        req.generated.append(next_token)

        # Update attention to use the new token
        mask = req.attention_mask
        mask = torch.cat([mask, mask.new_ones((mask.size(0), 1))], dim=-1)
        req.attention_mask = mask 

        req.clear_hidden_states()

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

        compute_stream = self._streams[stage_index][0]
        with compute_stream:
            with torch.no_grad():
                out = stage.forward(req1)

            # Move the activation values to the next stage
            # req1.next_token_ids = out.last_hidden_state
            # print('Req', req1.id, 'size:', req1.bytes())

            if finish:
                # At this point an iteration of req1 is done
                self._finish_req_iter(req1)
                # Get ready for the generation of next token
                req1.hidden_states = req1.next_token_ids

            req1.move_hidden_state_to(next_stage.device, non_blocking=True)

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

        spare_device = self._select_spare_memory_device(stage_index)
        s1 = self._streams[stage_index][1]
        s2 = self._streams[stage_index][2]

        # offload layer
        req1.move_single_layer_to(layer_index, spare_device, s1, s2, non_blocking=True)

        s1 = self._streams[stage_index][3]
        s2 = self._streams[stage_index][4]
        # bring the layer for the other KV cache to this device
        req2.move_single_layer_to(layer_index, stage.device, s1, s2, non_blocking=True)

    def _move_kv_cache_layer(self, stage_index, layer_index):
        """
        This is called when forward pass finishes processing a layer.
        Start moving that layers KV-cache to spare memory

        stage: user provided state when registering the callback
        layer_index: index of layer that got finished
        """
        self._do_move_kv_cache_layer(stage_index, layer_index)

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

        # set this to bring us back to main thread 
        promise = worker.add_task(fn, req1, stage_index, finish)

        return promise

    def process_requests(self)-> None:
        assert len(self.rx_queue) == 0
        if not self.run_queue:
            return

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]

        dev_spare_mem: List[torch.device] = []
        for stage_index, stage in enumerate(self.replica.stages):
            dev = self._select_spare_memory_device(stage_index)
            num_stage_layers = len(stage.layers)
            dev_spare_mem.extend([dev,] * num_stage_layers)

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

            # TODO: try to overlap loading new requests with processing of previous ones
            # Move batch of requests to GPUs
            req1.move_to(dev_map, non_blocking=True)
            req2.move_to(dev_spare_mem, non_blocking=True)
            torch.cuda.synchronize()

            req1.hidden_states = req1.next_token_ids
            req2.hidden_states = req2.next_token_ids
            # stat = ExecutionStatistics(self.num_stages)

            # Start by running processing on stage 0 and passing the hidden-state to stage 1
            # while loading the req2 to stage 0
            self.swapping_decode_on_stage(req1, req2, 0, finish=False).wait()

            # TODO: Improve the code to support multiple stages not just two
            for _ in range(self.max_length):
                x = self.swapping_decode_on_stage(req1, req2, 1, finish=True)
                y = self.swapping_decode_on_stage(req2, req1, 0, finish=False)
                x.wait()
                y.wait()
                torch.cuda.synchronize()

                # Complete a swapping cycle
                # TODO: on the last iteration we should not do this
                x = self.swapping_decode_on_stage(req1, req2, 0, finish=False) 
                y = self.swapping_decode_on_stage(req2, req1, 1, finish=True)
                x.wait()
                y.wait()
                torch.cuda.synchronize()

            self.print_output(req1)
            self.print_output(req2)

            # stat.report()

            # free memory of requests
            print('Req', req1.id, 'size:', req1.bytes())
            print('Req', req2.id, 'size:', req2.bytes())

            req1.free()
            req2.free()
            # torch.cuda.empty_cache()

        self._unregister_from_layer_completion()
