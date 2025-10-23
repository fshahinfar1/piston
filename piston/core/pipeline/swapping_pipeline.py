from typing import *
import torch

from piston.constants import MPROC_ENABLED
from piston.core.pipeline.simple_pipeline import SimplePipeline
from piston.core.entity.mproc.mproc_swapping_replica import MPROC_Swapping_Replica


class SwappingPipeline(SimplePipeline):
    def __init__(self, spare_memory_devices: List[torch.device],
                 model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int = 64, do_print=False):
        
        assert MPROC_ENABLED is True
        assert num_stages == 2, 'The implementation assumes there are two stages'

        # List of device on which we hold temporary data
        self.spare_memory_devices = spare_memory_devices

        # NOTE: To use a different replica manager implementation I'm using a
        # level indirection which has made things obscure. We should call the
        # super class's constructor after settitng self.spare_memory_devices
        # because _build_replca version of this class needs it.
        super().__init__(model_name, num_stages, device_list, max_length,
                            batch_size, do_print=do_print)
        
        self.fallback_req_processing = super()._do_process_reqeust

    def _build_replica(self, model_name, num_stages, device_list) -> Any:
        return MPROC_Swapping_Replica(model_name, num_stages, device_list,
                                      self.spare_memory_devices)
    def process_requests(self)-> None:
        assert len(self.rx_queue) == 0
        if not self.run_queue:
            return
        
        swapping_state = False

        while self.run_queue:
            # Check if there is only one request/batch of request left then fall
            # back to simple pipeline
            if len(self.run_queue) == 1:
                if swapping_state:
                    print('disable swapping')
                    self.replica.disable_swapping()
                    swapping_state = False

                req = self.run_queue.pop()
                self.replica.set_active_request_swap_ver(req, None)
                self.fallback_req_processing(req)
                continue

            # Work with two request and swap them in and out
            req1 = self.run_queue.pop()
            req2 = self.run_queue.pop()
            self.replica.set_active_request_swap_ver(req1, req2)

            # Make sure pipeline is in swapping mode
            if not swapping_state:
                # NOTE: for enabling swapping we must have set two active
                # requests for the pipeline
                print('enable swapping')
                self.replica.enable_swapping()
                swapping_state = True

            # NOTE: make sure the underlying Pipe object has enough buffer
            # so that no one is blocked when sending message. Otherwise it
            # hinders parallel execution.
            for _ in range(self.max_length):
                # submit the two request to the pipeline
                self.replica.async_start_iteration(req1)
                self.replica.async_start_iteration(req2)

                next_token = self.replica.await_iteration_result(req1)
                req1.generated.append(next_token)
                req1.next_token_ids = next_token

                next_token = self.replica.await_iteration_result(req2)
                req2.generated.append(next_token)
                req2.next_token_ids = next_token

            self.print_output(req1)
            self.print_output(req2)

            # report stats about req
            # tmp = req1.cache.layers[0].keys.shape
            # print('Req', req1.id, 'size:', req1.cache_size_bytes(), 'number of tokens:', tmp[2], f'(shape: {tmp})')

            # tmp = req2.cache.layers[0].keys.shape
            # print('Req', req2.id, 'size:', req2.cache_size_bytes(), 'number of tokens:', tmp[2], f'(shape: {tmp})')

            # free memory of requests
            req1.free()
            req2.free()
        
        self.replica.disable_swapping()
