from typing import *

import torch
import torch.multiprocessing as multiprocessing
from transformers.generation.utils import DynamicCache
from transformers.cache_utils import DynamicLayer

from piston.core.entity.mproc.command_codes import *
from piston.core.entity.mproc.mproc_submodel import MPROC_SubModel, MPROC_SubModelInput


class MPROC_Swapping_SubModel(MPROC_SubModel):
    """
    Implement each stage of the pipeline as a separate process to fully utilize
    multiple cores and utilize GPU devices in parallel. The down side is
    each process must own its own tensor objects which complicates sharing them.

    This variant of stage is aware of KV-Cache swapping logic
    """
    def __init__(self, stage: int, pipe, ctrl_pipe, spare_device):
        super().__init__(stage, pipe, ctrl_pipe)

        self.swap_enabled = False
        self.count_active_request = 0
        self.spare_device = spare_device

        # The cache that we are maintaining on the spare device
        self.other_cache: DynamicCache = None

    def _enable_swapping(self):
        assert self.count_active_request > 1
        self.register('layer_finish', self._swap, None)
        self.swap_enabled = True
    
    def _disable_swapping(self):
        self.unregister('layer_finish', self._swap)
        self.swap_enabled = False
    
    def _die(self):
        del self.other_cache
        super()._die()
    
    def _setup_new_req_cache_swapable(self, req1, req2) -> None:
        """
        When we start processing a new request, load its relevant part of the
        cache to the memory managed by this process
        """
        # Remove the old cache and create a new one
        del self.cache
        self.cache = DynamicCache()
        del self.other_cache
        self.other_cache = DynamicCache()
        self.count_active_request = 0

        if req2 is None:
            # This happens when we do not have enough requests to overlap. So we
            # fallback on baseline pipeline execution
            assert self.swap_enabled is False
            self.count_active_request = 1
            return self._setup_new_req_cache(req1)

        L = len(req1.cache.layers)
        assert len(req2.cache.layers) == L

        # If the request cache does not have layers ready, it means this is the
        # prefill phase
        if L < self.last_layer_index:
            return

        self.cache.layers.clear()
        self.other_cache.layers.clear()
        # Add some empty layers
        for i in range(self.first_layer_index):
            self.cache.layers.append(DynamicLayer())
            self.other_cache.layers.append(DynamicLayer())

        # Bring the assigned kv-cache layers to the GPU device dedicated to this stage
        for i in range(self.first_layer_index, self.last_layer_index):
            lyr = DynamicLayer()
            lyr.keys = req1.cache.layers[i].keys.to(self.device, non_blocking=True)
            lyr.values = req1.cache.layers[i].values.to(self.device, non_blocking=True) 
            self.cache.layers.append(lyr)

            lyr_other = DynamicLayer()
            lyr_other.keys = req2.cache.layers[i].keys.to(self.spare_device, non_blocking=True)
            lyr_other.values = req2.cache.layers[i].values.to(self.spare_device, non_blocking=True)
            self.other_cache.layers.append(lyr_other)

            assert len(self.cache.layers) == i + 1, f'{len(self.cache.layers)} != {i+1}'
            assert len(self.other_cache.layers) == i + 1, f'{len(self.other_cache.layers)} != {i+1}'
        
        self.count_active_request = 2
    
    def _swap(self, _, layer_index):
        with self.copy_stream[0]:
            tmp_key = self.cache.layers[layer_index].keys.to(self.spare_device, non_blocking=True)
            tmp_val = self.cache.layers[layer_index].values.to(self.spare_device, non_blocking=True)

        with self.copy_stream[1]:
            tmp2_key = self.other_cache.layers[layer_index].keys.to(self.device, non_blocking=True)
            tmp2_val = self.other_cache.layers[layer_index].values.to(self.device, non_blocking=True)

        self.cache.layers[layer_index].keys = tmp2_key
        self.cache.layers[layer_index].values = tmp2_val

        self.other_cache.layers[layer_index].keys = tmp_key
        self.other_cache.layers[layer_index].values = tmp_val
    
    def _forward(self, cmd: MPROC_SubModelInput) -> None:
        # Wait for in-flight copy operations to finish before starting
        # processing the next request
        for s in self.copy_stream:
            s.synchronize()
        return super()._forward(cmd)
         
    def _handle_command(self, kind, payload):
        """
        Command formats

        new request: 2 requests objects
        do forward: MPROC_SubModelInput object
        extract kv cache: None
        set swap state: bool
        """
        # TODO: use enums for commands and pattern matching syntax for better
        # quality code
        if kind == COMMAND_NEW_REQ:
            self._setup_new_req_cache_swapable(payload[0], payload[1])
            if self.is_last_stage:
                self.output_pipe.send((COMMAND_NEW_REQ_ACK ,None))
            else:
                # pass the information of new request to next stages
                self.output_pipe.send((kind, payload))
        elif kind == COMMAND_SET_SWAP_STATE:
            assert isinstance(payload, bool)
            if payload:
                self._enable_swapping()
            else:
                self._disable_swapping()
            self.ctrl_pipe.send((COMMAND_SET_SWAP_STATE_ACK, payload))
        else:
            super()._handle_command(kind, payload)

    def _initialize_stage(self):
        super()._initialize_stage()
        # Create a stream for swapping kv-cache

        self.copy_stream = [torch.cuda.Stream(self.device, priority=0) for i in range(2)]
