from typing import *
import collections
import multiprocessing as mp

import torch
import torch.multiprocessing as multiprocessing
from transformers.generation.utils import DynamicCache
from transformers.cache_utils import DynamicLayer

from piston.core.entity.submodel import SubModel, SubModelOutput
from piston.core.entity.mproc.command_codes import *


MPROC_SubModelInput = collections.namedtuple('MPROC_SubModelInput',
    ['inputs_embeds', 'attention_mask', 'cache_position',
    'position_ids', 'causal_mask', 'position_embeddings'])


class MPROC_SubModel(SubModel):
    """
    Implement each stage of the pipeline as a separate process to fully utilize
    multiple cores and launch kernel on the GPU in parallel. The down side is
    each process must own its own tensor objects which complicates sharing them.
    """
    def __init__(self, stage: int, pipe: multiprocessing.Pipe,
                 ctrl_pipe: multiprocessing.Pipe):
        super().__init__(stage)

        self.is_last_stage = False

        # The version of cache owned by this stage
        self.cache = None 

        # Multiprocessing
        self.proc = multiprocessing.Process(target=self._main)
        self.input_pipe = pipe
        self.output_pipe_end, self.output_pipe = multiprocessing.Pipe(False)
        self.ctrl_pipe = ctrl_pipe

    def ready(self) -> None:
        self.proc.start()
    
    def _die(self):
        del self.layers
        del self.cache
        self.ctrl_pipe.close()
        self.input_pipe.close()
        self.output_pipe.close()
        self.output_pipe_end.close()
    
    def _setup_new_req_cache(self, req) -> None:
        """
        When we start processing a new request, load its relevant part of the
        cache to the memory managed by this process
        """
        # Remove the old cache and create a new one
        del self.cache
        self.cache = DynamicCache()

        # If the request cache does not have layers ready, it means this is the
        # prefill phase
        if len(req.cache.layers) < self.last_layer_index:
            return

        self.cache.layers.clear()
        # Add some empty layers
        for i in range(self.first_layer_index):
            self.cache.layers.append(DynamicLayer())

        # Bring the assigned kv-cache layers to the GPU device dedicated to this stage
        for i in range(self.first_layer_index, self.last_layer_index):
            lyr = DynamicLayer()
            lyr.keys = req.cache.layers[i].keys.to(self.device, non_blocking=True)
            lyr.values = req.cache.layers[i].values.to(self.device, non_blocking=True) 
            self.cache.layers.append(lyr)
            assert len(self.cache.layers) == i + 1, f'{len(self.cache.layers)} != {i+1}'
         
    def _forward(self, cmd: MPROC_SubModelInput) -> None:
        # Move inputs to this stage's device memory
        inputs_embeds = cmd.inputs_embeds
        attention_mask = cmd.attention_mask
        cache_position = cmd.cache_position
        position_ids = cmd.position_ids
        causal_mask = cmd.causal_mask
        position_embeddings = cmd.position_embeddings

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)
        if cache_position is not None:
            cache_position = cache_position.to(self.device, non_blocking=True)
        if position_ids is not None:
            position_ids = position_ids.to(self.device, non_blocking=True)
        if causal_mask is not None:
            causal_mask = causal_mask.to(self.device, non_blocking=True)
        if position_embeddings is not None:
            position_embeddings = position_embeddings.to(self.device, non_blocking=True)

        out: SubModelOutput = self._do_forward(inputs_embeds,
                                            self.cache,
                                            attention_mask,
                                            True,
                                            cache_position,
                                            position_ids,
                                            causal_mask,
                                            position_embeddings)
        
        next_stage_input = MPROC_SubModelInput(out.hidden_states,
                                            attention_mask,
                                            out.cache_position,
                                            out.position_ids, out.causal_mask,
                                            position_embeddings)

        if self.is_last_stage:
            next_stage_cmd = (COMMAND_DO_FORWARD_ACK, next_stage_input)
        else:
            next_stage_cmd = (COMMAND_DO_FORWARD, next_stage_input)

        self.output_pipe.send(next_stage_cmd)
    
    def _handle_command(self, kind, payload):
        if kind == COMMAND_NEW_REQ:
            self._setup_new_req_cache(payload)
            if self.is_last_stage:
                self.output_pipe.send((COMMAND_NEW_REQ_ACK ,None))
            else:
                # pass the information of new request to next stages
                self.output_pipe.send((kind, payload))
        elif kind == COMMAND_DO_FORWARD:
            with torch.no_grad():
                self._forward(payload) 
        elif kind == COMMAND_EXTRACT_KV_CACHE:
            self.ctrl_pipe.send((COMMAND_EXTRACT_KV_CACHE_ACK, self.cache))
        else:
            raise RuntimeError(f'Unexpected command code: {kind}')
    
    def _main(self) -> None:
        """
        Main loop of the process handling the logic of this stage.
        """
        assert self.input_pipe is not None
        # Move the model weights to GPU
        super().ready()

        torch.cuda.synchronize(self.device)
        tmp = torch.cuda.memory_allocated(self.device) 
        print(str(self.device), ':', 'Memory usage:', tmp)

        while True:
            pipes = mp.connection.wait([self.input_pipe, self.ctrl_pipe])
            for pipe in pipes:
                kind, payload = pipe.recv()
                if kind == COMMAND_TERMINATE:
                    self._die()
                    return
                self._handle_command(kind, payload)
