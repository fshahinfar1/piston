from typing import *
import time
import torch
import torch.multiprocessing as multiprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer

from piston.constants import LOCAL_FILE_ONLY, PIPE_SIZE
from piston.core.entity.request import Request
from piston.core.entity.mproc.mproc_submodel import MPROC_SubModel, MPROC_SubModelInput
from piston.core.entity.mproc.command_codes import *
from piston.utils.pipe import SharedMemoryPipe


class MPROC_Replica:
    def __init__(self, model_name, num_stages, device_list):
        """
        This class abstracts a full pipeline. A model splitted over multiple
        stages.
        """
        assert num_stages > 0
        assert len(device_list) >= num_stages

        model = AutoModelForCausalLM.from_pretrained(model_name,
                device_map='cpu', torch_dtype=torch.float16,
                local_files_only=LOCAL_FILE_ONLY)
        model.eval()
        print('Loaded. Model size:', model.get_memory_footprint())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            local_files_only=LOCAL_FILE_ONLY)

        num_layers = model.model.config.num_hidden_layers
        layers = model.model.layers[:num_layers]
        #print('Number of', num_layers)
        stage_num_layers = num_layers // num_stages

        self.config = model.model.config

        self.active_request = None

        # We communicate with the first stage using this pipe
        self.pipe_other_end, self.pipe = SharedMemoryPipe(PIPE_SIZE, False)

        self.ctrl_pipe = []
    
        # prepare stages
        self.stages = []
        next_pipe = self.pipe_other_end
        for i in range(num_stages):
            ctrl_recv, ctrl_send = SharedMemoryPipe(PIPE_SIZE, True)
            self.ctrl_pipe.append(ctrl_send)

            stage = MPROC_SubModel(i, next_pipe, ctrl_recv)
            self.stages.append(stage)
            next_pipe = stage.output_pipe_end

        for s_index, s in enumerate(self.stages):
            # assign device
            s.device = device_list[s_index]
            # assign layers
            prev = s_index * stage_num_layers
            next = prev + stage_num_layers
            s.layers = layers[prev:next]
            s.first_layer_index = prev
            s.last_layer_index = next

            s.config = self.config
            s.rotary_emb = model.model.rotary_emb

        # the first stage will apply the embed tokens
        self.stages[0].embed_tokens = model.model.embed_tokens

        # last stage will apply the nrom
        self.stages[-1].norm = model.model.norm
        self.stages[-1].is_last_stage = True

        self.last_stage_pipe = self.stages[-1].output_pipe_end

        # We should also apply the lm_head after the last stage
        # the code implementing the pipeline will do that
        self.lm_head = model.lm_head

        for s in self.stages:
            # Launch submodel processes and move their model weights to their
            # device
            s.ready()

        # move the lm_head to the last stage's device
        self.lm_head = self.lm_head.to(self.stages[-1].device)

    def terminate(self):
        self.pipe.close()
        for pipe in self.ctrl_pipe:
            pipe.send((COMMAND_TERMINATE, None))
            pipe.close()
    
    def extract_kv_cache(self, req) -> None:
        """
        get KV cache layers of current active request in the pipeline
        and wirtes it on the given request object.

        NOTE: this function call is blocking
        """
        req.cache.layers.clear() # the cache layers is probably already empty
        for i, stage in enumerate(self.stages):
            pipe = self.ctrl_pipe[i]
            pipe.send((COMMAND_EXTRACT_KV_CACHE, None))
            kind, internal_cache = pipe.recv()
            assert kind == COMMAND_EXTRACT_KV_CACHE_ACK
            for l in range(stage.first_layer_index, stage.last_layer_index):
                lyr = internal_cache.layers[l]
                req.cache.layers.append(lyr) 
        # print('After extracing kv:', len(req.cache.layers))

    def set_active_request(self, req) -> None:
        cmd = (COMMAND_NEW_REQ, req)
        self.pipe.send(cmd)
        # block until the last stage acknowledges configuring the kv-cache
        cmd = self.last_stage_pipe.recv()
        assert cmd[0] == COMMAND_NEW_REQ_ACK
        torch.cuda.synchronize() # make sure cuda copy operations are finished

    def do_one_iteration(self, req: Request) -> torch.Tensor:
        """
        Do one full iteration on the request through all the stages of the pipeline.
        This will return the next token.
        The request must have gone through tokenization phase and have the next_token_ids set to the input for the first stage.
        """
        payload = MPROC_SubModelInput(
            req.next_token_ids,
            req.attention_mask,
            req.cache_position,
            req.position_ids,
            req.causal_mask,
            req.position_embeddings,
            )
        cmd = (COMMAND_DO_FORWARD, payload)
        self.pipe.send(cmd)

        # block until the last stage returns the new token
        kind, submodel_inp = self.last_stage_pipe.recv()
        assert kind == COMMAND_DO_FORWARD_ACK
        hidden_states = submodel_inp[0]

        # Select the most probable token as next toekn
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Update attention to use the new token
        mask = req.attention_mask
        mask = torch.cat([mask, mask.new_ones((mask.size(0), 1))], dim=-1)
        req.attention_mask = mask 

        req.clear_hidden_states()

        return next_token
