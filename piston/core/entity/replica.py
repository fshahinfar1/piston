from typing import *
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from piston.constants import LOCAL_FILE_ONLY
from piston.core.entity.submodel import SubModel


class Replica:
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

        # prepare stages
        self.stages = [SubModel(i) for i in range(num_stages)]
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

        # We should also apply the lm_head after the last stage
        # the code implementing the pipeline will do that
        self.lm_head = model.lm_head

        for s in self.stages:
            print('loading:', s.stage_index)
            # Move the submodels to their device
            s.ready()
            # input('continue? ')
        # move the lm_head to the last stage's device
        self.lm_head = self.lm_head.to(self.stages[-1].device)

        # Let's see how much memory we are using after moving things to each GPU
        torch.cuda.synchronize()
        for d in device_list:
            tmp = torch.cuda.memory_allocated(d) 
            print(str(d), ':', 'Memory usage:', tmp)

    def get_kv_cache_token_size(self) -> int:
        """
        Approximately how much memory one token of kv cache will consume
        """
        # two: key and value
        # two: two bytes for float 6
        # one: batch size
        # one: sequence size
        # num heads
        # hidden size
        # num tokens
        bytes = (2 * 2 * 1 * 1 * self.config.num_key_value_heads * self.config.hidden_size)
        return bytes

    def get_max_kv_cache_size(self, max_length) -> int:
        """
        Approximately what is the size of a kv cache with max-length number of tokens
        """
        bytes = self.get_kv_cache_token_size() * max_length
        return bytes

    def do_one_iteration(self, req, stats=None) -> torch.Tensor:
        """
        Do one full iteration on the request through all the stages of the pipeline.
        This will return the next token.
        The request must have gone through tokenization phase and have the next_token_ids set to the input for the first stage.
        """

        with torch.no_grad():
            req.hidden_states = req.next_token_ids
            for stage in self.stages:
                cache = req.cache

                # bring the input/hidden state to device
                start = time.time()
                req.move_hidden_state_to(stage.device, non_blocking=False)

                if stats:
                    torch.cuda.synchronize(stage.device)
                    duration = (time.time() - start) * 1000
                    stats.hidden_state_transfer_times[stage.stage_index].append(duration)
                    sz = req.hidden_states.numel() * req.hidden_states.element_size()
                    stats.hidden_state_transfer_size[stage.stage_index].append(sz)

                start = time.time()
                stage.forward(req)
                if stats:
                    # wait until all computation on this device is over
                    torch.cuda.synchronize(stage.device)
                    duration = (time.time() - start) * 1000
                    stats.stage_exec_times[stage.stage_index].append(duration)

        # Select the most probable token as next toekn
        logits = self.lm_head(req.hidden_states)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Update attention to use the new token
        mask = req.attention_mask
        mask = torch.cat([mask, mask.new_ones((mask.size(0), 1))], dim=-1)
        req.attention_mask = mask 

        req.clear_hidden_states()

        return next_token
