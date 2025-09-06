from typing import *
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )

from constants import LOCAL_FILE_ONLY
from utils.observer import Observable


class SubModel(Observable):
    """
    This class abstacts a single stage of the pipeline.
    """
    def __init__(self, stage):
        super().__init__()

        self.stage_index = stage
        self.device = torch.device('cpu')
        self.layers = []
        self.first_layer_index = 0

        # Only the first stage will have embed_tokens set
        self.embed_tokens = None

        # Only the last stage will have the norm
        self.norm = None

        # Not sure about this
        self.config = None
        self.rotary_emb = None

    def ready(self) -> None:
        for layer in self.layers:
            layer.to(self.device)

        if self.embed_tokens:
            self.embed_tokens.to(self.device)

        if self.norm:
            self.norm.to(self.device)

    def forward(self, req):
        input_ids = req.hidden_states
        attention_mask = req.attention_mask
        use_cache=True # always

        inputs_embeds = input_ids
        if self.embed_tokens is not None:
            # First stage
            inputs_embeds = self.embed_tokens(input_ids)

        if req.cache is None:
            raise Exception('The request does not have a cache!')
        past_key_values = req.cache

        cache_position = req.cache_position
        if req.cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=self.device
            )

        position_ids = req.position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = req.causal_mask
        if causal_mask is None:
            mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
            causal_mask = mask_function(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        position_embeddings = req.position_embeddings
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for k, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # if there is code that wants to work layer by later ...
            self._notify('layer_finish', self.first_layer_index + k)

        # Last stage
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        # hidden request state
        req.position_ids = position_ids
        req.cache_position = cache_position
        req.causal_mask = causal_mask
        req.position_embeddings = position_embeddings
        req.hidden_states = hidden_states


class Replica:
    def __init__(self, model_name, num_stages, device_list):
        model = AutoModelForCausalLM.from_pretrained(model_name,
                torch_dtype=torch.float16, device_map='cpu',
                local_files_only=LOCAL_FILE_ONLY)
        model.eval()

        assert num_stages > 0
        assert len(device_list) >= num_stages

        # TODO: is this not on a GPU?
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

            # TODO: I am not sure about these
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
            # Move the submodels to their device
            s.ready()
            # input('continue? ')
        # move the lm_head to the last stage's device
        self.lm_head = self.lm_head.to(self.stages[-1].device)

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
                req.hidden_states = req.hidden_states.to(stage.device, non_blocking=True)

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

