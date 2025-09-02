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

    def forward(self, input_ids, attention_mask=None, use_cache=None, past_key_values=None):
        position_ids = None
        cache_position = None

        # First stage
        inputs_embeds = input_ids
        if self.embed_tokens is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=self.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
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
                self._notify('layer_finish', self.first_layer_index + k)

            # Last stage
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)

        out = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        return out


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
        hidden_state = req.next_token_ids

        with torch.no_grad():
            for stage in self.stages:
                cache = req.cache

                start = time.time()
                # bring the input/hidden state to device
                hidden_state = hidden_state.to(stage.device, non_blocking=True)

                # TODO: can we do something here to use the time?

                # wait for data transfer to finish
                torch.cuda.synchronize(stage.device)
                duration = (time.time() - start) * 1000
                if stats:
                    stats.hidden_state_transfer_times[stage.stage_index].append(duration)
                    sz = hidden_state.numel() * hidden_state.element_size()
                    stats.hidden_state_transfer_size[stage.stage_index].append(sz)

                start = time.time()
                out = stage.forward(hidden_state, attention_mask=req.attention_mask,
                                        use_cache=True, past_key_values=cache)
                # wait until all computation on this device is over
                torch.cuda.synchronize(stage.device)
                duration = (time.time() - start) * 1000
                if stats:
                    stats.stage_exec_times[stage.stage_index].append(duration)

                hidden_state = out.last_hidden_state

        logits = self.lm_head(out.last_hidden_state)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        return next_token


class Request:
    counter = 1

    def __init__(self, prompt: str):
        self.id = Request.counter
        Request.counter += 1

        # prompt string
        self.prompt = prompt

        # List of tokens
        # NOTICE: not all tokens are on the same device!
        self.generated: List[torch.Tensor] = []

        # next token
        self.next_token_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None

        # KV Cache of each stage
        self.cache = DynamicCache()

        self._pre_move_state: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
    
    def move_to(self, device_map, non_blocking=False) -> None:
        """
        Move KV cache of request to the given devices device map indicates for
        each layer of KV-Cache which device it should go to.
        """

        cache = self.cache
        num_layers = len(cache.layers)
        for i in range(num_layers):
            dev = device_map[i]
            if dev is None or cache.layers[i].keys.device == dev:
                # do not move this layer
                continue
            cache.layers[i].keys = cache.layers[i].keys.to(dev, non_blocking=non_blocking)
            cache.layers[i].values = cache.layers[i].values.to(dev, non_blocking=non_blocking)
    
    def pre_move_to(self, pre_move_key, device_map, non_blocking=False) -> None:
        """
        Similar as move_to, but does not update the KV-Cache poitners until
        pre_move_apply is called.
        
        Attention: if using non-blocking mode, make sure to call pre_move_apply
        after pre_move_to is finished.
        """

        if pre_move_key in self._pre_move_state:
            raise Exception('Overwirting a previous pre-move state', pre_move_key)

        state: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        self._pre_move_state[pre_move_key] = state
        
        for i, layer in enumerate(self.cache.layers):
            target_dev = device_map[i]
            if target_dev is None or layer.keys.device == target_dev:
                # do not move this layer
                continue
            else:
                new_key = layer.keys.to(target_dev, non_blocking=non_blocking)
                new_value = layer.values.to(target_dev, non_blocking=non_blocking)
                state.append((i, new_key, new_value))
    
    def apply_pre_move(self, pre_move_key):
        state = self._pre_move_state.get(pre_move_key)
        if state is None:
            raise Exception('The premove state is empty. Probably calling apply before pre-moving!')
        
        for index, new_keys, new_values in state:
            self.cache.layers[index].keys = new_keys
            self.cache.layers[index].values = new_values

        del self._pre_move_state[pre_move_key]
    
    def cache_size_bytes(self) -> int:
        dc = self.cache
        nbytes = 0
        print('--->', 'num layers:', len(dc.layers), 'num floats:',
        dc.layers[0].keys.numel(), 'shape:', dc.layers[0].keys.shape)
        for layer in dc.layers:
            nbytes += layer.keys.numel()   * layer.keys.element_size()
            nbytes += layer.values.numel() * layer.values.element_size()
        return nbytes
    
    def bytes(self) -> int:
        b = self.cache_size_bytes()
        for t in self.generated:
            b += t.numel() * t.element_size()
        return b 
    
    def free(self):
        # TODO: Do I need to do this? why?
        self.cache = None
        self.generated.clear()
        self.next_token_ids = None
