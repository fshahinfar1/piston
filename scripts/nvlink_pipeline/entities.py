import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )


class SubModel:
    def __init__(self, stage):
        self.stage_index = stage
        self.device = torch.device('cpu')
        self.layers = []

        # Only the first stage will have embed_tokens set
        self.embed_tokens = None

        # Only the last stage will have the norm
        self.norm = None

        # Not sure about this
        self.config = None
        self.rotary_emb = None

    def ready(self):
        for layer in self.layers:
            layer.to(self.device)

        if self.embed_tokens:
            self.embed_tokens.to(self.device)
        
        if self.norm:
            self.norm.to(self.device)

    def forward(self, input_ids, use_cache=None, past_key_values=None):
        position_ids = None
        cache_position = None
        attention_mask = None

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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

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
                torch_dtype=torch.float16, device_map='cpu')
        model.eval()

        assert num_stages > 0
        assert len(device_list) >= num_stages

        # TODO: is this not on a GPU?
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        num_layers = model.model.config.num_hidden_layers
        layers = model.model.layers[:num_layers]
        #print('Number of', num_layers)
        stage_num_layers = num_layers // num_stages

        # prepare stages
        self.stages = [SubModel(i) for i in range(num_stages)]
        for s_index, s in enumerate(self.stages):
            # assign device
            s.device = device_list[s_index]
            # assign layers
            prev = s_index * stage_num_layers
            next = prev + stage_num_layers
            s.layers = layers[prev:next]

            # TODO: I am not sure about these
            s.config = model.model.config
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


class Request:
    def __init__(self, prompt):
        # prompt string
        self.prompt = prompt
        # Tensor of generated tokens
        self.generated = torch.tensor([], dtype=torch.int, device='cpu')
        # next token
        self.next_token_ids = None
        # KV Cache of each stage
        self.stage_cache = {}
        

