from typing import *
import collections
import torch

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )

from piston.utils.observer import Observable

SubModelOutput = collections.namedtuple('SubModelOutput', ['position_ids',
                            'cache_position', 'causal_mask', 'hidden_states'])

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
        self.last_layer_index = 0

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
    
    def _do_forward(self, inputs_embeds, past_key_values, attention_mask,
                    use_cache, cache_position, position_ids, causal_mask,
                    position_embeddings):
        if self.embed_tokens is not None:
            # First stage
            inputs_embeds = self.embed_tokens(inputs_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=self.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

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

        out = SubModelOutput(position_ids, cache_position, causal_mask, hidden_states)
        # req.position_embeddings = position_embeddings
        return out

    def forward(self, req):
        if req.cache is None:
            raise Exception('The request does not have a cache!')

        out = self._do_forward(req.hidden_states, req.cache, req.attention_mask,
                        True, req.cache_position, req.position_ids, req.causal_mask,
                        req.position_embeddings)

        # hidden request state
        req.position_ids = out.position_ids
        req.cache_position = out.cache_position
        req.causal_mask = out.causal_mask
        # req.position_embeddings = out.position_embeddings
        req.hidden_states = out.hidden_states

        return out
    
    def set_request(self, req):
        return
