import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import DynamicCache


class SubModel:
    def __init__(self, stage):
        self.stage_index = stage
        self.device = torch.device('cpu')
        self.layers = []

        # Only the first stage will have embed_tokens set
        self.embed_tokens = None

        # Only the last stage will have the norm
        self.norm = None

    def ready(self):
        for layer in self.layers:
            layer.to(self.device)

        if self.embed_tokens:
            self.embed_tokens.to(self.device)
        
        if self.norm:
            self.norm.to(self.device)

    def forward(self, input_ids, use_cache=None, past_key_values=None):
        print(input_ids.device, self.device)
        assert(input_ids.device == self.device)
        attention_mask = None
        output_attentions = None
        cache_position = None

        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            assert isinstance(past_key_values, DynamicCache)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_length,
                    device=self.device)

        position_ids = cache_position.unsqueeze(0)

        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            raise RuntimeError('not implemented')
            # attention_mask = _prepare_4d_causal_attention_mask(
            #     attention_mask,
            #     (batch_size, seq_length),
            #     inputs_embeds,
            #     past_key_values_length,
            #     sliding_window=self.config.sliding_window,
            # )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    )
            hidden_states = layer_outputs[0]

        if self.norm:
            hidden_state = self.norm(hidden_state)

        return (hidden_states, past_key_values)


class Replica:
    def __init__(self, model_name, num_stages, device_list):
        model = AutoModelForCausalLM.from_pretrained(model_name,
                torch_dtype=torch.float16, device_map='cpu')
        model.eval()

        assert num_stages > 0
        assert len(device_list) >= num_stages

        # TODO: is this not on a GPU?
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        layers = model.model.layers
        num_layers = len(layers)
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

            # TODO: what is happening here?
            # not sure about this
            # s._attn_implementation = model.model._attn_implementation
            s._attn_implementation = "flash_attention_2"

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


class Request:
    def __init__(self, prompt):
        # prompt string
        self.prompt = prompt
        # Tensor of generated tokens
        self.generated = torch.tensor([], device='cpu')
        # next token
        self.next_token_ids = None
        # KV Cache of each stage
        self.stage_cache = {}
        


