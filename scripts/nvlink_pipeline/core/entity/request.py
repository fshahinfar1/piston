from typing import *
import torch
from transformers.generation.utils import DynamicCache


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

        # ??
        self.hidden_states = None
        self.position_ids = None
        self.cache_position = None
        self.causal_mask = None
        self.position_embeddings = None

        # KV Cache of each stage
        self.cache = DynamicCache()

        self._pre_move_state: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}

    def move_hidden_state_to(self, device, non_blocking=False) -> None:
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device, non_blocking=non_blocking)

        if self.position_ids is not None:
            self.position_ids = self.position_ids.to(device, non_blocking=non_blocking)

        if self.cache_position is not None:
            self.cache_position = self.cache_position.to(device, non_blocking=non_blocking)

        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(device, non_blocking=non_blocking)
        # self.position_embeddings = self.position_embeddings.to(device, non_blocking=non_blocking)

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
        # print('--->', 'num layers:', len(dc.layers), 'num floats:',
        #     dc.layers[0].keys.numel(), 'shape:', dc.layers[0].keys.shape)
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
        self.clear_hidden_states()

    def clear_hidden_states(self):
        self.hidden_states = None
        self.position_ids = None
        self.cache_position = None
        self.causal_mask = None
        self.position_embeddings = None

