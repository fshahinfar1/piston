from typing import *
import time
import torch
from queue import Queue

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )

from piston.core.pipeline.simple_pipeline import SimplePipeline
from piston.core.entity.request import Request
from piston.core.entities import SubModel
from piston.utils.worker import Worker, Promise
# from piston.utils.circular_range import CircularRange


class RangeTracker:
    def __init__(self, size):
        self._state = [False] * size
        self._size = size
    
    def __contains__(self, index) -> bool:
        return self.__getitem__(index)
    
    def __getitem__(self, index) -> bool:
        if index >= self._size:
            return False
        return self._state[index]
    
    def __setitem__(self, index, value) -> None: 
        assert isinstance(value, bool)
        if index >= self._size:
            raise RuntimeError('index out of range')
        self._state[index] = value
    
    def set_range(self, start, end, val):
        assert isinstance(val, bool)
        if (start < 0 or end < 0
            or start >= self._size
            or end > self._size
            or end <= start):
            raise RuntimeError('wrong range indicator')
        
        for index in range(start, end):
            self._state[index] = val


def free_mem(dev):
    r = torch.cuda.memory_reserved(dev)
    a = torch.cuda.memory_allocated(dev)
    f = r-a  # free inside reserved
    return f


class OverCommitedSingleStagePipeline(SimplePipeline):
    def __init__(self, spare_memory_device, model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int = 64, do_print=False):
        assert num_stages == 1, 'The implementation assumes single stages'

        super().__init__(model_name, num_stages, device_list, max_length,
                            batch_size, do_print=False)

        # Device on which we hold temporary data
        self.spare_memory_device = spare_memory_device

        # For moving kv caches
        self._in_flight_copies: Queue[Tuple[Any, int]] = Queue()

        self._active_request: Optional[torch.Tensor] = None

        # How much memory do we have for KV-cache (only use 90% of available memory and leave the rest for hidden state)
        # self.available_memory = free_mem(self.replica.stages[0]) * 0.9
        self.available_memory = 7 * 1024 * 1024 * 1024
        print('Single Stage Swapping Pipeline: Available memory:', self.available_memory)

        self._count_layers = 0
        self._load_layers = RangeTracker(32)

        self._report_movement_measurements = False
        self._data_movement_measurements = []

        # skipping moving if we have this many layers ahead
        self._layers_a_head = 4

        self._count_stream = self._layers_a_head # A100 has 2 copy engines
        self._next_stream = 0
        self._streams = tuple(torch.cuda.Stream() for i in range(self._count_stream))

        self._is_last_iter = False

    def close(self) -> None:
        pass

    def _move_kv_cache_layer(self, layer_index):
        """
        We have just finished processing a layer. Let's decide if we need to
        move layers in/out of the device and start the process in the background.
        """
        prefetch_layer_index = (layer_index + self._layers_a_head) % self._count_layers
        if prefetch_layer_index in self._load_layers:
            # print('skip: @', layer_index, f'({prefetch_layer_index})')
            return

        # if self._is_last_iter and prefetch_layer_index < layer_index:
        #     # This is the last iteration, we do not need to move more layers in or out
        #     return

        # select a stream
        stream = self._streams[self._next_stream]
        self._next_stream = (self._next_stream + 1) % self._count_stream

        req = self._active_request
        assert req is not None

        num_layers = len(req.cache.layers)

        start = time.time()
        offload_layer_index = layer_index
        assert offload_layer_index < num_layers
        assert prefetch_layer_index < num_layers
        with torch.cuda.stream(stream):
            # move this layer of KV cache to the spare device. First update the
            # ring point, so that we no this layer is not available
            # self._load_layers.begin = (self._load_layers.begin + 1) % self._load_layers.size
            self._load_layers[offload_layer_index] = False
            req.move_single_layer_to(offload_layer_index, self.spare_memory_device, non_blocking=True)

            # prefetch the layer of KV cache that we need in near future. We
            # must update the end pointer of the ring after the copy is
            # finished. Otherwise we might work on garbage data
            req.move_single_layer_to(prefetch_layer_index, self.replica.stages[0].device, non_blocking=True)
            # self._load_layers.end = (prefetch_layer_index + 1) % num_layers

        if self._report_movement_measurements:
            stream.synchronize()
            dur = time.time() - start
            S = lambda x: x.numel() * x.element_size()
            K = lambda l: S(l.keys) + S(l.values)
            sz = K(req.cache.layers[offload_layer_index])
            self._data_movement_measurements.append((dur, offload_layer_index, prefetch_layer_index, sz))

        # print('in/out:', self._load_layers)

        t = stream.record_event() # synchronize on current event
        self._in_flight_copies.put((t, prefetch_layer_index))

    def _initial_load_kv_cache(self, req: Request) -> None:
        keys = req.cache.layers[0].keys
        vals = req.cache.layers[0].values

        # [batch_size, num_heads, seq_len, head_dim]

        batch_sz, num_head, seq_len, head_dim = keys.shape
        assert keys.shape == vals.shape, f'{keys.shape} != {vals.shape}'

        # Statically reserve memory for KV cache
        # x2 bytes for float16
        # x2 for key and value
        layer_size =  batch_sz * num_head * head_dim * 2 * (self.max_length + seq_len) * 2 # bytes

        num_layers = len(req.cache.layers)

        # each layer has the same size
        load_layers = int(self.available_memory // layer_size)
        if load_layers > num_layers:
            # we are not going to load more than what is available
            load_layers = num_layers
        assert load_layers > 1, 'can not load even one layer!!!'
        print('Number of layers on device memory:', load_layers, '/', num_layers)

        # Move what is possible to main device and others to the spare memory
        dev_map = [None] * num_layers
        for i in range(load_layers):
            dev_map[i] = self.replica.stages[0].device
        for i in range(load_layers, num_layers):
            dev_map[i] = self.spare_memory_device
        req.move_to(dev_map, non_blocking=True)

        self._count_layers = num_layers
        self._load_layers = RangeTracker(num_layers)
        self._load_layers.set_range(0, load_layers, True)
    
    def _forward(self, stage: SubModel, req: Request) -> None:
        """
        This code is repeated from SubModel.forward because I want to have
        control over which layers should be executed.
        I must block if the KV-cache of a layer has not been loaded yet
        """
        input_ids = req.hidden_states
        attention_mask = req.attention_mask
        use_cache = True # always

        inputs_embeds = input_ids
        if stage.embed_tokens is not None:
            # First stage
            inputs_embeds = stage.embed_tokens(input_ids)

        if req.cache is None:
            raise Exception('The request does not have a cache!')
        past_key_values = req.cache

        cache_position = req.cache_position
        if req.cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=stage.device
            )

        position_ids = req.position_ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = req.causal_mask
        if causal_mask is None:
            mask_function = create_causal_mask if stage.config.sliding_window is None else create_sliding_window_causal_mask
            causal_mask = mask_function(
                config=stage.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        position_embeddings = req.position_embeddings
        if position_embeddings is None:
            position_embeddings = stage.rotary_emb(hidden_states, position_ids)

        for k, decoder_layer in enumerate(stage.layers):
            # print('-- layer:', k)
            # we should block until the KV cache of the layer is loaded
            while k not in self._load_layers:
                if not self._in_flight_copies.empty():
                    event, layer_index = self._in_flight_copies.get()
                    if not event.query():
                        event.synchronize()
                    self._load_layers[layer_index] = True
                else:
                    print('error: no in flight copies but the layers are not found')
                    torch.cuda.synchronize(stage.device)
                    time.sleep(1)
                    print('warning: waiting for', k, self._load_layers)
            
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            # Load next layer of kv-cache 
            self._move_kv_cache_layer(stage.first_layer_index + k)

        # Last stage
        if stage.norm is not None:
            hidden_states = stage.norm(hidden_states)

        # hidden request state
        req.position_ids = position_ids
        req.cache_position = cache_position
        req.causal_mask = causal_mask
        # req.position_embeddings = position_embeddings
        req.hidden_states = hidden_states
    
    def _do_one_iteration(self, req: Request, stats: Optional[Any]) -> torch.Tensor:
        with torch.no_grad():
            req.hidden_states = req.next_token_ids
            cache = req.cache

            start = time.time()
            # there is only one stage, so ...
            self._forward(self.replica.stages[0], req)
            if stats:
                # wait until all computation on this device is over
                torch.cuda.synchronize(self.replica.stages[0].device)
                duration = (time.time() - start) * 1000
                stats.stage_exec_times[0].append(duration)

        # Select the most probable token as next toekn
        logits = self.replica.lm_head(req.hidden_states)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Update attention to use the new token
        mask = req.attention_mask
        mask = torch.cat([mask, mask.new_ones((mask.size(0), 1))], dim=-1)
        req.attention_mask = mask 

        req.clear_hidden_states()

        return next_token

    def process_requests(self)-> None:
        assert len(self.rx_queue) == 0
        if not self.run_queue:
            return

        # How to split KV cache between stages
        count = len(self.run_queue[0].cache.layers)
        r = count // self.num_stages
        dev_map = [self.replica.stages[i // r].device for i in range(count)]

        dev_spare_mem = [self.spare_memory_device] * count

        while self.run_queue:
            req = self.run_queue.pop()

            self._active_request = req

            # Load as much as KV cache as we can
            self._initial_load_kv_cache(req)
            # Make sure initial token is on the device
            req.next_token_ids = req.next_token_ids.to(self.replica.stages[0].device, non_blocking=True)
            torch.cuda.synchronize(self.replica.stages[0]) 

            # stat = ExecutionStatistics(self.num_stages)
            stat = None
            for i in range(self.max_length):
                self._is_last_iter =  i == self.max_length - 1
                x = self._do_one_iteration(req, stat)
                req.generated.append(x)
                req.next_token_ids = x

            self.print_output(req)
            # stat.report()

            # make sure do not have anything in flight
            # assert self._in_flight_copies.empty()
            while not self._in_flight_copies.empty():
                self._in_flight_copies.get()

            # free memory of requests
            print('Req', req.id, 'size:', req.bytes())

            if self._report_movement_measurements:
                for m in self._data_movement_measurements:
                    print(round(m[0] * 1000, 2), 'ms', '    ', m[-1], 'B')
                print('-----')

                self._data_movement_measurements.clear()
            req.free()