from typing import *
import time
import torch
from queue import deque

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )

from piston.core.pipeline.simple_pipeline import SimplePipeline
from piston.core.entity.request import Request
from piston.core.entity.replica import SubModel

# from piston.utils.worker import Worker, Promise
# from piston.utils.circular_range import CircularRange
from piston.utils.range_tracker import RangeTracker


def free_mem(dev):
    r = torch.cuda.memory_reserved(dev)
    a = torch.cuda.memory_allocated(dev)
    f = r-a  # free inside reserved
    return f


class Waitable:
    def __init__(self):
        self.to_sync = []
        self.callbacks: List[Callable] = []
    
    def synchronize(self):
        for e in self.to_sync:
            e.synchronize()
        for fn in self.callbacks:
            fn()


def create_slice(large_tensor: torch.Tensor, shape) -> torch.Tensor :
    """
    Create a slice of tensor with the given shape
    """
    t = tuple(slice(0, d) for d in shape)
    return large_tensor[t]


class PartialKVCache:
    OFFLOAD = 0
    PREFETCH = 1

    REAL = 0
    PHANTOM = 1
    INVALID = -1

    def __init__(self, cache, main_dev, spare_memory, num_layers, load_layers):
        """
        @main_dev: main device where we use KV cache 
        @spare_memory: device on which we temporarily keep KV cache layers
        @num_layers: number of layers of the model
        @shape: shape of a KV_cache layer entry
        @dtype:
        """
        assert load_layers > 0
        assert load_layers < num_layers

        self.cache_refs = cache
        self.main_dev = main_dev
        self.spare_mem_dev = spare_memory

        dtype = cache.layers[0].keys.dtype
        cur_shape = list(cache.layers[0].keys.shape)
        shape = list(cur_shape)
        shape[2] = 512 # TODO: change this from hard coded

        extra_buffers = 1

        self.real = [ [torch.empty(*shape, dtype=dtype, device=self.main_dev), 
                        torch.empty(*shape, dtype=dtype, device=self.main_dev)]
                    for i in range(load_layers + extra_buffers)]
        self.real_used = RangeTracker(len(self.real)) 

        phantom_layers = max(0, num_layers - load_layers)
        self.phantom = [ [torch.empty(*shape, dtype=dtype, device=self.spare_mem_dev), 
                            torch.empty(*shape, dtype=dtype, device=self.spare_mem_dev)]
                        for i in range(phantom_layers + extra_buffers)]
        self.phantom_used = RangeTracker(len(self.phantom))

        self.layer_at: List[Tuple[int, int]] = [(PartialKVCache.INVALID, -1), ] * num_layers

        for i in range(load_layers):
            keys = create_slice(self.real[i][0], cur_shape)
            values = create_slice(self.real[i][1], cur_shape)
            keys.copy_(cache.layers[i].keys, non_blocking=True)
            values.copy_(cache.layers[i].values, non_blocking=True)
            cache.layers[i].keys = keys
            cache.layers[i].values = values
            self.layer_at[i] = (PartialKVCache.REAL, i)
        self.real_used.set_range(0, load_layers, True)
        assert self.real_used._state.index(False) == load_layers

        for i in range(load_layers, num_layers):
            ph_index = i - load_layers
            keys = create_slice(self.phantom[ph_index][0], cur_shape)
            values = create_slice(self.phantom[ph_index][1], cur_shape)
            keys.copy_(cache.layers[i].keys, non_blocking=True)
            values.copy_(cache.layers[i].values, non_blocking=True)
            cache.layers[i].keys = keys
            cache.layers[i].values = values
            self.layer_at[i] = (PartialKVCache.PHANTOM, ph_index)
        self.phantom_used.set_range(0, phantom_layers, True)
        assert self.phantom_used._state.index(False) == phantom_layers

        self._count_layers = num_layers
        self._load_layers = RangeTracker(num_layers)
        self._load_layers.set_range(0, load_layers, True)
        self._streams = [
            torch.cuda.Stream(self.main_dev, priority=0),
            torch.cuda.Stream(self.main_dev, priority=0),
            torch.cuda.Stream(self.main_dev, priority=0),
            torch.cuda.Stream(self.main_dev, priority=0),
        ]
    
    def __contains__(self, i) -> bool:
        """
        Return true if the layer ``i'' is on the main device otherwise false
        """
        return i in self._load_layers
     
    def _move(self, layer_index: int, dir: int) -> Waitable:
        if dir == PartialKVCache.OFFLOAD:
            s1 = self._streams[0]
            s2 = self._streams[1]

            src_index = self.layer_at[layer_index]
            assert src_index[0] == PartialKVCache.REAL
            assert src_index[1] in self.real_used

            free_buf_index = self.phantom_used._state.index(False)
            assert free_buf_index not in [x[1] for x in self.layer_at if x[0] == PartialKVCache.PHANTOM], 'some one is using this?'

            target_index = (PartialKVCache.PHANTOM, free_buf_index)
            self.phantom_used[free_buf_index] = True

            dst_buf_list = self.phantom
            src_buf_track = self.real_used

            target_dev = self.spare_mem_dev
        else:
            s1 = self._streams[2]
            s2 = self._streams[3]

            src_index = self.layer_at[layer_index]
            assert src_index[0] == PartialKVCache.PHANTOM
            assert src_index[1] in self.phantom_used

            free_buf_index = self.real_used._state.index(False)
            assert free_buf_index not in [x[1] for x in self.layer_at if x[0] == PartialKVCache.REAL], 'some one is using this?'

            target_index = (PartialKVCache.REAL, free_buf_index)
            self.real_used[free_buf_index] = True

            dst_buf_list = self.real
            src_buf_track = self.phantom_used

            target_dev = self.main_dev

        src_keys = self.cache_refs.layers[layer_index].keys
        src_values = self.cache_refs.layers[layer_index].values
        assert src_keys.shape == src_values.shape


        # find a pair of buffer to copy data to
        dst_keys, dst_values = dst_buf_list[free_buf_index]
        assert dst_keys.device == target_dev

        shape = src_keys.shape
        dst_keys = create_slice(dst_keys, shape)
        dst_values = create_slice(dst_values, shape)

        with s1:
            dst_keys = dst_keys.copy_(src_keys, non_blocking=True)
        with s2:
            dst_values = dst_values.copy_(src_values, non_blocking=True)

        self.cache_refs.layers[layer_index].keys = dst_keys
        self.cache_refs.layers[layer_index].values = dst_values

        t1 = s1.record_event()
        t2 = s2.record_event()

        # torch.cuda.synchronize()
        # src_buf_track[src_index[1]] = False
        self.layer_at[layer_index] = (PartialKVCache.INVALID, -1)
        # self.layer_at[layer_index] = target_index

        # when copy is finished
        def new_func():
            # print(layer_index, src_index, '-->', target_index, '|', shape)
            # a = src_keys.cpu()
            # b = dst_keys.cpu()
            # torch.cuda.synchronize()
            # assert torch.equal(a, b)
            src_buf_track[src_index[1]] = False
            self.layer_at[layer_index] = target_index

        w = Waitable()
        w.to_sync.append(t1)
        w.to_sync.append(t2)
        w.callbacks.append(new_func)
        return w
 
    def offload(self, layer_index: int) -> Waitable:
        assert layer_index in self._load_layers
        self._load_layers[layer_index] = False
        w = self._move(layer_index, PartialKVCache.OFFLOAD)
        return w
    
    def prefetch(self, layer_index: int) -> Waitable:
        assert layer_index not in self._load_layers
        w = self._move(layer_index, PartialKVCache.PREFETCH)

        # when copy is finished
        def new_func():
            # print(layer_index, 'available')
            self._load_layers[layer_index] = True
        w.callbacks.append(new_func)
        return w


class OverCommitedSingleStagePipeline(SimplePipeline):
    def __init__(self, spare_memory_device, model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int = 64, do_print=False):
        assert num_stages == 1, 'The implementation assumes single stages'
        super().__init__(model_name, num_stages, device_list, max_length,
                            batch_size, do_print=do_print)

        # Device on which we hold temporary data
        self.spare_memory_device = spare_memory_device

        # Used to synchronize with in flight kv caches
        self._in_flight_copies: deque[Tuple[int, Waitable]] = deque()
        self._in_flight_layers: Set[int] = set()

        # How much memory do we have for KV-cache (only use 90% of available
        # memory and leave the rest for hidden state)
        # self.available_memory = free_mem(self.replica.stages[0]) * 0.9
        self.available_memory = 7 * 1024 * 1024 * 1024
        print('Single Stage Swapping Pipeline: Available memory:', self.available_memory)

        # Only keep part of the request's kv-cache on the main device
        self._partial_kv: Optional[PartialKVCache] = None

        self._report_movement_measurements = False
        self._data_movement_measurements = []

        # skipping moving if we have this many layers ahead
        self._layers_a_head = 8
        self._count_layers = 32
        self._is_last_iter = False

        self._compute_stream = torch.cuda.Stream(self.replica.stages[0].device, priority=-1)
    
    def _service_in_flight_queue(self):
        layer_index, w = self._in_flight_copies.popleft()
        w.synchronize()
        self._in_flight_layers.remove(layer_index)

    def _move_kv_cache_layer(self, req: Request, layer_index: int):
        """
        We have just finished processing a layer. Let's decide if we need to
        move layers in/out of the device and start the process in the background.
        """
        if self._partial_kv is None:
            # We don't have a partial KV cache
            return

        # prefetch_layer_index = (layer_index + self._layers_a_head) % self._count_layers
        prefetch_layer_index = (layer_index + self._layers_a_head) % self._count_layers
        if prefetch_layer_index in self._partial_kv:
            # print('skip: @', layer_index, f'({prefetch_layer_index})')
            return

        if self._is_last_iter and prefetch_layer_index < layer_index:
            # This is the last iteration, we do not need to move more layers in or out
            return

        num_layers = len(req.cache.layers)
        offload_layer_index = layer_index
        assert offload_layer_index < num_layers
        assert prefetch_layer_index < num_layers

        while (prefetch_layer_index in self._in_flight_layers or
               offload_layer_index in self._in_flight_layers):
            self._service_in_flight_queue()

        start = time.time()
        p1 = self._partial_kv.prefetch(prefetch_layer_index)
        p2 = self._partial_kv.offload(offload_layer_index)

        if self._report_movement_measurements:
            torch.cuda.synchronize()
            print('sync event')
            dur = time.time() - start
            S = lambda x: x.numel() * x.element_size()
            K = lambda l: S(l.keys) + S(l.values)
            sz = K(req.cache.layers[offload_layer_index])
            self._data_movement_measurements.append((dur, offload_layer_index, prefetch_layer_index, sz))

        self._in_flight_copies.append((prefetch_layer_index, p1))
        self._in_flight_layers.add(prefetch_layer_index)
        self._in_flight_copies.append((offload_layer_index, p2))
        self._in_flight_layers.add(offload_layer_index)

    def _initial_load_kv_cache(self, req: Request) -> None:
        # TODO: instead of statically allocate layers on the GPU, check their
        # size after each token is added and then move the layers that will not
        # fit to the spare memory

        # [batch_size, num_heads, seq_len, head_dim]
        shape = req.cache.layers[0].keys.shape
        batch_sz, num_head, seq_len, head_dim = shape

        # Statically reserve memory for KV cache
        # x2 bytes for float16
        # x2 for key and value
        layer_size =  batch_sz * num_head * head_dim * 2 * (self.max_length + seq_len) * 2 # bytes
        num_layers = len(req.cache.layers)
        # each layer has the same size, how many layers will fit?
        load_layers = min(int(self.available_memory // layer_size), num_layers)
        assert load_layers > 1, 'can not load even one layer!!!'
        print('Number of layers on device memory:', load_layers, '/', num_layers)

        # The partial kv cache object will manaage the requests cache and expose
        # APIs for prefetch/offload tensors
        main_dev = self.replica.stages[0].device
        if load_layers != num_layers:
            self._partial_kv = PartialKVCache(req.cache, main_dev,
                                self.spare_memory_device, num_layers, load_layers)
        else:
            self._partial_kv = None

        self._count_layers = num_layers
        self._layers_a_head = min(8, load_layers)
    
    def _stall_if_needed_for_layers(self, k) -> None:
        if self._partial_kv is None:
            # we don't have a partial kv cache
            return
        
        # while self._in_flight_copies:
        #     w = self._in_flight_copies.popleft()
        #     w.synchronize()

        # print('-- layer:', k)
        # we should block until the KV cache of the layer is loaded
        while k not in self._partial_kv or k in self._in_flight_layers:
            if self._in_flight_copies:
                self._service_in_flight_queue()
            else:
                # This is an error
                print('error: no in flight copies but the layers are not found')
                torch.cuda.synchronize(self.replica.stages[0].device)
                time.sleep(1)
                print('warning: waiting for', k, self._load_layers)
    
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
            self._stall_if_needed_for_layers(k)
            
            with self._compute_stream:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            self._compute_stream.synchronize()

            # Load next layer of kv-cache 
            self._move_kv_cache_layer(req, stage.first_layer_index + k)

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

            # Load as much as KV cache as we can
            self._initial_load_kv_cache(req)
            # Make sure initial token is on the device
            req.next_token_ids = req.next_token_ids.to(self.replica.stages[0].device, non_blocking=True)
            torch.cuda.synchronize() 

            # stat = ExecutionStatistics(self.num_stages)
            stat = None
            for i in range(self.max_length):
                self._is_last_iter =  (i == (self.max_length - 1))
                x = self._do_one_iteration(req, stat)
                req.generated.append(x)
                req.next_token_ids = x

            self.print_output(req)
            # stat.report()

            # make sure do not have anything in flight
            # assert self._in_flight_copies.empty()
            self._in_flight_copies.clear()

            # free memory of requests
            print('Req', req.id, 'size:', req.bytes())

            if self._report_movement_measurements:
                for m in self._data_movement_measurements:
                    print(round(m[0] * 1000, 2), 'ms', '    ', m[-1], 'B')
                print('-----')

                self._data_movement_measurements.clear()
            req.free()