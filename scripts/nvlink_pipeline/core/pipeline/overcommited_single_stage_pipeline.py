from typing import *
import time
import torch

# TODO: FIXME: I have changed the library to expose this
from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask
        )

from core.pipeline.simple_pipeline import SimplePipeline
from core.entity.request import Request
from core.entities import SubModel
from utils.worker import Worker, Promise


class CircularRange:
    def __init__(self, begin:int, end:int, size):
        # end is not inclusive: The range is [begin, end)
        assert begin >= 0 and end >= 0 and begin < end
        self.begin = begin
        self.end = end
        self.dist = end - begin
        self.size = size

    def __getitem__(self, index) -> int:
        if index == 0:
            return self.begin
        if index == 1:
            return self.end
        raise RuntimeError('invalid index')

    def __len__(self) -> int:
        return self.dist

    def __contains__(self, other:int) -> bool:
        if other >= self.size:
            return False

        if self.begin < self.end:
            # straight range
            return other >= self.begin and other < self.end

        # rotated range
        return other < self.end or other >= self.begin

    def __add__(self, other: int):
        self.begin = (self.begin + other) % self.size
        self.end = (self.end + other) % self.size
        return self

    def __str__(self) -> str:
        return f'[{self.begin}, {self.end}) / {self.size}'


class OverCommitedSingleStagePipeline(SimplePipeline):
    def __init__(self, spare_memory_device, model_name: str, num_stages: int, device_list: List,
                    max_length=32, batch_size: int = 64, do_print=False):
        assert num_stages == 1, 'The implementation assumes single stages'

        super().__init__(model_name, num_stages, device_list, max_length,
                            batch_size, do_print=False)

        # Device on which we hold temporary data
        self.spare_memory_device = spare_memory_device

        # For computing forward pass in background
        self._count_worker = 2
        self._next_worker = 0
        self._workers = [Worker() for _ in range(self._count_worker)]

        # For moving kv caches
        self._kv_cache_workers = [Worker() for _ in range(num_stages)]
        self._in_flight_copies: Dict[int, List[Promise]] = {i: [] for i in range(num_stages)}

        self._active_request: List[Optional[torch.Tensor]] = [None,] * len(self.replica.stages)

        self.dev = self.replica.stages[0].device
        # How much memory do we have for KV-cache (only use 90% of available memory and leave the rest for hidden state)
        self.available_memory = torch.cuda.memory_allocated(self.dev) * 0.9
        self.available_memory = 128 * 1024 * 1024
        print('Single Stage Swapping Pipeline: Available memory:', self.available_memory)

        self._count_layers = 0
        self._load_layers = CircularRange(0,1,32)

    def __del__(self):
        self.close()

    def close(self) -> None:
        """
        Close pipeline and release resources specially worker threads.
        NOTE: without terminating the worker threads the python program does not close properly
        """
        for w in self._workers + self._kv_cache_workers:
            w.die()

    def _do_move_kv_cache_layer(self, stage_index, layer_index):
        """
        offload one layer of the request and prefetch the layer of the second.

        NOTE: This function should run in a kv-cache worker thread. So the blocking
        move should be fine.
        """
        req = self._active_request[stage_index]
        assert req is not None
        stage = self.replica.stages[stage_index]

        assert layer_index == self._load_layers[0]

        with torch.Stream(device=stage.device) as s_cuda:
            # move this layer of KV cache to the spare device in parallel
            dev_map = [None] * len(req.cache.layers)
            dev_map[self._load_layers[0]] = self.spare_memory_device
            req.move_to(dev_map, non_blocking=False)
            torch.cuda.synchronize(stage.device)

            # prefetch the layer of KV cache that we need in near future
            dev_map = [None] * len(req.cache.layers)
            dev_map[self._load_layers[1]] = stage.device
            req.move_to(dev_map, non_blocking=False)
            torch.cuda.synchronize(stage.device)

        # print('in/out:', self._load_layers)
        self._load_layers + 1

    def _move_kv_cache_layer(self, stage_index, layer_index):
        """
        This is called when forward pass finishes processing a layer.
        Start moving that layers KV-cache to spare memory
        """
        if len(self._load_layers) == self._count_layers:
            # All the requests KV cache is present in the memory, nothing to worry about
            return

        w = self._kv_cache_workers[stage_index]
        p = w.add_task(self._do_move_kv_cache_layer, stage_index, layer_index)
        self._in_flight_copies[stage_index].append(p)

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

        dev_map = [None] * num_layers
        for i in range(load_layers):
            dev_map[i] = self.dev
        req.move_to(dev_map, non_blocking=False)

        self._count_layers = num_layers
        self._load_layers = CircularRange(0, load_layers, num_layers)
        print('Number of layers on device memory:', load_layers, '/', num_layers)
    
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
                while self._in_flight_copies[0]:  
                    self._in_flight_copies[0].pop().wait()
                torch.cuda.synchronize(stage.device)
                # print('waiting:', k, self._load_layers)
            
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
            self._move_kv_cache_layer(0, stage.first_layer_index + k)

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
                torch.cuda.synchronize(self.dev)
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

            self._active_request[0] = req

            # Load as much as KV cache as we can
            self._initial_load_kv_cache(req)
            # Make sure initial token is on the device
            req.next_token_ids = req.next_token_ids.to(self.dev, non_blocking=False)
            torch.cuda.synchronize(self.dev)

            # stat = ExecutionStatistics(self.num_stages)
            stat = None
            for _ in range(self.max_length):
                x = self._do_one_iteration(req, stat)
                req.generated.append(x)
                req.next_token_ids = x

            self.print_output(req)
            # stat.report()

            # TODO: we are doing some extra and unnecessary data copy :|
            while self._in_flight_copies[0]:
                self._in_flight_copies[0].pop().wait()
            torch.cuda.synchronize()

            # free memory of requests
            print('Req', req.id, 'size:', req.bytes())
            req.free()
            # torch.cuda.empty_cache()