
import multiprocessing as mp
import pickle
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
import fcntl, os


_HEADER_FORMAT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
MB = 1024 ** 2

def create_os_pipe(bidi=True):
    # NOTE: the maximum buffer size is limited by value at /proc/sys/fs/pipe-max-size 
    buf_size = 1 * MB
    p1, p2 = mp.Pipe(bidi)

    # fd = p1.fileno()
    # fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, buf_size)

    # fd = p2.fileno()
    # fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, buf_size)

    return p1, p2


@dataclass
class _RingState:
    offset: int
    capacity: int
    lock: mp.RLock
    not_empty: mp.Condition
    not_full: mp.Condition
    head: mp.Value
    tail: mp.Value
    count: mp.Value
    writer_closed: mp.Value
    reader_closed: mp.Value


class _RingBuffer:
    def __init__(self, shm, state: _RingState):
        self._shm = shm
        self._state = state

    @property
    def _mv(self):
        start = self._state.offset
        end = start + self._state.capacity
        return self._shm.buf[start:end]

    def close_writer(self):
        with self._state.lock:
            if not self._state.writer_closed.value:
                self._state.writer_closed.value = 1
                self._state.not_empty.notify_all()

    def close_reader(self):
        with self._state.lock:
            if not self._state.reader_closed.value:
                self._state.reader_closed.value = 1
                self._state.not_full.notify_all()

    def has_data(self):
        with self._state.lock:
            return bool(self._state.count.value)

    def poll(self, timeout):
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        end_time = None if timeout is None else time.monotonic() + timeout
        with self._state.lock:
            while not self._state.count.value:
                if self._state.writer_closed.value:
                    return False
                if timeout == 0:
                    return False
                remaining = None if end_time is None else end_time - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._state.not_empty.wait(timeout=remaining)
            return True

    def send_bytes(self, data: bytes, block=True, timeout=None):
        if not block and timeout is not None:
            raise ValueError("cannot set timeout when block is False")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        needed = _HEADER_SIZE + len(data)
        if needed > self._state.capacity:
            raise ValueError("message too large for shared buffer")
        end_time = None if timeout is None else time.monotonic() + timeout
        with self._state.lock:
            if self._state.writer_closed.value:
                raise BrokenPipeError("send on closed connection")
            while self._state.capacity - self._state.count.value < needed:
                if self._state.reader_closed.value:
                    raise BrokenPipeError("peer closed the read end")
                if not block:
                    raise BufferError("ring buffer full")
                remaining = None if end_time is None else end_time - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("send timed out")
                self._state.not_full.wait(timeout=remaining)
            mv = self._mv
            tail = self._state.tail.value
            tail = self._write_header(mv, tail, len(data))
            tail = self._write_bytes(mv, tail, data)
            self._state.tail.value = tail
            self._state.count.value += needed
            self._state.not_empty.notify_all()

    def recv_bytes(self, block=True, timeout=None):
        if not block and timeout is not None:
            raise ValueError("cannot set timeout when block is False")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        end_time = None if timeout is None else time.monotonic() + timeout
        with self._state.lock:
            while not self._state.count.value:
                if self._state.writer_closed.value:
                    raise EOFError
                if not block:
                    raise TimeoutError("no data available")
                remaining = None if end_time is None else end_time - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("recv timed out")
                self._state.not_empty.wait(timeout=remaining)
            mv = self._mv
            head = self._state.head.value
            length, head = self._read_header(mv, head)
            payload, head = self._read_bytes(mv, head, length)
            self._state.head.value = head
            self._state.count.value -= _HEADER_SIZE + length
            self._state.not_full.notify_all()
            return payload

    def _write_header(self, mv, pos, length):
        if pos <= self._state.capacity - _HEADER_SIZE:
            struct.pack_into(_HEADER_FORMAT, mv, pos, length)
            return (pos + _HEADER_SIZE) % self._state.capacity
        header = struct.pack(_HEADER_FORMAT, length)
        return self._write_bytes(mv, pos, header)

    def _write_bytes(self, mv, pos, data):
        cap = self._state.capacity
        end = pos + len(data)
        if end <= cap:
            mv[pos:end] = data
            return end % cap
        first = cap - pos
        mv[pos:pos + first] = data[:first]
        mv[0:len(data) - first] = data[first:]
        return (len(data) - first) % cap

    def _read_header(self, mv, pos):
        if pos <= self._state.capacity - _HEADER_SIZE:
            value = struct.unpack_from(_HEADER_FORMAT, mv, pos)[0]
            return value, (pos + _HEADER_SIZE) % self._state.capacity
        header, pos = self._read_bytes(mv, pos, _HEADER_SIZE)
        return struct.unpack(_HEADER_FORMAT, header)[0], pos

    def _read_bytes(self, mv, pos, size):
        cap = self._state.capacity
        end = pos + size
        if end <= cap:
            return bytes(mv[pos:end]), end % cap
        first = cap - pos
        first_part = bytes(mv[pos:pos + first])
        second_len = size - first
        second_part = bytes(mv[0:second_len])
        return first_part + second_part, second_len % cap


class SharedPipeConnection:
    def __init__(self, shm_name, send_state, recv_state, refcount, refcount_lock, can_send, can_recv):
        self._shm_name = shm_name
        self._shm = shared_memory.SharedMemory(name=shm_name)
        self._send_state = send_state
        self._recv_state = recv_state
        self._send_ring = _RingBuffer(self._shm, send_state) if send_state else None
        self._recv_ring = _RingBuffer(self._shm, recv_state) if recv_state else None
        self._refcount = refcount
        self._refcount_lock = refcount_lock
        self._can_send = can_send
        self._can_recv = can_recv
        self._local_closed = False

    def send(self, obj):
        self._ensure_sendable()
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.send_bytes(data, block=False)

    def send_bytes(self, data, block=True, timeout=None):
        self._ensure_sendable()
        self._send_ring.send_bytes(bytes(data), block=block, timeout=timeout)

    def recv(self):
        payload = self.recv_bytes()
        return pickle.loads(payload)

    def recv_bytes(self, block=True, timeout=None):
        self._ensure_recvable()
        return self._recv_ring.recv_bytes(block=block, timeout=timeout)

    def poll(self, timeout=0.0):
        self._ensure_recvable()
        return self._recv_ring.poll(timeout)

    def close(self):
        if self._local_closed:
            return
        self._local_closed = True
        try:
            if self._send_ring:
                self._send_ring.close_writer()
            if self._recv_ring:
                self._recv_ring.close_reader()
        finally:
            with self._refcount_lock:
                self._refcount.value -= 1
                finalize = self._refcount.value == 0
            self._shm.close()
            if finalize:
                try:
                    shm = shared_memory.SharedMemory(name=self._shm_name)
                    shm.unlink()
                    shm.close()
                except FileNotFoundError:
                    pass

    @property
    def closed(self):
        return self._local_closed

    def fileno(self):
        raise OSError("shared memory connections do not have a file descriptor")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_send_ring'] = None
        state['_recv_ring'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shm = shared_memory.SharedMemory(name=self._shm_name)
        self._send_ring = _RingBuffer(self._shm, self._send_state) if self._send_state else None
        self._recv_ring = _RingBuffer(self._shm, self._recv_state) if self._recv_state else None

    def _ensure_sendable(self):
        if not self._can_send:
            raise OSError("connection is read-only")
        if self._local_closed:
            raise BrokenPipeError("connection is closed")

    def _ensure_recvable(self):
        if not self._can_recv:
            raise OSError("connection is write-only")
        if self._local_closed:
            raise EOFError


def _make_ring_state(ctx, offset, capacity):
    lock = ctx.RLock()
    return _RingState(
        offset=offset,
        capacity=capacity,
        lock=lock,
        not_empty=ctx.Condition(lock),
        not_full=ctx.Condition(lock),
        head=ctx.Value('I', 0),
        tail=ctx.Value('I', 0),
        count=ctx.Value('I', 0),
        writer_closed=ctx.Value('b', 0),
        reader_closed=ctx.Value('b', 0),
    )


def _SharedMemoryPipe(size=1_048_576, duplex=True, ctx=None):
    context = ctx or mp.get_context('spawn')
    if size <= _HEADER_SIZE:
        raise ValueError("size must be greater than header size")
    total = size if not duplex else size * 2
    shm = shared_memory.SharedMemory(create=True, size=total)
    refcount_lock = context.Lock()
    refcount = context.Value('i', 2)
    ring_ab = _make_ring_state(context, 0, size)
    ring_ba = _make_ring_state(context, size, size) if duplex else None
    conn1 = SharedPipeConnection(
        shm.name,
        send_state=ring_ab if duplex else None,
        recv_state=ring_ba if duplex else ring_ab,
        refcount=refcount,
        refcount_lock=refcount_lock,
        can_send=duplex,
        can_recv=True,
    )
    conn2 = SharedPipeConnection(
        shm.name,
        send_state=ring_ba if duplex else ring_ab,
        recv_state=ring_ab if duplex else None,
        refcount=refcount,
        refcount_lock=refcount_lock,
        can_send=True,
        can_recv=duplex,
    )
    shm.close()
    return conn1, conn2


def SharedMemoryPipe(size=1_048_576, duplex=True, ctx=None):
    return create_os_pipe(duplex)