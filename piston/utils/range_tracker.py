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
