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
        raise Exception('Not implemented')

    def __str__(self) -> str:
        return f'[{self.begin}, {self.end}) / {self.size}'
    
    def increment(self, other: int):
        self.begin = (self.begin + other) % self.size
        self.end = (self.end + other) % self.size
        return self
