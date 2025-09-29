#! python

import numpy as np
import matplotlib.pyplot as plt


class ParseError(RuntimeError):
    pass


class O:
    def __init__(self):
        self.batch_size = 0
        self.iterations = 0
        self.kv_size = 0
        self.layer_time = 0.0  # ms

    @classmethod
    def from_line(self, line):
        tmp = line.split('::')[1].strip()
        if not tmp:
            raise ParseError('Failed to parse the line')

        layer_time = float(tmp.split()[0].split('=')[1])  # ms

        tmp = line.split()
        btch = int(tmp[1])
        iter = int(tmp[3])
        kv_sz = float(tmp[6])

        o = O()
        o.batch_size = btch
        o.iterations = iter
        o.kv_size = kv_sz
        o.layer_time = layer_time
        return o


def extract_data(path: str = './data.txt'):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('B:'):
                continue
            try:
                o = O.from_line(line)
                data.append(o)
            except ParseError:
                continue
    return data


def main():
    data = extract_data()

    number_of_layers = 32

    # Filter to iterations == 1 (minimum measured layer time case per batch)
    data_i1 = [o for o in data if o.iterations == 1]
    if not data_i1:
        raise RuntimeError('No entries with iterations == 1 found in data.txt')

    # Group by batch size; for safety, if multiple records exist, pick the one with min layer_time
    by_batch = {}
    for o in data_i1:
        if o.batch_size not in by_batch:
            by_batch[o.batch_size] = o
        else:
            if o.layer_time < by_batch[o.batch_size].layer_time:
                by_batch[o.batch_size] = o

    # Prepare sorted arrays by batch size
    batches = np.array(sorted(by_batch.keys()))
    BANDWIDTH_MB_PER_MS = 16.0
    movable_mb = np.array([by_batch[b].layer_time * BANDWIDTH_MB_PER_MS * number_of_layers for b in batches])
    kv_mb = np.array([by_batch[b].kv_size for b in batches])

    # Plot: line for movable data, scatter for KV size
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(batches, movable_mb, '-o', label='Movable Data (MB) @ 16 MB/ms')
    ax.scatter(batches, kv_mb, s=40, c='crimson', edgecolors='black', linewidths=0.6, alpha=0.9, label='KV Size (MB) [I=1]')

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Amount of Data (MB)')
    ax.set_title('Movable Data vs Batch Size (I=1) with KV Size Scatter')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    plt.tight_layout()
    out_path = 'data_moved_vs_batch.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


if __name__ == '__main__':
    main()
