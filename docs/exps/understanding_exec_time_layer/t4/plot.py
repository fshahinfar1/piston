#! python

import numpy as np
from scipy.interpolate import griddata
import matplotlib
# Use a non-GUI backend so this works in headless environments and avoids Qt issues
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ParseError(RuntimeError):
    pass


class O:
    def __init__(self):
        self.batch_size = 0
        self.iterations = 0
        self.kv_size = 0
        self.layer_time = 0

    @classmethod
    def from_line(self, line):
        tmp = line.split('::')[1].strip()
        if  not tmp:
            raise ParseError('Failed to parse the line')

        layer_time = float(tmp.split()[0].split('=')[1]) # ms

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


def extract_data():
    data = []
    with open('./data.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('B:'):
                continue
            try:
                o = O.from_line(line)
                data.append(o)
            except ParseError as e:
                # failed to parse
                continue
    return data


def main():
    data = extract_data()

    # Extract data
    x = np.array([o.batch_size for o in data])
    y = np.array([o.iterations for o in data])
    z = np.array([o.layer_time for o in data])

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create grid for interpolation (start at 0 so axes can start at origin)
    xi = np.linspace(0, max(x), 50)
    yi = np.linspace(0, max(y), 50)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate z values for smooth heatmap; try cubic then fall back to linear
    ZI = griddata(
        points=np.column_stack([x, y]),
        values=z,
        xi=(XI, YI),
        method='linear' # cubic
    )

    # Mask invalid regions (outside convex hull)
    ZI = np.ma.masked_invalid(ZI)

    # 2D filled contour (heatmap-like) plus contour lines for precise reading
    levels = 20
    cf = ax.contourf(XI, YI, ZI, levels=levels, cmap='viridis')
    c = ax.contour(XI, YI, ZI, levels=levels, colors='k', linewidths=0.6, alpha=0.7)
    ax.clabel(c, inline=True, fontsize=8, fmt='%.3f')

    # Optional: overlay sample points for reference
    ax.scatter(x, y, s=12, c='white', edgecolors='black', linewidths=0.5, alpha=0.8)

    # Color bar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, aspect=30)
    cbar.set_label('Layer Time (ms)', rotation=270, labelpad=15)

    # Labels and title
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Number of Iterations', fontsize=12)
    ax.set_xlim(0, np.max(x))
    ax.set_ylim(0, np.max(y))

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_path = 'heatmap.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

