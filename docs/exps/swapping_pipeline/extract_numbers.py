import os

data_dir = './data'
results = {}
for exp in os.listdir(data_dir):
    for i in range(1, 17):
        fname = os.path.join(data_dir, exp, f'{i}.txt')
        with open(fname, 'r') as f:
            last_line = f.readlines()[-1]
            seconds = float(last_line.split()[-2])

            if exp not in results:
                results[exp] = []
            results[exp].append((i, seconds))

for exp, measurements in results.items():
    print(exp,':')
    measurements.sort(key=lambda t: t[0])
    X = [t[0] for t in measurements]
    Y = [t[1] for t in measurements]
    print('x:', X)
    print('y:', Y)