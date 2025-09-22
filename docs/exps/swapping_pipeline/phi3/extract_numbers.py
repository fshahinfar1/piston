import os

data_dir = './data'
results = {}
req_sizes = []
for exp in os.listdir(data_dir):
    for i in range(1, 26):
        fname = os.path.join(data_dir, exp, f'{i}.txt')
        with open(fname, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            seconds = float(last_line.split()[-2])

            if exp not in results:
                results[exp] = []
            results[exp].append((i, seconds))

            if exp == 'simple':
                req_size = round(int(lines[-2].split()[-1]) / 1024 ** 3, 2)
                req_sizes.append((i, req_size))


for exp, measurements in results.items():
    print(exp,':')
    measurements.sort(key=lambda t: t[0])
    X = [t[0] for t in measurements]
    Y = [t[1] for t in measurements]
    print('x:', X)
    print('y:', Y)

req_sizes.sort(key=lambda t: t[0])

print('size:', [t[1] for t in req_sizes])
