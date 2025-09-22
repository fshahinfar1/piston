#! python3

def extract_data(fstream):
    lines = fstream.readlines()
    data = {}
    for line in lines:
        if line.startswith('python'):
            tmp = line.split()
            batch_size = int(tmp[3])
            pipeline = tmp[7]
        elif line.startswith('Req '):
            tmp = line.split()
            kv_size = round(int(tmp[3]) / (1024 ** 3), 2)  # GB
        elif line.startswith('Time'):
            tmp = line.split()
            dur = round(float(tmp[5]) / 60, 2) # minute
            lst = data.setdefault(pipeline, [])
            lst.append((batch_size, kv_size, dur))
        else:
            continue
    return data

def report_data(data):
    xtick = [1,2,4,8,16,32,64,128,256,512,1024]
    labels = []
    for pipeline, measurements in data.items():
        # sort based on batch size
        tmp = sorted(measurements, key=lambda x: x[0])
        x = [t[0] for t in tmp]
        ticks = [t[0] for t in tmp if t[0] in xtick] 
        labels = [f'"{t[0]}\\n{t[1]} GB"' for t in tmp if t[0] in xtick]
        y = [t[2] for t in tmp]
        print(pipeline)
        print('x:', x)
        print('y:', y)
        print('xticks:', ticks)
        print('xtick_labels:', '[', ','.join(labels), ']')


def main():
    with open('./runtime_log.txt', 'r') as f:
        data = extract_data(f)
    report_data(data)

if __name__ == '__main__':
    main()