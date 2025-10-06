#! python3
import os
import sys
import json
from pprint import pprint

import gzip
import io

class Obj:
    def __init__(self):
        self.files = []
        self.layer_directory = {}

    def add_file(self, path):
        file = open(path, 'rb')
        header_size = file.read(8)
        header_size = int.from_bytes(header_size, byteorder='little', signed=False)
        # print('Header size:', header_size)

        meta_data = file.read(header_size)
        meta_data = json.loads(meta_data)
        # print('Meta data:')
        # pprint(meta_data)

        f = { 'path': path,
              'header_size': header_size,
              'header': meta_data,
              'file': file,
            }
        self.files.append(f)

        # create a directory of which layer are on which file and where on the
        # file
        index = len(self.files) - 1
        raw_param_base = 8 + header_size
        for k, m in meta_data.items():
            # find out this key belongs to which layer
            tmp = k.split('.')
            if len(tmp) < 3:
                continue
            if tmp[0] != 'model' or tmp[1] != 'layers':
                continue
            layer = int(tmp[2])
            directory = self.layer_directory.get(layer, [])

            begin, end = m['data_offsets']
            size = end - begin
            # which file (index), what absolute offset in the file, and how
            # many bytes
            directory.append([index, begin + raw_param_base, size])
            self.layer_directory[layer] = directory

    def get_layer_weights(self, layer):
        layer_bytes = []
        directory = self.layer_directory.get(layer, [])
        for x in directory:
            index, off, sz = x
            file = self.files[index]['file']
            file.seek(off)
            x = file.read(sz)
            layer_bytes.append(x)
        layer_bytes = b''.join(layer_bytes)
        return layer_bytes

def name_is_for_layer(name, target_layer):
    tmp = name.split('.')
    if len(tmp) < 3:
        return False
    if tmp[0] != 'model' or tmp[1] != 'layers':
        return False
    return int(tmp[2]) == target_layer

def do_gzip(bytes):
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as gz:
        gz.write(bytes)
    return compressed_buffer.getvalue()

def main():
    o = Obj()

    dir = '/home/farbod/data/mixtral_8x7b_instruct_v0.1/bf8/raw/'
    files = [x for x in os.listdir(dir) if x.endswith('.safetensors')]
    for f in files:
        o.add_file(os.path.join(dir, f))


    results = []

    number_of_layers = 32
    print('layer & compressed (%) & orig & comp')
    for layer in range(0, number_of_layers):
        x = o.get_layer_weights(layer)
        comp = do_gzip(x)
        orig_size = len(x)
        comp_size = len(comp)
        if orig_size != 0:
            percent = (comp_size - orig_size) / orig_size * 100
        else:
            percent = 'NaN'
        results.append((layer, percent, orig_size, comp_size))
        print(layer,percent, orig_size, comp_size)

    print('Done')

if __name__ == '__main__':
    main()

