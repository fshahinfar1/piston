import numpy as np
import cupy as cp
from nvidia import nvcomp


def do_experiment(algo):
    # Prepare your data (as a bytes object)
    data = "your uncompressed data..." * 100
    data_arr = nvcomp.as_array(data.encode())
    # print(data)
    original_size = data_arr.size * data_arr.item_size

    # Algorithms: LZ4, Cascaded, GDeflate
    codec = nvcomp.Codec(algorithm=algo)
    comp_arr = codec.encode(data_arr)
    comp_size = comp_arr.size * data_arr.item_size

    decomp_arr = codec.decode(comp_arr)
    decomp_size = decomp_arr.size * decomp_arr.item_size

    decomp_data = bytes(decomp_arr.cpu()).decode()

    # print('original:', data)
    # print('decompressed:', decomp_data)

    assert decomp_data == data  # Should match original
    assert decomp_size == original_size

    print(original_size, '-->', comp_size)


for x in ["LZ4", "Cascaded", "GDeflate"]:
    print(x)
    do_experiment(x)

