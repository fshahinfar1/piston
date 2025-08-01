import numpy as np
import cupy as cp
from nvidia import nvcomp
import torch


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


def do_exp_with_cupy(algo):
    # codec = nvcomp.Codec(algorithm='GDeflate', algorithm_type=5)
    codec = nvcomp.Codec(algorithm=algo)

    print('Testing with an array of uint8')
    raw_data = np.arange(0, 10000)
    arr = cp.array(raw_data, dtype=cp.uint8) 
    comp = codec.encode(nvcomp.as_array(arr))

    decomp = codec.decode(comp).cuda()
    t2 = cp.asarray(decomp, dtype=cp.uint8)

    print('orig shape:', arr.size, arr.itemsize)
    print('comp shape:', comp.size, comp.item_size)
    print('decomp shape:', t2.size, t2.itemsize)


    for i, (a,b) in enumerate(zip(arr, t2)):
        if a != b:
            print('@', i, ':', a, 'vs', b)
            raise RuntimeError('test failed')
    print('Okay')
    print('-------------')

    print('Testing with an array of float16')
    raw_data = np.arange(0, 10000)
    arr = cp.array(raw_data, dtype=cp.float16) 
    arr_wrap = arr.view(cp.uint8)
    comp = codec.encode(nvcomp.as_array(arr_wrap))

    decomp = codec.decode(comp).cuda()
    t2 = cp.fromDlpack(decomp.to_dlpack()).view(cp.float16)

    print('orig shape:', arr.size, arr.itemsize)
    print('comp shape:', comp.size, comp.item_size)
    print('decomp shape:', t2.size, t2.itemsize)

    assert t2.size == arr.size

    for i, (a,b) in enumerate(zip(arr, t2)):
        if a != b:
            print('@', i, ':', a, 'vs', b)
            raise RuntimeError('test failed')
    print('Okay')
    print('-------------')


def do_exp_with_torch(algo):
    codec = nvcomp.Codec(algorithm=algo)

    raw_data = np.arange(0, 4552704, dtype=np.float16)
    tensor = torch.tensor(raw_data, dtype=torch.float16).to('cuda:0')

    arr = cp.from_dlpack(tensor)
    arr_wrap = arr.view(cp.uint8)

    comp = codec.encode(nvcomp.as_array(arr_wrap))

    decomp = codec.decode(comp).cuda()
    t2 = cp.from_dlpack(decomp.to_dlpack()).view(cp.float16)

    tensor2 = torch.tensor(t2)

    print('orig shape:', tensor.numel(), tensor.element_size())
    print('comp shape:', comp.size, comp.item_size)
    print('decomp shape:', tensor2.numel(), tensor2.element_size())

    assert tensor2.numel() == tensor.numel()

    for i, (a,b) in enumerate(zip(tensor, tensor2)):
        if a != b:
            print('@', i, ':', a, 'vs', b)
            raise RuntimeError('test failed')
    print('Okay')
    print('-------------') 


def do_exp_list_of_tensors(algo):
    codec = nvcomp.Codec(algorithm=algo)
    raw_data = [np.random.rand(4552704).astype(np.float16)
                                            for j in range(64)]
    raw_data = [torch.tensor(d, dtype=torch.float16).to('cuda')
                                                for d in raw_data]

    comp_arr = []
    for tensor in raw_data:
        tmp = cp.from_dlpack(tensor).view(cp.uint8)
        comp = codec.encode(nvcomp.as_array(tmp))
        comp_arr.append(comp)

    # Create a new codec
    codec = nvcomp.Codec(algorithm=algo)
    decomp_arr = []
    for comp in comp_arr:
        decomp = codec.decode(comp)
        tmp = cp.from_dlpack(decomp.to_dlpack()).view(cp.float16)
        tensor = torch.tensor(tmp)
        decomp_arr.append(tensor)

    # del comp_arr

    for q, (t1, t2) in enumerate(zip(raw_data, decomp_arr)):
        # print('orig shape:', t1.numel(), t1.element_size())
        # print('decomp shape:', t2.numel(), t2.element_size())

        assert t2.numel() == t1.numel()

        if not torch.all(t1.eq(t2)):
            print('tensor:', q)
            raise RuntimeError('test failed')


for x in ["ANS", "LZ4", "GDeflate"]: # "Cascaded",
    print(x)
    # do_experiment(x)
    # do_exp_with_cupy(x)
    # do_exp_with_torch(x)
    do_exp_list_of_tensors(x)

