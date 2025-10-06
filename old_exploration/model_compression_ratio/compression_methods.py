# NOTE: these were not fast enough
# import time
# import zlib
# import gzip
# import bz2
# import lzma
# 
# def compress_zlib(input_file, output_file):
#     with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
#         data = f_in.read()
#         compressed = zlib.compress(data)
#         f_out.write(compressed)
# 
# def compress_gzip(input_file, output_file):
#     with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
#         f_out.writelines(f_in)
# 
# def compress_bz2(input_file, output_file):
#     with open(input_file, 'rb') as f_in, bz2.open(output_file, 'wb') as f_out:
#         f_out.writelines(f_in)
# 
# def compress_lzma(input_file, output_file):
#     with open(input_file, 'rb') as f_in, lzma.open(output_file, 'wb') as f_out:
#         f_out.writelines(f_in)
# 
# methods = {
#     'zlib': compress_zlib,
#     'gzip': compress_gzip,
#     'bz2': compress_bz2,
#     'lzma': compress_lzma
# }


from typing import *
import subprocess
import os

def compress_with_gzip(testfile):
    subprocess.run(["gzip", "-k", testfile], check=True)
    compressed_file = f"{testfile}.gz"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_pigz(testfile):
    subprocess.run(["pigz", "-k", testfile], check=True)
    compressed_file = f"{testfile}.gz"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_bzip2(testfile):
    subprocess.run(["bzip2", "-k", testfile], check=True)
    compressed_file = f"{testfile}.bz2"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_pbzip2(testfile):
    subprocess.run(["pbzip2", "-k", testfile], check=True)
    compressed_file = f"{testfile}.bz2"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_xz(testfile):
    subprocess.run(["xz", "-k", "-T0", testfile], check=True)
    compressed_file = f"{testfile}.xz"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_zstd(testfile):
    subprocess.run(["zstd", "-k", "-T0", testfile], check=True)
    compressed_file = f"{testfile}.zst"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_lz4(testfile):
    subprocess.run(["lz4", "-k", testfile], check=True)
    compressed_file = f"{testfile}.lz4"
    size = os.path.getsize(compressed_file)
    return size

def compress_with_7z(testfile):
    output_archive = f"{testfile}.7z"
    subprocess.run(["7z", "a", "-t7z", "-mmt=on", output_archive, testfile], check=True)
    size = os.path.getsize(output_archive)
    return size

methods: dict[str, Callable] = {
    'gzip': compress_with_gzip,
    'pigz': compress_with_pigz,
    'bzip2': compress_with_bzip2,
    'pbzip2': compress_with_pbzip2,
    'xz': compress_with_xz,
    'zstd': compress_with_zstd,
    'lz4': compress_with_lz4,
    '7z': compress_with_7z,
}
