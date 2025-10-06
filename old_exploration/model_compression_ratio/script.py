import os
import sys
import shutil
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from compression_methods import methods as M
import torch


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


class ExpResult:
    def __init__(self, name: str):
        self.model_name = name
        self.model_orig_size = 0
        self.tokenizer_orig_size = 0
        self.compressed_sizes: dict[str, List[int]] = {}

        name_e = name.replace('/', '.')
        self.mpath = f"./_data/models/{name_e}"
        self.tpath = f"./_data/tokenizers/{name_e}"
        self.fetch_files()

    def get_model_files(self) -> List[str]:
        # tmp = os.listdir(self.mpath)
        # tmp = [x for x in tmp if x.endswith('.safetensors')]
        # return [os.path.join(self.mpath, x) for x in tmp]
        return [self.mpath,]

    def fetch_files(self) -> None:
        # check if we have created the model file then do not refetch the model
        if not os.path.exists(self.mpath):
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Save (export) model and tokenizer
            torch.save(model.state_dict(), self.mpath)
            # model.save_pretrained(self.mpath)
            # tokenizer.save_pretrained(self.tpath)

        size = 0
        for f in self.get_model_files():
            size += os.stat(f).st_size
        self.model_orig_size = size
        self.tokenizer_orig_size = 0

    def apply_compression(self, comp_method: str) -> None:
        comp_func = M[comp_method]

        output_path = '/tmp/compression_test'
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        mkdir_if_not_exist(output_path)

        model_comp_size = 0
        for i, f in enumerate(self.get_model_files()):
            # output_file = os.path.join(output_path, f'file_{i}')
            # comp_func(f, output_file)
            # model_comp_size += os.stat(output_file).st_size
            tmp_comp_sz = comp_func(f)
            model_comp_size + tmp_comp_sz
        self.compressed_sizes[comp_method] = [model_comp_size, 0]

    def get_model_ratio(self, comp_method: str) -> float:
        return self.compressed_sizes[comp_method][0] / self.model_orig_size

    def get_tokenizer_ratio(self, comp_method)-> float:
        return 0
        # return self.compressed_sizes[comp_method][1] / self.tokenizer_orig_size

    def get_output_lines(self) -> str:
        build = []
        for key,val in self.compressed_sizes.items():
            l1 =  f'{self.model_name},{key},model,{self.model_orig_size},{val[0]},{self.get_model_ratio(key):.3f}'
            l2 =  f'{self.model_name},{key},token,{self.tokenizer_orig_size},{val[1]},{self.get_tokenizer_ratio(key):.3f}'
            build.append(l1)
            build.append(l2)
        return '\n'.join(build)


def main():
    mkdir_if_not_exist('./_data')
    mkdir_if_not_exist('./_data/models')
    mkdir_if_not_exist('./_data/tokenizers')
    output_file = "./results.txt"

    # Specify model checkpoint; replace with any model you prefer
    # "distilbert-base-uncased-finetuned-sst-2-english",
    # "bert-large-uncased-whole-word-masking-finetuned-squad",
    # "Helsinki-NLP/opus-mt-en-fr",
    # "t5-base",
    # "facebook/bart-large-cnn",
    # "google/pegasus-xsum",
    # "dbmdz/bert-large-cased-finetuned-conll03-english",
    list_of_models = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/phi-4",
            "meta-llama/Llama-3.1-8B-Instruct",
            ]

    # list_of_compression_methods = [
    #         "lzma",
    #         "lzma2",
    #         "zstd",
    #         "deflate",
    #         "bzip2",
    #         "lz77",
    #         "lzss",
    #         "huffman",
    #         "arithmetic"
    #         ]

    tmp = []
    for model_name in list_of_models:
        R: ExpResult = ExpResult(model_name)
        print('model:', model_name, 'size:', R.model_orig_size)
        for method in M.keys():
            R.apply_compression(method)
            print('    ', method, ':', R.get_model_ratio(method))
        tmp.append(R)

    with open(output_file, 'w') as f:
        for r in tmp:
            f.write(r.get_output_lines())
    print('Done')


if __name__ == '__main__':
    main()
