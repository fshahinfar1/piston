import os
from vllm import LLM, SamplingParams
from datetime import datetime

def main():
    config = {
            "model": "facebook/opt-125m",
            "cpu_offload_gb": 100,
            "compilation_config": {
                "use_cudagraph": True,
                "cache_dir": "/home/farbod/__vllm_cache_dir/"
                },
            }

    question_path = os.path.join(os.path.dirname(__file__), '../data/questions.txt')
    print(question_path)
    with open(question_path, 'r') as f:
        prompts = f.readlines()
    print('Number of prompts:', len(prompts))

    llm = LLM(**config)
    start = datetime.now()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95))
    end = datetime.now()
    print('End to end time: ', end - start)

    # for prompt, output in zip(prompts, outputs):
    #     print('Prompt:', prompt)
    #     print(output.outputs[0].text)
    #     print('------')


if __name__ == '__main__':
    main()
