# Piston: Overcommiting vRAM To Overcome the Memory Limitation in Inference Systems

## About

Decode phase of LLM inference is bottelneck by available vRAM. In this project
we explore the possiblity of overcomming the barrier through a careful
synchronization of compute and data movement ballet.

> python3 -m piston.main -h
```
usage: main.py [-h] [--batch BATCH] [--num-requests NUM_REQUESTS] [--pipeline PIPELINE] [--iters ITERS] [--num-stages NUM_STAGES]

options:
  -h, --help            show this help message and exit
  --batch BATCH         number of request in a batch
  --num-requests NUM_REQUESTS
                        total number of requests to process
  --pipeline PIPELINE   which type of pipeline use for requests processing (simple, swapping)
  --iters ITERS         number of tokens to generate
  --num-stages NUM_STAGES
```
