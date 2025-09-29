# Piston: Overcommiting vRAM To Overcome the Memory Limitation in Inference Systems

## About

Decode phase of LLM inference is bottelneck by available vRAM. In this project
we explore the possiblity of overcomming the barrier through a careful
synchronization of compute and data movement ballet.

```
.
├── docs
│   └── exps  <-- some experiments, data, and figures
├── README.md
├── scripts
│   ├── compression_across_layers  <-- measureing standard compression techniques on model layers
│   ├── install.sh
│   ├── layers_exec_time <-- measure execution time of layers
│   ├── model_compression_ratio <-- compression investigated
│   ├── nvlink_pipeline <-- A inference system based on transformer API that does swapping
│   ├── nvlink_speed  <-- measure NVLink speed
│   ├── on_gpu_compression <-- compression tests
├── src
```

