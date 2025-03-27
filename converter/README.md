# TorchScript Runtime


## Bottleneck Issue
 - In batch inference, the inference time depends on the batch size because the flow-matching approach 
   uses a loop within `nfe_step` to solve ODEs. This process is slow as each iteration must wait for the previous one to complete.

## Performance ⚡

Testing on :
 - CPU: AMD Ryzen 7 5800H with Radeon Graphics 3.20 GHz Processor
 - GPU: NVIDIA GeForce RTX 3060 Laptop GPU
 
### Matcha (Condition Flow Matching)
![Latency](https://raw.githubusercontent.com/ccyrene/matcha-tts-integration/main/converter/perf.svg)

### Vocoder

### Batch Inference

### A Parallel ODE Solver