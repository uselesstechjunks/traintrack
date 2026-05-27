# 0. Topics
1. Distributed training reasoning
2. GPU mental model
3. Performance debugging intuition

# 1. GPU Performance Engineering Resource
- [NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html)
- [Learning Guide: Performance Engineering for AI Infra](https://github.com/wafer-ai/gpu-perf-engineering-resources)
- [Introduction to CUDA Programming With GPU Puzzles](https://henryhmko.github.io/posts/cuda/cuda.html)
- [LeetGPU](https://leetgpu.com/challenges)

# 2. Beyond Tools
## 2.1 Distributed Training Architecture
- Data parallel
- Tensor parallel
- Pipeline parallel
- Hybrid parallel
- FSDP vs ZeRO
- Why tensor parallel increases communication

### 2.1.1 Reading List
1. [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
2. [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
3. [PyTorch FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html)
4. [Hugging Face blog: “Parallelism Strategies for Training LLMs”](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)

## 2.2 Memory Accounting
- Parameters memory
- Optimizer states memory
- Gradient memory
- Activation memory
- KV cache (for inference)
- Checkpointing tradeoffs

### 2.2.1 Reading List
- [“How to Train a 175B Parameter Model” blog](https://markaicode.com/multi-node-training-175b-parameter-models/) (Eleuther or similar breakdowns)
- Other blogs such as [this](https://markaicode.com/llm-memory-optimization-reduce-vram-usage/), [this](https://blog.ezyang.com/)
- Any blog explaining memory breakdown in LLM training

### 2.2.2 Exercise
If model is 30B in bf16: How much memory for weights? How much for Adam? What happens with ZeRO stage 3?

## 2.3 Communication & Scaling
### 2.3.1 Reading List
- NCCL ring AllReduce explanation - [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- Communication vs compute overlap articles - [paper1](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959), [paper2](https://arxiv.org/abs/2507.03114)
- Megatron communication diagrams - [this](https://danielvegamyhre.github.io/ml/performance/2025/03/30/illustrated-megatron.html), [this](https://deepwiki.com/ROCm/Megatron-LM/3.3-communication-primitives), [this](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

### 2.3.2 Exercise
- Why scaling stalls
- Why adding GPUs doesn’t linearly speed training
- What bandwidth saturation means

## 2.4 GPU Execution Model
### 2.4.1 Reading List
- CUDA programming model overview - [this](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- Ampere architecture summary - [this](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- Triton tutorial (matmul example) - [here](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
### 2.4.2 Exercise
- Why FlashAttention improves memory locality.
- Why kernel fusion helps.

## 2.5 Profiling Mental Model
### 2.5.1 Reading List
- Nsight Systems tutorial - [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- PyTorch profiler docs - [here](https://docs.pytorch.org/docs/stable/profiler.html)
### 2.5.2 Exercise
If utilization is low:
- Is it CPU bound?
- Is it comm bound?
- Is it kernel launch overhead?
- Is it memory bound?
