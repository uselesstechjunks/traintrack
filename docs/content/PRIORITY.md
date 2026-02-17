1. Distributed training reasoning
2. GPU mental model
3. Performance debugging intuition

# Reading Map (Sharper Version)

This is breadth-first but targeted.

## Layer 1 — Distributed Training Architecture

You must deeply understand:

* Data parallel
* Tensor parallel
* Pipeline parallel
* Hybrid parallel
* FSDP vs ZeRO
* Why tensor parallel increases communication

Read in this order:

1. [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
2. [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
3. [PyTorch FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html)
4. [Hugging Face blog: “Parallelism Strategies for Training LLMs”](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)

Do not skim.
Build a one-page summary for yourself after each.

## Layer 2 — Memory Accounting

This is what separates RE from product MLE.

You must know:

* Parameters memory
* Optimizer states memory
* Gradient memory
* Activation memory
* KV cache (for inference)
* Checkpointing tradeoffs

Read:

* [“How to Train a 175B Parameter Model” blog](https://markaicode.com/multi-node-training-175b-parameter-models/) (Eleuther or similar breakdowns)
* Other blogs such as [this](https://markaicode.com/llm-memory-optimization-reduce-vram-usage/), [this](https://blog.ezyang.com/)
* Any blog explaining memory breakdown in LLM training

Be able to calculate:
If model is 30B in bf16:
How much memory for weights?
How much for Adam?
What happens with ZeRO stage 3?

## Layer 3 — Communication & Scaling

Read:

* NCCL ring AllReduce explanation - [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
* Communication vs compute overlap articles - [paper1](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959), [paper2](https://arxiv.org/abs/2507.03114)
* Megatron communication diagrams - [this](https://danielvegamyhre.github.io/ml/performance/2025/03/30/illustrated-megatron.html), [this](https://deepwiki.com/ROCm/Megatron-LM/3.3-communication-primitives), [this](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)

Be able to explain:

* Why scaling stalls
* Why adding GPUs doesn’t linearly speed training
* What bandwidth saturation means

## Layer 4 — GPU Execution Model

Read:

* CUDA programming model overview - [this](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
* Ampere architecture summary - [this](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
* Triton tutorial (matmul example) - [here](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

You don’t need to write kernels.
But you must understand:

Why FlashAttention improves memory locality.
Why kernel fusion helps.

## Layer 5 — Profiling Mental Model

Read:

* Nsight Systems tutorial - [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
* PyTorch profiler docs - [here](https://docs.pytorch.org/docs/stable/profiler.html)

Understand:

If utilization is low:
Is it CPU bound?
Is it comm bound?
Is it kernel launch overhead?
Is it memory bound?
