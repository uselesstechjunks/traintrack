# I. The LLM Training Stack
### 1. End-to-End View of Modern LLM Training

Start here to frame everything:

* Stanford CS336: *Training Large Language Models* (lecture notes + videos)
* Hugging Face blog: “The Illustrated Transformer” (fast skim if rusty)
* “Scaling Laws for Neural Language Models” (Kaplan et al.)
* Chinchilla paper (compute-optimal scaling)

Purpose:
Understand compute regimes, data regimes, batch size scaling, optimizer choices, throughput constraints.

# II. Distributed Training Systems (Core)

## A. PyTorch Distributed & Parallelism

Read:

* PyTorch Distributed Overview (official docs)
* FSDP tutorial
* DeepSpeed ZeRO papers (ZeRO-1/2/3)

Concepts to understand:

* Data parallel vs tensor parallel vs pipeline parallel
* FSDP sharding vs ZeRO sharding
* Communication cost models
* AllReduce vs ReduceScatter vs AllGather
* Gradient accumulation
* Activation checkpointing

## B. Megatron-LM Design

Read:

* Megatron-LM paper
* Megatron-LM GitHub README
* Blog posts summarizing tensor parallelism

Key concepts:

* Column/row parallel linear layers
* Attention partitioning
* Overlapping communication and compute

# III. GPU Architecture & Performance Engineering (filter for product MLEs)

## A. GPU Fundamentals

Read:

* CUDA Programming Guide (first 3–4 chapters only)
* NVIDIA GPU architecture overview (Ampere or Hopper whitepaper)

Understand:

* SMs, warps, blocks
* Memory hierarchy (register, shared, L2, HBM)
* Occupancy
* Memory bandwidth vs compute bound

Reason: “If training is slow, what could be happening at hardware level?”

## B. Profiling & Bottleneck Diagnosis

Read:

* Nsight Systems documentation overview
* Nsight Compute documentation
* NVTX instrumentation tutorial
* PyTorch profiler guide

Understand:

* Kernel launch timeline
* GPU utilization vs CPU starvation
* Communication stalls
* Memory fragmentation
* Host-device synchronization

# IV. Custom Kernel Development

## A. Triton

Read:

* Triton language tutorial
* Triton matmul tutorial
* Triton fused attention examples

Goal:
Understand how Triton differs from CUDA.
Understand block-level programming model.

## B. PyTorch Extensions

Read:

* PyTorch C++/CUDA extension tutorial
* torch.utils.cpp_extension docs

Goal:
Understand integration path.

# V. Communication & NCCL

Read:

* NCCL developer guide
* Ring AllReduce algorithm explanation
* Tree vs ring collective tradeoffs

Understand:

* Bandwidth vs latency regimes
* Scaling inefficiencies
* Topology effects (NVLink vs PCIe vs Infiniband)

# VI. Transformer Performance Techniques

Likely architecture questions.

Read:

* FlashAttention paper
* FlashAttention-2
* GQA / MQA explanations
* ALiBi vs RoPE scaling

Understand:

* Why attention is quadratic
* IO-aware algorithms
* Memory vs compute tradeoffs

# VII. Precision & Memory Optimization

Read:

* Mixed Precision Training (Micikevicius et al.)
* FP8 training (NVIDIA blog / Hopper whitepaper)
* Gradient checkpointing explanation

Understand:

* Loss scaling
* Numerical instability sources
* Memory reduction strategies

# VIII. Large Training Stability & Reproducibility

Read:

* OpenAI RLHF implementation blog posts
* Fault tolerance in distributed training (checkpointing best practices)
* Deterministic training in PyTorch

Understand:

* Reproducibility challenges
* Random seed issues
* Divergence debugging

# IX. Orchestration & Infra Layer

They mention Kubernetes / SLURM.

Read:

* SLURM basics
* Kubernetes GPU scheduling overview
* Ray distributed training docs (optional)

Understand:

* Job scheduling
* Resource allocation
* Multi-node environment setup

# X. Mellum Context

Read:

* JetBrains Mellum blog posts
* Any available papers or blog on their code LLM
* Code LLM landscape: Code Llama, StarCoder

Understand:

* What makes code LLMs different
* Long context importance
* Evaluation metrics for code models
