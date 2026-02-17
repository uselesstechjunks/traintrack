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

1. Megatron-LM paper
2. DeepSpeed ZeRO paper
3. PyTorch FSDP docs
4. Hugging Face blog: “Parallelism Strategies for Training LLMs”

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

* “How to Train a 175B Parameter Model” blog (Eleuther or similar breakdowns)
* Any blog explaining memory breakdown in LLM training

Be able to calculate:
If model is 30B in bf16:
How much memory for weights?
How much for Adam?
What happens with ZeRO stage 3?

## Layer 3 — Communication & Scaling

Read:

* NCCL ring AllReduce explanation
* Communication vs compute overlap articles
* Megatron communication diagrams

Be able to explain:

* Why scaling stalls
* Why adding GPUs doesn’t linearly speed training
* What bandwidth saturation means

## Layer 4 — GPU Execution Model

Read:

* CUDA programming model overview
* Ampere architecture summary
* Triton tutorial (matmul example)

You don’t need to write kernels.
But you must understand:

Why FlashAttention improves memory locality.
Why kernel fusion helps.

## Layer 5 — Profiling Mental Model

Read:

* Nsight Systems tutorial
* PyTorch profiler docs

Understand:

If utilization is low:
Is it CPU bound?
Is it comm bound?
Is it kernel launch overhead?
Is it memory bound?
