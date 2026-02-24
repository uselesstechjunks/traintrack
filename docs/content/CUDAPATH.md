# Cuda Learning Path
Goal: Build deep CUDA/GPU programming skills through reading production kernels, writing exercises, and learning concepts

# Modules
## Module 1: Thread Indexing & Kernel Launch Mechanics
- Expectation: Can write basic CUDA kernels
- Concepts
  - Global thread ID: `blockIdx.x * blockDim.x + threadIdx.x` — the universal CUDA indexing pattern
  - Grid/block dimensions: how `<<<gridSize, blockSize>>>` maps to hardware; max 1024 threads per block
  - Host ↔ device memory flow: `cudaMalloc` → `cudaMemcpy(H2D)` → `kernel launch` → `cudaMemcpy(D2H)` → `cudaFree`
  - Bounds checking: grid is rounded up, so trailing threads must not write out of bounds
- Exercise
  - Write a CUDA kernel that computes `c[i] = a[i] * b[i] + scalar` (fused multiply-add) for an array of floats.
  - Include the full host program: allocate, copy, launch, copy back, verify on CPU.
  - Experiment with different block sizes (32, 128, 256, 512, 1024).

## Module 2: Shared Memory & Synchronization
- Expectation: Can write basic CUDA kernels with shared memory
- Concepts
  - GPU memory hierarchy:
    - Registers: ~1 cycle, per-thread (`PTX %r`, `%f vars`)
    - Shared memory (`__shared__`): ~5 cycles, per-block, programmer-managed L1
    - Global memory (DRAM): ~400-800 cycles, all threads
  - `__syncthreads()`: barrier within a block — no thread passes until all reach it
  - When shared memory matters: when threads in a block need to cooperate (reductions, tiling, stencils) — not for independent element-wise ops
- Exercise
  - Write a 1D stencil kernel: `out[i] = 0.25*in[i-1] + 0.5*in[i] + 0.25*in[i+1]`.
  - Use shared memory so each block loads a tile (with halo elements) from global memory once, then all threads read from shared memory.
  - Handle boundary conditions.

## Module 3: PyTorch's Element-wise Kernel Abstraction
- Expectation: Understands PyTorch's CUDA kernel patterns
- Concepts
  - `gpu_kernel(iter, lambda)`: PyTorch's abstraction that generates the CUDA launch — you write the per-element math, it handles threading/indexing
  - `AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16, ...)`: compile-time type dispatch for `dtype` support
  - `GPU_LAMBDA`: makes a lambda callable from device code (`__device__`)
  - `opmath_type<scalar_t>`: promotes `half/bfloat16` to `float32` for numerical accuracy during computation, then casts back
  - How this connects to raw CUDA: `gpu_kernel` ultimately generates the same `blockIdx * blockDim + threadIdx` pattern internally
- Exercise
  - Implement a `GELU` kernel without using gpu_kernel — write the raw `__global__` kernel with `blockIdx/threadIdx` indexing, grid-stride loop, and `AT_DISPATCH` type dispatch.
  - Compare your version's structure to the `gpu_kernel` version.

## Module 4: Warp-Level Primitives
Expectation: Can write warp-level reductions
Concepts
 - Warp: 32 threads that execute in lockstep (SIMT). No synchronization needed within a warp.
 - Warp shuffle intrinsics (`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`):
   - `shfl_down`: each thread gets the value from lane + delta — used for tree reductions
   - `shfl_xor`: each thread exchanges with lane XOR mask — used for butterfly reductions
   - `shfl`: each thread reads from a specific source lane
   - All require a mask (typically 0xffffffff = all 32 lanes active)
 - Butterfly reduction pattern: for `mask = W/2; mask > 0; mask >>= 1` with `shfl_xor` — `O(log W)` steps
 - Two-level block reduction: (1) warp-level reduce → (2) lane 0 writes to shared memory → (3) `__syncthreads` → (4) first warp reduces the partial results
 - Why this matters: reductions are the building block of softmax, normalization, loss functions, attention
- Exercise
  - Implement a parallel sum reduction kernel from scratch.
  - Take an array of N floats and produce a single sum.
  - Use warp shuffles for intra-warp reduction and shared memory for inter-warp communication.
  - Verify against a CPU sum.

## Module 5: Memory Coalescing & Occupancy
- Expectation: Understands performance optimization levers
- Concepts
  - Memory coalescing: adjacent threads should access adjacent memory addresses.
    - A warp accessing 32 consecutive floats = 1 memory transaction (128 bytes). Random access = up to 32 transactions.
    - Vectorized loads: `float4` loads 4 floats in one instruction; `aligned_vector<T, N>` pattern
    - Occupancy: ratio of active warps to maximum warps per SM. Affected by:
    - Registers per thread (more registers → fewer concurrent warps)
    - Shared memory per block (more shared mem → fewer concurrent blocks)
    - Block size (must be multiple of warp size for efficiency)
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: runtime API to query optimal block count
  - Persistent kernels: launch fewer blocks than data elements, keep blocks alive processing multiple elements — amortizes launch overhead, enables register-resident computation
- Exercise
  - Take your Module 4 reduction kernel and optimize it:
    - (1) use float4 vectorized loads,
    - (2) experiment with different block sizes and measure occupancy impact.
  - Profile with `nvprof` or `ncu` if available, or reason about the coalescing patterns analytically.

## Module 6: Shared Memory Tiling (Matrix Multiply)
- Expectation: Can write tiled `GEMM` (the most important GPU algorithm)
- Concepts
  - Why tiling: naive matmul has `O(N)` global memory reads per output element; tiled matmul has `O(N/TILE_K)` — shared memory reuse across the tile
  - Three-level hierarchy: global → shared memory tiles → register fragments
  - Cooperative loading: all threads in a block collaborate to load tiles from global memory, then each thread computes its portion
  - Double buffering (advanced): overlap next tile load with current tile computation
  - Bank conflicts: shared memory is banked (32 banks); two threads accessing same bank = serialization. Padding (`__shared__ float A[TILE][TILE+1]`) avoids this.
- Exercise
  - Write a tiled matrix multiply kernel.
  - Start with `TILE_SIZE=16`. Load tiles of A and B into shared memory, `__syncthreads`, accumulate the partial dot products in registers, repeat for all tiles along K.
  - Verify correctness against CPU matmul.
  - Then experiment with `TILE_SIZE=32`.

## Module 7: Softmax — Three Implementations Compared
- Expectation: Can implement softmax — combines all prior concepts
- Concepts
  - Numerically stable softmax: subtract max before exp to prevent overflow
  - Three-pass algorithm: (1) find max, (2) sum of exp(x - max), (3) divide by sum
  - Design tradeoff: shared memory approach vs. register-resident vs. CUB library
  - When each is best: register-resident when softmax dim fits in warp registers; shared memory for larger dims; CUB for simplicity/correctness
- Exercise:
  - Write a softmax kernel for a 2D tensor `[batch, dim]` where dim <= 1024.
  - Use the warp-shuffle approach.
  - Verify numerical correctness against `torch.softmax`.

## Module 8: Triton — Python-Based GPU Kernels
- Expectation: Can write Triton kernels (faster prototyping path)
- Concepts
  - Triton programming model: write kernels in Python with `@triton.jit`, think in terms of blocks (tiles) not individual threads
  - `tl.program_id`: equivalent to blockIdx; tl.arange: equivalent to threadIdx range
  - `tl.load / tl.store`: block-level memory operations with masking for bounds
  - Auto-tuning: `@triton.autotune` with configs tries different block sizes automatically
  - Triton vs. CUDA tradeoff: Triton is ~80-95% of hand-tuned CUDA performance for most ops, with 5x less code. CUDA still wins for highly specialized kernels (flash attention, CUTLASS GEMMs).
- Exercise
  - Rewrite your Module 6 matrix multiply in Triton.
  - Compare the code complexity and (if possible) performance against your CUDA version.

## Module 9: Advanced Reading
Expectation: Recognize the patterns and understand the architecture when you encounter them in issues/PRs.

### 9a: Attention Mechanisms
Key idea: attention = two GEMMs (Q×K^T, scores×V) with softmax in between, fused into one kernel to avoid materializing the N×N attention matrix
 
### 9b: Quantized Inference
Key idea: lower precision (FP8, INT4) = 2-4x throughput via Tensor Cores, but requires careful scaling to maintain accuracy

### 9c: Distributed GPU Communication
Key idea: symmetric memory = GPU memory accessible by all GPUs without explicit copies; multicast = hardware-accelerated broadcast

9d: Vector Search
Key idea: similarity search = distance matrix (GEMM) + top-k selection (custom kernels with warp/block reductions)
