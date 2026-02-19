# In-depth GPU exec
Putting the first example from [this](https://henryhmko.github.io/posts/cuda/cuda.html) into [godbolt](https://godbolt.org/), we find 2 sections. This is a great example to learn from. Godbolt is showing you three layers of compiled output from your CUDA kernel. Let me walk through each one.

## 1. CUDA Source Code                                                                                                                                                                                                                                                                                             
```
__global__ void addTenKernel(const float *a, float *out) {                                                                                                                                                                                                                                                           
  int local_x = threadIdx.x;                          
  out[local_x] = a[local_x] + 10;
}
```
`__global__` marks this as a kernel — a function that runs on the GPU but is called from the CPU. Each GPU thread executes this function independently. `threadIdx.x` is a built-in variable that gives each thread its unique index within a block. So thread 0 processes element 0, thread 1 processes element 1, etc.

## 2. The PTX Output (GPU "Assembly")

PTX (Parallel Thread Execution) is NVIDIA's intermediate representation — think of it as GPU assembly language. This is the most instructive part. Let me annotate it line by line:
```
ld.param.u64  %rd1, [... _param_0];   // Load pointer 'a' (param 0) into register rd1
ld.param.u64  %rd2, [... _param_1];   // Load pointer 'out' (param 1) into register rd2
```
These load the two 64-bit pointer arguments from the kernel's parameter space.
```
cvta.to.global.u64  %rd3, %rd2;       // Convert 'out' pointer to global address space
cvta.to.global.u64  %rd4, %rd1;       // Convert 'a' pointer to global address space
```
CUDA has multiple address spaces (global, shared, local). `cvta` (convert to address) translates the generic parameter-space pointers into global memory pointers that can be dereferenced.
```
mov.u32  %r1, %tid.x;                 // r1 = threadIdx.x
```
This is the PTX equivalent of `int local_x = threadIdx.x`. `%tid.x` is a special register holding the thread's X index.
```
mul.wide.s32  %rd5, %r1, 4;           // rd5 = threadIdx.x * 4 (byte offset)
```
Each float is 4 bytes. This computes the byte offset into the array. mul.wide multiplies a 32-bit value and produces a 64-bit result (needed because pointers are 64-bit).
```
add.s64  %rd6, %rd4, %rd5;            // rd6 = &a[threadIdx.x]
ld.global.f32  %f1, [%rd6];           // f1 = a[threadIdx.x]
```
Compute the address of `a[local_x]`, then load the float from global memory.
```
add.f32  %f2, %f1, 0f41200000;        // f2 = f1 + 10.0
```
Here's the `+ 10`. The value `0f41200000` is the IEEE 754 hex representation of `10.0` as a 32-bit float. You can verify: `sign=0, exponent=130` (bias `127`, so `2^3=8`), `mantissa=1.25`, giving `8 × 1.25 = 10.0`.
```
add.s64  %rd7, %rd3, %rd5;            // rd7 = &out[threadIdx.x]
st.global.f32  [%rd7], %f2;           // out[threadIdx.x] = f2
ret;
```
Compute the address of `out[local_x]`, store the result, and return.

Summary of PTX data flow:
```
a[threadIdx.x]  -->  load  -->  + 10.0  -->  store  -->  out[threadIdx.x]
```
## 3. The x86-64 Host Code (CPU Side)

This is the part that runs on the CPU. It's the runtime scaffolding that CUDA generates automatically. None of this runs on the GPU.

Key functions:

`__sti____cudaRegisterAll()` — Called at program startup (before main). It:
1. Calls `__cudaRegisterFatBinary` to register the embedded GPU binary (`fatbinData`) with the CUDA runtime
2. Calls `__cudaRegisterFunction` to associate the host-side function name "`addTenKernel`" with the GPU code
3. Registers `__cudaUnregisterBinaryUtil` with `atexit()` for cleanup on exit

`fatbinData` — The giant block of `.quad` hex values is the fat binary: the compiled GPU machine code (SASS) and PTX, packaged into an ELF-like container and embedded directly into the host executable. This is what gets shipped to the GPU at launch time.

`addTenKernel(float const*, float*)` (host wrapper) — When you call `addTenKernel<<<blocks, threads>>>(a, out)` from host code, the compiler rewrites it into a call to this wrapper. It simply forwards to` __device_stub__....`

`__device_stub__Z12addTenKernelPKfPf` — The real launch logic:
1. Packs the two pointer arguments `(a, out)` into a `void**` array on the stack
2. Calls `__cudaPopCallConfiguration` to retrieve the `<<<blocks, threads>>>` launch parameters
3. Calls `cudaLaunchKernel(...)` with the kernel handle, grid dimensions, block dimensions, and argument array

This is what the `<<<>>>` syntax compiles down to — it's just a cudaLaunchKernel call.

## The Big Picture
```
Source code (.cu)
    │
    ├──► PTX (GPU intermediate repr) ──► SASS (GPU machine code) ──► embedded in fatbin
    │
    └──► x86-64 host code (launch stubs, runtime registration)
```
When you write `kernel<<<1, 100>>>(a, out)`, the compiler splits it into:
- GPU side: the PTX/SASS that each of the 100 threads executes
  - CPU side: boilerplate that registers the kernel at startup, packs arguments, and calls cudaLaunchKernel

For learning CUDA, the PTX section is the most valuable to study. The host-side x86 is mechanical boilerplate you'll never write by hand.
