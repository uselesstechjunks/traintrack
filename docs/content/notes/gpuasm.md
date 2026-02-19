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

## 4. The Big Picture
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
Here's the full PTX execution traced step by step. I'll use thread 3 (threadIdx.x = 3) as a concrete example, assuming a is at address 0x7F00 and out is at 0x8F00.                                                                                                                                                  
```                                                                 
   PARAMETER SPACE                    SPECIAL REGISTERS
  ┌─────────────────┐                ┌─────────────────┐                                                                                                                                                                                                                                                               
  │ param_0: 0x7F00 │ (ptr to a)     │ %tid.x:  3      │
  │ param_1: 0x8F00 │ (ptr to out)   └─────────────────┘                                                                                                                                                                                                                                                               
  └─────────────────┘                                   

   GLOBAL MEMORY (GPU DRAM)
  ┌────────┬────────┬────────┬────────┬────────┬─────┐
  │ a[0]   │ a[1]   │ a[2]   │ a[3]   │ a[4]   │ ... │  (input, read-only)
  │ 1.0    │ 2.0    │ 3.0    │ 5.0    │ 7.0    │     │
  ├────────┼────────┼────────┼────────┼────────┼─────┤
  │ +0     │ +4     │ +8     │ +12    │ +16    │     │  (byte offsets)
  └────────┴────────┴────────┴────────┴────────┴─────┘
  ┌────────┬────────┬────────┬────────┬────────┬─────┐
  │ out[0] │ out[1] │ out[2] │ out[3] │ out[4] │ ... │  (output)
  │  ??    │  ??    │  ??    │  ??    │  ??    │     │
  └────────┴────────┴────────┴────────┴────────┴─────┘
═══════════════════════════════════════════════════════════════════
    STEP   INSTRUCTION                        REGISTERS AFTER
═══════════════════════════════════════════════════════════════════
    ┌─ PHASE 1: LOAD PARAMETERS ─────────────────────────────────┐
    │                                                            │
    │  1 │ ld.param.u64  %rd1, [param_0]                         │
    │    │                                                       │
    │    │   param_0 ──────► %rd1 = 0x7F00  (ptr to a)           │
    │    │                                                       │
    │  2 │ ld.param.u64  %rd2, [param_1]                         │
    │    │                                                       │
    │    │   param_1 ──────► %rd2 = 0x8F00  (ptr to out)         │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    ┌─ PHASE 2: CONVERT TO GLOBAL ADDRESS SPACE ─────────────────┐
    │                                                            │
    │  3 │ cvta.to.global.u64  %rd3, %rd2                        │
    │    │                                                       │
    │    │   %rd2 ──[cvta]──► %rd3 = 0x8F00  (global out ptr)    │
    │    │                                                       │
    │  4 │ cvta.to.global.u64  %rd4, %rd1                        │
    │    │                                                       │
    │    │   %rd1 ──[cvta]──► %rd4 = 0x7F00  (global a ptr)      │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    ┌─ PHASE 3: COMPUTE BYTE OFFSET ─────────────────────────────┐
    │                                                            │
    │  5 │ mov.u32  %r1, %tid.x                                  │
    │    │                                                       │
    │    │   %tid.x ──────► %r1 = 3  (thread index)              │
    │    │                                                       │
    │  6 │ mul.wide.s32  %rd5, %r1, 4                            │
    │    │                                                       │
    │    │   %r1 ──[× 4]──► %rd5 = 12  (byte offset)             │
    │    │                   32-bit       64-bit                 │
    │    │                                                       │
    │    │   WHY × 4?  sizeof(float) = 4 bytes                   │
    │    │   Element 3 starts at byte 12                         │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    ┌─ PHASE 4: LOAD a[3] ───────────────────────────────────────┐
    │                                                            │
    │  7 │ add.s64  %rd6, %rd4, %rd5                             │
    │    │                                                       │
    │    │   %rd4 (0x7F00) + %rd5 (12) ──► %rd6 = 0x7F0C         │
    │    │                                                       │
    │  8 │ ld.global.f32  %f1, [%rd6]                            │
    │    │                                                       │
    │    │   GLOBAL MEMORY                                       │
    │    │   ┌──────────────────────────┐                        │
    │    │   │  addr 0x7F0C  ═  a[3]   │                         │
    │    │   │  value: 5.0             │───► %f1 = 5.0           │
    │    │   └──────────────────────────┘                        │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    ┌─ PHASE 5: ADD 10 ──────────────────────────────────────────┐
    │                                                            │
    │  9 │ add.f32  %f2, %f1, 0f41200000                         │
    │    │                                                       │
    │    │   %f1 (5.0) + 10.0 ──► %f2 = 15.0                     │
    │    │                                                       │
    │    │   0f41200000 is IEEE 754 for 10.0:                    │
    │    │   0 10000010 01000000000000000000000                  │
    │    │   │ ├──────┘ └──────────────────────┘                 │
    │    │   │ exp=130    mantissa=1.25                          │
    │    │   sign=+                                              │
    │    │   = +1 × 2^(130-127) × 1.25 = 8 × 1.25 = 10.0         │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    ┌─ PHASE 6: STORE TO out[3] ─────────────────────────────────┐
    │                                                            │
    │ 10 │ add.s64  %rd7, %rd3, %rd5                             │
    │    │                                                       │
    │    │   %rd3 (0x8F00) + %rd5 (12) ──► %rd7 = 0x8F0C         │
    │    │                                                       │
    │ 11 │ st.global.f32  [%rd7], %f2                            │
    │    │                                                       │
    │    │   %f2 (15.0) ───► GLOBAL MEMORY                       │
    │    │                   ┌──────────────────────────┐        │
    │    │                   │  addr 0x8F0C  ═  out[3]  │        │
    │    │                   │  value: 15.0              │       │
    │    │                   └──────────────────────────┘        │
    │    │                                                       │
    └────────────────────────────────────────────────────────────┘
    12 │ ret;
═══════════════════════════════════════════════════════════════════
    FINAL REGISTER STATE (thread 3)
═══════════════════════════════════════════════════════════════════
    64-bit address regs        32-bit int regs    32-bit float regs
    ┌──────────────────┐       ┌──────────────┐   ┌──────────────┐
    │ %rd1 = 0x7F00    │       │ %r1  = 3     │   │ %f1 = 5.0    │
    │ %rd2 = 0x8F00    │       └──────────────┘   │ %f2 = 15.0   │
    │ %rd3 = 0x8F00    │                          └──────────────┘
    │ %rd4 = 0x7F00    │
    │ %rd5 = 12        │  ← shared by both load and store paths
    │ %rd6 = 0x7F0C    │  ← &a[3]
    │ %rd7 = 0x8F0C    │  ← &out[3]
    └──────────────────┘
═══════════════════════════════════════════════════════════════════
    DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════
    param_0 ─► rd1 ─► rd4 ─┐
                           ├─► rd6 ──► ld.global ──► f1 ─┐
    %tid.x ──► r1 ──► rd5 ─┤                             ├► f2 ──► st.global ──► out[3]
                           ├─► rd7 ──────────────────────┘
    param_1 ─► rd2 ─► rd3 ─┘                          ▲
                                                      │
                                                10.0 (immediate)
═══════════════════════════════════════════════════════════════════
    ALL 100 THREADS EXECUTING IN PARALLEL
═══════════════════════════════════════════════════════════════════
    Thread 0:  a[0]  + 10 ──► out[0]     (%tid.x=0,  offset=0)
    Thread 1:  a[1]  + 10 ──► out[1]     (%tid.x=1,  offset=4)
    Thread 2:  a[2]  + 10 ──► out[2]     (%tid.x=2,  offset=8)
    Thread 3:  a[3]  + 10 ──► out[3]     (%tid.x=3,  offset=12)  ◄── traced above
    Thread 4:  a[4]  + 10 ──► out[4]     (%tid.x=4,  offset=16)
       ...        ...                        ...
    Thread 99: a[99] + 10 ──► out[99]    (%tid.x=99, offset=396)   
```
Each thread has its OWN copy of all registers (r1, rd1-rd7, f1-f2). No thread reads another thread's data — zero synchronization needed.

## 5. Full PTX code
```
.visible .entry addTenKernel(float const*, float*)(
	.param .u64 addTenKernel(float const*, float*)_param_0,
	.param .u64 addTenKernel(float const*, float*)_param_1
)
{
	ld.param.u64 	%rd1, [addTenKernel(float const*, float*)_param_0];
	ld.param.u64 	%rd2, [addTenKernel(float const*, float*)_param_1];
	cvta.to.global.u64 	%rd3, %rd2;
	cvta.to.global.u64 	%rd4, %rd1;
	mov.u32 	%r1, %tid.x;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.f32 	%f1, [%rd6];
	add.f32 	%f2, %f1, 0f41200000;
	add.s64 	%rd7, %rd3, %rd5;
	st.global.f32 	[%rd7], %f2;
	ret;
}
```
### 6. Full x86-64 Host Code
```
__nv_save_fatbinhandle_for_managed_rt(void**):
        pushq   %rbp
        movq    %rsp, %rbp
        movq    %rdi, -8(%rbp)
        movq    -8(%rbp), %rax
        movq    %rax, __nv_fatbinhandle_for_managed_rt(%rip)
        nop
        popq    %rbp
        ret
____nv_dummy_param_ref(void*):
        pushq   %rbp
        movq    %rsp, %rbp
        movq    %rdi, -8(%rbp)
        movq    -8(%rbp), %rax
        movq    %rax, ____nv_dummy_param_ref(void*)::__ref(%rip)
        nop
        popq    %rbp
        ret
__cudaUnregisterBinaryUtil():
        pushq   %rbp
        movq    %rsp, %rbp
        movl    $__cudaFatCubinHandle, %edi
        call    ____nv_dummy_param_ref(void*)
        movq    __cudaFatCubinHandle(%rip), %rax
        movq    %rax, %rdi
        call    __cudaUnregisterFatBinary
        nop
        popq    %rbp
        ret
__nv_init_managed_rt_with_module(void**):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $16, %rsp
        movq    %rdi, -8(%rbp)
        movq    -8(%rbp), %rax
        movq    %rax, %rdi
        call    __cudaInitModule
        leave
        ret
fatbinData:
.quad 0x00100001ba55ed50,0x00000000000010b0,0x0000005001010002,0x0000000000000e48
.quad 0x0000000000000000,0x0000003400010007,0x0000000f00000040,0x0000000000000011
.quad 0x0000000000000000,0x0000000000000000,0x6178652f7070612f,0x0075632e656c706d
.quad 0x33010102464c457f,0x0000000000000007,0x0000007d00be0002,0x0000000000000000
.quad 0x0000000000000da0,0x00000000000009e0,0x0038004000340534,0x0001000f00400003
.quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x746d79732e006261
.quad 0x78646e68735f6261,0x666e692e766e2e00,0x2e747865742e006f,0x5464646132315a5f
.quad 0x6c656e72654b6e65,0x6e2e006650664b50,0x5f2e6f666e692e76,0x655464646132315a
.quad 0x506c656e72654b6e,0x766e2e006650664b,0x2e6465726168732e,0x5464646132315a5f
.quad 0x6c656e72654b6e65,0x6e2e006650664b50,0x6174736e6f632e76,0x32315a5f2e30746e
.quad 0x654b6e6554646461,0x50664b506c656e72,0x6e2e6c65722e0066,0x6174736e6f632e76
.quad 0x32315a5f2e30746e,0x654b6e6554646461,0x50664b506c656e72,0x67756265642e0066
.quad 0x722e00656e696c5f,0x67756265642e6c65,0x6e2e00656e696c5f,0x5f67756265645f76
.quad 0x7361735f656e696c,0x6e2e6c65722e0073,0x5f67756265645f76,0x7361735f656e696c
.quad 0x65645f766e2e0073,0x5f7874705f677562,0x2e766e2e00747874,0x706172676c6c6163
.quad 0x72702e766e2e0068,0x00657079746f746f,0x2e6c65722e766e2e,0x00006e6f69746361
.quad 0x617472747368732e,0x61747274732e0062,0x61746d79732e0062,0x61746d79732e0062
.quad 0x0078646e68735f62,0x6f666e692e766e2e,0x5f2e747865742e00,0x655464646132315a
.quad 0x506c656e72654b6e,0x766e2e006650664b,0x5a5f2e6f666e692e,0x6e65546464613231
.quad 0x4b506c656e72654b,0x2e766e2e00665066,0x5f2e646572616873,0x655464646132315a
.quad 0x506c656e72654b6e,0x65722e006650664b,0x6e6f632e766e2e6c,0x5f2e30746e617473
.quad 0x655464646132315a,0x506c656e72654b6e,0x766e2e006650664b,0x6e6174736e6f632e
.quad 0x6132315a5f2e3074,0x72654b6e65546464,0x6650664b506c656e,0x5f67756265642e00
.quad 0x65722e00656e696c,0x5f67756265642e6c,0x766e2e00656e696c,0x6c5f67756265645f
.quad 0x737361735f656e69,0x766e2e6c65722e00,0x6c5f67756265645f,0x737361735f656e69
.quad 0x6265645f766e2e00,0x745f7874705f6775,0x632e766e2e007478,0x68706172676c6c61
.quad 0x6f72702e766e2e00,0x2e00657079746f74,0x612e6c65722e766e,0x5a5f006e6f697463
.quad 0x6e65546464613231,0x4b506c656e72654b,0x0000000000665066,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x000e000300000032,0x0000000000000000
.quad 0x0000000000000000,0x000d0003000000b6,0x0000000000000000,0x0000000000000000
.quad 0x00040003000000da,0x0000000000000000,0x0000000000000000,0x00050003000000f6
.quad 0x0000000000000000,0x0000000000000000,0x0006000300000122,0x0000000000000000
.quad 0x0000000000000000,0x0009000300000134,0x0000000000000000,0x0000000000000000
.quad 0x000a000300000150,0x0000000000000000,0x0000000000000000,0x000e10120000015f
.quad 0x0000000000000000,0x00000000000000c0,0x002800020000004c,0x000a0efb01010000
.quad 0x0100000001010101,0x786500007070612f,0x75632e656c706d61,0xc506ccdcd2970100
.quad 0x0000000209000003,0x0301040000000000,0xe002010380f00107,0x01010000c8020100
.quad 0x0010000200000046,0x000a0efb01010000,0x0100000001010101,0x0000000209000000
.quad 0x0300040000000000,0x08020c0308020112,0x0301180201038101,0x027f038101180201
.quad 0x0000c80281810118,0x0000000000000101,0x69737265762e0000,0x2e00352e38206e6f
.quad 0x7320746567726174,0x64612e0032355f6d,0x69735f7373657264,0x000000343620657a
.quad 0x6c62697369762e00,0x7972746e652e2065,0x64646132315a5f20,0x656e72654b6e6554
.quad 0x00286650664b506c,0x2e206d617261702e,0x32315a5f20343675,0x654b6e6554646461
.quad 0x50664b506c656e72,0x5f6d617261705f66,0x617261702e002c30,0x5f203436752e206d
.quad 0x655464646132315a,0x506c656e72654b6e,0x7261705f6650664b,0x7b002900315f6d61
.quad 0x662e206765722e00,0x333c662509203233,0x206765722e003b3e,0x722509203233622e
.quad 0x65722e003b3e323c,0x09203436622e2067,0x003b3e383c647225,0x61702e646c000000
.quad 0x203436752e6d6172,0x5b202c3164722509,0x5464646132315a5f,0x6c656e72654b6e65
.quad 0x61705f6650664b50,0x003b5d305f6d6172,0x6d617261702e646c,0x722509203436752e
.quad 0x315a5f5b202c3264,0x4b6e655464646132,0x664b506c656e7265,0x6d617261705f6650
.quad 0x766300003b5d315f,0x6c672e6f742e6174,0x3436752e6c61626f,0x202c336472250920
.quad 0x7663003b32647225,0x6c672e6f742e6174,0x3436752e6c61626f,0x202c346472250920
.quad 0x6f6d003b31647225,0x2509203233752e76,0x64697425202c3172,0x6c756d00003b782e
.quad 0x33732e656469772e,0x2c35647225092032,0x3b34202c31722520,0x3436732e64646100
.quad 0x202c366472250920,0x7225202c34647225,0x672e646c003b3564,0x33662e6c61626f6c
.quad 0x202c316625092032,0x003b5d366472255b,0x203233662e646461,0x6625202c32662509
.quad 0x3231346630202c31,0x61003b3030303030,0x09203436732e6464,0x7225202c37647225
.quad 0x35647225202c3364,0x6f6c672e7473003b,0x203233662e6c6162,0x2c5d376472255b09
.quad 0x7200003b32662520,0x00007d00003b7465,0x0000000800082f04,0x0008120400000007
.quad 0x0000000000000008,0x0000000800081104,0x0008120400000000,0x0000000000000008
.quad 0x0000007d00043704,0x00002a0100003001,0x0000000200080a04,0x0010190300100140
.quad 0x00000000000c1704,0x0021f00000080001,0x00000000000c1704,0x0021f00000000000
.quad 0x00041c0400ff1b03,0x0000000000000078,0x00000000ffffffff,0x00000000fffffffe
.quad 0x00000000fffffffd,0x00000000fffffffc,0x0000000000000073,0x3605002511000000
.quad 0x0000000000000035,0x0000000800000002,0x000000000000001d,0x0000000800000002
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x083fc400e3e007f6,0x4c98078000870001,0xf0c8000002170000,0x3848000000270004
.quad 0x001fc840fec007f5,0x3829000001e70000,0x4c10800005070402,0x4c10080005170003
.quad 0x001fdc00fcc007b1,0xeed4200000070202,0x4c10800005270404,0x4c10080005370005
.quad 0x001ffc00fe2107f2,0x3858004120070206,0xeedc200000070406,0xe30000000007000f
.quad 0x001f8000fc0007ff,0xe2400fffff07000f,0x50b0000000070f00,0x50b0000000070f00
.quad 0x001f8000fc0007e0,0x50b0000000070f00,0x50b0000000070f00,0x50b0000000070f00
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
.quad 0x0000000300000001,0x0000000000000000,0x0000000000000000,0x0000000000000040
.quad 0x000000000000015f,0x0000000000000000,0x0000000000000001,0x0000000000000000
.quad 0x000000030000000b,0x0000000000000000,0x0000000000000000,0x000000000000019f
.quad 0x0000000000000175,0x0000000000000000,0x0000000000000001,0x0000000000000000
.quad 0x0000000200000013,0x0000000000000000,0x0000000000000000,0x0000000000000318
.quad 0x00000000000000d8,0x0000000800000002,0x0000000000000008,0x0000000000000018
.quad 0x00000001000000da,0x0000000000000000,0x0000000000000000,0x00000000000003f0
.quad 0x0000000000000050,0x0000000000000000,0x0000000000000001,0x0000000000000000
.quad 0x00000001000000f6,0x0000000000000000,0x0000000000000000,0x0000000000000440
.quad 0x000000000000004a,0x0000000000000000,0x0000000000000001,0x0000000000000000
.quad 0x0000000100000122,0x0000000000000000,0x0000000000000000,0x000000000000048a
.quad 0x0000000000000265,0x0000000000000000,0x0000000000000001,0x0000000000000000
.quad 0x7000000000000029,0x0000000000000000,0x0000000000000000,0x00000000000006f0
.quad 0x0000000000000030,0x0000000000000003,0x0000000000000004,0x0000000000000000
.quad 0x700000000000004e,0x0000000000000040,0x0000000000000000,0x0000000000000720
.quad 0x000000000000004c,0x0000000e00000003,0x0000000000000004,0x0000000000000000
.quad 0x7000000100000134,0x0000000000000000,0x0000000000000000,0x000000000000076c
.quad 0x0000000000000020,0x0000000000000003,0x0000000000000004,0x0000000000000008
.quad 0x7000000b00000150,0x0000000000000000,0x0000000000000000,0x0000000000000790
.quad 0x0000000000000010,0x0000000000000000,0x0000000000000008,0x0000000000000008
.quad 0x00000009000000e6,0x0000000000000040,0x0000000000000000,0x00000000000007a0
.quad 0x0000000000000010,0x0000000400000003,0x0000000000000008,0x0000000000000010
.quad 0x000000090000010a,0x0000000000000040,0x0000000000000000,0x00000000000007b0
.quad 0x0000000000000010,0x0000000500000003,0x0000000000000008,0x0000000000000010
.quad 0x000000010000008e,0x0000000000000042,0x0000000000000000,0x00000000000007c0
.quad 0x0000000000000150,0x0000000e00000000,0x0000000000000004,0x0000000000000000
.quad 0x0000000100000032,0x0000000000000006,0x0000000000000000,0x0000000000000920
.quad 0x00000000000000c0,0x0700000800000003,0x0000000000000020,0x0000000000000000
.quad 0x0000000500000006,0x0000000000000da0,0x0000000000000000,0x0000000000000000
.quad 0x00000000000000a8,0x00000000000000a8,0x0000000000000008,0x0000000500000001
.quad 0x00000000000007c0,0x0000000000000000,0x0000000000000000,0x0000000000000220
.quad 0x0000000000000220,0x0000000000000008,0x0000000500000001,0x0000000000000da0
.quad 0x0000000000000000,0x0000000000000000,0x00000000000000a8,0x00000000000000a8
.quad 0x0000000000000008,0x0000007001010001,0x00000000000001a8,0x00000050000001a2
.quad 0x0000003400080005,0x0000000f00000040,0x0000000000002011,0x0000000000000000
.quad 0x00000000000002ab,0x6178652f7070612f,0x0075632e656c706d,0x0000001600000058
.quad 0x72656e65672d2d20,0x656e696c2d657461,0x0000206f666e692d,0x1ef300032f2f0a3c
.quad 0x6f69737265762e0a,0x742e0a352e38206e,0x6d73207465677261,0x6464612e0a32355f
.quad 0x7a69735f73736572,0xff00310a34362065,0x20656c6269736921,0x5f207972746e652e
.quad 0x655464646132315a,0x506c656e72654b6e,0x702e0a286650664b,0x36752e206d617261
.quad 0x00215f1103002334,0xf316002b2c305f3f,0x2e0a7b0a290a3107,0x3233662e20676572
.quad 0x113b3e333c662520,0x3c72360011621000,0x2520343680001132,0x6ce20012383c6472
.quad 0x302038203109636f,0x220071646c0a0a0a,0x202c314f0022752e,0x343b5d2d0a00a25b
.quad 0x31250c0034321f00,0x33203903f400755d,0x6f742e617476630a,0x456c61626f6c672e
.quad 0x3b1f004b2c332100,0x81001f341104001f,0x752e766f6d0a3b31,0x7425202c319500df
.quad 0x00f1005e782e6469,0x6c756d0a33203031,0x26732e656469772e,0x82002c2c35643200
.quad 0x732e6464610a3b34,0x1100562c36260050,0x4c0200700300d335,0x1200260001070001
.quad 0x322200150200355d,0x3134663060001a2c,0x2600530900013032,0x0a3b355800c82c37
.quad 0x20004e0000537473,0x00af321600415d37,0x65720a31203119f0,0x2e0a0a7d0a0a3b74
.quad 0x22203109656c6966,0x6178652f7070612f,0x2275632e656c706d,0x000000000000000a

__fatDeviceText:
        .long   1180844977
        .long   1
        .quad   fatbinData
        .quad   0
__device_stub__Z12addTenKernelPKfPf(float const*, float*):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $96, %rsp
        movq    %rdi, -88(%rbp)
        movq    %rsi, -96(%rbp)
        movl    $0, -4(%rbp)
        movl    -4(%rbp), %eax
        cltq
        leaq    -88(%rbp), %rdx
        movq    %rdx, -32(%rbp,%rax,8)
        addl    $1, -4(%rbp)
        movl    -4(%rbp), %eax
        cltq
        leaq    -96(%rbp), %rdx
        movq    %rdx, -32(%rbp,%rax,8)
        addl    $1, -4(%rbp)
        movq    $addTenKernel(float const*, float*), __device_stub__Z12addTenKernelPKfPf(float const*, float*)::__f(%rip)
        movl    $1, -44(%rbp)
        movl    $1, -40(%rbp)
        movl    $1, -36(%rbp)
        movl    $1, -56(%rbp)
        movl    $1, -52(%rbp)
        movl    $1, -48(%rbp)
        leaq    -72(%rbp), %rcx
        leaq    -64(%rbp), %rdx
        leaq    -56(%rbp), %rsi
        leaq    -44(%rbp), %rax
        movq    %rax, %rdi
        call    __cudaPopCallConfiguration
        testl   %eax, %eax
        setne   %al
        testb   %al, %al
        jne     .L6
        cmpl    $0, -4(%rbp)
        jne     .L9
        movq    -72(%rbp), %rdi
        movq    -64(%rbp), %rsi
        leaq    -32(%rbp), %rdx
        movl    -4(%rbp), %eax
        cltq
        salq    $3, %rax
        leaq    (%rdx,%rax), %r9
        movq    -56(%rbp), %rcx
        movl    -48(%rbp), %r8d
        movq    -44(%rbp), %rdx
        movl    -36(%rbp), %eax
        pushq   %rdi
        pushq   %rsi
        movq    %rdx, %rsi
        movl    %eax, %edx
        movl    $addTenKernel(float const*, float*), %edi
        call    cudaError cudaLaunchKernel<char>(char const*, dim3, dim3, void**, unsigned long, CUstream_st*)
        addq    $16, %rsp
        jmp     .L6
.L9:
        movq    -72(%rbp), %rdi
        movq    -64(%rbp), %rsi
        leaq    -32(%rbp), %r9
        movq    -56(%rbp), %rcx
        movl    -48(%rbp), %r8d
        movq    -44(%rbp), %rdx
        movl    -36(%rbp), %eax
        pushq   %rdi
        pushq   %rsi
        movq    %rdx, %rsi
        movl    %eax, %edx
        movl    $addTenKernel(float const*, float*), %edi
        call    cudaError cudaLaunchKernel<char>(char const*, dim3, dim3, void**, unsigned long, CUstream_st*)
        addq    $16, %rsp
.L6:
        leave
        ret
addTenKernel(float const*, float*):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $16, %rsp
        movq    %rdi, -8(%rbp)
        movq    %rsi, -16(%rbp)
        movq    -16(%rbp), %rdx
        movq    -8(%rbp), %rax
        movq    %rdx, %rsi
        movq    %rax, %rdi
        call    __device_stub__Z12addTenKernelPKfPf(float const*, float*)
        nop
        leave
        ret
.LC0:
        .string "addTenKernel(float const*, float*)"
__nv_cudaEntityRegisterCallback(void**):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $16, %rsp
        movq    %rdi, -8(%rbp)
        movq    -8(%rbp), %rax
        movq    %rax, __nv_cudaEntityRegisterCallback(void**)::__ref(%rip)
        movq    -8(%rbp), %rax
        movq    %rax, %rdi
        call    __nv_save_fatbinhandle_for_managed_rt(void**)
        movq    -8(%rbp), %rax
        pushq   $0
        pushq   $0
        pushq   $0
        pushq   $0
        movl    $0, %r9d
        movl    $-1, %r8d
        movl    $.LC0, %ecx
        movl    $.LC0, %edx
        movl    $addTenKernel(float const*, float*), %esi
        movq    %rax, %rdi
        call    __cudaRegisterFunction
        addq    $32, %rsp
        nop
        leave
        ret
__sti____cudaRegisterAll():
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $16, %rsp
        movl    $__fatDeviceText, %edi
        call    __cudaRegisterFatBinary
        movq    %rax, __cudaFatCubinHandle(%rip)
        movq    $__nv_cudaEntityRegisterCallback(void**), -8(%rbp)
        movq    __cudaFatCubinHandle(%rip), %rax
        movq    -8(%rbp), %rdx
        movq    %rax, %rdi
        call    *%rdx
        movq    __cudaFatCubinHandle(%rip), %rax
        movq    %rax, %rdi
        call    __cudaRegisterFatBinaryEnd
        movl    $__cudaUnregisterBinaryUtil(), %edi
        call    atexit
        nop
        leave
        ret
cudaError cudaLaunchKernel<char>(char const*, dim3, dim3, void**, unsigned long, CUstream_st*):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp
        movq    %rdi, -8(%rbp)
        movq    %rcx, %rax
        movl    %r8d, %ecx
        movq    %r9, -48(%rbp)
        movq    %rsi, -24(%rbp)
        movl    %edx, -16(%rbp)
        movq    %rax, -40(%rbp)
        movl    %ecx, -32(%rbp)
        movq    -48(%rbp), %r8
        movq    -40(%rbp), %rcx
        movl    -32(%rbp), %edi
        movq    -24(%rbp), %rsi
        movl    -16(%rbp), %edx
        movq    -8(%rbp), %rax
        pushq   24(%rbp)
        pushq   16(%rbp)
        movq    %r8, %r9
        movl    %edi, %r8d
        movq    %rax, %rdi
        call    cudaLaunchKernel
        addq    $16, %rsp
        leave
        ret
```
