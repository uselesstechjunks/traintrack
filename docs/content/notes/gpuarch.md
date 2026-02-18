# Compute FLOPs from #multiply-add operations.

1. Start with **throughput per SM per clock** expressed as **“multiply-add operations / clock / SM”** (from NVIDIA table/figure).
2. Convert multiply-adds to **floating-point operations (FLOPs)** by deciding how many FLOPs a multiply-add counts as.
3. Scale up by **#SMs** and **clock frequency** to get **FLOPs/second**.

NVIDIA’s convention is:

* **1 multiply-add (i.e., fused multiply-add, FMA)** = **2 FLOPs**
  (one multiply + one add). ([NVIDIA Docs][1])

So the conversion is:

$$\text{FLOPs/clock/SM} = 2 \times (\text{multiply-adds/clock/SM})$$

Then throughput for the whole GPU:

$$\text{FLOPs/s} = 2 \times (\text{multiply-adds/clock/SM}) \times (\\#\text{SMs}) \times (\text{clock in cycles/s})$$

And to express it in TFLOPs:

$$\text{TFLOPs} = \frac{\text{FLOPs/s}}{10^{12}}$$

## Working the A100 example backwards

A100: **108 SMs**, **1.41 GHz**, peak dense: **156 TF32 TFLOPs** and **312 FP16 TFLOPs**. ([NVIDIA Docs][1])

Let’s solve for the implied multiply-adds/clock/SM.

### TF32

$$
\text{MA/clock/SM} =
\frac{156\times 10^{12}}{2 \times 108 \times 1.41\times 10^{9}}
\approx 512
$$

So the table’s TF32 entry for A100 is effectively **512 multiply-adds per clock per SM** (dense).

Check the forward direction:

* FLOPs/clock/SM = 2 × 512 = 1024 FLOPs/clock/SM
* FLOPs/s = 1024 × 108 × 1.41e9 ≈ 156e12 = **156 TFLOPs**

### FP16

$$
\text{MA/clock/SM} =
\frac{312\times 10^{12}}{2 \times 108 \times 1.41\times 10^{9}}
\approx 1024
$$

So FP16 is **1024 multiply-adds per clock per SM** (dense), which matches how A100 Tensor Core dense FP16 FMA throughput is commonly described per SM. ([NVIDIA Images][2])

## Common gotchas

* **Dense vs sparse:** Many NVIDIA peak numbers have a “with sparsity” variant that is often 2× for Tensor Core MMA. The paragraph you quoted is explicitly about **peak dense**. ([servicedesk.surf.nl][3])
* **“FMA = 2 FLOPs” is a convention:** Some communities/tools count an FMA as 1 “operation” for different reasons; NVIDIA is being explicit that *for FLOP counting* they use 2 here. ([NVIDIA Docs][1])
* **Clock to use:** They used **SM clock rate** (1.41 GHz in the example), not memory clock, not “boost” marketing max, etc. In practice the instantaneous SM clock can vary with power/thermal limits, so achieved TFLOPs can be lower/higher depending on conditions. ([NVIDIA Docs][1])

[1]: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html?utm_source=chatgpt.com "GPU Performance Background User's Guide"
[2]: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf?utm_source=chatgpt.com "NVIDIA A100 Tensor Core GPU Architecture"
[3]: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30668820/Deep%2BLearning%2Bon%2BA100%2BGPUs?utm_source=chatgpt.com "Deep Learning on A100 GPUs - SURF User Knowledge Base"
