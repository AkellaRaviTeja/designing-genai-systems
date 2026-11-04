| Quantity | Value | How |
| :-- | --: | :-- |
| Weights in memory | 65.2 GB | checkpoint size; Hopper keeps MXFP4 experts packed and multiplies in BF16 |
| Read per decode step, dense part | 3.1 GB | attention, router and lm_head in BF16 |
| Expert weights | 61 GB | 128 experts x 36 layers, 13.2 MB each |
| Prefill ceiling per H100 | 88K tokens/s | 989 TFLOPS BF16 / 11.3 GFLOP per token at 7,000 tokens |
| Decode ceiling, 32 streams of 3,000 | 2,400 tokens/s | 3.35 TB/s; 64% of experts touched per step |
| Decode ceiling, 76 streams of 7,000 | 3,700 tokens/s | the batch an FP8 KV cache fits in 10 GB |
| Decode ceiling, 38 streams, BF16 KV | 2,300 tokens/s | the batch a BF16 KV cache fits in 10 GB |
