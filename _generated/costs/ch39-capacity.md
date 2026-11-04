| Quantity | Value | How |
| :-- | --: | :-- |
| KV cache per token | 72 KB | 2 x 36 layers x 8 KV heads x 64 dims x 2 bytes |
| Prefill per H100 | 24K tokens/s | 25% of 989 TFLOPS / (2 x 5.1B active params) |
| Decode per H100, batch 32 | 1,600 tokens/s | 32 streams, 3,000-token contexts, 70% of 3.35 TB/s |
| Speed per interactive stream | 50 tokens/s | one token per decode step |
| Decode per H100, batch 128 | 3,100 tokens/s | 128 streams, 4,000-token contexts |
