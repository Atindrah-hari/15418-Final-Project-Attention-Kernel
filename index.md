# 15418 Attention Kernel Project
Raymond He, Atindrah Harishankar

**URL:** https://atindrah-hari.github.io/15418-Final-Project-Attention-Kernel/

## Project Summary 
We will implement and optimize a custom CUDA kernel for the backward pass of the masked “queries attend to nodes” cross-attention in a MaskFormer-style transformer, where 512 to 1536 queries attend to 32k to 65k nodes with a head dimension of 8. FlashAttention cannot be used for this layer because it does not support our dynamic attention masks (it is used only in the encoder), so the decoder runs PyTorch’s memory-efficient attention kernel. Our baselines are PyTorch’s memory efficient kernel and unfused math attention, which we will compare across  different problem sizes and across several GPUs (RTX 4090, PSC GPUs, and a research-lab GPU), analyzing what limits performance on each.


## Reports
- [Proposal](Proposal.docx.pdf)
- Milestone Report (not finished)
- Final Report (not finished)

## Schedule
