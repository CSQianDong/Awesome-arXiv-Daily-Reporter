# ADEMA: A Knowledge-State Orchestration Architecture for Long-Horizon Knowledge Synthesis with LLMAgents 

**Authors**: Zhou Hanlin, Chan Huah Yong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25849)  

**Abstract**: Long-horizon LLM tasks often fail not because a single answer is unattainable, but because knowledge states drift across rounds, intermediate commitments remain implicit, and interruption fractures the evolving evidence chain. This paper presents ADEMA as a knowledge-state orchestration architecture for long-horizon knowledge synthesis rather than as a generic multi-agent runtime. The architecture combines explicit epistemic bookkeeping, heterogeneous dual-evaluator governance, adaptive task-mode switching, reputation-shaped resource allocation, checkpoint-resumable persistence, segment-level memory condensation, artifact-first assembly, and final-validity checking with safe fallback. Evidence is drawn entirely from existing materials: a four-scenario showcase package, a fixed 60-run mechanism matrix, targeted micro-ablation and artifact-chain supplements, and a repaired protocol-level benchmark in which code-oriented evaluation is the clearest quality-sensitive mechanism block. Across the fixed matrix, removing checkpoint/resume produced the only invalid run, and it did so in the interruption-sensitive resume condition. By contrast, dual evaluation, segment synthesis, and dynamic governance are best interpreted as supporting control mechanisms that shape trajectory discipline, explicit artifact progression, and cost-quality behavior rather than as universal binary prerequisites for completion. The contribution is therefore a knowledge-state orchestration architecture in which explicit epistemic state transition, evidence-bearing artifact progression, and recoverable continuity are the primary design commitments. 

---
# Salca: A Sparsity-Aware Hardware Accelerator for Efficient Long-Context Attention Decoding 

**Authors**: Wang Fan, Wei Cao, Xi Zha, Kedi Ma, MingQian Sun, Jialin Chen, Fengzhe Zhang, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24820)  

**Abstract**: Long contexts improve capabilities of large language models but pose serious hardware challenges: compute and memory footprints grow linearly with sequence length. Particularly, the decoding phase continuously accesses massive KV cache, dramatically increasing bandwidth and computing pressure. Existing accelerators are primarily designed and evaluated for short contexts. They suffer from significant performance degradation when processing long contexts. To bridge this gap, we identify the major bottleneck and present a hardware accelerator for long context attention decoding via hardware-software co-design. On the software side, we propose dual-compression dynamic sparse attention. It combines ultra-low-precision quantization with feature sparsity to minimize prediction overhead. A hardware-friendly approximate Top-K selection further reduces filter complexity from $O(n \log k)$ to $O(n)$. On the hardware side, we deeply optimize compute and memory access to tackle bottlenecks from intricate interplay between sparse attention and long contexts, and establish a performance model to derive the optimal co-design scheme. The resulting hardware adopts a fully pipelined parallel architecture and achieves $O(n)$ efficiency even for long sequences. Experiments show that our design delivers $3.82\times$ speedup and $74.19\times$ energy efficiency over A100. Compared to SOTA accelerators, this is the first ASIC accelerator that efficiently supports long context inference, with at least $3.5\times$ higher throughput and $2.08\times$ better energy efficiency. 

---
# Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model 

**Authors**: Maixent Chenebaux  

**Link**: [PDF](https://arxiv.org/pdf/2604.24809)  

**Abstract**: We present Nautile-370M, a 371-million-parameter small language model designed for efficient reasoning under strict parameter and inference budgets. Nautile-370M uses a hybrid backbone in which two SeqCond Attention (SCA) layers, a linear-time spectral sequence operator inspired by SeqCondenser, alternate with one transformer layer. This design aims to retain the long-context efficiency and state-tracking benefits of structured sequential models while preserving the expressive token-to-token routing of attention. The model was trained on a single Cloud TPU v4-64 pod slice provided through the Google TPU Research Cloud (TRC) program; the subsequent reinforcement learning stage was carried out on a single NVIDIA DGX Spark. We prove that the SCA readout mechanism can exactly retrieve any individual token from the prefix summary and can reproduce any output of softmax attention as a special case, establishing that SCA is at least as expressive as full self-attention in the continuous limit. We also describe the training data pipeline and outline a reinforcement learning stage specialized for reasoning, verification, and response quality. 

---
# Barriers to Universal Reasoning With Transformers (And How to Overcome Them) 

**Authors**: Oliver Kraus, Yash Sarrof, Yuekun Yao, Alexander Koller, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.25800)  

**Abstract**: Chain-of-Thought (CoT) has been shown to empirically improve Transformers' performance, and theoretically increase their expressivity to Turing completeness. However, whether Transformers can learn to generalize to CoT traces longer than those seen during training is understudied. We use recent theoretical frameworks for Transformer length generalization and find that -- under standard positional encodings and a finite alphabet -- Transformers with CoT cannot solve problems beyond $TC^0$, i.e. the expressivity benefits do not hold under the stricter requirement of length-generalizable learnability. However, if we allow the vocabulary to grow with problem size, we attain a length-generalizable simulation of Turing machines where the CoT trace length is linear in the simulated runtime up to a constant. Our construction overcomes two core obstacles to reliable length generalization: repeated copying and last-occurrence retrieval. We assign each tape position a unique signpost token, and log only value changes to enable recovery of the current tape symbol through counts circumventing both barriers. Further, we empirically show that the use of such signpost tokens and value change encodings provide actionable guidance to improve length generalization on hard problems. 

---
# PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference 

**Authors**: Ishan Patel, Ishan Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24971)  

**Abstract**: We present PolyKV, a system in which multiple concurrent inference agents share a single, asymmetrically compressed KV cache pool. Rather than allocating a separate KV cache per agent -- the standard paradigm -- PolyKV writes a compressed cache once and injects it into N independent agent contexts via HuggingFace DynamicCache objects. Compression is asymmetric: Keys are quantized at int8 (q8_0) to preserve softmax stability, while Values are compressed using TurboQuant MSE -- a Fast Walsh-Hadamard Transform (FWHT) rotation followed by 3-bit Lloyd-Max quantization with centroids tuned to N(0,1). We evaluate across two model scales (SmolLM2-1.7B-Instruct and Llama-3-8B-Instruct), three context lengths (600-7,194 tokens), and up to 15 concurrent agents. PolyKV achieves a stable 2.91x compression ratio across all configurations. On Llama-3-8B with 15 agents sharing a 4K-token context, PolyKV reduces KV cache memory from 19.8 GB to 0.45 GB -- a 97.7% reduction -- while maintaining only +0.57% perplexity degradation and a mean BERTScore F1 of 0.928. PPL delta does not grow with agent count and improves as context length increases, inverting to -0.26% at 1,851 coherent tokens. To our knowledge, no prior work combines a single shared, lossy-compressed KV pool with multi-reader concurrent agent access. 

---
