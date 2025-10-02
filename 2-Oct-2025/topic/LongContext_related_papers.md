# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 

**Authors**: Alessio Devoto, Maximilian Jeblick, Simon Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00636)  

**Abstract**: Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce $\textbf{Expected Attention, a training-free compression method}$ that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, $\textbf{we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques}$. 

---
# ACON: Optimizing Context Compression for Long-horizon LLM Agents 

**Authors**: Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00615)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. 

---
# SecureBERT 2.0: Advanced Language Model for Cybersecurity Intelligence 

**Authors**: Ehsan Aghaei, Sarthak Jain, Prashanth Arun, Arjun Sambamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.00240)  

**Abstract**: Effective analysis of cybersecurity and threat intelligence data demands language models that can interpret specialized terminology, complex document structures, and the interdependence of natural language and source code. Encoder-only transformer architectures provide efficient and robust representations that support critical tasks such as semantic search, technical entity extraction, and semantic analysis, which are key to automated threat detection, incident triage, and vulnerability assessment. However, general-purpose language models often lack the domain-specific adaptation required for high precision. We present SecureBERT 2.0, an enhanced encoder-only language model purpose-built for cybersecurity applications. Leveraging the ModernBERT architecture, SecureBERT 2.0 introduces improved long-context modeling and hierarchical encoding, enabling effective processing of extended and heterogeneous documents, including threat reports and source code artifacts. Pretrained on a domain-specific corpus more than thirteen times larger than its predecessor, comprising over 13 billion text tokens and 53 million code tokens from diverse real-world sources, SecureBERT 2.0 achieves state-of-the-art performance on multiple cybersecurity benchmarks. Experimental results demonstrate substantial improvements in semantic search for threat intelligence, semantic analysis, cybersecurity-specific named entity recognition, and automated vulnerability detection in code within the cybersecurity domain. 

---
# Rethinking RoPE Scaling in Quantized LLM: Theory, Outlier, and Channel-Band Analysis with Weight Rescaling 

**Authors**: Ye Qiao, Haocheng Xu, Xiaofan Zhang, Sitao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00028)  

**Abstract**: Extending the context window support of large language models (LLMs) is crucial for tasks with long-distance dependencies. RoPE-based interpolation and extrapolation methods, such as linear scaling and frequency-aware schemes, enable longer input length support without retraining, while post-training quantization (PTQ) makes deployment practical. However, we show that combining RoPE position interpolation (PI) with PTQ degrades accuracy due to coupled effects including long-context aliasing, dynamic-range dilation, anisotropy from axis-aligned quantizers vs. rotated RoPE pairs, and outlier shifting that produces position-dependent logit noise. We provide, to the best of our knowledge, the first systematic analysis of the PI+PTQ approach and introduce two practical diagnostics: interpolation pressure (per-band sensitivity to phase scaling) and tail-inflation ratios (outlier shift from short to long contexts). Following the analysis results, we propose Q-ROAR (Quantization, RoPE-interpolation, and Outlier Aware Rescaling), a weight-only, interpolation-aware stabilization of PI for quantized LLMs. Q-ROAR groups RoPE dimensions into a small number of frequency bands and performs a lightweight search over per-band scales for Key and Query weights (with an optional symmetric variant to preserve logit scale). The search is guided by our diagnostics and uses a tiny long-context development dataset, requiring no fine-tuning to the model, no architecture or kernel changes, and no additional deployment overhead. Empirically, Q-ROAR reduces the model's perplexity on long-context workloads by more than 14%, while preserving short-context performance, inference throughput, and compatibility with existing LLM system stacks. 

---
# LongCodeZip: Compress Long Context for Code Language Models 

**Authors**: Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00446)  

**Abstract**: Code generation under long contexts is becoming increasingly critical as Large Language Models (LLMs) are required to reason over extensive information in the codebase. While recent advances enable code LLMs to process long inputs, high API costs and generation latency remain substantial bottlenecks. Existing context pruning techniques, such as LLMLingua, achieve promising results for general text but overlook code-specific structures and dependencies, leading to suboptimal performance in programming tasks. In this paper, we propose LongCodeZip, a novel plug-and-play code compression framework designed specifically for code LLMs. LongCodeZip employs a dual-stage strategy: (1) coarse-grained compression, which identifies and ranks function-level chunks using conditional perplexity with respect to the instruction, retaining only the most relevant functions; and (2) fine-grained compression, which segments retained functions into blocks based on perplexity and selects an optimal subset under an adaptive token budget to maximize relevance. Evaluations across multiple tasks, including code completion, summarization, and question answering, show that LongCodeZip consistently outperforms baseline methods, achieving up to a 5.6x compression ratio without degrading task performance. By effectively reducing context size while preserving essential information, LongCodeZip enables LLMs to better scale to real-world, large-scale code scenarios, advancing the efficiency and capability of code intelligence applications. 

---
