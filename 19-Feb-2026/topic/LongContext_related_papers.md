# Reinforced Fast Weights with Next-Sequence Prediction 

**Authors**: Hee Seung Hwang, Xindi Wu, Sanghyuk Chun, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.16704)  

**Abstract**: Fast weight architectures offer a promising alternative to attention-based transformers for long-context modeling by maintaining constant memory overhead regardless of context length. However, their potential is limited by the next-token prediction (NTP) training paradigm. NTP optimizes single-token predictions and ignores semantic coherence across multiple tokens following a prefix. Consequently, fast weight models, which dynamically update their parameters to store contextual information, learn suboptimal representations that fail to capture long-range dependencies. We introduce REFINE (Reinforced Fast weIghts with Next sEquence prediction), a reinforcement learning framework that trains fast weight models under the next-sequence prediction (NSP) objective. REFINE selects informative token positions based on prediction entropy, generates multi-token rollouts, assigns self-supervised sequence-level rewards, and optimizes the model with group relative policy optimization (GRPO). REFINE is applicable throughout the training lifecycle of pre-trained language models: mid-training, post-training, and test-time training. Our experiments on LaCT-760M and DeltaNet-1.3B demonstrate that REFINE consistently outperforms supervised fine-tuning with NTP across needle-in-a-haystack retrieval, long-context question answering, and diverse tasks in LongBench. REFINE provides an effective and versatile framework for improving long-context modeling in fast weight architectures. 

---
# Doc-to-LoRA: Learning to Instantly Internalize Contexts 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2602.15902)  

**Abstract**: Long input sequences are central to in-context learning, document understanding, and multi-step reasoning of Large Language Models (LLMs). However, the quadratic attention cost of Transformers makes inference memory-intensive and slow. While context distillation (CD) can transfer information into model parameters, per-prompt distillation is impractical due to training costs and latency. To address these limitations, we propose Doc-to-LoRA (D2L), a lightweight hypernetwork that meta-learns to perform approximate CD within a single forward pass. Given an unseen prompt, D2L generates a LoRA adapter for a target LLM, enabling subsequent queries to be answered without re-consuming the original context, reducing latency and KV-cache memory consumption during inference of the target LLM. On a long-context needle-in-a-haystack task, D2L successfully learns to map contexts into adapters that store the needle information, achieving near-perfect zero-shot accuracy at sequence lengths exceeding the target LLM's native context window by more than 4x. On real-world QA datasets with limited compute, D2L outperforms standard CD while significantly reducing peak memory consumption and update latency. We envision that D2L can facilitate rapid adaptation of LLMs, opening up the possibility of frequent knowledge updates and personalized chat behavior. 

---
# CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill 

**Authors**: Bradley McDanel, Steven Li, Harshit Khaitan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16054)  

**Abstract**: The prefill stage in long-context LLM inference remains a computational bottleneck. Recent token-ranking heuristics accelerate inference by selectively processing a subset of semantically relevant tokens. However, existing methods suffer from unstable token importance estimation, often varying between layers. Evaluating token-ranking quality independently from heuristic-specific architectures is challenging. To address this, we introduce an Answer-Informed Oracle, which defines ground-truth token importance by measuring attention from generated answers back to the prompt. This oracle reveals that existing heuristics exhibit high variance across layers: rankings can degrade sharply at specific layers, a failure mode invisible to end-to-end benchmarks. The diagnosis suggests a simple fix: aggregate scores across layers rather than relying on any single one. We implement this as Cross-Layer Attention Aggregation (CLAA), which closes the gap to the oracle upper bound and reduces Time-to-First-Token (TTFT) by up to 39\% compared to the Full KV Cache baseline. 

---
# Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective 

**Authors**: Yunhao Liu, Zian Jia, Xinyu Gao, Kanjun Xu, Yun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.15856)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively grounds Large Language Models (LLMs) with external knowledge and is widely applied to Web-related tasks. However, its scalability is hindered by excessive context length and redundant retrievals. Recent research on soft context compression aims to address this by encoding long documents into compact embeddings, yet they often underperform non-compressed RAG due to their reliance on auto-encoder-like full-compression that forces the encoder to compress all document information regardless of relevance to the input query.
In this work, we conduct an analysis on this paradigm and reveal two fundamental limitations: (I) Infeasibility, full-compression conflicts with the LLM's downstream generation behavior; and (II) Non-necessity: full-compression is unnecessary and dilutes task-relevant information density. Motivated by these insights, we introduce SeleCom, a selector-based soft compression framework for RAG that redefines the encoder's role as query-conditioned information selector. The selector is decoder-only and is trained with a massive, diverse and difficulty-graded synthetic QA dataset with curriculum learning.
Extensive experiments show that SeleCom significantly outperforms existing soft compression approaches and achieves competitive or superior performance to non-compression baselines, while reducing computation and latency by 33.8%~84.6%. 

---
