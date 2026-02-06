# LongR: Unleashing Long-Context Reasoning via Reinforcement Learning with Dense Utility Rewards 

**Authors**: Bowen Ping, Zijun Chen, Yiyao Yu, Tingfeng Hui, Junchi Yan, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05758)  

**Abstract**: Reinforcement Learning has emerged as a key driver for LLM reasoning. This capability is equally pivotal in long-context scenarios--such as long-dialogue understanding and structured data analysis, where the challenge extends beyond consuming tokens to performing rigorous deduction. While existing efforts focus on data synthesis or architectural changes, recent work points out that relying solely on sparse, outcome-only rewards yields limited gains, as such coarse signals are often insufficient to effectively guide the complex long-context reasoning. To address this, we propose LongR, a unified framework that enhances long-context performance by integrating a dynamic "Think-and-Read" mechanism, which interleaves reasoning with document consultation, with a contextual density reward based on relative information gain to quantify the utility of the relevant documents. Empirically, LongR achieves a 9% gain on LongBench v2 and consistent improvements on RULER and InfiniteBench, demonstrating robust efficiency in navigating extensive contexts. Furthermore, LongR consistently enhances performance across diverse RL algorithms (e.g., DAPO, GSPO). Finally, we conduct in-depth analyses to investigate the impact of reasoning chain length on efficiency and the model's robustness against distractors. 

---
# RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference 

**Authors**: Siran Liu, Guoxia Wang, Sa Wang, Jinle Zeng, HaoYang Xie, Siyu Lou, JiaBin Yang, DianHai Yu, Haifeng Wang, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05853)  

**Abstract**: The quadratic complexity of attention mechanisms poses a critical bottleneck for large language models processing long contexts. While dynamic sparse attention methods offer input-adaptive efficiency, they face fundamental trade-offs: requiring preprocessing, lacking global evaluation, violating query independence, or incurring high computational overhead. We present RRAttention, a novel dynamic sparse attention method that simultaneously achieves all desirable properties through a head \underline{r}ound-\underline{r}obin (RR) sampling strategy. By rotating query sampling positions across attention heads within each stride, RRAttention maintains query independence while enabling efficient global pattern discovery with stride-level aggregation. Our method reduces complexity from $O(L^2)$ to $O(L^2/S^2)$ and employs adaptive Top-$\tau$ selection for optimal sparsity. Extensive experiments on natural language understanding (HELMET) and multimodal video comprehension (Video-MME) demonstrate that RRAttention recovers over 99\% of full attention performance while computing only half of the attention blocks, achieving 2.4$\times$ speedup at 128K context length and outperforming existing dynamic sparse attention methods. 

---
# CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs 

**Authors**: Haoran Li, Sucheng Ren, Alan Yuille, Feng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05258)  

**Abstract**: Rotary Positional Embedding (RoPE) is a key component of context scaling in Large Language Models (LLMs). While various methods have been proposed to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: (1) out-of-distribution (OOD) mitigation, which scales RoPE frequencies to accommodate unseen positions, and (2) Semantic Modeling, which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. In this work, we unify these seemingly distinct objectives through a minimalist intervention, namely CoPE: soft clipping lowfrequency components of RoPE. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping. Extensive experiments demonstrate that simply applying our soft clipping strategy to RoPE yields significant performance gains that scale up to 256k context length, validating our theoretical analysis and establishing CoPE as a new state-of-the-art for length generalization. Our code, data, and models are available at this https URL. 

---
# Locas: Your Models are Principled Initializers of Locally-Supported Parametric Memories 

**Authors**: Sidi Lu, Zhenwen Liang, Dongyang Ma, Yan Wang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05085)  

**Abstract**: In this paper, we aim to bridge test-time-training with a new type of parametric memory that can be flexibly offloaded from or merged into model parameters. We present Locas, a Locally-Supported parametric memory that shares the design of FFN blocks in modern transformers, allowing it to be flexibly permanentized into the model parameters while supporting efficient continual learning. We discuss two major variants of Locas: one with a conventional two-layer MLP design that has a clearer theoretical guarantee; the other one shares the same GLU-FFN structure with SOTA LLMs, and can be easily attached to existing models for both parameter-efficient and computation-efficient continual learning. Crucially, we show that proper initialization of such low-rank sideway-FFN-style memories -- performed in a principled way by reusing model parameters, activations and/or gradients -- is essential for fast convergence, improved generalization, and catastrophic forgetting prevention. We validate the proposed memory mechanism on the PG-19 whole-book language modeling and LoCoMo long-context dialogue question answering tasks. With only 0.02\% additional parameters in the lowest case, Locas-GLU is capable of storing the information from past context while maintaining a much smaller context window. In addition, we also test the model's general capability loss after memorizing the whole book with Locas, through comparative MMLU evaluation. Results show the promising ability of Locas to permanentize past context into parametric knowledge with minimized catastrophic forgetting of the model's existing internal knowledge. 

---
# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion 

**Authors**: Zhuokun Chen, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05305)  

**Abstract**: Generating long-form content, such as minute-long videos and extended texts, is increasingly important for modern generative models. Block diffusion improves inference efficiency via KV caching and block-wise causal inference and has been widely adopted in diffusion language models and video generation. However, in long-context settings, block diffusion still incurs substantial overhead from repeatedly computing attention over a growing KV cache. We identify an underexplored property of block diffusion: cross-step redundancy of attention within a block. Our analysis shows that attention outputs from tokens outside the current block remain largely stable across diffusion steps, while block-internal attention varies significantly. Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that reuses stable attention output, reducing attention computation and KV cache access without modifying the diffusion process. Moreover, FlashBlock is orthogonal to sparse attention and can be combined as a complementary residual reuse strategy, substantially improving model accuracy under aggressive sparsification. Experiments on diffusion language models and video generation demonstrate up to 1.44$\times$ higher token throughput and up to 1.6$\times$ reduction in attention time, with negligible impact on generation quality. Project page: this https URL. 

---
# DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search 

**Authors**: Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05014)  

**Abstract**: With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior. 

---
# Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs 

**Authors**: Wentao Ni, Kangqi Zhang, Zhongming Yu, Oren Nelson, Mingu Lee, Hong Cai, Fatih Porikli, Jongryool Kim, Zhijian Liu, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05191)  

**Abstract**: As long-context inference becomes central to large language models (LLMs), attention over growing key-value caches emerges as a dominant decoding bottleneck, motivating sparse attention for scalable inference. Fixed-budget top-k sparse attention cannot adapt to heterogeneous attention distributions across heads and layers, whereas top-p sparse attention directly preserves attention mass and provides stronger accuracy guarantees. Existing top-p methods, however, fail to jointly optimize top-p accuracy, selection overhead, and sparse attention cost, which limits their overall efficiency. We present Double-P, a hierarchical sparse attention framework that optimizes all three stages. Double-P first performs coarse-grained top-p estimation at the cluster level using size-weighted centroids, then adaptively refines computation through a second top-p stage that allocates token-level attention only when needed. Across long-context benchmarks, Double-P consistently achieves near-zero accuracy drop, reducing attention computation overhead by up to 1.8x and delivers up to 1.3x end-to-end decoding speedup over state-of-the-art fixed-budget sparse attention methods. 

---
# GLASS: A Generative Recommender for Long-sequence Modeling via SID-Tier and Semantic Search 

**Authors**: Shiteng Cao, Junda She, Ji Liu, Bin Zeng, Chengcheng Guo, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Zhiheng Li, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05663)  

**Abstract**: Leveraging long-term user behavioral patterns is a key trajectory for enhancing the accuracy of modern recommender systems. While generative recommender systems have emerged as a transformative paradigm, they face hurdles in effectively modeling extensive historical sequences. To address this challenge, we propose GLASS, a novel framework that integrates long-term user interests into the generative process via SID-Tier and Semantic Search. We first introduce SID-Tier, a module that maps long-term interactions into a unified interest vector to enhance the prediction of the initial SID token. Unlike traditional retrieval models that struggle with massive item spaces, SID-Tier leverages the compact nature of the semantic codebook to incorporate cross features between the user's long-term history and candidate semantic codes. Furthermore, we present semantic hard search, which utilizes generated coarse-grained semantic ID as dynamic keys to extract relevant historical behaviors, which are then fused via an adaptive gated fusion module to recalibrate the trajectory of subsequent fine-grained tokens. To address the inherent data sparsity in semantic hard search, we propose two strategies: semantic neighbor augmentation and codebook resizing. Extensive experiments on two large-scale real-world datasets, TAOBAO-MM and KuaiRec, demonstrate that GLASS outperforms state-of-the-art baselines, achieving significant gains in recommendation quality. Our codes are made publicly available to facilitate further research in generative recommendation. 

---
