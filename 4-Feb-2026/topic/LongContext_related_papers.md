# Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03249)  

**Abstract**: Scaling test-time compute via long Chain-ofThought unlocks remarkable gains in reasoning capabilities, yet it faces practical limits due to the linear growth of KV cache and quadratic attention complexity. In this paper, we introduce Accordion-Thinking, an end-to-end framework where LLMs learn to self-regulate the granularity of the reasoning steps through dynamic summarization. This mechanism enables a Fold inference mode, where the model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens. We apply reinforcement learning to incentivize this capability further, uncovering a critical insight: the accuracy gap between the highly efficient Fold mode and the exhaustive Unfold mode progressively narrows and eventually vanishes over the course of training. This phenomenon demonstrates that the model learns to encode essential reasoning information into compact summaries, achieving effective compression of the reasoning context. Our Accordion-Thinker demonstrates that with learned self-compression, LLMs can tackle complex reasoning tasks with minimal dependency token overhead without compromising solution quality, and it achieves a 3x throughput while maintaining accuracy on a 48GB GPU memory configuration, while the structured step summaries provide a human-readable account of the reasoning process. 

---
# ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs 

**Authors**: Xuancheng Li, Haitao Li, Yujia Zhou, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03226)  

**Abstract**: Long-context inputs in large language models (LLMs) often suffer from the "lost in the middle" problem, where critical information becomes diluted or ignored due to excessive length. Context compression methods aim to address this by reducing input size, but existing approaches struggle with balancing information preservation and compression efficiency. We propose Adaptive Task-Aware Compressor (ATACompressor), which dynamically adjusts compression based on the specific requirements of the task. ATACompressor employs a selective encoder that compresses only the task-relevant portions of long contexts, ensuring that essential information is preserved while reducing unnecessary content. Its adaptive allocation controller perceives the length of relevant content and adjusts the compression rate accordingly, optimizing resource utilization. We evaluate ATACompressor on three QA datasets: HotpotQA, MSMARCO, and SQUAD-showing that it outperforms existing methods in terms of both compression efficiency and task performance. Our approach provides a scalable solution for long-context processing in LLMs. Furthermore, we perform a range of ablation studies and analysis experiments to gain deeper insights into the key components of ATACompressor. 

---
# Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models 

**Authors**: Difan Deng, Andreas Bentzen Winje, Lukas Fehring, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2602.03681)  

**Abstract**: The quadratic computational complexity of softmax transformers has become a bottleneck in long-context scenarios. In contrast, linear attention model families provide a promising direction towards a more efficient sequential model. These linear attention models compress past KV values into a single hidden state, thereby efficiently reducing complexity during both training and inference. However, their expressivity remains limited by the size of their hidden state. Previous work proposed interleaving softmax and linear attention layers to reduce computational complexity while preserving expressivity. Nevertheless, the efficiency of these models remains bottlenecked by their softmax attention layers. In this paper, we propose Neural Attention Search Linear (NAtS-L), a framework that applies both linear attention and softmax attention operations within the same layer on different tokens. NAtS-L automatically determines whether a token can be handled by a linear attention model, i.e., tokens that have only short-term impact and can be encoded into fixed-size hidden states, or require softmax attention, i.e., tokens that contain information related to long-term retrieval and need to be preserved for future queries. By searching for optimal Gated DeltaNet and softmax attention combinations across tokens, we show that NAtS-L provides a strong yet efficient token-level hybrid architecture. 

---
# Context Compression via Explicit Information Transmission 

**Authors**: Jiangnan Ye, Hanqi Yan, Zhenyi Shen, Heng Chang, Ye Mao, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.03784)  

**Abstract**: Long-context inference with Large Language Models (LLMs) is costly due to quadratic attention and growing key-value caches, motivating context compression. In this work, we study soft context compression, where a long context is condensed into a small set of continuous representations. Existing methods typically re-purpose the LLM itself as a trainable compressor, relying on layer-by-layer self-attention to iteratively aggregate information. We argue that this paradigm suffers from two structural limitations: (i) progressive representation overwriting across layers (ii) uncoordinated allocation of compression capacity across tokens. We propose ComprExIT (Context Compression via Explicit Information Transmission), a lightweight framework that formulates soft compression into a new paradigm: explicit information transmission over frozen LLM hidden states. This decouples compression from the model's internal self-attention dynamics. ComprExIT performs (i) depth-wise transmission to selectively transmit multi-layer information into token anchors, mitigating progressive overwriting, and (ii) width-wise transmission to aggregate anchors into a small number of slots via a globally optimized transmission plan, ensuring coordinated allocation of information. Across six question-answering benchmarks, ComprExIT consistently outperforms state-of-the-art context compression methods while introducing only ~1% additional parameters, demonstrating that explicit and coordinated information transmission enables more effective and robust long-context compression. 

---
# Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection 

**Authors**: Dongwon Jo, Beomseok Kang, Jiwon Song, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.03216)  

**Abstract**: The quadratic complexity of attention remains the central bottleneck in long-context inference for large language models. Prior acceleration methods either sparsify the attention map with structured patterns or permanently evict tokens at specific layers, which can retain irrelevant tokens or rely on irreversible early decisions despite the layer-/head-wise dynamics of token importance. In this paper, we propose Token Sparse Attention, a lightweight and dynamic token-level sparsification mechanism that compresses per-head $Q$, $K$, $V$ to a reduced token set during attention and then decompresses the output back to the original sequence, enabling token information to be reconsidered in subsequent layers. Furthermore, Token Sparse Attention exposes a new design point at the intersection of token selection and sparse attention. Our approach is fully compatible with dense attention implementations, including Flash Attention, and can be seamlessly composed with existing sparse attention kernels. Experimental results show that Token Sparse Attention consistently improves accuracy-latency trade-off, achieving up to $\times$3.23 attention speedup at 128K context with less than 1% accuracy degradation. These results demonstrate that dynamic and interleaved token-level sparsification is a complementary and effective strategy for scalable long-context inference. 

---
# ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution 

**Authors**: Zican Dong, Peiyu Liu, Junyi Li, Zhipeng Chen, Han Peng, Shuo Wang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03203)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning abilities by producing long reasoning traces. However, as the sequence length grows, the key-value (KV) cache expands linearly, incurring significant memory and computation costs. Existing KV cache eviction methods mitigate this issue by discarding less important KV pairs, but often fail to capture complex KV dependencies, resulting in performance degradation. To better balance efficiency and performance, we introduce ForesightKV, a training-based KV cache eviction framework that learns to predict which KV pairs to evict during long-text generations. We first design the Golden Eviction algorithm, which identifies the optimal eviction KV pairs at each step using future attention scores. These traces and the scores at each step are then distilled via supervised training with a Pairwise Ranking Loss. Furthermore, we formulate cache eviction as a Markov Decision Process and apply the GRPO algorithm to mitigate the significant language modeling loss increase on low-entropy tokens. Experiments on AIME2024 and AIME2025 benchmarks of three reasoning models demonstrate that ForesightKV consistently outperforms prior methods under only half the cache budget, while benefiting synergistically from both supervised and reinforcement learning approaches. 

---
# FASA: Frequency-aware Sparse Attention 

**Authors**: Yifei Wang, Yueqi Wang, Zhenrui Yue, Huimin Zeng, Yong Wang, Ismini Lourentzou, Zhengzhong Tu, Xiangxiang Chu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2602.03152)  

**Abstract**: The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance. We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance. FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens. %making them a powerful and efficient proxy for token importance. Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset. % Since accessing only a small fraction of the KV cache, FASA drastically lowers memory bandwidth requirements and computational cost. Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24. 

---
# AmharicStoryQA: A Multicultural Story Question Answering Benchmark in Amharic 

**Authors**: Israel Abebe Azime, Abenezer Kebede Angamo, Hana Mekonen Tamiru, Dagnachew Mekonnen Marilign, Philipp Slusallek, Seid Muhie Yimam, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2602.02774)  

**Abstract**: With the growing emphasis on multilingual and cultural evaluation benchmarks for large language models, language and culture are often treated as synonymous, and performance is commonly used as a proxy for a models understanding of a given language. In this work, we argue that such evaluations overlook meaningful cultural variation that exists within a single language. We address this gap by focusing on narratives from different regions of Ethiopia and demonstrate that, despite shared linguistic characteristics, region-specific and domain-specific content substantially influences language evaluation outcomes. To this end, we introduce \textbf{\textit{AmharicStoryQA}}, a long-sequence story question answering benchmark grounded in culturally diverse narratives from Amharic-speaking regions. Using this benchmark, we reveal a significant narrative understanding gap in existing LLMs, highlight pronounced regional differences in evaluation results, and show that supervised fine-tuning yields uneven improvements across regions and evaluation settings. Our findings emphasize the need for culturally grounded benchmarks that go beyond language-level evaluation to more accurately assess and improve narrative understanding in low-resource languages. 

---
# InfMem: Learning System-2 Memory Control for Long-Context Agent 

**Authors**: Xinyu Wang, Mingze Li, Peng Lu, Xiao-Wen Chang, Lifeng Shang, Jinping Li, Fei Mi, Prasanna Parthasarathi, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.02704)  

**Abstract**: Reasoning over ultra-long documents requires synthesizing sparse evidence scattered across distant segments under strict memory constraints. While streaming agents enable scalable processing, their passive memory update strategy often fails to preserve low-salience bridging evidence required for multi-hop reasoning. We propose InfMem, a control-centric agent that instantiates System-2-style control via a PreThink-Retrieve-Write protocol. InfMem actively monitors evidence sufficiency, performs targeted in-document retrieval, and applies evidence-aware joint compression to update a bounded memory. To ensure reliable control, we introduce a practical SFT-to-RL training recipe that aligns retrieval, writing, and stopping decisions with end-task correctness. On ultra-long QA benchmarks from 32k to 1M tokens, InfMem consistently outperforms MemAgent across backbones. Specifically, InfMem improves average absolute accuracy by +10.17, +11.84, and +8.23 points on Qwen3-1.7B, Qwen3-4B, and Qwen2.5-7B, respectively, while reducing inference time by $3.9\times$ on average (up to $5.1\times$) via adaptive early stopping. 

---
# ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching 

**Authors**: Yunao Zheng, Xiaojie Wang, Lei Ren, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02499)  

**Abstract**: Long-context capability and computational efficiency are among the central challenges facing today's large language models. Existing efficient attention methods reduce computational complexity, but they typically suffer from a limited coverage of the model state. This paper proposes ROSA-Tuning, a retrieval-and-recall mechanism for enhancing the long-context modeling ability of pretrained models. Beyond the standard attention mechanism, ROSA-Tuning introduces in parallel a CPU-based ROSA (RWKV Online Suffix Automaton) retrieval module, which efficiently locates historical positions in long contexts that are relevant to the current query, and injects the retrieved information into the model state in a trainable manner; subsequent weighted fusion can then be handled by range-restricted attention. To enable end-to-end training, we design a binary discretization strategy and a counterfactual gradient algorithm, and further optimize overall execution efficiency via an asynchronous CPU-GPU pipeline. Systematic evaluations on Qwen3-Base-1.7B show that ROSA-Tuning substantially restores the long-context modeling ability of windowed-attention models, achieving performance close to and in some cases matching global attention on benchmarks such as LongBench, while maintaining computational efficiency and GPU memory usage that are nearly comparable to windowed-attention methods, offering a new technical path for efficient long-context processing. The example code can be found at this https URL. 

---
# DynSplit-KV: Dynamic Semantic Splitting for KVCache Compression in Efficient Long-Context LLM Inference 

**Authors**: Jiancai Ye, Jun Liu, Qingchen Li, Tianlang Zhao, Hanbin Zhang, Jiayi Pan, Ningyi Xu, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.03184)  

**Abstract**: Although Key-Value (KV) Cache is essential for efficient large language models (LLMs) inference, its growing memory footprint in long-context scenarios poses a significant bottleneck, making KVCache compression crucial. Current compression methods rely on rigid splitting strategies, such as fixed intervals or pre-defined delimiters. We observe that rigid splitting suffers from significant accuracy degradation (ranging from 5.5% to 55.1%) across different scenarios, owing to the scenario-dependent nature of the semantic boundaries. This highlights the necessity of dynamic semantic splitting to match semantics. To achieve this, we face two challenges. (1) Improper delimiter selection misaligns semantics with the KVCache, resulting in 28.6% accuracy loss. (2) Variable-length blocks after splitting introduce over 73.1% additional inference overhead. To address the above challenges, we propose DynSplit-KV, a KVCache compression method that dynamically identifies delimiters for splitting. We propose: (1) a dynamic importance-aware delimiter selection strategy, improving accuracy by 49.9%. (2) A uniform mapping strategy that transforms variable-length semantic blocks into a fixed-length format, reducing inference overhead by 4.9x. Experiments show that DynSplit-KV achieves the highest accuracy, 2.2x speedup compared with FlashAttention and 2.6x peak memory reduction in long-context scenarios. 

---
# ALPBench: A Benchmark for Attribution-level Long-term Personal Behavior Understanding 

**Authors**: Lu Ren, Junda She, Xinchen Luo, Tao Wang, Xin Ye, Xu Zhang, Muxuan Wang, Xiao Yang, Chenguang Wang, Fei Xie, Yiwei Zhou, Danjun Wu, Guodong Zhang, Yifei Hu, Guoying Zheng, Shujie Yang, Xingmei Wang, Shiyao Wang, Yukun Zhou, Fan Yang, Size Li, Kuo Cai, Qiang Luo, Ruiming Tang, Han Li, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2602.03056)  

**Abstract**: Recent advances in large language models have highlighted their potential for personalized recommendation, where accurately capturing user preferences remains a key challenge. Leveraging their strong reasoning and generalization capabilities, LLMs offer new opportunities for modeling long-term user behavior. To systematically evaluate this, we introduce ALPBench, a Benchmark for Attribution-level Long-term Personal Behavior Understanding. Unlike item-focused benchmarks, ALPBench predicts user-interested attribute combinations, enabling ground-truth evaluation even for newly introduced items. It models preferences from long-term historical behaviors rather than users' explicitly expressed requests, better reflecting enduring interests. User histories are represented as natural language sequences, allowing interpretable, reasoning-based personalization. ALPBench enables fine-grained evaluation of personalization by focusing on the prediction of attribute combinations task that remains highly challenging for current LLMs due to the need to capture complex interactions among multiple attributes and reason over long-term user behavior sequences. 

---
