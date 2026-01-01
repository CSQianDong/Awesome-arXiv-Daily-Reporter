# MEIC-DT: Memory-Efficient Incremental Clustering for Long-Text Coreference Resolution with Dual-Threshold Constraints 

**Authors**: Kangyang Luo, Shuzheng Si, Yuzhuo Bai, Cheng Gao, Zhitong Wang, Cheng Huang, Yingli Shen, Yufeng Han, Wenhao Li, Cunliang Kong, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.24711)  

**Abstract**: In the era of large language models (LLMs), supervised neural methods remain the state-of-the-art (SOTA) for Coreference Resolution. Yet, their full potential is underexplored, particularly in incremental clustering, which faces the critical challenge of balancing efficiency with performance for long texts. To address the limitation, we propose \textbf{MEIC-DT}, a novel dual-threshold, memory-efficient incremental clustering approach based on a lightweight Transformer. MEIC-DT features a dual-threshold constraint mechanism designed to precisely control the Transformer's input scale within a predefined memory budget. This mechanism incorporates a Statistics-Aware Eviction Strategy (\textbf{SAES}), which utilizes distinct statistical profiles from the training and inference phases for intelligent cache management. Furthermore, we introduce an Internal Regularization Policy (\textbf{IRP}) that strategically condenses clusters by selecting the most representative mentions, thereby preserving semantic integrity. Extensive experiments on common benchmarks demonstrate that MEIC-DT achieves highly competitive coreference performance under stringent memory constraints. 

---
# Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models 

**Authors**: Junru Lu, Jiarui Qin, Lingfeng Qiao, Yinghui Li, Xinyi Dai, Bo Ke, Jianfeng He, Ruizhi Qiao, Di Yin, Xing Sun, Yunsheng Wu, Yinsong Liu, Shuangyin Liu, Mingkong Tang, Haodong Lin, Jiayi Kuang, Fanxu Meng, Xiaojuan Tang, Yunjia Xi, Junjie Huang, Haotong Yang, Zhenyi Shen, Yangning Li, Qianwen Zhang, Yifei Yu, Siyu An, Junnan Dong, Qiufeng Wang, Jie Wang, Keyu Chen, Wei Wen, Taian Guo, Zhifeng Shen, Daohai Yu, Jiahao Li, Ke Li, Zongyi Li, Xiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2512.24618)  

**Abstract**: We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities. 

---
# R-Debater: Retrieval-Augmented Debate Generation through Argumentative Memory 

**Authors**: Maoyuan Li, Zhongsheng Wang, Haoyuan Li, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.24684)  

**Abstract**: We present R-Debater, an agentic framework for generating multi-turn debates built on argumentative memory. Grounded in rhetoric and memory studies, the system views debate as a process of recalling and adapting prior arguments to maintain stance consistency, respond to opponents, and support claims with evidence. Specifically, R-Debater integrates a debate knowledge base for retrieving case-like evidence and prior debate moves with a role-based agent that composes coherent utterances across turns. We evaluate on standardized ORCHID debates, constructing a 1,000-item retrieval corpus and a held-out set of 32 debates across seven domains. Two tasks are evaluated: next-utterance generation, assessed by InspireScore (subjective, logical, and factual), and adversarial multi-turn simulations, judged by Debatrix (argument, source, language, and overall). Compared with strong LLM baselines, R-Debater achieves higher single-turn and multi-turn scores. Human evaluation with 20 experienced debaters further confirms its consistency and evidence use, showing that combining retrieval grounding with structured planning yields more faithful, stance-aligned, and coherent debates across turns. 

---
# Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling 

**Authors**: Chulun Zhou, Chunkang Zhang, Guoxin Yu, Fandong Meng, Jie Zhou, Wai Lam, Mo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23959)  

**Abstract**: Multi-step retrieval-augmented generation (RAG) has become a widely adopted strategy for enhancing large language models (LLMs) on tasks that demand global comprehension and intensive reasoning. Many RAG systems incorporate a working memory module to consolidate retrieved information. However, existing memory designs function primarily as passive storage that accumulates isolated facts for the purpose of condensing the lengthy inputs and generating new sub-queries through deduction. This static nature overlooks the crucial high-order correlations among primitive facts, the compositions of which can often provide stronger guidance for subsequent steps. Therefore, their representational strength and impact on multi-step reasoning and knowledge evolution are limited, resulting in fragmented reasoning and weak global sense-making capacity in extended contexts. We introduce HGMem, a hypergraph-based memory mechanism that extends the concept of memory beyond simple storage into a dynamic, expressive structure for complex reasoning and global understanding. In our approach, memory is represented as a hypergraph whose hyperedges correspond to distinct memory units, enabling the progressive formation of higher-order interactions within memory. This mechanism connects facts and thoughts around the focal problem, evolving into an integrated and situated knowledge structure that provides strong propositions for deeper reasoning in subsequent steps. We evaluate HGMem on several challenging datasets designed for global sense-making. Extensive experiments and in-depth analyses show that our method consistently improves multi-step RAG and substantially outperforms strong baseline systems across diverse tasks. 

---
# Retrieval Augmented Question Answering: When Should LLMs Admit Ignorance? 

**Authors**: Dingmin Wang, Ji Ma, Shankar Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.23836)  

**Abstract**: The success of expanded context windows in Large Language Models (LLMs) has driven increased use of broader context in retrieval-augmented generation. We investigate the use of LLMs for retrieval augmented question answering. While longer contexts make it easier to incorporate targeted knowledge, they introduce more irrelevant information that hinders the model's generation process and degrades its performance. To address the issue, we design an adaptive prompting strategy which involves splitting the retrieved information into smaller chunks and sequentially prompting a LLM to answer the question using each chunk. Adjusting the chunk size allows a trade-off between incorporating relevant information and reducing irrelevant information. Experimental results on three open-domain question answering datasets demonstrate that the adaptive strategy matches the performance of standard prompting while using fewer tokens. Our analysis reveals that when encountering insufficient information, the LLM often generates incorrect answers instead of declining to respond, which constitutes a major source of error. This finding highlights the need for further research into enhancing LLMs' ability to effectively decline requests when faced with inadequate information. 

---
# Efficient Context Scaling with LongCat ZigZag Attention 

**Authors**: Chen Zhang, Yang Bai, Jiahuan Li, Anchun Gui, Keheng Wang, Feifan Liu, Guanyu Wu, Yuwei Jiang, Defei Bu, Li Wei, Haihang Jing, Hongyin Tang, Xin Chen, Xiangzhou Huang, Fengcun Li, Rongxiang Weng, Yulei Qian, Yifan Lu, Yerui Sun, Jingang Wang, Yuchen Xie, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2512.23966)  

**Abstract**: We introduce LongCat ZigZag Attention (LoZA), which is a sparse attention scheme designed to transform any existing full-attention models into sparse versions with rather limited compute budget. In long-context scenarios, LoZA can achieve significant speed-ups both for prefill-intensive (e.g., retrieval-augmented generation) and decode-intensive (e.g., tool-integrated reasoning) cases. Specifically, by applying LoZA to LongCat-Flash during mid-training, we serve LongCat-Flash-Exp as a long-context foundation model that can swiftly process up to 1 million tokens, enabling efficient long-term reasoning and long-horizon agentic capabilities. 

---
# Recursive Language Models 

**Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2512.24601)  

**Abstract**: We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. We find that RLMs successfully handle inputs up to two orders of magnitude beyond model context windows and, even for shorter prompts, dramatically outperform the quality of base LLMs and common long-context scaffolds across four diverse long-context tasks, while having comparable (or cheaper) cost per query. 

---
# Probing the Limits of Compressive Memory: A Study of Infini-Attention in Small-Scale Pretraining 

**Authors**: Ruizhe Huang, Kexuan Zhang, Yihao Fang, Baifeng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23862)  

**Abstract**: This study investigates small-scale pretraining for Small Language Models (SLMs) to enable efficient use of limited data and compute, improve accessibility in low-resource settings and reduce costs. To enhance long-context extrapolation in compact models, we focus on Infini-attention, which builds a compressed memory from past segments while preserving local attention. In our work, we conduct an empirical study using 300M-parameter LLaMA models pretrained with Infini-attention. The model demonstrates training stability and outperforms the baseline in long-context retrieval. We identify the balance factor as a key part of the model performance, and we found that retrieval accuracy drops with repeated memory compressions over long sequences. Even so, Infini-attention still effectively compensates for the SLM's limited parameters. Particularly, despite performance degradation at a 16,384-token context, the Infini-attention model achieves up to 31% higher accuracy than the baseline. Our findings suggest that achieving robust long-context capability in SLMs benefits from architectural memory like Infini-attention. 

---
# Trellis: Learning to Compress Key-Value Memory in Attention Models 

**Authors**: Mahdi Karami, Ali Behrouz, Praneeth Kacham, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2512.23852)  

**Abstract**: Transformers, while powerful, suffer from quadratic computational complexity and the ever-growing Key-Value (KV) cache of the attention mechanism. This paper introduces Trellis, a novel Transformer architecture with bounded memory that learns how to compress its key-value memory dynamically at test time. Trellis replaces the standard KV cache with a fixed-size memory and train a two-pass recurrent compression mechanism to store new keys and values into memory. To achieve this, it leverages an online gradient descent procedure with a forget gate, enabling the compressed memory to be updated recursively while learning to retain important contextual information from incoming tokens at test time. Extensive experiments on language modeling, common-sense reasoning, recall-intensive tasks, and time series show that the proposed architecture outperforms strong baselines. Notably, its performance gains increase as the sequence length grows, highlighting its potential for long-context applications. 

---
# SPARK: Search Personalization via Agent-Driven Retrieval and Knowledge-sharing 

**Authors**: Gaurab Chhetri, Subasish Das, Tausif Islam Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2512.24008)  

**Abstract**: Personalized search demands the ability to model users' evolving, multi-dimensional information needs; a challenge for systems constrained by static profiles or monolithic retrieval pipelines. We present SPARK (Search Personalization via Agent-Driven Retrieval and Knowledge-sharing), a framework in which coordinated persona-based large language model (LLM) agents deliver task-specific retrieval and emergent personalization. SPARK formalizes a persona space defined by role, expertise, task context, and domain, and introduces a Persona Coordinator that dynamically interprets incoming queries to activate the most relevant specialized agents. Each agent executes an independent retrieval-augmented generation process, supported by dedicated long- and short-term memory stores and context-aware reasoning modules. Inter-agent collaboration is facilitated through structured communication protocols, including shared memory repositories, iterative debate, and relay-style knowledge transfer. Drawing on principles from cognitive architectures, multi-agent coordination theory, and information retrieval, SPARK models how emergent personalization properties arise from distributed agent behaviors governed by minimal coordination rules. The framework yields testable predictions regarding coordination efficiency, personalization quality, and cognitive load distribution, while incorporating adaptive learning mechanisms for continuous persona refinement. By integrating fine-grained agent specialization with cooperative retrieval, SPARK provides insights for next-generation search systems capable of capturing the complexity, fluidity, and context sensitivity of human information-seeking behavior. 

---
# PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression 

**Authors**: Bo Jiang, Taolue Yang, Youyuan Liu, Xubin He, Sheng Di, Sian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.24449)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable potential across a wide range of practical applications. However, long-context inference remains a significant challenge due to the substantial memory requirements of the key-value (KV) cache, which can scale to several gigabytes as sequence length and batch size increase. In this paper, we present \textbf{PackKV}, a generic and efficient KV cache management framework optimized for long-context generation. %, which synergistically supports both latency-critical and throughput-critical inference scenarios. PackKV introduces novel lossy compression techniques specifically tailored to the characteristics of KV cache data, featuring a careful co-design of compression algorithms and system architecture. Our approach is compatible with the dynamically growing nature of the KV cache while preserving high computational efficiency. Experimental results show that, under the same and minimum accuracy drop as state-of-the-art quantization methods, PackKV achieves, on average, \textbf{153.2}\% higher memory reduction rate for the K cache and \textbf{179.6}\% for the V cache. Furthermore, PackKV delivers extremely high execution throughput, effectively eliminating decompression overhead and accelerating the matrix-vector multiplication operation. Specifically, PackKV achieves an average throughput improvement of \textbf{75.7}\% for K and \textbf{171.7}\% for V across A100 and RTX Pro 6000 GPUs, compared to cuBLAS matrix-vector multiplication kernels, while demanding less GPU memory bandwidth. Code available on this https URL 

---
