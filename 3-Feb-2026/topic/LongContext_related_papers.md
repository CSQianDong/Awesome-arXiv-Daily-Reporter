# GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm 

**Authors**: Shaopeng Chen, Chuyue Xie, Huimin Ren, Shaozong Zhang, Han Zhang, Ruobing Cheng, Zhiqiang Cao, Zehao Ju, Gao Yu, Jie Ding, Xiaodong Chen, Xuewu Jiao, Shuanglong Li, Liu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.01865)  

**Abstract**: Traditional Deep Learning Recommendation Models (DLRMs) face increasing bottlenecks in performance and efficiency, often struggling with generalization and long-sequence modeling. Inspired by the scaling success of Large Language Models (LLMs), we propose Generative Ranking for Ads at Baidu (GRAB), an end-to-end generative framework for Click-Through Rate (CTR) prediction. GRAB integrates a novel Causal Action-aware Multi-channel Attention (CamA) mechanism to effectively capture temporal dynamics and specific action signals within user behavior sequences. Full-scale online deployment demonstrates that GRAB significantly outperforms established DLRMs, delivering a 3.05% increase in revenue and a 3.49% rise in CTR. Furthermore, the model demonstrates desirable scaling behavior: its expressive power shows a monotonic and approximately linear improvement as longer interaction sequences are utilized. 

---
# SPARC-RAG: Adaptive Sequential-Parallel Scaling with Context Management for Retrieval-Augmented Generation 

**Authors**: Yuxin Yang, Gangda Deng, Ömer Faruk Akgül, Nima Chitsazan, Yash Govilkar, Akasha Tigalappanavara, Shi-Xiong Zhang, Sambit Sahu, Viktor Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2602.00083)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language model outputs in external evidence, but remains challenged on multi-hop question answering that requires long reasoning. Recent works scale RAG at inference time along two complementary dimensions: sequential depth for iterative refinement and parallel width for coverage expansion. However, naive scaling causes context contamination and scaling inefficiency, leading to diminishing or negative returns despite increased computation. To address these limitations, we propose SPARC-RAG, a multi-agent framework that coordinates sequential and parallel inference-time scaling under a unified context management mechanism. SPARC-RAG employs specialized agents that maintain a shared global context and provide explicit control over the scaling process. It generates targeted, complementary sub-queries for each branch to enable diverse parallel exploration, and explicitly regulates exiting decisions based on answer correctness and evidence grounding. To optimize scaling behavior, we further introduce a lightweight fine-tuning method with process-level verifiable preferences, which improves the efficiency of sequential scaling and effectiveness of parallel scaling. Across single- and multi-hop QA benchmarks, SPARC-RAG consistently outperforms previous RAG baselines, yielding an average +6.2 F1 improvement under lower inference cost. 

---
# Unlocking Electronic Health Records: A Hybrid Graph RAG Approach to Safe Clinical AI for Patient QA 

**Authors**: Samuel Thio, Matthew Lewis, Spiros Denaxas, Richard JB Dobson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00009)  

**Abstract**: Electronic health record (EHR) systems present clinicians with vast repositories of clinical information, creating a significant cognitive burden where critical details are easily overlooked. While Large Language Models (LLMs) offer transformative potential for data processing, they face significant limitations in clinical settings, particularly regarding context grounding and hallucinations. Current solutions typically isolate retrieval methods focusing either on structured data (SQL/Cypher) or unstructured semantic search but fail to integrate both simultaneously. This work presents MediGRAF (Medical Graph Retrieval Augmented Framework), a novel hybrid Graph RAG system that bridges this gap. By uniquely combining Neo4j Text2Cypher capabilities for structured relationship traversal with vector embeddings for unstructured narrative retrieval, MediGRAF enables natural language querying of the complete patient journey. Using 10 patients from the MIMIC-IV dataset (generating 5,973 nodes and 5,963 relationships), we generated enough nodes and data for patient level question answering (QA), and we evaluated this architecture across varying query complexities. The system demonstrated 100\% recall for factual queries which means all relevant information was retrieved and in the output, while complex inference tasks achieved a mean expert quality score of 4.25/5 with zero safety violations. These results demonstrate that hybrid graph-grounding significantly advances clinical information retrieval, offering a safer, more comprehensive alternative to standard LLM deployments. 

---
# Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts 

**Authors**: Wenhao Li, Daohai Yu, Gen Luo, Yuxin Zhang, Fei Chao, Rongrong Ji, Yifan Wu, Jiaxin Liu, Ziyang Gong, Zimu Liao  

**Link**: [PDF](https://arxiv.org/pdf/2602.02108)  

**Abstract**: Training Large Language Models (LLMs) on long contexts is severely constrained by prohibitive GPU memory overhead, not training time. The primary culprits are the activations, whose memory footprints scale linearly with sequence length. We introduce OOMB, a highly memory-efficient training system that directly confronts this barrier. Our approach employs a chunk-recurrent training framework with on-the-fly activation recomputation, which maintains a constant activation memory footprint (O(1)) and shifts the primary bottleneck to the growing KV cache. To manage the KV cache, OOMB integrates a suite of synergistic optimizations: a paged memory manager for both the KV cache and its gradients to eliminate fragmentation, asynchronous CPU offloading to hide data transfer latency, and page-level sparse attention to reduce both computational complexity and communication overhead. The synergy of these techniques yields exceptional efficiency. Our empirical results show that for every additional 10K tokens of context, the end-to-end training memory overhead increases by a mere 10MB for Qwen2.5-7B. This allows training Qwen2.5-7B with a 4M-token context on a single H200 GPU, a feat that would otherwise require a large cluster using context parallelism. This work represents a substantial advance in resource efficiency for long-context LLM training. The source code is available at this https URL. 

---
# Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing 

**Authors**: Lingkun Long, Yushi Huang, Shihao Bai, Ruihao Gong, Jun Zhang, Ao Zhou, Jianlei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02159)  

**Abstract**: Diffusion Large Language Models (dLLMs) deliver strong long-context processing capability in a non-autoregressive decoding paradigm. However, the considerable computational cost of bidirectional full attention limits the inference efficiency. Although sparse attention is promising, existing methods remain ineffective. This stems from the need to estimate attention importance for tokens yet to be decoded, while the unmasked token positions are unknown during diffusion. In this paper, we present Focus-dLLM, a novel training-free attention sparsification framework tailored for accurate and efficient long-context dLLM inference. Based on the finding that token confidence strongly correlates across adjacent steps, we first design a past confidence-guided indicator to predict unmasked regions. Built upon this, we propose a sink-aware pruning strategy to accurately estimate and remove redundant attention computation, while preserving highly influential attention sinks. To further reduce overhead, this strategy reuses identified sink locations across layers, leveraging the observed cross-layer consistency. Experimental results show that our method offers more than $29\times$ lossless speedup under $32K$ context length. The code is publicly available at: this https URL 

---
# Read As Human: Compressing Context via Parallelizable Close Reading and Skimming 

**Authors**: Jiwei Tang, Shilei Liu, Zhicheng Zhang, Qingsong Lv, Runsong Zhao, Tingwei Lu, Langming Liu, Haibin Chen, Yujin Yuan, Hai-Tao Zheng, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01840)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capability across diverse tasks. However, their deployment in long-context scenarios is hindered by two challenges: computational inefficiency and redundant information. We propose RAM (Read As HuMan), a context compression framework that adopts an adaptive hybrid reading strategy, to address these challenges. Inspired by human reading behavior (i.e., close reading important content while skimming less relevant content), RAM partitions the context into segments and encodes them with the input query in parallel. High-relevance segments are fully retained (close reading), while low-relevance ones are query-guided compressed into compact summary vectors (skimming). Both explicit textual segments and implicit summary vectors are concatenated and fed into decoder to achieve both superior performance and natural language format interpretability. To refine the decision boundary between close reading and skimming, we further introduce a contrastive learning objective based on positive and negative query-segment pairs. Experiments demonstrate that RAM outperforms existing baselines on multiple question answering and summarization benchmarks across two backbones, while delivering up to a 12x end-to-end speedup on long inputs (average length 16K; maximum length 32K). 

---
# ES-MemEval: Benchmarking Conversational Agents on Personalized Long-Term Emotional Support 

**Authors**: Tiantian Chen, Jiaqi Lu, Ying Shen, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01885)  

**Abstract**: Large Language Models (LLMs) have shown strong potential as conversational agents. Yet, their effectiveness remains limited by deficiencies in robust long-term memory, particularly in complex, long-term web-based services such as online emotional support. However, existing long-term dialogue benchmarks primarily focus on static and explicit fact retrieval, failing to evaluate agents in critical scenarios where user information is dispersed, implicit, and continuously evolving. To address this gap, we introduce ES-MemEval, a comprehensive benchmark that systematically evaluates five core memory capabilities: information extraction, temporal reasoning, conflict detection, abstention, and user modeling, in long-term emotional support settings, covering question answering, summarization, and dialogue generation tasks. To support the benchmark, we also propose EvoEmo, a multi-session dataset for personalized long-term emotional support that captures fragmented, implicit user disclosures and evolving user states. Extensive experiments on open-source long-context, commercial, and retrieval-augmented (RAG) LLMs show that explicit long-term memory is essential for reducing hallucinations and enabling effective personalization. At the same time, RAG improves factual consistency but struggles with temporal dynamics and evolving user states. These findings highlight both the potential and limitations of current paradigms and motivate more robust integration of memory and retrieval for long-term personalized dialogue systems. 

---
# Data Distribution Matters: A Data-Centric Perspective on Context Compression for Large Language Model 

**Authors**: Kangtao Lv, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Shilei Liu, Yongwei Wang, Yujin Yuan, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01778)  

**Abstract**: The deployment of Large Language Models (LLMs) in long-context scenarios is hindered by computational inefficiency and significant information redundancy. Although recent advancements have widely adopted context compression to address these challenges, existing research only focus on model-side improvements, the impact of the data distribution itself on context compression remains largely unexplored. To bridge this gap, we are the first to adopt a data-centric perspective to systematically investigate how data distribution impacts compression quality, including two dimensions: input data and intrinsic data (i.e., the model's internal pretrained knowledge). We evaluate the semantic integrity of compressed representations using an autoencoder-based framework to systematically investigate it. Our experimental results reveal that: (1) encoder-measured input entropy negatively correlates with compression quality, while decoder-measured entropy shows no significant relationship under a frozen-decoder setting; and (2) the gap between intrinsic data of the encoder and decoder significantly diminishes compression gains, which is hard to mitigate. Based on these findings, we further present practical guidelines to optimize compression gains. 

---
# COMI: Coarse-to-fine Context Compression via Marginal Information Gain 

**Authors**: Jiwei Tang, Shilei Liu, Zhicheng Zhang, Yujin Yuan, Libin Zheng, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01719)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks. However, their deployment in long context scenarios remains hindered by computational inefficiency and information redundancy. Context compression methods address these challenges by significantly reducing input length and eliminating redundancy. We propose COMI, a coarse-to-fine adaptive context compression framework that jointly optimizes for semantic relevance and diversity under high compression rates. We introduce Marginal Information Gain (MIG), a metric defined as the relevance of a unit to the input query minus its semantic redundancy with other units, guiding the compression process to prioritize information that is both relevant and low redundant. The framework operates in two stages: (1) Coarse-Grained Group Reallocation, where the context is partitioned into groups and dynamically assigned compression rates based on inter-group MIG, ensuring compression budgets align with information value distribution; and (2) Fine-Grained Token Merging, where tokens within each group are fused via an intra-group MIG-based weighting mechanism, thereby preserving key semantics while avoiding the accumulation of redundancy. Extensive experiments across question-answering (e.g., NaturalQuestions, 2WikiMQA, HotpotQA and NarrativeQA), summarization (e.g., MultiNews) with various backbones (e.g., LLaMA-2-7B, Qwen2-7B) show that COMI outperforms existing baselines by a large margin, e.g., approximately 25-point Exact Match (EM) improvement under 32x compression constraint with Qwen2-7B on NaturalQuestions. 

---
# FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents 

**Authors**: Chiwei Zhu, Benfeng Xu, Mingxuan Du, Shaohan Wang, Xiaorui Wang, Zhendong Mao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.01566)  

**Abstract**: Deep research is emerging as a representative long-horizon task for large language model (LLM) agents. However, long trajectories in deep research often exceed model context limits, compressing token budgets for both evidence collection and report writing, and preventing effective test-time scaling. We introduce FS-Researcher, a file-system-based, dual-agent framework that scales deep research beyond the context window via a persistent workspace. Specifically, a Context Builder agent acts as a librarian which browses the internet, writes structured notes, and archives raw sources into a hierarchical knowledge base that can grow far beyond context length. A Report Writer agent then composes the final report section by section, treating the knowledge base as the source of facts. In this framework, the file system serves as a durable external memory and a shared coordination medium across agents and sessions, enabling iterative refinement beyond the context window. Experiments on two open-ended benchmarks (DeepResearch Bench and DeepConsult) show that FS-Researcher achieves state-of-the-art report quality across different backbone models. Further analyses demonstrate a positive correlation between final report quality and the computation allocated to the Context Builder, validating effective test-time scaling under the file-system paradigm. The code and data are anonymously open-sourced at this https URL. 

---
# EverMemBench: Benchmarking Long-Term Interactive Memory in Large Language ModelsEverMemBench: Benchmarking Long-Term Interactive Memory in Large Language Models 

**Authors**: Chuanrui Hu, Tong Li, Xingze Gao, Hongda Chen, Dannong Xu, Yi Bai, Tianwei Lin, Xinda Zhao, Xiaohong Li, Jiaqi An, Yunyun Han, Jian Pei, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01313)  

**Abstract**: Long-term conversational memory is essential for LLM-based assistants, yet existing benchmarks focus on dyadic, single-topic dialogues that fail to capture real-world complexity. We introduce EverMemBench, a benchmark featuring multi-party, multi-group conversations spanning over 1 million tokens with temporally evolving information, cross-topic interleaving, and role-specific personas. EverMemBench evaluates memory systems across three dimensions through 1,000+ QA pairs: fine-grained recall, memory awareness, and user profile understanding. Our evaluation reveals critical limitations: (1) multi-hop reasoning collapses in multi-party settings, with even oracle models achieving only 26%; (2) temporal reasoning remains unsolved, requiring version semantics beyond timestamp matching; (3) memory awareness is bottlenecked by retrieval, where current similarity-based methods fail to bridge the semantic gap between queries and implicitly relevant memories. EverMemBench provides a challenging testbed for developing next-generation memory architectures. 

---
# Long-range Modeling and Processing of Multimodal Event Sequences 

**Authors**: Jichu Li, Yilun Zhong, Zhiting Li, Feng Zhou, Quyu Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.01125)  

**Abstract**: Temporal point processes (TPPs) have emerged as powerful tools for modeling asynchronous event sequences. While recent advances have extended TPPs to handle textual information, existing approaches are limited in their ability to generate rich, multimodal content and reason about event dynamics. A key challenge is that incorporating multimodal data dramatically increases sequence length, hindering the ability of attention-based models to generate coherent, long-form textual descriptions that require long-range understanding. In this paper, we propose a novel framework that extends LLM-based TPPs to the visual modality, positioning text generation as a core capability alongside time and type prediction. Our approach addresses the long-context problem through an adaptive sequence compression mechanism based on temporal similarity, which reduces sequence length while preserving essential patterns. We employ a two-stage paradigm of pre-training on compressed sequences followed by supervised fine-tuning for downstream tasks. Extensive experiments, including on the challenging DanmakuTPP-QA benchmark, demonstrate that our method outperforms state-of-the-art baselines in both predictive accuracy and the quality of its generated textual analyses. 

---
# HyLRA: Hybrid Layer Reuse Attention for Efficient Long-Context Inference 

**Authors**: Xuan Ai, Qingqing Yang, Peng Wang, Lei Deng, Lin Zhang, Renhai Chen, Gong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00777)  

**Abstract**: Long-context inference in Large Language Models (LLMs) is bottlenecked by the quadratic computation complexity of attention and the substantial memory footprint of Key-Value (KV) caches. While existing sparse attention mechanisms attempt to mitigate this by exploiting inherent sparsity, they often rely on rigid patterns or aggressive pruning, failing to achieve an optimal balance between efficiency and accuracy. In this paper, we introduce {\bf HyLRA} ({\bf Hy}brid {\bf L}ayer {\bf R}euse {\bf A}ttention), a novel framework driven by layer-wise sparsity profiling. Our empirical analysis uncovers a dual characteristic in attention mechanics: \textit{intra-layer sensitivity}, where specific layers necessitate full attention to prevent feature distortion, and \textit{inter-layer similarity}, where consecutive layers share substantial critical tokens. Based on these observations, HyLRA employs an offline dynamic programming approach to derive an optimal layer-wise policy. This hybrid strategy retains full attention for sensitive layers to ensure robustness, while enabling tolerant layers to bypass quadratic calculations by directly reusing top-$k$ indices from preceding layers. This approach allows LLMs to restrict computation to the most critical tokens, effectively overcoming the quadratic bottleneck of dense attention. Extensive evaluations demonstrate that HyLRA improves inference throughput by 6\%--46\% while maintaining comparable performance (with $<1\%$ accuracy degradation), consistently outperforming state-of-the-art sparse attention methods. HyLRA is open source at \href{this https URL}{\texttt{/r/unified-cache-management-CF80/}} 

---
# G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning in Large Language Models 

**Authors**: Xun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.00015)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, yet they remain constrained by the finite capacity of their context windows and the inherent difficulty of maintaining long-term factual consistency during multi-hop reasoning. While existing methods utilize context compression or recurrent tokens, they often suffer from ``context rot'' or the dilution of information over long horizons. In this paper, we propose \textbf{G-MemLLM}, a memory-augmented architecture that integrates a frozen LLM backbone with a trainable \textbf{Latent Memory Bank}. Our key innovation is a GRU-style gated update logic that allows the model to selectively update, preserve, or overwrite latent memory slots, preventing the vanishing gradients of knowledge common in recurrent systems. We evaluate G-MemLLM across scales, from GPT-2 (124M) to Llama 3.1 (8B), on the HotpotQA and Zero-Shot Relation Extraction (ZsRE) benchmarks. Our results demonstrate that G-MemLLM significantly enhances multi-hop reasoning and relational precision, achieving a 13.3\% accuracy boost on ZsRE for Llama 3.1-8B, and it also yields improvements across model scales, boosting Answer F1 by 8.56 points for GPT-2 and increasing Supporting Fact F1 by 6.89 points for Llama 3.1-8B on HotpotQA. 

---
# More Than a Quick Glance: Overcoming the Greedy Bias in KV-Cache Compression 

**Authors**: Aryan Sood, Tanvi Sharma, Vansh Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2602.02199)  

**Abstract**: While Large Language Models (LLMs) can theoretically support extensive context windows, their actual deployment is constrained by the linear growth of Key-Value (KV) cache memory. Prevailing compression strategies mitigate this through various pruning mechanisms, yet trade-off semantic recall for memory efficiency. In this work, we present LASER-KV (Layer Accumulated Selection with Exact-LSH Recall), a framework designed to test the limits of KV compression under a strict accumulative budgeting policy. We deviate from the standard fixed summary size approach by implementing a block-wise accumulation strategy governed by a protection divisor (n). This allows us to isolate the effects of compression from sliding window artifacts. Our experiments on the Babilong benchmark reveal performance degradation in previous compression methods by 15-30% on various long context tasks. LASER-KV maintains stable performance, achieving superior accuracies by a margin of upto 10% at 128k. These findings challenge the prevailing assumption that attention scores alone are a sufficient proxy for token utility. 

---
# A State-Transition Framework for Efficient LLM Reasoning 

**Authors**: Liang Zhang, Yu Zhao, Longyue Wang, Tianqi Shi, Weihua Luo, Kaifu Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.01198)  

**Abstract**: While Long Chain-of-Thought (CoT) reasoning significantly improves Large Language Models (LLMs) performance on complex reasoning tasks, the substantial computational and memory costs of generating long CoT sequences limit their efficiency and practicality. Existing studies usually enhance the reasoning efficiency of LLMs by compressing CoT sequences. However, this approach conflicts with test-time scaling, limiting the reasoning capacity of LLMs. In this paper, we propose an efficient reasoning framework that models the reasoning process of LLMs as a state-transition process. Specifically, we first apply a linear attention mechanism to estimate the LLM's reasoning state, which records the historical reasoning information from previous reasoning steps. Then, based on the query prompt and the reasoning state, the LLM can efficiently perform the current reasoning step and update the state. With the linear attention, each token in the current reasoning step can directly retrieve relevant historical reasoning information from the reasoning state, without explicitly attending to tokens in previous reasoning steps. In this way, the computational complexity of attention is reduced from quadratic to linear, significantly improving the reasoning efficiency of LLMs. In addition, we propose a state-based reasoning strategy to mitigate the over-thinking issue caused by noisy reasoning steps. Extensive experiments across multiple datasets and model sizes demonstrate that our framework not only improves the reasoning efficiency of LLMs but also enhances their reasoning performance. 

---
# Cross-Modal Memory Compression for Efficient Multi-Agent Debate 

**Authors**: Jing Wu, Yue Sun, Tianpei Xie, Suiyao Chen, Jingyuan Bao, Yaopengxiao Xu, Gaoyuan Du, Inseok Heo, Alexander Gutfraind, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00454)  

**Abstract**: Multi-agent debate can improve reasoning quality and reduce hallucinations, but it incurs rapidly growing context as debate rounds and agent count increase. Retaining full textual histories leads to token usage that can exceed context limits and often requires repeated summarization, adding overhead and compounding information loss. We introduce DebateOCR, a cross-modal compression framework that replaces long textual debate traces with compact image representations, which are then consumed through a dedicated vision encoder to condition subsequent rounds. This design compresses histories that commonly span tens to hundreds of thousands of tokens, cutting input tokens by more than 92% and yielding substantially lower compute cost and faster inference across multiple benchmarks. We further provide a theoretical perspective showing that diversity across agents supports recovery of omitted information: although any single compressed history may discard details, aggregating multiple agents' compressed views allows the collective representation to approach the information bottleneck with exponentially high probability. 

---
# CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling 

**Authors**: Runsong Zhao, Shilei Liu, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Yujin Yuan, Tong Xiao, Jingbo Zhu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.01766)  

**Abstract**: The quadratic complexity and indefinitely growing key-value (KV) cache of standard Transformers pose a major barrier to long-context processing. To overcome this, we introduce the Collaborative Memory Transformer (CoMeT), a novel architecture that enables LLMs to handle arbitrarily long sequences with constant memory usage and linear time complexity. Designed as an efficient, plug-in module, CoMeT can be integrated into pre-trained models with only minimal fine-tuning. It operates on sequential data chunks, using a dual-memory system to manage context: a temporary memory on a FIFO queue for recent events, and a global memory with a gated update rule for long-range dependencies. These memories then act as a dynamic soft prompt for the next chunk. To enable efficient fine-tuning on extremely long contexts, we introduce a novel layer-level pipeline parallelism strategy. The effectiveness of our approach is remarkable: a model equipped with CoMeT and fine-tuned on 32k contexts can accurately retrieve a passkey from any position within a 1M token sequence. On the SCROLLS benchmark, CoMeT surpasses other efficient methods and achieves performance comparable to a full-attention baseline on summarization tasks. Its practical effectiveness is further validated on real-world agent and user behavior QA tasks. The code is available at: this https URL 

---
# Softmax Linear Attention: Reclaiming Global Competition 

**Authors**: Mingwei Xu, Xuan Lin, Xinnan Guo, Wanqing Xu, Wanyun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.01744)  

**Abstract**: While linear attention reduces the quadratic complexity of standard Transformers to linear time, it often lags behind in expressivity due to the removal of softmax normalization. This omission eliminates \emph{global competition}, a critical mechanism that enables models to sharply focus on relevant information amidst long-context noise. In this work, we propose \textbf{Softmax Linear Attention (SLA)}, a framework designed to restore this competitive selection without sacrificing efficiency. By lifting the softmax operation from the token level to the head level, SLA leverages attention heads as coarse semantic slots, applying a competitive gating mechanism to dynamically select the most relevant subspaces. This reintroduces the ``winner-take-all'' dynamics essential for precise retrieval and robust long-context understanding. Distinct from prior methods that focus on refining local kernel functions, SLA adopts a broader perspective by exploiting the higher-level multi-head aggregation structure. Extensive experiments demonstrate that SLA consistently enhances state-of-the-art linear baselines (RetNet, GLA, GDN) across language modeling and long-context benchmarks, particularly in challenging retrieval scenarios where it significantly boosts robustness against noise, validating its capability to restore precise focus while maintaining linear complexity. 

---
# P-EAGLE: Parallel-Drafting EAGLE with Scalable Training 

**Authors**: Mude Hui, Xin Huang, Jaime Campos Salas, Yue Sun, Nathan Pemberton, Xiang Song, Ashish Khetan, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2602.01469)  

**Abstract**: Reasoning LLMs produce longer outputs, requiring speculative decoding drafters trained on extended sequences. Parallel drafting - predicting multiple tokens per forward pass - offers latency benefits over sequential generation, but training complexity scales quadratically with the product of sequence length and parallel positions, rendering long-context training impractical. We present P(arallel)-EAGLE, which transforms EAGLE from autoregressive to parallel multi-token prediction via a learnable shared hidden state. To scale training to long contexts, we develop a framework featuring attention mask pre-computation and sequence partitioning techniques, enabling gradient accumulation within individual sequences for parallel-prediction training. We implement P-EAGLE in vLLM and demonstrate speedups of 1.10-1.36x over autoregressive EAGLE-3 across GPT-OSS 120B, 20B, and Qwen3-Coder 30B. 

---
# Self-Attention at Constant Cost per Token via Symmetry-Aware Taylor Approximation 

**Authors**: Franz A. Heinsen, Leo Kozachkov  

**Link**: [PDF](https://arxiv.org/pdf/2602.00294)  

**Abstract**: The most widely used artificial intelligence (AI) models today are Transformers employing self-attention. In its standard form, self-attention incurs costs that increase with context length, driving demand for storage, compute, and energy that is now outstripping society's ability to provide them. To help address this issue, we show that self-attention is efficiently computable to arbitrary precision with constant cost per token, achieving orders-of-magnitude reductions in memory use and computation. We derive our formulation by decomposing the conventional formulation's Taylor expansion into expressions over symmetric chains of tensor products. We exploit their symmetry to obtain feed-forward transformations that efficiently map queries and keys to coordinates in a minimal polynomial-kernel feature basis. Notably, cost is fixed inversely in proportion to head size, enabling application over a greater number of heads per token than otherwise feasible. We implement our formulation and empirically validate its correctness. Our work enables unbounded token generation at modest fixed cost, substantially reducing the infrastructure and energy demands of large-scale Transformer models. The mathematical techniques we introduce are of independent interest. 

---
