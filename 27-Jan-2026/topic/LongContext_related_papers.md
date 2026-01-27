# Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction 

**Authors**: Weijiang Lai, Beihong Jin, Di Zhang, Siru Chen, Jiongyan Zhang, Yuhang Gou, Jian Dong, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17836)  

**Abstract**: In recent years, the success of large language models (LLMs) has driven the exploration of scaling laws in recommender systems. However, models that demonstrate scaling laws are actually challenging to deploy in industrial settings for modeling long sequences of user behaviors, due to the high computational complexity of the standard self-attention mechanism. Despite various sparse self-attention mechanisms proposed in other fields, they are not fully suited for recommendation scenarios. This is because user behaviors exhibit personalization and temporal characteristics: different users have distinct behavior patterns, and these patterns change over time, with data from these users differing significantly from data in other fields in terms of distribution. To address these challenges, we propose SparseCTR, an efficient and effective model specifically designed for long-term behaviors of users. To be precise, we first segment behavior sequences into chunks in a personalized manner to avoid separating continuous behaviors and enable parallel processing of sequences. Based on these chunks, we propose a three-branch sparse self-attention mechanism to jointly identify users' global interests, interest transitions, and short-term interests. Furthermore, we design a composite relative temporal encoding via learnable, head-specific bias coefficients, better capturing sequential and periodic relationships among user behaviors. Extensive experimental results show that SparseCTR not only improves efficiency but also outperforms state-of-the-art methods. More importantly, it exhibits an obvious scaling law phenomenon, maintaining performance improvements across three orders of magnitude in FLOPs. In online A/B testing, SparseCTR increased CTR by 1.72\% and CPM by 1.41\%. Our source code is available at this https URL. 

---
# Exploring Fine-Tuning for In-Context Retrieval and Efficient KV-Caching in Long-Context Language Models 

**Authors**: Francesco Maria Molfese, Momchil Hardalov, Rexhina Blloshmi, Bill Byrne, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2601.18527)  

**Abstract**: With context windows of millions of tokens, Long-Context Language Models (LCLMs) can encode entire document collections, offering a strong alternative to conventional retrieval-augmented generation (RAG). However, it remains unclear whether fine-tuning strategies can improve long-context performance and translate to greater robustness under KV-cache compression techniques. In this work, we investigate which training strategies most effectively enhance LCLMs' ability to identify and use relevant information, as well as enhancing their robustness under KV-cache compression. Our experiments show substantial in-domain improvements, achieving gains of up to +20 points over the base model. However, out-of-domain generalization remains task dependent with large variance -- LCLMs excels on finance questions (+9 points), while RAG shows stronger performance on multiple-choice questions (+6 points) over the baseline models. Finally, we show that our fine-tuning approaches bring moderate improvements in robustness under KV-cache compression, with gains varying across tasks. 

---
# FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning 

**Authors**: Lin Sun, Linglin Zhang, Jingang Huang, Change Jia, Zhengwei Cheng, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18116)  

**Abstract**: The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis.
We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs.
Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval. 

---
# S$^3$-Attention:Attention-Aligned Endogenous Retrieval for Memory-Bounded Long-Context Inference 

**Authors**: Qingsen Ma, Dianyun Wang, Yaoye Wang, Lechen Ning, Sujie Zhu, Xiaohang Zhang, Jiaming Lyu, Linhao Ren, Zhenbo Xu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2601.17702)  

**Abstract**: Large language models are increasingly applied to multi-document and long-form inputs, yet long-context inference remains memory- and noise-inefficient. Key-value (KV) caching scales linearly with context length, while external retrieval methods often return lexically similar but causally irrelevant passages.
We present S3-Attention, a memory-first inference-time framework that treats long-context processing as attention-aligned endogenous retrieval. S3-Attention decodes transient key and query projections into top-k sparse feature identifiers using lightweight sparse autoencoders, and constructs a CPU-based inverted index mapping features to token positions or spans during a single streaming scan. This design allows the KV cache to be discarded entirely and bounds GPU memory usage by the scan chunk size.
At generation time, feature co-activation is used to retrieve compact evidence spans, optionally fused with BM25 for exact lexical matching. Under a unified LongBench evaluation protocol with fixed prompting, decoding, and matched token budgets, S3-Hybrid closely matches full-context inference across multiple model families and improves robustness in several information-dense settings. We also report an engineering limitation of the current prototype, which incurs higher wall-clock latency than optimized full-KV baselines, motivating future kernel-level optimization. 

---
# Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers 

**Authors**: Zecheng Tang, Quantong Qiu, Yi Yang, Zhiyi Hong, Haiya Xiang, Kebin Liu, Qingqing Dang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17367)  

**Abstract**: The quadratic complexity of standard attention mechanisms poses a significant scalability bottleneck for large language models (LLMs) in long-context scenarios. While hybrid attention strategies that combine sparse and full attention within a single model offer a viable solution, they typically employ static computation ratios (i.e., fixed proportions of sparse versus full attention) and fail to adapt to the varying sparsity sensitivities of downstream tasks during inference. To address this issue, we propose Elastic Attention, which allows the model to dynamically adjust its overall sparsity based on the input. This is achieved by integrating a lightweight Attention Router into the existing pretrained model, which dynamically assigns each attention head to different computation modes. Within only 12 hours of training on 8xA800 GPUs, our method enables models to achieve both strong performance and efficient inference. Experiments across three long-context benchmarks on widely-used LLMs demonstrate the superiority of our method. 

---
# Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content 

**Authors**: Parth Bhalerao, Diola Dsouza, Ruiwen Guan, Oana Ignat  

**Link**: [PDF](https://arxiv.org/pdf/2601.17173)  

**Abstract**: Question answering systems are typically evaluated on factual correctness, yet many real-world applications-such as education and career guidance-require mentorship: responses that provide reflection and guidance. Existing QA benchmarks rarely capture this distinction, particularly in multilingual and long-form settings. We introduce MentorQA, the first multilingual dataset and evaluation framework for mentorship-focused question answering from long-form videos, comprising nearly 9,000 QA pairs from 180 hours of content across four languages. We define mentorship-focused evaluation dimensions that go beyond factual accuracy, capturing clarity, alignment, and learning value. Using MentorQA, we compare Single-Agent, Dual-Agent, RAG, and Multi-Agent QA architectures under controlled conditions. Multi-Agent pipelines consistently produce higher-quality mentorship responses, with especially strong gains for complex topics and lower-resource languages. We further analyze the reliability of automated LLM-based evaluation, observing substantial variation in alignment with human judgments. Overall, this work establishes mentorship-focused QA as a distinct research problem and provides a multilingual benchmark for studying agentic architectures and evaluation design in educational AI. The dataset and evaluation framework are released at this https URL. 

---
# Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction 

**Authors**: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun  

**Link**: [PDF](https://arxiv.org/pdf/2601.17668)  

**Abstract**: Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. The results are consistent across a wide range of tasks, including long-context understanding, code comprehension, and mathematical reasoning, demonstrating the generality of our approach. 

---
# When Personalization Legitimizes Risks: Uncovering Safety Vulnerabilities in Personalized Dialogue Agents 

**Authors**: Jiahe Guo, Xiangran Guo, Yulin Hu, Zimo Long, Xingyu Sui, Xuda Zhi, Yongbo Huang, Hao He, Weixiang Zhao, Yanyan Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.17887)  

**Abstract**: Long-term memory enables large language model (LLM) agents to support personalized and sustained interactions. However, most work on personalized agents prioritizes utility and user experience, treating memory as a neutral component and largely overlooking its safety implications. In this paper, we reveal intent legitimation, a previously underexplored safety failure in personalized agents, where benign personal memories bias intent inference and cause models to legitimize inherently harmful queries. To study this phenomenon, we introduce PS-Bench, a benchmark designed to identify and quantify intent legitimation in personalized interactions. Across multiple memory-augmented agent frameworks and base LLMs, personalization increases attack success rates by 15.8%-243.7% relative to stateless baselines. We further provide mechanistic evidence for intent legitimation from internal representations space, and propose a lightweight detection-reflection method that effectively reduces safety degradation. Overall, our work provides the first systematic exploration and evaluation of intent legitimation as a safety failure mode that naturally arises from benign, real-world personalization, highlighting the importance of assessing safety under long-term personal context. WARNING: This paper may contain harmful content. 

---
# RegGuard: AI-Powered Retrieval-Enhanced Assistant for Pharmaceutical Regulatory Compliance 

**Authors**: Siyuan Yang, Xihan Bian, Jiayin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17826)  

**Abstract**: The increasing frequency and complexity of regulatory updates present a significant burden for multinational pharmaceutical companies. Compliance teams must interpret evolving rules across jurisdictions, formats, and agencies, often manually, at high cost and risk of error. We introduce RegGuard, an industrial-scale AI assistant designed to automate the interpretation of heterogeneous regulatory texts and align them with internal corporate policies. The system ingests heterogeneous document sources through a secure pipeline and enhances retrieval and generation quality with two novel components: HiSACC (Hierarchical Semantic Aggregation for Contextual Chunking) semantically segments long documents into coherent units while maintaining consistency across non-contiguous sections. ReLACE (Regulatory Listwise Adaptive Cross-Encoder for Reranking), a domain-adapted cross-encoder built on an open-source model, jointly models user queries and retrieved candidates to improve ranking relevance. Evaluations in enterprise settings demonstrate that RegGuard improves answer quality specifically in terms of relevance, groundedness, and contextual focus, while significantly mitigating hallucination risk. The system architecture is built for auditability and traceability, featuring provenance tracking, access control, and incremental indexing, making it highly responsive to evolving document sources and relevant for any domain with stringent compliance demands. 

---
