# CelloAI: Leveraging Large Language Models for HPC Software Development in High Energy Physics 

**Authors**: Mohammad Atif, Kriti Chopra, Ozgur Kilic, Tianle Wang, Zhihua Dong, Charles Leggett, Meifeng Lin, Paolo Calafiura, Salman Habib  

**Link**: [PDF](https://arxiv.org/pdf/2508.16713)  

**Abstract**: Next-generation High Energy Physics (HEP) experiments will generate unprecedented data volumes, necessitating High Performance Computing (HPC) integration alongside traditional high-throughput computing. However, HPC adoption in HEP is hindered by the challenge of porting legacy software to heterogeneous architectures and the sparse documentation of these complex scientific codebases. We present CelloAI, a locally hosted coding assistant that leverages Large Language Models (LLMs) with retrieval-augmented generation (RAG) to support HEP code documentation and generation. This local deployment ensures data privacy, eliminates recurring costs and provides access to large context windows without external dependencies. CelloAI addresses two primary use cases, code documentation and code generation, through specialized components. For code documentation, the assistant provides: (a) Doxygen style comment generation for all functions and classes by retrieving relevant information from RAG sources (papers, posters, presentations), (b) file-level summary generation, and (c) an interactive chatbot for code comprehension queries. For code generation, CelloAI employs syntax-aware chunking strategies that preserve syntactic boundaries during embedding, improving retrieval accuracy in large codebases. The system integrates callgraph knowledge to maintain dependency awareness during code modifications and provides AI-generated suggestions for performance optimization and accurate refactoring. We evaluate CelloAI using real-world HEP applications from ATLAS, CMS, and DUNE experiments, comparing different embedding models for code retrieval effectiveness. Our results demonstrate the AI assistant's capability to enhance code understanding and support reliable code generation while maintaining the transparency and safety requirements essential for scientific computing environments. 

---
# VQL: An End-to-End Context-Aware Vector Quantization Attention for Ultra-Long User Behavior Modeling 

**Authors**: Kaiyuan Li, Yongxiang Tang, Yanhua Cheng, Yong Bai, Yanxiang Zeng, Chao Wang, Xialong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17125)  

**Abstract**: In large-scale recommender systems, ultra-long user behavior sequences encode rich signals of evolving interests. Extending sequence length generally improves accuracy, but directly modeling such sequences in production is infeasible due to latency and memory constraints. Existing solutions fall into two categories: (1) top-k retrieval, which truncates the sequence and may discard most attention mass when L >> k; and (2) encoder-based compression, which preserves coverage but often over-compresses and fails to incorporate key context such as temporal gaps or target-aware signals. Neither class achieves a good balance of low-loss compression, context awareness, and efficiency.
We propose VQL, a context-aware Vector Quantization Attention framework for ultra-long behavior modeling, with three innovations. (1) Key-only quantization: only attention keys are quantized, while values remain intact; we prove that softmax normalization yields an error bound independent of sequence length, and a codebook loss directly supervises quantization quality. This also enables L-free inference via offline caches. (2) Multi-scale quantization: attention heads are partitioned into groups, each with its own small codebook, which reduces quantization error while keeping cache size fixed. (3) Efficient context injection: static features (e.g., item category, modality) are directly integrated, and relative position is modeled via a separable temporal kernel. All context is injected without enlarging the codebook, so cached representations remain query-independent.
Experiments on three large-scale datasets (KuaiRand-1K, KuaiRec, TMALL) show that VQL consistently outperforms strong baselines, achieving higher accuracy while reducing inference latency, establishing a new state of the art in balancing accuracy and efficiency for ultra-long sequence recommendation. 

---
# Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering 

**Authors**: Julius Gun, Timo Oksanen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18093)  

**Abstract**: We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges. 

---
# ILRe: Intermediate Layer Retrieval for Context Compression in Causal Language Models 

**Authors**: Manlai Liang, Mandi Liu, Jiangzhou Ji, Huaijun Li, Haobo Yang, Yaohan He, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17892)  

**Abstract**: Large Language Models (LLMs) have demonstrated success across many benchmarks. However, they still exhibit limitations in long-context scenarios, primarily due to their short effective context length, quadratic computational complexity, and high memory overhead when processing lengthy inputs. To mitigate these issues, we introduce a novel context compression pipeline, called Intermediate Layer Retrieval (ILRe), which determines one intermediate decoder layer offline, encodes context by streaming chunked prefill only up to that layer, and recalls tokens by the attention scores between the input query and full key cache in that specified layer. In particular, we propose a multi-pooling kernels allocating strategy in the token recalling process to maintain the completeness of semantics. Our approach not only reduces the prefilling complexity from $O(L^2)$ to $O(L)$, but also achieves performance comparable to or better than the full context in the long context scenarios. Without additional post training or operator development, ILRe can process a single $1M$ tokens request in less than half a minute (speedup $\approx 180\times$) and scores RULER-$1M$ benchmark of $\approx 79.8$ with model Llama-3.1-UltraLong-8B-1M-Instruct on a Huawei Ascend 910B NPU. 

---
# Efficient Zero-Shot Long Document Classification by Reducing Context Through Sentence Ranking 

**Authors**: Prathamesh Kokate, Mitali Sarnaik, Manavi Khopade, Mukta Takalikar, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17490)  

**Abstract**: Transformer-based models like BERT excel at short text classification but struggle with long document classification (LDC) due to input length limitations and computational inefficiencies. In this work, we propose an efficient, zero-shot approach to LDC that leverages sentence ranking to reduce input context without altering the model architecture. Our method enables the adaptation of models trained on short texts, such as headlines, to long-form documents by selecting the most informative sentences using a TF-IDF-based ranking strategy. Using the MahaNews dataset of long Marathi news articles, we evaluate three context reduction strategies that prioritize essential content while preserving classification accuracy. Our results show that retaining only the top 50\% ranked sentences maintains performance comparable to full-document inference while reducing inference time by up to 35\%. This demonstrates that sentence ranking is a simple yet effective technique for scalable and efficient zero-shot LDC. 

---
# Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding 

**Authors**: Thi-Nhung Nguyen, Hoang Ngo, Dinh Phung, Thuy-Trang Vu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17005)  

**Abstract**: Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets. 

---
# RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System 

**Authors**: Zui Chen, Han Li, Xinhao Zhang, Xiaoyu Chen, Chunyin Dong, Yifeng Wang, Xin Cai, Su Zhang, Ziqi Li, Chi Ding, Jinxu Li, Shuai Wang, Dousheng Zhao, Sanhai Gao, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17590)  

**Abstract**: We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research. 

---
