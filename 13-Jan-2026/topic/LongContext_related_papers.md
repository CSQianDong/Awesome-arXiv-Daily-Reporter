# RLPO: Residual Listwise Preference Optimization for Long-Context Review Ranking 

**Authors**: Hao Jiang, Zhi Yang, Annan Wang, Yichi Zhang, Weisi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.07449)  

**Abstract**: Review ranking is pivotal in e-commerce for prioritizing diagnostic and authentic feedback from the deluge of user-generated content. While large language models have improved semantic assessment, existing ranking paradigms face a persistent trade-off in long-context settings. Pointwise scoring is efficient but often fails to account for list-level interactions, leading to miscalibrated top-$k$ rankings. Listwise approaches can leverage global context, yet they are computationally expensive and become unstable as candidate lists grow. To address this, we propose Residual Listwise Preference Optimization (RLPO), which formulates ranking as listwise representation-level residual correction over a strong pointwise LLM scorer. RLPO first produces calibrated pointwise scores and item representations, then applies a lightweight encoder over the representations to predict listwise score residuals, avoiding full token-level listwise processing. We also introduce a large-scale benchmark for long-context review ranking with human verification. Experiments show RLPO improves NDCG@k over strong pointwise and listwise baselines and remains robust as list length increases. 

---
# TeleMem: Building Long-Term and Multimodal Memory for Agentic AI 

**Authors**: Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06037)  

**Abstract**: Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark. 

---
# Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths 

**Authors**: Xuezhe Ma, Shicheng Wen, Linghao Jin, Bilge Acun, Ruihang Lai, Bohan Hou, Will Lin, Hao Zhang, Songlin Yang, Ryan Lee, Mengxi Wu, Jonathan May, Luke Zettlemoyer, Carole-Jean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06463)  

**Abstract**: Designing a unified neural network to efficiently and inherently process sequential data with arbitrary lengths is a central and challenging problem in sequence modeling. The design choices in Transformer, including quadratic complexity and weak length extrapolation, have limited their ability to scale to long sequences. In this work, we propose Gecko, a neural architecture that inherits the design of Mega and Megalodon (exponential moving average with gated attention), and further introduces multiple technical components to improve its capability to capture long range dependencies, including timestep decay normalization, sliding chunk attention mechanism, and adaptive working memory. In a controlled pretraining comparison with Llama2 and Megalodon in the scale of 7 billion parameters and 2 trillion training tokens, Gecko achieves better efficiency and long-context scalability. Gecko reaches a training loss of 1.68, significantly outperforming Llama2-7B (1.75) and Megalodon-7B (1.70), and landing close to Llama2-13B (1.67). Notably, without relying on any context-extension techniques, Gecko exhibits inherent long-context processing and retrieval capabilities, stably handling sequences of up to 4 million tokens and retrieving information from contexts up to $4\times$ longer than its attention window. Code: this https URL 

---
# Active Context Compression: Autonomous Memory Management in LLM Agents 

**Authors**: Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2601.07190)  

**Abstract**: Large Language Model (LLM) agents struggle with long-horizon software engineering tasks due to "Context Bloat." As interaction history grows, computational costs explode, latency increases, and reasoning capabilities degrade due to distraction by irrelevant past errors. Existing solutions often rely on passive, external summarization mechanisms that the agent cannot control. This paper proposes Focus, an agent-centric architecture inspired by the biological exploration strategies of Physarum polycephalum (slime mold). The Focus Agent autonomously decides when to consolidate key learnings into a persistent "Knowledge" block and actively withdraws (prunes) the raw interaction history. Using an optimized scaffold matching industry best practices (persistent bash + string-replacement editor), we evaluated Focus on N=5 context-intensive instances from SWE-bench Lite using Claude Haiku 4.5. With aggressive prompting that encourages frequent compression, Focus achieves 22.7% token reduction (14.9M -> 11.5M tokens) while maintaining identical accuracy (3/5 = 60% for both agents). Focus performed 6.0 autonomous compressions per task on average, with token savings up to 57% on individual instances. We demonstrate that capable models can autonomously self-regulate their context when given appropriate tools and prompting, opening pathways for cost-aware agentic systems without sacrificing task performance. 

---
# Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents 

**Authors**: Miao Su, Yucan Guo, Zhongni Hou, Long Bai, Zixuan Li, Yufei Zhang, Guojun Yin, Wei Lin, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.07468)  

**Abstract**: Memory enables Large Language Model (LLM) agents to perceive, store, and use information from past dialogues, which is essential for personalization. However, existing methods fail to properly model the temporal dimension of memory in two aspects: 1) Temporal inaccuracy: memories are organized by dialogue time rather than their actual occurrence time; 2) Temporal fragmentation: existing methods focus on point-wise memory, losing durative information that captures persistent states and evolving patterns. To address these limitations, we propose Temporal Semantic Memory (TSM), a memory framework that models semantic time for point-wise memory and supports the construction and utilization of durative memory. During memory construction, it first builds a semantic timeline rather than a dialogue one. Then, it consolidates temporally continuous and semantically related information into a durative memory. During memory utilization, it incorporates the query's temporal intent on the semantic timeline, enabling the retrieval of temporally appropriate durative memories and providing time-valid, duration-consistent context to support response generation. Experiments on LongMemEval and LoCoMo show that TSM consistently outperforms existing methods and achieves up to 12.2% absolute improvement in accuracy, demonstrating the effectiveness of the proposed method. 

---
# HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents 

**Authors**: Ningning Zhang, Xingxing Yang, Zhizhong Tan, Weiping Deng, Wenyong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06377)  

**Abstract**: Although long-term memory systems have made substantial progress in recent years, they still exhibit clear limitations in adaptability, scalability, and self-evolution under continuous interaction settings. Inspired by cognitive theories, we propose HiMem, a hierarchical long-term memory framework for long-horizon dialogues, designed to support memory construction, retrieval, and dynamic updating during sustained interactions. HiMem constructs cognitively consistent Episode Memory via a Topic-Aware Event--Surprise Dual-Channel Segmentation strategy, and builds Note Memory that captures stable knowledge through a multi-stage information extraction pipeline. These two memory types are semantically linked to form a hierarchical structure that bridges concrete interaction events and abstract knowledge, enabling efficient retrieval without sacrificing information fidelity. HiMem supports both hybrid and best-effort retrieval strategies to balance accuracy and efficiency, and incorporates conflict-aware Memory Reconsolidation to revise and supplement stored knowledge based on retrieval feedback. This design enables continual memory self-evolution over long-term use. Experimental results on long-horizon dialogue benchmarks demonstrate that HiMem consistently outperforms representative baselines in accuracy, consistency, and long-term reasoning, while maintaining favorable efficiency. Overall, HiMem provides a principled and scalable design paradigm for building adaptive and self-evolving LLM-based conversational agents. The code is available at this https URL. 

---
# Towards Infinite Length Extrapolation: A Unified Approach 

**Authors**: Nitin Vetcha  

**Link**: [PDF](https://arxiv.org/pdf/2601.06113)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, but their ability to process long sequences is fundamentally limited by the context window size during training. Existing length extrapolation methods often suffer from performance degradation or computational inefficiencies. We thereby use a unified framework that reinterprets positional encoding methods as a decomposition of the attention score into a multiplicative transformation and an additive bias. This perspective not only subsumes popular approaches such as relative position embeddings and attention-bias moderated approaches but also exposes their inherent limitations in handling long-range dependencies. To address these shortcomings, motivated by our framework, we introduce Adaptive Positional Encoding (APE), which leverages adaptive frequency modulation and an intricately designed decay bias that incorporates linear, logarithmic, and square-root terms. Our theoretical analysis establishes conditions for infinite-context extrapolation, ensuring that the softmax normalization remains well-defined over unbounded sequences while preserving long-distance correlations, entropy boundedness and gradient positional sensitivity. We substantiate our claims with an experimental case study on TinyStories dataset as well as a new synthetic dataset, \emph{Long Tiny Stories} featuring stories up to 32,000 words. Relevant code, dataset and model weights are available at this https URL. 

---
# Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning 

**Authors**: Yue Zhou, Xiaobo Guo, Belhassen Bayar, Srinivasan H. Sengamedu  

**Link**: [PDF](https://arxiv.org/pdf/2601.06282)  

**Abstract**: Long-term conversational agents face a fundamental scalability challenge as interactions extend over time: repeatedly processing entire conversation histories becomes computationally prohibitive. Current approaches attempt to solve this through memory frameworks that predominantly fragment conversations into isolated embeddings or graph representations and retrieve relevant ones in a RAG style. While computationally efficient, these methods often treat memory formation minimally and fail to capture the subtlety and coherence of human memory. We introduce Amory, a working memory framework that actively constructs structured memory representations through enhancing agentic reasoning during offline time. Amory organizes conversational fragments into episodic narratives, consolidates memories with momentum, and semanticizes peripheral facts into semantic memory. At retrieval time, the system employs coherence-driven reasoning over narrative structures. Evaluated on the LOCOMO benchmark for long-term reasoning, Amory achieves considerable improvements over previous state-of-the-art, with performance comparable to full context reasoning while reducing response time by 50%. Analysis shows that momentum-aware consolidation significantly enhances response quality, while coherence-driven retrieval provides superior memory coverage compared to embedding-based approaches. 

---
# Semantic Event Graphs for Long-Form Video Question Answering 

**Authors**: Aradhya Dixit, Tianxi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06097)  

**Abstract**: Long-form video question answering remains challenging for modern vision-language models, which struggle to reason over hour-scale footage without exceeding practical token and compute budgets. Existing systems typically downsample frames or feed dense visual embeddings to large-context language models, trading off temporal coverage against cost. We propose Semantic Event Graphs (SEG), a lightweight symbolic interface between video and language that replaces raw frames with compact temporal interaction logs. Our pipeline detects and tracks objects with YOLOv11, converts proximity patterns into START/END human-object events, and organizes them into a Temporal Scene Graph (TSG). At inference time, a query-aware pruning module identifies anchor entities and lexically relevant events, returning only a small subgraph which is verbalized and passed to Gemini 2.5 Flash for answer generation. On five YouTube videos (300-500 interactions each) and 120 automatically generated long-horizon questions, SEG achieves 65.0% accuracy using only 3.47k tokens per query, closely matching a full-log baseline (62.5% at 40.39k tokens) while reducing token usage by 91.4%. A short-context baseline restricted to the last 30 seconds collapses to 2.5% accuracy, underscoring the need for explicit temporal memory. These results show that symbolic temporal graphs can serve as an effective, plug-and-play memory layer for off-the-shelf vision-language models, preserving long-range reasoning ability while making long-form video question answering substantially more token- and cost-efficient. Code, logs, and event-extraction tools will be released for reproducibility. 

---
