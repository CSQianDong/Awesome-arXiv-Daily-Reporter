# SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG 

**Authors**: Xuechen Zhang, Koustava Goswami, Samet Oymak, Jiasi Chen, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2602.22225)  

**Abstract**: Retrieval-augmented generation (RAG) has strong potential for producing accurate and factual outputs by combining language models (LMs) with evidence retrieved from large text corpora. However, current pipelines are limited by static chunking and flat retrieval: documents are split into short, predetermined, fixed-size chunks, embeddings are retrieved uniformly, and generation relies on whatever chunks are returned. This design brings challenges, as retrieval quality is highly sensitive to chunk size, often introduces noise from irrelevant or misleading chunks, and scales poorly to large corpora. We present SmartChunk retrieval, a query-adaptive framework for efficient and robust long-document question answering (QA). SmartChunk uses (i) a planner that predicts the optimal chunk abstraction level for each query, and (ii) a lightweight compression module that produces high-level chunk embeddings without repeated summarization. By adapting retrieval granularity on the fly, SmartChunk balances accuracy with efficiency and avoids the drawbacks of fixed strategies. Notably, our planner can reason about chunk abstractions through a novel reinforcement learning scheme, STITCH, which boosts accuracy and generalization. To reflect real-world applications, where users face diverse document types and query styles, we evaluate SmartChunk on five QA benchmarks plus one out-of-domain dataset. Across these evaluations, SmartChunk outperforms state-of-the-art RAG baselines, while reducing cost. Further analysis demonstrates strong scalability with larger corpora and consistent gains on out-of-domain datasets, highlighting its effectiveness as a general framework for adaptive retrieval. 

---
# AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications 

**Authors**: Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.22769)  

**Abstract**: Large Language Models (LLMs) are deployed as autonomous agents in increasingly complex applications, where enabling long-horizon memory is critical for achieving strong performance. However, a significant gap exists between practical applications and current evaluation standards for agent memory: existing benchmarks primarily focus on dialogue-centric, human-agent interactions. In reality, agent memory consists of a continuous stream of agent-environment interactions that are primarily composed of machine-generated representations. To bridge this gap, we introduce AMA-Bench (Agent Memory with Any length), which evaluates long-horizon memory for LLMs in real agentic applications. It features two key components: (1) a set of real-world agentic trajectories across representative agentic applications, paired with expert-curated QA, and (2) a set of synthetic agentic trajectories that scale to arbitrary horizons, paired with rule-based QA. Our comprehensive study shows that existing memory systems underperform on AMA-Bench primarily because they lack causality and objective information and are constrained by the lossy nature of similarity-based retrieval employed by many memory systems. To address these limitations, we propose AMA-Agent, an effective memory system featuring a causality graph and tool-augmented retrieval. Our results demonstrate that AMA-Agent achieves 57.22% average accuracy on AMA-Bench, surpassing the strongest memory system baselines by 11.16%. 

---
# SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning 

**Authors**: Sanjay Kariyappa, G. Edward Suh  

**Link**: [PDF](https://arxiv.org/pdf/2602.22603)  

**Abstract**: Long-running agentic tasks, such as deep research, require multi-hop reasoning over information distributed across multiple webpages and documents. In such tasks, the LLM context is dominated by tokens from external retrieval, causing memory usage to grow rapidly and limiting decode performance. While several KV cache compression techniques exist for long-context inputs, we find that existing heuristics fail to support multi-step reasoning models effectively. We address this challenge with SideQuest -- a novel approach that leverages the Large Reasoning Model (LRM) itself to perform KV cache compression by reasoning about the usefulness of tokens in its context. To prevent the tokens associated with this management process from polluting the model's memory, we frame KV cache compression as an auxiliary task executed in parallel to the main reasoning task. Our evaluations, using a model trained with just 215 samples, show that SideQuest reduces peak token usage by up to 65% on agentic tasks with minimal degradation in accuracy, outperforming heuristic-based KV cache compression techniques. 

---
# S2O: Early Stopping for Sparse Attention via Online Permutation 

**Authors**: Yu Zhang, Songwei Liu, Chenqian Yan, Sheng Lin, Beichen Ning, Fangmin Chen, Xing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.22575)  

**Abstract**: Attention scales quadratically with sequence length, fundamentally limiting long-context inference. Existing block-granularity sparsification can reduce latency, but coarse blocks impose an intrinsic sparsity ceiling, making further improvements difficult even with carefully engineered designs. We present S2O, which performs early stopping for sparse attention via online permutation. Inspired by virtual-to-physical address mapping in memory systems, S2O revisits and factorizes FlashAttention execution, enabling inference to load non-contiguous tokens rather than a contiguous span in the original order. Motivated by fine-grained structures in attention heatmaps, we transform explicit permutation into an online, index-guided, discrete loading policy; with extremely lightweight preprocessing and index-remapping overhead, it concentrates importance on a small set of high-priority blocks. Building on this importance-guided online permutation for loading, S2O further introduces an early-stopping rule: computation proceeds from high to low importance; once the current block score falls below a threshold, S2O terminates early and skips the remaining low-contribution blocks, thereby increasing effective sparsity and reducing computation under a controlled error budget.
As a result, S2O substantially raises the practical sparsity ceiling. On Llama-3.1-8B under a 128K context, S2O reduces single-operator MSE by 3.82$\times$ at matched sparsity, and reduces prefill compute density by 3.31$\times$ at matched MSE; meanwhile, it preserves end-to-end accuracy and achieves 7.51$\times$ attention and 3.81$\times$ end-to-end speedups. 

---
# Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents 

**Authors**: Cosmo Santoni  

**Link**: [PDF](https://arxiv.org/pdf/2602.22402)  

**Abstract**: As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base64 images, and metadata. A single-user case-study evaluation across 76 real-world coding sessions demonstrates that trimming remains economically viable under prompt caching, with the strongest gains in mixed tool-use sessions, which average 39% reduction and reach break-even within 10 turns. A reference implementation is available at this https URL. 

---
