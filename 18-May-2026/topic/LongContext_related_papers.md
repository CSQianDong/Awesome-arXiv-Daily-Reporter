# Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning 

**Authors**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15315)  

**Abstract**: LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one score and one transition matrix. We show that this formulation creates a modeling bottleneck: a single CRF transition prior must serve heterogeneous retention patterns, including contiguous semantic spans and sparse structural support lines. We propose LaMR (Latent Multi-Rubric), a structured pruning framework that decomposes code relevance into two interpretable quality dimensions, semantic evidence and dependency support, each modeled by a dedicated CRF with dimension-specific transition dynamics. A mixture-of-experts gating network dynamically weights the per-rubric emissions conditioned on the query, and a final CRF layer on the fused emissions produces the aggregate keep-or-prune decision. To supervise each dimension without additional annotation cost, we derive multi-rubric labels from the existing training corpus via AST-based program analysis, simultaneously denoising the teacher's binary labels. By effectively filtering distracting noise, LaMR frequently matches or even outperforms unpruned full-context baselines. Experiments on four benchmarks (SWE-Bench Verified, SWE-QA, LCC, LongCodeQA) show that LaMR wins 12 of 16 head-to-head multi-turn comparisons. It saves up to 31% more tokens on multi-turn agent tasks and improves Exact Match by up to +3.5 on single-turn tasks, while performance is frequently enhanced by denoising the context, and any remaining drops are marginal. 

---
# Argus: Evidence Assembly for Scalable Deep Research Agents 

**Authors**: Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16217)  

**Abstract**: Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens. 

---
# RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents 

**Authors**: Zijie Dai, Shiyuan Deng, Sheng Guan, Yizhou Tian, Xin Yao, Xiao Yan, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16045)  

**Abstract**: Memory systems often organize user-agent interactions as retrievable external memory and are crucial for long-running agents by overcoming the limited context windows of LLMs. However, existing memory systems invoke LLMs to process every incoming interaction for memory extraction, and such an eager memory consolidation scheme leads to substantial token consumption. To tackle this problem, we propose RecMem by rethinking when memory consolidation should be conducted. RecMem stores incoming interactions in a subconscious memory layer and encode them using lightweight embedding models for retrieval. LLMs are only invoked to extract episodic and semantic memory when sustained recurrence are observed for semantically similar interactions. Such recurrence-based consolidation works because these interactions correspond to a semantic cluster with rich information and thus are worth extraction and summarization. To improve accuracy, RecMem also incorporates a semantic refinement mechanism that recovers the fine-grained facts omitted by memory extraction. Experiments show that RecMem reduces the memory construction token cost of three SOTA memory systems by up to 87% while exceeding their accuracy. 

---
# Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation 

**Authors**: Shuaiyi Li, Zhisong Zhang, Yan Wang, Lei Zhu, Dongyang Ma, Chenlong Deng, Yang Deng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.15913)  

**Abstract**: Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention. 

---
# RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably 

**Authors**: Yufeng Du, Phillip Harris, Minyang Tian, Eliu A Huerta, Srikanth Ronanki, Subendhu Rongali, Aram Galstyan, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15514)  

**Abstract**: We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models. 

---
# GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding 

**Authors**: Fanxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15250)  

**Abstract**: Multi-head Latent Attention (MLA), the attention used in DeepSeek-V2/V3, jointly compresses keys and values into a low-rank latent and matches the H100 roofline almost perfectly. Its trained weights, however, expose only one decoding path - an absorbed MQA form - which ties efficient inference to H100-class compute-bandwidth ratios, forfeits tensor parallelism along the head axis, and yields no Multi-Token Prediction (MTP) gain on commodity inference GPUs such as the export-restricted H20. We propose Group-Query Latent Attention (GQLA), a minimal modification of MLA whose trained weights expose two algebraically equivalent decoding paths over the same parameters: an MQA-absorb path identical to MLA's, and a GQA path with a per-group expanded cache. The runtime picks the path that matches the target hardware - no retraining, no custom kernels - so a single set of GQLA weights pins the rooflines of both H100 (MQA-absorb, s_q=1) and H20 (GQA + MTP, s_q=2), while supporting up to 8-way zero-redundancy tensor parallelism on the GQA path. To avoid pretraining from scratch we extend TransMLA into TransGQLA, which converts a pretrained GQA checkpoint into a GQLA model; on LLaMA-3-8B it compresses the per-token KV cache to 28.125% of the GQA baseline on the MQA-absorb path while structurally preserving GQA-level traffic on the per-group path. 

---
# STS: Efficient Sparse Attention with Speculative Token Sparsity 

**Authors**: Ceyu Xu, Jiangnan Yu, Yongji Wu, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.15508)  

**Abstract**: The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to dense attention. STS establishes a new state-of-the-art on the sparsity-accuracy trade-off, outperforming prior techniques by enabling higher sparsity levels for a given accuracy budget. 

---
