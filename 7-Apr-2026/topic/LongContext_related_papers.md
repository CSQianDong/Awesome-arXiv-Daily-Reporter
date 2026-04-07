# LightThinker++: From Reasoning Compression to Memory Management 

**Authors**: Yuqi Zhu, Jintian Zhang, Zhenjie Wan, Yujie Luo, Shuofei Qiao, Zhengke Gui, Da Zheng, Lei Liang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03679)  

**Abstract**: Large language models (LLMs) excel at complex reasoning, yet their efficiency is limited by the surging cognitive overhead of long thought traces. In this paper, we propose LightThinker, a method that enables LLMs to dynamically compress intermediate thoughts into compact semantic representations. However, static compression often struggles with complex reasoning where the irreversible loss of intermediate details can lead to logical bottlenecks. To address this, we evolve the framework into LightThinker++, introducing Explicit Adaptive Memory Management. This paradigm shifts to behavioral-level management by incorporating explicit memory primitives, supported by a specialized trajectory synthesis pipeline to train purposeful memory scheduling. Extensive experiments demonstrate the framework's versatility across three dimensions. (1) LightThinker reduces peak token usage by 70% and inference time by 26% with minimal accuracy loss. (2) In standard reasoning, LightThinker++ slashes peak token usage by 69.9% while yielding a +2.42% accuracy gain under the same context budget for maximum performance. (3) Most notably, in long-horizon agentic tasks, it maintains a stable footprint beyond 80 rounds (a 60%-70% reduction), achieving an average performance gain of 14.8% across different complex scenarios. Overall, our work provides a scalable direction for sustaining deep LLM reasoning over extended horizons with minimal overhead. 

---
# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression 

**Authors**: Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04921)  

**Abstract**: Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions -- Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5x higher throughput or 10.7x KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency. TriAttention enables OpenClaw deployment on a single consumer GPU, where long context would otherwise cause out-of-memory with Full Attention. 

---
# DeonticBench: A Benchmark for Reasoning over Rules 

**Authors**: Guangyao Dou, Luis Brena, Akhil Deo, William Jurayj, Jingyu Zhang, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2604.04443)  

**Abstract**: Reasoning with complex, context-specific rules remains challenging for large language models (LLMs). In legal and policy settings, this manifests as deontic reasoning: reasoning about obligations, permissions, and prohibitions under explicit rules. While many recent benchmarks emphasize short-context mathematical reasoning, fewer focus on long-context, high-stakes deontic reasoning. To address this gap, we introduce DEONTICBENCH, a benchmark of 6,232 tasks across U.S. federal taxes, airline baggage policies, U.S. immigration administration, and U.S. state housing law. These tasks can be approached in multiple ways, including direct reasoning in language or with the aid of symbolic computation. Besides free-form chain-of-thought reasoning, DEONTICBENCH enables an optional solver-based workflow in which models translate statutes and case facts into executable Prolog, leading to formal problem interpretations and an explicit program trace. We release reference Prolog programs for all instances. Across frontier LLMs and coding models, best hard-subset performance reaches only 44.4% on SARA Numeric and 46.6 macro-F1 on Housing. We further study training with supervised fine-tuning and reinforcement learning for symbolic program generation. Although training improves Prolog generation quality, current RL methods still fail to solve these tasks reliably. Overall, DEONTICBENCH provides a benchmark for studying context-grounded rule reasoning in real-world domains under both symbolic and non-symbolic settings. 

---
# GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering 

**Authors**: Tianyi Zhang, Andreas Marfurt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04359)  

**Abstract**: Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis. 

---
# CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling 

**Authors**: Dejan Čugalj, Aleksandar Jevremovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.04250)  

**Abstract**: Modern Large Language Models (LLMs) rely on Transformer self-attention, which scales quadratically with sequence length. Recent linear-time alternatives, like State Space Models (SSMs), often suffer from signal degradation over extended contexts. We introduce the Continuous Acoustic Wave Network (CAWN), a fully continuous sequence-mixing architecture. Instead of discrete matrix-based attention, CAWN projects hidden states into multi-headed complex-domain phasors, achieving sequence mixing through a causal, $O(L)$ Phase Accumulation mechanism. To prevent signal degradation over ultra-long contexts, we introduce a dual-gated Selective Phase Resonance mechanism incorporating Frequency-Dependent Retention, Hard-Threshold Gating via Straight-Through Estimation, and a Temporal Syntax Cache to capture short-term local dependencies. We also replace standard dense linear projections with Depth-wise Harmonic Convolutions for optimal spatial frequency mixing, augmented by Block Attention Residuals for depth-wise state routing. Scaled to a 150M-parameter model, CAWN utilizes custom Triton kernels for hardware-efficient, true-complex phase accumulation in float32. Trained via a continuous streaming loop on a 100-Billion-token corpus, the prototype is evaluated at a 5-Billion-token milestone. Empirical evaluations via a Targeted Semantic Retrieval protocol demonstrate robust vocabulary acquisition and extended explicitly learned contextual denoising. By leveraging $O(1)$ state-passing via chunked prefill, the model retrieves targeted information across 2,000,000 tokens while strictly plateauing at 8.72 GB of Peak VRAM, empirically overcoming the $O(L^2)$ context memory wall. 

---
# Document-Level Numerical Reasoning across Single and Multiple Tables in Financial Reports 

**Authors**: Yi-Cheng Wang, Wei-An Wang, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03664)  

**Abstract**: Despite the strong language understanding abilities of large language models (LLMs), they still struggle with reliable question answering (QA) over long, structured documents, particularly for numerical reasoning. Financial annual reports exemplify this difficulty: financial statement analysis often hinges on accurate arithmetic, and analysts derive key indicators by integrating evidence scattered across multiple tables and narrative text. However, existing benchmarks focus largely on single-table settings, leaving cross-table document-level numerical reasoning underexplored. To address this gap, we introduce FinLongDocQA, a dataset for both single-table and cross-table financial numerical reasoning in long-context reports. Evaluating both closed-source and open-source LLMs on FinLongDocQA reveals two bottlenecks: (1) annual reports often exceed 129k tokens, exacerbating the context rot problem for locating relevant tables; and (2) even when relevant evidence is located, LLMs remain prone to errors in multi-step numerical reasoning. We propose FinLongDocAgent, a Multi-Agent Multi-Round Retrieval-Augmented Generation (RAG) approach that iteratively retrieves evidence, performs intermediate calculations, and verifies results across rounds. Experiments highlight the importance of iterative retrieval and verification for reliable numerical QA in long financial documents. 

---
# LPC-SM: Local Predictive Coding and Sparse Memory for Long-Context Language Modeling 

**Authors**: Keqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.03263)  

**Abstract**: Most current long-context language models still rely on attention to handle both local interaction and long-range state, which leaves relatively little room to test alternative decompositions of sequence modeling. We propose LPC-SM, a hybrid autoregressive architecture that separates local attention, persistent memory, predictive correction, and run-time control within the same block, and we use Orthogonal Novelty Transport (ONT) to govern slow-memory writes. We evaluate a 158M-parameter model in three stages spanning base language modeling, mathematical continuation, and 4096-token continuation. Removing mHC raises the Stage-A final LM loss from 12.630 to 15.127, while adaptive sparse control improves the Stage-B final LM loss from 12.137 to 10.787 relative to a matched fixed-ratio continuation. The full route remains stable at sequence length 4096, where Stage C ends with final LM loss 11.582 and improves the delayed-identifier diagnostic from 14.396 to 12.031 in key cross-entropy. Taken together, these results show that long-context autoregressive modeling can be organized around a broader division of labor than attention alone. 

---
# Why Attend to Everything? Focus is the Key 

**Authors**: Hengshuai Yao, Xing Chen, Ahmed Murtadha, Jin Li, Shuai Shao, Yasin Abbasi Yadkori, Guan Wang, Mingli Yuan, William Chen, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.03260)  

**Abstract**: We introduce Focus, a method that learns which token pairs matter rather than approximating all of them. Learnable centroids assign tokens to groups; distant attention is restricted to same-group pairs while local attention operates at full resolution. Because all model weights stay frozen, Focus is purely additive: centroid-only training (as few as 148K parameters) improves domain perplexity with zero degradation on downstream benchmarks--from 124M to 70B parameters, across five attention architectures. No existing efficient attention method achieves this in the retrofit setting. At 124M, Focus surpasses full attention (30.3 vs 31.4 PPL); trained from scratch at 7B scale (2B tokens), Focus again beats full attention (13.82 vs 13.89 PPL). At inference, restricting each token to its top-k highest-scoring groups discretizes the soft routing into a hard sparsity pattern, yielding 2x speedup while beating the pretrained baseline (41.3 vs 42.8 PPL); decomposing this pattern into two standard FlashAttention calls reaches 8.6x wall-clock speedup at 1M tokens with no custom kernels. Unlike LoRA, centroid routing preserves alignment: instruction-tuned models retain TruthfulQA scores after adaptation, while LoRA degrades at every learning rate and rank. Sinkhorn normalization enforces balanced groups as a hard constraint, and the resulting groups discover interpretable linguistic categories without supervision. 

---
# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents 

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04853)  

**Abstract**: Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memory system that integrates short-term, long-term episodic, and profile memory within a ground-truth-preserving architecture that stores entire conversational episodes and reduces lossy LLM-based extraction. MemMachine uses contextualized retrieval that expands nucleus matches with surrounding context, improving recall when relevant evidence spans multiple dialogue turns. Across benchmarks, MemMachine achieves strong accuracy-efficiency tradeoffs: on LoCoMo it reaches 0.9169 using gpt4.1-mini; on LongMemEvalS (ICLR 2025), a six-dimension ablation yields 93.0 percent accuracy, with retrieval-stage optimizations -- retrieval depth tuning (+4.2 percent), context formatting (+2.0 percent), search prompt design (+1.8 percent), and query bias correction (+1.4 percent) -- outperforming ingestion-stage gains such as sentence chunking (+0.8 percent). GPT-5-mini exceeds GPT-5 by 2.6 percent when paired with optimized prompts, making it the most cost-efficient setup. Compared to Mem0, MemMachine uses roughly 80 percent fewer input tokens under matched conditions. A companion Retrieval Agent adaptively routes queries among direct retrieval, parallel decomposition, or iterative chain-of-query strategies, achieving 93.2 percent on HotpotQA-hard and 92.6 percent on WikiMultiHop under randomized-noise conditions. These results show that preserving episodic ground truth while layering adaptive retrieval yields robust, efficient long-term memory for personalized LLM agents. 

---
# Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures 

**Authors**: Benjamin Rombaut  

**Link**: [PDF](https://arxiv.org/pdf/2604.03515)  

**Abstract**: LLM-based coding agents can localize bugs, generate patches, and run tests with diminishing human oversight, yet the scaffolding code that surrounds the language model (the control loop, tool definitions, state management, and context strategy) remains poorly understood. Existing surveys classify agents by abstract capabilities (tool use, planning, reflection) that cannot distinguish between architecturally distinct systems, and trajectory studies observe what agents do without examining the scaffold code that determines why. This paper presents a source-code-level architectural taxonomy derived from analysis of 13 open-source coding agent scaffolds at pinned commit hashes. Each agent is characterized across 12 dimensions organized into three layers: control architecture, tool and environment interface, and resource management. The analysis reveals that scaffold architectures resist discrete classification: control strategies range from fixed pipelines to Monte Carlo Tree Search, tool counts range from 0 to 37, and context compaction spans seven distinct strategies. Five loop primitives (ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search) function as composable building blocks that agents layer in different combinations; 11 of 13 agents compose multiple primitives rather than relying on a single control structure. Dimensions converge where external constraints dominate (tool capability categories, edit formats, execution isolation) and diverge where open design questions remain (context compaction, state management, multi-model routing). All taxonomic claims are grounded in file paths and line numbers, providing a reusable reference for researchers studying agent behavior and practitioners designing new scaffolds. 

---
