# Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents 

**Authors**: Natchanon Pollertlam, Witchayut Kornsuwannawit  

**Link**: [PDF](https://arxiv.org/pdf/2603.04814)  

**Abstract**: Persistent conversational AI systems face a choice between passing full conversation histories to a long-context large language model (LLM) and maintaining a dedicated memory system that extracts and retrieves structured facts. We compare a fact-based memory system built on the Mem0 framework against long-context LLM inference on three memory-centric benchmarks - LongMemEval, LoCoMo, and PersonaMemv2 - and evaluate both architectures on accuracy and cumulative API cost. Long-context GPT-5-mini achieves higher factual recall on LongMemEval and LoCoMo, while the memory system is competitive on PersonaMemv2, where persona consistency depends on stable, factual attributes suited to flat-typed extraction. We construct a cost model that incorporates prompt caching and show that the two architectures have structurally different cost profiles: long-context inference incurs a per-turn charge that grows with context length even under caching, while the memory system's per-turn read cost remains roughly fixed after a one-time write phase. At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns, with the break-even point decreasing as context length grows. These results characterize the accuracy-cost trade-off between the two approaches and provide a concrete criterion for selecting between them in production deployments. 

---
# Stacked from One: Multi-Scale Self-Injection for Context Window Extension 

**Authors**: Wei Han, Pan Zhou, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04759)  

**Abstract**: The limited context window of contemporary large language models (LLMs) remains a primary bottleneck for their broader application across diverse domains. Although continual pre-training on long-context data offers a straightforward solution, it incurs prohibitive data acquisition and computational costs. To address this challenge, we propose~\modelname, a novel framework based on multi-grained context compression and query-aware information acquisition. SharedLLM comprises two stacked short-context LLMs: a lower model serving as a compressor and an upper model acting as a decoder. The lower model compresses long inputs into compact, multi-grained representations, which are then forwarded to the upper model for context-aware processing. To maximize efficiency, this information transfer occurs exclusively at the lowest layers, bypassing lengthy forward passes and redundant cross-attention operations. This entire process, wherein the upper and lower models are derived from the same underlying LLM layers, is termed~\textit{self-injection}. To support this architecture, a specialized tree-based data structure enables the efficient encoding and query-aware retrieval of contextual information. Despite being trained on sequences of only 8K tokens, \modelname~effectively generalizes to inputs exceeding 128K tokens. Across a comprehensive suite of long-context modeling and understanding benchmarks, \modelname~achieves performance superior or comparable to strong baselines, striking an optimal balance between efficiency and accuracy. Furthermore, these design choices allow \modelname~to substantially reduce the memory footprint and yield notable inference speedups ($2\times$ over streaming and $3\times$ over encoder-decoder architectures). 

---
# EchoGuard: An Agentic Framework with Knowledge-Graph Memory for Detecting Manipulative Communication in Longitudinal Dialogue 

**Authors**: Ratna Kandala, Niva Manchanda, Akshata Kishore Moharir, Ananth Kandala  

**Link**: [PDF](https://arxiv.org/pdf/2603.04815)  

**Abstract**: Manipulative communication, such as gaslighting, guilt-tripping, and emotional coercion, is often difficult for individuals to recognize. Existing agentic AI systems lack the structured, longitudinal memory to track these subtle, context-dependent tactics, often failing due to limited context windows and catastrophic forgetting. We introduce EchoGuard, an agentic AI framework that addresses this gap by using a Knowledge Graph (KG) as the agent's core episodic and semantic memory. EchoGuard employs a structured Log-Analyze-Reflect loop: (1) users log interactions, which the agent structures as nodes and edges in a personal, episodic KG (capturing events, emotions, and speakers); (2) the system executes complex graph queries to detect six psychologically-grounded manipulation patterns (stored as a semantic KG); and (3) an LLM generates targeted Socratic prompts grounded by the subgraph of detected patterns, guiding users toward self-discovery. This framework demonstrates how the interplay between agentic architectures and Knowledge Graphs can empower individuals in recognizing manipulative communication while maintaining personal autonomy and safety. We present the theoretical foundation, framework design, a comprehensive evaluation strategy, and a vision to validate this approach. 

---
# VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory 

**Authors**: Yuheng Lei, Zhixuan Liang, Hongyuan Zhang, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.04910)  

**Abstract**: Imitation learning from human demonstrations has achieved significant success in robotic control, yet most visuomotor policies still condition on single-step observations or short-context histories, making them struggle with non-Markovian tasks that require long-term memory. Simply enlarging the context window incurs substantial computational and memory costs and encourages overfitting to spurious correlations, leading to catastrophic failures under distribution shift and violating real-time constraints in robotic systems. By contrast, humans can compress important past experiences into long-term memories and exploit them to solve tasks throughout their lifetime. In this paper, we propose VPWEM, a non-Markovian visuomotor policy equipped with working and episodic memories. VPWEM retains a sliding window of recent observation tokens as short-term working memory, and introduces a Transformer-based contextual memory compressor that recursively converts out-of-window observations into a fixed number of episodic memory tokens. The compressor uses self-attention over a cache of past summary tokens and cross-attention over a cache of historical observations, and is trained jointly with the policy. We instantiate VPWEM on diffusion policies to exploit both short-term and episode-wide information for action generation with nearly constant memory and computation per step. Experiments demonstrate that VPWEM outperforms state-of-the-art baselines including diffusion policies and vision-language-action (VLA) models by more than 20% on the memory-intensive manipulation tasks in MIKASA and achieves an average 5% improvement on the mobile manipulation benchmark MoMaRT. Code is available at this https URL. 

---
# VSPrefill: Vertical-Slash Sparse Attention with Lightweight Indexing for Long-Context Prefilling 

**Authors**: Chen Guanzhong  

**Link**: [PDF](https://arxiv.org/pdf/2603.04460)  

**Abstract**: The quadratic complexity of self-attention during the prefill phase impedes long-context inference in large language models. Existing sparse attention methods face a trade-off among context adaptivity, sampling overhead, and fine-tuning costs. We propose VSPrefill, a mechanism requiring lightweight training that uses the vertical-slash structural pattern in attention distributions. Our compact VSIndexer module predicts context-aware importance scores for vertical columns and slash diagonals from key-value representations augmented with RoPE. This approach constructs sparse masks with linear complexity without modifying the backbone parameters. During inference, an adaptive cumulative-threshold strategy allocates sparsity budgets per layer, while a fused kernel executes attention with on-the-fly index merging. Evaluated on Qwen3-4B-Instruct and LLaMA-3.1-8B-Instruct across the LongBench and RULER benchmarks, VSPrefill preserves 98.35% of the full attention accuracy while delivering a 4.95x average speedup at a context length of 128k. These results establish a new Pareto frontier in the trade-off between accuracy and efficiency. 

---
