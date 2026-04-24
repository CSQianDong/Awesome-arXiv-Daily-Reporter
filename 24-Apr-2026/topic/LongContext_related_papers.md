# How English Print Media Frames Human-Elephant Conflicts in India 

**Authors**: Bonala Sai Punith, Salveru Jayati, Garima Shakya, Shubham Kumar Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2604.21496)  

**Abstract**: Human-elephant conflict (HEC) is rising across India as habitat loss and expanding human settlements force elephants into closer contact with people. While the ecological drivers of conflict are well-studied, how the news media portrays them remains largely unexplored. This work presents the first large-scale computational analysis of media framing of HEC in India, examining 1,968 full-length news articles consisting of 28,986 sentences, from a major English-language outlet published between January 2022 and September 2025. Using a multi-model sentiment framework that combines long-context transformers, large language models, and a domain-specific Negative Elephant Portrayal Lexicon, we quantify sentiment, extract rationale sentences, and identify linguistic patterns that contribute to negative portrayals of elephants. Our findings reveal a dominance of fear-inducing and aggression-related language. Since the media framing can shape public attitudes toward wildlife and conservation policy, such narratives risk reinforcing public hostility and undermining coexistence efforts. By providing a transparent, scalable methodology and releasing all resources through an anonymized repository, this study highlights how Web-scale text analysis can support responsible wildlife reporting and promote socially beneficial media practices. 

---
# StructMem: Structured Memory for Long-Horizon Behavior in LLMs 

**Authors**: Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.21748)  

**Abstract**: Long-term conversational agents need memory systems that capture relationships between events, not merely isolated facts, to support temporal reasoning and multi-hop question answering. Current approaches face a fundamental trade-off: flat memory is efficient but fails to model relational structure, while graph-based memory enables structured reasoning at the cost of expensive and fragile construction. To address these issues, we propose \textbf{StructMem}, a structure-enriched hierarchical memory framework that preserves event-level bindings and induces cross-event connections. By temporally anchoring dual perspectives and performing periodic semantic consolidation, StructMem improves temporal reasoning and multi-hop performance on \texttt{LoCoMo}, while substantially reducing token usage, API calls, and runtime compared to prior memory systems, see this https URL . 

---
# Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation 

**Authors**: Hanwen Gu, Chao Guo, Junle Wang, Wenda Xie, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.21253)  

**Abstract**: While LLMs demonstrate remarkable fluency in narrative generation, existing methods struggle to maintain global narrative coherence, contextual logical consistency, and smooth character development, often producing monotonous scripts with structural fractures. To this end, we introduce PLOTTER, a framework that performs narrative planning on structural graph representations instead of the direct sequential text representations used in existing work. Specifically, PLOTTER executes the Evaluate-Plan-Revise cycle on the event graph and character graph. By diagnosing and repairing issues of the graph topology under rigorous logical constraints, the model optimizes the causality and narrative skeleton before complete context generation. Experiments demonstrate that PLOTTER significantly outperforms representative baselines across diverse narrative scenarios. These findings verify that planning narratives on structural graph representations-rather than directly on text-is crucial to enhance the long context reasoning of LLMs in complex narrative generation. 

---
# EngramaBench: Evaluating Long-Term Conversational Memory with Structured Graph Retrieval 

**Authors**: Julian Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.21229)  

**Abstract**: Large language model assistants are increasingly expected to retain and reason over information accumulated across many sessions. We introduce EngramaBench, a benchmark for long-term conversational memory built around five personas, one hundred multi-session conversations, and one hundred fifty queries spanning factual recall, cross-space integration, temporal reasoning, adversarial abstention, and emergent synthesis. We evaluate Engrama, a graph-structured memory system, against GPT-4o full-context prompting and Mem0, an open-source vector-retrieval memory system. All three use the same answering model (GPT-4o), isolating the effect of memory architecture. GPT-4o full-context achieves the highest composite score (0.6186), while Engrama scores 0.5367 globally but is the only system to score higher than full-context prompting on cross-space reasoning (0.6532 vs. 0.6291, n=30). Mem0 is cheapest but substantially weaker (0.4809). Ablations reveal that the components driving Engrama's cross-space advantage trade off against global composite score, exposing a systems-level tension between structured memory specialization and aggregate optimization. 

---
# Absorber LLM: Harnessing Causal Synchronization for Test-Time Training 

**Authors**: Zhixin Zhang, Shabo Zhang, Chengcan Wu, Zeming Wei, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.20915)  

**Abstract**: Transformers suffer from a high computational cost that grows with sequence length for self-attention, making inference in long streams prohibited by memory consumption. Constant-memory alternatives such as RNNs and SSMs compress history into states with fixed size and thus lose long-tail dependencies, while methods that memorize contexts into parameters, such as Test-Time Training (TTT), are prone to overfitting token-level projection and fail to preserve the causal effect of context in pretrained LLMs. We propose Absorber LLM, which formulates long-context retention as a self-supervised causal synchronization: after absorbing historical contexts into parameters, a contextless model should match the original model with full context on future generations. We optimize this objective by synchronizing internal behaviors of the updated model with the original one, ensuring context absorption and generalization. Experiments on long-context and streaming benchmarks show that Absorber LLM reduces inference memory and improves accuracy over prior parameter-as-memory baselines. 

---
# Omission Constraints Decay While Commission Constraints Persist in Long-Context LLM Agents 

**Authors**: Yeran Gamage  

**Link**: [PDF](https://arxiv.org/pdf/2604.20911)  

**Abstract**: LLM agents deployed in production operate under operator-defined behavioral policies (system-prompt instructions such as prohibitions on credential disclosure, data exfiltration, and unauthorized output) that safety evaluations assume hold throughout a conversation. Prohibition-type constraints decay under context pressure while requirement-type constraints persist; we term this asymmetry Security-Recall Divergence (SRD). In a 4,416-trial three-arm causal study across 12 models and 8 providers at six conversation depths, omission compliance falls from 73% at turn 5 to 33% at turn 16 while commission compliance holds at 100% (Mistral Large 3, $p < 10^{-33}$). In the two models with token-matched padding controls, schema semantic content accounts for 62-100% of the dilution effect. Re-injecting constraints before the per-model Safe Turn Depth (STD) restores compliance without retraining. Production security policies consist of prohibitions such as never revealing credentials, never executing untrusted code, and never forwarding user data. Commission-type audit signals remain healthy while omission constraints have already failed, leaving the failure invisible to standard monitoring. 

---
# DWTSumm: Discrete Wavelet Transform for Document Summarization 

**Authors**: Rana Salama, Abdou Youssef, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2604.21070)  

**Abstract**: Summarizing long, domain-specific documents with large language models (LLMs) remains challenging due to context limitations, information loss, and hallucinations, particularly in clinical and legal settings. We propose a Discrete Wavelet Transform (DWT)-based multi-resolution framework that treats text as a semantic signal and decomposes it into global (approximation) and local (detail) components. Applied to sentence- or word-level embeddings, DWT yields compact representations that preserve overall structure and critical domain-specific details, which are used directly as summaries or to guide LLM generation. Experiments on clinical and legal benchmarks demonstrate comparable ROUGE-L scores. Compared to a GPT-4o baseline, the DWT based summarization consistently improve semantic similarity and grounding, achieving gains of over 2% in BERTScore, more than 4\% in Semantic Fidelity, factual consistency in legal tasks, and large METEOR improvements indicative of preserved domain-specific semantics. Across multiple embedding models, Fidelity reaches up to 97%, suggesting that DWT acts as a semantic denoising mechanism that reduces hallucinations and strengthens factual grounding. Overall, DWT provides a lightweight, generalizable method for reliable long-document and domain-specific summarization with LLMs. 

---
# Spatial Metaphors for LLM Memory: A Critical Analysis of the MemPalace Architecture 

**Authors**: Robin Dey, Panyanon Viradecha  

**Link**: [PDF](https://arxiv.org/pdf/2604.21284)  

**Abstract**: MemPalace is an open-source AI memory system that applies the ancient method of loci (memory palace) spatial metaphor to organize long-term memory for large language models; launched in April 2026, it accumulated over 47,000 GitHub stars in its first two weeks and claims state-of-the-art retrieval performance on the LongMemEval benchmark (96.6% Recall@5) without requiring any LLM inference at write time. Through independent codebase analysis, benchmark replication, and comparison with competing systems, we find that MemPalace's headline retrieval performance is attributable primarily to its verbatim storage philosophy combined with ChromaDB's default embedding model (all-MiniLM-L6-v2), rather than to its spatial organizational metaphor per se -- the palace hierarchy (Wings->Rooms->Closets->Drawers) operates as standard vector database metadata filtering, an effective but well-established technique. However, MemPalace makes several genuinely novel contributions: (1) a contrarian verbatim-first storage philosophy that challenges extraction-based competitors, (2) an extremely low wake-up cost (approximately 170 tokens) through its four-layer memory stack, (3) a fully deterministic, zero-LLM write path enabling offline operation at zero API cost, and (4) the first systematic application of spatial memory metaphors as an organizing principle for AI memory systems. We also note that the competitive landscape is evolving rapidly, with Mem0's April 2026 token-efficient algorithm raising their LongMemEval score from approximately 49% to 93.4%, narrowing the gap between extraction-based and verbatim approaches. Our analysis concludes that MemPalace represents significant architectural insight wrapped in overstated claims -- a pattern common in rapidly adopted open-source projects where marketing velocity exceeds scientific rigor. 

---
# Sub-Token Routing in LoRA for Adaptation and Query-Aware KV Compression 

**Authors**: Wei Jiang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.21335)  

**Abstract**: Sub-token routing offers a finer control axis for transformer efficiency than the coarse units used in most prior work, such as tokens, pages, heads, or layers. In this paper, we study routing within a token representation itself in LoRA-adapted transformers. The motivation is that a relevant token need not be internally uniform: under a retention budget, preserved value groups are distributed unevenly both across tokens and within tokens, which suggests that KV compression need not be an all-or-nothing decision at token level. We study this fine-grained routing mechanism in two settings. For compression-aware language modeling, we introduce a query-independent design that combines routed subspace LoRA with value-group routing on the KV path. For downstream-task-preserving KV compression, we introduce a query-aware design in which a predictor-based selector allocates a global retention budget over context-token/value-group pairs using query-conditioned relevance. Experiments show that the query-independent design improves the quality-compression tradeoff for language modeling, while the query-aware design preserves downstream behavior under reduced KV budgets. We further examine the relation between token-level and sub-token-level query-aware routing, and show that they form complementary compression axes: token-level methods determine which tokens survive globally, while sub-token routing determines how the surviving tokens are compressed internally. 

---
# The Root Theorem of Context Engineering 

**Authors**: Borja Odriozola Schick  

**Link**: [PDF](https://arxiv.org/pdf/2604.20874)  

**Abstract**: Every system that maintains a large language model conversation beyond a single session faces two inescapable constraints: the context window is finite, and information quality degrades with accumulated volume. We formalize these constraints as axioms and derive a single governing principle -- the Root Theorem of Context Engineering: \emph{maximize signal-to-token ratio within bounded, lossy channels.} From this principle, we derive five consequences without additional assumptions: (1)~a quality function $F(P)$ that degrades monotonically with injected token volume, independent of window size; (2)~the independence of signal and token count as optimization variables; (3)~a necessary gate mechanism triggered by fidelity thresholds, not capacity limits; (4)~the inevitability of homeostatic persistence -- accumulate, compress, rewrite, shed -- as the only architecture that sustains understanding indefinitely; and (5)~the self-referential property that the compression mechanism operates inside the channel it compresses, requiring an external verification gate. We show that append-only systems necessarily exceed their effective window in finite time, that retrieval-augmented generation solves search but not continuity, and that the theorem's constraint structure converges with biological memory architecture through independent derivation from shared principles. Engineering proof is provided through a 60+-session persistent architecture demonstrating stable memory footprint under continuous operation -- the divergence prediction made concrete. The Root Theorem establishes context engineering as an information-theoretic discipline with formal foundations, distinct from prompt engineering in both scope and method. Shannon solved point-to-point transmission. Context engineering solves continuity. 

---
