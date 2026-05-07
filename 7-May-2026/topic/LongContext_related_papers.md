# The Impossibility Triangle of Long-Context Modeling 

**Authors**: Yan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.05066)  

**Abstract**: We identify and prove a fundamental trade-off governing long-sequence models: no model can simultaneously achieve (i) per-step computation independent of sequence length (Efficiency), (ii) state size independent of sequence length (Compactness), and (iii) the ability to recall a number of historical facts proportional to sequence length (Recall). We formalize this trade-off within an Online Sequence Processor abstraction that unifies Transformers, state space models, linear recurrent networks, and their hybrids. Using the Data Processing Inequality and Fano's Inequality, we prove that any model satisfying Efficiency and Compactness can recall at most O(poly(d)/log V) key-value pairs from a sequence of arbitrary length, where d is the model dimension and V is the vocabulary size. We classify 52 architectures published before March 2026 into the triangle, showing that each achieves at most two of the three properties and that hybrid architectures trace continuous trajectories in the interior. Experiments on synthetic associative recall tasks with five representative architectures validate the theoretical bound: empirical recall capacity lies strictly below the information-theoretic limit, and no architecture escapes the triangle. 

---
# Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall 

**Authors**: Joshua Adler, Guy Zehavi  

**Link**: [PDF](https://arxiv.org/pdf/2605.04897)  

**Abstract**: Extraction at ingestion is the wrong primitive for agent memory: content discarded before the query is known cannot be recovered at retrieval time. We propose True Memory, a six-layer architecture that shifts the center of the system from a storage schema to a multi-stage retrieval pipeline operating over events preserved verbatim. The full system runs as a single SQLite file on commodity CPU with no external database, vector index, graph store, or GPU. On LoCoMo (1,540 questions across 10 multi-session conversations), True Memory Pro reaches 93.0% accuracy (3-run mean) against 61.4% for Mem0, 65.4% for Supermemory, approximately 71% for Zep, and 94.5% for EverMemOS under a matched gpt-4.1-mini answer model. On LongMemEval (500 questions), True Memory Pro reaches 87.8% (3-run mean). On BEAM-1M (700 questions at the 1-million-token scale), True Memory Pro reaches 76.6% (3-run mean), above the prior published result of 73.9% for Hindsight. A 56-configuration ablation shows a 1.3-percentage-point spread within the top-performing configuration family. 

---
# Telegraph English: Semantic Prompt Compression via Structured Symbolic Rewriting 

**Authors**: Mikhail L. Arbuzov, Sisong Bei, Ziwei Dong, Dmitri Kalaev, Alexey A. Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2605.04426)  

**Abstract**: We introduce Telegraph English (TE), a prompt-compression protocol that rewrites natural language into a symbol-rich, formally-structured dialect. Where token-deletion methods such as LLMLingua-2 train a classifier to delete low-importance tokens at a fixed ratio, TE performs a full semantic rewrite: it decomposes the input into atomic fact lines, substitutes verbose phrases with $\sim$40 logical and relational symbols, and lets the compression ratio adapt to each document's information density. A consequence of the line-structure rule is that compression and semantic chunking become the same operation -- each output line is an independently addressable fact, so the compressed representation is simultaneously a semantic index. We evaluate TE on 4{,}081 question-answer pairs from LongBench-v2 across five OpenAI models and two difficulty levels. At roughly 50\% token reduction, TE preserves 99.1\% accuracy on key facts with GPT-4.1 and outperforms LLMLingua-2 at matched compression ratios on every model and task tested. The gap widens on smaller models -- up to 11 percentage points on fine-detail tasks -- suggesting that explicit relational structure compensates for limited model capacity. We release the grammar specification, compression prompt, benchmark data, and reference implementation. 

---
# SCOUT: Active Information Foraging for Long-Text Understanding with Decoupled Epistemic States 

**Authors**: Zhenliang Zhang, Wenqing Wang, Yong Hu, Yaming Yang, Jiaheng Gao, Chen Shen, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.04496)  

**Abstract**: Long-Text Understanding (LTU) at million-token scale requires balancing reasoning fidelity with computational efficiency. Frontier long-context LLMs can process millions of token contexts end-to-end, but they suffer from high token consumption and attention dilution. In parallel, specialized LTU agents often sacrifice fidelity through task-agnostic abstractions like graph construction or indexing. We identify a key insight for LTU: query-relevant information is typically sparse relative to the full document, so effective reasoning should rely on a query-sufficient subset rather than the entire context. To address this, we propose SCOUT, a new paradigm for LTU that shifts from passive processing to active information foraging. It treats the document as an explorable environment and answers from a compact, provenance-grounded epistemic state. Guided by state-level gap diagnosis, SCOUT adaptively alternates between coarse-to-fine exploration and anchored state updates that progressively contract its epistemic state toward query sufficiency. Experiments show that SCOUT matches state-of-the-art proprietary models while reducing token consumption by up to 8x. Moreover, SCOUT remains stable as context length scales, substantially alleviating the practical cost-performance trade-off. 

---
# RetentiveKV: State-Space Memory for Uncertainty-Aware Multimodal KV Cache Eviction 

**Authors**: Sihao Liu, YuFan Xiong, Zhonghua Jiang, Zhaode Wang, chengfei lv Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.04075)  

**Abstract**: Multimodal Large Language Models face severe challenges in computational efficiency and memory consumption due to the substantial expansion of the visual KV cache when processing long visual contexts. Existing KV cache compression methods typically rely on the "persistence of importance" hypothesis to prune tokens. However, this approach proves fragile in multimodal settings due to two key issues: 1) Visual tokens display "deferred importance," initially exhibiting low salience but becoming pivotal during later decoding, which can lead to premature eviction. 2) Discrete pruning disrupts the inherent spatial continuity of visual cues. To address these challenges, we propose RetentiveKV, an entropy-driven KV cache optimization method that reformulates KV eviction from "discrete context truncation" to "continuous memory evolution" based on State Space Models. Our method leverages information entropy to quantify the information potential of low-attention tokens and integrates tokens scheduled for eviction into a continuous state space through entropy-guided state transitions, enabling their dynamic reactivation when semantic relevance arises during subsequent decoding. Extensive experiments on multimodal benchmarks demonstrate that RetentiveKV achieves 5.0 times KV cache compression and 1.5 times decoding acceleration. 

---
# LongSeeker: Elastic Context Orchestration for Long-Horizon Search Agents 

**Authors**: Yijun Lu, Rui Ye, Yuwen Du, Jiajun Wang, Songhua Liu, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05191)  

**Abstract**: Long-horizon search agents must manage a rapidly growing working context as they reason, call tools, and observe information. Naively accumulating all intermediate content can overwhelm the agent, increasing costs and the risk of errors. We propose that effective context management should be adaptive: parts of the agent's trajectory are maintained at different levels of detail depending on their current relevance to the task. To operationalize this principle, we introduce Context-ReAct, a general agentic paradigm for elastic context orchestration that integrates reasoning, context management, and tool use in a unified loop. Context-ReAct provides five atomic operations: Skip, Compress, Rollback, Snippet and Delete, which allow the agent to dynamically reshape its working context, preserving important evidence, summarizing resolved information, discarding unhelpful branches, and controlling context size. We prove that the Compress operator is expressively complete, while the other specialized operators provide efficiency and fidelity guarantees that reduce generation cost and hallucination risk. Building on this paradigm, we develop LongSeeker, a long-horizon search agent fine-tuned from Qwen3-30B-A3B on 10k synthesized trajectories. Across four representative search benchmarks, LongSeeker achieves 61.5% on BrowseComp and 62.5% on BrowseComp-ZH, substantially outperforming Tongyi DeepResearch (43.2% and 46.7%) and AgentFold (36.2% and 47.3%). These results highlight the potential of adaptive context management, showing that agents can achieve more reliable and efficient long-horizon reasoning by actively shaping their working memory. 

---
# LCM: Lossless Context Management 

**Authors**: Clint Ehrlich, Theodore Blackman  

**Link**: [PDF](https://arxiv.org/pdf/2605.04050)  

**Abstract**: We introduce Lossless Context Management (LCM), a deterministic architecture for LLM memory that outperforms Claude Code on long-context tasks. When benchmarked using Opus 4.6, our LCM-augmented coding agent, Volt, achieves higher scores than Claude Code on the OOLONG long-context eval, including at every context length between 32K and 1M tokens.
LCM may be considered both a vindication and extension of the recursive paradigm pioneered by Recursive Language Models (RLMs). Our results demonstrate that recursive context manipulation can outperform not just conventional LLMs, but frontier coding agents with native file-system access.
LCM departs from RLM by decomposing symbolic recursion into two deterministic, engine-managed mechanisms: recursive context compression, in which a hierarchical summary DAG automatically compacts older messages while retaining lossless pointers to every original; and recursive task partitioning, in which engine-managed parallel primitives like LLM-Map replace model-written loops. This trade-off, analogous to the move from GOTO to structured control flow in program-ming language design, sacrifices maximal flexibility for termination guarantees, zero-cost continuity on short tasks, and lossless retrievability of all prior state. 

---
