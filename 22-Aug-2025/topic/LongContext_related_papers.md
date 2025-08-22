# LongRetriever: Towards Ultra-Long Sequence based Candidate Retrieval for Recommendation 

**Authors**: Ren Qin, Chai Zheng, Xiao Xijun, Zheng Yuchao, Wu Di  

**Link**: [PDF](https://arxiv.org/pdf/2508.15486)  

**Abstract**: Precisely modeling user ultra-long sequences is critical for industrial recommender systems. Current approaches predominantly focus on leveraging ultra-long sequences in the ranking stage, whereas research for the candidate retrieval stage remains under-explored. This paper presents LongRetriever, a practical framework for incorporating ultra-long sequences into the retrieval stage of recommenders. Specifically, we propose in-context training and multi-context retrieval, which enable candidate-specific interaction between user sequence and candidate item, and ensure training-serving consistency under the search-based paradigm. Extensive online A/B testing conducted on a large-scale e-commerce platform demonstrates statistically significant improvements, confirming the framework's effectiveness. Currently, LongRetriever has been fully deployed in the platform, impacting billions of users. 

---
# LongRecall: A Structured Approach for Robust Recall Evaluation in Long-Form Text 

**Authors**: MohamamdJavad Ardestani, Ehsan Kamalloo, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15085)  

**Abstract**: LongRecall. The completeness of machine-generated text, ensuring that it captures all relevant information, is crucial in domains such as medicine and law and in tasks like list-based question answering (QA), where omissions can have serious consequences. However, existing recall metrics often depend on lexical overlap, leading to errors with unsubstantiated entities and paraphrased answers, while LLM-as-a-Judge methods with long holistic prompts capture broader semantics but remain prone to misalignment and hallucinations without structured verification. We introduce LongRecall, a general three-stage recall evaluation framework that decomposes answers into self-contained facts, successively narrows plausible candidate matches through lexical and semantic filtering, and verifies their alignment through structured entailment checks. This design reduces false positives and false negatives while accommodating diverse phrasings and contextual variations, serving as a foundational building block for systematic recall assessment. We evaluate LongRecall on three challenging long-form QA benchmarks using both human annotations and LLM-based judges, demonstrating substantial improvements in recall accuracy over strong lexical and LLM-as-a-Judge baselines. 

---
# SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15190)  

**Abstract**: Tokenization plays a critical role in language modeling, yet existing approaches such as Byte-Pair Encoding (BPE) or WordPiece operate purely on frequency statistics, ignoring the underlying semantic structure of text. This leads to over-tokenization of semantically redundant spans and underutilization of contextual coherence, particularly in long-context scenarios. In this work, we propose \textbf{SemToken}, a semantic-aware tokenization framework that jointly reduces token redundancy and improves computation efficiency. SemToken first extracts contextual semantic embeddings via lightweight encoders and performs local semantic clustering to merge semantically equivalent tokens. Then, it allocates heterogeneous token granularity based on semantic density, allowing finer-grained tokenization in content-rich regions and coarser compression in repetitive or low-entropy spans. SemToken can be seamlessly integrated with modern language models and attention acceleration methods. Experiments on long-context language modeling benchmarks such as WikiText-103 and LongBench show that SemToken achieves up to $2.4\times$ reduction in token count and $1.9\times$ speedup, with negligible or no degradation in perplexity and downstream accuracy. Our findings suggest that semantic structure offers a promising new axis for optimizing tokenization and computation in large language models. 

---
# Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory 

**Authors**: Siddharth Chaudhary, Bennett Browning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15099)  

**Abstract**: We present Hydra as an architectural proposal for hybrid long-context language models that combine conditional computation, long-context memory mechanisms, and sparse mixture-of-experts within an approximately 1.6B parameter design envelope. Hydra integrates a Mamba-style Structured State Space Model (SSM) backbone with intermittent sparse global attention, chunk-level MoE feed-forward routing, and dual (workspace plus factual PKM) memories. We formalize the component interfaces, give transparent parameter and complexity accounting, and outline a staged curriculum intended to stably activate the parts. We accompany the specification with illustrative toy-scale prototype measurements (tens of millions of parameters on synthetic data) whose sole purpose is to demonstrate implementation feasibility and qualitative scaling behaviors (for example, long-context throughput crossover and controllable expert routing), not to claim competitive full-scale performance. We explicitly delineate assumptions and open risks (training complexity, memory utilization, specialization dynamics) and position Hydra as a blueprint to stimulate empirical follow-up rather than a finished system. By combining SSM efficiency, selective sparse attention, MoE capacity, and learnable memory, Hydra sketches a path toward modular, input-adaptive long-context language models; validating end-task gains at target scale remains future work. 

---
# Position Bias Mitigates Position Bias:Mitigate Position Bias Through Inter-Position Knowledge Distillation 

**Authors**: Yifei Wang, Feng Xiong, Yong Wang, Linjing Li, Xiangxiang Chu, Daniel Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.15709)  

**Abstract**: Positional bias (PB), manifesting as non-uniform sensitivity across different contextual locations, significantly impairs long-context comprehension and processing capabilities. While prior work seeks to mitigate PB through modifying the architectures causing its emergence, significant PB still persists. To address PB effectively, we introduce \textbf{Pos2Distill}, a position to position knowledge distillation framework. Pos2Distill transfers the superior capabilities from advantageous positions to less favorable ones, thereby reducing the huge performance gaps. The conceptual principle is to leverage the inherent, position-induced disparity to counteract the PB itself. We identify distinct manifestations of PB under \textbf{\textsc{r}}etrieval and \textbf{\textsc{r}}easoning paradigms, thereby designing two specialized instantiations: \emph{Pos2Distill-R\textsuperscript{1}} and \emph{Pos2Distill-R\textsuperscript{2}} respectively, both grounded in this core principle. By employing the Pos2Distill approach, we achieve enhanced uniformity and significant performance gains across all contextual positions in long-context retrieval and reasoning tasks. Crucially, both specialized systems exhibit strong cross-task generalization mutually, while achieving superior performance on their respective tasks. 

---
# Multiple Memory Systems for Enhancing the Long-term Memory of Agent 

**Authors**: Gaoke Zhang, Bo Wang, Yunlong Ma, Dongming Zhao, Zifei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15294)  

**Abstract**: An agent powered by large language models have achieved impressive results, but effectively handling the vast amounts of historical data generated during interactions remains a challenge. The current approach is to design a memory module for the agent to process these data. However, existing methods, such as MemoryBank and A-MEM, have poor quality of stored memory content, which affects recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multiple memory system (MMS) inspired by cognitive psychology theory. The system processes short-term memory to multiple long-term memory fragments, and constructs retrieval memory units and contextual memory units based on these fragments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. Experiments on LoCoMo dataset compared our method with three others, proving its effectiveness. Ablation studies confirmed the rationality of our memory units. We also analyzed the robustness regarding the number of selected memory segments and the storage overhead, demonstrating its practical value. 

---
