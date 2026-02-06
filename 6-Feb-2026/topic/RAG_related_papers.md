# CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering 

**Authors**: Hao Yang, Zhiyu Yang, Xupeng Zhang, Wei Wei, Yunjie Zhang, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05728)  

**Abstract**: Retrieval-augmented generation (RAG) has become a key paradigm for knowledge-intensive question answering. However, existing multi-hop RAG systems remain inefficient, as they alternate between retrieval and reasoning at each step, resulting in repeated LLM calls, high token consumption, and unstable entity grounding across hops. We propose CompactRAG, a simple yet effective framework that decouples offline corpus restructuring from online reasoning.
In the offline stage, an LLM reads the corpus once and converts it into an atomic QA knowledge base, which represents knowledge as minimal, fine-grained question-answer pairs. In the online stage, complex queries are decomposed and carefully rewritten to preserve entity consistency, and are resolved through dense retrieval followed by RoBERTa-based answer extraction. Notably, during inference, the LLM is invoked only twice in total - once for sub-question decomposition and once for final answer synthesis - regardless of the number of reasoning hops.
Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue demonstrate that CompactRAG achieves competitive accuracy while substantially reducing token consumption compared to iterative RAG baselines, highlighting a cost-efficient and practical approach to multi-hop reasoning over large knowledge corpora. The implementation is available at GitHub. 

---
# FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters 

**Authors**: Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2602.05235)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge to improve factuality and reduce hallucinations. Yet most deployments assume a centralized corpus, which is infeasible in privacy aware domains where knowledge remains siloed. This motivates federated RAG (FedRAG), where a central LLM server collaborates with distributed silos without sharing raw documents. In context RAG violates this requirement by transmitting verbatim documents, whereas parametric RAG encodes documents into lightweight adapters that merge with a frozen LLM at inference, avoiding raw-text exchange. We adopt the parametric approach but face two unique challenges induced by FedRAG: high storage and communication from per-document adapters, and destructive aggregation caused by indiscriminately merging multiple adapters. We present FedMosaic, the first federated RAG framework built on parametric adapters. FedMosaic clusters semantically related documents into multi-document adapters with document-specific masks to reduce overhead while preserving specificity, and performs selective adapter aggregation to combine only relevance-aligned, nonconflicting adapters. Experiments show that FedMosaic achieves an average 10.9% higher accuracy than state-of-the-art methods in four categories, while lowering storage costs by 78.8% to 86.3% and communication costs by 91.4%, and never sharing raw documents. 

---
# Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Exploration 

**Authors**: Chuangtao Ma, Zeyu Zhang, Arijit Khan, Sebastian Schelter, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2602.05708)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLM reasoning in knowledge-intensive tasks, but existing RAG pipelines incur substantial retrieval and generation overhead when applied to large-scale entity matching. To address this limitation, we introduce CE-RAG4EM, a cost-efficient RAG architecture that reduces computation through blocking-based batch retrieval and generation. We also present a unified framework for analyzing and evaluating RAG systems for entity matching, focusing on blocking-aware optimizations and retrieval granularity. Extensive experiments suggest that CE-RAG4EM can achieve comparable or improved matching quality while substantially reducing end-to-end runtime relative to strong baselines. Our analysis further reveals that key configuration parameters introduce an inherent trade-off between performance and overhead, offering practical guidance for designing efficient and scalable RAG systems for entity matching and data integration. 

---
# Mitigating Hallucination in Financial Retrieval-Augmented Generation via Fine-Grained Knowledge Verification 

**Authors**: Taoye Yin, Haoyuan Hu, Yaxin Fan, Xinhao Chen, Xinya Wu, Kai Deng, Kezun Zhang, Feng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05723)  

**Abstract**: In financial Retrieval-Augmented Generation (RAG) systems, models frequently rely on retrieved documents to generate accurate responses due to the time-sensitive nature of the financial domain. While retrieved documents help address knowledge gaps, model-generated responses still suffer from hallucinations that contradict the retrieved information. To mitigate this inconsistency, we propose a Reinforcement Learning framework enhanced with Fine-grained Knowledge Verification (RLFKV). Our method decomposes financial responses into atomic knowledge units and assesses the correctness of each unit to compute the fine-grained faithful reward. This reward offers more precise optimization signals, thereby improving alignment with the retrieved documents. Additionally, to prevent reward hacking (e.g., overly concise replies), we incorporate an informativeness reward that encourages the policy model to retain at least as many knowledge units as the base model. Experiments conducted on the public Financial Data Description (FDD) task and our newly proposed FDD-ANT dataset demonstrate consistent improvements, confirming the effectiveness of our approach. 

---
# Traceable Cross-Source RAG for Chinese Tibetan Medicine Question Answering 

**Authors**: Fengxian Chen, Zhilong Tao, Jiaxuan Li, Yunlong Li, Qingguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.05195)  

**Abstract**: Retrieval-augmented generation (RAG) promises grounded question answering, yet domain settings with multiple heterogeneous knowledge bases (KBs) remain challenging. In Chinese Tibetan medicine, encyclopedia entries are often dense and easy to match, which can dominate retrieval even when classics or clinical papers provide more authoritative evidence. We study a practical setting with three KBs (encyclopedia, classics, and clinical papers) and a 500-query benchmark (cutoff $K{=}5$) covering both single-KB and cross-KB questions. We propose two complementary methods to improve traceability, reduce hallucinations, and enable cross-KB verification. First, DAKS performs KB routing and budgeted retrieval to mitigate density-driven bias and to prioritize authoritative sources when appropriate. Second, we use an alignment graph to guide evidence fusion and coverage-aware packing, improving cross-KB evidence coverage without relying on naive concatenation. All answers are generated by a lightweight generator, \textsc{openPangu-Embedded-7B}. Experiments show consistent gains in routing quality and cross-KB evidence coverage, with the full system achieving the best CrossEv@5 while maintaining strong faithfulness and citation correctness. 

---
# HugRAG: Hierarchical Causal Knowledge Graph Design for RAG 

**Authors**: Nengbo Wang, Tuo Liang, Vikash Singh, Chaoda Song, Van Yang, Yu Yin, Jing Ma, Jagdip Singh, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2602.05143)  

**Abstract**: Retrieval augmented generation (RAG) has enhanced large language models by enabling access to external knowledge, with graph-based RAG emerging as a powerful paradigm for structured retrieval and reasoning. However, existing graph-based methods often over-rely on surface-level node matching and lack explicit causal modeling, leading to unfaithful or spurious answers. Prior attempts to incorporate causality are typically limited to local or single-document contexts and also suffer from information isolation that arises from modular graph structures, which hinders scalability and cross-module causal reasoning. To address these challenges, we propose HugRAG, a framework that rethinks knowledge organization for graph-based RAG through causal gating across hierarchical modules. HugRAG explicitly models causal relationships to suppress spurious correlations while enabling scalable reasoning over large-scale knowledge graphs. Extensive experiments demonstrate that HugRAG consistently outperforms competitive graph-based RAG baselines across multiple datasets and evaluation metrics. Our work establishes a principled foundation for structured, scalable, and causally grounded RAG systems. 

---
# Automated Customization of LLMs for Enterprise Code Repositories Using Semantic Scopes 

**Authors**: Ulrich Finkler, Irene Manotas, Wei Zhang, Geert Janssen, Octavian Popescu, Shyam Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2602.05780)  

**Abstract**: Code completion (CC) is a task frequently used by developers when working in collaboration with LLM-based programming assistants. Despite the increased performance of LLMs on public benchmarks, out of the box LLMs still have a hard time generating code that aligns with a private code repository not previously seen by the model's training data. Customizing code LLMs to a private repository provides a way to improve the model performance. In this paper we present our approach for automated LLM customization based on semantic scopes in the code. We evaluate LLMs on real industry cases with two private enterprise code repositories with two customization strategies: Retrieval-Augmented Generation (RAG) and supervised Fine-Tuning (FT). Our mechanism for ingesting the repository's data and formulating the training data pairs with semantic scopes helps models to learn the underlying patterns specific to the repository, providing more precise code to developers and helping to boost their productivity. The code completions of moderately sized customized models can be significantly better than those of uncustomized models of much larger capacity. We also include an analysis of customization on two public benchmarks and present opportunities for future work. 

---
# RAG without Forgetting: Continual Query-Infused Key Memory 

**Authors**: Yuntong Hu, Sha Li, Naren Ramakrishnan, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05152)  

**Abstract**: Retrieval-augmented generation (RAG) systems commonly improve robustness via query-time adaptations such as query expansion and iterative retrieval. While effective, these approaches are inherently stateless: adaptations are recomputed for each query and discarded thereafter, precluding cumulative learning and repeatedly incurring inference-time cost. Index-side approaches like key expansion introduce persistence but rely on offline preprocessing or heuristic updates that are weakly aligned with downstream task utility, leading to semantic drift and noise accumulation. We propose Evolving Retrieval Memory (ERM), a training-free framework that transforms transient query-time gains into persistent retrieval improvements. ERM updates the retrieval index through correctness-gated feedback, selectively attributes atomic expansion signals to the document keys they benefit, and progressively evolves keys via stable, norm-bounded updates. We show that query and key expansion are theoretically equivalent under standard similarity functions and prove convergence of ERM's selective updates, amortizing optimal query expansion into a stable index with zero inference-time overhead. Experiments on BEIR and BRIGHT across 13 domains demonstrate consistent gains in retrieval and generation, particularly on reasoning-intensive tasks, at native retrieval speed. 

---
