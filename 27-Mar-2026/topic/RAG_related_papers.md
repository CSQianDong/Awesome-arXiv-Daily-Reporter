# Adaptive Chunking: Optimizing Chunking-Method Selection for RAG 

**Authors**: Paulo Roberto de Moura Júnior, Jean Lelong, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2603.25333)  

**Abstract**: The effectiveness of Retrieval-Augmented Generation (RAG) is highly dependent on how documents are chunked, that is, segmented into smaller units for indexing and retrieval. Yet, commonly used "one-size-fits-all" approaches often fail to capture the nuanced structure and semantics of diverse texts. Despite its central role, chunking lacks a dedicated evaluation framework, making it difficult to assess and compare strategies independently of downstream performance. We challenge this paradigm by introducing Adaptive Chunking, a framework that selects the most suitable chunking strategy for each document based on a set of five novel intrinsic, document-based metrics: References Completeness (RC), Intrachunk Cohesion (ICC), Document Contextual Coherence (DCC), Block Integrity (BI), and Size Compliance (SC), which directly assess chunking quality across key dimensions. To support this framework, we also introduce two new chunkers, an LLM-regex splitter and a split-then-merge recursive splitter, alongside targeted post-processing techniques. On a diverse corpus spanning legal, technical, and social science domains, our metric-guided adaptive method significantly improves downstream RAG performance. Without changing models or prompts, our framework increases RAG outcomes, raising answers correctness to 72% (from 62-64%) and increasing the number of successfully answered questions by over 30% (65 vs. 49). These results demonstrate that adaptive, document-aware chunking, guided by a complementary suite of intrinsic metrics, offers a practical and effective path to more robust RAG systems. Code available at this https URL. 

---
# Training the Knowledge Base through Evidence Distillation and Write-Back Enrichment 

**Authors**: Yuxing Lu, Xukai Zhao, Wei Wu, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25737)  

**Abstract**: The knowledge base in a retrieval-augmented generation (RAG) system is typically assembled once and never revised, even though the facts a query requires are often fragmented across documents and buried in irrelevant content. We argue that the knowledge base should be treated as a trainable component and propose WriteBack-RAG, a framework that uses labeled examples to identify where retrieval succeeds, isolate the relevant documents, and distill them into compact knowledge units that are indexed alongside the original corpus. Because the method modifies only the corpus, it can be applied once as an offline preprocessing step and combined with any RAG pipeline. Across four RAG methods, six benchmarks, and two LLM backbones, WriteBack-RAG improves every evaluated setting, with gains averaging +2.14%. Cross-method transfer experiments further show that the distilled knowledge benefits RAG pipelines other than the one used to produce it, confirming that the improvement resides in the corpus itself. 

---
# Supercharging Federated Intelligence Retrieval 

**Authors**: Dimitris Stripelis, Patrick Foley, Mohammad Naseri, William Lindskog-Münzing, Chong Shen Ng, Daniel Janes Beutel, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2603.25374)  

**Abstract**: RAG typically assumes centralized access to documents, which breaks down when knowledge is distributed across private data silos. We propose a secure Federated RAG system built using Flower that performs local silo retrieval, while server-side aggregation and text generation run inside an attested, confidential compute environment, enabling confidential remote LLM inference even in the presence of honest-but-curious or compromised servers. We also propose a cascading inference approach that incorporates a non-confidential third-party model (e.g., Amazon Nova) as auxiliary context without weakening confidentiality. 

---
# GraphER: An Efficient Graph-Based Enrichment and Reranking Method for Retrieval-Augmented Generation 

**Authors**: Ruizhong Miao, Yuying Wang, Rongguang Wang, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.24925)  

**Abstract**: Semantic search in retrieval-augmented generation (RAG) systems is often insufficient for complex information needs, particularly when relevant evidence is scattered across multiple sources. Prior approaches to this problem include agentic retrieval strategies, which expand the semantic search space by generating additional queries. However, these methods do not fully leverage the organizational structure of the data and instead rely on iterative exploration, which can lead to inefficient retrieval. Another class of approaches employs knowledge graphs to model non-semantic relationships through graph edges. Although effective in capturing richer proximities, such methods incur significant maintenance costs and are often incompatible with the vector stores used in most production systems. To address these limitations, we propose GraphER, a graph-based enrichment and reranking method that captures multiple forms of proximity beyond semantic similarity. GraphER independently enriches data objects during offline indexing and performs graph-based reranking over candidate objects at query time. This design does not require a knowledge graph, allowing GraphER to integrate seamlessly with standard vector stores. In addition, GraphER is retriever-agnostic and introduces negligible latency overhead. Experiments on multiple retrieval benchmarks demonstrate the effectiveness of the proposed approach. 

---
# AutoSAM: an Agentic Framework for Automating Input File Generation for the SAM Code with Multi-Modal Retrieval-Augmented Generation 

**Authors**: Zaid Abulawi, Zavier Ndum Ndum, Eric Cervi, Rui Hu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.24736)  

**Abstract**: In the design and safety analysis of advanced reactor systems, constructing input files for system-level thermal-hydraulics codes such as the System Analysis Module (SAM) remains a labor-intensive task. Analysts must extract and reconcile design data from heterogeneous engineering documents and manually translate it into solver-specific syntax. In this paper, we present AutoSAM, an agentic framework that automates SAM input file generation. The framework combines a large language model agent with retrieval-augmented generation over the solver's user guide and theory manual, together with specialized tools for analyzing PDFs, images, spreadsheets, and text files. AutoSAM ingests unstructured engineering documents, including system diagrams, design reports, and data tables, extracts simulation-relevant parameters into a human-auditable intermediate representation, and synthesizes validated, solver-compatible input decks. Its multimodal retrieval pipeline integrates scientific text extraction, vision-based figure interpretation, semantic embedding, and query answering. We evaluate AutoSAM on four case studies of increasing complexity: a single-pipe steady-state model, a solid-fuel channel with temperature reactivity feedback, the Advanced Burner Test Reactor core, and the Molten Salt Reactor Experiment primary loop. Across all cases, the agent produces runnable SAM models consistent with expected thermal-hydraulic behavior while explicitly identifying missing data and labeling assumed values. The framework achieves 100% utilization of structured inputs, about 88% extraction from PDF text, and 100% completeness in vision-based geometric extraction. These results demonstrate a practical path toward prompt-driven reactor modeling, in which analysts provide system descriptions and supporting documentation while the agent translates them into transparent, and executable, SAM simulations. 

---
# PIDP-Attack: Combining Prompt Injection with Database Poisoning Attacks on Retrieval-Augmented Generation Systems 

**Authors**: Haozhen Wang, Haoyue Liu, Jionghao Zhu, Zhichao Wang, Yongxin Guo, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25164)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications. However, their practical deployment is often hindered by issues such as outdated knowledge and the tendency to generate hallucinations. To address these limitations, Retrieval-Augmented Generation (RAG) systems have been introduced, enhancing LLMs with external, up-to-date knowledge sources. Despite their advantages, RAG systems remain vulnerable to adversarial attacks, with data poisoning emerging as a prominent threat. Existing poisoning-based attacks typically require prior knowledge of the user's specific queries, limiting their flexibility and real-world applicability. In this work, we propose PIDP-Attack, a novel compound attack that integrates prompt injection with database poisoning in RAG. By appending malicious characters to queries at inference time and injecting a limited number of poisoned passages into the retrieval database, our method can effectively manipulate LLM response to arbitrary query without prior knowledge of the user's actual query. Experimental evaluations across three benchmark datasets (Natural Questions, HotpotQA, MS-MARCO) and eight LLMs demonstrate that PIDP-Attack consistently outperforms the original PoisonedRAG. Specifically, our method improves attack success rates by 4% to 16% on open-domain QA tasks while maintaining high retrieval precision, proving that the compound attack strategy is both necessary and highly effective. 

---
# AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation 

**Authors**: Zhihui Yao, Hengran Zhang, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25092)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) with external knowledge but remains vulnerable to low-authority sources that can propagate misinformation. We investigate whether LLMs can perceive information authority - a capability extending beyond semantic understanding. To address this, we introduce AuthorityBench, a comprehensive benchmark for evaluating LLM authority perception comprising three datasets: DomainAuth (10K web domains with PageRank-based authority), EntityAuth (22K entities with popularity-based authority), and RAGAuth (120 queries with documents of varying authority for downstream evaluation). We evaluate five LLMs using three judging methods (PointJudge, PairJudge, ListJudge) across multiple output formats. Results show that ListJudge and PairJudge with PointScore output achieve the strongest correlation with ground-truth authority, while ListJudge offers optimal cost-effectiveness. Notably, incorporating webpage text consistently degrades judgment performance, suggesting authority is distinct from textual style. Downstream experiments on RAG demonstrate that authority-guided filtering largely improves answer accuracy, validating the practical importance of authority perception for reliable knowledge retrieval. Code and benchmark are available at: this https URL. 

---
# UniAI-GraphRAG: Synergizing Ontology-Guided Extraction, Multi-Dimensional Clustering, and Dual-Channel Fusion for Robust Multi-Hop Reasoning 

**Authors**: Jie Wang, Honghua Huang, Xi Ge, Jianhui Su, Wen Liu, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2603.25152)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems face significant challenges in complex reasoning, multi-hop queries, and domain-specific QA. While existing GraphRAG frameworks have made progress in structural knowledge organization, they still have limitations in cross-industry adaptability, community report integrity, and retrieval performance. This paper proposes UniAI-GraphRAG, an enhanced framework built upon open-source GraphRAG. The framework introduces three core innovations: (1) Ontology-Guided Knowledge Extraction that uses predefined Schema to guide LLMs in accurately identifying domain-specific entities and relations; (2) Multi-Dimensional Community Clustering Strategy that improves community completeness through alignment completion, attribute-based clustering, and multi-hop relationship clustering; (3) Dual-Channel Graph Retrieval Fusion that balances QA accuracy and performance through hybrid graph and community retrieval. Evaluation results on MultiHopRAG benchmark show that UniAI-GraphRAG outperforms mainstream open source solutions (this http URL) in comprehensive F1 scores, particularly in inference and temporal queries. The code is available at this https URL. 

---
