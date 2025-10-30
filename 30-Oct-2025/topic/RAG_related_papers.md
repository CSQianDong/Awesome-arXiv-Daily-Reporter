# FARSIQA: Faithful and Advanced RAG System for Islamic Question Answering 

**Authors**: Mohammad Aghajani Asl, Behrooz Minaei Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.25621)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized Natural Language Processing, yet their application in high-stakes, specialized domains like religious question answering is hindered by challenges like hallucination and unfaithfulness to authoritative sources. This issue is particularly critical for the Persian-speaking Muslim community, where accuracy and trustworthiness are paramount. Existing Retrieval-Augmented Generation (RAG) systems, relying on simplistic single-pass pipelines, fall short on complex, multi-hop queries requiring multi-step reasoning and evidence aggregation. To address this gap, we introduce FARSIQA, a novel, end-to-end system for Faithful Advanced Question Answering in the Persian Islamic domain. FARSIQA is built upon our innovative FAIR-RAG architecture: a Faithful, Adaptive, Iterative Refinement framework for RAG. FAIR-RAG employs a dynamic, self-correcting process: it adaptively decomposes complex queries, assesses evidence sufficiency, and enters an iterative loop to generate sub-queries, progressively filling information gaps. Operating on a curated knowledge base of over one million authoritative Islamic documents, FARSIQA demonstrates superior performance. Rigorous evaluation on the challenging IslamicPCQA benchmark shows state-of-the-art performance: the system achieves a remarkable 97.0% in Negative Rejection - a 40-point improvement over baselines - and a high Answer Correctness score of 74.3%. Our work establishes a new standard for Persian Islamic QA and validates that our iterative, adaptive architecture is crucial for building faithful, reliable AI systems in sensitive domains. 

---
# Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation 

**Authors**: Alexander Martin, William Walden, Reno Kriz, Dengjia Zhang, Kate Sanders, Eugene Yang, Chihsheng Jin, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2510.24870)  

**Abstract**: We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a prevalent source of information online, it is essential for RAG systems to integrate information from these sources into generation. However, existing evaluations for RAG are text-centric, limiting their applicability to multimodal, reasoning intensive settings because they don't verify information against sources. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, evaluating factuality and information coverage, and CiteF1, measuring citation support and completeness. We show that MiRAGE, when applied by humans, strongly aligns with extrinsic quality judgments. We additionally introduce automatic variants of MiRAGE and three prominent TextRAG metrics -- ACLE, ARGUE, and RAGAS -- demonstrating the limitations of text-centric work and laying the groundwork for automatic evaluation. We release open-source implementations and outline how to assess multimodal RAG. 

---
# Secure Retrieval-Augmented Generation against Poisoning Attacks 

**Authors**: Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25025)  

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling applications from content generation to decision support. Retrieval-Augmented Generation (RAG) improves LLMs by incorporating external knowledge but also introduces security risks, particularly from data poisoning, where the attacker injects poisoned texts into the knowledge database to manipulate system outputs. While various defenses have been proposed, they often struggle against advanced attacks. To address this, we introduce RAGuard, a detection framework designed to identify poisoned texts. RAGuard first expands the retrieval scope to increase the proportion of clean texts, reducing the likelihood of retrieving poisoned content. It then applies chunk-wise perplexity filtering to detect abnormal variations and text similarity filtering to flag highly similar texts. This non-parametric approach enhances RAG security, and experiments on large-scale datasets demonstrate its effectiveness in detecting and mitigating poisoning attacks, including strong adaptive attacks. 

---
# Retrieval Augmented Generation (RAG) for Fintech: Agentic Design and Evaluation 

**Authors**: Thomas Cook, Richard Osuagwu, Liman Tsatiashvili, Vrynsia Vrynsia, Koustav Ghosal, Maraim Masoud, Riccardo Mattivi  

**Link**: [PDF](https://arxiv.org/pdf/2510.25518)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often face limitations in specialized domains such as fintech, where domain-specific ontologies, dense terminology, and acronyms complicate effective retrieval and synthesis. This paper introduces an agentic RAG architecture designed to address these challenges through a modular pipeline of specialized agents. The proposed system supports intelligent query reformulation, iterative sub-query decomposition guided by keyphrase extraction, contextual acronym resolution, and cross-encoder-based context re-ranking. We evaluate our approach against a standard RAG baseline using a curated dataset of 85 question--answer--reference triples derived from an enterprise fintech knowledge base. Experimental results demonstrate that the agentic RAG system outperforms the baseline in retrieval precision and relevance, albeit with increased latency. These findings suggest that structured, multi-agent methodologies offer a promising direction for enhancing retrieval robustness in complex, domain-specific settings. 

---
