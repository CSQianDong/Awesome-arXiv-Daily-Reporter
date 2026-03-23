# URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models 

**Authors**: Vinh Nguyen, Cuong Dang, Jiahao Zhang, Hoa Tran, Minh Tran, Trinh Chau, Thai Le, Lu Cheng, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19281)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a widely adopted approach for enhancing LLMs in scenarios that demand extensive factual knowledge. However, current RAG evaluations concentrate primarily on correctness, which may not fully capture the impact of retrieval on LLM uncertainty and reliability. To bridge this gap, we introduce URAG, a comprehensive benchmark designed to assess the uncertainty of RAG systems across various fields like healthcare, programming, science, math, and general text. By reformulating open-ended generation tasks into multiple-choice question answering, URAG allows for principled uncertainty quantification via conformal prediction. We apply the evaluation pipeline to 8 standard RAG methods, measuring their performance through both accuracy and prediction-set sizes based on LAC and APS metrics. Our analysis shows that (1) accuracy gains often coincide with reduced uncertainty, but this relationship breaks under retrieval noise; (2) simple modular RAG methods tend to offer better accuracy-uncertainty trade-offs than more complex reasoning pipelines; and (3) no single RAG approach is universally reliable across domains. We further show that (4) retrieval depth, parametric knowledge dependence, and exposure to confidence cues can amplify confident errors and hallucinations. Ultimately, URAG establishes a systematic benchmark for analyzing and enhancing the trustworthiness of retrieval-augmented systems. Our code is available on GitHub. 

---
# From Flat to Structural: Enhancing Automated Short Answer Grading with GraphRAG 

**Authors**: Yucheng Chu, Haoyu Han, Shen Dong, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Joseph Krajcik, Namsoo Shin, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19276)  

**Abstract**: Automated short answer grading (ASAG) is critical for scaling educational assessment, yet large language models (LLMs) often struggle with hallucinations and strict rubric adherence due to their reliance on generalized pre-training. While Rretrieval-Augmented Generation (RAG) mitigates these issues, standard "flat" vector retrieval mechanisms treat knowledge as isolated fragments, failing to capture the structural relationships and multi-hop reasoning essential for complex educational content. To address this limitation, we introduce a Graph Retrieval-Augmented Generation (GraphRAG) framework that organizes reference materials into a structured knowledge graph to explicitly model dependencies between concepts. Our methodology employs a dual-phase pipeline: utilizing Microsoft GraphRAG for high-fidelity graph construction and the HippoRAG neurosymbolic algorithm to execute associative graph traversals, thereby retrieving comprehensive, connected subgraphs of evidence. Experimental evaluations on a Next Generation Science Standards (NGSS) dataset demonstrate that this structural approach significantly outperforms standard RAG baselines across all metrics. Notably, the HippoRAG implementation achieved substantial improvements in evaluating Science and Engineering Practices (SEP), confirming the superiority of structural retrieval in verifying the logical reasoning chains required for higher-order academic assessment. 

---
# Grounded Multimodal Retrieval-Augmented Drafting of Radiology Impressions Using Case-Based Similarity Search 

**Authors**: Himadri Samanta  

**Link**: [PDF](https://arxiv.org/pdf/2603.17765)  

**Abstract**: Automated radiology report generation has gained increasing attention with the rise of deep learning and large language models. However, fully generative approaches often suffer from hallucinations and lack clinical grounding, limiting their reliability in real-world workflows. In this study, we propose a multimodal retrieval-augmented generation (RAG) system for grounded drafting of chest radiograph impressions. The system combines contrastive image-text embeddings, case-based similarity retrieval, and citation-constrained draft generation to ensure factual alignment with historical radiology reports. A curated subset of the MIMIC-CXR dataset was used to construct a multimodal retrieval database. Image embeddings were generated using CLIP encoders, while textual embeddings were derived from structured impression sections. A fusion similarity framework was implemented using FAISS indexing for scalable nearest-neighbor retrieval. Retrieved cases were used to construct grounded prompts for draft impression generation, with safety mechanisms enforcing citation coverage and confidence-based refusal. Experimental results demonstrate that multimodal fusion significantly improves retrieval performance compared to image-only retrieval, achieving Recall@5 above 0.95 on clinically relevant findings. The grounded drafting pipeline produces interpretable outputs with explicit citation traceability, enabling improved trustworthiness compared to conventional generative approaches. This work highlights the potential of retrieval-augmented multimodal systems for reliable clinical decision support and radiology workflow augmentation 

---
# Enhancing Legal LLMs through Metadata-Enriched RAG Pipelines and Direct Preference Optimization 

**Authors**: Suyash Maniyar, Deepali Singh, Rohith Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2603.19251)  

**Abstract**: Large Language Models (LLMs) perform well in short contexts but degrade on long legal documents, often producing hallucinations such as incorrect clauses or precedents. In the legal domain, where precision is critical, such errors undermine reliability and trust.
Retrieval Augmented Generation (RAG) helps ground outputs but remains limited in legal settings, especially with small, locally deployed models required for data privacy. We identify two failure modes: retrieval errors due to lexical redundancy in legal corpora, and decoding errors where models generate answers despite insufficient context.
To address this, we propose Metadata Enriched Hybrid RAG to improve document level retrieval, and apply Direct Preference Optimization (DPO) to enforce safe refusal when context is inadequate. Together, these methods improve grounding, reliability, and safety in legal language models. 

---
