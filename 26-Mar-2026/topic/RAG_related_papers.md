# Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework 

**Authors**: Zeinab Dehghani, Rameez Raja Kureshi, Koorosh Aslansefat, Faezeh Alsadat Abedi, Dhavalkumar Thakker, Lisa Greaves, Bhupesh Kumar Mishra, Baseer Ahmad, Tanaya Maslekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.23625)  

**Abstract**: Artificial intelligence (AI) is increasingly being explored in health and social care to reduce administrative workload and allow staff to spend more time on patient care. This paper evaluates a voice-enabled Care Home Smart Speaker designed to support everyday activities in residential care homes, including spoken access to resident records, reminders, and scheduling tasks. A safety-focused evaluation framework is presented that examines the system end-to-end, combining Whisper-based speech recognition with retrieval-augmented generation (RAG) approaches (hybrid, sparse, and dense). Using supervised care-home trials and controlled testing, we evaluated 330 spoken transcripts across 11 care categories, including 184 reminder-containing interactions. These evaluations focus on (i) correct identification of residents and care categories, (ii) reminder recognition and extraction, and (iii) end-to-end scheduling correctness under uncertainty (including safe deferral/clarification). Given the safety-critical nature of care homes, particular attention is also paid to reliability in noisy environments and across diverse accents, supported by confidence scoring, clarification prompts, and human-in-the-loop oversight. In the best-performing configuration (GPT-5.2), resident ID and care category matching reached 100% (95% CI: 98.86-100), while reminder recognition reached 89.09\% (95% CI: 83.81-92.80) with zero missed reminders (100% recall) but some false positives. End-to-end scheduling via calendar integration achieved 84.65% exact reminder-count agreement (95% CI: 78.00-89.56), indicating remaining edge cases in converting informal spoken instructions into actionable events. The findings suggest that voice-enabled systems, when carefully evaluated and appropriately safeguarded, can support accurate documentation, effective task management, and trustworthy use of AI in care home settings. 

---
# Retrieval Improvements Do Not Guarantee Better Answers: A Study of RAG for AI Policy QA 

**Authors**: Saahil Mathur, Ryan David Rittner, Vedant Ajit Thakur, Daniel Stuart Schiff, Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2603.24580)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly used to analyze complex policy documents, but achieving sufficient reliability for expert usage remains challenging in domains characterized by dense legal language and evolving, overlapping regulatory frameworks. We study the application of RAG to AI governance and policy analysis using the AI Governance and Regulatory Archive (AGORA) corpus, a curated collection of 947 AI policy documents. Our system combines a ColBERT-based retriever fine-tuned with contrastive learning and a generator aligned to human preferences using Direct Preference Optimization (DPO). We construct synthetic queries and collect pairwise preferences to adapt the system to the policy domain. Through experiments evaluating retrieval quality, answer relevance, and faithfulness, we find that domain-specific fine-tuning improves retrieval metrics but does not consistently improve end-to-end question answering performance. In some cases, stronger retrieval counterintuitively leads to more confident hallucinations when relevant documents are absent from the corpus. These results highlight a key concern for those building policy-focused RAG systems: improvements to individual components do not necessarily translate to more reliable answers. Our findings provide practical insights for designing grounded question-answering systems over dynamic regulatory corpora. 

---
# Evaluating Chunking Strategies For Retrieval-Augmented Generation in Oil and Gas Enterprise Documents 

**Authors**: Samuel Taiwo, Mohd Amaluddin Yusoff  

**Link**: [PDF](https://arxiv.org/pdf/2603.24556)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a framework to address the constraints of Large Language Models (LLMs). Yet, its effectiveness fundamentally hinges on document chunking - an often-overlooked determinant of its quality. This paper presents an empirical study quantifying performance differences across four chunking strategies: fixed-size sliding window, recursive, breakpoint-based semantic, and structure-aware. We evaluated these methods using a proprietary corpus of oil and gas enterprise documents, including text-heavy manuals, table-heavy specifications, and piping and instrumentation diagrams (P and IDs). Our findings show that structure-aware chunking yields higher overall retrieval effectiveness, particularly in top-K metrics, and incurs significantly lower computational costs than semantic or baseline strategies. Crucially, all four methods demonstrated limited effectiveness on P and IDs, underscoring a core limitation of purely text-based RAG within visually and spatially encoded documents. We conclude that while explicit structure preservation is essential for specialised domains, future work must integrate multimodal models to overcome current limitations. 

---
# Who Benefits from RAG? The Role of Exposure, Utility and Attribution Bias 

**Authors**: Mahdi Dehghan, Graham McDonald  

**Link**: [PDF](https://arxiv.org/pdf/2603.24218)  

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have achieved substantial improvements in accuracy by grounding their responses in external documents that are relevant to the user's query. However, relatively little work has investigated the impact of RAG in terms of fairness. Particularly, it is not yet known if queries that are associated with certain groups within a fairness category systematically receive higher accuracy, or accuracy improvements in RAG systems compared to LLM-only, a phenomenon we refer to as query group fairness. In this work, we conduct extensive experiments to investigate the impact of three key factors on query group fairness in RAG, namely: Group exposure, i.e., the proportion of documents from each group appearing in the retrieved set, determined by the retriever; Group utility, i.e., the degree to which documents from each group contribute to improving answer accuracy, capturing retriever-generator interactions; and Group attribution, i.e., the extent to which the generator relies on documents from each group when producing responses. We examine group-level average accuracy and accuracy improvements disparities across four fairness categories using three datasets derived from the TREC 2022 Fair Ranking Track for two tasks: article generation and title generation. Our findings show that RAG systems suffer from the query group fairness problem and amplify disparities in terms of average accuracy across queries from different groups, compared to an LLM-only setting. Moreover, group utility, exposure, and attribution can exhibit strong positive or negative correlations with average accuracy or accuracy improvements of queries from that group, highlighting their important role in fair RAG. Our data and code are publicly available from Github. 

---
# Mixture of Demonstrations for Textual Graph Understanding and Question Answering 

**Authors**: Yukun Wu, Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23554)  

**Abstract**: Textual graph-based retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) in domain-specific question answering. While existing approaches primarily focus on zero-shot GraphRAG, selecting high-quality demonstrations is crucial for improving reasoning and answer accuracy. Furthermore, recent studies have shown that retrieved subgraphs often contain irrelevant information, which can degrade reasoning performance. In this paper, we propose MixDemo, a novel GraphRAG framework enhanced with a Mixture-of-Experts (MoE) mechanism for selecting the most informative demonstrations under diverse question contexts. To further reduce noise in the retrieved subgraphs, we introduce a query-specific graph encoder that selectively attends to information most relevant to the query. Extensive experiments across multiple textual graph benchmarks show that MixDemo significantly outperforms existing methods. 

---
# MDKeyChunker: Single-Call LLM Enrichment with Rolling Keys and Key-Based Restructuring for High-Accuracy RAG 

**Authors**: Bhavik Mangla  

**Link**: [PDF](https://arxiv.org/pdf/2603.23533)  

**Abstract**: RAG pipelines typically rely on fixed-size chunking, which ignores document structure, fragments semantic units across boundaries, and requires multiple LLM calls per chunk for metadata extraction. We present MDKeyChunker, a three-stage pipeline for Markdown documents that (1) performs structure-aware chunking treating headers, code blocks, tables, and lists as atomic units; (2) enriches each chunk via a single LLM call extracting title, summary, keywords, typed entities, hypothetical questions, and a semantic key, while propagating a rolling key dictionary to maintain document-level context; and (3) restructures chunks by merging those sharing the same semantic key via bin-packing, co-locating related content for retrieval. The single-call design extracts all seven metadata fields in one LLM invocation, eliminating the need for separate per-field extraction passes. Rolling key propagation replaces hand-tuned scoring with LLM-native semantic matching. An empirical evaluation on 30 queries over an 18-document Markdown corpus shows Config D (BM25 over structural chunks) achieves Recall@5=1.000 and MRR=0.911, while dense retrieval over the full pipeline (Config C) reaches Recall@5=0.867. MDKeyChunker is implemented in Python with four dependencies and supports any OpenAI-compatible endpoint. 

---
# S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering 

**Authors**: Rong Fu, Yemin Wang, Tianxiang Xu, Yongtai Liu, Weizhi Tang, Wangyu Wu, Xiaowen Ma, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2603.23512)  

**Abstract**: We present S-Path-RAG, a semantic-aware shortest-path Retrieval-Augmented Generation framework designed to improve multi-hop question answering over large knowledge graphs. S-Path-RAG departs from one-shot, text-heavy retrieval by enumerating bounded-length, semantically weighted candidate paths using a hybrid weighted $k$-shortest, beam, and constrained random-walk strategy, learning a differentiable path scorer together with a contrastive path encoder and lightweight verifier, and injecting a compact soft mixture of selected path latents into a language model via cross-attention. The system runs inside an iterative Neural-Socratic Graph Dialogue loop in which concise diagnostic messages produced by the language model are mapped to targeted graph edits or seed expansions, enabling adaptive retrieval when the model expresses uncertainty. This combination yields a retrieval mechanism that is both token-efficient and topology-aware while preserving interpretable path-level traces for diagnostics and intervention. We validate S-Path-RAG on standard multi-hop KGQA benchmarks and through ablations and diagnostic analyses. The results demonstrate consistent improvements in answer accuracy, evidence coverage, and end-to-end efficiency compared to strong graph- and LLM-based baselines. We further analyze trade-offs between semantic weighting, verifier filtering, and iterative updates, and report practical recommendations for deployment under constrained compute and token budgets. 

---
# CVPD at QIAS 2026: RAG-Guided LLM Reasoning for Al-Mawarith Share Computation and Heir Allocation 

**Authors**: Wassim Swaileh, Mohammed-En-Nadhir Zighem, Hichem Telli, Salah Eddine Bekhouche, Abdellah Zakaria Sellam, Fadi Dornaika, Dimitrios Kotzinos  

**Link**: [PDF](https://arxiv.org/pdf/2603.24012)  

**Abstract**: Islamic inheritance (Ilm al-Mawarith) is a multi-stage legal reasoning task requiring the identification of eligible heirs, resolution of blocking rules (hajb), assignment of fixed and residual shares, handling of adjustments such as awl and radd, and generation of a consistent final distribution. The task is further complicated by variations across legal schools and civil-law codifications, requiring models to operate under explicit legal configurations.
We present a retrieval-augmented generation (RAG) pipeline for this setting, combining rule-grounded synthetic data generation, hybrid retrieval (dense and BM25) with cross-encoder reranking, and schema-constrained output validation. A symbolic inheritance calculator is used to generate a large high-quality synthetic corpus with full intermediate reasoning traces, ensuring legal and numerical consistency.
The proposed system achieves a MIR-E score of 0.935 and ranks first on the official QIAS 2026 blind-test leaderboard. Results demonstrate that retrieval-grounded, schema-aware generation significantly improves reliability in high-precision Arabic legal reasoning tasks. 

---
# CoCR-RAG: Enhancing Retrieval-Augmented Generation in Web Q&A via Concept-oriented Context Reconstruction 

**Authors**: Kaize Shi, Xueyao Sun, Qika Lin, Firoj Alam, Qing Li, Xiaohui Tao, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23989)  

**Abstract**: Retrieval-augmented generation (RAG) has shown promising results in enhancing Q&A by incorporating information from the web and other external sources. However, the supporting documents retrieved from the heterogeneous web often originate from multiple sources with diverse writing styles, varying formats, and inconsistent granularity. Fusing such multi-source documents into a coherent and knowledge-intensive context remains a significant challenge, as the presence of irrelevant and redundant information can compromise the factual consistency of the inferred answers. This paper proposes the Concept-oriented Context Reconstruction RAG (CoCR-RAG), a framework that addresses the multi-source information fusion problem in RAG through linguistically grounded concept-level integration. Specifically, we introduce a concept distillation algorithm that extracts essential concepts from Abstract Meaning Representation (AMR), a stable semantic representation that structures the meaning of texts as logical graphs. The distilled concepts from multiple retrieved documents are then fused and reconstructed into a unified, information-intensive context by Large Language Models, which supplement only the necessary sentence elements to highlight the core knowledge. Experiments on the PopQA and EntityQuestions datasets demonstrate that CoCR-RAG significantly outperforms existing context-reconstruction methods across these Web Q&A benchmarks. Furthermore, CoCR-RAG shows robustness across various backbone LLMs, establishing itself as a flexible, plug-and-play component adaptable to different RAG frameworks. 

---
# Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith 

**Authors**: Somaya Eltanbouly, Samer Rashwani  

**Link**: [PDF](https://arxiv.org/pdf/2603.23972)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in many language tasks, yet they continue to struggle with complex historical and religious Arabic texts such as the Quran and Hadith. To address this limitation, we develop a retrieval-augmented generation (RAG) framework grounded in diachronic lexicographic knowledge. Unlike prior RAG systems that rely on general-purpose corpora, our approach retrieves evidence from the Doha Historical Dictionary of Arabic (DHDA), a large-scale resource documenting the historical development of Arabic vocabulary. The proposed pipeline combines hybrid retrieval with an intent-based routing mechanism to provide LLMs with precise, contextually relevant historical information. Our experiments show that this approach improves the accuracy of Arabic-native LLMs, including Fanar and ALLaM, to over 85\%, substantially reducing the performance gap with Gemini, a proprietary large-scale model. Gemini also serves as an LLM-as-a-judge system for automatic evaluation in our experiments. The automated judgments were verified through human evaluation, demonstrating high agreement (kappa = 0.87). An error analysis further highlights key linguistic challenges, including diacritics and compound expressions. These findings demonstrate the value of integrating diachronic lexicographic resources into retrieval-augmented generation frameworks to enhance Arabic language understanding, particularly for historical and religious texts. The code and resources are publicly available at: this https URL. 

---
# Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems 

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Haichen Zhang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23508)  

**Abstract**: Retrieval-augmented generation (RAG) is increasingly deployed in enterprise search and document-centric assistants, where responses must be grounded in long and complex source materials. In practice, verifying that generated answers faithfully reflect retrieved documents is difficult: large language models can check long contexts but are too slow and costly for interactive services, while lightweight classifiers operate within strict context limits and frequently miss evidence outside truncated passages. We present the design of a real-time verification component integrated into a production RAG pipeline that enables full-document grounding under latency constraints. The system processes documents up to 32K tokens and employs adaptive inference strategies to balance response time and verification coverage across workloads. We describe the architectural decisions, operational trade-offs, and evaluation methodology used to deploy the verifier, and show that full-context verification substantially improves detection of unsupported responses compared with truncated validation. Our experience highlights when long-context verification is necessary, why chunk-based checking often fails in real documents, and how latency budgets shape model design. These findings provide practical guidance for practitioners building reliable large-scale retrieval-augmented applications. (Model, benchmark, and code: this https URL) 

---
# VILLA: Versatile Information Retrieval From Scientific Literature Using Large LAnguage Models 

**Authors**: Blessy Antony, Amartya Dutta, Sneha Aggarwal, Vasu Gatne, Ozan Gökdemir, Samantha Grimes, Adam Lauring, Brian R. Wasik, Anuj Karpatne, T. M. Murali  

**Link**: [PDF](https://arxiv.org/pdf/2603.23849)  

**Abstract**: The lack of high-quality ground truth datasets to train machine learning (ML) models impedes the potential of artificial intelligence (AI) for science research. Scientific information extraction (SIE) from the literature using LLMs is emerging as a powerful approach to automate the creation of these datasets. However, existing LLM-based approaches and benchmarking studies for SIE focus on broad topics such as biomedicine and chemistry, are limited to choice-based tasks, and focus on extracting information from short and well-formatted text. The potential of SIE methods in complex, open-ended tasks is considerably under-explored. In this study, we used a domain that has been virtually ignored in SIE, namely virology, to address these research gaps. We design a unique, open-ended SIE task of extracting mutations in a given virus that modify its interaction with the host. We develop a new, multi-step retrieval augmented generation (RAG) framework called VILLA for SIE. In parallel, we curate a novel dataset of 629 mutations in ten influenza A virus proteins obtained from 239 scientific publications to serve as ground truth for the mutation extraction task. Finally, we demonstrate VILLA's superior performance using a novel and comprehensive evaluation and comparison with vanilla RAG and other state-of-the art RAG- and agent-based tools for SIE. 

---
