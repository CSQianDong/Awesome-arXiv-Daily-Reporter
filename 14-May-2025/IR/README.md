# Interest Changes: Considering User Interest Life Cycle in Recommendation System 

**Authors**: Yinjiang Cai, Jiangpan Hou, Yangping Zhu, Yuan Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.08471)  

**Abstract**: In recommendation systems, user interests are always in a state of constant flux. Typically, a user interest experiences a emergent phase, a stable phase, and a declining phase, which are referred to as the "user interest life-cycle". Recent papers on user interest modeling have primarily focused on how to compute the correlation between the target item and user's historical behaviors, without thoroughly considering the life-cycle features of user interest. In this paper, we propose an effective method called Deep Interest Life-cycle Network (DILN), which not only captures the interest life-cycle features efficiently, but can also be easily integrated to existing ranking models. DILN contains two key components: Interest Life-cycle Encoder Module constructs historical activity histograms of the user interest and then encodes them into dense representation. Interest Life-cycle Fusion Module injects the encoded dense representation into multiple expert networks, with the aim of enabling the specific phase of interest life-cycle to activate distinct experts. Online A/B testing reveals that DILN achieves significant improvements of +0.38% in CTR, +1.04% in CVR and +0.25% in duration per user, which demonstrates its effectiveness. In addition, DILN inherently increase the exposure of users' emergent and stable interests while decreasing the exposure of declining interests. DILN has been deployed on the Lofter App. 

---
# Lost in Transliteration: Bridging the Script Gap in Neural IR 

**Authors**: Andreas Chari, Iadh Ounis, Sean MacAvaney  

**Link**: [PDF](https://arxiv.org/pdf/2505.08411)  

**Abstract**: Most human languages use scripts other than the Latin alphabet. Search users in these languages often formulate their information needs in a transliterated -- usually Latinized -- form for ease of typing. For example, Greek speakers might use Greeklish, and Arabic speakers might use Arabizi. This paper shows that current search systems, including those that use multilingual dense embeddings such as BGE-M3, do not generalise to this setting, and their performance rapidly deteriorates when exposed to transliterated queries. This creates a ``script gap" between the performance of the same queries when written in their native or transliterated form. We explore whether adapting the popular ``translate-train" paradigm to transliterations can enhance the robustness of multilingual Information Retrieval (IR) methods and bridge the gap between native and transliterated scripts. By exploring various combinations of non-Latin and Latinized query text for training, we investigate whether we can enhance the capacity of existing neural retrieval techniques and enable them to apply to this important setting. We show that by further fine-tuning IR models on an even mixture of native and Latinized text, they can perform this cross-script matching at nearly the same performance as when the query was formulated in the native script. Out-of-domain evaluation and further qualitative analysis show that transliterations can also cause queries to lose some of their nuances, motivating further research in this direction. 

---
# TikTok Search Recommendations: Governance and Research Challenges 

**Authors**: Taylor Annabell, Robert Gorwa, Rebecca Scharlach, Jacob van de Kerkhof, Thales Bertaglia  

**Link**: [PDF](https://arxiv.org/pdf/2505.08385)  

**Abstract**: Like other social media, TikTok is embracing its use as a search engine, developing search products to steer users to produce searchable content and engage in content discovery. Their recently developed product search recommendations are preformulated search queries recommended to users on videos. However, TikTok provides limited transparency about how search recommendations are generated and moderated, despite requirements under regulatory frameworks like the European Union's Digital Services Act. By suggesting that the platform simply aggregates comments and common searches linked to videos, it sidesteps responsibility and issues that arise from contextually problematic recommendations, reigniting long-standing concerns about platform liability and moderation. This position paper addresses the novelty of search recommendations on TikTok by highlighting the challenges that this feature poses for platform governance and offering a computational research agenda, drawing on preliminary qualitative analysis. It sets out the need for transparency in platform documentation, data access and research to study search recommendations. 

---
# Hyperbolic Contrastive Learning with Model-augmentation for Knowledge-aware Recommendation 

**Authors**: Shengyin Sun, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.08157)  

**Abstract**: Benefiting from the effectiveness of graph neural networks (GNNs) and contrastive learning, GNN-based contrastive learning has become mainstream for knowledge-aware recommendation. However, most existing contrastive learning-based methods have difficulties in effectively capturing the underlying hierarchical structure within user-item bipartite graphs and knowledge graphs. Moreover, they commonly generate positive samples for contrastive learning by perturbing the graph structure, which may lead to a shift in user preference learning. To overcome these limitations, we propose hyperbolic contrastive learning with model-augmentation for knowledge-aware recommendation. To capture the intrinsic hierarchical graph structures, we first design a novel Lorentzian knowledge aggregation mechanism, which enables more effective representations of users and items. Then, we propose three model-level augmentation techniques to assist Hyperbolic contrastive learning. Different from the classical structure-level augmentation (e.g., edge dropping), the proposed model-augmentations can avoid preference shifts between the augmented positive pair. Finally, we conduct extensive experiments to demonstrate the superiority (maximum improvement of $11.03\%$) of proposed methods over existing baselines. 

---
# Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation 

**Authors**: Linus Stuhlmann, Michael Alexander Saxer, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07917)  

**Abstract**: Biomedical question-answering (QA) systems require effective retrieval and generation components to ensure accuracy, efficiency, and scalability. This study systematically examines a Retrieval-Augmented Generation (RAG) system for biomedical QA, evaluating retrieval strategies and response time trade-offs. We first assess state-of-the-art retrieval methods, including BM25, BioBERT, MedCPT, and a hybrid approach, alongside common data stores such as Elasticsearch, MongoDB, and FAISS, on a ~10% subset of PubMed (2.4M documents) to measure indexing efficiency, retrieval latency, and retriever performance in the end-to-end RAG system. Based on these insights, we deploy the final RAG system on the full 24M PubMed corpus, comparing different retrievers' impact on overall performance. Evaluations of the retrieval depth show that retrieving 50 documents with BM25 before reranking with MedCPT optimally balances accuracy (0.90), recall (0.90), and response time (1.91s). BM25 retrieval time remains stable (82ms), while MedCPT incurs the main computational cost. These results highlight previously not well-known trade-offs in retrieval depth, efficiency, and scalability for biomedical QA. With open-source code, the system is fully reproducible and extensible. 

---
# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval 

**Authors**: Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07879)  

**Abstract**: Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems. 

---
# Securing RAG: A Risk Assessment and Mitigation Framework 

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.08728)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems. 

---
# Development of a WAZOBIA-Named Entity Recognition System 

**Authors**: S.E Emedem, I.E Onyenwe, E. G Onyedinma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07884)  

**Abstract**: Named Entity Recognition NER is very crucial for various natural language processing applications, including information extraction, machine translation, and sentiment analysis. Despite the ever-increasing interest in African languages within computational linguistics, existing NER systems focus mainly on English, European, and a few other global languages, leaving a significant gap for under-resourced languages. This research presents the development of a WAZOBIA-NER system tailored for the three most prominent Nigerian languages: Hausa, Yoruba, and Igbo. This research begins with a comprehensive compilation of annotated datasets for each language, addressing data scarcity and linguistic diversity challenges. Exploring the state-of-the-art machine learning technique, Conditional Random Fields (CRF) and deep learning models such as Bidirectional Long Short-Term Memory (BiLSTM), Bidirectional Encoder Representation from Transformers (Bert) and fine-tune with a Recurrent Neural Network (RNN), the study evaluates the effectiveness of these approaches in recognizing three entities: persons, organizations, and locations. The system utilizes optical character recognition (OCR) technology to convert textual images into machine-readable text, thereby enabling the Wazobia system to accept both input text and textual images for extraction purposes. The system achieved a performance of 0.9511 in precision, 0.9400 in recall, 0.9564 in F1-score, and 0.9301 in accuracy. The model's evaluation was conducted across three languages, with precision, recall, F1-score, and accuracy as key assessment metrics. The Wazobia-NER system demonstrates that it is feasible to build robust NER tools for under-resourced African languages using current NLP frameworks and transfer learning. 

---
# SweRank: Software Issue Localization with Code Ranking 

**Authors**: Revanth Gangi Reddy, Tarun Suresh, JaeHyeok Doo, Ye Liu, Xuan Phi Nguyen, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Heng Ji, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.07849)  

**Abstract**: Software issue localization, the task of identifying the precise code locations (files, classes, or functions) relevant to a natural language issue description (e.g., bug report, feature request), is a critical yet time-consuming aspect of software development. While recent LLM-based agentic approaches demonstrate promise, they often incur significant latency and cost due to complex multi-step reasoning and relying on closed-source LLMs. Alternatively, traditional code ranking models, typically optimized for query-to-code or code-to-code retrieval, struggle with the verbose and failure-descriptive nature of issue localization queries. To bridge this gap, we introduce SweRank, an efficient and effective retrieve-and-rerank framework for software issue localization. To facilitate training, we construct SweLoc, a large-scale dataset curated from public GitHub repositories, featuring real-world issue descriptions paired with corresponding code modifications. Empirical results on SWE-Bench-Lite and LocBench show that SweRank achieves state-of-the-art performance, outperforming both prior ranking models and costly agent-based systems using closed-source LLMs like Claude-3.5. Further, we demonstrate SweLoc's utility in enhancing various existing retriever and reranker models for issue localization, establishing the dataset as a valuable resource for the community. 

---
