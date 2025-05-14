# WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation 

**Authors**: Dvir Cohen, Lin Burg, Sviatoslav Pykhnivskyi, Hagit Gur, Stanislav Kovynov, Olga Atzmon, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08643)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a cornerstone of modern question answering (QA) systems, enabling grounded answers based on external knowledge. Although recent progress has been driven by open-domain datasets, enterprise QA systems need datasets that mirror the concrete, domain-specific issues users raise in day-to-day support scenarios. Critically, evaluating end-to-end RAG systems requires benchmarks comprising not only question--answer pairs but also the specific knowledge base (KB) snapshot from which answers were derived. To address this need, we introduce WixQA, a benchmark suite featuring QA datasets precisely grounded in the released KB corpus, enabling holistic evaluation of retrieval and generation components. WixQA includes three distinct QA datasets derived from this http URL customer support interactions and grounded in a snapshot of the public Wix Help Center KB: (i) WixQA-ExpertWritten, 200 real user queries with expert-authored, multi-step answers; (ii) WixQA-Simulated, 200 expert-validated QA pairs distilled from user dialogues; and (iii) WixQA-Synthetic, 6,222 LLM-generated QA pairs, with one pair systematically derived from each article in the knowledge base. We release the KB snapshot alongside the datasets under MIT license and provide comprehensive baseline results, forming a unique benchmark for evaluating enterprise RAG systems in realistic enterprise environments. 

---
# Securing RAG: A Risk Assessment and Mitigation Framework 

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.08728)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems. 

---
# Hakim: Farsi Text Embedding Model 

**Authors**: Mehran Sarmadi, Morteza Alikhani, Erfan Zinvandi, Zahra Pourbahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.08435)  

**Abstract**: Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding. 

---
# Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency 

**Authors**: Adel Ammar, Anis Koubaa, Omer Nacar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2505.08445)  

**Abstract**: Large language models achieve high task performance yet often hallucinate or rely on outdated knowledge. Retrieval-augmented generation (RAG) addresses these gaps by coupling generation with external search. We analyse how hyperparameters influence speed and quality in RAG systems, covering Chroma and Faiss vector stores, chunking policies, cross-encoder re-ranking, and temperature, and we evaluate six metrics: faithfulness, answer correctness, answer relevancy, context precision, context recall, and answer similarity. Chroma processes queries 13% faster, whereas Faiss yields higher retrieval precision, revealing a clear speed-accuracy trade-off. Naive fixed-length chunking with small windows and minimal overlap outperforms semantic segmentation while remaining the quickest option. Re-ranking provides modest gains in retrieval quality yet increases runtime by roughly a factor of 5, so its usefulness depends on latency constraints. These results help practitioners balance computational cost and accuracy when tuning RAG systems for transparent, up-to-date responses. Finally, we re-evaluate the top configurations with a corrective RAG workflow and show that their advantages persist when the model can iteratively request additional evidence. We obtain a near-perfect context precision (99%), which demonstrates that RAG systems can achieve extremely high retrieval accuracy with the right combination of hyperparameters, with significant implications for applications where retrieval quality directly impacts downstream task performance, such as clinical decision support in healthcare. 

---
# Aitomia: Your Intelligent Assistant for AI-Driven Atomistic and Quantum Chemical Simulations 

**Authors**: Jinming Hu, Hassan Nawaz, Yuting Rui, Lijie Chi, Arif Ullah, Pavlo O. Dral  

**Link**: [PDF](https://arxiv.org/pdf/2505.08195)  

**Abstract**: We have developed Aitomia - a platform powered by AI to assist in performing AI-driven atomistic and quantum chemical (QC) simulations. This intelligent assistant platform is equipped with chatbots and AI agents to help experts and guide non-experts in setting up and running the atomistic simulations, monitoring their computation status, analyzing the simulation results, and summarizing them for the user in text and graphical forms. We achieve these goals by exploiting fine-tuned open-source large language models (LLMs), rule-based agents, and a retrieval-augmented generation (RAG) system. Aitomia leverages the versatility of our MLatom ecosystem for AI-enhanced computational chemistry. This intelligent assistant is going to be integrated into the Aitomistic Hub and XACS online computing services, with some functionality already publicly available as described at this http URL. Aitomia is expected to lower the barrier to performing atomistic simulations, accelerating research and development in the relevant fields. 

---
# Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation 

**Authors**: Linus Stuhlmann, Michael Alexander Saxer, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07917)  

**Abstract**: Biomedical question-answering (QA) systems require effective retrieval and generation components to ensure accuracy, efficiency, and scalability. This study systematically examines a Retrieval-Augmented Generation (RAG) system for biomedical QA, evaluating retrieval strategies and response time trade-offs. We first assess state-of-the-art retrieval methods, including BM25, BioBERT, MedCPT, and a hybrid approach, alongside common data stores such as Elasticsearch, MongoDB, and FAISS, on a ~10% subset of PubMed (2.4M documents) to measure indexing efficiency, retrieval latency, and retriever performance in the end-to-end RAG system. Based on these insights, we deploy the final RAG system on the full 24M PubMed corpus, comparing different retrievers' impact on overall performance. Evaluations of the retrieval depth show that retrieving 50 documents with BM25 before reranking with MedCPT optimally balances accuracy (0.90), recall (0.90), and response time (1.91s). BM25 retrieval time remains stable (82ms), while MedCPT incurs the main computational cost. These results highlight previously not well-known trade-offs in retrieval depth, efficiency, and scalability for biomedical QA. With open-source code, the system is fully reproducible and extensible. 

---
# TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking 

**Authors**: Ching Nam Hang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07891)  

**Abstract**: In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT , a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age. 

---
# Patchwork: A Unified Framework for RAG Serving 

**Authors**: Bodun Hu, Luis Pabon, Saurabh Agarwal, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2505.07833)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a new paradigm for enhancing Large Language Model reliability through integration with external knowledge sources. However, efficient deployment of these systems presents significant technical challenges due to their inherently heterogeneous computational pipelines comprising LLMs, databases, and specialized processing components. We introduce Patchwork, a comprehensive end-to-end RAG serving framework designed to address these efficiency bottlenecks. Patchwork's architecture offers three key innovations: First, it provides a flexible specification interface enabling users to implement custom RAG pipelines. Secondly, it deploys these pipelines as distributed inference systems while optimizing for the unique scalability characteristics of individual RAG components. Third, Patchwork incorporates an online scheduling mechanism that continuously monitors request load and execution progress, dynamically minimizing SLO violations through strategic request prioritization and resource auto-scaling. Our experimental evaluation across four distinct RAG implementations demonstrates that Patchwork delivers substantial performance improvements over commercial alternatives, achieving throughput gains exceeding 48% while simultaneously reducing SLO violations by ~24%. 

---
# Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models 

**Authors**: Hussien Al-Asi, Jordan P Reynolds, Shweta Agarwal, Bryan J Dangott, Aziza Nassar, Zeynettin Akkus  

**Link**: [PDF](https://arxiv.org/pdf/2505.08590)  

**Abstract**: Advancements in artificial intelligence (AI) are transforming pathology by integrat-ing large language models (LLMs) with retrieval-augmented generation (RAG) and domain-specific foundation models. This study explores the application of RAG-enhanced LLMs coupled with pathology foundation models for thyroid cytology diagnosis, addressing challenges in cytological interpretation, standardization, and diagnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dy-namic retrieval of relevant case studies, diagnostic criteria, and expert interpreta-tion, improving the contextual understanding of LLMs. Meanwhile, pathology foun-dation models, trained on high-resolution pathology images, refine feature extrac-tion and classification capabilities. The fusion of these AI-driven approaches en-hances diagnostic consistency, reduces variability, and supports pathologists in dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate that integrating RAG with pathology-specific LLMs significantly improves diagnostic efficiency and interpretability, paving the way for AI-assisted thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for correct prediction of surgi-cal pathology diagnosis from thyroid cytology samples. 

---
# IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation 

**Authors**: Kazuki Hayashi, Hidetaka Kamigaito, Shinya Kouda, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.08450)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability. 

---
# Assessing and Mitigating Medical Knowledge Drift and Conflicts in Large Language Models 

**Authors**: Weiyi Wu, Xinwen Xu, Chongyang Gao, Xingjian Diao, Siting Li, Lucas A. Salas, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07968)  

**Abstract**: Large Language Models (LLMs) have great potential in the field of health care, yet they face great challenges in adapting to rapidly evolving medical knowledge. This can lead to outdated or contradictory treatment suggestions. This study investigated how LLMs respond to evolving clinical guidelines, focusing on concept drift and internal inconsistencies. We developed the DriftMedQA benchmark to simulate guideline evolution and assessed the temporal reliability of various LLMs. Our evaluation of seven state-of-the-art models across 4,290 scenarios demonstrated difficulties in rejecting outdated recommendations and frequently endorsing conflicting guidance. Additionally, we explored two mitigation strategies: Retrieval-Augmented Generation and preference fine-tuning via Direct Preference Optimization. While each method improved model performance, their combination led to the most consistent and reliable results. These findings underscore the need to improve LLM robustness to temporal shifts to ensure more dependable applications in clinical practice. 

---
