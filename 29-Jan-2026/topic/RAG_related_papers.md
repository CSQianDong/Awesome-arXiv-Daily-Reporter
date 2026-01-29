# Harnessing Large Language Models for Precision Querying and Retrieval-Augmented Knowledge Extraction in Clinical Data Science 

**Authors**: Juan Jose Rubio Jan, Jack Wu, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2601.20674)  

**Abstract**: This study applies Large Language Models (LLMs) to two foundational Electronic Health Record (EHR) data science tasks: structured data querying (using programmatic languages, Python/Pandas) and information extraction from unstructured clinical text via a Retrieval Augmented Generation (RAG) pipeline. We test the ability of LLMs to interact accurately with large structured datasets for analytics and the reliability of LLMs in extracting semantically correct information from free text health records when supported by RAG. To this end, we presented a flexible evaluation framework that automatically generates synthetic question and answer pairs tailored to the characteristics of each dataset or task. Experiments were conducted on a curated subset of MIMIC III, (four structured tables and one clinical note type), using a mix of locally hosted and API-based LLMs. Evaluation combined exact-match metrics, semantic similarity, and human judgment. Our findings demonstrate the potential of LLMs to support precise querying and accurate information extraction in clinical workflows. 

---
# Attribution Techniques for Mitigating Hallucinated Information in RAG Systems: A Survey 

**Authors**: Yuqing Zhao, Ziyao Liu, Yongsen Zheng, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2601.19927)  

**Abstract**: Large Language Models (LLMs)-based question answering (QA) systems play a critical role in modern AI, demonstrating strong performance across various tasks. However, LLM-generated responses often suffer from hallucinations, unfaithful statements lacking reliable references. Retrieval-Augmented Generation (RAG) frameworks enhance LLM responses by incorporating external references but also introduce new forms of hallucination due to complex interactions between the retriever and generator. To address these challenges, researchers have explored attribution-based techniques that ensure responses are verifiably supported by retrieved content. Despite progress, a unified pipeline for these techniques, along with a clear taxonomy and systematic comparison of their strengths and weaknesses, remains lacking. A well-defined taxonomy is essential for identifying specific failure modes within RAG systems, while comparative analysis helps practitioners choose appropriate solutions based on hallucination types and application context. This survey investigates how attribution-based techniques are used within RAG systems to mitigate hallucinations and addresses the gap by: (i) outlining a taxonomy of hallucination types in RAG systems, (ii) presenting a unified pipeline for attribution techniques, (iii) reviewing techniques based on the hallucinations they target, and (iv) discussing strengths and weaknesses with practical guidelines. This work offers insights for future research and practical use of attribution techniques in RAG systems. 

---
# CiMRAG: Cim-Aware Domain-Adaptive and Noise-Resilient Retrieval-Augmented Generation for Edge-Based LLMs 

**Authors**: Shih-Hsuan Chiu, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.20041)  

**Abstract**: Personalized virtual assistants powered by large language models (LLMs) on edge devices are attracting growing attention, with Retrieval-Augmented Generation (RAG) emerging as a key method for personalization by retrieving relevant profile data and generating tailored responses. However, deploying RAG on edge devices faces efficiency hurdles due to the rapid growth of profile data, such as user-LLM interactions and recent updates. While Computing-in-Memory (CiM) architectures mitigate this bottleneck by eliminating data movement between memory and processing units via in-situ operations, they are susceptible to environmental noise that can degrade retrieval precision. This poses a critical issue in dynamic, multi-domain edge-based scenarios (e.g., travel, medicine, and law) where both accuracy and adaptability are paramount. To address these challenges, we propose Task-Oriented Noise-resilient Embedding Learning (TONEL), a framework that improves noise robustness and domain adaptability for RAG in noisy edge environments. TONEL employs a noise-aware projection model to learn task-specific embeddings compatible with CiM hardware constraints, enabling accurate retrieval under noisy conditions. Extensive experiments conducted on personalization benchmarks demonstrate the effectiveness and practicality of our methods relative to strong baselines, especially in task-specific noisy scenarios. 

---
