# Optimizing Retrieval Strategies for Financial Question Answering Documents in Retrieval-Augmented Generation Systems 

**Authors**: Sejong Kim, Hyunseo Song, Hyunwoo Seo, Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15191)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework to mitigate hallucinations in Large Language Models (LLMs), yet its overall performance is dependent on the underlying retrieval system. In the finance domain, documents such as 10-K reports pose distinct challenges due to domain-specific vocabulary and multi-hierarchical tabular data. In this work, we introduce an efficient, end-to-end RAG pipeline that enhances retrieval for financial documents through a three-phase approach: pre-retrieval, retrieval, and post-retrieval. In the pre-retrieval phase, various query and corpus preprocessing techniques are employed to enrich input data. During the retrieval phase, we fine-tuned state-of-the-art (SOTA) embedding models with domain-specific knowledge and implemented a hybrid retrieval strategy that combines dense and sparse representations. Finally, the post-retrieval phase leverages Direct Preference Optimization (DPO) training and document selection methods to further refine the results. Evaluations on seven financial question answering datasets-FinDER, FinQABench, FinanceBench, TATQA, FinQA, ConvFinQA, and MultiHiertt-demonstrate substantial improvements in retrieval performance, leading to more accurate and contextually appropriate generation. These findings highlight the critical role of tailored retrieval techniques in advancing the effectiveness of RAG systems for financial applications. A fully replicable pipeline is available on GitHub: this https URL. 

---
# Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval 

**Authors**: Hang Li, Xiao Wang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2503.14887)  

**Abstract**: Pseudo-relevance feedback (PRF) refines queries by leveraging initially retrieved documents to improve retrieval effectiveness. In this paper, we investigate how large language models (LLMs) can facilitate PRF for zero-shot LLM-based dense retrieval, extending the recently proposed PromptReps method. Specifically, our approach uses LLMs to extract salient passage features-such as keywords and summaries-from top-ranked documents, which are then integrated into PromptReps to produce enhanced query representations. Experiments on passage retrieval benchmarks demonstrate that incorporating PRF significantly boosts retrieval performance. Notably, smaller rankers with PRF can match the effectiveness of larger rankers without PRF, highlighting PRF's potential to improve LLM-driven search while maintaining an efficient balance between effectiveness and resource usage. 

---
# Graph-Based Re-ranking: Emerging Techniques, Limitations, and Opportunities 

**Authors**: Md Shahir Zaoad, Niamat Zawad, Priyanka Ranade, Richard Krogman, Latifur Khan, James Holt  

**Link**: [PDF](https://arxiv.org/pdf/2503.14802)  

**Abstract**: Knowledge graphs have emerged to be promising datastore candidates for context augmentation during Retrieval Augmented Generation (RAG). As a result, techniques in graph representation learning have been simultaneously explored alongside principal neural information retrieval approaches, such as two-phased retrieval, also known as re-ranking. While Graph Neural Networks (GNNs) have been proposed to demonstrate proficiency in graph learning for re-ranking, there are ongoing limitations in modeling and evaluating input graph structures for training and evaluation for passage and document ranking tasks. In this survey, we review emerging GNN-based ranking model architectures along with their corresponding graph representation construction methodologies. We conclude by providing recommendations on future research based on community-wide challenges and opportunities. 

---
# Long Context Modeling with Ranked Memory-Augmented Retrieval 

**Authors**: Ghadir Alselwi, Hao Xue, Shoaib Jameel, Basem Suleiman, Flora D. Salim, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2503.14800)  

**Abstract**: Effective long-term memory management is crucial for language models handling extended contexts. We introduce a novel framework that dynamically ranks memory entries based on relevance. Unlike previous works, our model introduces a novel relevance scoring and a pointwise re-ranking model for key-value embeddings, inspired by learning-to-rank techniques in information retrieval. Enhanced Ranked Memory Augmented Retrieval ERMAR achieves state-of-the-art results on standard benchmarks. 

---
# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving 

**Authors**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14649)  

**Abstract**: Retrieval-augmented generation (RAG), which combines large language models (LLMs) with retrievals from external knowledge databases, is emerging as a popular approach for reliable LLM serving. However, efficient RAG serving remains an open challenge due to the rapid emergence of many RAG variants and the substantial differences in workload characteristics across them. In this paper, we make three fundamental contributions to advancing RAG serving. First, we introduce RAGSchema, a structured abstraction that captures the wide range of RAG algorithms, serving as a foundation for performance optimization. Second, we analyze several representative RAG workloads with distinct RAGSchema, revealing significant performance variability across these workloads. Third, to address this variability and meet diverse performance requirements, we propose RAGO (Retrieval-Augmented Generation Optimizer), a system optimization framework for efficient RAG serving. Our evaluation shows that RAGO achieves up to a 2x increase in QPS per chip and a 55% reduction in time-to-first-token latency compared to RAG systems built on LLM-system extensions. 

---
# When Pigs Get Sick: Multi-Agent AI for Swine Disease Detection 

**Authors**: Tittaya Mairittha, Tanakon Sawanglok, Panuwit Raden, Sorrawit Treesuk  

**Link**: [PDF](https://arxiv.org/pdf/2503.15204)  

**Abstract**: Swine disease surveillance is critical to the sustainability of global agriculture, yet its effectiveness is frequently undermined by limited veterinary resources, delayed identification of cases, and variability in diagnostic accuracy. To overcome these barriers, we introduce a novel AI-powered, multi-agent diagnostic system that leverages Retrieval-Augmented Generation (RAG) to deliver timely, evidence-based disease detection and clinical guidance. By automatically classifying user inputs into either Knowledge Retrieval Queries or Symptom-Based Diagnostic Queries, the system ensures targeted information retrieval and facilitates precise diagnostic reasoning. An adaptive questioning protocol systematically collects relevant clinical signs, while a confidence-weighted decision fusion mechanism integrates multiple diagnostic hypotheses to generate robust disease predictions and treatment recommendations. Comprehensive evaluations encompassing query classification, disease diagnosis, and knowledge retrieval demonstrate that the system achieves high accuracy, rapid response times, and consistent reliability. By providing a scalable, AI-driven diagnostic framework, this approach enhances veterinary decision-making, advances sustainable livestock management practices, and contributes substantively to the realization of global food security. 

---
