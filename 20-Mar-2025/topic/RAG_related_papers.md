# Optimizing Retrieval Strategies for Financial Question Answering Documents in Retrieval-Augmented Generation Systems 

**Authors**: Sejong Kim, Hyunseo Song, Hyunwoo Seo, Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.15191)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework to mitigate hallucinations in Large Language Models (LLMs), yet its overall performance is dependent on the underlying retrieval system. In the finance domain, documents such as 10-K reports pose distinct challenges due to domain-specific vocabulary and multi-hierarchical tabular data. In this work, we introduce an efficient, end-to-end RAG pipeline that enhances retrieval for financial documents through a three-phase approach: pre-retrieval, retrieval, and post-retrieval. In the pre-retrieval phase, various query and corpus preprocessing techniques are employed to enrich input data. During the retrieval phase, we fine-tuned state-of-the-art (SOTA) embedding models with domain-specific knowledge and implemented a hybrid retrieval strategy that combines dense and sparse representations. Finally, the post-retrieval phase leverages Direct Preference Optimization (DPO) training and document selection methods to further refine the results. Evaluations on seven financial question answering datasets-FinDER, FinQABench, FinanceBench, TATQA, FinQA, ConvFinQA, and MultiHiertt-demonstrate substantial improvements in retrieval performance, leading to more accurate and contextually appropriate generation. These findings highlight the critical role of tailored retrieval techniques in advancing the effectiveness of RAG systems for financial applications. A fully replicable pipeline is available on GitHub: this https URL. 

---
# When Pigs Get Sick: Multi-Agent AI for Swine Disease Detection 

**Authors**: Tittaya Mairittha, Tanakon Sawanglok, Panuwit Raden, Sorrawit Treesuk  

**Link**: [PDF](https://arxiv.org/pdf/2503.15204)  

**Abstract**: Swine disease surveillance is critical to the sustainability of global agriculture, yet its effectiveness is frequently undermined by limited veterinary resources, delayed identification of cases, and variability in diagnostic accuracy. To overcome these barriers, we introduce a novel AI-powered, multi-agent diagnostic system that leverages Retrieval-Augmented Generation (RAG) to deliver timely, evidence-based disease detection and clinical guidance. By automatically classifying user inputs into either Knowledge Retrieval Queries or Symptom-Based Diagnostic Queries, the system ensures targeted information retrieval and facilitates precise diagnostic reasoning. An adaptive questioning protocol systematically collects relevant clinical signs, while a confidence-weighted decision fusion mechanism integrates multiple diagnostic hypotheses to generate robust disease predictions and treatment recommendations. Comprehensive evaluations encompassing query classification, disease diagnosis, and knowledge retrieval demonstrate that the system achieves high accuracy, rapid response times, and consistent reliability. By providing a scalable, AI-driven diagnostic framework, this approach enhances veterinary decision-making, advances sustainable livestock management practices, and contributes substantively to the realization of global food security. 

---
# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving 

**Authors**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14649)  

**Abstract**: Retrieval-augmented generation (RAG), which combines large language models (LLMs) with retrievals from external knowledge databases, is emerging as a popular approach for reliable LLM serving. However, efficient RAG serving remains an open challenge due to the rapid emergence of many RAG variants and the substantial differences in workload characteristics across them. In this paper, we make three fundamental contributions to advancing RAG serving. First, we introduce RAGSchema, a structured abstraction that captures the wide range of RAG algorithms, serving as a foundation for performance optimization. Second, we analyze several representative RAG workloads with distinct RAGSchema, revealing significant performance variability across these workloads. Third, to address this variability and meet diverse performance requirements, we propose RAGO (Retrieval-Augmented Generation Optimizer), a system optimization framework for efficient RAG serving. Our evaluation shows that RAGO achieves up to a 2x increase in QPS per chip and a 55% reduction in time-to-first-token latency compared to RAG systems built on LLM-system extensions. 

---
# Evaluating Bias in Retrieval-Augmented Medical Question-Answering Systems 

**Authors**: Yuelyu Ji, Hang Zhang, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15454)  

**Abstract**: Medical QA systems powered by Retrieval-Augmented Generation (RAG) models support clinical decision-making but may introduce biases related to race, gender, and social determinants of health. We systematically evaluate biases in RAG-based LLM by examining demographic-sensitive queries and measuring retrieval discrepancies. Using datasets like MMLU and MedMCQA, we analyze retrieval overlap and correctness disparities. Our findings reveal substantial demographic disparities within RAG pipelines, emphasizing the critical need for retrieval methods that explicitly account for fairness to ensure equitable clinical decision-making. 

---
