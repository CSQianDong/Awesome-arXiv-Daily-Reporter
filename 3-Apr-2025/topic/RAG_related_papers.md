# LARGE: Legal Retrieval Augmented Generation Evaluation Tool 

**Authors**: Minhu Park, Hongseok Oh, Eunkyung Choi, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01840)  

**Abstract**: Recently, building retrieval-augmented generation (RAG) systems to enhance the capability of large language models (LLMs) has become a common practice. Especially in the legal domain, previous judicial decisions play a significant role under the doctrine of stare decisis which emphasizes the importance of making decisions based on (retrieved) prior documents. However, the overall performance of RAG system depends on many components: (1) retrieval corpora, (2) retrieval algorithms, (3) rerankers, (4) LLM backbones, and (5) evaluation metrics. Here we propose LRAGE, an open-source tool for holistic evaluation of RAG systems focusing on the legal domain. LRAGE provides GUI and CLI interfaces to facilitate seamless experiments and investigate how changes in the aforementioned five components affect the overall accuracy. We validated LRAGE using multilingual legal benches including Korean (KBL), English (LegalBench), and Chinese (LawBench) by demonstrating how the overall accuracy changes when varying the five components mentioned above. The source code is available at this https URL. 

---
# GTR: Graph-Table-RAG for Cross-Table Question Answering 

**Authors**: Jiaru Zou, Dongqi Fu, Sirui Chen, Xinrui He, Zihao Li, Yada Zhu, Jiawei Han, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2504.01346)  

**Abstract**: Beyond pure text, a substantial amount of knowledge is stored in tables. In real-world scenarios, user questions often require retrieving answers that are distributed across multiple tables. GraphRAG has recently attracted much attention for enhancing LLMs' reasoning capabilities by organizing external knowledge to address ad-hoc and complex questions, exemplifying a promising direction for cross-table question answering. In this paper, to address the current gap in available data, we first introduce a multi-table benchmark, MutliTableQA, comprising 60k tables and 25k user queries collected from real-world sources. Then, we propose the first Graph-Table-RAG framework, namely GTR, which reorganizes table corpora into a heterogeneous graph, employs a hierarchical coarse-to-fine retrieval process to extract the most relevant tables, and integrates graph-aware prompting for downstream LLMs' tabular reasoning. Extensive experiments show that GTR exhibits superior cross-table question-answering performance while maintaining high deployment efficiency, demonstrating its real-world practical applicability. 

---
# CoRAG: Collaborative Retrieval-Augmented Generation 

**Authors**: Aashiq Muhamed, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.01883)  

**Abstract**: Retrieval-Augmented Generation (RAG) models excel in knowledge-intensive tasks, especially under few-shot learning constraints. We introduce CoRAG, a framework extending RAG to collaborative settings, where clients jointly train a shared model using a collaborative passage store. To evaluate CoRAG, we introduce CRAB, a benchmark for collaborative homogeneous open-domain question answering. Our experiments demonstrate that CoRAG consistently outperforms both parametric collaborative learning methods and locally trained RAG models in low-resource scenarios. Further analysis reveals the critical importance of relevant passages within the shared store, the surprising benefits of incorporating irrelevant passages, and the potential for hard negatives to negatively impact performance. This introduces a novel consideration in collaborative RAG: the trade-off between leveraging a collectively enriched knowledge base and the potential risk of incorporating detrimental passages from other clients. Our findings underscore the viability of CoRAG, while also highlighting key design challenges and promising avenues for future research. 

---
# Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding 

**Authors**: Sakhinana Sagar Srinivas, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.01281)  

**Abstract**: We present a comprehensive framework for enhancing Retrieval-Augmented Generation (RAG) systems through dynamic retrieval strategies and reinforcement fine-tuning. This approach significantly improves large language models on knowledge-intensive tasks, including opendomain question answering and complex reasoning. Our framework integrates two complementary techniques: Policy-Optimized RetrievalAugmented Generation (PORAG), which optimizes the use of retrieved information, and Adaptive Token-Layer Attention Scoring (ATLAS), which dynamically determines retrieval timing and content based on contextual needs. Together, these techniques enhance both the utilization and relevance of retrieved content, improving factual accuracy and response quality. Designed as a lightweight solution compatible with any Transformer-based LLM without requiring additional training, our framework excels in knowledge-intensive tasks, boosting output accuracy in RAG settings. We further propose CRITIC, a novel method to selectively compress key-value caches by token importance, mitigating memory bottlenecks in long-context applications. The framework also incorporates test-time scaling techniques to dynamically balance reasoning depth and computational resources, alongside optimized decoding strategies for faster inference. Experiments on benchmark datasets show that our framework reduces hallucinations, strengthens domain-specific reasoning, and achieves significant efficiency and scalability gains over traditional RAG systems. This integrated approach advances the development of robust, efficient, and scalable RAG systems across diverse applications. 

---
# From Code Generation to Software Testing: AI Copilot with Context-Based RAG 

**Authors**: Yuchen Wang, Shangxin Guo, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01866)  

**Abstract**: The rapid pace of large-scale software development places increasing demands on traditional testing methodologies, often leading to bottlenecks in efficiency, accuracy, and coverage. We propose a novel perspective on software testing by positing bug detection and coding with fewer bugs as two interconnected problems that share a common goal, which is reducing bugs with limited resources. We extend our previous work on AI-assisted programming, which supports code auto-completion and chatbot-powered Q&A, to the realm of software testing. We introduce Copilot for Testing, an automated testing system that synchronizes bug detection with codebase updates, leveraging context-based Retrieval Augmented Generation (RAG) to enhance the capabilities of large language models (LLMs). Our evaluation demonstrates a 31.2% improvement in bug detection accuracy, a 12.6% increase in critical test coverage, and a 10.5% higher user acceptance rate, highlighting the transformative potential of AI-driven technologies in modern software development practices. 

---
# Reasoning LLMs for User-Aware Multimodal Conversational Agents 

**Authors**: Hamed Rahimi, Jeanne Cattoni, Meriem Beghili, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01700)  

**Abstract**: Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment. 

---
# GeoRAG: A Question-Answering Approach from a Geographical Perspective 

**Authors**: Jian Wang, Zhuo Zhao, Zheng Jie Wang, Bo Da Cheng, Lei Nie, Wen Luo, Zhao Yuan Yu, Ling Wang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01458)  

**Abstract**: Geographic Question Answering (GeoQA) addresses natural language queries in geographic domains to fulfill complex user demands and improve information retrieval efficiency. Traditional QA systems, however, suffer from limited comprehension, low retrieval accuracy, weak interactivity, and inadequate handling of complex tasks, hindering precise information acquisition. This study presents GeoRAG, a knowledge-enhanced QA framework integrating domain-specific fine-tuning and prompt engineering with Retrieval-Augmented Generation (RAG) technology to enhance geographic knowledge retrieval accuracy and user interaction. The methodology involves four components: (1) A structured geographic knowledge base constructed from 3267 corpora (research papers, monographs, and technical reports), categorized via a multi-agent approach into seven dimensions: semantic understanding, spatial location, geometric morphology, attribute characteristics, feature relationships, evolutionary processes, and operational mechanisms. This yielded 145234 classified entries and 875432 multi-dimensional QA pairs. (2) A multi-label text classifier based on BERT-Base-Chinese, trained to analyze query types through geographic dimension classification. (3) A retrieval evaluator leveraging QA pair data to assess query-document relevance, optimizing retrieval precision. (4) GeoPrompt templates engineered to dynamically integrate user queries with retrieved information, enhancing response quality through dimension-specific prompting. Comparative experiments demonstrate GeoRAG's superior performance over conventional RAG across multiple base models, validating its generalizability. This work advances geographic AI by proposing a novel paradigm for deploying large language models in domain-specific contexts, with implications for improving GeoQA systems scalability and accuracy in real-world applications. 

---
