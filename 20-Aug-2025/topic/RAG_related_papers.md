# Revisiting RAG Ensemble: A Theoretical and Mechanistic Analysis of Multi-RAG System Collaboration 

**Authors**: Yifei Chen, Guanting Dong, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13828)  

**Abstract**: Retrieval-Augmented Generation (RAG) technology has been widely applied in recent years. However, despite the emergence of various RAG frameworks, a single RAG framework still cannot adapt well to a broad range of downstream tasks. Therefore, how to leverage the advantages of multiple RAG systems has become an area worth exploring. To address this issue, we have conducted a comprehensive and systematic investigation into ensemble methods based on RAG systems. Specifically, we have analyzed the RAG ensemble framework from both theoretical and mechanistic analysis perspectives. From the theoretical analysis, we provide the first explanation of the RAG ensemble framework from the perspective of information entropy. In terms of mechanism analysis, we have explored the RAG ensemble framework from both the pipeline and module levels. We carefully select four different pipelines (Branching, Iterative, Loop, and Agentic) and three different modules (Generator, Retriever, and Reranker) to solve seven different research questions. The experiments show that aggregating multiple RAG systems is both generalizable and robust, whether at the pipeline level or the module level. Our work lays the foundation for similar research on the multi-RAG system ensemble. 

---
# CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support 

**Authors**: Yuting Zhang, Karina V. Bunting, Asgher Champsi, Xiaoxia Wang, Wenqi Lu, Alexander Thorley, Sandeep S Hothi, Zhaowen Qiu, Dipak Kotecha, Jinming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13256)  

**Abstract**: Cardiovascular diseases (CVDs) remain the foremost cause of mortality worldwide, a burden worsened by a severe deficit of healthcare workers. Artificial intelligence (AI) agents have shown potential to alleviate this gap via automated early detection and proactive screening, yet their clinical application remains limited by: 1) prompt-based clinical role assignment that relies on intrinsic model capabilities without domain-specific tool support; or 2) rigid sequential workflows, whereas clinical care often requires adaptive reasoning that orders specific tests and, based on their results, guides personalised next steps; 3) general and static knowledge bases without continuous learning capability; and 4) fixed unimodal or bimodal inputs and lack of on-demand visual outputs when further clarification is needed. In response, a multimodal framework, CardAIc-Agents, was proposed to augment models with external tools and adaptively support diverse cardiac tasks. Specifically, a CardiacRAG agent generated general plans from updatable cardiac knowledge, while the chief agent integrated tools to autonomously execute these plans and deliver decisions. To enable adaptive and case-specific customization, a stepwise update strategy was proposed to dynamically refine plans based on preceding execution results, once the task was assessed as complex. In addition, a multidisciplinary discussion tool was introduced to interpret challenging cases, thereby supporting further adaptation. When clinicians raised concerns, visual review panels were provided to assist final validation. Experiments across three datasets showed the efficiency of CardAIc-Agents compared to mainstream Vision-Language Models (VLMs), state-of-the-art agentic systems, and fine-tuned VLMs. 

---
# EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation 

**Authors**: Yi Wang, Haoran Luo, Lu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13735)  

**Abstract**: With the widespread application of electroencephalography (EEG) in neuroscience and clinical practice, efficiently retrieving and semantically interpreting large-scale, multi-source, heterogeneous EEG data has become a pressing challenge. We propose EEG-MedRAG, a three-layer hypergraph-based retrieval-augmented generation framework that unifies EEG domain knowledge, individual patient cases, and a large-scale repository into a traversable n-ary relational hypergraph, enabling joint semantic-temporal retrieval and causal-chain diagnostic generation. Concurrently, we introduce the first cross-disease, cross-role EEG clinical QA benchmark, spanning seven disorders and five authentic clinical perspectives. This benchmark allows systematic evaluation of disease-agnostic generalization and role-aware contextual understanding. Experiments show that EEG-MedRAG significantly outperforms TimeRAG and HyperGraphRAG in answer accuracy and retrieval, highlighting its strong potential for real-world clinical decision support. Our data and code are publicly available at this https URL. 

---
