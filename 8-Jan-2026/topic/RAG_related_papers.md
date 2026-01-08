# Trade-R1: Bridging Verifiable Rewards to Stochastic Environments via Process-Level Reasoning Verification 

**Authors**: Rui Sun, Yifan Sun, Sheng Xu, Li Zhao, Jing Li, Daxin Jiang, Chen Hua, Zuo Bai  

**Link**: [PDF](https://arxiv.org/pdf/2601.03948)  

**Abstract**: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to achieve remarkable reasoning in domains like mathematics and coding, where verifiable rewards provide clear signals. However, extending this paradigm to financial decision is challenged by the market's stochastic nature: rewards are verifiable but inherently noisy, causing standard RL to degenerate into reward hacking. To address this, we propose Trade-R1, a model training framework that bridges verifiable rewards to stochastic environments via process-level reasoning verification. Our key innovation is a verification method that transforms the problem of evaluating reasoning over lengthy financial documents into a structured Retrieval-Augmented Generation (RAG) task. We construct a triangular consistency metric, assessing pairwise alignment between retrieved evidence, reasoning chains, and decisions to serve as a validity filter for noisy market returns. We explore two reward integration strategies: Fixed-effect Semantic Reward (FSR) for stable alignment signals, and Dynamic-effect Semantic Reward (DSR) for coupled magnitude optimization. Experiments on different country asset selection demonstrate that our paradigm reduces reward hacking, with DSR achieving superior cross-market generalization while maintaining the highest reasoning consistency. 

---
# Decide Then Retrieve: A Training-Free Framework with Uncertainty-Guided Triggering and Dual-Path Retrieval 

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03908)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but existing approaches indiscriminately trigger retrieval and rely on single-path evidence construction, often introducing noise and limiting performance gains. In this work, we propose Decide Then Retrieve (DTR), a training-free framework that adaptively determines when retrieval is necessary and how external information should be selected. DTR leverages generation uncertainty to guide retrieval triggering and introduces a dual-path retrieval mechanism with adaptive information selection to better handle sparse and ambiguous queries. Extensive experiments across five open-domain QA benchmarks, multiple model scales, and different retrievers demonstrate that DTR consistently improves EM and F1 over standard RAG and strong retrieval-enhanced baselines, while reducing unnecessary retrievals. The code and data used in this paper are available at this https URL. 

---
# VietMed-MCQ: A Consistency-Filtered Data Synthesis Framework for Vietnamese Traditional Medicine Evaluation 

**Authors**: Huynh Trung Kiet, Dao Sy Duy Minh, Nguyen Dinh Ha Duong, Le Hoang Minh Huy, Long Nguyen, Dien Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2601.03792)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in general medical domains. However, their performance significantly degrades in specialized, culturally specific domains such as Vietnamese Traditional Medicine (VTM), primarily due to the scarcity of high-quality, structured benchmarks. In this paper, we introduce VietMed-MCQ, a novel multiple-choice question dataset generated via a Retrieval-Augmented Generation (RAG) pipeline with an automated consistency check mechanism. Unlike previous synthetic datasets, our framework incorporates a dual-model validation approach to ensure reasoning consistency through independent answer verification, though the substring-based evidence checking has known limitations. The complete dataset of 3,190 questions spans three difficulty levels and underwent validation by one medical expert and four students, achieving 94.2 percent approval with substantial inter-rater agreement (Fleiss' kappa = 0.82). We benchmark seven open-source models on VietMed-MCQ. Results reveal that general-purpose models with strong Chinese priors outperform Vietnamese-centric models, highlighting cross-lingual conceptual transfer, while all models still struggle with complex diagnostic reasoning. Our code and dataset are publicly available to foster research in low-resource medical domains. 

---
# Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey 

**Authors**: Xiantao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03262)  

**Abstract**: Visually rich documents (VRDs) challenge retrieval-augmented generation (RAG) with layout-dependent semantics, brittle OCR, and evidence spread across complex figures and structured tables. This survey examines how Multimodal Large Language Models (MLLMs) are being used to make VRD retrieval practical for RAG. We organize the literature into three roles: Modality-Unifying Captioners, Multimodal Embedders, and End-to-End Representers. We compare these roles along retrieval granularity, information fidelity, latency and index size, and compatibility with reranking and grounding. We also outline key trade-offs and offer some practical guidance on when to favor each role. Finally, we identify promising directions for future research, including adaptive retrieval units, model size reduction, and the development of evaluation methods. 

---
# Enhancing Retrieval-Augmented Generation with Two-Stage Retrieval: FlashRank Reranking and Query Expansion 

**Authors**: Sherine George  

**Link**: [PDF](https://arxiv.org/pdf/2601.03258)  

**Abstract**: Retrieval-Augmented Generation (RAG) couples a retriever with a large language model (LLM) to ground generated responses in external evidence. While this framework enhances factuality and domain adaptability, it faces a key bottleneck: balancing retrieval recall with limited LLM context. Retrieving too few passages risks missing critical context, while retrieving too many overwhelms the prompt window, diluting relevance and increasing cost.
We propose a two-stage retrieval pipeline that integrates LLM-driven query expansion to improve candidate recall and FlashRank, a fast marginal-utility reranker that dynamically selects an optimal subset of evidence under a token budget. FlashRank models document utility as a weighted combination of relevance, novelty, brevity, and cross-encoder evidence. Together, these modules form a generalizable solution that increases answer accuracy, faithfulness, and computational efficiency. 

---
