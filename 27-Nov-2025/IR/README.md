# RIA: A Ranking-Infused Approach for Optimized listwise CTR Prediction 

**Authors**: Guoxiao Zhang, Tan Qu, Ao Li, DongLin Ni, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21394)  

**Abstract**: Reranking improves recommendation quality by modeling item interactions. However, existing methods often decouple ranking and reranking, leading to weak listwise evaluation models that suffer from combinatorial sparsity and limited representational power under strict latency constraints. In this paper, we propose RIA (Ranking-Infused Architecture), a unified, end-to-end framework that seamlessly integrates pointwise and listwise evaluation. RIA introduces four key components: (1) the User and Candidate DualTransformer (UCDT) for fine-grained user-item-context modeling; (2) the Context-aware User History and Target (CUHT) module for position-sensitive preference learning; (3) the Listwise Multi-HSTU (LMH) module to capture hierarchical item dependencies; and (4) the Embedding Cache (EC) module to bridge efficiency and effectiveness during inference. By sharing representations across ranking and reranking, RIA enables rich contextual knowledge transfer while maintaining low latency. Extensive experiments show that RIA outperforms state-of-the-art models on both public and industrial datasets, achieving significant gains in AUC and LogLoss. Deployed in Meituan advertising system, RIA yields a +1.69% improvement in Click-Through Rate (CTR) and a +4.54% increase in Cost Per Mille (CPM) in online A/B tests. 

---
# FITRep: Attention-Guided Item Representation via MLLMs 

**Authors**: Guoxiao Zhang, Ao Li, Tan Qu, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21389)  

**Abstract**: Online platforms usually suffer from user experience degradation due to near-duplicate items with similar visuals and text. While Multimodal Large Language Models (MLLMs) enable multimodal embedding, existing methods treat representations as black boxes, ignoring structural relationships (e.g., primary vs. auxiliary elements), leading to local structural collapse problem. To address this, inspired by Feature Integration Theory (FIT), we propose FITRep, the first attention-guided, white-box item representation framework for fine-grained item deduplication. FITRep consists of: (1) Concept Hierarchical Information Extraction (CHIE), using MLLMs to extract hierarchical semantic concepts; (2) Structure-Preserving Dimensionality Reduction (SPDR), an adaptive UMAP-based method for efficient information compression; and (3) FAISS-Based Clustering (FBC), a FAISS-based clustering that assigns each item a unique cluster id using FAISS. Deployed on Meituan's advertising system, FITRep achieves +3.60% CTR and +4.25% CPM gains in online A/B tests, demonstrating both effectiveness and real-world impact. 

---
# Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval 

**Authors**: Anup Roy, Rishabh Gyanendra Upadhyay, Animesh Rameshbhai Panara, Robin Mills  

**Link**: [PDF](https://arxiv.org/pdf/2511.21121)  

**Abstract**: Document centric RAG pipelines usually begin with OCR, followed by brittle heuristics for chunking, table parsing, and layout reconstruction. These text first workflows are costly to maintain, sensitive to small layout shifts, and often lose the spatial cues that contain the answer. Vision first retrieval has emerged as a strong alternative. By operating directly on page images, systems like ColPali and ColQwen preserve structure and reduce pipeline complexity while achieving strong benchmark performance. However, these late interaction models tie retrieval to a specific vision backbone and require storing hundreds of patch embeddings per page, creating high memory overhead and complicating large scale deployment.
We introduce VisionRAG, a multimodal retrieval system that is OCR free and model agnostic. VisionRAG indexes documents directly as images, preserving layout, tables, and spatial cues, and builds semantic vectors without committing to a specific extraction. Our three pass pyramid indexing framework creates vectors using global page summaries, section headers, visual hotspots, and fact level cues. These summaries act as lightweight retrieval surrogates. At query time, VisionRAG retrieves the most relevant pages using the pyramid index, then forwards the raw page image encoded as base64 to a multimodal LLM for final question answering. During retrieval, reciprocal rank fusion integrates signals across the pyramid to produce robust ranking.
VisionRAG stores only 17 to 27 vectors per page, matching the efficiency of patch based methods while staying flexible across multimodal encoders. On financial document benchmarks, it achieves 0.8051 accuracy at 10 on FinanceBench and 0.9629 recall at 100 on TAT DQA. These results show that OCR free, summary guided multimodal retrieval is a practical and scalable alternative to traditional text extraction pipelines. 

---
# Generating Querying Code from Text for Multi-Modal Electronic Health Record 

**Authors**: Mengliang ZHang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20904)  

**Abstract**: Electronic health records (EHR) contain extensive structured and unstructured data, including tabular information and free-text clinical notes. Querying relevant patient information often requires complex database operations, increasing the workload for clinicians. However, complex table relationships and professional terminology in EHRs limit the query accuracy. In this work, we construct a publicly available dataset, TQGen, that integrates both \textbf{T}ables and clinical \textbf{T}ext for natural language-to-query \textbf{Gen}eration. To address the challenges posed by complex medical terminology and diverse types of questions in EHRs, we propose TQGen-EHRQuery, a framework comprising a medical knowledge module and a questions template matching module. For processing medical text, we introduced the concept of a toolset, which encapsulates the text processing module as a callable tool, thereby improving processing efficiency and flexibility. We conducted extensive experiments to assess the effectiveness of our dataset and workflow, demonstrating their potential to enhance information querying in EHR systems. 

---
# E-GEO: A Testbed for Generative Engine Optimization in E-Commerce 

**Authors**: Puneet S. Bagga, Vivek F. Farias, Tamar Korkotashvili, Tianyi Peng, Yuhang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20867)  

**Abstract**: With the rise of large language models (LLMs), generative engines are becoming powerful alternatives to traditional search, reshaping retrieval tasks. In e-commerce, for instance, conversational shopping agents now guide consumers to relevant products. This shift has created the need for generative engine optimization (GEO)--improving content visibility and relevance for generative engines. Yet despite its growing importance, current GEO practices are ad hoc, and their impacts remain poorly understood, especially in e-commerce. We address this gap by introducing E-GEO, the first benchmark built specifically for e-commerce GEO. E-GEO contains over 7,000 realistic, multi-sentence consumer product queries paired with relevant listings, capturing rich intent, constraints, preferences, and shopping contexts that existing datasets largely miss. Using this benchmark, we conduct the first large-scale empirical study of e-commerce GEO, evaluating 15 common rewriting heuristics and comparing their empirical performance. To move beyond heuristics, we further formulate GEO as a tractable optimization problem and develop a lightweight iterative prompt-optimization algorithm that can significantly outperform these baselines. Surprisingly, the optimized prompts reveal a stable, domain-agnostic pattern--suggesting the existence of a "universally effective" GEO strategy. Our data and code are publicly available at this https URL. 

---
# ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning 

**Authors**: Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2511.21005)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates significant potential in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing RLVR methods are often constrained by issues such as coarse-grained rewards, reward noise, and inefficient exploration, which lead to unstable training and entropy collapse. To address this challenge, we propose the Intrinsic Confidence-Driven Group Relative Preference Optimization method (ICPO). The intuition behind it lies in the fact that the probabilities of an LLM generating different responses can inherently and directly reflect its self-assessment of the reasoning process. Inspired by the idea of preference modeling, ICPO calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt, and integrates this score with verifiable rewards to guide the exploration process. We have discovered that the preference advantage score not only alleviates the issues of coarse-grained rewards and reward noise but also effectively curbs overconfident errors, enhances the relative superiority of undervalued high-quality responses, and prevents the model from overfitting to specific strategies, thereby facilitating more thorough exploration. Comprehensive experiments across four general-domain benchmarks and three mathematical benchmarks demonstrate that ICPO steadily boosts reasoning compared to GRPO. 

---
# Semantics Meet Signals: Dual Codebook Representationl Learning for Generative Recommendation 

**Authors**: Zheng Hui, Xiaokai Wei, Reza Shirkavand, Chen Wang, Weizhi Zhang, Alejandro Peláez, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.20673)  

**Abstract**: Generative recommendation has recently emerged as a powerful paradigm that unifies retrieval and generation, representing items as discrete semantic tokens and enabling flexible sequence modeling with autoregressive models. Despite its success, existing approaches rely on a single, uniform codebook to encode all items, overlooking the inherent imbalance between popular items rich in collaborative signals and long-tail items that depend on semantic understanding. We argue that this uniform treatment limits representational efficiency and hinders generalization. To address this, we introduce FlexCode, a popularity-aware framework that adaptively allocates a fixed token budget between a collaborative filtering (CF) codebook and a semantic codebook. A lightweight MoE dynamically balances CF-specific precision and semantic generalization, while an alignment and smoothness objective maintains coherence across the popularity spectrum. We perform experiments on both public and industrial-scale datasets, showing that FlexCode consistently outperform strong baselines. FlexCode provides a new mechanism for token representation in generative recommenders, achieving stronger accuracy and tail robustness, and offering a new perspective on balancing memorization and generalization in token-based recommendation models. 

---
