# Cost-Efficient Cross-Lingual Retrieval-Augmented Generation for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory 

**Authors**: Md. Asif Hossain, Nabil Subhan, Mantasha Rahman Mahi, Jannatul Ferdous Nabila  

**Link**: [PDF](https://arxiv.org/pdf/2601.02065)  

**Abstract**: Access to reliable agricultural advisory remains limited in many developing regions due to a persistent language barrier: authoritative agricultural manuals are predominantly written in English, while farmers primarily communicate in low-resource local languages such as Bengali. Although recent advances in Large Language Models (LLMs) enable natural language interaction, direct generation in low-resource languages often exhibits poor fluency and factual inconsistency, while cloud-based solutions remain cost-prohibitive. This paper presents a cost-efficient, cross-lingual Retrieval-Augmented Generation (RAG) framework for Bengali agricultural advisory that emphasizes factual grounding and practical deployability. The proposed system adopts a translation-centric architecture in which Bengali user queries are translated into English, enriched through domain-specific keyword injection to align colloquial farmer terminology with scientific nomenclature, and answered via dense vector retrieval over a curated corpus of English agricultural manuals (FAO, IRRI). The generated English response is subsequently translated back into Bengali to ensure accessibility. The system is implemented entirely using open-source models and operates on consumer-grade hardware without reliance on paid APIs. Experimental evaluation demonstrates reliable source-grounded responses, robust rejection of out-of-domain queries, and an average end-to-end latency below 20 seconds. The results indicate that cross-lingual retrieval combined with controlled translation offers a practical and scalable solution for agricultural knowledge access in low-resource language settings 

---
# Tackling the Inherent Difficulty of Noise Filtering in RAG 

**Authors**: Jingyu Liu, Jiaen Lin, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01896)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a widely adopted approach to enhance Large Language Models (LLMs) by incorporating external knowledge and reducing hallucinations. However, noisy or irrelevant documents are often introduced during RAG, potentially degrading performance and even causing hallucinated outputs. While various methods have been proposed to filter out such noise, we argue that identifying irrelevant information from retrieved content is inherently difficult and limited number of transformer layers can hardly solve this. Consequently, retrievers fail to filter out irrelevant documents entirely. Therefore, LLMs must be robust against such noise, but we demonstrate that standard fine-tuning approaches are often ineffective in enabling the model to selectively utilize relevant information while ignoring irrelevant content due to the structural constraints of attention patterns. To address this, we propose a novel fine-tuning method designed to enhance the model's ability to distinguish between relevant and irrelevant information within retrieved documents. Extensive experiments across multiple benchmarks show that our approach significantly improves the robustness and performance of LLMs. 

---
# Reasoning Over Recall: Evaluating the Efficacy of Generalist Architectures vs. Specialized Fine-Tunes in RAG-Based Mental Health Dialogue Systems 

**Authors**: Md Abdullah Al Kafi, Raka Moni, Sumit Kumar Banshal  

**Link**: [PDF](https://arxiv.org/pdf/2601.01341)  

**Abstract**: The deployment of Large Language Models (LLMs) in mental health counseling faces the dual challenges of hallucinations and lack of empathy. While the former may be mitigated by RAG (retrieval-augmented generation) by anchoring answers in trusted clinical sources, there remains an open question as to whether the most effective model under this paradigm would be one that is fine-tuned on mental health data, or a more general and powerful model that succeeds purely on the basis of reasoning. In this paper, we perform a direct comparison by running four open-source models through the same RAG pipeline using ChromaDB: two generalist reasoners (Qwen2.5-3B and Phi-3-Mini) and two domain-specific fine-tunes (MentalHealthBot-7B and TherapyBot-7B). We use an LLM-as-a-Judge framework to automate evaluation over 50 turns. We find a clear trend: the generalist models outperform the domain-specific ones in empathy (3.72 vs. 3.26, $p < 0.001$) in spite of being much smaller (3B vs. 7B), and all models perform well in terms of safety, but the generalist models show better contextual understanding and are less prone to overfitting as we observe in the domain-specific models. Overall, our results indicate that for RAG-based therapy systems, strong reasoning is more important than training on mental health-specific vocabulary; i.e. a well-reasoned general model would provide more empathetic and balanced support than a larger narrowly fine-tuned model, so long as the answer is already grounded in clinical evidence. 

---
# SRAS: A Lightweight Reinforcement Learning-based Document Selector for Edge-Native RAG Pipelines 

**Authors**: Rajiv Chaitanya Muttur  

**Link**: [PDF](https://arxiv.org/pdf/2601.01785)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often rely on fixed top-k document selection mechanisms that ignore downstream generation quality and impose computational overheads. We propose SRAS (Sparse Reward-Aware Selector), a lightweight document selector trained via reinforcement learning (RL) for edge-native RAG deployment. Unlike prior RL-based retrievers that assume large memory and latency budgets, SRAS learns a compact (~0.76MB) policy using Proximal Policy Optimization (PPO), guided by a hybrid reward signal combining Relaxed F1 and BERTScore. Our method operates under tight token and compute constraints, maintaining <1s latency on CPU. SRAS outperforms supervised and random selectors on a synthetic QA benchmark, and generalizes to real-world data, achieving BERTScore F1 of 0.8546 on SQuAD v2 without domain-specific tuning. This work is the first to demonstrate that RL-based document selection can be made ultra-lightweight, latency-aware, and effective for on-device RAG pipelines. 

---
# Enhancing Retrieval-Augmented Generation with Topic-Enriched Embeddings: A Hybrid Approach Integrating Traditional NLP Techniques 

**Authors**: Rodrigo Kataishi  

**Link**: [PDF](https://arxiv.org/pdf/2601.00891)  

**Abstract**: Retrieval-augmented generation (RAG) systems rely on accurate document retrieval to ground large language models (LLMs) in external knowledge, yet retrieval quality often degrades in corpora where topics overlap and thematic variation is high. This work proposes topic-enriched embeddings that integrate term-based signals and topic structure with contextual sentence embeddings. The approach combines TF-IDF with topic modeling and dimensionality reduction, using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) to encode latent topical organization, and fuses these representations with a compact contextual encoder (all-MiniLM). By jointly capturing term-level and topic-level semantics, topic-enriched embeddings improve semantic clustering, increase retrieval precision, and reduce computational burden relative to purely contextual baselines. Experiments on a legal-text corpus show consistent gains in clustering coherence and retrieval metrics, suggesting that topic-enriched embeddings can serve as a practical component for more reliable knowledge-intensive RAG pipelines. 

---
# Yuan3.0 Flash: An Open Multimodal Large Language Model for Enterprise Applications 

**Authors**: YuanLab.ai, Shawn Wu, Sean Wang, Louie Li, Darcy Chen, Allen Wang, Jiangang Luo, Xudong Zhao, Joseph Shen, Gawain Ma, Jasper Jia, Marcus Mao, Claire Wang, Hunter He, Carol Wang, Zera Zhang, Jason Wang, Chonly Shen, Leo Zhang, Logan Chen, Qasim Meng, James Gong, Danied Zhao, Penn Zheng, Owen Zhu, Tong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01718)  

**Abstract**: We introduce Yuan3.0 Flash, an open-source Mixture-of-Experts (MoE) MultiModal Large Language Model featuring 3.7B activated parameters and 40B total parameters, specifically designed to enhance performance on enterprise-oriented tasks while maintaining competitive capabilities on general-purpose tasks. To address the overthinking phenomenon commonly observed in Large Reasoning Models (LRMs), we propose Reflection-aware Adaptive Policy Optimization (RAPO), a novel RL training algorithm that effectively regulates overthinking behaviors. In enterprise-oriented tasks such as retrieval-augmented generation (RAG), complex table understanding, and summarization, Yuan3.0 Flash consistently achieves superior performance. Moreover, it also demonstrates strong reasoning capabilities in domains such as mathematics, science, etc., attaining accuracy comparable to frontier model while requiring only approximately 1/4 to 1/2 of the average tokens. Yuan3.0 Flash has been fully open-sourced to facilitate further research and real-world deployment: this https URL. 

---
# Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation 

**Authors**: Udiptaman Das, Krishnasai B. Atmakuri, Duy Ho, Chi Lee, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.01844)  

**Abstract**: Large language models (LLMs) offer new opportunities for constructing knowledge graphs (KGs) from unstructured clinical narratives. However, existing approaches often rely on structured inputs and lack robust validation of factual accuracy and semantic consistency, limitations that are especially problematic in oncology. We introduce an end-to-end framework for clinical KG construction and evaluation directly from free text using multi-agent prompting and a schema-constrained Retrieval-Augmented Generation (KG-RAG) strategy. Our pipeline integrates (1) prompt-driven entity, attribute, and relation extraction; (2) entropy-based uncertainty scoring; (3) ontology-aligned RDF/OWL schema generation; and (4) multi-LLM consensus validation for hallucination detection and semantic refinement. Beyond static graph construction, the framework supports continuous refinement and self-supervised evaluation, enabling iterative improvement of graph quality. Applied to two oncology cohorts (PDAC and BRCA), our method produces interpretable, SPARQL-compatible, and clinically grounded knowledge graphs without relying on gold-standard annotations. Experimental results demonstrate consistent gains in precision, relevance, and ontology compliance over baseline methods. 

---
# FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation 

**Authors**: Gen Li, Peiyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01513)  

**Abstract**: Vision-Language Models (VLMs) excel at visual reasoning but still struggle with integrating external knowledge. Retrieval-Augmented Generation (RAG) is a promising solution, but current methods remain inefficient and often fail to maintain high answer quality. To address these challenges, we propose VideoSpeculateRAG, an efficient VLM-based RAG framework built on two key ideas. First, we introduce a speculative decoding pipeline: a lightweight draft model quickly generates multiple answer candidates, which are then verified and refined by a more accurate heavyweight model, substantially reducing inference latency without sacrificing correctness. Second, we identify a major source of error - incorrect entity recognition in retrieved knowledge - and mitigate it with a simple yet effective similarity-based filtering strategy that improves entity alignment and boosts overall answer accuracy. Experiments demonstrate that VideoSpeculateRAG achieves comparable or higher accuracy than standard RAG approaches while accelerating inference by approximately 2x. Our framework highlights the potential of combining speculative decoding with retrieval-augmented reasoning to enhance efficiency and reliability in complex, knowledge-intensive multimodal tasks. 

---
