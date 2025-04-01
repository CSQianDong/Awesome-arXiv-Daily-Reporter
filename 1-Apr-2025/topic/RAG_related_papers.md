# GRASP: Municipal Budget AI Chatbots for Enhancing Civic Engagement 

**Authors**: Jerry Xu, Justin Wang, Joley Leung, Jasmine Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23299)  

**Abstract**: There are a growing number of AI applications, but none tailored specifically to help residents answer their questions about municipal budget, a topic most are interested in but few have a solid comprehension of. In this research paper, we propose GRASP, a custom AI chatbot framework which stands for Generation with Retrieval and Action System for Prompts. GRASP provides more truthful and grounded responses to user budget queries than traditional information retrieval systems like general Large Language Models (LLMs) or web searches. These improvements come from the novel combination of a Retrieval-Augmented Generation (RAG) framework ("Generation with Retrieval") and an agentic workflow ("Action System"), as well as prompt engineering techniques, the incorporation of municipal budget domain knowledge, and collaboration with local town officials to ensure response truthfulness. During testing, we found that our GRASP chatbot provided precise and accurate responses for local municipal budget queries 78% of the time, while GPT-4o and Gemini were only accurate 60% and 35% of the time, respectively. GRASP chatbots greatly reduce the time and effort needed for the general public to get an intuitive and correct understanding of their town's budget, thus fostering greater communal discourse, improving government transparency, and allowing citizens to make more informed decisions. 

---
# A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG 

**Authors**: Arshia Kermani, Veronica Perez-Rosas, Vangelis Metsis  

**Link**: [PDF](https://arxiv.org/pdf/2503.24307)  

**Abstract**: This study presents a systematic comparison of three approaches for the analysis of mental health text using large language models (LLMs): prompt engineering, retrieval augmented generation (RAG), and fine-tuning. Using LLaMA 3, we evaluate these approaches on emotion classification and mental health condition detection tasks across two datasets. Fine-tuning achieves the highest accuracy (91% for emotion classification, 80% for mental health conditions) but requires substantial computational resources and large training sets, while prompt engineering and RAG offer more flexible deployment with moderate performance (40-68% accuracy). Our findings provide practical insights for implementing LLM-based solutions in mental health applications, highlighting the trade-offs between accuracy, computational requirements, and deployment flexibility. 

---
# Better wit than wealth: Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancement 

**Authors**: Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23895)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources and incorporating them into the context. While it improves reliability by providing factual texts, it significantly increases inference costs as context length grows and introduces challenging issue of RAG hallucination, primarily caused by the lack of corresponding parametric knowledge in LLMs. An efficient solution is to enhance the knowledge of LLMs at test-time. Parametric RAG (PRAG) addresses this by embedding document into LLMs parameters to perform test-time knowledge enhancement, effectively reducing inference costs through offline training. However, its high training and storage costs, along with limited generalization ability, significantly restrict its practical adoption. To address these challenges, we propose Dynamic Parametric RAG (DyPRAG), a novel framework that leverages a lightweight parameter translator model to efficiently convert documents into parametric knowledge. DyPRAG not only reduces inference, training, and storage costs but also dynamically generates parametric knowledge, seamlessly enhancing the knowledge of LLMs and resolving knowledge conflicts in a plug-and-play manner at test-time. Extensive experiments on multiple datasets demonstrate the effectiveness and generalization capabilities of DyPRAG, offering a powerful and practical RAG paradigm which enables superior knowledge fusion and mitigates RAG hallucination in real-world applications. Our code is available at this https URL. 

---
# DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation 

**Authors**: Hsin-Ling Hsu, Jengnan Tzeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.23013)  

**Abstract**: Hybrid retrieval techniques in Retrieval-Augmented Generation (RAG) systems enhance information retrieval by combining dense and sparse (e.g., BM25-based) retrieval methods. However, existing approaches struggle with adaptability, as fixed weighting schemes fail to adjust to different queries. To address this, we propose DAT (Dynamic Alpha Tuning), a novel hybrid retrieval framework that dynamically balances dense retrieval and BM25 for each query. DAT leverages a large language model (LLM) to evaluate the effectiveness of the top-1 results from both retrieval methods, assigning an effectiveness score to each. It then calibrates the optimal weighting factor through effectiveness score normalization, ensuring a more adaptive and query-aware weighting between the two approaches. Empirical results show that DAT consistently significantly outperforms fixed-weighting hybrid retrieval methods across various evaluation metrics. Even on smaller models, DAT delivers strong performance, highlighting its efficiency and adaptability. 

---
# Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation 

**Authors**: Dun Yuan, Hao Zhou, Di Wu, Xue Liu, Hao Chen, Yan Xin, Jianzhong, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24245)  

**Abstract**: Large language models (LLMs) have made significant progress in general-purpose natural language processing tasks. However, LLMs are still facing challenges when applied to domain-specific areas like telecommunications, which demands specialized expertise and adaptability to evolving standards. This paper presents a novel framework that combines knowledge graph (KG) and retrieval-augmented generation (RAG) techniques to enhance LLM performance in the telecom domain. The framework leverages a KG to capture structured, domain-specific information about network protocols, standards, and other telecom-related entities, comprehensively representing their relationships. By integrating KG with RAG, LLMs can dynamically access and utilize the most relevant and up-to-date knowledge during response generation. This hybrid approach bridges the gap between structured knowledge representation and the generative capabilities of LLMs, significantly enhancing accuracy, adaptability, and domain-specific comprehension. Our results demonstrate the effectiveness of the KG-RAG framework in addressing complex technical queries with precision. The proposed KG-RAG model attained an accuracy of 88% for question answering tasks on a frequently used telecom-specific dataset, compared to 82% for the RAG-only and 48% for the LLM-only approaches. 

---
# SCORE: Story Coherence and Retrieval Enhancement for AI Narratives 

**Authors**: Qiang Yi, Yangfan He, Jianhui Wang, Xinyuan Song, Shiyao Qian, Miao Zhang, Li Sun, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23512)  

**Abstract**: Large Language Models (LLMs) excel at generating creative narratives but struggle with long-term coherence and emotional consistency in complex stories. To address this, we propose SCORE (Story Coherence and Retrieval Enhancement), a framework integrating three components: 1) Dynamic State Tracking (monitoring objects/characters via symbolic logic), 2) Context-Aware Summarization (hierarchical episode summaries for temporal progression), and 3) Hybrid Retrieval (combining TF-IDF keyword relevance with cosine similarity-based semantic embeddings). The system employs a temporally-aligned Retrieval-Augmented Generation (RAG) pipeline to validate contextual consistency. Evaluations show SCORE achieves 23.6% higher coherence (NCI-2.0 benchmark), 89.7% emotional consistency (EASM metric), and 41.8% fewer hallucinations versus baseline GPT models. Its modular design supports incremental knowledge graph construction for persistent story memory and multi-LLM backend compatibility, offering an explainable solution for industrial-scale narrative systems requiring long-term consistency. 

---
# CrossFormer: Cross-Segment Semantic Fusion for Document Segmentation 

**Authors**: Tongke Ni, Yang Fan, Junru Zhou, Xiangping Wu, Qingcai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23671)  

**Abstract**: Text semantic segmentation involves partitioning a document into multiple paragraphs with continuous semantics based on the subject matter, contextual information, and document structure. Traditional approaches have typically relied on preprocessing documents into segments to address input length constraints, resulting in the loss of critical semantic information across segments. To address this, we present CrossFormer, a transformer-based model featuring a novel cross-segment fusion module that dynamically models latent semantic dependencies across document segments, substantially elevating segmentation accuracy. Additionally, CrossFormer can replace rule-based chunk methods within the Retrieval-Augmented Generation (RAG) system, producing more semantically coherent chunks that enhance its efficacy. Comprehensive evaluations confirm CrossFormer's state-of-the-art performance on public text semantic segmentation datasets, alongside considerable gains on RAG benchmarks. 

---
# A Retrieval-Augmented Knowledge Mining Method with Deep Thinking LLMs for Biomedical Research and Clinical Support 

**Authors**: Yichun Feng, Jiawei Wang, Ruikun He, Lu Zhou, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.23029)  

**Abstract**: Knowledge graphs and large language models (LLMs) are key tools for biomedical knowledge integration and reasoning, facilitating structured organization of scientific articles and discovery of complex semantic relationships. However, current methods face challenges: knowledge graph construction is limited by complex terminology, data heterogeneity, and rapid knowledge evolution, while LLMs show limitations in retrieval and reasoning, making it difficult to uncover cross-document associations and reasoning pathways. To address these issues, we propose a pipeline that uses LLMs to construct a biomedical knowledge graph (BioStrataKG) from large-scale articles and builds a cross-document question-answering dataset (BioCDQA) to evaluate latent knowledge retrieval and multi-hop reasoning. We then introduce Integrated and Progressive Retrieval-Augmented Reasoning (IP-RAR) to enhance retrieval accuracy and knowledge reasoning. IP-RAR maximizes information recall through Integrated Reasoning-based Retrieval and refines knowledge via Progressive Reasoning-based Generation, using self-reflection to achieve deep thinking and precise contextual understanding. Experiments show that IP-RAR improves document retrieval F1 score by 20\% and answer generation accuracy by 25\% over existing methods. This framework helps doctors efficiently integrate treatment evidence for personalized medication plans and enables researchers to analyze advancements and research gaps, accelerating scientific discovery and decision-making. 

---
