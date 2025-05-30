# Privacy-Enhancing Paradigms within Federated Multi-Agent Systems 

**Authors**: Zitong Shi, Guancheng Wan, Wenke Huang, Guibin Zhang, Jiawei Shao, Mang Ye, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08175)  

**Abstract**: LLM-based Multi-Agent Systems (MAS) have proven highly effective in solving complex problems by integrating multiple agents, each performing different roles. However, in sensitive domains, they face emerging privacy protection challenges. In this paper, we introduce the concept of Federated MAS, highlighting the fundamental differences between Federated MAS and traditional FL. We then identify key challenges in developing Federated MAS, including: 1) heterogeneous privacy protocols among agents, 2) structural differences in multi-party conversations, and 3) dynamic conversational network structures. To address these challenges, we propose Embedded Privacy-Enhancing Agents (EPEAgent), an innovative solution that integrates seamlessly into the Retrieval-Augmented Generation (RAG) phase and the context retrieval stage. This solution minimizes data flows, ensuring that only task-relevant, agent-specific information is shared. Additionally, we design and generate a comprehensive dataset to evaluate the proposed paradigm. Extensive experiments demonstrate that EPEAgent effectively enhances privacy protection while maintaining strong system performance. The code will be availiable at this https URL 

---
# LLM-based Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification 

**Authors**: Siyuan Wang, James R. Foulds, Md Osman Gani, Shimei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07937)  

**Abstract**: In this paper, we introduce CIBER (Claim Investigation Based on Evidence Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework designed to identify corroborating and refuting documents as evidence for scientific claim verification. CIBER addresses the inherent uncertainty in Large Language Models (LLMs) by evaluating response consistency across diverse interrogation probes. By focusing on the behavioral analysis of LLMs without requiring access to their internal information, CIBER is applicable to both white-box and black-box models. Furthermore, CIBER operates in an unsupervised manner, enabling easy generalization across various scientific domains. Comprehensive evaluations conducted using LLMs with varying levels of linguistic proficiency reveal CIBER's superior performance compared to conventional RAG approaches. These findings not only highlight the effectiveness of CIBER but also provide valuable insights for future advancements in LLM-based scientific claim verification. 

---
# DeepRAG: Building a Custom Hindi Embedding Model for Retrieval Augmented Generation from Scratch 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2503.08213)  

**Abstract**: In this paper, I present our work on DeepRAG, a specialized embedding model we built specifically for Hindi language in RAG systems. While LLMs have gotten really good at generating text, their performance in retrieval tasks still depends heavily on having quality embeddings - something that's been lacking for Hindi despite being one of the world's most spoken languages. We tackled this by creating embeddings from the ground up rather than just fine-tuning existing models. Our process involved collecting diverse Hindi texts (over 2.7M samples), training a custom SentencePiece tokenizer that actually understands Hindi morphology, designing transformer architecture with Hindi-specific attention mechanisms, and optimizing with contrastive learning. Results were honestly better than I expected - we saw a 23% improvement in retrieval precision compared to the multilingual models everyone's been using. The paper details our methodology, which I think could help others working with low-resource languages where the one-size-fits-all multilingual models fall short. We've also integrated our embeddings with LangChain to build complete Hindi RAG systems, which might be useful for practitioners. While there's still tons more to explore, I believe this work addresses a critical gap for Hindi NLP and demonstrates why language-specific approaches matter. 

---
# TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster 

**Authors**: Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y. Zhang, Kashif Rasul, Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, Dongjin Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.07649)  

**Abstract**: Recently, Large Language Models (LLMs) and Foundation Models (FMs) have become prevalent for time series forecasting tasks. However, fine-tuning large language models (LLMs) for forecasting enables the adaptation to specific domains but may not generalize well across diverse, unseen datasets. Meanwhile, existing time series foundation models (TSFMs) lack inherent mechanisms for domain adaptation and suffer from limited interpretability, making them suboptimal for zero-shot forecasting. To this end, we present TS-RAG, a retrieval-augmented generation based time series forecasting framework that enhances the generalization capability and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant time series segments from a dedicated knowledge database, incorporating contextual patterns for the given time series query. Next, we develop a learnable Mixture-of-Experts (MoE)-based augmentation module, which dynamically fuses retrieved time series patterns with the TSFM's representation of the input query, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming TSFMs by up to 6.51% across diverse domains and showcasing desired interpretability. 

---
# OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning 

**Authors**: Jiawei Zhou, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08398)  

**Abstract**: In this paper, we analyze and empirically show that the learned relevance for conventional information retrieval (IR) scenarios may be inconsistent in retrieval-augmented generation (RAG) scenarios. To bridge this gap, we introduce OpenRAG, a RAG framework that is optimized end-to-end by tuning the retriever to capture in-context relevance, enabling adaptation to the diverse and evolving needs. Extensive experiments across a wide range of tasks demonstrate that OpenRAG, by tuning a retriever end-to-end, leads to a consistent improvement of 4.0% over the original retriever, consistently outperforming existing state-of-the-art retrievers by 2.1%. Additionally, our results indicate that for some tasks, an end-to-end tuned 0.2B retriever can achieve improvements that surpass those of RAG-oriented or instruction-tuned 8B large language models (LLMs), highlighting the cost-effectiveness of our approach in enhancing RAG systems. 

---
