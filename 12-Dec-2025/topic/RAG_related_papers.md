# AgriRegion: Region-Aware Retrieval for High-Fidelity Agricultural Advice 

**Authors**: Mesafint Fanuel, Mahmoud Nabil Mahmoud, Crystal Cook Marshal, Vishal Lakhotia, Biswanath Dari, Kaushik Roy, Shaohu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10114)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in democratizing access to information. However, in the domain of agriculture, general-purpose models frequently suffer from contextual hallucination, which provides non-factual advice or answers are scientifically sound in one region but disastrous in another due to variations in soil, climate, and local regulations. We introduce AgriRegion, a Retrieval-Augmented Generation (RAG) framework designed specifically for high-fidelity, region-aware agricultural advisory. Unlike standard RAG approaches that rely solely on semantic similarity, AgriRegion incorporates a geospatial metadata injection layer and a region-prioritized re-ranking mechanism. By restricting the knowledge base to verified local agricultural extension services and enforcing geo-spatial constraints during retrieval, AgriRegion ensures that the advice regarding planting schedules, pest control, and fertilization is locally accurate. We create a novel benchmark dataset, AgriRegion-Eval, which comprises 160 domain-specific questions across 12 agricultural subfields. Experiments demonstrate that AgriRegion reduces hallucinations by 10-20% compared to state-of-the-art LLMs systems and significantly improves trust scores according to a comprehensive evaluation. 

---
# Suzume-chan: Your Personal Navigator as an Embodied Information Hub 

**Authors**: Maya Grace Torii, Takahito Murakami, Shuka Koseki, Yoichi Ochiai  

**Link**: [PDF](https://arxiv.org/pdf/2512.09932)  

**Abstract**: Access to expert knowledge often requires real-time human communication. Digital tools improve access to information but rarely create the sense of connection needed for deep understanding. This study addresses this issue using Social Presence Theory, which explains how a feeling of "being together" enhances communication. An "Embodied Information Hub" is proposed as a new way to share knowledge through physical and conversational interaction. The prototype, Suzume-chan, is a small, soft AI agent running locally with a language model and retrieval-augmented generation (RAG). It learns from spoken explanations and responds through dialogue, reducing psychological distance and making knowledge sharing warmer and more human-centered. 

---
# Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers 

**Authors**: Youmin Ko, Sungjong Seo, Hyunjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10422)  

**Abstract**: Since large language models (LLMs) have a tendency to generate factually inaccurate output, retrieval-augmented generation (RAG) has gained significant attention as a key means to mitigate this downside of harnessing only LLMs. However, existing RAG methods for simple and multi-hop question answering (QA) are still prone to incorrect retrievals and hallucinations. To address these limitations, we propose CoopRAG, a novel RAG framework for the question answering task in which a retriever and an LLM work cooperatively with each other by exchanging informative knowledge, and the earlier and later layers of the retriever model work cooperatively with each other to accurately rank the retrieved documents relevant to a given query. In this framework, we (i) unroll a question into sub-questions and a reasoning chain in which uncertain positions are masked, (ii) retrieve the documents relevant to the question augmented with the sub-questions and the reasoning chain, (iii) rerank the documents by contrasting layers of the retriever, and (iv) reconstruct the reasoning chain by filling the masked positions via the LLM. Our experiments demonstrate that CoopRAG consistently outperforms state-of-the-art QA methods on three multi-hop QA datasets as well as a simple QA dataset in terms of both the retrieval and QA performances. Our code is available.\footnote{this https URL} 

---
