# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

---
# EnronQA: Towards Personalized RAG over Private Documents 

**Authors**: Michael J. Ryan, Danmei Xu, Chris Nivera, Daniel Campos  

**Link**: [PDF](https://arxiv.org/pdf/2505.00263)  

**Abstract**: Retrieval Augmented Generation (RAG) has become one of the most popular methods for bringing knowledge-intensive context to large language models (LLM) because of its ability to bring local context at inference time without the cost or data leakage risks associated with fine-tuning. A clear separation of private information from the LLM training has made RAG the basis for many enterprise LLM workloads as it allows the company to augment LLM's understanding using customers' private documents. Despite its popularity for private documents in enterprise deployments, current RAG benchmarks for validating and optimizing RAG pipelines draw their corpora from public data such as Wikipedia or generic web pages and offer little to no personal context. Seeking to empower more personal and private RAG we release the EnronQA benchmark, a dataset of 103,638 emails with 528,304 question-answer pairs across 150 different user inboxes. EnronQA enables better benchmarking of RAG pipelines over private data and allows for experimentation on the introduction of personalized retrieval settings over realistic data. Finally, we use EnronQA to explore the tradeoff in memorization and retrieval when reasoning over private documents. 

---
# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00039)  

**Abstract**: This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to significantly advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support. 

---
