# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

---
