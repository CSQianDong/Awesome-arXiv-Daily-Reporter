# eSapiens: A Real-World NLP Framework for Multimodal Document Understanding and Enterprise Knowledge Processing 

**Authors**: Isaac Shi, Zeyuan Li, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16768)  

**Abstract**: We introduce eSapiens, a unified question-answering system designed for enterprise settings, which bridges structured databases and unstructured textual corpora via a dual-module architecture. The system combines a Text-to-SQL planner with a hybrid Retrieval-Augmented Generation (RAG) pipeline, enabling natural language access to both relational data and free-form documents. To enhance answer faithfulness, the RAG module integrates dense and sparse retrieval, commercial reranking, and a citation verification loop that ensures grounding consistency. We evaluate eSapiens on the RAGTruth benchmark across five leading large language models (LLMs), analyzing performance across key dimensions such as completeness, hallucination, and context utilization. Results demonstrate that eSapiens outperforms a FAISS baseline in contextual relevance and generation quality, with optional strict-grounding controls for high-stakes scenarios. This work provides a deployable framework for robust, citation-aware question answering in real-world enterprise applications. 

---
# RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering 

**Authors**: Ines Besrour, Jingbo He, Tobias Schreieder, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.16988)  

**Abstract**: We present RAGentA, a multi-agent retrieval-augmented generation (RAG) framework for attributed question answering (QA). With the goal of trustworthy answer generation, RAGentA focuses on optimizing answer correctness, defined by coverage and relevance to the question and faithfulness, which measures the extent to which answers are grounded in retrieved documents. RAGentA uses a multi-agent architecture that iteratively filters retrieved documents, generates attributed answers with in-line citations, and verifies completeness through dynamic refinement. Central to the framework is a hybrid retrieval strategy that combines sparse and dense methods, improving Recall@20 by 12.5% compared to the best single retrieval model, resulting in more correct and well-supported answers. Evaluated on a synthetic QA dataset derived from the FineWeb index, RAGentA outperforms standard RAG baselines, achieving gains of 1.09% in correctness and 10.72% in faithfulness. These results demonstrate the effectiveness of the multi-agent architecture and hybrid retrieval in advancing trustworthy QA. 

---
# cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree 

**Authors**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15655)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become essential for large-scale code generation, grounding predictions in external code corpora to improve actuality. However, a critical yet underexplored aspect of RAG pipelines is chunking -- the process of dividing documents into retrievable units. Existing line-based chunking heuristics often break semantic structures, splitting functions or merging unrelated code, which can degrade generation quality. We propose chunking via Abstract Syntax Trees (\ourwork), a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits. This approach generates self-contained, semantically coherent units across programming languages and tasks, improving performance on diverse code generation tasks, e.g., boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation. Our work highlights the importance of structure-aware chunking for scaling retrieval-enhanced code intelligence. 

---
# Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding 

**Authors**: Vishesh Tripathi, Tanmay Odapally, Indraneel Das, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16035)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval and question answering, but traditional text-based chunking methods struggle with complex document structures, multi-page tables, embedded figures, and contextual dependencies across page boundaries. We present a novel multimodal document chunking approach that leverages Large Multimodal Models (LMMs) to process PDF documents in batches while maintaining semantic coherence and structural integrity. Our method processes documents in configurable page batches with cross-batch context preservation, enabling accurate handling of tables spanning multiple pages, embedded visual elements, and procedural content. We evaluate our approach on a curated dataset of PDF documents with manually crafted queries, demonstrating improvements in chunk quality and downstream RAG performance. Our vision-guided approach achieves better accuracy compared to traditional vanilla RAG systems, with qualitative analysis showing superior preservation of document structure and semantic coherence. 

---
# SGIC: A Self-Guided Iterative Calibration Framework for RAG 

**Authors**: Guanhua Chen, Yutong Yao, Lidia S. Chao, Xuebo Liu, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16172)  

**Abstract**: Recent research in retrieval-augmented generation (RAG) has concentrated on retrieving useful information from candidate documents. However, numerous methodologies frequently neglect the calibration capabilities of large language models (LLMs), which capitalize on their robust in-context reasoning prowess. This work illustrates that providing LLMs with specific cues substantially improves their calibration efficacy, especially in multi-round calibrations. We present a new SGIC: Self-Guided Iterative Calibration Framework that employs uncertainty scores as a tool. Initially, this framework calculates uncertainty scores to determine both the relevance of each document to the query and the confidence level in the responses produced by the LLMs. Subsequently, it reevaluates these scores iteratively, amalgamating them with prior responses to refine calibration. Furthermore, we introduce an innovative approach for constructing an iterative self-calibration training set, which optimizes LLMs to efficiently harness uncertainty scores for capturing critical information and enhancing response accuracy. Our proposed framework significantly improves performance on both closed-source and open-weight LLMs. 

---
# Enhancing Document-Level Question Answering via Multi-Hop Retrieval-Augmented Generation with LLaMA 3 

**Authors**: Xinyue Huang, Ziqi Lin, Fang Sun, Wenchao Zhang, Kejian Tong, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16037)  

**Abstract**: This paper presents a novel Retrieval-Augmented Generation (RAG) framework tailored for complex question answering tasks, addressing challenges in multi-hop reasoning and contextual understanding across lengthy documents. Built upon LLaMA 3, the framework integrates a dense retrieval module with advanced context fusion and multi-hop reasoning mechanisms, enabling more accurate and coherent response generation. A joint optimization strategy combining retrieval likelihood and generation cross-entropy improves the model's robustness and adaptability. Experimental results show that the proposed system outperforms existing retrieval-augmented and generative baselines, confirming its effectiveness in delivering precise, contextually grounded answers. 

---
# From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents 

**Authors**: Mohammad Amaan Sayeed, Mohammed Talha Alam, Raza Imam, Shahab Saquib Sohail, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2506.15911)  

**Abstract**: Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering. 

---
