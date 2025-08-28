# SoK: Large Language Model Copyright Auditing via Fingerprinting 

**Authors**: Shuo Shao, Yiming Li, Yu He, Hongwei Yao, Wenyuan Yang, Dacheng Tao, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19843)  

**Abstract**: The broad capabilities and substantial resources required to train Large Language Models (LLMs) make them valuable intellectual property, yet they remain vulnerable to copyright infringement, such as unauthorized use and model theft. LLM fingerprinting, a non-intrusive technique that extracts and compares the distinctive features from LLMs to identify infringements, offers a promising solution to copyright auditing. However, its reliability remains uncertain due to the prevalence of diverse model modifications and the lack of standardized evaluation. In this SoK, we present the first comprehensive study of LLM fingerprinting. We introduce a unified framework and formal taxonomy that categorizes existing methods into white-box and black-box approaches, providing a structured overview of the state of the art. We further propose LeaFBench, the first systematic benchmark for evaluating LLM fingerprinting under realistic deployment scenarios. Built upon mainstream foundation models and comprising 149 distinct model instances, LeaFBench integrates 13 representative post-development techniques, spanning both parameter-altering methods (e.g., fine-tuning, quantization) and parameter-independent mechanisms (e.g., system prompts, RAG). Extensive experiments on LeaFBench reveal the strengths and weaknesses of existing methods, thereby outlining future research directions and critical open problems in this emerging field. The code is available at this https URL. 

---
# LFD: Layer Fused Decoding to Exploit External Knowledge in Retrieval-Augmented Generation 

**Authors**: Yang Sun, Lixin Zou, Dan Luo, Zhiyong Xie, Long Zhang, Liming Dong, Yunwei Zhao, Xixun Lin, Yanxiong Lu, Chenliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19614)  

**Abstract**: Retrieval-augmented generation (RAG) incorporates external knowledge into large language models (LLMs), improving their adaptability to downstream tasks and enabling information updates. Surprisingly, recent empirical evidence demonstrates that injecting noise into retrieved relevant documents paradoxically facilitates exploitation of external knowledge and improves generation quality. Although counterintuitive and challenging to apply in practice, this phenomenon enables granular control and rigorous analysis of how LLMs integrate external knowledge. Therefore, in this paper, we intervene on noise injection and establish a layer-specific functional demarcation within the LLM: shallow layers specialize in local context modeling, intermediate layers focus on integrating long-range external factual knowledge, and deeper layers primarily rely on parametric internal knowledge. Building on this insight, we propose Layer Fused Decoding (LFD), a simple decoding strategy that directly combines representations from an intermediate layer with final-layer decoding outputs to fully exploit the external factual knowledge. To identify the optimal intermediate layer, we introduce an internal knowledge score (IKS) criterion that selects the layer with the lowest IKS value in the latter half of layers. Experimental results across multiple benchmarks demonstrate that LFD helps RAG systems more effectively surface retrieved context knowledge with minimal cost. 

---
# Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning 

**Authors**: Junnan Dong, Siyu An, Yifei Yu, Qian-Wen Zhang, Linhao Luo, Xiao Huang, Yunsheng Wu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.19855)  

**Abstract**: Graph retrieval-augmented generation (GraphRAG) has effectively enhanced large language models in complex reasoning by organizing fragmented knowledge into explicitly structured graphs. Prior efforts have been made to improve either graph construction or graph retrieval in isolation, yielding suboptimal performance, especially when domain shifts occur. In this paper, we propose a vertically unified agentic paradigm, Youtu-GraphRAG, to jointly connect the entire framework as an intricate integration. Specifically, (i) a seed graph schema is introduced to bound the automatic extraction agent with targeted entity types, relations and attribute types, also continuously expanded for scalability over unseen domains; (ii) To obtain higher-level knowledge upon the schema, we develop novel dually-perceived community detection, fusing structural topology with subgraph semantics for comprehensive knowledge organization. This naturally yields a hierarchical knowledge tree that supports both top-down filtering and bottom-up reasoning with community summaries; (iii) An agentic retriever is designed to interpret the same graph schema to transform complex queries into tractable and parallel sub-queries. It iteratively performs reflection for more advanced reasoning; (iv) To alleviate the knowledge leaking problem in pre-trained LLM, we propose a tailored anonymous dataset and a novel 'Anonymity Reversion' task that deeply measures the real performance of the GraphRAG frameworks. Extensive experiments across six challenging benchmarks demonstrate the robustness of Youtu-GraphRAG, remarkably moving the Pareto frontier with up to 90.71% saving of token costs and 16.62% higher accuracy over state-of-the-art baselines. The results indicate our adaptability, allowing seamless domain transfer with minimal intervention on schema. 

---
# AI for Statutory Simplification: A Comprehensive State Legal Corpus and Labor Benchmark 

**Authors**: Emaan Hariri, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2508.19365)  

**Abstract**: One of the emerging use cases of AI in law is for code simplification: streamlining, distilling, and simplifying complex statutory or regulatory language. One U.S. state has claimed to eliminate one third of its state code using AI. Yet we lack systematic evaluations of the accuracy, reliability, and risks of such approaches. We introduce LaborBench, a question-and-answer benchmark dataset designed to evaluate AI capabilities in this domain. We leverage a unique data source to create LaborBench: a dataset updated annually by teams of lawyers at the U.S. Department of Labor, who compile differences in unemployment insurance laws across 50 states for over 101 dimensions in a six-month process, culminating in a 200-page publication of tables. Inspired by our collaboration with one U.S. state to explore using large language models (LLMs) to simplify codes in this domain, where complexity is particularly acute, we transform the DOL publication into LaborBench. This provides a unique benchmark for AI capacity to conduct, distill, and extract realistic statutory and regulatory information. To assess the performance of retrieval augmented generation (RAG) approaches, we also compile StateCodes, a novel and comprehensive state statute and regulatory corpus of 8.7 GB, enabling much more systematic research into state codes. We then benchmark the performance of information retrieval and state-of-the-art large LLMs on this data and show that while these models are helpful as preliminary research for code simplification, the overall accuracy is far below the touted promises for LLMs as end-to-end pipelines for regulatory simplification. 

---
# Context-Adaptive Synthesis and Compression for Enhanced Retrieval-Augmented Generation in Complex Domains 

**Authors**: Peiran Zhou, Junnan Zhu, Yichen Shen, Ruoxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19357)  

**Abstract**: Large Language Models (LLMs) excel in language tasks but are prone to hallucinations and outdated knowledge. Retrieval-Augmented Generation (RAG) mitigates these by grounding LLMs in external knowledge. However, in complex domains involving multiple, lengthy, or conflicting documents, traditional RAG suffers from information overload and inefficient synthesis, leading to inaccurate and untrustworthy answers. To address this, we propose CASC (Context-Adaptive Synthesis and Compression), a novel framework that intelligently processes retrieved contexts. CASC introduces a Context Analyzer & Synthesizer (CAS) module, powered by a fine-tuned smaller LLM, which performs key information extraction, cross-document consistency checking and conflict resolution, and question-oriented structured synthesis. This process transforms raw, scattered information into a highly condensed, structured, and semantically rich context, significantly reducing the token count and cognitive load for the final Reader LLM. We evaluate CASC on SciDocs-QA, a new challenging multi-document question answering dataset designed for complex scientific domains with inherent redundancies and conflicts. Our extensive experiments demonstrate that CASC consistently outperforms strong baselines. 

---
# Heterogeneous LLM Methods for Ontology Learning (Few-Shot Prompting, Ensemble Typing, and Attention-Based Taxonomies) 

**Authors**: Aleksandra Beliaeva, Temurbek Rahmatullaev  

**Link**: [PDF](https://arxiv.org/pdf/2508.19428)  

**Abstract**: We present a comprehensive system for addressing Tasks A, B, and C of the LLMs4OL 2025 challenge, which together span the full ontology construction pipeline: term extraction, typing, and taxonomy discovery. Our approach combines retrieval-augmented prompting, zero-shot classification, and attention-based graph modeling -- each tailored to the demands of the respective task. For Task A, we jointly extract domain-specific terms and their ontological types using a retrieval-augmented generation (RAG) pipeline. Training data was reformulated into a document to terms and types correspondence, while test-time inference leverages semantically similar training examples. This single-pass method requires no model finetuning and improves overall performance through lexical augmentation Task B, which involves assigning types to given terms, is handled via a dual strategy. In the few-shot setting (for domains with labeled training data), we reuse the RAG scheme with few-shot prompting. In the zero-shot setting (for previously unseen domains), we use a zero-shot classifier that combines cosine similarity scores from multiple embedding models using confidence-based weighting. In Task C, we model taxonomy discovery as graph inference. Using embeddings of type labels, we train a lightweight cross-attention layer to predict is-a relations by approximating a soft adjacency matrix. These modular, task-specific solutions enabled us to achieve top-ranking results in the official leaderboard across all three tasks. Taken together these strategies showcase the scalability, adaptability, and robustness of LLM-based architectures for ontology learning across heterogeneous domains.
Code is available at: this https URL 

---
# RAGAPHENE: A RAG Annotation Platform with Human Enhancements and Edits 

**Authors**: Kshitij Fadnis, Sara Rosenthal, Maeda Hanafi, Yannis Katsis, Marina Danilevsky  

**Link**: [PDF](https://arxiv.org/pdf/2508.19272)  

**Abstract**: Retrieval Augmented Generation (RAG) is an important aspect of conversing with Large Language Models (LLMs) when factually correct information is important. LLMs may provide answers that appear correct, but could contain hallucinated information. Thus, building benchmarks that can evaluate LLMs on multi-turn RAG conversations has become an increasingly important task. Simulating real-world conversations is vital for producing high quality evaluation benchmarks. We present RAGAPHENE, a chat-based annotation platform that enables annotators to simulate real-world conversations for benchmarking and evaluating LLMs. RAGAPHENE has been successfully used by approximately 40 annotators to build thousands of real-world conversations. 

---
