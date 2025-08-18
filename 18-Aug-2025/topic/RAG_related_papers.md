# Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering 

**Authors**: Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, Ning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11247)  

**Abstract**: Multi-hop question answering (MHQA) requires integrating knowledge scattered across multiple passages to derive the correct answer. Traditional retrieval-augmented generation (RAG) methods primarily focus on coarse-grained textual semantic similarity and ignore structural associations among dispersed knowledge, which limits their effectiveness in MHQA tasks. GraphRAG methods address this by leveraging knowledge graphs (KGs) to capture structural associations, but they tend to overly rely on structural information and fine-grained word- or phrase-level retrieval, resulting in an underutilization of textual semantics. In this paper, we propose a novel RAG approach called HGRAG for MHQA that achieves cross-granularity integration of structural and semantic information via hypergraphs. Structurally, we construct an entity hypergraph where fine-grained entities serve as nodes and coarse-grained passages as hyperedges, and establish knowledge association through shared entities. Semantically, we design a hypergraph retrieval method that integrates fine-grained entity similarity and coarse-grained passage similarity via hypergraph diffusion. Finally, we employ a retrieval enhancement module, which further refines the retrieved results both semantically and structurally, to obtain the most relevant passages as context for answer generation with the LLM. Experimental results on benchmark datasets demonstrate that our approach outperforms state-of-the-art methods in QA performance, and achieves a 6$\times$ speedup in retrieval efficiency. 

---
# Retrieval-augmented reasoning with lean language models 

**Authors**: Ryan Sze-Yin Chan, Federico Nanni, Tomas Lazauskas, Rosie Wood, Penelope Yong, Lionel Tarassenko, Mark Girolami, James Geddes, Andrew Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11386)  

**Abstract**: This technical report details a novel approach to combining reasoning and retrieval augmented generation (RAG) within a single, lean language model architecture. While existing RAG systems typically rely on large-scale models and external APIs, our work addresses the increasing demand for performant and privacy-preserving solutions deployable in resource-constrained or secure environments. Building on recent developments in test-time scaling and small-scale reasoning models, we develop a retrieval augmented conversational agent capable of interpreting complex, domain-specific queries using a lightweight backbone model. Our system integrates a dense retriever with fine-tuned Qwen2.5-Instruct models, using synthetic query generation and reasoning traces derived from frontier models (e.g., DeepSeek-R1) over a curated corpus, in this case, the NHS A-to-Z condition pages. We explore the impact of summarisation-based document compression, synthetic data design, and reasoning-aware fine-tuning on model performance. Evaluation against both non-reasoning and general-purpose lean models demonstrates that our domain-specific fine-tuning approach yields substantial gains in answer accuracy and consistency, approaching frontier-level performance while remaining feasible for local deployment. All implementation details and code are publicly released to support reproducibility and adaptation across domains. 

---
# CryptoScope: Utilizing Large Language Models for Automated Cryptographic Logic Vulnerability Detection 

**Authors**: Zhihao Li, Zimo Ji, Tao Zheng, Hao Ren, Xiao Lan  

**Link**: [PDF](https://arxiv.org/pdf/2508.11599)  

**Abstract**: Cryptographic algorithms are fundamental to modern security, yet their implementations frequently harbor subtle logic flaws that are hard to detect. We introduce CryptoScope, a novel framework for automated cryptographic vulnerability detection powered by Large Language Models (LLMs). CryptoScope combines Chain-of-Thought (CoT) prompting with Retrieval-Augmented Generation (RAG), guided by a curated cryptographic knowledge base containing over 12,000 entries. We evaluate CryptoScope on LLM-CLVA, a benchmark of 92 cases primarily derived from real-world CVE vulnerabilities, complemented by cryptographic challenges from major Capture The Flag (CTF) competitions and synthetic examples across 11 programming languages. CryptoScope consistently improves performance over strong LLM baselines, boosting DeepSeek-V3 by 11.62%, GPT-4o-mini by 20.28%, and GLM-4-Flash by 28.69%. Additionally, it identifies 9 previously undisclosed flaws in widely used open-source cryptographic projects. 

---
# RAG for Geoscience: What We Expect, Gaps and Opportunities 

**Authors**: Runlong Yu, Shiyuan Luo, Rahul Ghosh, Lingyao Li, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.11246)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by combining retrieval with generation. However, its current workflow remains largely text-centric, limiting its applicability in geoscience. Many geoscientific tasks are inherently evidence-hungry. Typical examples involve imputing missing observations using analog scenes, retrieving equations and parameters to calibrate models, geolocating field photos based on visual cues, or surfacing historical case studies to support policy analyses. A simple ``retrieve-then-generate'' pipeline is insufficient for these needs. We envision Geo-RAG, a next-generation paradigm that reimagines RAG as a modular retrieve $\rightarrow$ reason $\rightarrow$ generate $\rightarrow$ verify loop. Geo-RAG supports four core capabilities: (i) retrieval of multi-modal Earth data; (ii) reasoning under physical and domain constraints; (iii) generation of science-grade artifacts; and (iv) verification of generated hypotheses against numerical models, ground measurements, and expert assessments. This shift opens new opportunities for more trustworthy and transparent geoscience workflows. 

---
