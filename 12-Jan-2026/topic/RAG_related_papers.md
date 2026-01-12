# Retrieval-Augmented Multi-LLM Ensemble for Industrial Part Specification Extraction 

**Authors**: Muzakkiruddin Ahmed Mohammed, John R. Talburt, Leon Claasssens, Adriaan Marais  

**Link**: [PDF](https://arxiv.org/pdf/2601.05266)  

**Abstract**: Industrial part specification extraction from unstructured text remains a persistent challenge in manufacturing, procurement, and maintenance, where manual processing is both time-consuming and error-prone. This paper introduces a retrieval-augmented multi-LLM ensemble framework that orchestrates nine state-of-the-art Large Language Models (LLMs) within a structured three-phase pipeline. RAGsemble addresses key limitations of single-model systems by combining the complementary strengths of model families including Gemini (2.0, 2.5, 1.5), OpenAI (GPT-4o, o4-mini), Mistral Large, and Gemma (1B, 4B, 3n-e4b), while grounding outputs in factual data using FAISS-based semantic retrieval. The system architecture consists of three stages: (1) parallel extraction by diverse LLMs, (2) targeted research augmentation leveraging high-performing models, and (3) intelligent synthesis with conflict resolution and confidence-aware scoring. RAG integration provides real-time access to structured part databases, enabling the system to validate, refine, and enrich outputs through similarity-based reference retrieval. Experimental results using real industrial datasets demonstrate significant gains in extraction accuracy, technical completeness, and structured output quality compared to leading single-LLM baselines. Key contributions include a scalable ensemble architecture for industrial domains, seamless RAG integration throughout the pipeline, comprehensive quality assessment mechanisms, and a production-ready solution suitable for deployment in knowledge-intensive manufacturing environments. 

---
# Engineering the RAG Stack: A Comprehensive Review of the Architecture and Trust Frameworks for Retrieval-Augmented Generation Systems 

**Authors**: Dean Wampler, Dave Nielson, Alireza Seddighi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05264)  

**Abstract**: This article provides a comprehensive systematic literature review of academic studies, industrial applications, and real-world deployments from 2018 to 2025, providing a practical guide and detailed overview of modern Retrieval-Augmented Generation (RAG) architectures. RAG offers a modular approach for integrating external knowledge without increasing the capacity of the model as LLM systems expand. Research and engineering practices have been fragmented as a result of the increasing diversity of RAG methodologies, which encompasses a variety of fusion mechanisms, retrieval strategies, and orchestration approaches. We provide quantitative assessment frameworks, analyze the implications for trust and alignment, and systematically consolidate existing RAG techniques into a unified taxonomy. This document is a practical framework for the deployment of resilient, secure, and domain-adaptable RAG systems, synthesizing insights from academic literature, industry reports, and technical implementation guides. It also functions as a technical reference. 

---
# TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Wenbiao Tao, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.05254)  

**Abstract**: Retrieval-Augmented Generation enhances language models by retrieving external knowledge to support informed and grounded responses. However, traditional RAG methods rely on fragment-level retrieval, limiting their ability to address query-focused summarization queries. GraphRAG introduces a graph-based paradigm for global knowledge reasoning, yet suffers from inefficiencies in information extraction, costly resource consumption, and poor adaptability to incremental updates. To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework designed for efficient global reasoning and scalable graph maintenance. TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extracts object tags and their relationships from documents and organizes them into hierarchical domain tag chains for structured knowledge representation, and (2) Tag-Guided Retrieval-Augmented Generation, which retrieves domain-centric tag chains to localize and synthesize relevant knowledge during inference. This design significantly adapts to smaller language models, improves retrieval granularity, and supports efficient knowledge increment. Extensive experiments on UltraDomain datasets spanning Agriculture, Computer Science, Law, and cross-domain settings demonstrate that TagRAG achieves an average win rate of 95.41\% against baselines while maintaining about 14.6x construction and 1.9x retrieval efficiency compared with GraphRAG. 

---
# Cross-Document Topic-Aligned Chunking for Retrieval-Augmented Generation 

**Authors**: Mile Stankovic  

**Link**: [PDF](https://arxiv.org/pdf/2601.05265)  

**Abstract**: Chunking quality determines RAG system performance. Current methods partition documents individually, but complex queries need information scattered across multiple sources: the knowledge fragmentation problem. We introduce Cross-Document Topic-Aligned (CDTA) chunking, which reconstructs knowledge at the corpus level. It first identifies topics across documents, maps segments to each topic, and synthesizes them into unified chunks.
On HotpotQA multi-hop reasoning, our method reached 0.93 faithfulness versus 0.83 for contextual retrieval and 0.78 for semantic chunking, a 12% improvement over current industry best practice (p < 0.05). On UAE Legal texts, it reached 0.94 faithfulness with 0.93 citation accuracy. At k = 3, it maintains 0.91 faithfulness while semantic methods drop to 0.68, with a single CDTA chunk containing information requiring multiple traditional fragments.
Indexing costs are higher, but synthesis produces information-dense chunks that reduce query-time retrieval needs. For high-query-volume applications with distributed knowledge, cross-document synthesis improves measurably over within-document optimization. 

---
# Quantifying Document Impact in RAG-LLMs 

**Authors**: Armin Gerami, Kazem Faghih, Ramani Duraiswami  

**Link**: [PDF](https://arxiv.org/pdf/2601.05260)  

**Abstract**: Retrieval Augmented Generation (RAG) enhances Large Language Models (LLMs) by connecting them to external knowledge, improving accuracy and reducing outdated information. However, this introduces challenges such as factual inconsistencies, source conflicts, bias propagation, and security vulnerabilities, which undermine the trustworthiness of RAG systems. A key gap in current RAG evaluation is the lack of a metric to quantify the contribution of individual retrieved documents to the final output. To address this, we introduce the Influence Score (IS), a novel metric based on Partial Information Decomposition that measures the impact of each retrieved document on the generated response. We validate IS through two experiments. First, a poison attack simulation across three datasets demonstrates that IS correctly identifies the malicious document as the most influential in $86\%$ of cases. Second, an ablation study shows that a response generated using only the top-ranked documents by IS is consistently judged more similar to the original response than one generated from the remaining documents. These results confirm the efficacy of IS in isolating and quantifying document influence, offering a valuable tool for improving the transparency and reliability of RAG systems. 

---
# Naiad: Novel Agentic Intelligent Autonomous System for Inland Water Monitoring 

**Authors**: Eirini Baltzi, Tilemachos Moumouris, Athena Psalta, Vasileios Tsironis, Konstantinos Karantzalos  

**Link**: [PDF](https://arxiv.org/pdf/2601.05256)  

**Abstract**: Inland water monitoring is vital for safeguarding public health and ecosystems, enabling timely interventions to mitigate risks. Existing methods often address isolated sub-problems such as cyanobacteria, chlorophyll, or other quality indicators separately. NAIAD introduces an agentic AI assistant that leverages Large Language Models (LLMs) and external analytical tools to deliver a holistic solution for inland water monitoring using Earth Observation (EO) data. Designed for both experts and non-experts, NAIAD provides a single-prompt interface that translates natural-language queries into actionable insights. Through Retrieval-Augmented Generation (RAG), LLM reasoning, external tool orchestration, computational graph execution, and agentic reflection, it retrieves and synthesizes knowledge from curated sources to produce tailored reports. The system integrates diverse tools for weather data, Sentinel-2 imagery, remote-sensing index computation (e.g., NDCI), chlorophyll-a estimation, and established platforms such as CyFi. Performance is evaluated using correctness and relevancy metrics, achieving over 77% and 85% respectively on a dedicated benchmark covering multiple user-expertise levels. Preliminary results show strong adaptability and robustness across query types. An ablation study on LLM backbones further highlights Gemma 3 (27B) and Qwen 2.5 (14B) as offering the best balance between computational efficiency and reasoning performance. 

---
# Open World Knowledge Aided Single-Cell Foundation Model with Robust Cross-Modal Cell-Language Pre-training 

**Authors**: Haoran Wang, Xuanyi Zhang, Shuangsang Fang, Longke Ran, Ziqing Deng, Yong Zhang, Yuxiang Li, Shaoshuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05648)  

**Abstract**: Recent advancements in single-cell multi-omics, particularly RNA-seq, have provided profound insights into cellular heterogeneity and gene regulation. While pre-trained language model (PLM) paradigm based single-cell foundation models have shown promise, they remain constrained by insufficient integration of in-depth individual profiles and neglecting the influence of noise within multi-modal data. To address both issues, we propose an Open-world Language Knowledge-Aided Robust Single-Cell Foundation Model (OKR-CELL). It is built based on a cross-modal Cell-Language pre-training framework, which comprises two key innovations: (1) leveraging Large Language Models (LLMs) based workflow with retrieval-augmented generation (RAG) enriches cell textual descriptions using open-world knowledge; (2) devising a Cross-modal Robust Alignment (CRA) objective that incorporates sample reliability assessment, curriculum learning, and coupled momentum contrastive learning to strengthen the model's resistance to noisy data. After pretraining on 32M cell-text pairs, OKR-CELL obtains cutting-edge results across 6 evaluation tasks. Beyond standard benchmarks such as cell clustering, cell-type annotation, batch-effect correction, and few-shot annotation, the model also demonstrates superior performance in broader multi-modal applications, including zero-shot cell-type annotation and bidirectional cell-text retrieval. 

---
# FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG 

**Authors**: Maxime Dassen, Rebecca Kotula, Kenton Murray, Andrew Yates, Dawn Lawrie, Efsun Kayi, James Mayfield, Kevin Duh  

**Link**: [PDF](https://arxiv.org/pdf/2601.05866)  

**Abstract**: Retrieval-Augmented Generation (RAG) models are critically undermined by citation hallucinations, a deceptive failure where a model confidently cites a source that fails to support its claim. Existing work often attributes hallucination to a simple over-reliance on the model's parametric knowledge. We challenge this view and introduce FACTUM (Framework for Attesting Citation Trustworthiness via Underlying Mechanisms), a framework of four mechanistic scores measuring the distinct contributions of a model's attention and FFN pathways, and the alignment between them. Our analysis reveals two consistent signatures of correct citation: a significantly stronger contribution from the model's parametric knowledge and greater use of the attention sink for information synthesis. Crucially, we find the signature of a correct citation is not static but evolves with model scale. For example, the signature of a correct citation for the Llama-3.2-3B model is marked by higher pathway alignment, whereas for the Llama-3.1-8B model, it is characterized by lower alignment, where pathways contribute more distinct, orthogonal information. By capturing this complex, evolving signature, FACTUM outperforms state-of-the-art baselines by up to 37.5% in AUC. Our findings reframe citation hallucination as a complex, scale-dependent interplay between internal mechanisms, paving the way for more nuanced and reliable RAG systems. 

---
