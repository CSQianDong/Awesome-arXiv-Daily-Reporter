# RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing 

**Authors**: Timothy Rupprecht, Enfu Nan, Arash Akbari, Arman Akbari, Lei Lu, Priyanka Maan, Sean Duffy, Pu Zhao, Yumei He, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12168)  

**Abstract**: Role-playing Large language models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, education, and governance, where failures can directly impact user trust and well-being. A cost effective paradigm for LLM role-playing is few-shot learning, but existing approaches often cause models to break character in unexpected and potentially harmful ways, especially when interacting with hostile users. Inspired by Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a text retrieval problem and propose a new prompting framework called RAGs-to-Riches, which leverages curated reference demonstrations to condition LLM responses. We evaluate our framework with LLM-as-a-judge preference voting and introduce two novel token-level ROUGE metrics: Intersection over Output (IOO) to quantity how much an LLM improvises and Intersection over References (IOR) to measure few-shot demonstrations utilization rate during the evaluation tasks. When simulating interactions with a hostile user, our prompting strategy incorporates in its responses during inference an average of 35% more tokens from the reference demonstrations. As a result, across 453 role-playing interactions, our models are consistently judged as being more authentic, and remain in-character more often than zero-shot and in-context Learning (ICL) methods. Our method presents a scalable strategy for building robust, human-aligned LLM role-playing frameworks. 

---
# A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students 

**Authors**: Guy Tel-Zur  

**Link**: [PDF](https://arxiv.org/pdf/2509.11947)  

**Abstract**: This project addresses a critical pedagogical need: offering students continuous, on-demand academic assistance beyond conventional reception hours. I present a domain-specific Retrieval-Augmented Generation (RAG) system powered by a quantized Mistral-7B Instruct model and deployed as a Telegram bot. The assistant enhances learning by delivering real-time, personalized responses aligned with the "Introduction to Parallel Processing" course materials. GPU acceleration significantly improves inference latency, enabling practical deployment on consumer hardware. This approach demonstrates how consumer GPUs can enable affordable, private, and effective AI tutoring for HPC education. 

---
# MMORE: Massive Multimodal Open RAG & Extraction 

**Authors**: Alexandre Sallinen, Stefan Krsteski, Paul Teiletche, Marc-Antoine Allard, Baptiste Lecoeur, Michael Zhang, Fabrice Nemo, David Kalajdzic, Matthias Meyer, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2509.11937)  

**Abstract**: We introduce MMORE, an open-source pipeline for Massive Multimodal Open RetrievalAugmented Generation and Extraction, designed to ingest, transform, and retrieve knowledge from heterogeneous document formats at scale. MMORE supports more than fifteen file types, including text, tables, images, emails, audio, and video, and processes them into a unified format to enable downstream applications for LLMs. The architecture offers modular, distributed processing, enabling scalable parallelization across CPUs and GPUs. On processing benchmarks, MMORE demonstrates a 3.8-fold speedup over single-node baselines and 40% higher accuracy than Docling on scanned PDFs. The pipeline integrates hybrid dense-sparse retrieval and supports both interactive APIs and batch RAG endpoints. Evaluated on PubMedQA, MMORE-augmented medical LLMs improve biomedical QA accuracy with increasing retrieval depth. MMORE provides a robust, extensible foundation for deploying task-agnostic RAG systems on diverse, real-world multimodal data. The codebase is available at this https URL. 

---
# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking 

**Authors**: Wensheng Lu, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11552)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems. 

---
# Intelligent Reservoir Decision Support: An Integrated Framework Combining Large Language Models, Advanced Prompt Engineering, and Multimodal Data Fusion for Real-Time Petroleum Operations 

**Authors**: Seyed Kourosh Mahjour, Seyed Saman Mahjour  

**Link**: [PDF](https://arxiv.org/pdf/2509.11376)  

**Abstract**: The petroleum industry faces unprecedented challenges in reservoir management, requiring rapid integration of complex multimodal datasets for real-time decision support. This study presents a novel integrated framework combining state-of-the-art large language models (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Pro) with advanced prompt engineering techniques and multimodal data fusion for comprehensive reservoir analysis. The framework implements domain-specific retrieval-augmented generation (RAG) with over 50,000 petroleum engineering documents, chain-of-thought reasoning, and few-shot learning for rapid field adaptation. Multimodal integration processes seismic interpretations, well logs, and production data through specialized AI models with vision transformers. Field validation across 15 diverse reservoir environments demonstrates exceptional performance: 94.2% reservoir characterization accuracy, 87.6% production forecasting precision, and 91.4% well placement optimization success rate. The system achieves sub-second response times while maintaining 96.2% safety reliability with no high-risk incidents during evaluation. Economic analysis reveals 62-78% cost reductions (mean 72%) relative to traditional methods with 8-month payback period. Few-shot learning reduces field adaptation time by 72%, while automated prompt optimization achieves 89% improvement in reasoning quality. The framework processed real-time data streams with 96.2% anomaly detection accuracy and reduced environmental incidents by 45%. We provide detailed experimental protocols, baseline comparisons, ablation studies, and statistical significance testing to ensure reproducibility. This research demonstrates practical integration of cutting-edge AI technologies with petroleum domain expertise for enhanced operational efficiency, safety, and economic performance. 

---
# CultureSynth: A Hierarchical Taxonomy-Guided and Retrieval-Augmented Framework for Cultural Question-Answer Synthesis 

**Authors**: Xinyu Zhang, Pei Zhang, Shuang Luo, Jialong Tang, Yu Wan, Baosong Yang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10886)  

**Abstract**: Cultural competence, defined as the ability to understand and adapt to multicultural contexts, is increasingly vital for large language models (LLMs) in global environments. While several cultural benchmarks exist to assess LLMs' cultural competence, current evaluations suffer from fragmented taxonomies, domain specificity, and heavy reliance on manual data annotation. To address these limitations, we introduce CultureSynth, a novel framework comprising (1) a comprehensive hierarchical multilingual cultural taxonomy covering 12 primary and 130 secondary topics, and (2) a Retrieval-Augmented Generation (RAG)-based methodology leveraging factual knowledge to synthesize culturally relevant question-answer pairs. The CultureSynth-7 synthetic benchmark contains 19,360 entries and 4,149 manually verified entries across 7 languages. Evaluation of 14 prevalent LLMs of different sizes reveals clear performance stratification led by ChatGPT-4o-Latest and Qwen2.5-72B-Instruct. The results demonstrate that a 3B-parameter threshold is necessary for achieving basic cultural competence, models display varying architectural biases in knowledge processing, and significant geographic disparities exist across models. We believe that CultureSynth offers a scalable framework for developing culturally aware AI systems while reducing reliance on manual annotation\footnote{Benchmark is available at this https URL.}. 

---
# Quality Assessment of Tabular Data using Large Language Models and Code Generation 

**Authors**: Ashlesha Akella, Akshar Kaul, Krishnasuri Narayanam, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.10572)  

**Abstract**: Reliable data quality is crucial for downstream analysis of tabular datasets, yet rule-based validation often struggles with inefficiency, human intervention, and high computational costs. We present a three-stage framework that combines statistical inliner detection with LLM-driven rule and code generation. After filtering data samples through traditional clustering, we iteratively prompt LLMs to produce semantically valid quality rules and synthesize their executable validators through code-generating LLMs. To generate reliable quality rules, we aid LLMs with retrieval-augmented generation (RAG) by leveraging external knowledge sources and domain-specific few-shot examples. Robust guardrails ensure the accuracy and consistency of both rules and code snippets. Extensive evaluations on benchmark datasets confirm the effectiveness of our approach. 

---
# Real-Time RAG for the Identification of Supply Chain Vulnerabilities 

**Authors**: Jesse Ponnock, Grace Kenneally, Michael Robert Briggs, Elinor Yeo, Tyrone Patterson III, Nicholas Kinberg, Matthew Kalinowski, David Hechtman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10469)  

**Abstract**: New technologies in generative AI can enable deeper analysis into our nation's supply chains but truly informative insights require the continual updating and aggregation of massive data in a timely manner. Large Language Models (LLMs) offer unprecedented analytical opportunities however, their knowledge base is constrained to the models' last training date, rendering these capabilities unusable for organizations whose mission impacts rely on emerging and timely information. This research proposes an innovative approach to supply chain analysis by integrating emerging Retrieval-Augmented Generation (RAG) preprocessing and retrieval techniques with advanced web-scraping technologies. Our method aims to reduce latency in incorporating new information into an augmented-LLM, enabling timely analysis of supply chain disruptors. Through experimentation, this study evaluates the combinatorial effects of these techniques towards timeliness and quality trade-offs. Our results suggest that in applying RAG systems to supply chain analysis, fine-tuning the embedding retrieval model consistently provides the most significant performance gains, underscoring the critical importance of retrieval quality. Adaptive iterative retrieval, which dynamically adjusts retrieval depth based on context, further enhances performance, especially on complex sup- ply chain queries. Conversely, fine-tuning the LLM yields limited improvements and higher resource costs, while techniques such as downward query abstraction significantly outperforms upward abstraction in practice. 

---
# DSRAG: A Domain-Specific Retrieval Framework Based on Document-derived Multimodal Knowledge Graph 

**Authors**: Mengzheng Yang, Yanfei Ren, David Osei Opoku, Ruochang Li, Peng Ren, Chunxiao Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.10467)  

**Abstract**: Current general-purpose large language models (LLMs) commonly exhibit knowledge hallucination and insufficient domain-specific adaptability in domain-specific tasks, limiting their effectiveness in specialized question answering scenarios. Retrieval-augmented generation (RAG) effectively tackles these challenges by integrating external knowledge to enhance accuracy and relevance. However, traditional RAG still faces limitations in domain knowledge accuracy and context this http URL enhance domain-specific question answering performance, this work focuses on a graph-based RAG framework, emphasizing the critical role of knowledge graph quality during the generation process. We propose DSRAG (Domain-Specific RAG), a multimodal knowledge graph-driven retrieval-augmented generation framework designed for domain-specific applications. Our approach leverages domain-specific documents as the primary knowledge source, integrating heterogeneous information such as text, images, and tables to construct a multimodal knowledge graph covering both conceptual and instance layers. Building on this foundation, we introduce semantic pruning and structured subgraph retrieval mechanisms, combining knowledge graph context and vector retrieval results to guide the language model towards producing more reliable responses. Evaluations using the Langfuse multidimensional scoring mechanism show that our method excels in domain-specific question answering, validating the efficacy of integrating multimodal knowledge graphs with retrieval-augmented generation. 

---
# Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL 

**Authors**: Xiaoying Song, Anirban Saha Anik, Dibakar Barua, Pengcheng Luo, Junhua Ding, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01058)  

**Abstract**: Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation 

---
# FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval 

**Authors**: Ying Li, Mengyu Wang, Miguel de Carvalho, Sotirios Sabanis, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.12042)  

**Abstract**: Financial disclosures such as 10-K filings present challenging retrieval problems due to their length, regulatory section hierarchy, and domain-specific language, which standard retrieval-augmented generation (RAG) models underuse. We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a retrieval framework tailored to financial documents. FinGEAR combines a finance lexicon for Item-level guidance (FLAM), dual hierarchical indices for within-Item search (Summary Tree and Question Tree), and a two-stage cross-encoder reranker. This design aligns retrieval with disclosure structure and terminology, enabling fine-grained, query-aware context selection. Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR delivers consistent gains in precision, recall, F1, and relevancy, improving F1 by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over prior tree-based systems, while also increasing downstream answer accuracy with a fixed reader. By jointly modeling section hierarchy and domain lexicon signals, FinGEAR improves retrieval fidelity and provides a practical foundation for high-stakes financial analysis. 

---
