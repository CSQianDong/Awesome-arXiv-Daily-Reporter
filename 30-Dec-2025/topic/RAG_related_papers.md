# HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG 

**Authors**: Cattalyya Nuengsigkapian  

**Link**: [PDF](https://arxiv.org/pdf/2512.22442)  

**Abstract**: Retrieval-Augmented Generation (RAG) in open-domain settings faces significant challenges regarding irrelevant information in retrieved documents and the alignment of generated answers with user intent. We present HiFi-RAG (Hierarchical Filtering RAG), the winning closed-source system in the Text-to-Text static evaluation of the MMU-RAGent NeurIPS 2025 Competition. Our approach moves beyond standard embedding-based retrieval via a multi-stage pipeline. We leverage the speed and cost-efficiency of Gemini 2.5 Flash (4-6x cheaper than Pro) for query formulation, hierarchical content filtering, and citation attribution, while reserving the reasoning capabilities of Gemini 2.5 Pro for final answer generation. On the MMU-RAGent validation set, our system outperformed the baseline, improving ROUGE-L to 0.274 (+19.6%) and DeBERTaScore to 0.677 (+6.2%). On Test2025, our custom dataset evaluating questions that require post-cutoff knowledge (post January 2025), HiFi-RAG outperforms the parametric baseline by 57.4% in ROUGE-L and 14.9% in DeBERTaScore. 

---
# DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for Retrieval-Augmented Generation 

**Authors**: Shiyan Liu, Jian Ma, Rui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2512.22629)  

**Abstract**: As Retrieval-Augmented Generation (RAG) systems evolve toward more sophisticated architectures, ensuring their trustworthiness through explainable and robust evaluation becomes critical. Existing scalar metrics suffer from limited interpretability, inadequate uncertainty quantification, and computational inefficiency in multi-system comparisons, hindering responsible deployment of RAG technologies. We introduce DICE (Discrete Interpretable Comparative Evaluation), a two-stage, evidence-coupled framework that advances explainability and robustness in RAG evaluation. DICE combines deep analytical reasoning with probabilistic $\{A, B, Tie\}$ scoring to produce transparent, confidence-aware judgments that support accountable system improvement through interpretable reasoning traces, enabling systematic error diagnosis and actionable insights. To address efficiency challenges at scale, DICE employs a Swiss-system tournament that reduces computational complexity from $O(N^2)$ to $O(N \log N)$, achieving a 42.9% reduction in our eight-system evaluation while preserving ranking fidelity. Validation on a curated Chinese financial QA dataset demonstrates that DICE achieves 85.7% agreement with human experts, substantially outperforming existing LLM-based metrics such as RAGAS. Our results establish DICE as a responsible, explainable, and efficient paradigm for trustworthy RAG system assessment. 

---
# The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction 

**Authors**: Haoyu Pei, Zhongyang Liu, Xiangyi Xiao, Xiaocong Du, Haipeng Zhang, Kunpeng Zhang, Suting Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.23489)  

**Abstract**: Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: this https URL. 

---
# Bidirectional RAG: Safe Self-Improving Retrieval-Augmented Generation Through Multi-Stage Validation 

**Authors**: Teja Chinthala  

**Link**: [PDF](https://arxiv.org/pdf/2512.22199)  

**Abstract**: Retrieval-Augmented Generation RAG systems enhance large language models by grounding responses in external knowledge bases, but conventional RAG architectures operate with static corpora that cannot evolve from user interactions. We introduce Bidirectional RAG, a novel RAG architecture that enables safe corpus expansion through validated write back of high quality generated responses. Our system employs a multi stage acceptance layer combining grounding verification (NLI based entailment, attribution checking, and novelty detection to prevent hallucination pollution while enabling knowledge accumulation. Across four datasets Natural Questions, TriviaQA, HotpotQA, Stack Overflow with three random seeds 12 experiments per system, Bidirectional RAG achieves 40.58% average coverage nearly doubling Standard RAG 20.33% while adding 72% fewer documents than naive write back 140 vs 500. Our work demonstrates that self improving RAG is feasible and safe when governed by rigorous validation, offering a practical path toward RAG systems that learn from deployment. 

---
# FasterPy: An LLM-based Code Execution Efficiency Optimization Framework 

**Authors**: Yue Wu, Minghao Han, Ruiyin Li, Peng Liang, Amjed Tahir, Zengyang Li, Qiong Feng, Mojtaba Shahin  

**Link**: [PDF](https://arxiv.org/pdf/2512.22827)  

**Abstract**: Code often suffers from performance bugs. These bugs necessitate the research and practice of code optimization. Traditional rule-based methods rely on manually designing and maintaining rules for specific performance bugs (e.g., redundant loops, repeated computations), making them labor-intensive and limited in applicability. In recent years, machine learning and deep learning-based methods have emerged as promising alternatives by learning optimization heuristics from annotated code corpora and performance measurements. However, these approaches usually depend on specific program representations and meticulously crafted training datasets, making them costly to develop and difficult to scale. With the booming of Large Language Models (LLMs), their remarkable capabilities in code generation have opened new avenues for automated code optimization. In this work, we proposed FasterPy, a low-cost and efficient framework that adapts LLMs to optimize the execution efficiency of Python code. FasterPy combines Retrieval-Augmented Generation (RAG), supported by a knowledge base constructed from existing performance-improving code pairs and corresponding performance measurements, with Low-Rank Adaptation (LoRA) to enhance code optimization performance. Our experimental results on the Performance Improving Code Edits (PIE) benchmark demonstrate that our method outperforms existing models on multiple metrics. The FasterPy tool and the experimental results are available at this https URL. 

---
# ReGAIN: Retrieval-Grounded AI Framework for Network Traffic Analysis 

**Authors**: Shaghayegh Shajarian, Kennedy Marsh, James Benson, Sajad Khorsandroo, Mahmoud Abdelsalam  

**Link**: [PDF](https://arxiv.org/pdf/2512.22223)  

**Abstract**: Modern networks generate vast, heterogeneous traffic that must be continuously analyzed for security and performance. Traditional network traffic analysis systems, whether rule-based or machine learning-driven, often suffer from high false positives and lack interpretability, limiting analyst trust. In this paper, we present ReGAIN, a multi-stage framework that combines traffic summarization, retrieval-augmented generation (RAG), and Large Language Model (LLM) reasoning for transparent and accurate network traffic analysis. ReGAIN creates natural-language summaries from network traffic, embeds them into a multi-collection vector database, and utilizes a hierarchical retrieval pipeline to ground LLM responses with evidence citations. The pipeline features metadata-based filtering, MMR sampling, a two-stage cross-encoder reranking mechanism, and an abstention mechanism to reduce hallucinations and ensure grounded reasoning. Evaluated on ICMP ping flood and TCP SYN flood traces from the real-world traffic dataset, it demonstrates robust performance, achieving accuracy between 95.95% and 98.82% across different attack types and evaluation benchmarks. These results are validated against two complementary sources: dataset ground truth and human expert assessments. ReGAIN also outperforms rule-based, classical ML, and deep learning baselines while providing unique explainability through trustworthy, verifiable responses. 

---
# ReCollab: Retrieval-Augmented LLMs for Cooperative Ad-hoc Teammate Modeling 

**Authors**: Conor Wallace, Umer Siddique, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.22129)  

**Abstract**: Ad-hoc teamwork (AHT) requires agents to infer the behavior of previously unseen teammates and adapt their policy accordingly. Conventional approaches often rely on fixed probabilistic models or classifiers, which can be brittle under partial observability and limited interaction. Large language models (LLMs) offer a flexible alternative: by mapping short behavioral traces into high-level hypotheses, they can serve as world models over teammate behavior. We introduce \Collab, a language-based framework that classifies partner types using a behavior rubric derived from trajectory features, and extend it to \ReCollab, which incorporates retrieval-augmented generation (RAG) to stabilize inference with exemplar trajectories. In the cooperative Overcooked environment, \Collab effectively distinguishes teammate types, while \ReCollab consistently improves adaptation across layouts, achieving Pareto-optimal trade-offs between classification accuracy and episodic return. These findings demonstrate the potential of LLMs as behavioral world models for AHT and highlight the importance of retrieval grounding in challenging coordination settings. 

---
