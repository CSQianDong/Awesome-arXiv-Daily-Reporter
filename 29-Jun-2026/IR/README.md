# Fast and Feasible: Permutation-based Constrained Reranking for Revenue Maximization 

**Authors**: Svetlana Shirokovskikh, Anastasiia Soboleva, Ekaterina Solodneva, Aleksandr Katrutsa, Roman Loginov, Egor Samosvat  

**Link**: [PDF](https://arxiv.org/pdf/2606.28059)  

**Abstract**: Search and recommender systems have produced highly relevant search results. A natural next step in the development of such systems in e-commerce is to rerank these results to increase the platform's revenue from paid promotion products. However, maximizing revenue alone may degrade the user experience by reducing relevance or increasing fraud risk. To avoid this, we state the reranking problem as an integer linear program ($ILP$) that maximizes revenue subject to per-query constraints on other metrics, e.g., relevance. Since solving $ILP$ exactly for every query is slow for deployment to the online service, we propose a lightweight permutation-based reranking approximation algorithm PermR. At each step, the algorithm selects a pair of neighboring items and swaps them to either improve the objective or repair a violated constraint. We evaluate PermR across multiple categories of a large classified platform in offline and online settings. PermR achieves about 63\% of the ILP revenue improvement, within production latency limits, preserving all constraints. In a 14-day online A/B test over 56 million search queries, PermR increased revenue by $2$\%. 

---
# Listwise Explanation of Embedding-Based Rankings via Semantic Chunk Grouping 

**Authors**: Hyunkyu Kim, Yeeun Yoo, Youngjun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2606.27980)  

**Abstract**: Dense embedding rankers score documents through contextual sentence- and passage-level representations. Yet many listwise explanation methods still attribute rankings to isolated words. This feature-unit mismatch leaves word-level features too fragmented for dense semantic ranking. We introduce ChunkGroupSHAP, a listwise Shapley method that clusters semantically related chunks into shared cross-document features. Masking a group perturbs all documents with related evidence, attributing rankings at a granularity closer to dense representations while preserving the listwise setup. Our findings across MS MARCO, FinanceBench, AILACaseDocs, and FinQA with E5 rankers and BM25 show that the best explanation unit is setting-dependent: word features for lexical BM25, corpus-level groups for dense rankers, and query-local grouping for heterogeneous web retrieval. Feature units should thus follow both the ranker's representational granularity and the structure of the retrieved corpus. 

---
# An LLM-Powered Semantic Alignment Framework for Journal Recommendation 

**Authors**: Yanglin Yan, Zicheng Xie, Tianchen Gao, Rui Pan, Hansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27930)  

**Abstract**: Journal recommendation is an important task in scholarly information systems. Existing approaches typically rely on supervised learning models, manually engineered features, or historical interaction data, which may limit their generalizability and interpretability. We propose an LLM-powered semantic alignment framework that formulates journal recommendation as a semantic matching problem between manuscript content and journal scope descriptions. The framework enables large language models (LLMs) to infer journal suitability directly from article titles, abstracts, keywords, and candidate journal information without task-specific training. Experiments are conducted using DeepSeek-V3 on a dataset of 23,609 articles from 49 journals in statistics and related fields. The proposed framework achieves Top-3, Top-5, and Top-10 accuracies of 40.23\%, 53.67\%, and 70.05\%, respectively. Additional analyses show that incorporating reference information generally improves recommendation performance and that recommendations remain highly stable across repeated runs, with an average Top-5 Jaccard similarity of 84\%. The framework also generates interpretable reasoning outputs that provide insights into the recommendation process. These findings demonstrate the potential of LLMs as a training-free and scalable paradigm for journal recommendation and scholarly decision support. 

---
# From Bootstrapping to Sequence Modeling: A Unified Generative Framework for Personalized Landing-Page Modeling 

**Authors**: Fan Li, Chang Meng, Jiaqi Fu, Shuchang Liu, Tianke Zhang, Xueliang Wang, Xiaoqiang Feng, Yongqi Liu, Kaiqiao Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2606.27865)  

**Abstract**: Modern online platforms increasingly adopt multi-page architectures to accommodate diverse user needs. On these platforms, page navigation (the process of directing users to specific functional pages upon app entry) serves as a critical gateway that shapes user's first impression and significantly influences subsequent engagement. To optimize this process, Kuaishou formulated the task of Personalized Landing Page Modeling (PLPM) and proposed KLAN, a reinforcement learning framework built upon Conservative Q-Learning (CQL). However, CQL-based approaches suffer from two fundamental limitations: (1) the Markov assumption fails to capture the strong non-Markovian temporal dependencies inherent in real-world user behaviors, and (2) TD learning with bootstrapping incurs severe cumulative errors and credit assignment difficulties under delayed rewards, particularly in long-horizon settings where users enter the app multiple times daily. To address these limitations, we propose GLAN (Generative Landing-page Adaptive Navigator), a sequence modeling framework built on Decision Transformer to tackle PLPM from a unified global-local perspective. Specifically, GLAN incorporates two key modules. First, we design the L-RTG module that captures users' inter-day consumption dynamics to provide accurate global guidance for all page assignments within a day. Furthermore, we propose the HRM module that decomposes session-level feedback into fine-grained signals, enabling precise local supervision for each page assignment. Extensive online experiments conducted on the Kuaishou platform demonstrate the effectiveness of GLAN, achieving +0.158\% and +0.108\% improvements on Daily Active Users (DAU) and user Lifetime (LT) respectively. 

---
# End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inference 

**Authors**: Yuhang Chen, Jinhao Duan, Ruichen Zhang, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Parish Aggarwal, Frank Shyu, Luke Simon, Sandeep Pandey, Tianlong Chen, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27743)  

**Abstract**: Large Language Models (LLMs) inference is typically deployed under a static resource assumption, where models execute a fixed computational graph regardless of the runtime environment. However, real-world cloud infrastructure is inherently dynamic, characterized by fluctuating availability (e.g., spot instance preemption) and tiered Quality-of-Service requirements. In such volatile settings, static models are inflexible: they either crash under resource constraints or waste compute on redundant operations. To bridge this gap, we propose Learning to Allocate (L2A), an end-to-end framework for resource-adaptive inference. Unlike prior methods that condition only on input difficulty, we formulate inference as a constrained allocation problem conditioned on both the input and the runtime resource budget itself. We introduce lightweight, budget-conditioned and input-aware gating networks integrated into the LLM. These gates are trained via a unified objective that jointly optimizes task performance, logical consistency, and resource costs along three axes matching how real-world dynamics manifest: layer skipping for memory and depth pressure, head pruning for throughput contention, and reasoning-token reduction for latency tightening. This lets the model learn a budget-aware policy beyond input difficulty alone: it adaptively configures its computational footprint with respect to real-time resource dynamics, maximizing reasoning depth when resources permit while enforcing strict frugality when budgets tighten. A single L2A model traces the entire compute-accuracy Pareto frontier on Llama-3-8B and Qwen-3-4B: at up to 34% realized layer sparsity, it stays within 0.6% of the dense baseline on GSM8K, with the same gap holding zero-shot on out-of-distribution tasks, while every static or heuristic baseline requires a separately tuned model and still drops by 5-10% at comparable inference time. 

---
# Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation 

**Authors**: Yuhang Chen, Xianfeng Wu, Jinhao Duan, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Parish Aggarwal, Frank Shyu, Luke Simon, Sandeep Pandey, Xi Liu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27732)  

**Abstract**: Discrete diffusion language models (dLLMs) recover masked tokens in parallel, offering significant speedups over autoregressive (AR) generation. However, such promising frameworks face a fundamental architectural design dilemma: \ding{182} Adopting bidirectional attention achieves strong generation quality by allowing each position to access the full context, but is inherently incompatible with KV caching, limiting inference throughput in batch-serving scenarios; \ding{183} Conversely, causal attention enables efficient cached inference but loses all right-side context, substantially degrading generation quality. This paper introduces Bifocal dLLMs, a new paradigm that resolves this dilemma through \emph{asymmetric bidirectional context}. Analogous to bifocal lenses, we instantiate the paradigm as \textbf{R2LM} (Right-to-Left Mamba), which combines two complementary mechanisms: $a$) standard causal attention providing precise left-context with full KV cache compatibility, while $b$) a lightweight reverse Mamba SSM sidecar supplying compressed right-side context without breaking cacheability. Comprehensive experiments on continued pretraining of Qwen3-1.7B with 60B tokens demonstrate that R2LM achieves $2.4\times$ to $12.9\times$ higher throughput than bidirectional dLLMs and $1.9\times$ to $2.9\times$ speedup over AR baselines in batch serving through parallel decoding with KV caching, while exceeding the causal baseline on most benchmarks and surpassing the bidirectional dLLM on average. 

---
# Intuition-Guided Latent Reasoning for LLM-Based Recommendation 

**Authors**: Chang Liu, Yimeng Bai, Xiaoyan Zhao, Yang Zhang, Qifan Wang, Fuli Feng, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2606.27684)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive reasoning capabilities in complex problem-solving tasks, motivating their use for preference reasoning in recommender systems. Latent reasoning, which operates in continuous hidden spaces rather than discrete tokens, has recently emerged as a promising paradigm for LLM-based recommendation. However, existing methods often start from unconstrained reasoning points, where hidden representations are misaligned with target item embeddings, leading to suboptimal reasoning trajectories.
Inspired by cognitive neuroscience, which suggests that human multi-step reasoning is guided by intuition as a latent prior, we propose \emph{IntuRec}, a two-stage framework that anchors latent reasoning with \emph{recommendation intuition}. In the extraction stage, the LLM-based recommender generates a top-$K$ candidate set based on users' histories as the source of intuition. In the injection stage, the candidate set is transformed into a preference-aligned intuition embedding using self- and cross-attention mechanisms, which initializes the reasoning start point and guides subsequent latent reasoning. By providing a semantically grounded starting point, IntuRec efficiently explores the preference space along more accurate reasoning trajectories. Extensive experiments on multiple real-world datasets demonstrate that IntuRec consistently outperforms state-of-the-art baselines. We release our code at this https URL. 

---
# A Sensitivity-Aware Test Collection for Search Among Personal Information 

**Authors**: Jack McKechnie, Graham McDonald, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2606.27559)  

**Abstract**: Traditional search tasks aim to satisfy user information needs by returning a subset of a collection of documents, ranked by the documents' relevance to a user query. However, some collections that contain useful information also contain sensitive personal information. Recently, there has been increasing interest in the development of Sensitivity-Aware Search (SAS) retrieval models to provide users with effective retrieval results without revealing such sensitive information. To develop such systems, test collections containing both sensitive and non-sensitive information, a set of queries, and query-document relevance assessments are required. The Enron email corpus contains real business-related emails, where some emails also contain sensitive personal information. However, the original Enron collection does not contain queries or query-relevance assessments. To this end, we crowdsource 150 query formulations for 50 different topics and 11,471 query-relevance assessments for a subset of the Enron documents that have been manually labelled for sensitivity. We follow best practices for using large language models (LLMs) in Information Retrieval evaluation to extend the collection further with additional LLM judged query-relevance assessments and sensitivity labels. We present baseline performances for relevance, sensitivity classification, and sensitivity-aware search on the collection. We make the collection available, including through the popular ir_datasets package, and provide pre-built sparse and dense indices on Huggingface to facilitate easy experimentation. 

---
# Context-Aware Explanations for Spatialized Document Layouts 

**Authors**: Wei Liu, John Wenskovitch, Chris North, Rebecca Faust  

**Link**: [PDF](https://arxiv.org/pdf/2606.28081)  

**Abstract**: Spatialized document layouts are widely used for exploratory analysis of text corpora, but interpreting the spatial organization of documents and the relationships between regions remains challenging. Existing approaches primarily summarize document content or explain how layouts are generated, providing limited support for understanding spatial relationships within the layout itself. We present CAPE, a context-aware explanation framework that generates natural-language explanations grounded in both document semantics and layout-derived spatial context. CAPE identifies salient spatial patterns (e.g., clusters, subgroups, outliers, and bridging documents) and constructs multi-level contextual representations to guide LLM-based explanation generation. It supports both AI-guided overview and user-driven exploration, with explanations available at multiple levels of detail. We demonstrate CAPE on news and scholarly document layouts and evaluate it in a controlled user study against keyword-based and content-only LLM baselines. Our results suggest that spatially grounded explanations are perceived as more helpful than content-only baselines for interpreting the spatial organization of document layouts. 

---
# Single and Multi Truth Data Fusion using Large Language Models 

**Authors**: Hira Beril Kucuk, Norman W Paton, Jiaoyan Chen, Zhenyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28062)  

**Abstract**: Data fusion, also known as truth discovery, is a data integration problem that aims to determine the correct value or set of values for each attribute of an object when presented with potentially conflicting values from multiple sources. Data fusion tasks belong to two main categories: single-truth scenarios, where each attribute has only one correct value, and multi-truth scenarios, where multiple values can be valid simultaneously. This paper investigates the use of Large Language Models (LLMs) in data fusion tasks for tabular data. Various prompting strategies, encompassing both single-truth and multi-truth scenarios, are investigated empirically. Domain-dependent, domain-independent, zero-shot and one-shot prompts are evaluated on three different benchmark datasets. Experimental results demonstrate that LLM-based approaches outperform traditional unsupervised truth discovery methods, such as DART and LTM, across all datasets. The codebase of this study has been made publicly available on GitHub. 

---
# SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieval 

**Authors**: Sergey Kurilenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.27976)  

**Abstract**: Dense embeddings underpin semantic search and RAG, yet a leaked vector store hands much of the underlying text back to whoever holds it. The attacks that make this possible (few-shot alignment, zero-shot inversion, unsupervised cross-space translation) share one weakness: the protected store is a single global geometry that can be aligned to a known one. A secret global rotation, the usual lightweight defence, is no exception: orthogonal Procrustes recovers it once the attacker has about the subspace dimension in known pairs.
We introduce Shard, a retrieval-preserving embedding transform that removes this weak axis. The centred embedding is split into a short public prefix (for stage-1 retrieval) and a private residual sharded into C cells under separate secret keys; the residual is reranked under CKKS, where the keys cancel and leave the inner product exact. A single parameter C runs the design from the global-linear baseline it replaces (C=1) to per-document micro-keys (C=N). Because the rerank is full-dimensional, Shard returns the raw-space nDCG@10 that half-SVD truncation gives up; and because the residual is keyed cell-locally, mapping it back to a common frame under a diffuse known-plaintext leak costs roughly C times more anchors (median 200 to 102,400 at C=256), for a few encrypted queries. The short public prefix leaks far less neighbour structure, and a micro-key limit drives the residual graph to zero with an unlinkable, renewable template. The barrier holds against learned, non-linear and unsupervised aligners, and where a matched-utility noise defence de-anonymises almost every probe, Shard de-anonymises none. We are plain about the limits: within a cell the keys cancel, a targeted attacker needs only about d_priv anchors, and an overlapping reference corpus still leaks through the prefix. Shard is an attack-aware geometric defence, not a cryptographic guarantee. 

---
# DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums 

**Authors**: Dana Rezazadegan, Atie Kia, Phongpadid Nandavong, Dominique Carlon, Jeremy Nguyen, Abhik Banerjee, James Marshall, Anthony McCosker, Yong-Bin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27619)  

**Abstract**: Dyslexic learners increasingly use artificial intelligence (AI) tools to support reading, writing, organisation, and study-related tasks. However, their lived experiences with these tools remain largely underexamined. This paper proposes DysLexLens, a low-resource LLM framework, designed to analyse dyslexic learners experience with AI through online forum discussions. DysLexLens is designed as an end-to-end, evidence-traceable architecture which transforms noisy social media posts into a dictionary-driven corpora, provides knowledge-graph (KG)-based question reasoning, generates verifiable query responses, and enables response evaluation through quantitative and human-grounded assessment. DysLexLens has four key features. First, it employs a dictionary-driven filtering method to construct a more focused Reddit corpus on dyslexia and AI, filtering out noisy and weakly related posts to improve the relevance of data collected from low-resource forum contexts. Second, it integrates LLM-assisted semantic analysis with KG-based query reasoning to uncover meaningful patterns. Third, it has quantitative evaluation metrics (RAGAS and Query Robustness) to measure LLM-generated response performance. Fourth, it provides structured qualitative validation guidelines for assessing response quality, with a specific focus on hallucination and evidence alignment. We demonstrate the effectiveness of DysLexLens using dyslexia-related Reddit forum data and 30 questions. The results show its potential generalisability to other low-resource forum data contexts. DysLexLens, sample data, questions and evaluation results are available at Github to support reproducibility. 

---
# Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval 

**Authors**: Leonardo Venuta, Francesco Tosoni, Paolo Ferragina  

**Link**: [PDF](https://arxiv.org/pdf/2606.27401)  

**Abstract**: Semantic code search and clone detection are essential for software development, maintenance, and reuse. This paper evaluates the effectiveness, efficiency, and scalability of contemporary deep learning models for first-stage recall in large-scale code-to-code search engines. Benchmarking across multiple programming languages and datasets reveals critical limits in the precision and scalability of these models on Terabyte-scale source-code collections. We present LLM-based code normalisation and query-rewriting schemes that yield significant gains in precision for lower-performing models. Our results question the sustainability of resource-constrained deployment and the assumed robustness of current code-specialised LLMs across datasets. We conclude with actionable insights for building scalable, efficient code-retrieval systems. 

---
