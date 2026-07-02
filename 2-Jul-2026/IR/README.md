# Diffusion-GR2: Diffusion Generative Reasoning Re-ranker 

**Authors**: Zhuoxuan Zhang, Kangqi Ni, Yuhang Chen, Mingfu Liang, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Frank Shyu, Adam, Song, Sandeep Pandey, Luke Simon, Tianlong Chen, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.01170)  

**Abstract**: Generative reasoning re-rankers achieve strong recommendation accuracy by emitting a chain-of-thought before re-ordering a candidate list, but they are slow at inference: an autoregressive (AR) decoder spends one sequential forward pass per reasoning token, and the reasoning trace far exceeds the ranking it produces. To reduce this cost, block-diffusion language models decode many positions in parallel over a few denoising steps and are substantially faster, yet naively converting an AR re-ranker into one opens two accuracy gaps: (1) a structural gap: answer positions are denoised in parallel and scored independently, so the decoder emits invalid rankings (duplicated, dropped, or out-of-set identifiers) that AR avoids through left-to-right masking; and (2) a distributional gap: fine-tuning the converted model on fixed teacher trajectories is off-policy relative to its own decoding at inference, leaving a residual accuracy gap. To close both gaps while keeping the speedup, we propose \textbf{Diffusion-GR2}, a recipe that converts our AR reasoning re-ranker (GR2) into a block-diffusion re-ranker. First, conversion fine-tuning (CFT) adapts the AR-initialized diffusion model to denoise the answer into a valid permutation on its own, without an external constrained decoder. Next, on-policy distillation (OPD) then supervises the model on its own decoded trajectories with dense per-token targets from the AR teacher. Finally, we apply a reinforcement-learning (RL) stage against a re-ranking reward on top of OPD's on-policy policy. Experiments on Amazon Beauty demonstrate that Diffusion-GR2 recovers to near-parity with the AR re-ranker, while block-parallel decoding raises decode throughput by $2.4$--$3.5\times$ at the model's reasoning output length. Ablations show that CFT recovers most of the conversion gap, and that on-policy distillation further closes it to the AR reference. 

---
# Trie-based Experiment Plans for Efficient IR Pipeline Experiments 

**Authors**: Irene Anu, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2607.01162)  

**Abstract**: Search engines are often formulated as cascading pipelines, where successive stages combine the results of different retrievers, and iteratively refine the ranking of candidate documents to obtain a final ranking, which can be presented to a user, or provided as context to an LLM. Such pipelines can be complex to evaluate in an end-to-end manner, necessitating measurement of Recall of early stages, and Precision of later stages, which are often interchangeable. PyTerrier is ideal for building and evaluating cascading retrieval pipelines, due to its declarative nature for pipeline construction and wide ecosystem of retrievers and rerankers. However, comparative evaluation of pipelines can be expensive due to repeated components. In this work, we describe the use of a trie data structure to formulate an experiment plan for comparative pipeline experiments that enhances experiment efficiency compared to a sequential "linear" plan. Empirically, on a demonstration experiment involving BM25, MonoT5 and DuoT5 on MSMARCO v2, we observe a 26% reduction in experiment duration. Finally, we report on a user study of undergraduate and postgraduate research students' use of the experiment plans. 

---
# MemSyco-Bench: Benchmarking Sycophancy in Agent Memory 

**Authors**: Zhishang Xiang, Zerui Chen, Yunbo Tang, Zhimin Wei, Ruqin Ning, Yujie Lin, Qinggang Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2607.01071)  

**Abstract**: Memory has emerged as a cornerstone of modern LLM-based agents, supporting their evolution from single-turn assistants to long-term collaborators. However, memory is not always beneficial: retrieved memories often induce a critical issue of sycophancy, causing agents to over-align with the user at the cost of factual accuracy or objective reasoning. Despite this emerging risk, existing memory benchmarks primarily evaluate whether memories are correctly stored, retrieved, or updated, while overlooking how retrieved memories influence downstream reasoning and decision-making. To bridge this gap, we propose MemSyco-Bench, a comprehensive benchmark for evaluating memory-induced sycophancy in agent systems. MemSyco-Bench measures when memory should influence a decision and how valid memory should be used. Specifically, it covers five tasks that assess whether agents can reject memory as factual evidence, respect its applicable scope, resolve conflicts between memory and objective evidence, track memory updates, and use valid memory for personalization. All related resources are collected for the community at this https URL. 

---
# As It Was: Aligning LLM Search Evaluation with Historical User Preferences 

**Authors**: Ali Vardasbi, Gustavo Penha, Enrico Palumbo, Claudia Hauff, Hugues Bouchard, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2607.01040)  

**Abstract**: Large-scale search systems evolve faster than human quality assurance can scale, especially for long-tail intents and multilingual queries. LLM-as-a-judge approaches provide a scalable alternative for evaluating the relevance of search engine result pages (SERPs), but judgments based solely on semantic similarity or world knowledge can drift from actual user preferences, particularly for ambiguous queries. We introduce a behavior-grounded LLM judge that augments each SERP item with a lightweight and auditable behavioral prior in the form of a Query-Relevance-Impressions (QRI) card. Each card summarizes how users have historically interacted with similar queries and results, providing compact empirical evidence that the judge can cite to resolve ambiguity and make more consistent relevance judgments while still relying on semantic reasoning. In a large-scale music search evaluation at Spotify, using relevance estimates derived from historical user interactions across 6,000 recomposed SERPs, the behavior-grounded judge achieves stronger alignment with user preferences, improving Spearman rank correlation by approximately 5% overall and yielding a 91% relative improvement on disagreement cases. On a multilingual human-judged dataset spanning five languages, grounding further increases correlation with human relevance judgments by 15%. Importantly, when evaluated against outcomes from a live A/B test, the grounded judge shows consistently higher alignment with the observed winning model. While absolute alignment remains moderate, these findings demonstrate that lightweight behavioral grounding can improve the reliability and practical usefulness of LLM-based evaluation in real-world search systems. 

---
# When RAG Meets Query Planning: Logical Query Trees for Resolving Exploratory Reasoning Problems 

**Authors**: Ganlin Xu, Linghao Zhang, Zhitao Yin, Hongda Xi, Chen Yang, Jiaqing Liang, Weijia Lu, Sihang Jiang, Yanghua Xiao, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.00508)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively grounds large language models (LLMs) in external knowledge but struggles with \textbf{exploratory reasoning problems (ERPs)} that are the complex queries involving high uncertainty and ambiguity. Resolving ERPs requires complex reasoning with unclear paths, tending to result in retrieval noise and error accumulation. Furthermore, the absence of an end-to-end planning mechanism makes it difficult to generate effective trajectories for ERPs. Motivated by database query planning, we introduce \emph{PlanRAG}, an RAG framework that models ERPs of natural language as \textbf{logical query trees (LQTs)}. However, translating ERPs into LQTs is non-trivial due to representation and optimization gaps between structured SQL and unstructured natural language, making it highly challenging to construct high-quality LQTs. To address these problems, we first decompose ERPs into atomic queries and then organize them into LQTs using dynamic programming guided by a cost model involving multiple complementary dimensions. Finally, we execute iterative aggregation, rewriting, retrieval, and generation over LQTs, processing nodes concurrently and propagating intermediate results upward, with further parallelization across multiple threads for efficiency. Our experimental results show that PlanRAG outperforms state-of-the-art iteration-based and graph-based RAG systems on our newly constructed dataset, \textbf{WikiWeb-ERP}, thereby providing a new formulation for optimizing natural language queries. Our source code and dataset are available at this https URL. 

---
# Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval 

**Authors**: Ivan Ji, Liuyi Hu, Harrison, Zhao, Lei Huang, Qunshu Zhang, Aameek Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.00448)  

**Abstract**: The two-tower model has been widely used for large-scale recommendation systems, particularly in the retrieval stage. Industry standards for training two-tower models typically involve in-batch and/or out-of-batch negative sampling. However, these methods often produce easy negatives that models can quickly learn, failing to sufficiently challenge the model. To address this issue, a novel self-supervised hard negative sampling technique is proposed that leverages a large language model (LLM) to generate hard negatives from the same cluster during model training. By utilizing the LLM to learn media representations, the proposed approach ensures that the generated negatives are more challenging and informative. This real-time sampling framework is designed for seamless integration into production models, capable of handling billions of training data points with minimal computational complexity. Experiments on public datasets, along with deployment to a large-scale online system, demonstrate that the proposed negative sampling technique outperforms widely used industry methods. Furthermore, analysis in industrial applications reveals that this sampling method can help break inherent feedback loops in recommendations and significantly reduce popularity bias. 

---
# Attribute-Prompted Kernel Hashing for Unsupervised Data-Efficient Cross-Modal Retrieval 

**Authors**: Runhao Li, Xiaoxu Ma, Zhenyu Weng, Yue Zhang, Guibo Luo, Huiping Zhuang, Zhiping Lin, Yap-Peng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2607.00379)  

**Abstract**: Unsupervised cross-modal hashing enables efficient retrieval of semantically related instances across different modalities without requiring manual semantic annotation. However, existing unsupervised methods rely heavily on large-scale image-text pairs. Collecting such data can be costly, particularly in scenarios where well-aligned pairs are scarce due to privacy and specialized constraints. More critically, existing methods tend to overfit to seen training data, restricting their generalization performance on unseen categories that the constrained training data cannot cover. To address these limitations, we propose Attribute-Prompted Kernel Hashing (APKH), a novel data-efficient approach that constructs a compact, modality-aligned Hamming space driven by the generalized attribute priors of vision-language foundation models. Specifically, APKH introduces two core modules: Context-optimized Attribute Kernel Mapping (CAKM) and Kernel-Smoothed Contrastive Alignment (KSCA). CAKM formulates cross-modal alignment through hyperspherical Radial Basis Function kernel mapping, optimizing dynamic attribute kernels via prompt learning to capture modality-invariant semantics. Furthermore, KSCA extends conventional point-to-point contrastive learning by modeling limited paired data as continuous kernel distributions. This explicit smoothing of the modality gap alleviates overfitting to sparse pairwise correlations. Extensive experiments demonstrate that APKH outperforms state-of-the-art hashing methods in the challenging cross-modal retrieval tasks from seen to unseen categories under data-constrained scenarios. 

---
# AGE: Adaptive-masking for Graph Embedding in Graph Retrieval-Augmented Generation 

**Authors**: Bao Long Nguyen Huu, Atsushi Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2607.00052)  

**Abstract**: GraphRAG is an extension of retrieval-augmented generation (RAG) that supports large language models (LLMs) by referring to graph-structured data as external knowledge. While this technique ideally captures intricate relationships, it often struggles with graph representations for LLMs, particularly for frozen LLMs, due to the misalignment between graph-based and text-based latent features. We tackle this issue by introducing the {\it Adaptive-masking for Graph Embedding (AGE)}. AGE employs a Transformer in a mask-based self-supervised learning (SSL) approach. We designed the architecture similar to text embedding encoders, addressing the latent feature misalignment. In contrast to natural language texts, graphs are concise representations, and there exist {\it key nodes} that hold dominant contextual information, which are challenging to predict from their surroundings. Masking such key nodes leads to inefficiency in the SSL process. Therefore, AGE focuses on predicting nodes apart from key nodes, utilizing a learnable node sampler. Our experimental results indicate that AGE significantly improves approaches using non-parametric search component in GraphQA tasks, achieving superior accuracy across four benchmark datasets with distinct characteristics. 

---
# Aligning Sentence Embeddings to Human Concepts via Sparse Autoencoders 

**Authors**: Wonseok Shin, Songkuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.00023)  

**Abstract**: Dense sentence embeddings are fundamental to modern Retrieval-Augmented Generation (RAG) systems but suffer from a lack of interpretability due to feature superposition. This opacity hinders the alignment of retrieval processes with human intent, as the entangled representations are difficult to analyze or control. In this work, we propose a method to disentangle the dense representations of sentence transformers (e.g., E5) into human-interpretable concepts using Top-k Sparse Autoencoders (SAEs). We demonstrate that these disentangled features align with specific semantic, syntactic, and pragmatic categories. Furthermore, we introduce an activation steering mechanism that allows for precise intervention in the retrieval process. By clamping specific latent features, we show that it is possible to re-rank search results to better align with user constraints without retraining the backbone model. Our findings suggest that SAE-based decomposition offers a viable path toward transparent and steerable neural information retrieval. 

---
# Learning User-Aware Recall: Personalized Retrieval in Long-Term Conversational Memory 

**Authors**: ZhiShu Jiang, Haibo Liu, Xin Shen, Guanqiang QI, Chenxi Miao, Weikang Li, Liwei Qian, Xin Pei, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.00017)  

**Abstract**: Long-term conversational agents are expected to remember past interactions, but memory is useful only when the right evidence is recalled for the right user. Existing memory-augmented LLM agents have made progress in building compact memory banks, yet retrieval is still often driven by query-centered similarity or fixed ranking rules, leaving user-conditioned relevance this http URL address this gap, we propose Profile-guided Personalized Retrieval Optimization (PPRO), a retrieval-centric framework that makes memory retrieval both user-aware and this http URL builds episodic and semantic memory banks from dialogue histories and derives a user profile from accumulated this http URL profile serves as an explicit personalized prior in memory ranking, allowing retrieval to account for stable user attributes, preferences, and this http URL further trains a query rewriter with Group Relative Policy Optimization, using both evidence retrieval quality and downstream answer quality as feedback while keeping the memory banks and answer model this http URL on LoCoMo and LongMemEval-S show consistent gains over training-free memory systems and training-based this http URL studies further show that both profile-guided ranking and retrieval-oriented rewriting contribute substantially to performance, highlighting retrieval optimization as a key factor in personalized long-term memory use. 

---
# Libra: Training the Environment for Agentic Information Retrieval 

**Authors**: Xuan Zhao, Andy Chiu, Gengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.00016)  

**Abstract**: Information localization within massive repositories is a cornerstone of agentic LLM systems. While synthetic data-driven optimization has proven successful in training LLMs, little attention has been paid to optimizing the agent's working environment (the repository itself) in a data-driven manner. To bridge this gap, we present Libra, a self-evolving framework that introduces mutable "catalogs" (hierarchical Markdown files serving as navigable indices) into the repository. Libra runs an LLM-driven optimization loop where a Prompter generates synthetic queries, a frozen Solver attempts to resolve them by navigating the catalogs, and a Healer rewrites the catalogs in response to the Solver's localization failures. Evaluations across 12 SWE-bench Lite repositories demonstrate that this environmental healing yields continual, logarithmic improvements in code localization accuracy. Furthermore, these environmental improvements transfer zero-shot across different LLMs and problem sets. Although the focus of this paper is to study the general behavior of such a system, we also demonstrate that a minimalist coding agent equipped with Libra-optimized catalogs outperforms state-of-the-art baselines. Code is available at this https URL and data at this https URL. 

---
# GRACE-RAG: Governed Retrieval Architecture for Canonical Evidence Synthesis, Enabling Lightweight Deployment in Closed-Domain Institutional Settings 

**Authors**: Asit Desai, Aman Kumar, Prashant Devadiga  

**Link**: [PDF](https://arxiv.org/pdf/2607.00013)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely used in institutional question answering settings where responses must be grounded in authoritative documentation (Gao et al., 2023). In entity-dense domains where relevant information is distributed across heterogeneous documents, vector-only retrieval often produces fragmented evidence and increases dependence on inference-time reasoning (Zhao et al., 2024). This paper introduces GRACE-RAG, a retrieval-governed, graph-augmented RAG architecture that externalizes structural reasoning from the generative stage to a structured retrieval layer, resolving structural ambiguity offline, enabling deployment on self-hosted lightweight models calibrated to closed-domain institutional vocabulary. Experiments across three model capacities: Mistral 24B, GPT OSS 120B, and Gemini 2.5 Flash show consistent improvements in completeness, depth, and anticipatory coverage, with overall quality gains of up to 20% under mid-scale models, indicating that retrieval architecture governs structural quality over model scale, reducing computational and latency footprint without dependence on proprietary systems. 

---
# PRA-RAG: Provably Robust Aggregation in Retrieval-Augmented Generation against Retrieval Corruption 

**Authors**: Xue Tan, Yi Zheng, Chang Huo, Yunruo Zhang, Yu Liu, Hao Luan, Zhuyang Yu, Xiaoyan Sun, Ping Chen, Jun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.00012)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge, effectively mitigating their inherent knowledge limitations. However, RAG remains vulnerable to poisoning attacks that manipulate retrieved texts to mislead model outputs. Existing defense mechanisms often lack theoretical robustness guarantees and perform unreliably when the LLM has limited knowledge of the retrieved content. In this work, we propose PRA-RAG, a provably robust retrieval aggregation algorithm designed to defend against poisoning attacks on retrieved texts. PRA-RAG samples multiple combinations of retrieved texts and utilizes geometric structures in the embedding space to identify a robust subset, from which a stable aggregated representation is derived. We provide theoretical bounds on the maximum impact of poisoned retrieved content and establish a quantitative measure of RAG's robustness. Experiments across multiple benchmarks and RAG architectures demonstrate that PRA-RAG reduces the attack success rate to as low as 1% while maintaining an accuracy of 71%, significantly outperforming representative state-of-the-art methods. 

---
# SkillSelect-Serve: Budget-Controllable and QoS-Aware Skill Service Recommendation and Composition for Small LLM Agents 

**Authors**: Jingyuan Zheng, Dongjing Wang, Xin Zhang, Butian Huang, Haiping Zhang, Dongjin Yu, Shuguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.00011)  

**Abstract**: Reusable skill libraries are becoming important infrastructure for large language model (LLM) agents, yet existing selection methods often treat skills as retrievable documents and return fixed top-k lists. This paper presents SkillSelect-Serve, a budget-controllable and QoS-aware framework that formulates agent skill selection as Skill Service Recommendation and Composition. SkillSelect-Serve represents raw skills as structured Skill Services with functional descriptions, dependencies, context cost, risk, and QoS-related attributes. A local Micro-Agent Requirement Planner converts natural-language tasks into structured service requirements, while a shared discovery backbone retrieves candidate services from a large registry. The framework then performs dual-granularity utility modeling with skill-level marginal suitability estimation and bundle-level calibration for coverage, redundancy, cost, and risk trade-offs. Experiments on 35,353 skills and 586 task queries show that SkillSelect-Serve consistently improves same-budget bundle recall and mean utility over fixed top-k retrieval baselines. 

---
# Prompt Optimization for User Simulation in Conversational Recommender Systems: A Multi-Objective Framework 

**Authors**: Nipun B Nair, Tongtong Wu, Weiqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.00010)  

**Abstract**: Conversational recommender systems (CRSs) are a core component of next-generation intelligent recommender systems because they enable users to actively elicit preferences, clarify intentions, and adapt recommendations in real time. However, there are two key obstacles in the CRS domain: evaluation and access to training data. Evaluating CRSs through real human studies is more critical than for traditional recommender systems, yet such studies are both costly and time-consuming. Moreover, CRS interaction data are often difficult to obtain for model training due to privacy concerns. Large language model (LLM)-based user simulators have shown promise in addressing both challenges by generating synthetic user interactions for evaluation and training. However, existing approaches suffer from systematic positive bias, data leakage, and limited behavioral diversity, and they rely on brittle manual prompt engineering that requires extensive domain expertise. In this paper, we propose a framework to automatically optimize prompts for LLM-based user simulators in CRSs, simultaneously mitigating these issues. Experimental results demonstrate that the proposed framework achieves improved behavioral alignment with human interaction patterns compared to baseline methods across diverse prompt settings. 

---
# SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information Extraction 

**Authors**: Sin Yu Bonnie Ho, Arlie Coles, Erik Larsson, Eric Marshall, Nathan Bodenstab, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2607.00008)  

**Abstract**: Extracting structured data from unstructured text using large language models (LLMs) becomes challenging when target schemas are large and complex. In such cases, including the full schema in the prompt increases cost and latency, risks lost-in-the-middle performance degradation, and can exceed context length limits. We propose SchemaRAG, a retrieval-augmented generation (RAG) framework that dynamically prunes the output schema space for schema-conditioned information extraction tasks by leveraging schema metadata and few-shot examples when available. We evaluate SchemaRAG on real-world healthcare and e-commerce datasets. Our results show that SchemaRAG can achieve up to an 8.8% increase in micro-F1, a 47% reduction in latency, and a 48% reduction in token costs, demonstrating its practicality for large-schema extraction. 

---
# BaRA: BFS-and-Reflection Web Data Collection Agent 

**Authors**: Soojeong Lee, Joseph Lee, Yongseong Cho, Sunjae Kim, Youngwoo Moon, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2607.00007)  

**Abstract**: Large language model (LLM)-based web agents reduce manual scripting for web data collection, yet on live websites, they often miss relevant pages, return incomplete multimodal outputs, or return media URLs that are not directly downloadable. We present BFS-and-Reflection Agent (BaRA), a framework for site-level collection under a fixed interaction budget. The framework combines bounded breadth-first search (BFS) traversal with history-based self-reflection. We evaluate BaRA on 50 synthetic websites with ground-truth reference sets. We additionally test on three public websites with cluttered or dynamic layouts. BaRA outperforms Pure LLM, SeeAct-Vision, and Browser-use on link discovery and downloadable multimodal extraction, with the largest gains in download-valid image and video recovery. Our code is available at this https URL. 

---
# Topological Void Analysis A Mathematical Framework for Systematic Technical Innovation Discovery in Knowledge Spaces 

**Authors**: Kris Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.00005)  

**Abstract**: Identifying where to innovate in a dense technical domain - such as operating systems or hardware/software co-design - is fundamentally a search problem in a high-dimensional knowledge space. Existing approaches rely on keyword search, citation proximity, or human intuition, none of which formalise the notion of an unexplored region that is simultaneously relevant to a target goal and absent from prior art.
We present Topological Void Analysis (TVA), a mathematical framework that defines topological voids as triads (A, B, C) in a dense-sparse hybrid embedding space. A void requires three conditions: (i) both concepts A and B are semantically cohesive with domain anchor C; (ii) their pairwise similarity falls within a calibrated marginality band - avoiding both obvious combinations and unrelated noise; and (iii) they share a sparse lexical bridge while the geodesic midpoint on the embedding hypersphere is unoccupied.
Applied to ~140k indexed documents, TVA generates 2,128 invention candidates across 96 targets; 90% survive automated quality filtering, yielding 191 REVISE and 1 APPROVE verdict from four-specialist adversarial review (0.05% end-to-end). Two case studies demonstrate the framework surfaces non-obvious connective tissue rather than merely obvious related pairs. 

---
# Why Advanced Encoders Lag on Sparse Retrieval? The Answer and an Approach to Bridging Vocabulary Gaps 

**Authors**: Zhichao Geng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.00004)  

**Abstract**: While advanced foundation models like ModernBERT significantly outperform older architectures in dense retrieval, they surprisingly lag behind the aging BERT-base baseline in learned sparse retrieval (LSR). We identify the root cause as the \textit{Vocabulary Gap}: modern tokenizers utilize raw, case-sensitive vocabularies designed for lossless reconstruction, which map single semantic units to redundant surface forms, wasting model capacity on morphological noise and hindering lexical matching. We formalize this intuition through a theoretical framework, demonstrating that appropriate vocabulary coarse-graining can tighten the generalization bounds by reducing complexity of the hypothesis class, provided that semantic integrity is preserved. To resolve this, we propose \textbf{Vocabulary Transfer (VT)}, a model-agnostic framework that migrates advanced encoders to sparse-friendly, normalized vocabularies with minimal computational cost.
VT utilizes a novel \textbf{Semantic Initialization} via spatial topology to preserve geometric structure and an \textbf{Activation Potential Calibration (APC)} mechanism to align pre-trained manifolds with sparsity constraints, preventing the dead neuron and dense collapse observed in standard fine-tuning. Empirically, VT is universally effective: it enables ModernBERT to achieve state-of-the-art performance on the BEIR benchmark (\textbf{52.4} nDCG, a \textbf{+4.7} improvement), resuscitates failing models like RoBERTa-large, and generalizes seamlessly to inference-free architectures and specialized domains. These results confirm that the performance lag is not an architectural deficiency but a solvable vocabulary mismatch. We've released our code and models.\footnote{this https URL. All details included.} 

---
# From "Strings" to "Things" for Personal Knowledge Graphs: Evaluating LLM Triple Extraction for Recommendation Systems 

**Authors**: Abhirup Dasgupta, Fernando Spadea, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2607.00003)  

**Abstract**: Personal Knowledge Graphs (PKGs) offer a privacy-preserving framework for modeling user preferences, yet constructing them from unstructured, decentralized conversational data remains a challenge. This paper bridges the gap between conversational "strings" and semantic "things" by presenting a reproducible pipeline for extracting structured user-preference triples using lightweight Large Language Models. We evaluate Qwen- and Gemma-based models on their ability to extract RDF-compliant triples linked to Wikidata identifiers from conversational data for PKG construction. Our evaluation assesses both the semantic extraction fidelity and the utility of the resulting graphs in a downstream recommendation task. We found that certain models performed well and had proportionally high downstream performance relative to their triple extraction performance. 

---
# RACORN-1: Adaptive Recall-Preserving Speedup for Low-Selectivity Filtered Vector Search 

**Authors**: Yoonseok Kim, Gyusik Choe  

**Link**: [PDF](https://arxiv.org/pdf/2607.00768)  

**Abstract**: Filtered Vector Search (FVS), which combines vector embedding similarity with structured metadata predicates, has emerged as a core requirement in RAG and production retrieval systems. ACORN-1, the representative In-filtering algorithm that reuses an existing HNSW index, substantially reduces latency at low selectivity but suffers connectivity instability below 5% selectivity and recall collapse below 1%. We propose RACORN-1, an in-place extension of ACORN-1 that resolves this collapse via (i) Adaptive Search Fallback (ASF) -- repurposing filter-failing nodes as transient bridges to detour around severed paths; bridge and two-hop candidate selection uses stride sampling for spatial diversity. While filter-first ACORN-family methods have a structural recall trade-off relative to distance-first HNSW, RACORN-1 improves the trade-off curve via ASF, minimizing recall loss while substantially reducing latency. Across three 1M-scale and one 40M-scale dataset, RACORN-1 delivers approximately 9-26x latency reduction over HNSW in the sweet spot (1%-0.3%), and recovers ACORN-1's recall collapse from 0.45-0.72 (1%) and 0.03-0.10 (0.3%) to 0.70-0.96 and 0.77-0.98 respectively. For the extreme-low-selectivity regime where linear scan can outperform graph search, we combine RACORN-1 with (ii) Adaptive Exact Fallback (AEF) in a variant RACORN-1+, achieving recall 1.00 with 20-75x speedup at 1M <=0.1% and 13x speedup at 40M 0.01%. Under a Negative Correlation evaluation (K-means clusters), where ACORN-1 collapses (recall 0.08-0.41), RACORN-1 maintains recall 0.80-0.98 with a 5-9x latency advantage over HNSW. Together, RACORN-1 and RACORN-1+ form an ACORN-1-compatible mechanism robust to both extreme-low-selectivity and adversarial query-filter correlation. 

---
# When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair Protects Tail Recall Under Bursty Churn 

**Authors**: Madhulatha Mandarapu, Sandeep Kunkunuru  

**Link**: [PDF](https://arxiv.org/pdf/2607.00728)  

**Abstract**: Graph approximate-nearest-neighbor (ANN) indexes (HNSW, DiskANN/Vamana) lose recall under insert/delete churn, because deletions orphan the greedy-search paths that route through removed nodes. Production systems restore navigability by repairing the graph on a fixed schedule (consolidate every X operations). We ask whether triggering local edge repair on a measured navigability-degradation signal, rather than a blind clock, spends a fixed repair budget better. On two real ANN datasets (SIFT-128 and Fashion-MNIST-784) under a controlled bursty churn stream, and comparing repair policies at matched amortized repair budget (equal consolidation count), signal-triggered repair Pareto-dominates fixed-cadence repair. The gain is concentrated on worst-case (tail) recall at scarce budget: at roughly one consolidation it improves the minimum recall@10 by +0.014 (SIFT) to +0.050 (Fashion-MNIST) across four stream seeds, with 95% confidence intervals excluding zero, while the mean-recall gain is small (<0.005). The advantage follows a clean drift-severity gradient -- larger for sparser, more fragile graphs -- and fades to parity when the index is robust or budget is ample. A cheap probe-recall signal is a valid, leading indicator of true recall (Spearman rho ~= 0.95). We contribute the mechanism, a budget-matched evaluation protocol that separates repair scheduling from repair spend, and an open, reproducible churn-repair harness. We deliberately do not claim a mean-recall improvement or a new index; a recall-versus-repair-cost bound and data-distribution-drift coupling are left as future work. 

---
# What Survives Into Context: A Diagnostic for Budget-Constrained Multi-Hop RAG and When Submodular Evidence Packing Improves It 

**Authors**: Ananto Nayan Bala  

**Link**: [PDF](https://arxiv.org/pdf/2607.00725)  

**Abstract**: Retrieval-augmented generation (RAG) under a fixed reader-context budget forces a selection problem: of the evidence retrieved, only a fraction can be shown to the reader. We argue that document recall -- the standard retrieval metric -- is the wrong quantity to optimize in this regime, and we make two contributions. First, as a general contribution, we introduce answer-in-context, a diagnostic that measures whether a gold answer survives as a contiguous span in the packed reader context (not the retrieved set). It predicts answer F1 better than recall (r=0.39-0.55 vs. about 0.31), separates answer quality roughly five-fold (0.60 vs. 0.12 on HotpotQA), and carries information beyond retrieval: it adds Delta R squared=0.17 over recall and shows a 4.6x EM gap even among questions where all gold was retrieved. We also confirm it interventionally: on 2WikiMultiHopQA a packing change that raises coverage but not answer-in-context yields no accuracy gain. Second, as a conditional contribution, we cast reader-context construction as budgeted monotone submodular maximization and build a packer that jointly optimizes relevance, query coverage, representativeness, and diversity. On HotpotQA with a 160-token budget and a 3B reader it beats a strong focused heuristic, MMR, and naive packing -- by up to +5.1 F1 at equal-or-lower token cost, across three seeds. Crucially, we map the scope of this win honestly: it requires the conjunction of (i) multi-hop complementary structure, (ii) retrieval that surfaces the evidence, (iii) a binding but not extreme budget, and (iv) a reader weak enough that evidence density, not reading capacity, is the bottleneck. A quantization-controlled reader-scale ladder (3B to 7B to 14B) shows the edge over the heuristic is absorbed by 7B and significantly reverses by 14B, while the diagnostic explains every boundary with a single variable. 

---
# Multi-Turn Agentic Scientific Literature Search via Workflow Induction 

**Authors**: Jisen Li, Bingxuan Li, Nanyi Jiang, Xuying Ning, Xiyao Wang, Yifan Shen, Heng Wang, Yuqing Jian, Xiaoxia Wu, Ben Athiwaratkun, Pan Lu, Jiaxuan You, Bingxin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.00597)  

**Abstract**: Scientific literature search often requires more than retrieving papers from a single query: users' intents are underspecified, preference-dependent, and evolve through interaction. Existing search agents typically rely on fixed pipelines or implicit language-only reasoning, making their search strategies difficult to control, inspect, and refine. We introduce PaperPilot, a multi-turn literature search agent that frames scientific search as workflow induction. Given an anchor paper and a user query, PaperPilot constructs an executable DAG of paper-search operators, including keyword search, citation expansion, filtering, scoring, reranking, and evidence extraction. User feedback is then used to refine both the query and the workflow itself. We train PaperPilot with supervised workflow imitation and preference optimization over controlled workflow corruptions. Experiments show that PaperPilot-9B improves over the base Qwen3.5-9B toolset agent under multi-turn interaction, increasing Hit@5 from 58.0 to 77.0, MRR from 47.5 to 59.4, and nDCG@10 from 26.8 to 32.5, while reducing workflow execution errors from 9.5% to 0%. These results show that explicit, editable search workflows provide an effective and controllable interface for aligning literature search agents with complex scientific intent. 

---
# Learning to Compose: Revisiting Proxy Task Design for Zero-Shot Composed Image Retrieval 

**Authors**: Jingjing Zhang, Lei Zhang, Zheren Fu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.00374)  

**Abstract**: Composed Image Retrieval (CIR) retrieves a target image from a reference image and a textual modification. While supervised CIR relies on costly triplets, Zero-Shot CIR (ZS-CIR) alleviates this reliance through proxy tasks trained on image-text pairs. However, existing proxy tasks primarily enhance visual and textual representations to accommodate a predefined composition mechanism such as pseudo-word injection into a frozen text encoder or linear feature arithmetic. As a result, the composition function itself remains unlearned, limiting the model's ability to express diverse and fine-grained semantic modifications. To address this, we propose FoCo, which models composition as two coordinated stages: focusing on modification-relevant visual content, and then completing the target semantics. We realize these through two proxy tasks: text-anchored visual aggregation to selectively gather visual content guided by localized textual semantics, and context-conditioned semantic completion to transform these aggregated visuals with the remaining scene context into a coherent composed representation. The tasks are trained jointly with a cross-instance contrastive objective, encouraging semantic diversity and discouraging shortcut composition strategies. Extensive experiments on four ZS-CIR benchmarks show FoCo's state-of-the-art performance and improved generalization. 

---
# Identifying and Resolving Pitfalls of Knowledge-Based VQA Benchmarks: Auditing, Repairing, and Augmenting 

**Authors**: Qian Ma, S M Rayeed, Charles V. Stewart, Qiong Wu, Yao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.00159)  

**Abstract**: Knowledge-Based Visual Question Answering (KB-VQA) aims to evaluate whether Visual Language Models (VLMs) can retrieve, ground, and reason over external structured knowledge beyond visual evidence. In practice, answer accuracy is widely adopted as the primary evaluation metric, implicitly treating correctness as a proxy for knowledge-grounded reasoning. However, for existing KB-VQA benchmarks, this proxy relies on critical assumptions that are often overlooked and rendered unreliable by benchmark issues: annotated answer must be derivable from the associated knowledge base, question must be well-posed with sufficient constraints, and visual setting must meaningfully require grounded disambiguation. In this work, we show that these assumptions are systematically violated in existing KB-VQA benchmarks. Our audit reveals substantial instances with missing or contradicted answers and underspecified questions that render accuracy a misleading metric. Furthermore, we find that existing datasets rely on visually trivial, single-entity scenes that bypass the need for sophisticated visual-to-knowledge mapping. We demonstrate that even with controlled architectures, these flaws lead to distorted model rankings and overestimations of reasoning capabilities. To address this, we introduce (1) a principled audit-and-repair protocol that restores answer derivability and question clarity, and (2) a controlled multi-entity augmentation protocol that introduces visual ambiguity to challenge initial retrieval and grounded reasoning. Re-evaluation under corrected and augmented settings yields markedly different performance trends. Our findings call for rethinking evaluation protocols and designing more interaction-aware KB-VQA benchmarks that prioritize verifiable reasoning over simple matching. 

---
