# SAGE-PAIR: Source-Pretrained Routed Residual Adaptation for Sparse-Feedback Recommendation 

**Authors**: Weiqing Zeng, Zida Yang, Guquan You, Jiexin Zheng, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.09728)  

**Abstract**: Recommendation systems often enter a new serving window before enough outcomes are available to retrain the deployed model. We study this regime with 0.05-1.0% target feedback, a range that spans our stress tests while keeping supervision far below the degrees of freedom of the source recommender. At this scale, existing fine-tuning and meta-learning methods still estimate target-specific updates from extremely sparse feedback, leaving the update direction weakly identified. SAGE-PAIR instead transfers a source-supported residual correction space and only calibrates this space using target outcomes. We introduce SAGE-PAIR, a staged adapter that pretrains a low-dimensional global residual on an adjacent source period and adds a soft router over residual experts conditioned on source user profiles. Target feedback then calibrates this residual state rather than relearning the recommender. On a locked test split of KuaiSAR-v2, SAGE-PAIR (Full) achieves a 1.77% relative improvement in NDCG@10, while a core-protocol comparison shows a further 0.25% gain from routed experts beyond source-pretrained PAIR. Matched source-pretrained LoRA and Adapter-Tuning baselines remain near the source model, supporting the need for a different adaptation space. Thus, SAGE-PAIR improves ranking quality with millisecond-scale adaptation and inference costs. Sparse-feedback adaptation is fundamentally governed by the identifiability of the update space rather than by label count alone. Source pretraining supplies estimable correction directions for scarce labels, while user-conditioned routing contributes a consistent, differentiated cross-user correction on top of the global correction. Our code is available at this https URL. 

---
# SPARSEUP: Towards Better Recipes for Training Modern Learned Sparse Retrievers 

**Authors**: Thibault Formal, Pierre Erbacher, Cosimo Rulli  

**Link**: [PDF](https://arxiv.org/pdf/2611.09680)  

**Abstract**: Learned sparse embeddings are a powerful alternative to dense retrieval, yet they have fallen behind single-vector and late interaction models. One reason for their limited adoption is that they are notoriously difficult to train, generally requiring very tight conditions to converge. We present a recipe for training Splade-like encoders on modern backbones and data that achieves a better efficiency-effectiveness trade-off. We discuss the role of logit distribution and its impact on the training stability and the final sparsity of the encoder. We also introduce a new per-position fan-out control - a hard cap on how many vocabulary dimensions each position may emit. We further apply a simple vocabulary merging strategy that removes redundant entries - such as case and word-boundary variants of the same word - which are common in cased BPE-based backbones. We apply our recipe to the LateOn backbone and training data, obtaining SPARSEUP, a sparse encoder that reaches 56.4 average nDCG@10 on BEIR-13, the highest we know of among public vocabulary-based sparse encoders below 150M parameters. 

---
# Orthogonal Norm Growth in Two-Tower Retrieval: A Popularity Mechanism for Unified Discovery 

**Authors**: Andrei Atamaniuk  

**Link**: [PDF](https://arxiv.org/pdf/2611.09294)  

**Abstract**: Unified search and recommendation systems may update the same item vectors from both interaction streams. We study how repeated updates can increase an item vector's norm and affect retrieval under an unnormalized dot product. A cosine-only loss makes the output gradient perpendicular to the current item vector - tangent to its constant-norm sphere. Training changes encoder parameters, however, so the realized item-vector movement need not remain tangent. We derive sufficient conditions under which it does; every nonzero update then increases squared norm by its squared length. When the tasks share an item encoder, sequential updates from both can accumulate in the same norm, which raw-inner-product serving can expose for either task. This conditional result is not a theorem about joint SGD. Controlled ablations test these conditions. On MovieLens 32M and Amazon Books, accumulated tangent movement accompanies frequency-norm association and additional head-item exposure under raw-dot rather than normalized-item ranking. In a nonlinear tower combining a per-item ID vector with a shared content encoding, active item movement remains near-tangent. We separate its norm change into the ID-update contribution and other output changes, then use a matched training intervention that removes immediate norm growth caused by the ID-row update. The intervention reduces norm-related exposure but also recommendation utility, so it is diagnostic rather than a proposed mitigation. Our logs are recommendation-only: the formulation supplies a shared-representation audit, while prevalence in joint traffic remains open. 

---
# Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL 

**Authors**: Duc-Vu Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.09200)  

**Abstract**: We study Vietnamese statutory Article retrieval on DRiLL using features for provision identity and local form, same-law support, and citation and lineage evidence. Across five random seeds, selectors sharing candidates, listwise Qwen evidence, and learner settings improve official $F_2$ from 71.77% to 72.91%, with the largest descriptive gain on two-Article questions. Added features account for 18.20% of the proposed selector's total decision-tree split gain, measured as training-objective reduction, mainly from same-law and citation evidence. Cases reveal base provisions and amendments, companion recovery, distractor removal, and failures. These findings motivate research on other datasets, representations, and learners without establishing individual feature effects or requiring LightGBM, hand-crafted scores, or explicit graphs. 

---
# PAGSL: Path-Augmenting Graph Structure Learning for explainable scholar recommendation 

**Authors**: Jiale Xu, Mingshu Ai, Junjie Huang, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.09126)  

**Abstract**: Explainable scholar recommendation aims to identify potential scholars of interest and provide reasonable explanations for the recommendations. However, existing frameworks struggle with optimization-induced noise and "hub node bias", yielding trivial explanations that lack hierarchical depth and personalization. To this end, we propose a novel explainable scholar recommendation model, Path-Augmenting Graph Structure Learning (named PAGSL). PAGSL combines representation learning and semantic path knowledge to ensure recommendation performance while providing interpretable results from two perspectives. Specifically, it activates latent academic paths among entities through graph augmentation and employs a path-based neural network to encode these paths for recommendation. A masking mechanism with path score optimization is designed to generate concise and informative explanations. We evaluate PAGSL on three academic datasets (Scholat, AMiner, and ResearchGate), comprising approximately 49,000 authors and over 330,000 interaction records. Experimental evaluations validate that PAGSL outperforms state-of-the-art baselines, achieving significant improvements of 6.31\% in F1-score, 6.29\% in Accuracy and 5.95\% in AUC for recommendation. Crucially, in terms of explainability, we provide quantitative results and visualization effects. Furthermore, ablation studies validate the effectiveness of our heterogeneous convolution and path representation learning. PAGSL achieves strong performance in both recommendation accuracy and explainability. 

---
# MV-IVF: Multi-Vector Retrieval via Multi-Vector Clustering 

**Authors**: Kishen N Gowda, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram  

**Link**: [PDF](https://arxiv.org/pdf/2611.09040)  

**Abstract**: Multi-vector retrieval represents each query and document as a set of embeddings and scores relevance via fine-grained set-to-set interactions such as MaxSim (Chamfer similarity). This expressiveness improves retrieval quality over single-vector representations, but makes search substantially more expensive. Existing systems address this challenge primarily through proxy representations or filter-and-rerank pipelines that reduce candidate generation to single-vector search. We study the complementary direction of building a native IVF-style index directly over multi-vector data.
We introduce MV-IVF, an inverted-file index for multi-vector retrieval built on MV-Lloyd, a new Lloyd-style algorithm for clustering point clouds under a Chamfer objective. Our key observation is that once point clouds are assigned to clusters, the center-update step reduces to standard single-vector clustering over the flattened embeddings in each cluster. This enables efficient construction of a hierarchical IVF tree with center point clouds at internal nodes.
At query time, MV-IVF performs best-first beam search over the tree to identify promising leaf clusters, then scores the documents in those leaves. We further accelerate search using low-bit quantization, query compression, and cache-efficient batched search. We implement MV-IVF in MVSIC, an optimized C++ library and unified evaluation framework for multi-vector retrieval that includes several major approaches. Across BEIR and LoTTE text retrieval benchmarks, MV-IVF achieves strong latency--recall tradeoffs, the best batched throughput frontiers, and favorable build-time and index-size profiles. At $90\%$ Recall-$10$@$10$, MV-IVF achieves up to $8.03\times$ lower single-query latency and up to $173.5\times$ higher QPS than the baselines, demonstrating that native IVF-style indexing is practical and effective for scalable multi-vector retrieval. 

---
# Rethinking Latent Reasoning for Generative Recommendation: Beyond Teacher-Generated CoT and Recurrent Hidden-State Feedback 

**Authors**: Yiwen Chen, Fuwei Zhang, Zehao Chen, Deqing Wang, Hehan Li, Peizhi Xu, Hanmeng Liu, Shuanglong Li, Xin Pei, Fuzhen Zhuang, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.09006)  

**Abstract**: Latent reasoning in generative recommendation faces two practical barriers: constructing teacher-generated chain-of-thought (CoT) supervision is costly, and recurrent hidden-state feedback requires a specialized execution loop. We revisit both the supervision source and the computation interface. A step-level diagnosis measures how recommendation CoT supports next-item prediction and trace coherence. In the analyzed traces, history analysis is the only step with positive mean contributions on both dimensions, while later preference and intent steps have negative mean answer contributions. Guided by this finding, we propose PHiT (Prefilled, History-Informed Thought Tokens), which learns intermediate states from semantic targets extracted directly from observed item histories. PHiT aggregates cached sparse item encodings to construct these targets and supervises the contextual states of distinct vocabulary thought tokens. All thought tokens are supplied together as ordinary input IDs during a single causal prefill, removing the need for recurrent final-state feedback. With supervised fine-tuning alone, PHiT outperforms the compared baselines across four recommendation benchmarks. Controlled ablations support the value of input-matched semantic supervision, while cost measurements show fast target construction with training time comparable to the recommendation-only thought-token baseline. 

---
# SkinMind: Equipping Agentic RAG with Human-Inspired Graph Probing for Reliable Dermatology AI 

**Authors**: Jing Hao, Weinan Zhou, Zhihong Lu, Yaling Shen, Yiming Xu, Yefeng Liu, Siyuan Yan, Shuaibo Li, Ming Dai, Junjun He, Zuyi Zhu, Jinjie Gu, Lijun Wu, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08935)  

**Abstract**: Reliable dermatology AI requires access to verifiable evidence distributed across heterogeneous textual and visual sources. However, conventional retrieval-augmented generation (RAG) typically relies on one-shot retrieval from flat corpora, limiting its ability to adapt retrieval granularity, explore multi-hop relations, and integrate multimodal clinical evidence. We introduce SkinMind, a multimodal agentic RAG framework that equips policy models with human-inspired graph probing for autonomous evidence seeking. SkinMind organizes retrieval as an iterative interaction between an orchestrator and a graph environment. Through eight hierarchical graph-probing tools, the orchestrator progressively identifies relevant entities, inspects node-level information, traverses relations and paths, and retrieves source-text and visual evidence. This progressive-disclosure design allows the model to adaptively determine the retrieval modality, exploration direction, and evidence granularity while constructing a query-specific evidence subgraph. To support autonomous graph probing, we construct SkinMind-KG from a 62M-token corpus of dermatology guidelines and textbooks and 423K images, guided by a dermatologist-informed schema and a conflict-resolution mechanism. The resulting graph contains over 1.0M nodes and 2.0M edges, unifying textual, relational, and visual evidence for retrieval. Across six dermatology VQA benchmarks and two policy-model settings, SkinMind improves the average score over direct answering by 8.70 and 8.89 points with GPT-5.4-mini and Gemini-2.5-Flash, respectively, while consistently outperforming competing RAG methods. Together, SkinMind and SkinMind-KG provide an effective foundation for evidence-grounded multimodal reasoning in dermatology. 

---
# Generative Search and Recommendation at Shopify 

**Authors**: Juhana Kangaspunta, Khalil Hajji, Darshan Fofadiya, Kyumin Park, Michael Garner, Anthony Maltsev, Ryan Synk, Jose Sepulveda, Joshua Crisologo, Shuying Sun, Akshay Soni  

**Link**: [PDF](https://arxiv.org/pdf/2611.08804)  

**Abstract**: Search and recommendation at Shopify require retrieving relevant products from a changing catalog of billions of products under strict latency constraints. We present SID-LLM, which casts both tasks as generating semantic IDs: short sequences of content-derived tokens representing products. We jointly pre-train SID-LLM on catalog and behavioral data, giving search and recommendation a shared representation and training objective. We show that reinforcement learning further improves judged search relevance. Furthermore, distillation from task-specific experts improves recommendation quality over joint supervised fine-tuning without a statistically detected loss in search quality. Finally, for efficient serving, we pair multi-token prediction with a lightweight ranker, improving retrieval quality over our autoregressive baseline while reducing latency. We evaluate SID-LLM both as a standalone retriever and as a component of Shopify's production search and recommendation systems. As a standalone search system, SID-LLM achieves quality comparable to Shopify's existing multi-model search stack while meeting production latency requirements. Integrating SID-LLM into the search and recommendation stacks provides significant gains in both offline evaluations and online A/B tests. Together, these results demonstrate the practical value of semantic-ID generation as a unified approach to product discovery at scale. 

---
# SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery 

**Authors**: Sunkyung Lee, Xiaokai Wei, Da Long, Hyunsik Jeon, Jiajun Wu, Rithwik Seth, Jongwuk Lee, Michelle Gong, Gungor Polatkan  

**Link**: [PDF](https://arxiv.org/pdf/2611.08790)  

**Abstract**: Generative recommendation aims to predict item identifiers from user interaction histories, but explicit control over the properties of recommended items remains underexplored. A key challenge is to connect human-readable concepts to internal features whose interventions effectively change the recommended items. To address this, we propose Steerable Generative Recommender (SteerGR), a framework that uses sparse autoencoders for concept-level control in game discovery. SteerGR proceeds through three stages: (i) Concept Definition and Grounding for defining play concepts with an LLM; (ii) Feature Discovery and Selection for discovering candidate features by their concept associations and selecting one by its steering effect; and (iii) Sparse Feature Intervention for amplifying or suppressing selected features to steer recommendations without retraining the recommender. Extensive experiments with two industrial generative recommendation models on a real-world game dataset show that the selected features successfully steer recommendations toward or away from target concepts. We believe this work takes a first step to extend SAE-based steering from LLMs to industrial generative recommendation. 

---
# CascadeRec: Deployed Adaptive LLM Compute Allocation for Multi-Tenant Content Recommendation 

**Authors**: Kai Zhao, Guilin Zhang, Xiang Cheng, Xiquan Cui, Henry Heng, Xu Chu, Alet Blanken  

**Link**: [PDF](https://arxiv.org/pdf/2611.08566)  

**Abstract**: Large language models (LLMs) are attractive rerankers for recommendation. Running an expensive model on every request is costly, and for users with sparse interaction histories it often lowers quality. We present CascadeRec, a two-tier LLM reranking cascade deployed in production at a large financial planning and analysis (FP&A) SaaS provider. It serves content recommendations to enterprise tenants on AWS EKS and GKE. A lightweight Tier-1 model reranks every request. A router escalates only the requests where extra inference-time compute is likely to help, and sends them to a more expensive Tier-2 procedure (self-consistency sampling, multi-path reasoning, or a larger model). The system has run in a production A/B experiment for over a month. Compared with uniform use of the expensive tier, CascadeRec has reduced inference cost by about 10%, with minimal latency impact and no quality regression observed to date. Tenant data cannot be released, so we also evaluate the pipeline on public benchmarks (MovieLens-1M and Amazon-Electronics). Segment-level analysis with paired-bootstrap confidence intervals shows that uniform LLM reranking lowers NDCG@10 for sparse-history users by up to 30%, and its only statistically significant gain appears on the segment with the longest histories. Escalating cold users can make the loss larger instead of repairing it. Which segments benefit from escalation depends on the domain, and this motivates the segment-gated rollout we use in production. We describe the deployment architecture, the measured benefits, and lessons learned, including inter-tenant load imbalance as a driver for cascading and the handling of LLM output-parsing failures at the serving layer. 

---
# RedMall: Generative Recommendation for Large-Scale E-commerce 

**Authors**: Kunyang Zhou, Xin Jia, Yi Yang, Yongjun Yang, Manjie Xu, Honglei Zhang, Yan Zhu, Kai Zuo, Xu Tang, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.08519)  

**Abstract**: Generative recommendation has shown considerable potential in industrial recommendation systems. However, the performance of existing generative methods is limited by several challenges. 1) Semantic IDs (SIDs) are treated as one-dimensional sequences, overlooking the hierarchical dependencies among different layers. 2) Next-token prediction used is unable to explicitly optimize session-wise recommendation. 3) Although reinforcement learning has been applied to align generated recommendations with user preferences, few existing methods explicitly optimize the ranking of positive items in session-wise recommendation. To address the above issues, we propose RedMall, a generative recommender designed for large-scale e-commerce. Firstly, RedMall constructs hierarchical SIDs through residual quantization over multimodal representations. Secondly, RedMall introduces a lightweight SID encoder to explicitly model layer-wise dependencies in SIDs. Moreover, RedMall incorporates sparse mixture-of-experts, dual gated attention, and multi-target prediction to scale model size, enhance user interest representation, and optimize session-wise recommendation. We further design an exposure-aware loss that treats exposed but unclicked items as negative feedback. To better align user preferences and list-wise ranking objectives, we combine Group Relative Policy Optimization with an auxiliary LambdaRank loss, using both real user feedback and item ranking to guide optimization. Experiments on four public datasets-Amazon Beauty, Amazon Toys, Amazon Food, and MovieLens-1M---show that RedMall outperforms the evaluated baselines across all reported Recall and NDCG metrics. Online A/B tests show improvements of 2.24% in advertiser value (ADVV) and 0.59% in click-through rate (CTR) for e-commerce advertising, as well as gains of 1.08% in GMV and 0.95% in daily active buyers (DAB) for e-commerce recommendations. 

---
# Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies 

**Authors**: Sakshi Garg, William B. Andreopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.08498)  

**Abstract**: The vast amount of information available in lengthy, distributed documents makes finding relevant answers difficult, especially when the answer requires connecting evidence from multiple sources. Standard Retrieval-Augmented Generation (RAG) systems often struggle with this type of "multi-hop" reasoning and frequently face issues with context fragmentation. In this paper, we propose a hybrid RAG framework that integrates the structured reasoning of knowledge graphs with the broad semantic coverage of dense vector retrieval. Our approach employs a multi-stage pipeline using transformer-based summarization and semantic chunking to improve context retention. We construct a dynamic knowledge graph to capture explicit relationships between entities while simultaneously maintaining a dense vector index. During inference, the system uses a hierarchical mechanism that prioritizes graph traversal for precision and falls back to vector retrieval when necessary. We evaluated our framework on the MultiHop-RAG dataset. The results show that this hybrid approach outperforms standalone vector and graph-based baselines, achieving higher Exact Match scores and improved retrieval accuracy for complex queries. This confirms that merging structured reasoning with standard vector retrieval creates a more robust system for handling complex knowledge-intensive questions. 

---
# ENGRAM: From Agent Experience to Weights through Verified Consolidation 

**Authors**: Ruiyi Yang, Zechen Li, Lihuan Li, Du Yin, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.09472)  

**Abstract**: Language-model agents use experience through external retrieval and parameter learning. Consolidating stored experience into model weights requires deciding both which experience to train on and when its external records can leave retrieval. Admission based solely on student mastery can exclude useful teacher demonstrations, while removal immediately after training can discard guidance that the model still needs. We introduce ENGRAM, a framework that separates training admission from memory release. A demonstration-aware gate admits skills based on student mastery or sufficient successful teacher demonstrations. The student learns from selected action traces with context from an Engram Trace Graph (ETG). A verification gate then compares the updated model without episodic retrieval against the base model with retrieval on reserved probes. Release requires valid training participation and scores meeting both relative and absolute criteria; records for skills that fail verification remain retrievable. Experiments across interactive and reasoning benchmarks show consistent gains over frozen-weight ETG controls, with ablations supporting demonstration-aware admission and selective adapter activation. A controlled held-out comparison shows that verified release substantially reduces the retrievable footprint while remaining close to retain-all performance and outperforming immediate and budget-matched random release. ENGRAM also improves task performance while reducing input-token usage and model calls. Skill-level verification further reduces retrievable experience through selective archival, retaining external guidance for skills that do not meet the release criteria. Code is available in this http URL. 

---
# FondueQA: A Benchmark for Evaluating Parametric Compensation in Multi-Hop QA 

**Authors**: Laurin Schmid, Suman Navaratnarajah, Koel Dutta Chowdhury, Steffen Eger, Josif Grabocka, Christian Frey  

**Link**: [PDF](https://arxiv.org/pdf/2611.09274)  

**Abstract**: Large Language Models (LLMs) show remarkable performance based on parametric knowledge, i.e., their ability to recall facts and associations internalized in a model's weights. However, studying this capability on established multi-hop benchmarks may be complicated by benchmark contamination, as many of these benchmarks have been publicly available long enough to plausibly appear in modern pretraining corpora. As a result, correct answers under missing evidence may reflect either general parametric knowledge or memorization of benchmark-specific instances. To study how models use internal knowledge when retrieval evidence is incomplete, we introduce FondueQA, in which every required multi-hop fact is uniquely localized to a designated evidence document. This enables controlled document-level evidence ablations to study parametric compensation under incomplete retrieval. Our empirical evaluation reveals that parametric compensation strongly depends on both entity popularity and the structural role of missing evidence. Generally, models compensate more successfully for facts involving popular entities. On FondueQA, we show that bridging evidence can be more useful than the answer-bearing document itself. Moreover, models may recall a missing fact in isolation but fail to use it under partial evidence, particularly when the context contains plausible competing entities, which can reduce Exact Match by up to 15 percentage points. 

---
# Large Language Model Consensus and Corpus-Adaptive Topic Modeling for Classifying and Mapping Patent Innovation 

**Authors**: Sercan Ozcan, Enes Ozgozler, C. Okan Sakar  

**Link**: [PDF](https://arxiv.org/pdf/2611.09194)  

**Abstract**: Patent analytics increasingly relies on transformer representations and large language models (LLMs), yet linking conceptual classification, topic formation and interpretation requires explicit decisions about model roles and the evidence retained at each stage. In this design-oriented study, we propose and demonstrate a patent analytics pipeline that links these tasks and keeps each decision open to audit. Three LLMs separately classify each patent by novelty (radical or incremental) and innovation type (product, process, service). Radical and innovation-type labels each require two supporting votes, with Incremental used as the fallback novelty label. Topics are discovered by clustering PatentSBERTa embeddings, with parameter ranges scaled to corpus size. A patent-specific stopword list is applied only when topics are represented with class-based term frequency--inverse document frequency (c-TF-IDF), and a second embedding model supports inter-topic similarity, hierarchical refinement and visualization. An LLM then labels each topic from its ranked keywords and representative patents. We demonstrate the pipeline on 30,964 public-sector-related patents retrieved from this http URL. Across five category corpora ranging from 591 to 29,215 patents, the pipeline produced 74 category-specific topic instances with a macro-average document coverage of 98.1% and topic diversity of 0.900--1.000, and fewer than 1% of classification outputs returned UNKNOWN. Because every model vote and the evidence behind each topic label are retained, analysts can audit and revise the outputs. The demonstration establishes the feasibility and traceability of the pipeline; testing its labels against expert coding and benchmarking it against alternative pipelines are identified as the next steps. The pipeline offers a reusable basis for LLM-assisted patent analysis in technology intelligence. 

---
# A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring 

**Authors**: Warren Jouanneau, Morgane Hoffmann, Emma Jouffroy, Marc Palyart  

**Link**: [PDF](https://arxiv.org/pdf/2611.09178)  

**Abstract**: While machine learning hiring tools are ubiquitous, the need for privacy regulations and commercial sensitivity makes public benchmarking legally complex. We address this challenge by introducing a taxonomy-driven, fully synthetic HR dataset and benchmark to simultaneously evaluate retrieval effectiveness and algorithmic fairness. Grounded in 660K real freelancer profiles, our multilingual corpus features candidate profiles, project briefs, and counterfactual demographic twins. Auditing state-of-the-art embedding models reveals performance divergence from general MTEB scores, alongside fairness disparities in grammatically gendered languages. This demonstrates that general benchmarks are insufficient for HR, requiring tailored evaluations to identify Pareto-optimal models that balance semantic accuracy and fairness. 

---
