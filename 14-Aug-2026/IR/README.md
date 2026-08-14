# Structure then Query: Enabling Precise Analytical Queries over Unstructured Documents 

**Authors**: Teng Lin, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13384)  

**Abstract**: Unstructured documents constitute the majority of enterprise and web data. With the rapid development of large language models(LLMs), researchers have started to build data systems that analyze unstructured textual documents like operating on databases. However, because mainstream retrieval methods still relies on fuzzy matching based on vector similarity, accurately obtaining information and performing structured analysis and reasoning remains a major challenge. To address these limitations, AnnoIndex introduces two core fundamental components. The first is Annotation Index. The system uses a module called SchemaLoop to automatically create hierarchical annotation schemas from the raw corpus, and then uses lightweight language model to extract specific values. It turns scattered unstructured text into a materialized, structured index that enables low-cost filtering and querying. The annotation index avoids the black-box matching of vector similarity and amortizes attribute extraction costs from online queries to a one-time build. The second innovation is a Structured Query Engine. It compiles user questions into execution plans based on SQL extension. It first uses the Annotation Index for precise documents filtering, then gradually applies extraction operations in ascending order of cost, resorting to LLMs only for the remaining minimal fraction of the corpus that require deep semantic understanding. The extracted attributions are merged into the annotation index, reducing the cost of future queries. Experiments on three real-world datasets demonstrate that AnnoIndex consistently outperforms state-of-the-art baselines, achieving the highest average F1 score (0.87) while maintaining robust performance on complex multi-hop join and progressive reasoning queries. 

---
# When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1 

**Authors**: Weimeng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.13237)  

**Abstract**: Multi-round retrieval-augmented generation (RAG) must decide when to stop searching as evidence accumulates. Because the deployed policy is determined by the first STOP on each trajectory, this is a sequential selection problem rather than an independent state-classification task. We adapt S2G-RAG's structured sufficiency-and-gap judgment to a frozen Search-R1 pipeline and train a Qwen3.5-2B judge on 3,009 states from 900 disjoint HotpotQA questions. Search-R1's reasoner, retriever, corpus, prompt, and search budget remain unchanged, while the judge checkpoint and stopping threshold are selected on grouped validation and frozen before confirmatory evaluation. On the confirmatory test set, the resulting policy reduces retrieval calls by 77 (3.70\%) relative to Native Search-R1, while Official Exact Match decreases by 0.625 percentage points. Thus, the trained S2G-style structured judge reduces retrieval while broadly preserving answer accuracy. The result does not imply unchanged or improved accuracy, safe stopping, or lower total inference cost. 

---
# Generative Universal Multimodal Retrieval with Dual-role Identifiers 

**Authors**: Kaipeng Li, Haitao Yu, Xuanchen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.12987)  

**Abstract**: Generative information retrieval (GIR) has emerged as a compelling alternative to the conventional index-retrieve-then-rank retrieval pipeline by training a generator to produce the identifiers of relevant items directly. Despite its promise, a number of open challenges still remain. First, constrained left-to-right decoding is vulnerable to prefix-level errors and local optima. Second, most prior GIR research remains largely unimodal, leaving instruction-aware retrieval across text, image, and mixed image-text items underexplored. Third, although discrete identifier-based GIR offers higher efficiency, its retrieval accuracy still lags behind that of the cutting-edge dense-vector-based retrieval methods. Motivated by these challenges, we propose DrIG, a novel Generative framework for universal multimodal retrieval featuring Dual-role Identifiers, which supports diverse retrieval tasks across multiple modalities and domains. Each candidate is assigned a single residual-quantized identifier that serves two complementary roles. In its sequential role, the identifier is decoded autoregressively, where the first token explicitly models modality and the remaining tokens capture progressively finer semantics. In its set-based role, the same tokens are reinterpreted as an unordered set to provide a prefix-independent relevance prior, which guides constrained beam search and alleviates local-optimum errors. Extensive experiments on the M-BEIR benchmark and the text-to-image evaluation datasets show that:(1)DrIG consistently outperforms state-of-the-art generative multimodal baselines across diverse tasks, while hybrid reranking achieves a favorable efficiency-effectiveness trade-off against strong dense retrievers. (2)Ablation and scaling analyses reveal how the base LMM, beam size, reranking depth, and fusion strategy affect retrieval performance, providing practical guidance for system design. 

---
# STAR: Structured Tokenization and Target-Aware Interest Representation for PCVR Prediction 

**Authors**: Yimeng Xu, Haorui Zhang, Yingqi Song, Ying Jiang, Lan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.12986)  

**Abstract**: Post-click conversion rate (PCVR) prediction is a core ranking task in industrial recommender systems. Modern ranking models must jointly capture heterogeneous non-sequential features, multi-behavior user sequences, and target-item-aware user interests, while remaining robust to high-cardinality sparse features, missing values, and train-inference inconsistencies. In this paper, we present STAR (Structured Tokenization and Target-Aware Interest Representation), a practical framework for the KDD Cup 2026 Tencent UniRec Challenge. STAR combines structured feature tokenization with target-aware interest representation on top of a HyFormer-style multi-sequence backbone. It introduces high-cardinality signal recovery, explicit user-item interaction tokens, target-aware sequence decoding, and a weighted user-item contrastive auxiliary objective inspired by InfoNCE. We further align the training and inference pipelines by reconstructing feature remapping tables and structural hyperparameters from the saved training configuration. Experiments on the challenge dataset identify the components that most reliably improve ranking AUC, while LogLoss is reported as a calibration diagnostic. The main ablation study shows a large gain from temporal context, with smaller but useful contributions from contrastive alignment, target-aware interest encoding, and high-cardinality sequence feature recovery. 

---
# FSGR: Mitigating Token Frequency Bias for Fair SID-Based Generative Recommendation 

**Authors**: Yuchen Zheng, Sihan Xu, Jingwen Yang, Xiangrui Cai, Haiwei Zhang, Xiaojie Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12845)  

**Abstract**: Semantic ID (SID)-based generative recommendation has recently achieved remarkable success. However, existing methods suffer from a previously overlooked fairness issue, which we term \textbf{Token Frequency Bias}, where high-frequency SID tokens are systematically over-predicted while low-frequency SID tokens are under-predicted. This bias originates from the combined effects of imbalanced semantic codebooks during SID construction, and popularity bias together with the maximum likelihood estimation objective during recommendation training, resulting in unfair exposure across item categories. Existing SID methods mainly focus on improving codebook quality and overlook the impact of token frequency imbalance on downstream recommendation fairness, while LLM debiasing methods often yield suboptimal results when directly applied to SID-based recommendation, due to the hierarchical semantics of SID tokens. To address this issue, we propose \textbf{FSGR}, a fairness optimization framework for SID-based generative recommendation. During SID construction, FSGR employs OT-based Assignment Optimization and Dual-Criteria Re-anchor mechanism to form a more balanced SID representation space. During recommendation training, it adopts a two-stage training strategy and introduces Hierarchical Frequency Calibration for layer-specific fairness fine-tuning. Experiments on three public datasets with three backbone models demonstrate that FSGR mitigates token frequency bias and delivers an average Gini fairness improvement of over 20\% while maintaining competitive recommendation accuracy. 

---
# Query Translation vs. Cross-Lingual Embeddings for Sinhala-Tamil E-Government Information Retrieval 

**Authors**: Dharshi Balasubramaniyam, Tiroshan Madushanka  

**Link**: [PDF](https://arxiv.org/pdf/2608.12820)  

**Abstract**: This paper presents a comparative evaluation of cross-lingual information retrieval (CLIR) methods for retrieving English government information using Sinhala and Tamil queries. Two CLIR paradigms are investigated: Query Translation (QT), employing Google Translate, NLLB, and mBART50, and Cross-Lingual Embeddings (CLE), using LaBSE, multilingual E5, and BGE-M3, with monolingual English retrieval as the baseline. Experiments are conducted on a human-verified benchmark comprising 500 Sinhala, Tamil, and English question-answer pairs derived from 1,699 segmented contexts from Sri Lanka's Government Information Center (GIC). Retrieval performance is evaluated using Recall@k (k = 1, 3, 5, 10, 15). Monolingual retrieval performs poorly (Recall@15 <10%), whereas all CLIR approaches substantially improve retrieval accuracy. Among them, BGE-M3 achieves the highest Recall@15, reaching 96.2% for Sinhala-English and 95.6% for Tamil-English, outperforming the best QT approach (Google Translate: 92.4% and 93.0%) while avoiding translation overhead. These results demonstrate that multilingual embedding models provide a more effective and scalable solution for cross-lingual retrieval-augmented generation (RAG) in low-resource government domains. 

---
# A Comprehensive Empirical Evaluation of Vector Database Systems for Approximate Nearest Neighbor Search: Performance, Quality, and Resource Trade-offs 

**Authors**: Ashen Rashmiks, Tiroshan Madushanka  

**Link**: [PDF](https://arxiv.org/pdf/2608.12812)  

**Abstract**: Vector databases have emerged as critical infrastructure for modern artificial intelligence applications, particularly retrieval-augmented generation (RAG), semantic search, and recommendation systems. Despite their growing importance, there remains a significant gap in comprehensive, reproducible benchmarks that jointly evaluate retrieval quality, query latency, throughput, and resource utilization. We present a systematic empirical evaluation of seven prominent vector database systems: FAISS, Qdrant, Milvus, Weaviate, Chroma, pgvector, and LanceDB. Our methodology spans six diverse datasets, from classical computer-vision descriptors (SIFT, GIST) to transformer-based text embeddings (MS MARCO, GloVe), encompassing over 4 million vectors at dimensionalities from 96 to 960. We measure 15 metrics spanning retrieval quality (Recall@K, Precision@K, MRR, NDCG@K, Hit Rate@K), query performance (latency percentiles, QPS, cold-start latency), and resource consumption (index build time, memory, storage). On SIFT1M, FAISS achieves the highest single-node throughput (866 QPS) but lacks database operational features; Weaviate provides the best out-of-the-box recall (> 99%); Qdrant offers the best latency among full databases (4.55~ms median); and LanceDB trades retrieval quality for substantially faster index construction. We derive system-selection guidelines for practitioners and release our benchmarking framework as open-source software. 

---
# DrEM: Dual-Side Robust Ensemble Ranking from Noisy User Preference Predictions in Video Recommendation 

**Authors**: Canwei Huang, Tiantian He, Xiaoxiao Xu, Jun Zhang, Ziran Deng, Weike Pan, Chunjie Chen, Kaiqiao Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12778)  

**Abstract**: Industrial video recommendation systems typically adopt a multi-stage architecture. At the ensemble ranking stage, multi-dimensional user preference predictions (pxtrs) from an upstream multi-task model are fused into a unified ranking score to reflect user satisfaction. Since users' true satisfaction is difficult to observe directly, ensemble ranking models commonly use pxtrs both as input features and as a source for constructing proxy preferences. However, as outputs of an upstream prediction model, pxtrs inevitably contain prediction noise, which propagates to downstream learning across two sides. On the supervision side, noisy pxtrs may flip proxy preferences and introduce erroneous gradients. On the feature side, pxtr noise may propagate through model inputs and destabilize ranking scores. Existing ensemble ranking methods typically treat pxtrs as reliable signals and overlook such prediction noise. To address this, we propose DrEM, a dual-side robust ensemble ranking framework. Our DrEM introduces a risk-denoising robust loss that corrects the empirical risk using estimated preference flip probability. Meanwhile, it samples perturbations from the distribution of prediction noise and introduces a preference-preserving ranking consistency regularizer to improve feature-side output stability. Theoretically, we obtain an approximate distribution of the prediction noise and prove that the robust loss remains superior under flip probability estimation error. Extensive offline experiments and large-scale online A/B tests demonstrate the effectiveness and robustness of our DrEM. 

---
# Knowledge Synthesis Review Framework: Task-Level Benchmarking of LLM-Based Systems for Multi-Source Evidence Synthesis 

**Authors**: Wafa Shafqat, Mark Patterson, Steven N. Liss  

**Link**: [PDF](https://arxiv.org/pdf/2608.12741)  

**Abstract**: Evidence in rapidly evolving fields is fragmented across academic studies, industry reports, policy documents, and media sources that differ in quality, structure, and purpose, making timely synthesis difficult. Large language models (LLMs) may accelerate this work, but their reliability across the distinct cognitive tasks of a review remains uncertain. We introduce the Knowledge Synthesis Review (KSR), a human-in-the-loop framework that decomposes evidence synthesis into screening, extraction, analysis, and synthesis, benchmarks LLM-based systems on each task against expert reference standards, and routes each task to the best-performing system under continuous expert validation. We evaluated GPT-5, Claude Sonnet 4, Gemini 2.5 Pro, and NotebookLM on a 244-document benchmark subset drawn from a 1,893-document corpus on AI and work spanning four source types, against a gold standard with high inter-rater reliability (92.2% agreement, kappa = 0.80). No system led on all tasks. Claude Sonnet 4 achieved the highest screening accuracy (82.8%) and GPT-5 the highest recall (91.8%) at the expense of lower specificity. Extraction exceeded 90% agreement for titles and sources but degraded in author and reference fields. Performance declined most in interpretive analysis and cross-source synthesis, where expert judgment remained essential. A contamination check on post-cutoff documents showed no evidence that prior exposure inflated results. Applied to the full corpus, the routed workflow surfaced cross-source asymmetries and blind spots that single-source synthesis would miss, including worker well-being, small firms, and the Global South. KSR offers a transparent, auditable, model-agnostic framework for governing LLM assistance in research synthesis while preserving human accountability. 

---
# Test-Time Optimization of Query Embeddings with Ranking Aware Reward Maximization 

**Authors**: Tianyu Chen, Jiaxing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12569)  

**Abstract**: Dense retrievers rank documents using vector similarity between a frozen encoder and a precomputed index. While test-time ranking rewards from a reranker or LLM judge can improve results, existing methods discard this signal after a single query. Updating the retriever's weights makes rewards reusable, but this requires parameter access, which is unavailable for closed-source models, and is computationally prohibitive. We propose TTT-Embed (Test-Time Tuning of Embeddings), a framework that distills ranking rewards into a lightweight, learned vector within the output embedding space of a frozen model. This vector is optimized purely from scalar ranking scores assigned to the retriever's own candidate documents, requiring no access to model weights, ground-truth labels, or modifications to index. A single scope parameter controls rewards reuse (global, task, or query), enabling a principled trade-off between reusability and specificity under a fixed reward computation budget. We demonstrate that as the available reward budget scales, the optimal sharing scope shifts dynamically from global-wise to task-wise and finally to query-wise. Evaluated across five embedding models and 15 MTEB retrieval tasks, TTT-Embed improves test-time retrieval by up to +8.36 nDCG@10. Crucially, the learned states generalize effectively to unseen queries (up to +8.57 nDCG@10) and unseen tasks (up to +4.71 nDCG@10). Furthermore, TTT-Embed successfully resolves catastrophic forgetting: by leaving base weights entirely frozen, it recovers degraded general capabilities (up to +8.00 nDCG@10, even surpassing the original base model) while preserving in-domain specialization. These results establish ranking rewards as a reusable test-time state, enabling budget-efficient adaptation for any embedding model, including closed-source APIs. 

---
# GEM: A Generative Embedding Model Bridging Reasoning and Retrieval 

**Authors**: Zhili Shen, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2608.13200)  

**Abstract**: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's generative nature allows test-time compute scaling via prompting to further enhance retrieval performance. Our code is available at: this https URL. 

---
# RAGSieve: Self-Referenced Local Contrast for Knowledge-Poison Detection in Retrieval-Augmented Generation 

**Authors**: Xinlong Xu, Yoshua Y. Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13010)  

**Abstract**: Retrieval-augmented generation treats an external corpus as inference evidence, allowing injected documents to promote attacker-chosen claims. Existing detectors depend on trusted references, specific attack artifacts, or global thresholds sensitive to corpus topology. We present RAGSieve, a self-referenced detection framework that constructs its reference from the inspected system. RAGSieve-Query (RSQ) performs query-local contrast, scoring top-five candidates against ranks 6-20 of the same retrieval to detect answer-anchor concentration and carrier transitions. RAGSieve-Graph (RSG) performs corpus-local contrast, comparing each document's semantically similar but lexically distinct neighbors with its local baseline to detect coordinated density before queries arrive. Across three QA datasets and six poisoning constructions, RSQ achieves 95.2% AUROC and detects 82.2% of poison at 5% clean-document removal, versus 81.1%/52.5% for GMTP. RSG achieves 93.3%/79.8%, versus 79.4%/37.6% for CleanBase. Joint deployment reduces attack success from 67.4% to 14.0% while retaining 41.3% F1 on unpoisoned retrieval, demonstrating practical protection at both corpus ingestion and query time without poison labels or trusted corpora. Source code is available at this https URL. 

---
# EviReform: Evidence-Guided Query Reformulation for Multi-Hop Graph Retrieval 

**Authors**: Xinlong Xu, Yoshua Y. Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13006)  

**Abstract**: Multi-hop retrieval must recover passages that provide sufficient evidence together. An initial passage often resolves an entity or relation implicit in the question, making the missing evidence easier to describe only after retrieval begins. Graph retrieval improves access to related evidence through stored corpus structure, but its retrieval signal is commonly derived from the original question. Complementary evidence must then be reached through stored relations even when an observed passage provides a more direct semantic cue. We introduce EviReform, which separates revising the retrieval request from aggregating evidence in the graph. Retrieved source passages formulate residual queries for the unresolved information need. The original and residual retrieval signals are normalized separately, combined, and propagated between propositions that share entities. On 2WikiMultiHopQA, HotpotQA, and MuSiQue, EviReform exceeds the strongest baseline by up to 5.59 Recall@5 points and 4.50 F1 points. These results show that observed evidence can guide graph retrieval toward the part of a supporting chain left underspecified by the original question. Code is available at this https URL. 

---
# HybridRAG-BN: A Retrieval-Augmented Framework with Fine-Tuned Verification for Bangla KBQA 

**Authors**: Rathijit Aich, Nirjhar Das, Mahfuzulhoq Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.13004)  

**Abstract**: Knowledge-base question answering (KBQA) systems rely on effective retrieval and reasoning mechanisms to generate accurate answers from external knowledge sources. However, developing reliable KBQA systems for low-resource languages such as Bangla remains challenging due to limited retrieval-focused research, scarce language resources, and difficulties in grounding generated responses in external knowledge. In this work, we propose HybridRAG-BN, a retrieval-augmented framework for Bangla KBQA that integrates hybrid retrieval using BM25 and BGE-M3, answer generation using the GGUF version of Gemma-4-31B-Instruct, and a LoRA-fine-tuned Gemma-4-31B-Instruct model for answer verification and refinement. To further improve robustness, the framework incorporates a post-processing stage that addresses unresolved cases through fallback answer replacement and DuckDuckGo-assisted retrieval. Experimental results demonstrate the effectiveness of the proposed framework, achieving token-level F1 scores of 0.71654 and 0.72912 on the public and private leaderboards, respectively, securing first place in the competition. 

---
# DTAMLP: Denoise Time-aware MLP for Session-based Recommendation 

**Authors**: Jiamu Zheng, Xiaojun Shan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12975)  

**Abstract**: This paper reports two empirical findings on session-based recommendation (SBR), unified in a single model, DTAMLP. First, existing time-aware and GNN-based models (e.g., TiSASRec, SR-GNN) treat every click-time interval as equally informative, even though very short dwell times often reflect accidental clicks carrying little preference signal -- a phenomenon we call sporadic noise. We show that a lightweight, plug-and-play weight fusion module, blending a model's attention weight with a threshold-capped time-interval weight, can be inserted into such models with almost no architectural change and yields a consistent accuracy gain; we view this as the most directly verifiable contribution of this work. Second, we revisit an under-explained observation from FMLP-Rec, where a learnable frequency-domain filter on item embeddings improves accuracy, and offer a possible explanation: time-domain behavior mixes several entangled psychological preferences, and a frequency-domain view may let a model separate and down-weight such preference noise more naturally -- an interpretive conjecture rather than a proven mechanism. Building on both insights, DTAMLP, an all-MLP framework combining weight fusion and FFT-based filtering, is validated on Diginetica and RetailRocket. While this system-level design reflects the state of the field circa 2023 rather than a state-of-the-art claim, ablations confirm the two mechanisms contribute complementary, non-redundant improvements. 

---
# CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives 

**Authors**: Chengyang He, Tahreem Arif, Marko Zivkovic, Lijing Wang, Yue Ning, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12779)  

**Abstract**: Understanding the temporal progression of symptoms in clinical narratives is critical for disease monitoring, safety surveillance, and causality assessment. Clinical narratives, however, rarely provide explicit temporal anchors. Current approaches to temporal information reasoning focus predominantly on pairwise relation classification across multi-visit and timestamp-rich records, leaving the reconstruction of structured symptom trajectories from individual anchor-sparse reports largely unaddressed. We propose CRAFT, an LLM framework that pairs a generator with a constraint-based verifier to iteratively produce and refine stage-wise symptom timelines through targeted feedback. We conduct evaluation on MedTempo, a new benchmark of 5,347 vaccine adverse-event narratives spanning three COVID-19 vaccine types, with expert-validated temporal stage annotations for 3,166 reports. Experiments across four LLM backbones demonstrate that CRAFT consistently improves temporal ordering accuracy, with ablation analysis isolating the contribution of generator and verifier components across model capability levels. 

---
# Attribute-Conditioned Multimodal Slot Factorization for Controllable Fashion Retrieval 

**Authors**: Najmeh Forouzandehmehr, Topojoy Biswas, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12570)  

**Abstract**: Fashion retrieval often requires satisfying multiple attributes at once, such as category, color, pattern, and demographic. Monolithic embeddings mix these signals into a single vector, making attribute-specific control difficult at retrieval time. Many existing semantic-ID methods provide discrete item codes, but these codes are typically optimized as item-level or residual addresses and do not expose named, independently controllable attribute slots.
We introduce MM-slotgate, a multimodal slot encoder that factorizes Fashion-CLIP text and image embeddings into four named attribute slots. Each slot learns its own text-image gate, so visually grounded attributes such as color and pattern can rely more on image evidence, while taxonomy-oriented attributes such as category and demographic can remain more text-driven.
On H&M, using a combined slot-similarity and slot-logit retrieval score, MM-slotgate achieves 0.7566 macro ConstraintSatisfied@10, outperforming equal-weight multimodal fusion (0.7142) and fCLIP text-only retrieval (0.4755). The largest gain is on color, which improves from 0.321 to 0.889 (+0.568 absolute), as the learned color gate assigns 57.4% weight to image evidence. The learned gates are interpretable without modality supervision: color is image-leaning, category is text-leaning, and pattern and demographic lie near the middle.
The resulting slots also remain controllable: linear probes show no measured excess leakage beyond the label-correlation baseline, and quantized slot codes support targeted intervention, including a 15.3x lift for color. These results suggest that controllable fashion retrieval benefits from typed, attribute-conditioned multimodal slots rather than either a single global embedding or opaque item-level semantic IDs. 

---
# MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval 

**Authors**: Aaryan Sharma, Vishak Prasad C, Virendra Singh, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12532)  

**Abstract**: Vision-Language Models (VLMs) are highly effective in retrieving semantically relevant images. However, in practice, relevance alone is often insufficient. Systems must also achieve Result Diversification (RD) across composite attributes such as geography and time, a task for which precise control remains challenging. Current re-ranking methods, such as Multi-Source Determinantal Point Processes (MS-DPP), address this using manifold-based repulsion over similarity representations. Although this strategy is effective for broad exploration, it exposes a key limitation in manifold-based models: when subjected to diversity-decrease tasks on discrete metadata, they suffer substantial degradation in early-rank recall.
To bridge this gap, we introduce MASCOT (Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval). Instead of relying on manifold repulsion, MASCOT formulates multi-attribute diversity as a resource allocation problem, projecting attributes into a soft-binning space weighted by query-driven importance. Averaged across the three PixelProse diversity-decrease tasks, MASCOT preserves an early-rank recall (R@10) of 88.58%, while MS-DPP retains 67.63%. The margin widens under composite constraints: on PP_geo_hour, where temporal and geographic diversity must be suppressed simultaneously, MS-DPP's recall collapses from 0.9737 to 0.4931 and its top-ranked result degrades to R@1 = 0.23, while MASCOT holds R@10 = 0.9410 and R@1 = 0.7202 at a diversity metric above the unconstrained baseline. We do not claim uniform superiority: on aggregate diversity-relevance scores our own simpler ablations attain higher harmonic means on all three decrease tasks, and MASCOT's advantage is specific to recall beyond rank 1 under composite constraints. 

---
# MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents 

**Authors**: Kaichao Liang, Yuqi Cui, Hao Kong, Xinyuan Huang, Guohaotian Hou, Qingcan Kang, Liang Chen, Yiyang Yin, Ke Ye, Jiaquan Guo, Da Chen, Lingan Zeng, Yixing Peng, Rong Yao, Shixiong Kai, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12428)  

**Abstract**: Memory is a core component of AI agents, enabling them to accumulate experience, maintain personalization, and adapt over long-term interactions. However, existing memory systems often remain fixed after development, limiting their ability to adapt their memory models, organization strategies, and procedural knowledge through continued use. We present MindMemOS, a portable and self-evolving memory operating layer that organizes open-world information using a unified entity property timestructure. MindMemOS supports scenario-adaptive memory modeling, higher-order pattern discovery, autonomous memory refinement, and continuous skill evolution. Its MindMemEvolve algorithm employs validation-driven evolutionary search to optimize memory schemas for target scenarios, whiledreaming consolidates accumulated memories by merging redundant records and resolving conflicts. In addition, implicit corrective feedback serves as a human-in-the-loop signal for identifying and revising potentially inaccurate or misaligned memories. Its MindSkillEvolve algorithm further transforms agent execution trajectories into reusable and progressively refined skills. MindMemOS achieves 94.03% accuracy on LOCOMO and 70.63% on PersonaMem. MindSkillEvolve improves SpreadsheetBench success by 9.2 percentage points over the initial-skill baseline. 

---
