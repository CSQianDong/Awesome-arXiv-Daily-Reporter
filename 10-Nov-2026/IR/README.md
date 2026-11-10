# NIFT-SQL: Customer-Supervised Table Scoring for Enterprise Text-to-SQL Retrieval 

**Authors**: Arjun Rao, Nishant Gopinath  

**Link**: [PDF](https://arxiv.org/pdf/2611.12731)  

**Abstract**: Enterprise text-to-SQL starts with finding the tables a question needs, which is hard when a schema spans several databases and hundreds of cryptic tables. We ask whether models trained separately on each customer's example questions, reading only table names, column names, and declared keys, improve table retrieval. NIFT-SQL (Necessity-Informed Fusion for Text-to-SQL) fuses hybrid search over a retriever fine-tuned on the customer's questions with a fine-tuned 560M cross-encoder that scores every table for whether the query needs it, through a logistic regression fitted on held-out questions. On BEAVER, LiveSQLBench, and LiveSQLBench-Large (244 to 971 tables; four splits, three trainings), NIFT-SQL places every reference table in the top 15 for 69.2%, 89.5%, and 85.6% of test questions, against 30.4%, 41.0%, and 21.8% for hybrid search and 61.1%, 81.7%, and 70.4% for the fine-tuned retriever with the same labels. With off-the-shelf search, the cross-encoder without fine-tuning is 21 points lower. A zero-shot 8B reranker needs no labels and, routed by database, shows no resolved complete@15 difference on two collections; adding its scores as two fusion features gives a high-accuracy mode that reaches 72.4%, 92.5%, and 88.6%, ahead of the validation-selected zero-shot reranker variant on two collections, with no resolved difference on the third. Rank fusion of the two trained parts, grouped by database, shows no resolved difference from the learned fusion. Scored once on a newer BEAVER release, NIFT-SQL beats search and leads the label-matched retriever at top 8 on the synthetic test, but trails it at top 15 on real questions. 

---
# TS-GEAR: Taobao Shangou GEnerative Ad Retrieval 

**Authors**: Guofeng Tong, Xu Zhang, Guangliang Yu, Haiqiang Wang, Guannan Cong, Xing Guo, Yang Zhou, Jingjing Duan, Shuang Li, Feng Shi, Hualei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12180)  

**Abstract**: On-demand delivery platforms such as DoorDash and Uber Eats impose a constraint absent in traditional e-commerce: merchants beyond the user's delivery range are invalid candidates rather than merely low-ranked. Generative retrieval has recently shown strong potential in large-scale retrieval systems. Adapting it to this setting, however, poses three challenges. First, existing semantic IDs encode geography only as a soft signal, so reachability can only rely on a non-differentiable post-generation hard truncation. Second, user preferences vary substantially across both regions and queries, so aggregating historical behaviors without conditioning on geography and the current query yields biased interest estimates. Third, the retrieved candidate set is evaluated by its collective value, yet autoregressive training scores each sequence independently, with no set-level signal to discourage redundant near-duplicates. To address the challenges, we propose TS-GEAR, a unified generative retrieval framework for on-demand delivery: (1) Geo-Semantic ID splits merchant identifiers into geographic prefixes and semantic suffixes, making reachability a prediction target rather than a post-hoc filter; (2) Query-Geo Target Attention aggregates historical behaviors using the real-time query and location as dual targets, while encoding each behavior's own location into its attention key; (3) GMPO (Group Marginal Policy Optimization) rewrites advantages using a deduplicated group reward and marginal advantage within GRPO, avoiding auxiliary diversity objectives. Deployed online with a lightweight 0.1B backbone, TS-GEAR serves over 150 million queries per day in the search advertising pipeline of Taobao Shangou, consistently outperforming state-of-the-art generative retrieval methods on both HitRate and geographic validity, and achieving 3.8% gains in ad impressions and 4.2% in ad revenue. 

---
# CHO-Bench: How Do Context Hints Affect Ad Rankings in LLM Advertising? 

**Authors**: Shiqi Zhang, Yang Fei, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.11955)  

**Abstract**: LLM advertising selects advertisements based on user conversations, where user needs may extend beyond explicit product searches. ChatGPT Ads allows advertisers to provide natural-language context hints describing what they offer, who may find it useful, and when it may be useful, as additional information for ranking advertisements across diverse user conversations. However, how to effectively construct these hints remains unclear. We formulate Context Hint Optimization (CHO) as the problem of constructing context hints to improve an advertisement's ranking under a given budget. Drawing on established marketing frameworks, we develop seven context hint generation strategies covering What, Who, and When, along with a Unified strategy that combines all seven dimensions. We introduce CHO-Bench and conduct over 1.19 million paired evaluations across five datasets using embedding-based and LLM-based rankers. Our results show that context hints improve advertisement rankings under the evaluated offline rankers. While individual strategies achieve similar average gains and Unified offers no consistent advantage, strategy differences become more pronounced under specific generation-ranking configurations, with embedding-based and LLM-based rankers favoring different information dimensions. Increasing the hint budget further improves rankings with diminishing marginal gains. These findings establish an offline benchmark for studying ad-side ranking optimization in LLM advertising and highlight opportunities for model-specific hint selection and budget-aware generation and aggregation. 

---
# Multi-Modal Agricultural Document Understanding at the DAGRI Task 

**Authors**: Zhefan Wang, Da Li, Keping Bi, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.11886)  

**Abstract**: Agricultural documents, such as prefectural technical reports and crop calendars, are critical knowledge resources. However, their complex table layouts, significant variations in document length, and dense multi-modal elements pose substantial challenges for automated understanding. To address these issues, the NTCIR-19 DAGRI (Data Analytics for aGRicultural Information) task introduced three subtasks focusing on different dimensions of document intelligence. In the Table Information Extraction (TableIE) task, we developed a multi-stage pipeline combining structure-preserving parsing via MinerU, region-adaptive segmentation for long documents, and a dual-modal (vision-text) extraction strategy for complex work schedules. This approach achieved first place with an exact match score of 0.6566. For Multi-Table Multi-Hop Question Answering (MTMH-QA) over interrelated JSON tables, we adopted an agentic search framework utilizing DCI-Agent to iteratively explore directories via bash commands, reinforced by a cross-model validation mechanism. This yielded a revised accuracy of 0.8444, initially ranked first. In the Figure and Table Question Answering (FiT-QA) task, which requires comprehending single large composite images, we implemented a dual-modal VLM pipeline enhanced by deterministic answer normalization, securing the highest BLEU score. Together, our system effectively addresses regional structural variations, dynamic cross-file evidence discovery, and multi-modal alignment, providing a robust framework for complex agricultural document understanding. 

---
# Proactive Dimension-Adaptive Denoising for Graph Collaborative Filtering via Representation Coherence 

**Authors**: Wenjun Li, Dong Zhao, Jiaying Shi, Juan Zhang, Zhaoming Pan, Zhimin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11801)  

**Abstract**: Implicit feedback in recommender systems contains noisy interactions (e.g., accidental clicks, impulse purchases) that propagate through user-item graphs and contaminate learned representations. Existing explicit denoising methods uniformly rely on training loss to estimate interaction reliability. However, loss is a fundamentally passive signal: it detects noise only after it has caused prediction errors, and it is heavily confounded by training stage, negative sampling, and hard positives. We propose Noise-aware Dimension Learning (NDL), a proactive dimension-adaptive denoising framework. NDL introduces Representation Coherence Detection (RCD), which evaluates the coherence between an interaction representation and the user's historical preference centroid in embedding space, providing an active noise signal independent of loss. RCD and the loss signal are fused via activity-aware weighting (active users rely more on coherence while cold-start users automatically fall back to loss), and the fused estimate drives an interaction-level dimension-adaptive mask that limits the effective representation capacity of unreliable interactions. A lightweight contrastive regularizer further improves embedding uniformity, stabilizing the geometry on which RCD discriminates. On three public benchmarks NDL consistently outperforms state-of-the-art denoising methods, lifting Recall@5 by +17.95% on Amazon-book, +12.33% on Yelp, and +5.63% on MovieLens-1M over the strongest baseline DAR; the core variant without contrastive learning already surpasses DAR on sparser datasets, confirming representation coherence as the primary driver of these gains. NDL is confined to the training loop and is serving-compatible with LightGCN-based production pipelines, adding no online latency or parameters at inference time. Code: this https URL 

---
# Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation 

**Authors**: Guy Hadad, Haggai Roitman, Bracha Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2611.11549)  

**Abstract**: Generative recommendation models represent items as sequences of discrete tokens known as semantic IDs. Existing approaches derive these identifiers using residual or hierarchical vector quantization, assuming that tokens carry independent semantic meaning across quantization levels. Revisiting this assumption, we argue that semantic IDs should behave as hierarchical paths where each token's meaning is conditioned on its prefix. This highlights a mismatch between conventional fixed-codebook quantization and the hierarchical structure implicitly learned by generative recommenders. Motivated by this insight, we propose Adaptive Silhouette Tree (AST), a top-down divisive clustering method that constructs semantic IDs as paths within a tree structure. Unlike prior methods with fixed branching factors, AST dynamically determines the number of branches at each node by maximizing the silhouette coefficient, enabling finer partitions in dense regions and compact groupings in homogeneous ones. We further introduce constrained variants that enforce minimum probability mass during splitting to improve supervision for long-tail items. Extensive experiments on five Amazon Review datasets of varying sizes and domains show that AST significantly outperforms strong baselines. These results demonstrate that aligning semantic ID construction with the inherent hierarchical structure of generative recommenders leads to more effective recommendations. 

---
# Beyond Ranking: Are Decision Models Calibrated Relevance Scorers? 

**Authors**: Shengyao Zhuang, Xueguang Ma, Matthias Petri, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.11480)  

**Abstract**: Decision models such as Jev are a new kind of relevance scorer. Their typed score primitive takes a question, an ordered set of textual relevance levels and a state, and returns a probability for each level directly, rather than token likelihoods or generated labels. Jev's vendor promises that these probabilities are calibrated. This motivates revisiting 'Beyond Yes and No', which improved zero-shot pointwise LLM reranking using likelihood-derived scores over fine-grained relevance labels but left calibration against human assessors as future work. In this paper, we first reproduce its open-model FLAN-UL2 setting on eight BEIR collections, confirming its ranking trends, and then run Jev and two open-weight decision models with the same protocol. Jev and the 27B pplx-decider rerank ahead of FLAN-UL2 and three cross-encoders on average. We then compare Jev's native probabilities with the human assessors' labels and find them overconfident: of the DL21 pairs it scores above 0.9, only 66% are relevant. Since this is mostly an error of scale, a two-parameter calibration fitted on 2,257 DL19 pairs removes much of it and transfers to DL21. After calibration, Jev has a lower DL21 Brier score than FLAN-UL2 and the three cross-encoders, mainly because it separates relevant from non-relevant pairs better. Finally, with calibrated probabilities, choosing each query's cut-off by expected F1 needs no parameter of its own, beats a fixed top-10 cut and matches a cut-off threshold tuned on DL19. 

---
# IntentLens: Grounding Underspecified Multimodal Queries for Recommendation via Tool-Augmented Reasoning 

**Authors**: Xiao Chen, Dong Fang, Haitao Li, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.11346)  

**Abstract**: As intelligent personal assistants are increasingly deployed in real-world scenarios, recommendation paradigms are shifting from passive preference matching to inferring user intent from underspecified multimodal inputs. In practical interactions, users typically issue a reference image paired with a vague natural language query, keeping key visual attributes, personalized preferences, and item-side constraints implicit. Existing recommendation methods either leverage historical user interactions augmented with multimodal item features or assume fully articulated textual queries; both paradigms struggle to resolve underspecified multimodal queries for accurate item matching. To address this limitation, we propose IntentLens, a tool-augmented recommendation framework that grounds ambiguous multimodal queries into query-relevant evidence prior to user-item matching. IntentLens follows a ground-then-rank pipeline: a shared multimodal LLM leverages visual tools, user memory modules, and item attribute retrieval tools to (i) infer latent user intent from image-text inputs and (ii) augment each candidate item with fine-grained, query-conditioned evidence. Subsequently, a lightweight ranker scores candidates based on the grounded representations. To facilitate research in this novel setting, we construct two benchmarks with underspecified multimodal queries, namely \textsc{GoogleReview-MMTool} and \textsc{Yelp-MMTool}. Extensive experiments on these benchmarks demonstrate that IntentLens outperforms competitive MLLM and retrieval baselines, and exhibits strong generalization to unseen categories and datasets. 

---
# An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users' Temporal Preference 

**Authors**: Jung-Hsien Chiang, Chung-Yao Ma, Chi-Shiang Wang, Pei-Yi Hao  

**Link**: [PDF](https://arxiv.org/pdf/2611.11278)  

**Abstract**: Recommendation systems have become more important since the widespread use of the Internet. The large amount of information means that it is difficult for users to discover they really need. However, users' preferences can change over time because of the age and the impact of social networks. Recommendation systems must capture users' preferences and recommend suitable items, which is a difficult challenge. This study proposes a novel recommendation system that adapts to the changing preferences of users. By learning and adapting to users' changing preferences better recommendations are provided. This study uses context factors as additional information to model users' preferences more precisely. For the proposed recommendation system, the context is based on users' interaction items within each defined time interval. This study proposes two novel attention mechanisms for the recommendation system. The contextual item attention module captures contextual information, changing pattern in users' preference and the importance of items. The multi-head attention module extends the diversity of users' preferences and adapts to changing preferences. The recommendation performance is improved using additional item's temporal information to model the contextual item's representation. Experiments compare the proposed algorithm with several state-of-the-art recommendation methods using three real-world datasets. The experimental results demonstrate that the proposed context-aware recommendation model outperforms traditional methods and demonstrate the effectiveness with which contextual information is captured by the attention mechanism. 

---
# Dhara: Evidence-Oriented Retrieval for Indian Knowledge Systems 

**Authors**: Avinash Holenarasipur Nagendra, Anand Kaniyaar Sriranga Prasad, Rahul Vishwakarma, Trupti Gavit, Anish Amol Tipnis, Yashaswini Mandayam Rangayyan, Vinod Appajanna, Kousik Sarathy Sridharan, Mohan Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2611.11197)  

**Abstract**: Relevant evidence in Indian Knowledge Systems (IKS) collections may be distributed across passages in different parts of a work, including verses, translations, and commentaries. Locating it requires matching natural-language questions and partially remembered verse fragments to source passages. We present Dhara, a retrieval engine combining hybrid search, chapter- and book-level Entity Notes (EN), Fuzzy Verse Retrieval (FVR), and neural reranking. Five comparisons on the 325 questions in Dhara Bench examine retrieval representations against BM25, EN's contribution, reranker quality and computation time, FVR, and Gemini. BGE-M3 single-vector retrieval exceeds BM25 in nDCG@10, Recall@20, and MRR@20 across all categories. Its scores are broadly comparable to ColBERTv2, with mean retrieval times of 0.108 s and 0.962 s under the same hardware configuration and a smaller memory footprint for chunk storage on disk. Adding EN to a direct branch with a matched candidate count improves nDCG@10 in all twelve categories with Qwen and ten with BGE; the combined ranking is capped at the same candidate count. Qwen improves nDCG@10, Precision@5, and MRR@20 over BGE in every category, at approximately 6.72 times the overall estimated serial computation time. FVR increases Verse-category nDCG@10 by approximately 20 percentage points with either reranker. Gemini is evaluated after source mapping and manual review with relevance annotations visible. The Qwen pipeline with EN and FVR has higher Recall@20 than Gemini in ten categories and stronger complete-evidence retrieval in Multi-hop questions, while Gemini retains early-relevance advantages in several categories. The results inform retrieval choices according to evidence requirements and resource budgets. 

---
# PCG-RAG: Progressive Compositional Grounding for Retrieval-Augmented Generation in Complex Web Question Answering 

**Authors**: Linyao Zheng, Xuhang Shi, Zhifang Mao, Chongqi Qu, Xiuquan Hou  

**Link**: [PDF](https://arxiv.org/pdf/2611.11113)  

**Abstract**: Complex Web question answering requires retrieval-augmented generation (RAG) systems to identify target entities from multi-clue descriptions and connect evidence scattered across long Web pages. Individual pages often support only some of the required semantic constraints, while clues discovered on one page guide subsequent searches on others. However, premature reliance on partially supported entity hypotheses can divert retrieval from the original information need. RAG systems must therefore use partial evidence to guide further searches while seeking the evidence needed to establish the intended entities. We propose PCG-RAG, a novel RAG framework based on Progressive Compositional Grounding (PCG). Progressive Evidence Acquisition directs subsequent searches toward missing evidence for candidate attributes and relations. Compositional Reasoning separates candidate-guided retrieval from evidence-based answering along a directed acyclic graph (DAG) of subquestions. Provisional names make dependent searches specific, while answering tasks use accepted results and unresolved identifying conditions, assessed against the original question. New passages connect the identifying clues to the requested relations, supporting evidence-based downstream answers. Evidence Grounding combines identifying and relational source passages with supported intermediate answers, connecting evidence across Web pages to the original conditions during final generation. On BrowseComp-Plus and WebWalkerQA, PCG-RAG improves accuracy over the leading compared baselines by 5.43 and 5.15 percentage points, respectively. 

---
# RoutingTrie: Metadata Routing for Constrained Generative Retrieval 

**Authors**: Darshan Fofadiya, Yang Liu, Shuying Sun, Kyumin Park, Juhana Kangaspunta  

**Link**: [PDF](https://arxiv.org/pdf/2611.11016)  

**Abstract**: Advances in large language models and semantic IDs (SIDs) have brought generative retrieval (GR) into industrial applications. These systems must return items that satisfy request metadata constraints as well as relevance requirements. A catalog trie ensures SID validity, but filtering metadata only after generation wastes computation on rejected candidates and may leave too few eligible results. We propose RoutingTrie to incorporate metadata constraints before beam selection. RoutingTrie maintains one shared SID trie and builds a compact index for each metadata field using Dense Bitset or Sparse List. We derive a storage cost formula to guide the choice between these index representations for different field distributions. During decoding, field probes determine whether requested values occur below candidate edges and produce a combined mask that guides beam selection. Experiments on ShopTrie, our configurable synthetic benchmark, demonstrate compact storage and efficient CPU and GPU probing across diverse metadata distributions. In a production replay, RoutingTrie raises Hit@100 from 67.5\% to 96.1\%, a gain of 28.6 percentage points, with only 1.1 ms latency increase. We will release the ShopTrie generator, benchmark scripts, and RoutingTrie CPU and Triton GPU implementations to support reproducibility. 

---
# Slim Foundation Models for Latency-Critical Recommendation Serving 

**Authors**: Danyal Aftab, Anas Zafar, Assadig Abakr, Steven Davy  

**Link**: [PDF](https://arxiv.org/pdf/2611.11013)  

**Abstract**: Click-through rate (CTR) prediction is a core component of recommendation and advertising systems. Large language models (LLMs) offer rich semantic priors that improve prediction quality, particularly in cold-start and long-tail settings. Injecting a full-scale LLM into a CTR pipeline is, however, prohibitively expensive at inference time: existing LLM-enhanced CTR methods avoid this cost by caching LLM outputs offline rather than serving the LLM online. We propose a resource-efficient pipeline, SlimFM, that makes online LLM-based semantic modeling practical: (1) dependency-aware structured pruning compresses LLaMA3-8B by 44% to a 4.5B-parameter backbone; (2) LoRA-based field-aware fine-tuning trains the pruned model to produce discriminative field-level semantic embeddings from lightweight textual prompts; (3) a semantic-guided fusion module injects these embeddings into the feature interactions of a conventional CTR model. Across four datasets and six CTR backbones, SlimFM retains over 99% of uncompressed-model accuracy while reducing inference latency by up to 3.4x relative to existing LLM-enhanced baselines. 

---
# DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback 

**Authors**: Minhyeong Yu, Wonduk Seo, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.10977)  

**Abstract**: Reliable function calling requires Large Language Models (LLMs) to select appropriate functions and generate arguments consistent with user intent. However, syntactically valid calls may still be semantically incorrect due to inappropriate function selection or mismatched arguments. In this paper, we propose Decision-Model-Guided Function Calling (DM-Call), a framework that combines local schema validation with separate decision-model verification of function selection and argument correctness. For rejected candidates, DM-Call accumulates verification probabilities, binary judgments, and schema errors across attempts. A hybrid feedback representation further combines numerical probabilities with verbal correctness labels to guide iterative refinement without executing external tools. Extensive evaluation across BFCL and API-Bank (first-call) demonstrates DM-Call boosts function-calling accuracy versus direct generation, at modest additional cost. 

---
# A Tool for Anonymized Clinical Knowledge Retrieval and Speaker Interaction Analysis of Healthcare Tele-mentoring Sessions 

**Authors**: Parthaw Goswami, Erfan Zareei, Amy Vogelsmeier, Dima Dandachi, Ben Casterline, Mirna Becevic, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2611.10922)  

**Abstract**: We present a tool for anonymized retrieval of clinical knowledge and analysis of multi-speaker interactions in tele-mentoring sessions of Project ECHO (Extension for Community Healthcare Outcomes). Our tool's novel design includes: (a) a compound metadata filter that enforces strict speaker- and session-level isolation during retrieval, preventing cross-session context contamination; (b) MedGemma-4B as a locally-hosted language model for clinically grounded answer generation across three query modes; (c) interaction graphs to compute betweenness centrality, PageRank influence, and turn-taking patterns to quantify participation equity; and (d) a consistent alias-based anonymization scheme that prevents real participant names from appearing in any rendered output. During the demo, a healthcare provider (user) will interactively query multi-year ECHO sessions, explore participation dynamics through interaction graphs, compare sessions side-by-side, and assess answer quality via six reference-free RAG metrics, all without any cloud API dependency. 

---
# From User Behaviors to Satisfaction Judgments: Reasoning about User Satisfaction with AI Overviews 

**Authors**: Xingzhu Wang, Yitong Xu, Yue Li, Daiting Shi, Jingzhou He, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.10715)  

**Abstract**: The growing presence of AI overviews in everyday search makes their effective evaluation essential. Response quality has been the primary focus of existing evaluation research, with only a small number of studies examining the user perspective. These evaluations rely on queries and response content, leaving the information conveyed by user behavior insufficiently utilized. This work investigates satisfaction prediction for AI overviews from a user-centered perspective and incorporates behavioral signals into the prediction process. Applying existing behavior-based satisfaction prediction methods from traditional information retrieval to this setting, however, presents limitations. We address these limitations with STaRS, an LLM-based approach designed specifically for predicting satisfaction with AI overviews. At its core is a structured reasoning procedure: individual behavioral signals are interpreted first, and their implications then inform the prediction of user satisfaction. Reinforcement learning further optimizes this procedure, while a reward shaping mechanism alleviates reward sparsity during training. Evaluation on two real-world datasets demonstrates that STaRS outperforms the comparison methods in satisfaction prediction, providing search providers with a practical means of accurately estimating users' satisfaction with AI overviews. 

---
# Formalizing Patient-Set Semantics for OMOP CDM-Based Clinical Trial Feasibility Assessment:A Neuro-Symbolic Methodology for Traceable Site-Feasibility Patient Counts 

**Authors**: Sohyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2611.10681)  

**Abstract**: This study examines the interpretation and execution of natural-language eligibility criteria for clinical trial site-feasibility patient-count estimation using the OMOP Common Data Model (CDM). It treats each count as the size of a patient set produced under explicitly declared choices about clinical concepts, temporal conditions, exclusions, proxies, and available data.
The study presents NS-CoT, a neuro-symbolic pipeline that uses an LLM to interpret and structure criteria, followed by symbolic processing and deterministic SQL compilation. Intermediate representations and execution records document how criteria are translated into patient sets and counts.
Evaluation uses synthetic OMOP CDM data, controlled benchmarks, and a multi-site simulation. Comparisons with a direct-SQL baseline examine execution success, count agreement, patient-set agreement, and preservation of reference-defined site-feasibility labels and rankings. Performance varies across evaluation settings. Layer-wise inspection also supports the identification of implementation and policy errors, followed by correction and re-evaluation.
The work investigates the reviewability and fidelity of cohort computation relative to declared reference definitions. It does not establish that these definitions capture a unique clinical ground truth or demonstrate improved recruitment, site selection, or trial outcomes in practice. 

---
# When Greedy Is Sorting: Exact and Certified Ideal Rankings for Novelty Metrics 

**Authors**: Kehan Liu, Zhaoyun Chen, Xianquan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.10576)  

**Abstract**: The ideal ranking that normalizes novelty metrics is NP-hard to compute in general, so widely used $\alpha$-nDCG evaluators approximate its denominator with a greedy rule that rescans every remaining document at each rank. We identify a regime in which this ideal ranking is exact and cheap for both $\alpha$-DCG and intent-aware expected reciprocal rank (ERR-IA). When each document is positively judged for at most one intent, each metric reduces to the classical allocation of a fixed budget across chains of decreasing marginal gains. A metric-specific virtual-score transformation followed by a multiway merge, which we call ChainMerge, returns the globally optimal top-$K$ ranking in $O(G+K\log G)$ time from $G$ preordered intent groups, or in $O(n+K\log G)$ time from $n$ globally preordered candidates, instead of the $\Theta(nK)$ of the greedy rule. When intent judgments overlap, exact $\alpha$-DCG optimization is NP-hard under the standard logarithmic discount and $\alpha=0.5$, even when every document has three intents and every intent occurs in two documents. For arbitrary overlap, a polynomial-size LP over nested ranking prefixes upper-bounds the ideal score and certifies the optimality gap of any feasible ranking. In the official TREC Web 2009-2012 judgments, 64.8% of positive documents serve a single intent, yet only 20 of 198 complete queries do so throughout. Beyond this exact regime, the LP certifies 189 of the 198 greedy denominators as globally optimal, and the resulting score intervals preserve the winner in all eight matched comparison groups. On 65,238 MIND-small candidates, ChainMerge returns identical optimal objectives 17-1,011 times faster than the greedy rule. For $\alpha$-DCG, these results trace the hardness of the ideal ranking to intent overlap. 

---
# Yuanbao-Reranker: Document-Grounded Synthesis and Preference Distillation for Text Reranking 

**Authors**: Hongyuan Xu, Dezhi Ye, Jiahao Cao, Jie Liu, Haijin Liang, Haibo Shi, Jin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.10405)  

**Abstract**: High-precision text reranking is a critical component in modern search, recommendation, and agentic retrieval-augmented generation (RAG) systems. We present Yuanbao-Reranker, a family of general-purpose dense rerankers spanning 0.6B, 4B, and 8B parameter scales built upon the Qwen3 backbone. To construct diverse and reliable ranking supervision at scale, we pair answer-first and query-type-guided document-grounded synthesis with a three-stage curation funnel that verifies positive support, eliminates semantic redundancy, and enforces balanced source-by-length quotas. For each curated pair, we screen real retrieval pools with a teacher score margin to filter out potential false negatives and sample Hard, Medium, and Moderate negative tiers, yielding 2.75M $1{:}3$ ranking groups. We train the models by jointly optimizing pointwise relevance cross-entropy and a bounded minimum-gap hinge objective over both positive--negative and negative--negative pairs. This formulation transfers fine-grained teacher preferences across candidate tiers without over-constraining well-separated predictions, supporting both cross-scale and same-scale distillation followed by layerwise adaptive model merging. Across five comprehensive benchmarks spanning Chinese (CMTEB-R), multilingual (MMTEB-R), English (BEIR), long-document (MLDR), and code (MTEB-Code) retrieval, Yuanbao-Reranker consistently leads every parameter tier: the 0.6B variant reaches 70.96 Avg. to surpass 1B--4B active-parameter baselines, the 4B variant reaches 73.63 Avg. to outperform the 8B teacher (72.68), and the 8B variant achieves a state-of-the-art 74.21 Avg., with substantial gains generalizing across distinct first-stage retrievers. 

---
# LEGIT: Live Eligibility Gating over Identifier Tries for Constrained Decoding of Heterogeneous Pages 

**Authors**: Felix Fang, Zheng Liu, Robert Saliba  

**Link**: [PDF](https://arxiv.org/pdf/2611.10357)  

**Abstract**: A generative recommender names each candidate with a short token sequence, a Semantic ID, and recommends by generating those tokens. Some generative recommenders, including the Uber Eats search engine we are building, write a whole results page, stores with their menu items, in one decoding pass. Every Semantic ID on such a page must name a real store or item that is eligible for the search, such as a store that delivers to the user's location, and that is not already on the page. Each item must also belong to the store above it. A trie over the catalog keeps identifiers real, but eligibility and repetition change with every request and token, and the grammar-constrained decoder we tried could not enforce these changing rules exactly and quickly. We call this constraint LEGIT, for live eligibility gating over identifier tries. Numbering trie leaves depth first turns each rule into a range lookup over one store trie and one item trie, and every generated page passes an independent catalog check. On one NVIDIA B200 GPU with one or two requests per decode step, a direct implementation that walks the trie on the GPU makes each token about 5 times as slow as unconstrained decoding. Across 36 implementations, hardware counters and ablations are consistent with kernel launches and CPU-GPU waits as the source of this slowdown. Our fastest implementation walks the trie on the CPU while the GPU runs the model, needs no custom kernel, and cuts time per output token by 65.9 percent against the direct GPU implementation, to about 1.7 times the unconstrained cost. This implementation is also the fastest on a page layout that puts each item before its store. 

---
# SepLRM: A Foundation Model Paradigm for Recommendation 

**Authors**: Di Chen, Xijun Xiao, Qin Ren, Weijiang Lai, Zhijian Pan, Zhentao Pan, Pengfei Nie, Cheng Yuan, Zixuan Tan, Xuyang Liu, Tao Wang, Yunwen Huang, Hao Fang, Yichao Wang, Hao Zhang, Siyuan Wu, Hao Chen, Han Diao, Mingyu Yang, Yongkang Feng, Beihong Jin, Jinfeng Yang, Deping Xie, Peng Xu, Zheng Chai, Yuchao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.10356)  

**Abstract**: Large recommendation models (LRMs) improve recommendation performance by scaling behavior sequence length, model capacity, and training compute. However, the return on investment (ROI) of continued scaling deteriorates as marginal gains diminish and long-sequence encoding costs rise, especially when this computation is repeatedly executed by distinct models for diverse recommendation stages and business scenarios. To address this issue, we propose SepLRM, a recommendation foundation model (RFM) paradigm that identifies reusable sequence encoding and reorganizes multiple end-to-end (E2E) LRMs into one RFM and multiple independently optimized vertical models (VMs). The RFM adopts an encoder-decoder architecture and exports its final-layer encoder state as the shared key-value (KV) cache that is independent of downstream task-offered candidates. Unlike candidate-specific logits or aggregated user embeddings, the KV cache preserves fine-grained behavioral information. Separation at this native KV interface also allows each VM to construct candidate-conditioned queries while retaining the same query-KV interaction as its E2E counterpart, enabling high-fidelity capability transfer. To ensure both transfer effectiveness and system efficiency, we co-design cache compression, the training and serving workflows, and LG-Store, a distributed storage system for KV caches. Experiments on three large-scale datasets show that SepLRM achieves state-of-the-art offline performance with a transfer rate of approximately 100%. It also demonstrates strong transferability across model iterations, recommendation stages, and business scenarios. Deployment in two large-scale industrial scenarios substantially improves the ROI of sequence scaling and delivers consistent gains in online A/B tests at ByteDance. 

---
# Edge-Deployable LLMs for Classification and Value-of-Information Ranking of Disaster-Related Tweets 

**Authors**: Christopher D. Molek, K. Brent Venable, Niranjan Suri, Stephen Russell  

**Link**: [PDF](https://arxiv.org/pdf/2611.10158)  

**Abstract**: Humanitarian-aid and disaster relief decisions often rely on noisy, incomplete, and uncertain information from social media. Although substantial research has focused on automating the classification of such data, limited work has examined whether individual posts contain sufficient information to support decisions to act. In this study, we use large language models (LLMs) to (1) classify social media content and (2) assess its Value-of-Information (VoI) in terms of operational actionability. We evaluate seven LLMs for the classification of 77,484 tweets across 10 humanitarian-aid categories and identify the Qwen3 (8B parameters) model as the top-performing classifier, achieving the highest F1 scores in all but one category. In a second task, the classified categories serve as contextual inputs for assessing VoI with respect to responder actionability across six well established metrics for each tweet. We observed that the Gemma 3 (4B parameters) model outperformed the other models on this task, achieving the highest recall across the majority of categories. This work demonstrates that relatively small, edge-deployable LLMs can prioritize actionable information in resource-constrained environments. By integrating VoI assessment, our framework advances the automated elicitation of high-utility knowledge from unstructured social media streams for disaster-response decision-making and highlights a pathway toward tactical edge applications using LLMs that fit on a single 8 GB GPU. 

---
# The Narrative Gap: Can LLMs Help Us Navigate Diverse Narratives Across Languages? 

**Authors**: Nikhil Sharma, Kelly Marchisio, Kenton Murray, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.12750)  

**Abstract**: Large language models (LLMs) increasingly mediate how people seek and consume information, including accounts that differ across languages and regions, especially on contested topics. Supporting diverse information seeking requires models to help users navigate competing narratives across languages and information streams. We define six capabilities necessary for diverse multilingual information seeking: three foundational capabilities for narrative inclusion, narrative exclusion, and missing narrative identification; and three higher-level capabilities for presenting diverse narratives, preserving narrative coverage under confirmatory queries, and maintaining information parity across languages. To guide the development of those information seeking capabilities, we introduce \textsc{NarrativeBench}, a benchmark comprising 1,333 news articles in 23 publication languages across 23 contested topics and 460 Reddit-grounded queries. Evaluating 14 LLMs across 24 query languages, we find that current models lack the capabilities needed to reliably navigate diverse narratives, limiting users' control over the perspectives they receive and leaving substantial gaps in exposure to competing accounts even when diversity is explicitly requested. Information access also differs across query framings and languages, risking echo chambers, linguistic filter bubbles, and information disparities that weaken the common ground needed for democratic discourse. \textsc{NarrativeBench} provides a foundation for developing information systems that give users across languages and regions agency to navigate diverse narratives. 

---
# Scaling Taxonomy-Free Image Labeling with Generative AI: A Multimodal 

**Authors**: Zhenjie Zhang, Sainan Chen, Kshiteesh Hegde, Mengze, Pouya Rezazadeh Kalehbasti, Munachim Ezema, Karina Sobhani  

**Link**: [PDF](https://arxiv.org/pdf/2611.12725)  

**Abstract**: Providing real-time multimodal insights into images is challenging because it requires low latency, high description accuracy, and scalable solutions that are both cost-effective and efficient. For example, given a new image of an outfit, we would like to generate insights that the image depicts "earth tones" and "linen pants" and allow users to explore similar ideas along these themes. Traditional strategies to generate such insights would typically involve building taxonomies producing a golden set labeled by humans, training models on these datasets, and deploying these models for online inference services. Such workflows are constrained by fixed taxonomy, high labeling cost, limited labeling coverage in sample space, and human errors and expertise gaps in labelers. In addition, these methods are expensive and cumbersome to iterate over time. Recent trends have started exploring generative AI models for online insights, but this method suffers from long latency and prohibitive costs to support a high volume of requests or large amounts of offline labeling needs.
We present Generative Labeling On Clusters (GLOC) as a solution to provide highly scalable, cost-effective, low latency, and high accuracy online image insight services. GLOC leverages highly optimized image embeddings to form image clusters that are semantically close to each other for the purpose of the type of insights, and summarizes the insights at the cluster level, which effectively generalizes to members in the cluster. At Pinterest, GLOC powers image descriptors and pivot query suggestions for Hybrid Search, which is a multimodal search service enabling users to search with both an image and a text query. Our offline and online A/B tests demonstrate that GLOC significantly outperforms previous methods, delivering improvements in quality and efficiency. 

---
# Multi-Agent Screening: What Agreement and Acceptance Cannot Tell You 

**Authors**: Benjamin Chang, Dell Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12705)  

**Abstract**: When AI agents screen items for a costly human assessor, as in agentic search, e-discovery and AI reviewing of papers, only what they pass gets judged. We show what each kind of evidence can establish about their errors, even with unlimited data. Agreement shows that agents concur, not that they are right, however many there are. Acceptance by the assessor gives precision but only a lower bound on recall. Knowing which agents found which relevant items gives an upper bound on recall if agents' finds are not negatively correlated, and caps what one more agent can add if agents are interchangeable. Without assumptions about the agents, recall can be pinned down only by judging items they did not pass: a random sample of these estimates how many relevant items were missed, to any chosen accuracy, and the sample needed does not grow with the number of agents. 

---
# Hidden Distances: How Likert-Scale Spacing Shapes Algorithmic Voting Advice 

**Authors**: Thomas Broholt Knudsen, Christian Høst-Madsen, Vedran Sekara, Jonas L. Juul  

**Link**: [PDF](https://arxiv.org/pdf/2611.12313)  

**Abstract**: Voting Advice Applications (VAAs) are algorithmic recommenders that match voters with political candidates on policy preferences. They are widely adopted across many multi-party democracies; in Denmark, 62\% of eligible voters used a VAA ahead of the 2022 general election, and 45\% of VAA users reported following its advice. VAAs typically ask voters and candidates to respond to policy statements on a Likert scale, from ``strongly disagree'' to ``strongly agree'', and recommend candidates whose responses are closest to the voter's. Computing these distances requires assigning numbers to response options, a design choice without an empirical basis, since it is unknown how people perceive the distances between options. Here we show that this seemingly technical choice substantially changes whom voters are matched with. We parameterize the distance between ``(dis)agree'' and ``strongly (dis)agree'' relative to the distance between ``neutral'' and ``(dis)agree'' as $\alpha$, and quantify how it changes recommendations for a popular Danish VAA. We find that moving from equidistant options to doubling this distance can change $37.7\%$ of top candidate matches and $21.9\%$ of top party matches, although no answers change. Low $\alpha$-values favor candidates with strong positions, with some parties receiving up to $ 19.6\%$ more recommendations, while high $\alpha$-values favor moderates. Similarly, female candidates receive fewer recommendations at low $\alpha$ and more at high $\alpha$-values. Our findings demonstrate that undisclosed quantification choices in algorithmic recommenders can carry considerable electoral consequences, and they call for transparency and sensitivity analysis in VAA design. As Likert-based matching also underpins dating platforms and roommate allocation systems, these implications extend well beyond elections. 

---
# RecoGEM: A Serving System for Production-Scale Generative Recommendation 

**Authors**: Xiangrui Yang, Yunfeng Zhao, Jiawei Yang, Fangyu Zhang, Enrui Zhao, Liang Zeng, Cheng Peng, Minhui Xie, Jiaqiang Liu, Xiao Liang, Yiming Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12296)  

**Abstract**: Generative recommendation (GR) is moving industrial recommender systems (RecSys) from scoring candidate items to autoregressively generating item identifiers. This shift promises higher recommendation quality, but makes serving substantially harder: each request executes a short sequence of dependent decoding steps that explore a large item space and select promising candidates on the latency-critical path. GR not only creates structural mismatches with conventional RecSys pipelines, but also challenges the architectural and optimization assumptions of LLM serving stacks, making direct adoption of existing systems difficult. We present RecoGEM, our production GR serving system. RecoGEM uses a runtime--engine decoupling architecture to move GR-specific serving responsibilities, such as candidate-state transitions and live artifact updates, to a dedicated runtime. Its execution engine exploits GR's short token sequences, large candidate populations, and predictable step shapes to optimize self-attention, candidate selection, and execution planning. RecoGEM has been deployed for over nine months across our production lines, serving applications with hundreds of millions of users and supporting more than ten GR model families. On our OneModel production workloads, RecoGEM reaches over 1,000 QPS per serving instance with P99 latency below 35 ms and a 2.88$\times$ throughput improvement over representative baselines. 

---
# Calibrating Document-Local Boundaries for Contextual Chunk Retrieval 

**Authors**: Yuanxiang Wang, Hui Jiang, Da Zhu, Zhihua Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12097)  

**Abstract**: Late Chunking encodes an entire document before pooling at chunk boundaries, preserving document context in local retrieval units. Chunk-level post-training improves context-dependent retrieval, but shared context can make semantically similar same-document negatives difficult to distinguish. We study the local decision boundary between a gold chunk and its hardest sibling. Starting from InSeNT, we analyze InSeNT's within-document gradients and show that the entropy of the conditional negative allocation decreases monotonically with the scaling factor at fixed logits. We propose a two-part framework for calibrating document-local chunk boundaries. The first component adapts the discriminative scale from within-document margins, while the second exposes chunk partitions to attention during document encoding. Experiments on ConTEB and NanoBEIR show that the resulting model improves contextual chunk retrieval across the evaluated encoder settings while retaining competitive passage-level retrieval quality; ablations further characterize the distinct roles of adaptive scaling and chunk-aware routing. 

---
# Search-by-Answer Bias in Retrieval Models 

**Authors**: Adiel Avigad, Shauli Ravfogel, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.11396)  

**Abstract**: Dense retrievers are meant to select passages that match a query's semantic specification. We show they suffer from search-by-answer bias: they favor passages that mention answer objects the model knows (scores as likely), over passages that support valid alternative answers. We test this with controlled factual contrasts from LAMA/TREx. For each retriever, we measure how strongly the model scores the canonical object over an alternative object using a token-prediction head, and ask whether this score predicts passage rankings. Across eight retrievers, including DPR, Spider variants, Contriever, E5, and Qwen3-Embedding, larger token-prediction scores consistently predict higher retrieval scores for passages containing the corresponding object. This holds when passages differ only in the answer object, and even when the passage that mentions the high-scoring object expresses a wrong relation. Dense retrieval thus exhibits search-by-answer bias during passage selection, shaping which evidence is retrieved. 

---
# Hybrid Product Search Without the Copy: Lexical, Semantic and Filtered Retrieval Served Directly from a Data Lake 

**Authors**: Raghvendra Yadav, Xiang Fu, Chinmay Soman  

**Link**: [PDF](https://arxiv.org/pdf/2611.11031)  

**Abstract**: E-commerce catalogs increasingly live in lakehouse tables: Apache Iceberg metadata over Parquet files in object storage. Product search, however, still runs on copies of that catalog, in a lexical engine and a vector database, each with its own ingestion pipeline, schema, staleness window and bill. We show that search applications can be hosted directly on the data lake, with the lakehouse table as the single system of record. We extend Apache Pinot, a real-time OLAP engine that already serves Iceberg tables in place, with BM25 relevance scoring over its Lucene text index, corpus-wide term statistics so that scores are comparable across files, and reciprocal rank fusion in plain SQL; a real-time table fed by a stream can sit behind the same table, merged by the broker. On the Amazon ESCI benchmark (1.2M US products, 22,458 test queries), hybrid search served from the lake matches exact retrieval (nDCG@10 .361). The lake table is queryable 149 s after its files are registered, and a lake commit is searchable in 3.8 s, close to indexing directly into a search engine. On a brand-sorted lake of 277M products (0.8 TB), one 16-vCPU node answers brand-scoped vector, BM25 and client-fused hybrid searches in 6.5-9 ms (p50), sustains 1,100 vector and 635 BM25 searches per second at under 180 ms p99, and makes a commit searchable in 4.3 s. Search adds only its indexes: no copy of the catalog, no sync pipeline and no second schema, which cuts storage, rebuild time and the operational work of keeping copies consistent. 

---
# Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion 

**Authors**: Uday Allu  

**Link**: [PDF](https://arxiv.org/pdf/2611.11010)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) improves multi-hop question answering, but deployed systems face two costs that research comparisons rarely measure: building the graph, and removing documents from it. We present Lightfoot, a GraphRAG system in which every entity, link, fact and synonym edge is a deterministic function of per-passage extractions carrying chunk-level provenance. Indexing makes one language-model call per passage, retrieval makes none, and deleting a document is a database operation with no model calls. On the full 11,402-passage MuSiQue corpus we verify that, after deleting 10% of the documents, the graph is identical to a fresh index of the surviving documents. We compare Lightfoot with HippoRAG 2, LightRAG and fast-graphrag under identical extraction, answer and embedding models on MuSiQue, 2WikiMultiHopQA and HotpotQA. Lightfoot builds the cheapest index on all three datasets, 26-31% below HippoRAG 2 and 3-6x below LightRAG and fast-graphrag. Against HippoRAG 2 it is significantly better on MuSiQue retrieval (+2.8 points recall@5 and +4.3 recall@10 on a held-out split) and on HotpotQA answers (+7.7 points exact match), and significantly worse on 2WikiMultiHopQA (-3.9 recall@2, -8.4 F1). Deleting the same documents took LightRAG 2,187 model calls and 3.7 hours; fast-graphrag has no deletion path. Code and per-question results: this https URL 

---
# Clinical Evidence-Guided Residual Learning for Medication Recommendation with Dynamic DDI Regularization 

**Authors**: Zesheng Li, Chengchang Pan, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.10613)  

**Abstract**: Medication recommendation must combine patient-specific treatment evidence with the interaction structure of a prescription. The central challenge is to determine which evidence supports each medication and how strongly to regularize interactions while preserving predictive quality. We propose EviDrug, a clinical evidence-guided residual learning framework. Its clinical-memory backbone integrates a self-trained visit-aware encoder, training-case memory and one medication-set interaction block to establish a patient-specific prescription. A gated refinement head combines the backbone states with clinical associations and history-conditioned medication identities through bounded logit corrections. Drug-drug interaction (DDI) feedback adapts regularization strength using training-query predictions. Across MIMIC-IV, NWICU and SICdb, EviDrug leads twelve published-method baselines in AP, F1 and Jaccard, improving AP by 2.26, 2.06 and 1.71 percentage points over the strongest comparator. These gains come at lower parameter and runtime costs than MetaDrug. Matched controls demonstrate DDI reductions with stable predictive quality. Code is available at this https URL. 

---
# Close and Sort: Exact Maximum-Likelihood Aggregation of Cyclic Rank-Two Preferences 

**Authors**: Kehan Liu, Zhaoyun Chen, Xianquan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.10538)  

**Abstract**: Pairwise probabilities arise in rank aggregation, recommendation, and human or LLM evaluation, yet converting them into a maximum-likelihood list is a weighted Kemeny problem and is NP-hard in general. Known exact algorithms for rank-two tournaments use only comparison signs and do not optimize this likelihood. We show that the weighted problem is exactly solvable when the skew-symmetric log-odds matrix $W_{ij}=\log(p_{ij}/p_{ji})$ has rank two, including when the induced preferences contain cycles. Writing $W_{ij}=\det(h_i,h_j)$ and adding the closing vector $h_0=-\sum_i h_i$ turns the objective into twice the signed area of a closed polygon. This geometry yields CloseSort, which polar-sorts the $n+1$ edge vectors, cuts after $h_0$, and returns a global maximum-likelihood order in $O(n\log n)$ time from the factors. The same geometry identifies the runner-up and gives a stability certificate beyond exact rank two. When the smallest determinant $\delta$ of two consecutive edges in the closed angular cycle is positive, it is exactly the best-to-runner-up log-likelihood gap. For $W=W_0+E$, the CloseSort order of $W_0$ has regret at most $\sum_{i<j}|E_{ij}|$ on $W$ and remains optimal whenever this sum is below $\delta$, a worst-case tight threshold. In 5,120 determinant-logit simulations, increasing the observations per pair from 4 to 64 raises the fraction on which rank-two projection outperforms the empirical plug-in optimum from 68.6% to 85.7%. On 3,600 Arena-Hard LLM-judge profiles drawn from 63 systems and ten judges, the leading rank-two component captures 89.2% of the log-odds energy. On profiles containing a directed majority triangle, CloseSort reduces normalized regret by 44% relative to Borda and improves the mean normalized objective over each of eight baselines. With factors supplied, it is 8,407 times faster than exact dynamic programming on 18-item rankings. 

---
# Track or Follow? A Jev-Compatible Multi-Agent Trading System When Web News Lags the Market 

**Authors**: Yebo Feng, Guangyan Gan, Liangrui Pan, Ruichao Liang, Cong Wu, Jiahua Xu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10414)  

**Abstract**: LLM agents increasingly trade on the Web news they read, yet much Web news about listed firms is published after, and about, the price move it reports, so a reader can agree with the market by following it instead of reading the event. We ask whether LLM judgments of Web news track the market's reaction to the reported event or follow it. To answer, we built a multi-agent trading system whose agents share only point-in-time stores and answer typed questions through the interface of Jev, a hosted typed decision model; code sets every position under a proved bound on how far text can move it. In a hash-committed test of a Jev-compatible reader built from a pinned LLM's option probabilities, item judgments track announcement-window abnormal returns (weekly Spearman 0.119), also after the LLM's release, and event types agree with the issuers' 8-K filings (AUC 0.833); Jev's judgments of news published after its own release track the reaction as well (descriptive). Exploratory analyses show where the association comes from: 72% of earnings news items appear after the issuer's filing, contributor-written items often report the price move, and from the first trade after publication the association falls to 0.023. At a weekly horizon, the confirmatory test detects no rank information over code-only features. Under attack, decision cards traced 88% of successful attacks to the planted item. We propose that market-graded tests of Web-text readers report the criterion from publication, not only from the event. 

---
# SituRecBench: Evaluating Frontier Models as Situated Recommendation Assistants in 3D Environments 

**Authors**: Jingtong Yang, Dongding Lin, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.10349)  

**Abstract**: Recommender systems are transitioning from static retrieval engines into interactive assistants grounded in physical environments. However, current studies largely neglect the combined complexities of spatial navigation, visual grounding, and multi-turn social interaction. To bridge this gap, we introduce SituRecBench, an interactive benchmark for evaluating frontier vision-language models (VLMs) as situated recommendation assistants within high-fidelity 3D environments. We reformulate recommendation as a situated assistance process in which an assistant interacts with a user while sharing the user's egocentric view. Our benchmark simulates real-world shopping environments (e.g., supermarkets and home furnishings stores), requiring assistants to elicit preferences, ground recommendations in the environment, and guide the user to relevant locations through dialogue. The benchmark evaluates dialogue quality, recommendation utility, and navigation effectiveness through a unified interactive protocol. Extensive experiments reveal persistent challenges in spatial grounding and preference elicitation, highlighting limitations in their ability to integrate global contextual information with local observations. SituRecBench provides a rigorous foundation for advancing the development of situated assistants in the real world. 

---
