# FINALLY: A Dataset Recommender System for Recommender-Systems Research 

**Authors**: Louis Owie  

**Link**: [PDF](https://arxiv.org/pdf/2609.08941)  

**Abstract**: Dataset selection shapes the empirical conditions under which recommender-system algorithms are evaluated, yet existing tools provide limited support for constructing complete dataset sets that jointly satisfy experimental constraints and set-level selection objectives. To address this problem, I developed FINALLY, a web-based dataset recommender for constructing configurable dataset sets for offline recommender-systems evaluations. FINALLY combines required datasets, candidate-pool restrictions, metadata filters, configurable target-set sizes, Random selection, and diverse and non-diverse strategies based on adapted Effective Covariance and Convex Hull objectives. I evaluated FINALLY through 420 recommendation runs across ten systematically varied configurations. All evaluated dataset sets satisfied the applicable target-size, duplicate-avoidance, snapshot-membership, required-dataset, and metadata-filter requirements. All 40 deterministic strategy--configuration combinations were reproducible. Both the Effective-Covariance-based and Convex-Hull-based strategies produced the expected diverse-versus-non-diverse score ordering in all ten configurations. Under their corresponding objectives, the diverse strategies produced scores above all 30 configuration-specific Random results, whereas the non-diverse strategies produced scores below all 30 Random results. These results establish technical consistency for the evaluated FINALLY workflow and show that the implemented strategies follow their intended optimization directions within the investigated configuration space. They do not establish the scientific suitability, global optimality, or practical superiority of the generated selections. 

---
# Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems 

**Authors**: Maximilian Schall, Sedigheh Eslami, Markus Krimmel, Antoine Chaffin, Louis Milliken, Bo Wang, Denis Bykov  

**Link**: [PDF](https://arxiv.org/pdf/2609.08887)  

**Abstract**: Evaluating first-stage retrievers in large-scale production RAG requires a benchmark that pairs a large-scale corpus with a large set of agent-reformulated search queries based on real user queries and their conversation threads, and that labels many relevant documents per query. No existing public benchmark evaluates this setting: large-scale collections typically provide only a small number of evaluation queries, whereas benchmarks with many queries generally contain only millions of documents. Moreover, most benchmarks assess human-written queries, while the first-stage retrievers in agentic RAG pipelines serve machine-written reformulations whose distribution differs from human search behavior. To overcome these evaluation gaps, we introduce Q2D-Web (Query2Doc-Web), a large-scale agentic retrieval benchmark consisting of a 190M-document web corpus and 70k agentic search queries in ten languages, reformulated from real-world user queries in production systems. Q2D-Web provides three sets of fixed relevance judgments: agent citations, production rankings, and a combined set that unions both signals and adds LLM-based judgments of unlabeled pooled documents to reduce false negatives. We benchmark 13 retrievers including lexical, dense, and late-interaction models and find that their relative ordering is largely insensitive to the choice of judgment set, while diverging substantially across topical domains, query languages, and query types. To enable fast evaluation, we also study subcorpus sampling as an approximation to full-corpus evaluations. Retaining a third of the corpus, selected by reciprocal rank fusion over pooled retriever runs, preserves the full-corpus model ranking under the combined judgments while raising absolute Recall@1000 only by 3 to 7 points. The public leaderboard is accessible under: this https URL 

---
# REDSI: Addressing the Reproducibility and Evaluation Consistency of Differentiable Search Indexing for Document Retrieval 

**Authors**: Vivien Nicolas, Hicham Randrianarivo, Pascale Sébillot, Caio Corro  

**Link**: [PDF](https://arxiv.org/pdf/2609.08860)  

**Abstract**: The differentiable search index (DSI) framework (Tay et al., 2022) has become the de facto baseline for generative retrieval. However, DSI is hard to reproduce: no public implementation covers all three original document identifier types (atomic, naive, semantic), reported results vary widely, and the ubiquitous NQ320K dataset is built from Natural Questions through diverse and underspecified preprocessing. We introduce ReDSI, the first open-source DSI implementation supporting all three identifier types, together with a parameterizable and well-documented NQ320K construction pipeline.
Experimentally, we achieve results that are competitive with or stronger than previous DSI baselines. Moreover, we conduct extensive experiments under model downscaling, covering retrieval effectiveness, parameter efficiency, training methods and decoding strategies, opening novel directions for future research. 

---
# PDMR: Passage-Driven Multi-ID Document Retrieval 

**Authors**: Smail Oussaidene, Mohand Boughanem  

**Link**: [PDF](https://arxiv.org/pdf/2609.08762)  

**Abstract**: Generative Retrieval (GR) models map queries directly to document identifiers, replacing conventional retrieval over external sparse or dense indexes with autoregressive identifier generation. However, most generative retrieval frameworks rely on a single-identifier assumption, mapping each document to a single target sequence. This forces the model to represent all document content with one sequence. Since documents are often multi-faceted, this can lead to lossy representations and reduced robustness to query variation, where multiple query intents must compete for a single generative access path.
In this work, we introduce Passage-Driven Multi-ID Retrieval (PDMR), a generative retrieval framework that represents documents through multiple passage-level identifiers. PDMR segments each document and assigns one identifier to each selected passage, which provides multiple semantic entry points for retrieving the same document. This multi-entry representation allows the model to align queries with specific semantic facets, thereby reducing the dependence on a single document-level target. To address the supervision ambiguity of this one-to-many mapping, we formulate training as a multi-target learning problem and explore an objective function designed to distribute probability mass across multiple valid passage-level identifiers.
We evaluate PDMR on NQ320K and MS MARCO Document. On NQ320K, PDMR improves over strong generative and non-generative baselines on Recall@1 and MRR@100. On MS MARCO Document, PDMR achieves the best Recall@1 and MRR@10 among the reported methods, while remaining competitive on Recall@10. Controlled ablations further show that passage-level supervision, identifier design, training-query augmentation, and multi-target learning contribute complementary gains. 

---
# Individual Text Corpora Predict User-Specific Knowledge: Benchmarks of Individualized Knowledge Simulation 

**Authors**: Christoph Wigbels, Ali Abusaleh, Markus T. Jansen, Alexander Mehler, Manuel Schaaf, Markus J. Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.08532)  

**Abstract**: This study examines whether individual text corpora (ICs) from search histories can be used to simulate individual knowledge. We collected ICs from 316 adults, who answered 36 multiple-choice knowledge items, and compared several large language models (LLMs) on this task, of which only Qwen3-1.7B proved viable. After task-specific fine-tuning via Low-Rank Adaptation (LoRA), Qwen3-1.7B outperformed both participants and a representative German norm sample on publicly available items. On non-public questions, however, the LLM performed worse than our participants, suggesting possible training data contamination for the public questions. When integrating ICs into retrieval-augmented generation to predict individual responses, LLM-participant Match accuracies significantly exceeded chance, which demonstrates a detectable individual knowledge signal. The probabilities assigned to the participants' answers were, however, low and far below the probability of correct answers, indicating poor calibration toward individual response patterns. Knowledge-gap prediction was sub-optimal, though it improved for corpora exceeding five million tokens. We discuss our entropy based evaluation benchmarks as calibration indices for individualized knowledge simulation. 

---
# SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching 

**Authors**: Lin Guan, Jia-Qi Yang, Zhishan Zhao, Jiaqi Huang, Hangyu Wang, Longbin Li, Beichuan Zhang, Haonan Jiang, Jinan Ni, Xiangyu Fan, Xiaowen Li, Ziyao Ren, Yuhang Qi, Xiaolong Zhu, Xuanyuan Luo, Qiwei Chen, Yi Cheng, Lele Yu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08443)  

**Abstract**: Modeling long-term user behavior is central to sequential recommendation and billion-scale industrial recommender systems, yet production ranking models operate under strict latency, memory, communication, and training-throughput constraints. At the 100K scale, the challenge extends beyond attention complexity: raw sequence features must be stored, transferred, and repeatedly processed during training and online serving. Existing approaches based on history truncation, multi-stage behavior retrieval, compressed lifelong histories, or train-short/infer-long extrapolation either weaken end-to-end optimization or retain substantial length-dependent cost. We present SequenceO1, an end-to-end framework for ultra-long user behavior sequence modeling, deployed at full traffic on Douyin with histories of up to 100K interactions. SequenceO1 follows a compress-then-reason design. Its Sketch Attention (SA) uses learnable prototypes and prototype-wise normalization to compress the raw history into a fixed-size, target-agnostic user representation. Target-conditioned Stacked Target-to-History Cross Attention (STCA) then models complementary time scales: a recent 10K suffix for short-term interests and the compact sketch for long-term preferences. To make training and inference practical, SequenceO1 combines low-rank user representation caching, multi-request user-level batching, pipeline lift, and a fused FlashSA kernel to amortize feature storage, communication, and computation across targets, training instances, and consecutive requests. Production experiments show consistent offline and online gains, while the compact cached sketch retains most of the benefit of directly scaling end-to-end sequence ranking to 100K. These results provide a practical model-system approach to efficient attention, sequence compression, and scalable long-sequence and long-context recommendation systems. 

---
# Exploring Bottom-Up Clustering for Creating Semantic IDs 

**Authors**: Leah Woldemariam, Sudhanshu Garg, Taha Belkhouja, Charles Kim-Yip, Ali Sahami  

**Link**: [PDF](https://arxiv.org/pdf/2609.08310)  

**Abstract**: The success of generative retrieval has largely been attributed to the use of Semantic IDs, which improve over arbitrary item-level identifiers such as hashes by capturing the semantics of items. The main challenges faced when constructing Semantic IDs, however, is in mapping each identifier to a unique product and capturing information valuable to downstream tasks. Past works have appended additional codewords to de-duplicate item identifiers and utilized residual quantization to create hierarchical clusters. In this work, we present an algorithm for generating Semantic IDs that ensure the identifiers are both unique and preserve the structure of the original embedding. Key to our work is the use of bottom-up clustering to preserve local structure in the embedding space, improving the clustering quality of the resulting Semantic IDs and their utility for downstream generative retrieval. 

---
# Cassette: Case-to-Case Structural Distillation for Efficient Legal Case Retrieval 

**Authors**: Yanran Tang, Ruihong Qiu, Hongzhi Yin, Xue Li, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08185)  

**Abstract**: Legal case retrieval (LCR) is an essential tool for not only assisting legal practitioners to efficiently retrieve precedents but also enabling ordinary individuals to find valuable legal case information without relying on expensive professional legal services. Our previous work CaseLink demonstrated the effectiveness of using case to case graph structures to improve retrieval accuracy. However, its high computational cost during inference on large-scale legal databases limits its practical use in real-world settings. The main inefficiency comes from constructing test time graphs and computing pairwise term frequency similarities of cases. This process has O(n^2) complexity for n legal cases, making the runtime prohibitive as the number of candidates grows. For example, the retrieval time for one query on a database (COLIEE2022) with 1,563 candidate cases is more than 500 milliseconds, while the runtime would increase drastically to more than 3,500 seconds for a database (LeCaRDv2) with 55,192 candidate cases. To further enhance the retrieval performance while achieving a significant speed-up, in this extension paper, Cassette framework is proposed with a distillation strategy involving ranking objective and eigen-matching objective for an effective transfer of knowledge from a powerful and well-trained heavy teacher retriever to a lightweight and efficient hybrid student dual encoder. Specifically, the student query encoder is implemented as a multilayer perceptron model designed for fast online processing, whereas the student candidate encoder adopts a GNN architecture, suitable for an offline manner within the case database. Extensive experiments are conducted on three benchmark datasets, and the results verify the effectiveness of the ranking distillation while achieving high efficiency. The code has been released on this https URL. 

---
# Noēsis: Deterministic-First Retrieval with Two-Tier Context Hydration for Factuality-Critical Queries on Small Local Models 

**Authors**: Nicola Cogotti  

**Link**: [PDF](https://arxiv.org/pdf/2609.07663)  

**Abstract**: A wrong number is worse than no answer. Across factuality-critical domains -- audience metrics, scheduling and rights in media; dosages and lab values in healthcare; figures and citations in finance and legal -- a confident but fabricated value is more damaging than an honest admission of uncertainty. Yet this is the dominant failure mode we observe on small local language models: even when correct evidence is present in context, models fabricate plausible numbers and timestamps. Recent work characterizes a real limit of this regime: below 7B parameters, the bottleneck of retrieval-augmented generation (RAG) is not retrieval quality but context utilization.
We present Noesis, the deterministic-first query plane of the Noesis architecture, which makes every deterministic judgment before generation. Its mechanisms follow from the ingestion architecture (subject of a separate patent application): (a) a producer-side fact layer rendering precomputed metric facts verbatim without ranking; (b) positional addressing with deterministic cross-source alignment, resolved ahead of query time at zero LLM cost; (c) provenance scoping as an attribution constraint with multi-tier named-reference routing; and (d) two-tier context with model-triggered verbatim hydration.
Across four ablations, a 2B model reaches parity with a 35B model on factual integrity (exact values in all runs; zero confabulated numbers on absent-entity traps); structured retrieval beats flat RAG by +11.4 points at 2B; skeleton-only context preserves quantitative answers at 20-30% smaller prompts; and hydration recovers verbatim narrative in ~8s versus ~29s. Two properties matter for regulated domains: each query resolves in a single generation call, and every reported value is traceable to its exact source and position by construction. 

---
# Open Tabular Insight Extraction: Where Do We Stand, and Where Should We Go? 

**Authors**: Daniel Gomm, Maarten de Rijke, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2609.07629)  

**Abstract**: Democratizing access to the knowledge held in large corpora of tables such as data lakes is emerging as a central research challenge. Research in this space is advancing and broadening in scope, increasingly supplying the components to satisfy a person's insight need end-to-end. Yet these efforts remain fragmented across communities that frame the problem under their own conventions, such as table question answering, text-to-SQL, and data analysis agents, with works six times as likely to cite within the same task label as across labels. To bring these communities onto common ground, we establish a holistic framework for this pursuit, which we refer to as Open Tabular Insight Extraction (OpenTI). We formalize OpenTI from first principles around the analytical knowledge a person needs, the procedure for deriving it from a corpus of tables, and how well a result serves the person who sought it. In doing so we consolidate frameworks and terminology across information retrieval, natural language processing, machine learning, databases, and human-computer interaction, and apply this grounding in a systematic review and analysis of systems and benchmarks that work towards OpenTI. We find that current systems do not cover the end-to-end scope of OpenTI, mainly focusing on the analysis itself, and that benchmarks are largely unfit for evaluations in an open setting as inputs presuppose knowledge of tables, and validation mechanisms do not match the setup. Finally, we distill a research agenda towards OpenTI systems, evaluation, and interaction paradigms that surface the insights users need. An interactive companion to our paper is available at this https URL. 

---
# EigenLI: Spectral Approximations to Late Interaction 

**Authors**: Archish S, Sabyasachi Basu, Ankit Garg, Ravishankar Krishnaswamy, Kirankumar Shiragur  

**Link**: [PDF](https://arxiv.org/pdf/2609.07561)  

**Abstract**: Late-interaction models such as ColBERT achieve strong effectiveness by representing each document with many token-level vectors, but this expressivity leads to large indexing cost, storage footprints and expensive MaxSim scoring. We show that late-interaction representations exhibit an intrinsic low-rank structure: document token embeddings concentrate in a low-dimensional subspace that preserves most of the retrieval signal. Leveraging this observation, we introduce EigenLI, a spectral approximation framework that compresses late-interaction representations via document-specific low-dimensional subspaces. Unlike clustering or pooling methods, EigenLI identifies the dominant eigendirections of each document and uses them to construct reduced interaction representations. Empirically, $k$-EigenLI with $k \le 32$ outperforms k-means and Ward clustering based pooling methods on ColBERTv2 and AnswerAI-ColBERT-small; GTE-ModernColBERT exhibits a different tradeoff at $k=32$, where clustering methods perform better. The same spectral construction also yields EigenLI-SV, an ANN-compatible single-vector representation derived from the second-order summary of the reduced structure. Across multiple datasets and all three text models, EigenLI-SV consistently outperforms comparable single-vector surrogates such as MUVERA. 

---
# Uncertainty Quantification for LLM Agents: A Taxonomy, an Evaluation Protocol, and an Empirical Study 

**Authors**: Moule Lin, Qizhen Lan, Shuhao Guan, Weipeng Jing, Jiexin Fan, David Gregg, Goetz Botterweck  

**Link**: [PDF](https://arxiv.org/pdf/2609.07395)  

**Abstract**: Large language models (LLMs) are no longer deployed only for single-turn conversation but increasingly act as agents that plan, call tools, retrieve evidence, maintain memory, and interact over long horizons, often together with other agents through multi-turn conversations. Therefore, knowing when to trust the agentic system is a prerequisite for safe deployment. However, existing work on quantifying uncertainty for LLMs was built almost entirely for single-turn question answering. This paper argues that errors and uncertainty arise from multi-turn conversations, environments, and tools rather than from a single-turn question answering setting. It comes late, however, and is compounded in a single score that is too coarse to represent the unreliability. We organize the literature with a three-axis taxonomy, (1) what the uncertainty is, (2) how it is estimated, and (3) where uncertainty arises during an agent pipeline. We investigate step-level and trajectory-level calibration and show with a simple counterexample that the first does not imply the second. Experiments on real agent traces across four models and up to a 50-step budget show that the proposed metric and reporting protocol (Trajectory-Checkpoint Expected Calibration Error, TC-ECE) can be computed and that step errors are coupled along a trajectory. We find that confidence estimates from the agent's own responses do not consistently outperform a simple baseline. The experiments also show that averaging all trajectories together can hide overconfidence at later stages, which becomes visible when results are analyzed across different horizons. In simpler terms, this paper identifies where the uncertainty comes from in the agentic system pipeline, how to teach agents to know when they are wrong, and why one confidence number is not enough. 

---
# Matryoshka Hash Representations for Model-Aware Compact Semantic Retrieval 

**Authors**: Peichun Hua, Yunming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.07276)  

**Abstract**: Retrieval-augmented generation (RAG) depends on dense retrieval: each document is stored as a learned vector, and a query is answered by finding its nearest neighbors in that vector space. Keeping one full-precision vector per document is the dominant index cost at corpus scale, so retrieval systems replace each vector with a short code of a few bytes---a step called quantization. Standard quantizers such as product quantization (PQ) pick the code that reconstructs the original vector most closely.
A single code is even more useful if it serves several byte budgets at once: when its short prefixes are each directly searchable, a deployment can set its efficiency--quality operating point without re-encoding the corpus. But training all prefixes under one objective makes the early bits a compromise across budgets---short codes improve while the full-width code degrades. Quantization to low-bit representation, such as binary codes, further sharpens the conflict. We introduce Matryoshka Hash Representations (MHR), a two-stage procedure that separates full-width training from prefix organization. MHR first learns a longer binary code, then freezes the model and trains additional zero-initialized residual code adaptors for directly searchable prefixes. Documents are stored at one bit per coordinate, while queries keep continuous logits like PQ to attain sufficient expressivity. We implement the search process with FAISS FastScan. Trained on MS MARCO and zero-shot transferred to seven BEIR datasets, MHR reaches .5561 NDCG@10 and .6535 Recall@100 at 32 bytes, surpassing the best baseline of the same budget. The advantage is more pronounced in lower budgets. The same code also strengthens two common pipelines: shortlisting candidates for full-precision reranking, and pruning a low-storage graph index such as LEANN. 

---
# Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones 

**Authors**: Yuchen Wang, Feng Niu, Qing Tan, Junting Lu, Baoxin Wu, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2609.07273)  

**Abstract**: Industrial ranking models for recommendation have scaled feature interaction and sequence modeling separately; recent architectures such as HyFormer and MixFormer unify both in a stackable backbone. Real-world recommender systems, however, nearly always require multi-task learning, yet existing unified architectures confine multi-task modeling to shallow post-backbone towers, leaving the backbone without task-aware information flow. We propose MORE (Multi-task cO-evolving Ranking modEl), which embeds multi-task information flow inside the backbone, enabling task-specific signals to co-evolve with sequence and feature representations at every layer rather than in a post-hoc fusion. It introduces Anchor Tokens that persist across backbone layers: Shared Anchors encode cross-task commonalities, while Private Anchors capture task-specific priors. In each block, Anchor Tokens (1) read task-conditioned signals from behavior sequences, (2) mix with non-sequential features under a task-boundary mask, and (3) refine per-task representations through independent branches; as blocks stack, each task obtains a differentiated representation refined through all backbone layers.
Experiments on large-scale industrial datasets show that MORE consistently outperforms baselines across all tasks under comparable parameter and FLOPs budgets, and scales well with model size. Online A/B tests on Momo, a leading Chinese social discovery platform with tens of millions of monthly active users, yield 3% improvement in usage duration, 3.6% in interaction rate, and 2% in deep-chat rate. MORE is deployed in production with request-level shared computation reducing scoring latency by about 30%. 

---
# Query-Aware Token Budgeting for Efficient Late-Interaction Visual Document Retrieval 

**Authors**: PS Rishi, Rajeev Ranjan Dwivedi, Vinod K Kurmi  

**Link**: [PDF](https://arxiv.org/pdf/2609.07262)  

**Abstract**: Late-interaction visual document retrievers preserve fine-grained page evidence by storing many token embeddings per page, but the resulting storage and query-time interaction costs make large-scale deployment expensive. Pooling document tokens before indexing offers a natural remedy, yet static pooling must decide which visual evidence to preserve before the query is known. We study an alternative: a heavily compressed hot-path index generates candidates, after which query-aware token budgeting operates on the original token sets of the shortlisted pages. We formulate this stage-two selection as a budgeted MaxSim coverage problem, show that a clipped version is monotone submodular, and compare coverage-only, cluster-guided, token-wise, and marginal-gain policies. On ten ViDoRe tasks with ColModernVBERT, direct static pooling reduces macro normalized discounted cumulative gain at rank five from 0.6309 without compression to 0.4738 at a thirty-two-fold pool factor. Under the same candidate-generation regime and a pool-factor-eight-equivalent reranking budget, token top-k recovers 93.93 percent of the full-token score, while greedy marginal-gain selection recovers 98.39 percent. Held-out and leave-one-dataset-out evaluations yield positive greedy improvements over token top-k on every dataset. The latency analysis reveals two useful operating points: token top-k for interactive retrieval and the naive greedy implementation as a quality upper envelope. Together, these results show that late-interaction visual retrieval benefits from query-aware allocation rather than query-agnostic pooling alone. 

---
# EAGER: Enrich-and-Align Generative Query Recommendation from Clicked Items in E-commerce Search 

**Authors**: Shuwei Yuan, Mingqian Ding, Luxin Liu, Rong Xiao, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2609.07143)  

**Abstract**: E-commerce platforms increasingly display clickable query suggestions alongside items in the user feed, enabling users to refine or expand their intent without manually reformulating queries. Existing approaches either mine suggestions from historical logs -- limited to past behavior and blind to long-tail, personalized intents -- or rely on off-the-shelf LLMs whose lack of platform-specific knowledge yields fluent but generic queries disconnected from real click behavior. We propose EAGER (Enrich-and-AliGn gEnerative Query Recommendation), a two-stage framework for generating query suggestions from clicked items. In the enrichment stage, supervised fine-tuning (SFT) follows a four-stage curriculum that scales information richness (from item-only to user-conditioned) and reasoning depth (from direct to chain-of-thought). Each stage incorporates rationale augmentation, diversity regularization, and self-distillation. In the alignment stage, we post-train via GRPO with a hybrid reward of multiple rule-based business signals and a preference-aware click reward. Extensive offline experiments and online A/B test demonstrate the effectiveness of EAGER, which has been deployed in production at a major e-commerce platform. 

---
# Tracing Query Expansion Effects through Sparse Autoencoder Features 

**Authors**: Fangan Dong, Weiran Shi, Zhiwei Xu, Xuri Ge, Ben He, Xin Xin, Zhumin Chen, Ying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.06968)  

**Abstract**: Query expansion (QE) is a critical technique in information retrieval that enriches underspecified queries with additional textual context. However, its effect is often unreliable in modern dense retrieval, especially for strong off-the-shelf retrievers without retraining. Existing studies mainly examine expansion quality, semantic drift, or retrieval outcomes, but rarely explain how QE changes dense retrievers internally. In this work, we trace QE effects through sparse autoencoder (SAE) features. Using paired original and expanded queries, we decompose layer-wise retriever representations into sparse latent activations, identify QE-related latents from expansion-induced activation shifts, and interpret them with natural-language descriptions and retrieval cases. Our analysis shows that effective QE induces layer-concentrated changes in sparse latents aligned with retrieval intent and entity attributes, rather than only perturbing final query embeddings. SAE-based activation steering further validates these latents improve retrieval more consistently than random interventions or vanilla QE across four benchmarks, suggesting that SAEs can explain QE effects and offer a lightweight option for precise retrieval behavior modulation without query rewriting or retriever fine-tuning. 

---
# FunnelAudit: Responsibility Auditing in Multi-Route Recommender Systems 

**Authors**: Jie Li, Dudu Luo, Jiayang Niu, Ke Deng, Yongli Ren  

**Link**: [PDF](https://arxiv.org/pdf/2609.06964)  

**Abstract**: Multi-route recommender systems combine retrieval, allocation, fusion, and ranking, making individual inclusions and exclusions difficult to audit. Route overlap can hide effects from one-at-a-time ablations, while freezing downstream stages produces counterfactuals inconsistent with serving behavior.
We introduce FunnelAudit, an executable framework for incident-level responsibility auditing. An accountability contract specifies the disputed Top-K event, controls and owners, permitted reference actions, and replay semantics. FunnelAudit evaluates every permitted control configuration and applies graded actual responsibility to find the smallest outcome-preserving contingency that makes each control pivotal. Its certificate records the contingency and paired serving executions needed to verify the judgment. We instantiate the framework in two-stage, nine-route funnels using fixed union, weighted quota allocation, or weighted reciprocal-rank fusion, followed by SASRec ranking.
Across 258,809 user-target incidents from three real interaction datasets, 4.24-16.24% admit a responsible control. Among responsible incident-control pairs, 92.55-99.64% require a nonempty contingency, so single-control ablation recovers only 0.36-7.45%. Policies differing in factual outcomes on only 0.31-2.39% of incidents yield 21.44-54.05% Jaccard distance between responsible-route sets on matched exclusions. Independent replay reproduces all 9,121,792 checked target-world outcomes; exhaustive search and a generic mixed-integer linear program agree with every sampled judgment. These findings demonstrate the importance of explicit serving semantics and checkable witnesses for recommender accountability. 

---
# Measuring GEO Visibility: Prompt Corpora Define the Answer Market 

**Authors**: Olivier Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2609.06811)  

**Abstract**: GEO (generative engine optimization) visibility scores aggregate source appearances, citations, or brand mentions in generated answers. The prompt corpus selects the situations evaluated, while weights determine their relative importance. Together they define an "answer market" that need not represent actual user demand.
Prompt wording can alter retrieval, competing sources, and generated answers. Scoring then requires identifying the appearances, citations, or mentions of interest. If a language model performs this task, its instruction can change the score assigned to an unchanged answer. Our critical survey examines how these choices help define what a GEO score measures. It draws on research into whether indicators measure the intended phenomenon, total survey error, and information retrieval evaluation. The framework specifies situation annotation, prompt formulations, execution conditions, weights, and scoring rules.
When weights are unknown or remain to be chosen, the framework reports sets of admissible scores. It distinguishes values compatible with data and assumptions about a target population (partial identification) from variation across weighting conventions (normative sensitivity).
A citation alone does not establish a source's contribution. The article defines a comparison of answers generated with and without a source in a controlled documentary context, distinct from an intervention on the full engine with competing sources. The framework is supported by reproducible calculations. No new experiments are reported; its general empirical validity remains to be assessed. 

---
# Who Anchors AI Overviews in Health? Baidu, Google, and the Geography of Authority 

**Authors**: Mingyue Zha, Ho-Chun Herbert Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.06798)  

**Abstract**: Artificial intelligence is being rapidly incorporated into traditional search systems, yet scant work audits the information disparities across platforms, geography, and languages. We address this gap by comparing Google and Baidu's AI Overview systems for health queries, and measure informational anchors that emerge. Auditing 1,920 health queries across 12 countries and 4 languages, we find that Google and Baidu exhibit vertical integration, routing users toward their own company platforms in AI Overviews rather than a diverse set of primary sources. Smaller, lower-localization countries receive fewer domestically sourced references for health queries. Issuing the same query in a country's official language rather than English raises the share of locally sourced citations approximately 3.5- to 13.5-fold. Comparing queries across health topics of varying severity and controversy, including Traditional Chinese Medicine as an example, we also show that health disclaimers are multidimensional and vary across language and culture. We discuss how generative search influences access to health information, and the urgent need for culturally-aware oversight of these systems that influence critical health decisions. 

---
# EviMap: Evidence-Grounded Hierarchical Topic Maps for Exploring Unlabeled Corpora 

**Authors**: Zhiyin Tan, Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06664)  

**Abstract**: Research teams and organizations often explore unfamiliar free-text collections, from survey comments and reviews to reports and domain documents, before labels, queries or coding schemes exist. At this stage, the first thematic map shapes what users notice, prioritize and carry into downstream analysis, so it should be trusted only insofar as it can be verified. Existing options force a trade-off between scale and verifiability. Qualitative coding preserves evidence but is slow. Search presupposes a query. Clustering and topic models scale but produce labels users must interpret. One-shot large language model (LLM) summaries are fluent yet difficult to reproduce or audit. We present EviMap, an interactive system providing researchers and practitioners with an auditable thematic overview of such corpora. Guided by model-generated context describing the corpus and hypothesized stakeholder concerns, EviMap extracts within-document evidence phrases and organizes them, rather than whole documents, into a three-level map of aspects, groups and fine-grained topics. Embedding-based clustering narrows the search space for finer semantic judgments by the LLM. Each node traces back to supporting phrase spans, so documents link to topics through evidence they contain and users can audit labels against the original text. Users can start from a top-level corpus map, drill into topics, inspect highlighted evidence in original documents, and combine two topics to find documents discussing both. We demonstrate this workflow across six heterogeneous corpora spanning 2,108 to 101,699 documents, with a comparison against flat and hierarchical LLM baselines. By grounding every label in verbatim source spans, EviMap makes a topic map not just readable, but verifiable. Code, demo video, and interactive dashboard are available at this https URL. 

---
# TaxoConf: Taxonomy-Guided Automatic Conference Program Organization 

**Authors**: Daomin Ji, Zhifeng Bao, Junhao Gan  

**Link**: [PDF](https://arxiv.org/pdf/2609.06604)  

**Abstract**: Conference program organization, the task of assembling accepted papers into a technical program, is labor-intensive. Papers must be grouped into topically coherent sessions under hard operational constraints, and existing methods rarely achieve both at once. To address this problem, we present TaxoConf, a system that organizes conference programs around a conference-specific topic taxonomy. TaxoConf constructs a canonicalized multi-parent taxonomy over the accepted papers and represents each paper by its frontier of most specific topics. It derives a specificity-weighted optimal-transport distance between papers from this taxonomy, then solves a binary integer programming problem that assigns papers to sessions by minimizing within-session distance subject to all hard constraints. On benchmarks built from four 2025 conferences, TaxoConf attains the highest session coherence (4.62 out of 5) and the closest agreement with human-curated sessions (NMI 0.761) while incurring no constraint violations. TaxoConf has also been deployed to generate the technical program of SIGIR 2026, where the organizers accepted the initial output with only a few edits, preserving 91.8% of oral assignments and the entire poster program while requiring no correction of hard-constraint violations, and an on-site survey of 47 attendees rated the overall program quality 4.26 out of 5. The system is publicly available at this https URL. 

---
# Relevance is not enough: A Communication-Oriented Retrieval System for Consequential Scientific Question Answering 

**Authors**: Avina Nakarmi, Naga Datha Saikiran Battula, Anthony Diaz, Aritra Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2609.06222)  

**Abstract**: AI systems increasingly answer scientific questions about health, safety, and the environment. But most retrieval-augmented generation systems are tuned to provide factually correct, on-topic answers rather than to help non-experts understand what those answers mean for their lives and decisions. We focus on consequential scientific questions whose results directly shape people's lives and study them through a public water-quality communication system, where residents and community leaders interpret the findings and choose actions. Their experiences show that on-topic answers can still be insufficient without explanation and context and that the emotional weight of risk information cannot be ignored. Our system first classifies each question by reasoning type (for example, causal versus policy-based), then generates follow-up questions to identify missing evidence and retrieve it. One component clearly distinguishes between what is known and what is uncertain, while another rewrites scientific details into accessible language, using persona-based styles, such as a caring neighbor or an administrative official, to adapt tone and readability. Ablations on over $160$ questions show that the system uses $\textit{an order of magnitude less context}$ and, in several configurations, improves human-rated completeness. A completeness metric co-designed with community members and a fine-tuned learned judge reveal that standard relevance scores explain about $1\%$ of variation in human completeness ratings, and even the tuned judge only moderately aligns with humans, indicating that completeness is a distinct human-centered objective that current metrics do not reliably capture. 

---
# Closing the Long-Short View Gap in Sequential Recommendation without Cached History 

**Authors**: Lingfeng Shi, Chengkai Huang, Lina Yao, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2609.06219)  

**Abstract**: Sequential recommenders are typically trained on long user histories to capture rich behavioral signals, yet serving with training-length sequences is often impractical due to real-time efficiency constraints. Directly using only recent behaviors leads to a severe performance drop. To bridge this gap, existing approaches compress user histories into persistent per-user states, storing and retrieving them at inference time; while effective, they impose non-trivial infrastructure overhead and offer little remedy in cold-start scenarios. In this paper, we empirically identify two structural flaws rooted in geometric properties and dataset sparsity, and propose a novel two-stage framework to close the long-short-view performance gap. Specifically, in the first stage, we replace the commonly used dot-product with angular similarity scoring and leverage a modified softmax to counter prefix position bias. In the second stage, we fine-tune only bias and LayerNorm components, which are universal to standard sequential backbones, for further improvement. Both stages are guided by carefully designed learning objectives. Extensive experiments on two representative backbones across three public datasets demonstrate the effectiveness of our proposed framework. 

---
# ExpertLens: Visualizing Embedding Spaces for Post-Hoc Explainability in MoE Enhanced Retrievers 

**Authors**: Effrosyni Sokli, Isaac Roberts, Alexander Schulz, Barbara Hammer, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2609.06155)  

**Abstract**: Neural models, including dense retrievers, have been widely adopted in Information Retrieval (IR), often delivering state-of-the-art performance. Despite their effectiveness, these models operate as black boxes, limiting the interpretability of their ranking decisions. Existing post-hoc explainability methods for neural rankers primarily focus on feature-level attributions, which can be insufficient to capture the complexity of learned embedding spaces. In this work, we propose ExpertLens, a post-hoc explainability framework for Mixture-of-Experts (MoE)-enhanced dense retrievers that shifts focus from local scalar feature importance to representation-level global interpretability. ExpertLens leverages discriminative embedding space visualizations jointly with automatically extracted Concept Activation Vectors to reveal how expert routing drives embedding space formulation and retrieval effectiveness. Experiments across five IR benchmarks and two MoE-enhanced dense retrievers show that expert routing consistently improves embedding space structure, positioning queries and their relevant documents into better-defined geometric neighborhoods. Analysis of expert subspaces further reveals general-purpose dominant experts, along with minority experts exhibiting distinct linguistic specialization, with subspaces arranged according to multi-semantic concept similarity. Our code is publicly available. 

---
# Visual Analysis of LLM-based Entity Resolution from Scientific Papers 

**Authors**: Siyu Wu, Yi Yang, Weize Wu, Ruiming Li, Yuyang Zhang, Ge Wang, Huobin Tan, Zipeng Liu, Lei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2609.06037)  

**Abstract**: This paper focuses on the visual analytics support for extracting domain-specific entity from extensive scientific literature, a task with inherent limitations using traditional named entity resolution methods. With the advent of large language models (LLMs) such as GPT-4, significant improvements over conventional machine learning approaches have been achieved due to LLM's capability on entity resolution integrate abilities such as understanding multiple types of text. This research introduces a new visual analysis pipeline that integrates these advanced LLMs with versatile visualization and interaction designs to support batch entity resolution. Specifically, we focus on a specific material science field of Metal-Organic Frameworks (MOFs) and a large data collection namely CSD-MOFs. Through collaboration with domain experts in material science, we obtain well-labeled synthesis paragraphs. We propose human-in-the-loop refinement over the entity resolution process using visual analytics techniques, which allows domain experts to interactively integrate insights into LLM intelligence, including error analysis and interpretation of the retrieval-augmented generation (RAG) algorithm. Our evaluation through the case study of example selection for RAG demonstrates that this human-machine collaborative approach improved single-document entity resolution accuracy by approximately 30%. 

---
# Do All Nodes Benefit Equally from Knowledge Graphs? Adaptive Node-Aware KG Fusion for Recommendation 

**Authors**: Jaehyun Park, Minseo Jeon, Daewon Gwak, Sunuk Kim, Hanvit Lee, Jinhong Jung  

**Link**: [PDF](https://arxiv.org/pdf/2609.05909)  

**Abstract**: KG-aware recommendation has been widely studied to alleviate data sparsity by using knowledge graphs (KGs), which represent items, entities, and their relations as graphs and provide item-side knowledge. However, existing methods incorporate item knowledge without considering how much each user or item node should rely on it. As a result, they apply KG signals indiscriminately across nodes, even to nodes whose collaborative filtering (CF) signals from the interaction graph (IG) are already reliable. In this paper, we propose AdaKG (Adaptive Node-Aware KG Fusion), a novel KG-aware recommendation method that adaptively adjusts the contribution of auxiliary knowledge for each node. Since user-item interactions and item knowledge provide different types of signals, directly mixing them can distort the CF signals. To avoid this, AdaKG separately encodes the IG and KG with view-specific encoders, allowing each view to capture its own information. It then estimates how strongly each node should rely on item knowledge by measuring the stability of its CF signals under small adversarial perturbations, assigning a larger KG contribution to less stable nodes. Finally, AdaKG adaptively aligns the IG and KG embeddings in a shared space and fuses them according to the estimated node-wise reliance. Through experiments, we show that AdaKG achieves strong performance compared with its baselines and the effectiveness of our adaptive fusion strategy. 

---
# What Price Fairness? Evaluating Energy - Fairness - Accuracy Trade-off in Recommender Systems 

**Authors**: Abhirup Mitra, Oleg Lesota, Antonela Tommasel  

**Link**: [PDF](https://arxiv.org/pdf/2609.05759)  

**Abstract**: Fairness-aware recommender systems aim to mitigate systematic imbalances in recommendation outcomes, including how visibility, relevance, and opportunities are distributed among users, items, and providers. However, these systems are usually evaluated in terms of accuracy and fairness alone, while their computational and environmental costs remain largely invisible. This omission matters because fairness interventions may affect the cost of recommendation in different ways. Training-time methods modify model optimization, post-processing methods add computation at inference time, and both may depend on the model, dataset, hardware, and deployment setting. We examine whether provider-side fairness in recommendation comes with a measurable green cost. We compare in-processing, graph-level reweighting and post-processing interventions across multiple models, two datasets, and two hardware settings. We measure recommendation quality, provider-side exposure, and energy consumption separately across training and inference stages. Our results show that the green cost of fairness is not uniform, post-processing shifts cost to repeated serving, while in-processing and graph-level methods avoid re-ranking overhead but vary substantially across models, datasets, and hardware. Findings call for evaluating fairness-aware recommendation as a three-way trade-off between accuracy, fairness, and computational cost. 

---
# Overview of ROMCIR 2026: The 6th Workshop on Reducing Online Misinformation through Credible Information Retrieval 

**Authors**: Marcos Fernández-Pichel, Marinella Petrocchi, Kevin Roitero, Marco Viviani  

**Link**: [PDF](https://arxiv.org/pdf/2609.05684)  

**Abstract**: In the digital online ecosystem, we are surrounded by distinct forms of information pollution, posing significant threats to both individuals and society. Fake news, for instance, wields power to sway public opinion on matters of politics and finance. Deceptive reviews can either bolster or tarnish the reputation of businesses, while unverified medical advice may steer people toward harmful health practices. In light of this challenging landscape, it has become imperative to ensure that users have access to both topically relevant and factually accurate information that does not warp their perception of reality, and there has been a surge of interest in various strategies to combat misinformation through different contexts and multiple tasks. The purpose of the ROMCIR Workshop, for some years now, is precisely that of engaging the Information Retrieval community to explore potential solutions that extend beyond conventional misinformation detection approaches. Key objectives include identifying subjective and objective factors associated with information credibility and truthfulness, respectively, and integrating such factors as fundamental dimensions of relevance within IR Systems (IRSs), achieving early detection of misinformation, and ensuring that the search results retrieved are not only truthful but also explainable to the users of IRSs. Moreover, it is essential to evaluate the role of generative models such as Large Language Models (LLMs) in inadvertently amplifying misinformation problems, and how they can be used to support IRSs, together with the contribution that the human-in-the-loop paradigm can have in this context. 

---
# Evaluating and Improving Evidence-Grounded Fact-Checking in LLMs via Multi-Round Evidence Ablation 

**Authors**: Xingyu Deng, Mingzi Cao, Nikolaos Aletras, Xi Wang, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2609.08943)  

**Abstract**: Automatic fact-checking systems assess the veracity of claims given evidence from relevant documents. Large Language Models (LLMs) have demonstrated strong performance in fact-checking due to their general reasoning capabilities. However, it remains unclear whether they faithfully make use of the evidence provided to reach veracity judgments or rely on parametric knowledge. To investigate this, we introduce Fact-Ablated Evaluation (FAE), a new evaluation framework that iteratively ablates the cited evidence to assess whether LLMs revise their predictions accordingly. Our empirical results show that current off-the-shelf LLMs as fact-checking systems rely more on their parametric knowledge than on the evidence provided. To bridge this gap between prediction accuracy and evidence grounding, we propose REAL (Rigorous Evidence Ablation Learning), a training framework that promotes evidence-dependent verification through counterfactual evidence supervision for the LLM-as-verifier models. Experiments on four fact-checking datasets across different domains demonstrate that models trained with REAL obtain superior evidence-dependent capabilities compared to standard fine-tuned models. Our findings highlight that strong fact-checking performance can still coexist with weak evidence dependency, while REAL encourages veracity predictions to remain more closely tied to the availability of supporting evidence. 

---
# Tool Retrievers Are Underestimated: Annotation Expansion Reveals True Capability 

**Authors**: Yanyu Zhu, Chenheng Zhang, Shaoshen Chen, Hoilam Pao, Yufei zhang, Jiajun Chai, Dongnian Wang, Zhaoyu Hu, Guojun Yin, Wei Lin, Hai-Tao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.08327)  

**Abstract**: In open-world scenarios with massive and evolving tool repositories, tool-augmented large language models rely on a retriever to surface relevant tools for a given query. Because such repositories often contain many tools that implement the same functionality, a single query can often be resolved by several distinct but functionally equivalent tool combinations, making the natural query-to-tool mapping inherently one-to-many. However, existing tool retrieval benchmarks annotate each query with a single relevant tool combination, collapsing this one-to-many mapping into a rigid one-to-one annotation and causing valid retrieved tools to be misjudged as failures. To address this, we propose ToolEX (Tool Equivalent eXpansion), a framework that automatically discovers and annotates the tool combinations functionally equivalent to the labeled ones. Applied to the 7,360-query Tool-DE benchmark, ToolEX finds that 67.9% of sub-queries admit equivalent alternatives, expanding the singular ground truth to an average of 5.3 valid combinations per query. Using the expanded benchmark ToolEQ, we re-evaluate eight base retrievers and two fine-tuned variants; metrics on ToolEQ rise substantially over Tool-DE, showing that one-to-one annotation systematically underestimates retrievers and that 30--47% of the reported fine-tuning gain is an evaluation artifact rather than genuine improvement. Applying the same pipeline to skill retrieval on SkillRet further confirms that the one-to-one problem extends beyond tool retrieval. 

---
# Evidence-Aligned Entity Verification for Hallucination Detection in Retrieval-Augmented Generation 

**Authors**: Runsong Jia, Zhen Fang, Mengjia Wu, Jie Lu, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.08267)  

**Abstract**: Hallucination detection is crucial for large language models (LLMs), as hallucinated content creates significant barriers in applications requiring factual accuracy. Current detection methods mainly depend on internal signals like uncertainty and self-consistency checks, using the model's pre-trained knowledge to identify unreliable outputs. However, pre-trained knowledge may become outdated and has coverage limitations, especially for specialized or recent information. To address these limitations, retrieval-augmented generation (RAG) has emerged as a promising solution by retrieving relevant evidence at inference time, grounding outputs beyond the model's parametric knowledge. In this paper, we target a critical and practical learning problem RAG-based hallucination detection (RHD), where RAG is employed to enhance hallucination detection by addressing information updating challenges. To address RHD, we propose a novel method Evidence-Aligned Entity Verification (EAEV), which detects entity-level hallucinations by leveraging RAG to align generated entities with retrieved evidence contexts. Specifically, EAEV evaluates entity-evidence alignment through three complementary dimensions and introduces counterfactual stability analysis to ensure robust alignments under evidence perturbations. Experiments across multiple RAG benchmarks demonstrate that EAEV achieves consistent improvements over existing methods with strong generalization capabilities. 

---
# Bridging the Semantic-Utility Gap in Multimodal RAG via Generator-in-the-Loop Alignment 

**Authors**: Zhan-Lun Chang, Dong-Jun Han, Seyyedali Hosseinalipour, Mung Chiang, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2609.08188)  

**Abstract**: Vision-language models (VLMs) augmented with retrieval-augmented generation (RAG) benefit from access to external evidence. However, standard retrievers and rerankers optimize for semantic similarity rather than answer utility, creating a preference gap: documents that appear relevant may not help the generator produce a correct answer. Motivated by this, we propose a two-stage generator-in-the-loop alignment framework that closes this gap without human document-level relevance annotations. Our framework consists of two stages: in Stage 1, a VLM generates a hypothetical text passage from the image-query pair, which is used as the retrieval query for dense text search, bridging the image-to-text modality gap. In Stage 2, a cross-encoder reranker adapted with low-rank adaptation (LoRA) is fine-tuned using answer-supervised preference pairs mined from the frozen VLM: given the dataset answer label, a candidate document is labeled positive if the VLM produces the correct answer when given that document as context, and negative otherwise. This generator-guided signal is compatible with multiple alignment loss functions, including contrastive (triplet) loss, pairwise direct preference optimization (DPO), and supervised fine-tuning (SFT), and supports periodic re-mining to refresh preference pairs as the reranker improves. Experiments on VQA-X and A-OKVQA with Qwen3.5-2B and Qwen3-VL-4B-Instruct show that our proposed framework consistently outperforms rank-order, random, and REPLUG-style likelihood baselines under various alignment losses and pool size settings, suggesting that answer-level generator feedback is an effective supervision signal for preference alignment. 

---
# Snugi-AI-v2 @ eRisk 2026 Task 2: Early Depression Detection via a Learned Stopping Policy with Sustained Confidence Gate 

**Authors**: Yuwen Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2609.08161)  

**Abstract**: We describe the Snugi-AI-v2 submission to eRisk 2026 Task 2, the second edition of contextualized early depression detection from Reddit discussions. Our central contribution is a learned MLP stopping policy trained to directly optimize ERDE50, replacing the fixed and tiered threshold strategies used in all prior eRisk Task 2 submissions. Combined with a sustained confidence gate that commits only after N=3 consecutive rounds of high policy confidence, the system reduces false positives caused by transient emotional posts without sacrificing recall. The pipeline encodes each discussion thread with a frozen MentalRoBERTa model, maps the accumulated representation to a depression probability via an MLP classifier, and delegates the timing decision to the learned policy. Our best run achieves F1 = 0.73 (Run 1) and F_latency = 0.70 (Runs 0 and 3), with a median alert round of 8 out of 500, completing the full evaluation in 1 hour 26 minutes, the fastest among all complete-submission teams. We report a systematic ablation across five runs spanning two encoder variants, four stopping strategies, and three gate values, along with negative results from GRPO policy training, BDI-II post filtering, MentalLongformer encoding, and DeBERTa ensembling. Code: this https URL 

---
# Artificial Intelligence-Assisted Digital Inventory of Cultural Heritage & Traditional Knowledge: Case for Indonesian Open Digital Library of Culture 

**Authors**: Hokky Situngkir  

**Link**: [PDF](https://arxiv.org/pdf/2609.08105)  

**Abstract**: The Indonesian Digital Library of Culture (Perpustakaan Digital Budaya Indonesia, PDBI; this http URL) is a participatory platform that has collected tens of thousands of entries on Nusantara cultural heritage through public contribution since 2007. Manual contribution faces three structural barriers: coverage (knowledge is scattered across languages and sites), integrity (open sources mix authentic documentation with noise), and completeness (subjects are recorded but their data remain shallow). This paper presents a methodological framework for autonomous, AI-based harvesting of cultural knowledge from the open web, designed to expand corpus coverage while intensifying per-entry data depth. The methodology is organised as a five-stage economic funnel: focused crawling, multilingual extraction and canonicalisation, vector encoding with blocking, agentic decision-making, and idempotent publication, under the principle of deterministic orchestration, agentic decisions. Each stage is formalised: funnel economics and optimal filter ordering; crawl-frontier dynamics as a subcritical branching process that explains the necessity of recurrent re-seeding; fact-level novelty via a containment measure; Bayesian multi-source evidence fusion with elevated publication thresholds for sacred categories; exactly-once effects via idempotent upserts and the transactional outbox; sliding-window inference budgeting with a reservation protocol; statistical quality auditing; and seed selection as submodular coverage maximisation. The framework retains four high-value human roles: curator of direction, escalation approver, quality auditor, and guardian of meaning, while machine autonomy is raised in stages. Ethical, legal, and cultural-sensitivity implications are discussed, including the architectural guarantee that the machine never overwrites human contributions. 

---
# From Citations to Contributions: LLM-Assisted Credit Scoring of Research Articles 

**Authors**: Sana Ebrahimi, Suraj Shetiya, Abolfazl Asudeh  

**Link**: [PDF](https://arxiv.org/pdf/2609.07673)  

**Abstract**: Citation-based measures of scientific influence typically treat citations as uniform signals, ignoring the different roles that cited works play in a paper's contribution. We introduce contribution-based credit scoring for research articles: a structured citation analysis that decomposes a paper's credit between its own original contribution and the prior work it builds on. Motivated by a cooperative-game view of scientific credit, we propose the contribution tree, a hierarchical framework that conserves importance across the document structure and separates original from citation-derived contribution. To make this framework scalable, we use LLMs as noisy comparative estimators of local importance. We further extend the model to article collections by propagating contributions through weighted citation graphs, yielding corpus-level contributions and normalized influence scores. Our experiments suggest that our framework captures contribution signals beyond surface-level heuristics. Our code is available at this https URL 

---
# Same Problem, Different Field: Cross-Domain Solution Import via Domain-Stripped Computational Fingerprints 

**Authors**: Eryk Kulikowski  

**Link**: [PDF](https://arxiv.org/pdf/2609.07595)  

**Abstract**: The same underlying computational problem is solved across unrelated fields under different names: recursive Bayesian state estimation appears as a "Kalman filter" in control, "Bayesian forecasting" in pharmacokinetics, and "data assimilation" in geoscience. Topical and citation-based scientific embeddings cannot see this shared problem. We distill each paper once into a domain- and method-name-stripped faceted computational fingerprint, a free-text mechanism skeleton plus controlled computational facets. We define a tunable, facet-selectable distance over it. The goal is solution import: surface cross-field pairs solving the same problem, so a bespoke implementation can be swapped for another field's standard, specialized solver. On a benchmark of 18 method families across 109 papers, the skeleton lifts cross-domain retrieval average precision over the abstract from 0.222 to 0.513, and the whole fingerprint reaches 0.557. Strikingly, four trained scientific embedders all fall below plain abstract+TF-IDF: they encode topical and citation similarity, the wrong signal for this task. The gain is the representation: the abstract-to-skeleton swap lifts every embedder, and the pipeline is one cached LLM call per paper plus a cheap embedder. An interventional re-skin / math-edit test shows the fingerprint tracks the computation, not the field. On a 501-paper wild corpus, known twins dominate the top of the ranking (23 of the top 30); with planted pairs excluded from the results, three blind LLM judges rate 3 of the top 5 and 8 of the top 30 pairs genuine import candidates, and 0 of 30 random ones. The human verification is the four executed imports: in one, an open standard solver reproduces a bespoke clinical dosing engine's output. We release the benchmark, the code, and the distillation prompt. 

---
# Where to Look and What to Use: Retrieve-Localize-Generate for Long-Term Conversational Memory Question Answering 

**Authors**: Yifan Wang, Xinkui Lin, Yongxiu Xu, Shen Gao, Ruochen Yang, Kun Huang, Yubin Wang, Jie Wu, Wei Liu, Jian Luan, Hongbo Xu, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2609.07093)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to answer questions by accessing external knowledge and has been widely adopted for long-term conversational memory question answering. However, existing methods suffer from two key challenges: (1) fragmented evidence scattered across temporally distant sessions, and (2) noisy content within retrieved sessions that triggers the lost-in-the-middle effect. To address these challenges, we propose MemLoc, a unified Retrieve-Localize-Generate framework for long-term conversational memory QA. For retrieval, MemLoc decomposes each session into multi-granularity memory units and performs query routing via an inner-memory graph with entropy-based granularity selection. It further models cross-session semantic and temporal dependencies through a cross-memory graph, enabling coarse-to-fine retrieval of top-K relevant memory candidates. For localization, we introduce a reasoning-based evidence locator trained with Self-reflective Hint Policy Optimization (SHPO), which performs progressive refinement by extracting query-relevant fragments within memory units to suppress noise and reranking across candidates to remove redundancy, producing a compact evidence set with lightweight location IDs. For generation, these IDs act as precise grounding signals that guide the LLM to the correct memory positions, mitigating the lost-in-the-middle effect while preserving original contextual integrity. Extensive experiments on four benchmarks demonstrate that MemLoc achieves state-of-the-art retrieval accuracy and response quality while maintaining efficiency. Our code is available at: this https URL. 

---
# Beyond One-Shot Expansion: Contrastive Evidence Exploration for Multi-Hop Retrieval 

**Authors**: JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.07050)  

**Abstract**: Retrieval-augmented generation (RAG) critically depends on retrieving the evidence necessary for effective reasoning. However, this remains particularly challenging in multi-hop question answering (QA), where supporting passages are often linked through intermediate entities and relations that must be progressively uncovered. Existing retrieval approaches typically rely on a single retrieval intent or one-shot query expansion, limiting their ability to adapt to newly retrieved evidence and potentially introducing noisy or redundant retrieval signals. To address these limitations, we propose a training-free multi-hop retrieval framework that integrates evidence-conditioned exploration, passage-specific contrastive refinement, and coverage-aware final ranking. During offline indexing, the framework constructs passage-specific contrastive facets that characterize each passage relative to its semantically similar neighbors, providing fine-grained signals to distinguish closely related candidates. At inference time, the framework iteratively retrieves evidence, generates probes targeting unresolved information needs, refines candidate relevance using the contrastive facets, and selects a complementary set of passages that collectively cover diverse evidence-seeking intents. Experiments on MuSiQue, HotpotQA, and 2WikiMultihopQA demonstrate consistent improvements in retrieval quality and downstream QA performance over baselines. 

---
# Customer Relationship Intelligence: Integrating CRM and MDM for Enhanced Customer Engagement 

**Authors**: Tejasvi c. Addagada  

**Link**: [PDF](https://arxiv.org/pdf/2609.06189)  

**Abstract**: This study examines how Customer Relationship Management (CRM), Master Data Management (MDM), and Customer Knowledge Management (CKM) jointly constitute a Customer Relationship Intelligence (CRI) framework for enhanced Customer Engagement (CE). A cross-sectional survey of 100 participants across retail, healthcare, IT, and telecommunications sectors was analysed using Spearman rho correlation and ordinal logistic regression (IBM SPSS). Bivariate correlations were weak and non-significant (r<0.19, p>0.06). Regression identified CRM (beta=0.717, p=0.002) and CKM (beta=0.581, p=0.009) as significant positive predictors of CE; MDM showed a positive but non-significant direct effect (beta=0.346, p=0.071). The model explained 20.5% of CE variance (Nagelkerke R^2=0.205). Parallel mediation analysis (Hayes PROCESS Model 4, 5,000 bootstrap samples) found no significant indirect effects of MDM on CE via CRM (IE=0.021, 95% BC CI [-0.072, 0.121]) or CKM (IE=0.032, 95% BC CI [-0.061, 0.126]); Hypothesis H4 was not supported. CRM and CKM emerge as the principal drivers of CE within the CRI framework, while MDM functions as a foundational data quality enabler whose strategic value is realised through its enabling effect on CRM execution and knowledge management. Findings should be treated as exploratory given the sample size and cross-sectional design. Future research should replicate with larger sector-specific samples and longitudinal designs, particularly in regulated BFSI contexts where MDM architecture is shaped by data governance mandates. 

---
# PAGR: Proof-Carrying Algebraic-Geometric Retrieval: A Quiver-, Provenance-, and Sheaf-Theoretic Framework for Grounded LLM Retrieval 

**Authors**: Xingting Wang, Min Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.06127)  

**Abstract**: Retrieval-augmented generation is usually formulated as a statistical information-retrieval problem. Graph-based variants add relational structure, but the mathematical status of that structure is often left underspecified. Three distinct questions tend to be conflated: which statements are certified as knowledge, which latent representations are useful for retrieval, and which multi-hop compositions are semantically admissible.
We propose Proof-Carrying Algebraic-Geometric Retrieval (PAGR), a framework that separates these questions mathematically. Its symbolic layer is a many-sorted relational theory generated by a typed quiver, path equations, and positive Horn inclusions. A quiver representation assigns inner-product spaces to entity types and linear operators to relations. A cellular sheaf measures local-to-global consistency. Semiring provenance records derivations and supports machine-checkable certificates.
The central principle is epistemic separation: learned geometry may rank and organize evidence, but cannot promote a hypothesis to certified ground truth. We show the certification criterion is invariant under arbitrary replacement of learned components. Further results include a conditional completeness bound, identification of the isometry group as the relevant symmetry for residual-based retrieval, a cohomological consistency diagnostic, and a bounded-bisimulation index for admissible-path expansion.
PAGR is a mathematical architecture for separating where a system should look from what it is allowed to treat as knowledge. 

---
# Evaluating Deep-Search Agents under Hierarchical Web Evidence Poisoning 

**Authors**: Zhongan Bi, Qiwen Wang, Jianrong Jiang, Jigang Ding, Wenwen Xiong, Changhua Meng, Xuanang Gao, Kepeng Lin, Changjiang Jiang, Yiang Chen, Huan Yao, Wei Wang, Zhenyu Ma, Wenhui Dong  

**Link**: [PDF](https://arxiv.org/pdf/2609.06027)  

**Abstract**: Search-augmented LLM agents are increasingly used for consumer decisions, making them vulnerable to Generative Engine Optimization (GEO) poisoning. Existing benchmarks largely measure whether manipulated content is retrieved or endorsed, but do not track whether an agent verifies suspicious evidence, revises adopted claims, or recovers before producing its final recommendation. We introduce HAE-GEO, a benchmark that tracks the full trajectory from exposure to recovery under progressively more persuasive Web poisoning. Agents interact via a multi-turn Search-Scrape interface across three attack levels (L1 direct assertion, L2 contextual camouflage, and L3 apparent corroboration), supported by a controlled corpus of 72,039 clean pages and 770 poisoned pages per level spanning 8 product categories and 154 brands. Evaluation combines deterministic behavioral measures with six semantic rubric dimensions. Evaluating 10 agents, we find three recurring patterns: evidence recognition degrades under the corroboration trap; agentic search improves final resistance without improving evidence recognition or utility; and defense prompting increases verification, yet rarely converts verification into recovery. 

---
# SurveyAgent-HKA: A multi-agent framework for scientific survey generation with LLMs and human knowledge augmentation 

**Authors**: Tong Bao, Mir Tafseer Nayeem, Yi Zhao, Davood Rafiei, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.05938)  

**Abstract**: Automatic scientific survey generation has become an important task in scientific document processing. The common approach of retrieving literature from a single source (e.g., arXiv) and generating surveys through a one-pass large language model (LLM) call often leads to limited reference coverage and, more importantly, fails to replicate the expert-driven revision process that is crucial for writing high-quality surveys. In this paper, we introduce SurveyAgent-HKA, a multi-agent framework that improves end-to-end scientific survey generation by incorporating knowledge derived from published surveys and peer-review comments. The framework decomposes survey generation into well-defined sub-tasks handled by LLM-powered agent. It first retrieves relevant papers from multiple sources and identifies key topics through clustering to construct an initial outline, which is then refined using outlines from related human-written surveys. Based on the refined outline, topic-focused papers are retrieved and re-ranked to select for drafting a well-grounded survey. Then, we identify common issues raised by experts in peer-review comments from published surveys to guide the revisions and finalize the survey. Experiments on two domains show that our approach outperforms mainstream baselines in citation quality, structural consistency, and content quality. Furthermore, our framework is efficient in both time and cost, making it a practical solution for broader AI-assisted scientific writing applications. 

---
# AtomCite: Verification and Correction of Supplied Page-Level Citations in Multi-Page Documents 

**Authors**: Chen Qian, Yimeng Wang, Yu Chen, Lingfei Wu, Andreas Stathopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2609.05802)  

**Abstract**: Large language models answering questions over multi-page documents are expected to cite the supporting pages, yet supplied citations are sometimes inaccurate, and current evaluations score citations at generation time or against text passages: no existing benchmark evaluates whether a system can verify and correct a page-level citation already attached to an answer. We propose AtomCite, an agentic framework that parses an answer into claims, checks each claim against the image of its cited page, and applies a deterministic repair policy. To evaluate it, we introduce DocCite, to our knowledge the first benchmark for systems that verify and correct page-level citations in document images. Built on MP-DocVQA and DUDE, it combines 928 validated injected instances with 2,468 candidate natural errors harvested from frontier- and efficiency-tier models, of which a two-annotator audit confirms 1,909 as genuine errors. Primary labels are assigned deterministically, not by LLM judges, with the human audit as a separate validation layer. Across three model families (Gemini, Claude, and GPT), AtomCite reaches around 93% binary verification accuracy on the injected benchmark, significantly outperforming every OCR-only condition, including a compute-matched control, and exceeding every prior text-based baseline given the same OCR text. Its repair policy lifts citation precision on the injected mix from a constructed 34% to 87-90% while retaining over 90% of correct claims. AtomCite also transfers: with frozen prompts and zero training, it raises the hallucination-detection scores of two open 7-8B models on five public benchmarks above the same models prompted as direct judges. Finally, the audit shows that noise in automatic labels biases measured verifier accuracy and can reverse system rankings, so evaluations relying only on synthetic or automatic labels risk mismeasuring verification capability. 

---
# RAGMark: A Comprehensive Framework for Benchmarking Retrieval-Augmented Generation Systems 

**Authors**: Zlatan Feric, Amir Taherin, Bin Ren, Yanzhi Wang, Jennifer Dy, David Kaeli  

**Link**: [PDF](https://arxiv.org/pdf/2609.05760)  

**Abstract**: We present RAGMark, a modular benchmarking framework for advanced Retrieval-Augmented Generation (RAG) systems targeting small-scale multi-GPU environments. RAGMark evaluates diverse RAG components, including retrievers, vector databases, prompt-processing methods, and generator models, while collecting detailed per-stage metrics such as latency, GPU utilization, memory consumption, power usage, time to first token (TTFT), throughput, and answer quality. The framework is highly extensible, separating RAG stages, timing, and resource monitoring into modular components, and is designed to efficiently sweep large configuration spaces while minimizing repeated model and database initialization overhead. Using RAGMark, we characterize five RAG workloads on open-domain QA datasets across varying retrieval depths, model scales, reranking, compression methods, and vector database configurations. We show that while autoregressive generation dominates latency in naive pipelines, context-reduction techniques shift bottlenecks across compute, memory bandwidth, and preprocessing stages. Reranking and compression produce compounding benefits: reranking reduces compression workload itself, while both jointly reduce prefill and KV-cache traversal costs, lowering energy consumption by up to 66%. We further observe strong cross-stage interactions, where small upstream context reductions cascade through downstream latency, memory traffic, and energy consumption. The RAGMark source code is publicly available at: this https URL. 

---
# A Multi-Source Ensemble Approach to Candidate Generation for Alternative Vacation Rental Property Recommendations 

**Authors**: Syed Mohammed Arshad Zaidi, Eric Rincon, Shayan Hassantabar  

**Link**: [PDF](https://arxiv.org/pdf/2609.05748)  

**Abstract**: Alternative property recommendations play a critical role in vacation rental marketplaces, helping users discover relevant options when viewing a specific listing. However, generating high-quality candidate alternatives presents unique challenges: heterogeneous inventory, geographic constraints, rapid availability changes, and long-tail property distributions. We present a comprehensive study of candidate generation (CG) approaches for vacation rental alternatives, comparing collaborative filtering, shallow embeddings, and graph neural network (GNN) methods.
Our experiments on a large-scale vacation rental platform (over 2M active properties) show that a hybrid architecture combining item-based collaborative filtering with GNN-based retrieval improves Recall@300 by 14.8% over the strongest baseline, by leveraging the complementary strengths of the two sources: collaborative filtering excels at early recall for properties with rich interaction history, while GNNs discover diverse, non-obvious alternatives and handle cold-start scenarios more effectively. As a component result, GNN-based embeddings alone substantially outperform shallow Hotel2Vec embeddings (48-68% relative recall improvement across K), motivating their inclusion in the ensemble.
Crucially, we examine how CG-stage gains carry through to the downstream ranking stage, and find that a stronger candidate pool yields higher downstream ranking quality, though attributing this effect cleanly is complicated by the coupling between candidate generation and ranker training. This recall-conversion gap is an important consideration for practitioners deploying new retrieval methods in two-stage recommendation systems. 

---
# Recovering Temporal and Geographic Signals from Language Model Embeddings 

**Authors**: Esteban Feuerstein, Victoria Klimkowski, Juan Manuel Ortiz de Zarate, Federico Hernán Suaiter  

**Link**: [PDF](https://arxiv.org/pdf/2609.05721)  

**Abstract**: Understanding whether language-model embeddings encode structured real-world information is important for both representation analysis and information retrieval. We study this question for temporal and geographic signals using a simple projection-based method that operates directly on output embeddings. Given a small set of seed examples, the method defines an axis in embedding space and ranks texts or entities by their projection onto that axis. Our approach is fully black-box and model-agnostic: it requires only embeddings, without access to model weights, internal activations, auxiliary probes, or additional training. This makes it applicable to modern embedding models available only through APIs and provides a lightweight way to analyze whether temporal and spatial dimensions are present in their representation spaces. We apply the method to temporal and geographic datasets and find that embedding projections recover meaningful chronological and spatial structure. These results provide evidence that output embeddings encode signals relevant to time and space, while also offering a practical tool for interpretability and for downstream temporal and geographic information retrieval tasks, such as temporal ordering, geographic ranking, and tagging. 

---
# Better Together: Complementary Query Rewriting Under a Strong RAG Baseline 

**Authors**: Sara Shanian, Xiaoqin Yi, Pavlo Ruban, Kurt MacDonald  

**Link**: [PDF](https://arxiv.org/pdf/2609.05637)  

**Abstract**: A popular way to improve Retrieval-Augmented Generation (RAG) is to rewrite the user's question into several variants and search with all of them. We test whether this actually helps once the underlying search is already strong. Under one fixed, competitive pipeline (BGE dense retrieval, cross-encoder reranking, and MMR diversification), we compare four query-rewriting strategies (S1-S4) against two strong LLM baselines (HyDE, Query2Doc) on three datasets (HotpotQA, AmbigNQ, and the 512K-document EnterpriseRAG-Bench) over three seeds with paired-bootstrap significance tests. Our headline result is that rewriting alone is at best competitive with a strong baseline, but combining methods yields outsized gains because different strategies fail on different questions. A post-hoc union of four methods (S1+S3+S4+HyDE) improves HIT@10 over the baseline by +12.5 points on enterprise data (51.70 vs 39.22), and a five-method union reaches 52.98 (+13.8). Budget-matched controls capture only ~40% of this gain, confirming that complementarity, not retrieval budget, is the primary driver. On HotpotQA the union adds +1.6 to +1.8 points (p<0.001), saturating the all-method oracle; on AmbigNQ the same fusion hurts (-2.4 below the best solo, p<0.001), and we analyze when and why. Because rewriting is expensive, we evaluate in simulation a confidence-gated router that runs rewriting only when the baseline's own top-1 score is low. It captures about half of the enterprise full-merge gain (+4.3 HIT@10) while paying rewriting cost on <40% of queries, and automatically declines to rewrite on AmbigNQ. A downstream answer-quality evaluation confirms the router improves F1 by +1.92 (p<0.01) at roughly 40% of the expansion cost. In short: treat query rewriting as a complementary coverage source, applied through cost-aware routing, not as a standalone replacement for a strong baseline. 

---
