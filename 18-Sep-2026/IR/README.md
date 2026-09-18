# Reasoning Quality Matters: Combating Reasoning Collapse in LLM-based Embedding Learning 

**Authors**: Zihan Gong, Xiaohan Ye, Jiangchao Yao, Jinsong Lan, Xiaoyong Zhu, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.20563)  

**Abstract**: Large Language Models (LLMs) have recently shown strong potential for producing context-rich text embeddings for retrieval. Most existing methods either treat embedding learning as passive feature extraction or exploit LLM reasoning through instruction following for better embedding optimization. However, specialization toward embedding objectives can suppress useful reasoning generation or produce retrieval-irrelevant text. We refer to these two forms of degradation as reasoning collapse. To address this issue, we propose CoFree (Collapse-Free Reasoning Embedding), a two-stage framework that progressively integrates LLM reasoning into query and document embedding optimization while preserving reasoning quality. At the first stage, CoFree applies reference-guided supervised fine-tuning to restore the reasoning ability and retain representational strength of the foundation embedding model. At the second stage, we introduce dual rewards, an embedding-oriented reward and a reasoning-oriented reward, to guarantee fine-grained reasoning of the relevance toward the embedding goal in reinforcement learning. This endpoint-coupled optimization transforms embedding learning from static alignment into a high-quality reasoning-guided search process for retrieval. Extensive experiments demonstrate the effectiveness of CoFree, with CoFree-4B achieving an average absolute improvement of 2.8 nDCG@10 points over Qwen3-Embedding-4B across 22 datasets from MTEB and BRIGHT. Online experiments in a real-world retrieval system further show consistent gains. Code, RTED, and model checkpoints will be made publicly available. 

---
# FacetCRS: Multi-Faceted Preference Learning for Pricking Filter Bubbles in Conversational Recommender System 

**Authors**: Yongsen Zheng, Ziliang Chen, Jinghui Qin, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.20175)  

**Abstract**: The filter bubble is a notorious issue in Recommender Systems (RSs), which describes the phenomenon whereby users are exposed to a limited and narrow range of information or content that reinforces their existing dominant preferences and beliefs. This results in a lack of exposure to diverse and varied content. Many existing works have predominantly examined filter bubbles in static or relatively-static recommendation settings. However, filter bubbles will be continuously intensified over time due to the feedback loop between the user and the system in the real-world online recommendation. To address these issues, we propose a novel paradigm, Multi-Facet Preference Learning for Pricking Filter Bubbles in Conversational Recommender System (FacetCRS), which aims to burst filter bubbles in the conversational recommender system (CRS) through timely user-item interactions via natural language conversations. By considering diverse user preferences and intentions, FacetCRS automatically model user preference into multi-facets, including entity-, word-, context-, and review-facet, to capture diverse and dynamic user preferences to prick filter bubbles in the CRS. It is an end-to-end CRS framework to adaptively learn representations of various levels of preference facet and diverse types of external knowledge. Extensive experiments on two publicly available benchmark datasets demonstrate that our proposed method achieves state-of-the-art performance in mitigating filter bubbles and enhancing recommendation quality in CRS. 

---
# Think Thrice Before Reranking: Multi-perspective Evidence and Reasoning Integration for Text Reranking 

**Authors**: Lijun Liu, Zhengzong Chen, Wenyan Li, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.20131)  

**Abstract**: Reasoning-based reranking with Large Language Models (LLMs) has shown promising improvements in text ranking. However, current methods predominantly rely on a single reasoning trajectory, resulting in rankings that are susceptible to reasoning errors and inherently constrained in modeling the multifaceted signals underlying document relevance. To resolve this dilemma, we propose MERIT-Rank(Multi-perspective Evidence and Reasoning Integration for Text Reranking), a framework that models complementary reasoning trajectories to improve reranking robustness. MERIT-Rank formulates a Multi-Trajectory Reasoning Space (MTRS) that evaluates query-document relevance from multiple perspectives and introduces a joint reranker that consolidates these reasoning paths into a unified ranking decision. We further develop Progressive Rank Policy Optimization (PRPO), a progressive training framework that stabilizes reasoning trajectories while continually improving ranking quality through staged optimization objectives. Experiments on both reasoning-intensive and traditional retrieval benchmarks show that MERIT-Rank consistently achieves superior performance over competitive baselines. The 4B model notably outperforms most 7B and even 32B rerankers on BRIGHT. 

---
# The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents 

**Authors**: Zhexi Feng, Ruiyi Zhang, Yongbo Yang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.20050)  

**Abstract**: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those states at five items and 80.6% at eight, against 61.4% and 72.4% for Qwen3 embedding with reranking. A matched control ranking by similarity alone reaches 66.6%, placing the gain in the set-level policy, not the computation. From frozen repository source with no gold-derived pool, the lead is 5.0 points. On AMA-Bench it answers from a 76.2% smaller answer prompt, with accuracy 2.08 points above that benchmark's own memory agent. Removing one required group from an otherwise complete set costs 12.3 and 11.1 points of repair-localization precision under two executors. Retrieval for agents is better posed as recovering what a decision lacks than re-ranking what an issue resembles. 

---
# Reproducing Transparent and Scrutable Recommendations: Exploring Open-Weight Models via Natural-Language User Profiles 

**Authors**: Noah Mamié, Laurin van den Bergh  

**Link**: [PDF](https://arxiv.org/pdf/2609.19831)  

**Abstract**: In this reproducibility study, we investigate the transparency and scrutability of recommender systems enhanced by incorporating generated natural-language user profiles that represent user preferences. The original paper explores the synthesis of user profiles from raw user-generated review text across domains such as movies and accommodations (Amazon Movies & TV, TripAdvisor). Crucially, these natural-language user profiles enable direct user interaction and intervention, allowing users to customize recommendations by correcting misattributed preferences or addressing cold-start settings. We successfully reproduce the core findings of the original study. Additionally, we extend the evaluation by conducting systematic context ablation experiments, multi-seed stability across five distinct random seeds to establish statistical reliability, and a mechanistic interpretability analysis using the nnsight framework to probe internal model representations under counterfactual profile perturbations. Our findings verify the original paper's claim that User Profile Recommendation (UPR) achieves competitive performance under its test-set reranking protocol and makes recommendations more transparent. Perturbing the natural-language profiles does change predictions, but it shifts predicted ratings uniformly across genres with no detectable genre-selective effect, leaving rankings unchanged even under direct activation steering. We trace this back to the rating-regression objective rather than the profile interface, with ranking-objective models clearly exceeding in this task. 

---
# Dense Feature Representation over Sequence Modeling: A Solution to the KDD Cup 2026 UniRec Challenge 

**Authors**: Yi Zhang, Weiliang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.19787)  

**Abstract**: We describe our 10th-place solution to the KDD Cup 2026 Tencent UniRec Challenge, industrial click-to-conversion (CVR) prediction over 34.82M records, and we ask which mechanisms actually move held-out AUC. Starting from the official PCVRHyFormer baseline, a 15-step single-variable chain raises test AUC from 0.813237 to 0.827816, and our final submission reaches 0.828535. A leave-one-out ablation from the full model attributes the gain: removing the dense-feature representation stack costs 0.0095 AUC and removing the orthogonalized optimizer costs 0.0028, while no sequence-modeling component (merged single-stream backbone, polarity channel, auxiliary head, per-token FFN) costs more than 0.0005, within or adjacent to a $\pm$0.0004 seed band. We also report a generalization hazard: the row-group train/validation split shares one time window, so validation AUC overstates the leaderboard by about 0.014; anti-memorization and high-cardinality-ID changes even invert sign against it, a divergence that traces to dump-to-dump distribution shift and survives a time-ordered re-split. Dense representation and optimization, not finer sequence modeling, drive CVR AUC at this scale, and verdicts must come from the held-out leaderboard. 

---
# Self-Evolving Search Index 

**Authors**: Sangam Lee, Wonjae Lee, Sunghwan Kim, Deogyong Kim, Jaehoon Kim, Daye Nam, SeongKu Kang, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.19656)  

**Abstract**: Information retrieval is increasingly important as LLM agents tackle complex tasks involving diverse information needs. Because retrieval relies on an index that represents each document through index keys, retrieval quality depends heavily on how effectively these keys expose the knowledge contained in each document. However, effective index representations vary across retrieval environments, making it difficult for any fixed optimization strategy to perform consistently. Yet evolving an index to its retrieval environment remains largely human-driven, requiring humans to diagnose retrieval failures, refine the optimization strategy, and reprocess the index accordingly. We propose SELF-INDEX, a framework that enables an index to self-evolve without human intervention. Its Optimizer autonomously diagnoses retrieval shortfalls, selectively revises the responsible index keys, and validates each revision before updating the index. Beyond reacting to observed retrieval demands, SELF-INDEX proactively explores additional demands through a Query Simulator, allowing the index to evolve beyond the queries already available for optimization. Across diverse corpora and retrievers, SELF-INDEX consistently improves retrieval performance while outperforming existing index optimization methods. We further show that these benefits extend to downstream applications, improving the effectiveness and efficiency of search agents and helping agent memory systems retrieve useful past interactions. 

---
# Beyond Similarity through Zero-Token Geometric Graphs for Multi-Hop RAG 

**Authors**: Zeliang Li, Xiaofen Xing, Kailing Guo, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.19622)  

**Abstract**: Multi-hop retrieval-augmented generation (RAG) requires evidence that remains relevant to a query while introducing enough novelty to bridge semantic gaps. Dense retrieval tends to concentrate on semantically similar documents, whereas graph-based alternatives often depend on costly Large Language Model (LLM) entity extraction and may propagate through noisy connections. We introduce Geometric Gain Graph RAG (G$^3$RAG), a document-only framework whose offline graph construction uses no LLM calls or generated tokens. G$^3$RAG assigns each edge a geometric gain score, $\cos\theta \cdot \sin\theta$, that jointly captures directional consistency and orthogonality between document representations. A density-aware topological penalty suppresses highly connected hubs, while single-step controlled diffusion expands from filtered query seeds toward complementary evidence. We evaluate G$^3$RAG on MusiQue, 2WikiMultiHopQA, and HotpotQA using Nv-embed-v2 and Qwen3-8B-embed. G$^3$RAG obtains the best average F1 and answer-document hit rate among the evaluated graph-based baselines in both embedding settings, with gains of up to 4.26 F1 points in average performance and 5.76 points on MusiQue. It also removes the graph-construction token cost incurred by entity-based graph methods. These results show that geometric structure can support efficient multi-hop evidence discovery without LLM-based graph construction. Code is available at this https URL 

---
# Algebraic Retrieval: Composable Search for Agents 

**Authors**: Damian Delmas  

**Link**: [PDF](https://arxiv.org/pdf/2609.19482)  

**Abstract**: Algebraic Retrieval lets AI agents compose search strategies at query time. Relevance criteria, eligibility constraints, and ranking preferences can be expressed together in a mathematical query. The query surface exposes available operations, so an agent can combine them for the question at hand and revise a program after inspecting results. We evaluate execution parity, not agent behavior or retrieval quality. Building on Programmatic Embedding Modulation (PEM), which exposes vector and score arithmetic during retrieval, we demonstrate contrastive scoring, candidate-pool reranking, and weighted ranking as composable queries, alongside executable SQL and PyTerrier counterparts. On the public 11,429-document Vaswani fixture, each program's implementations select the same document set with score differences below 1e-6; one tied pair orders differently across scoring paths. 

---
# Viveka-Insight: a cross-lingual concept graph and citation-grounded retrieval resource over the complete works of Swami Vivekananda in English and Bengali 

**Authors**: Tamal Maharaj  

**Link**: [PDF](https://arxiv.org/pdf/2609.20303)  

**Abstract**: Classical philosophical corpora pose three compounding challenges for language resources: they exist in several languages without parallel alignment, their vocabulary is remote from that of contemporary readers, and generated text over culturally sensitive material must be verifiably grounded. We present Viveka-Insight, a bilingual resource and open-source pipeline for the works of Swami Vivekananda (1863-1902): the nine-volume English Complete Works and the ten-volume Bengali Vani o Rachana, two related but non-parallel corpora of about 15 million characters. Four layers are released: (i) a structure-preserving parse (32,694 paragraphs, 168,842 sentences) with per-paragraph anchors deep-linking into the published editions; (ii) a cross-lingual concept graph of 8,362 language-agnostic concepts with 87,518 relation-typed paragraph-concept and 55,872 concept-concept edges, in which canonical English labels act as a string-equality key linking Bengali and English passages with no parallel data; (iii) a bilingual alias inventory of 60,850 surface forms (30,053 English, 30,797 Bengali); and (iv) a human-annotated set of 200 paragraph-concept edges judged by three annotators, released with all per-annotator judgments. We report known-item cross-lingual retrieval over 194 verified rendered lecture pairs (Recall@10 0.86 in both directions), a 30-question audit of citation integrity and modern-question bridging, and a human study placing concept-extraction precision at 0.60 under strict two-annotator consensus (Cohen's kappa = 0.61). The extractor's confidence weight is calibrated: restricting to weight >= 0.8 raises precision to 0.71 while retaining 98% of concept-bearing paragraphs. Precision is markedly lower in Bengali than English (0.54 vs 0.68), locating the weakness in exactly the half that cross-lingual access depends on. The design transfers to other multilingual classical corpora. 

---
# Intrinsic Sequence-Likelihood Confidence in Retrieval-Dominated Extractive QA: Two Pre-Specified Negatives, and What They Do and Do Not Attribute 

**Authors**: Gunwoo Lee, Changmin Sung, Sang-Hwan Gwak, Ina Kim, Ji-Young Choi, Kyong-Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.19942)  

**Abstract**: In extractive document question answering whose questions were generated from the passages that contain their answers -- so that retrieval recovers 92-99.8% of what any mode combination could reach, whatever its absolute accuracy -- confidence-driven mechanisms have little to gain. Fine-tuning an open language model on a specialized domain corpus yields a model whose own confidence is a tempting control signal: it could decide which queries warrant further adaptation, and which answers to trust. We evaluate both uses under criteria fixed before the runs were executed, across four 7-9B model families whose adaptation moved closed-book F1 by at most +0.03, and both fail: a distillation trigger on all four families, under its pre-specified three-step transfer budget, and a routing-and-abstention policy in its single-model pilot. Retrieval alone recovers 92-99.8% of best-case combined accuracy under every correctness criterion we test, leaving routers no meaningful gain. The sequence-likelihood signal is insufficient relative to that mode -- area under the receiver operating characteristic curve 0.65-0.81 under the registered criterion -- before adaptation as well as after, unchanged by scalar recalibration and not consistently improved by token-level temperature rescaling. And the finer diagnostics depend on the correctness criterion and on answer length; on the three adapted combinations where we could test it, selector ablations show no statistically detectable downstream benefit from the confidence term on any seed; on Gemma, removing it changes the selector from failing to passing both registered criteria. The usable product is a set of pre-specified negatives with their dependencies made explicit. 

---
# Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies 

**Authors**: Hang Xiao, Chuhong Xu, Kainan Zhou, Gangzhen Qian, Lu Yi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19844)  

**Abstract**: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; the exact saved artifact must pass the full production path. A subsequent follow-up is excluded because it did not satisfy the preregistered evidence-completeness gate and is treated only as future work. We release the benchmark, failure-preserving contract, incident provenance, and governance controls needed to prevent infrastructure behavior from being misreported as model behavior. 

---
# FINSKILLOPS: A Self-Evolving Multi-Agent System for SEC Filing QA 

**Authors**: Yanzhang Ma, Zhenghan Tai, Hanwei Wu, Sizhe Guan, Jianliang Lei, Hailin He, Chaolong Jiang, Jijun Chi, Tung Sum Thomas Kwok, Bohuai Xiao, Jingrui Tian, Xinlu Wu, Xingao Zhan, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma, Sicheng Lyu, Tianshuo Yan, Junhao Zhu, Yaqian Xu, Lei Ding, Yufei Cui, Ziquan Liu, Boyu Han, Hengli Liu, Ling Zhou, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.19680)  

**Abstract**: Financial QA systems are typically improved before deployment through better retrieval, prompting, or agent coordination, leaving their reliability behavior fixed thereafter. In practice, new SEC-filing questions repeatedly expose heterogeneous errors in period, entity, evidence use, and calculation. Existing self-improvement methods can turn failures into new behaviors, but offer limited control over where a correction should apply or which previously correct answers it may break. We therefore frame post-deployment improvement as controlled behavioral maintenance: recurring failures should become scoped skill patches, and each patch should earn deployment with- out introducing regressions. We instantiate this view in FINSKILLOPS, a multi-agent system for SEC filing QA. FINSKILLOPS derives reusable skills from evidence-grounded, typed failure diagnoses and governs them through targeted validation, protected-case regression checks, negative controls, and versioned replacement or retirement. Across six financial QA benchmarks, a single frozen skill registry achieves the highest verdict-weighted correctness and reference consistency among the evaluated systems. Evolved skills raise correctness from 3.70 to 4.55 on our enhanced benchmark. In a separate 12-round operational study, only six of 33 proposed skills are promoted, while the monitoring non-correct rate falls from 20.0% to 12.5%. These results establish controlled skill scope, admission, and lifecycle management as the foundation for reliable self-improvement. 

---
# Semantic Layer Induction from Raw Telemetry via Hierarchical LLM and RAG Abstraction 

**Authors**: Yuanzhe Jia, Ali Anaissi  

**Link**: [PDF](https://arxiv.org/pdf/2609.19615)  

**Abstract**: Modern applications generate massive volumes of raw telemetry data, but translating those noisy, heterogeneous event streams into actionable business insights remains a fundamental challenge. Data engineers and analysts expend substantial effort reconciling semantic discrepancies, hand-crafting parsing logics, and maintaining fragile mappings between raw data and business KPIs. In this paper, we present an end-to-end framework that fully automates the construction of a business semantic layer from application raw logs. Our approach introduces a two-stage semantic abstraction: first, high-level business features are identified via LLM inference augmented with domain-specific industry knowledge; second, fine-grained business nodes are derived through a structured pipeline comprising data refinement, hybrid retrieval, multi-stage filtering, semantic clustering, and canonical naming. Evaluation on production-scale telemetry demonstrates that our system improves human-assessed semantic quality from 50 to 80+ on a 100-point scale, reduces maintenance effort by 80%, filters out 74% of noise, and achieves 0.87 Cohen's kappa via an integrated LLM-as-Judge evaluation, enabling continuous, scalable quality assurance. Overall, our work distinguishes itself from prior work by addressing the novel problem of business semantic layer induction from raw telemetry, operating without labeled training data or manual rule engineering. 

---
# FootprintRAG: Visual Analytics for Evidence Context Refinement in RAG-based Scientific Literature Exploration 

**Authors**: Xingyu Liu, Yu Dong, Qizhen Yu, Shiyu Cheng, Zhe Wang, Guan Li, Guihua Shan, Dong Tian, Christy Jie Liang, Quang Vinh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2609.19601)  

**Abstract**: Retrieval-Augmented Generation (RAG) is increasingly used to ground large language model (LLM) outputs in scientific literature. However, in open-ended literature exploration, the evidence context used for generation is often produced through hidden retrieval, reranking, assessment, and filtering steps. Users may receive retrieval summaries without knowing how the system constructed the evidence context, which evidence units were retained or discarded, or whether potentially useful evidence was excluded before synthesis. We present FootprintRAG, an LLM-agent-powered visual analytics system for evidence context refinement in RAG-based scientific literature exploration. The core idea is to treat the RAG evidence context as an explicit, inspectable, and revisable analytical object before generation. FootprintRAG parses scientific literature into text and figure evidence units, expands an initial query into parallel query variants, retrieves and assesses evidence across iterative rounds, and surfaces ERS-ranked supplementary candidates from the corpus-level evidence space. Through coordinated views, the system connects retrieval trajectories, evidence-state revision, and provenance-aware summary generation into a user-steerable workflow. We evaluate FootprintRAG through two case studies, a user study, and a workflow-level comparison with representative RAG systems. The results show that FootprintRAG helps users compare retrieval directions, revise candidate evidence, recover potentially overlooked evidence, and trace generated summaries back to supporting evidence units. FootprintRAG is available at this https URL. 

---
# CliniCIRCA: A Modular LLM Framework for Constructing Longitudinal Mental Health Patient Journeys from Raw EHR Narratives 

**Authors**: Aiwei Ivy Zhang, Nimra Ishfaq, Mohit Chandra, Santiago Alvarez Lesmes, Adam Coscia, Khatiya Chelidze Moon, Xiaohan Ding, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2609.19585)  

**Abstract**: In mental health care, reasoning over patient journeys is a key task for clinicians. Yet these journeys, encompassing a longitudinal progression of biological, psychological, and social events, are often spread across disparate unstructured text narratives, making temporal recovery challenging. We present CliniCIRCA, a multi-stage LLM framework for Calendar-anchored, Imprecision-aware Reconstruction of Clinical Annals. To our knowledge, CliniCIRCA is the first to temporally classify clinical events across unstructured discharge summaries without event-level timestamps. From 14,882 MIMIC-III mental health admissions, we first construct a benchmark of 52 discharge summaries on which CliniCIRCA produces 15,891 temporally tagged events. After correcting 629 errors based on a clinician-in-the-loop evaluation, we produce verified gold-standard labels. Finally, the corrected timelines drive a temporally grounded summarization stage that compresses each source 1.52 times into a date-grouped chronological record. We then scale the framework to generate 1,000 silver-standard timelines and evaluate them as training data. Compared with zero- and few-shot prompting, instruction tuning generally improves five open-weight models on event extraction, temporal tagging, and summarization across silver and clinician-verified evaluations. 

---
# SCOUT: Sim-to-Real Text-Based Person Retrieval by Embedding-Space Prediction over Frozen Video Features 

**Authors**: Abdarahmane Traoré, Andy Couturier, Éric Hervet  

**Link**: [PDF](https://arxiv.org/pdf/2609.19483)  

**Abstract**: Text-based person retrieval under a sim-to-real gap (synthetic training data, a real-image gallery) is usually tackled with costly fine-tuned cross-encoders. We ask whether a frozen-encoder system can compete. We present SCOUT, which casts cross-modal retrieval as prediction in embedding space. A trainable predictor maps the patch tokens of a frozen video encoder into the embedding space of a frozen text encoder under a bidirectional InfoNCE objective, and no encoder is fine-tuned in the base model. The video encoder is V-JEPA, the text encoder is EmbeddingGemma, and the predictor is initialized from a Qwen3.5-0.8B decoder. We make three findings. First, the best frozen text encoder is simply the one whose geometry best matches the video features. A training-free alignment score ranks three candidate text encoders in the same order as their retrieval accuracy on our held-out split (Spearman $\rho = 1.0$); a fourth, LLM-based encoder shows the rule is metric-dependent, holding for a neighborhood-overlap score ($\rho = 0.8$) but not for a linear probe ($\rho = -0.2$). Second, two precision-targeted levers, parameter-efficient ExPLoRA adaptation of the video encoder and a training-free attribute-decomposed reranker built on a vision-language model, improve the top-rank precision that otherwise limits the frozen system, adding 2.2 points of leaderboard R@1. Third, a local-versus-public calibration study explains which interventions transfer to the real domain. On AI City Challenge 2026 Track 4 the full retrieve-fuse-rerank system reaches 84.25 mAP@10 on the final leaderboard, while a single frozen model submitted alone reaches 60.63. Our trained components cost about 95 GPU-hours. CMP, the dataset authors' fine-tuned cross-encoder that trains for sixteen GPU-days, is one fusion member of the full system, not an alternative. Code and annotations: this https URL 

---
# Beyond Private Training: The New Landscape of AI Privacy 

**Authors**: Sean Culatana, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.19456)  

**Abstract**: Retrieval-augmented systems increasingly rely on vector indexes that may retain deleted items in their search graph. Existing deletion interfaces can prevent deleted identifiers from appearing in returned results while still computing distances to their embeddings during graph traversal. We formalize this distinction as output safety versus traversal safety, and introduce TSD-AUDIT, a framework for auditing and enforcing traversal-safe deletion in graph-based approximate nearest-neighbor retrieval. On Faiss IndexHNSWFlat, native filtering leaves the number of distance computations unchanged relative to unfiltered search; at a 70% deletion rate, trace-faithful replay detects deleted-vector scoring in all 100 audited queries. Code inspection of hnswlib's mark_deleted path reveals the same scoring-before-liveness pattern. TSD-AUDIT enforces an alive-before-scoring invariant, repairs connectivity using only live candidates, and emits per-query scored-trace certificates that an independent verifier can check against the deletion snapshot. Under region-targeted deletion, TSD-AUDIT improves Recall@10 over native filtering by 4.3--42.2 percentage points across deletion fractions from 0.5 to 0.9, while remaining comparable under random deletion. These results show that output-only deletion audits can miss process-level exposure: auditing deletion in vector retrieval requires accounting for the vectors scored during search, not only the identifiers returned. 

---
# Characterizing Web Search by Conversational LLM Agents: From Search Decisions and Strategies to Results and Responses 

**Authors**: Mahsa Amani, Seungeon Lee, Abhisek Dash, Asmaa El Fraihi, Yunah Jang, Elisabeth Kirsten, Qinyuan Wu, Krishna P. Gummadi, Manish Gupta, Abhilasha Ravichander, Muhammad Bilal Zafar, Soumi Das  

**Link**: [PDF](https://arxiv.org/pdf/2609.19244)  

**Abstract**: Conversational LLM agents increasingly rely on Web search, yet the end-to-end lifecycle of agentic search remains poorly understood. We present the first study of Web search across four major conversational platforms (ChatGPT, Claude, Grok, and DeepSeek), combining real-world user interactions (invivo) with controlled experiments using the same platform's models by their APIs (invitro). We investigate the quality of agentic decisions to invoke Web search, their strategies to formulate queries, the potential domain preferences in the search results they receive, and the choices they make when transforming search results into grounded responses. We find that Web-search decisions vary substantially across platforms and models, while more frequent Web-search invocation does not necessarily yield better response quality. We further show that conversational agents employ different complex querying strategies and that platform specific search engines return search results from their preferred domains. Finally, although responses are largely grounded in search results, some claims rely on uncited search results, raising concerns about attribution and reliability. Our findings have important implications for the design of future AI agents and Web search tools optimized for conversational retrieval. 

---
# VisKG-LM: Compiling Knowledge Graphs into Visual Memory for Multiple-Choice Question Answering 

**Authors**: Yixin Peng, Er Jin, Shiwei Luo, Diego Collarana, Stefan Decker  

**Link**: [PDF](https://arxiv.org/pdf/2609.19158)  

**Abstract**: Knowledge graphs are usually integrated into question answering by encoding a retrieved subgraph with a graph neural network and fusing it with the language model in the online inference path. The same subgraph is therefore re-encoded from scratch every time a pair is scored, across training epochs, seeds, and evaluation runs, even though the knowledge graph never changes. We ask whether the retrieved knowledge graphs can instead be compiled once, offline, and then accessed as read-only memory. VisKG-LM shows that it can, by decoupling graph encoding from language reasoning. It serializes each retrieved candidate-specific subgraph as Relation-Labeled Paths and renders the result as an image whose two-dimensional layout preserves the branching structure of the paths. Each image is encoded once, offline, and cached for reuse. At inference, the language model contextualizes the question and candidate from text alone, and only its final layer consults the cached visual memory, reading both its global layout and its local relational detail. The graph information thus enters only after the text has been understood. On the test sets of CommonsenseQA, OpenBookQA, and MedQA-USMLE, VisKG-LMimproves over GreaseLM by $1.2$, $0.8$, and $4.3$ points, respectively, while matching or surpassing GraphVis, a $7$B vision-language model, with only about $400$M online parameters. Against a matched text-only control that receives the identical Relation-Labeled Paths, it gains $4.2$, $6.5$, and $5.1$ points across the three benchmarks. These gains show that the complete visual-memory interface adds value beyond path textualization alone and support compiled visual memory as an alternative to online graph propagation. 

---
# Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data 

**Authors**: Gregory M. Dickinson  

**Link**: [PDF](https://arxiv.org/pdf/2609.19153)  

**Abstract**: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,668 and 7,001 opinions. For each task the analysis approximates the best stoplist any expert could build, removing each of roughly 18,500 candidate words and measuring the effect directly. Three findings follow: generic stoplists in common use fall below the no-removal baseline in every test; even optimized stoplists are statistically indistinguishable from removing nothing; and meta-models trained on word-level features cannot predict which removals help, so list curation has nothing to target. The method generalizes to any inherited preprocessing default, and the result is a caution specific to interpretable legal text-as-data: a step that silently reshapes which features a model sees can distort the very doctrinal and ideological signal such research exists to recover. Leaving stopwords in place is a question of measurement validity. 

---
# What Users Think of Generative AI: A Cross-Platform NLP Analysis of Trust and Friction in App Store Reviews 

**Authors**: Md Jafrin Hossain, Umme Nusrat Jahan, Shouvaggo Sharif Shammo  

**Link**: [PDF](https://arxiv.org/pdf/2609.19151)  

**Abstract**: Generative AI (GenAI) applications have achieved rapid consumer adoption, yet little large-scale research examines user-perceived quality, trust, and adoption barriers. We present one of the first cross-application analyses of app store reviews for six major GenAI applications (ChatGPT, Gemini, Microsoft Copilot, Claude, DeepSeek, and Perplexity), comprising 17,012 English-language reviews from Google Play and the Apple App Store. We combine BERTopic topic modeling with RoBERTa sentiment classification and evaluate cross-application differences using chi-square, Kruskal-Wallis, and multinomial logistic regression with Bonferroni correction. Both components are validated against human coding using a stratified sample of 300 reviews. Results show that negative sentiment concentrates in advertising (91%), authentication (89%), server reliability (83%), and subscription pricing (73%). Sentiment differs significantly across applications, with Claude exhibiting the highest negative sentiment (47.7%) alongside a strongly enthusiastic user base, indicating statistically significant polarization. These findings are robust despite unequal review counts across applications. As exploratory observations, a subset of DeepSeek reviews raised geopolitical and data privacy concerns related to its Chinese origin, while a proposed Trust Friction Score summarizes application-specific trust and usability barriers into interpretable dimensions. The study provides validated and actionable evidence on user trust, usability, and adoption barriers in consumer generative AI applications. 

---
