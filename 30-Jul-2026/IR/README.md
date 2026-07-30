# Improving Item Discoverability in e-Commerce Search via Related Intent Generation 

**Authors**: Ji Xin, Xiao Xiao, Ishan Bhatt, Vinesh Gudla, Trace Levinson, Raochuan Fan, Shishir Kumar Prasad, Prakash Putta, Tejaswi Tenneti  

**Link**: [PDF](https://arxiv.org/pdf/2607.27172)  

**Abstract**: Traditional search systems are optimized to retrieve items that strictly match a query, often prioritizing precision over recall. In e-commerce marketplaces and particularly grocery, this paradigm is limiting, as user satisfaction and commercial outcomes depend heavily on the discoverability of substitute, complementary, and thematically related items. In this paper, we present a scalable system for discovery-augmented search that leverages intent-conditioned recall expansion. Our approach generates implicit user intents to expand candidate recall while maintaining relevance.
The system addresses the cost-quality tradeoff of generative retrieval through a two-stage hybrid architecture. First, we leverage closed-weight large language models (LLMs) to maximize discoverability for head queries. To extend these benefits to tail queries, we then introduce a finetuned small language model (SLM), trained via LoRA adapters and teacher-student distillation. We evaluate the system using a rigorous dual framework: (a) LLM-as-a-judge metrics validated against human preferences for semantic quality, and (b) end-to-end session-level purchase analysis. Results demonstrate that our approach improves both intent generation quality and downstream retrieval effectiveness, extending discovery coverage from approximately 60% to 80% of query traffic at roughly 30% of the teacher model's inference cost, offering a viable path for deployment in large-scale marketplaces. Beyond relevance gains, discovery-augmented search may serve as a marketplace-balancing mechanism, giving long-tail and emerging supply an opportunity for query-conditioned exposure. 

---
# KAMR: Grounding Generation via Knowledge-Aligned Multi-hop Retrieval 

**Authors**: Xiaochen Wang, Yuan Zhong, Haoyu Wang, Ting Wang, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2607.27136)  

**Abstract**: Graph-based retrieval-augmented generation increasingly relies on multi-hop retrieval, where answering a query requires composing multiple connected knowledge-graph triplets. However, existing retrievers often rank triplets independently via global semantic matching. Moreover, many multi-hop benchmarks provide only final answers, which limits supervision for query--triplet alignment and causes structurally necessary but weakly aligned facts to be missed. To address these issues, we propose a knowledge-aligned multi-hop retriever, KAMR, which distinguishes anchor triplets that are strongly constrained by the query from connected triplets that are weakly aligned yet structurally linked to the anchors. To mitigate the lack of query-triplet alignment supervision, we build a partial alignment dataset by masking triplet elements and prompting an LLM to generate corresponding queries, and optimize two contrastive objectives for pair-level and element-level matching. At inference time, KAMR retrieves anchors globally and then expands locally to collect connected evidence. Across four benchmarks, three LLM backbones, and fourteen baselines, KAMR consistently improves multi-hop retrieval and downstream question answering performance. 

---
# Learning from the Future: Privileged Self-Distillation for Sequential Recommendation 

**Authors**: Jiakai Tang, Yang Zhang, See-Kiong Ng, Xu Chen, Wen Chen, Jian Wu, Han Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27055)  

**Abstract**: Sequential recommenders are commonly trained with one-hot next-item labels under a causal (prefix-only) objective aligned with inference. While deployment-compatible, this supervision offers little insight into relative preferences among non-target items. Yet logged interaction sequences contain an additional supervisory source: interactions following the target often reveal how user intent evolves, making the target easier to interpret. We treat these future interactions as training-only privileged information, available during learning but not at inference. This raises a natural question: can future interactions provide richer supervision while keeping training aligned with inference-time prediction?
We propose Privileged Self-Distillation (PSD), a framework that separates learning-time information from inference-time information. PSD applies two attention masks to the same backbone: a future-aware view yields a privileged teacher distribution conditioned on past and future interactions, while a prefix-only view yields the student distribution used for deployment. Distilling the privileged distribution converts future interactions into training-only supervision rather than inference-time inputs. Since both views share a backbone, the teacher's advantage is purely informational, not architectural, removing the need for a separately pretrained teacher and letting its supervision adapt as the student evolves. PSD further uses an advantage-reachability gate to focus distillation on teacher signals likely supported by the observed prefix, along with a momentum-averaged teacher for stable targets. The framework is optimized end-to-end in a single stage, leaving the deployed model and inference cost unchanged. Experiments across public benchmarks and diverse backbones show consistent improvements. 

---
# IMFuse: Instance-Aware Multi-Layer Fusion for LLM-Enhanced Sequential Recommendation 

**Authors**: Yuheng Zheng, Yu Cui, Bin Wu, Jian Zhang, Ye Feng, Can Wang, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27002)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced sequential recommendation by encoding rich item textual information into semantic representations. However, existing methods typically rely on the final-layer hidden states of LLMs, overlooking potentially useful semantic signals encoded in other layers. Through empirical analysis, we reveal the limitations of this practice: final-layer representations often suffer from dimensional collapse, whereas intermediate layers preserve complementary, coarse-to-fine semantic knowledge. Furthermore, we observe that different items exhibit heterogeneous layer-wise representation evolution, making a uniform layer selection sub-optimal. To bridge this gap, we propose IMFuse, an instance-aware multi-layer fusion strategy designed for LLM-enhanced recommendation. Instead of relying on a single layer, IMFuse adaptively aggregates multi-layer semantic information by learning global dimension-wise layer preferences to capture general semantic contributions. To address item-level heterogeneity, IMFuse introduces an instance-aware expert modulation mechanism that dynamically adjusts these global preferences, generating personalized, item-specific semantic representations. Extensive experiments across four real-world datasets demonstrate the effectiveness of IMFuse. It consistently outperforms state-of-the-art baselines with an average relative improvement of 6.72%, while introducing limited parameter and computational overhead. 

---
# Beyond Action Imitation: Learning a Decision-Aware User Simulator for Online Advertising 

**Authors**: Zipeng Chen, Jiaer Zheng, Xiangyang Xu, Xinyu Lin, Zhaobin Wang, Zhaohui Liu, Qianjin Xiang, Xiaoyu Zhao, Zhuozhen Yu, Guangshuo Wang, Daxing Chen, Junwei Pan, Zhangbin Zhu, Chengguo Yin, Hao Chen, Tat-Seng Chua, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26893)  

**Abstract**: Recent advances in LLM-based user simulation have shown promise for offline evaluation of recommendation and advertising systems. However, existing simulators typically infer user preferences from single-domain interaction histories and are primarily optimized to reproduce observable actions such as clicks. Consequently, they capture only a partial view of user preferences, while action-only prediction easily induces model shortcuts and limits both the fidelity and diagnostic value of simulation. To address these challenges, we propose DASH, a decision-aware user simulator that jointly generates thinking traces and predicts behavioral actions from heterogeneous cross-domain histories. DASH first introduces a Context Engineering stage that folds heterogeneous cross-domain histories into decision-relevant context, together with prompt optimization for effective reasoning over the folded context. To train a user simulator, DASH distills thinking trajectories from strong LLMs as SFT data, and further tailors a rubric-based reward model that evaluates thinking traces along form, content, and logic for RL training. Combined with the action reward, these signals jointly improve action prediction and thinking quality. Extensive experiments on real-world Tencent advertising data spanning five heterogeneous content domains demonstrate the effectiveness, efficiency, fidelity, and diagnostic value of DASH. 

---
# MediaWiki Code2Code Search: Neural Retrieval for the Semantic Discovery of Open-Source Software Entities 

**Authors**: Francesco Tosoni  

**Link**: [PDF](https://arxiv.org/pdf/2607.26766)  

**Abstract**: Code search in large-scale ecosystems is often hindered by the lexical gap between user queries and implementation details, alongside the trade-off between the low latency of traditional Information Retrieval (IR) and the precision of Deep Learning (DL). We present MediaWiki Code2Code Search, a neural retrieval system for semantic code-to-code discovery. By indexing 1.29 million structural entities (functions, types, and templates) across 2,500+ MediaWiki repositories, our system enables retrieval based on computational intent rather than surface tokens. We employ a split-build architecture, decoupling GPU-intensive offline indexing from a CPU-only serving layer; our FAISS IVF-PQ index occupies 168.6 MB: a 96.6\% reduction compared to a flat float32 baseline, and achieves a median query latency of 1.85 seconds on commodity hardware, satisfying the 6 GiB RAM constraint of Wikimedia Toolforge. Our evaluation across a 27-query benchmark demonstrates superior performance over the BM25 baseline, achieving a P@10 of 0.87 compared to 0.64 (0.52 versus 0.34 for strict matching). Gains are most pronounced in name-obfuscated tasks where lexical methods fail. The system is available at this https URL under the Apache 2.0 licence and provides an open RESTful API. 

---
# CaIRec: Calibrated Modality Imputation for Incomplete Multimodal Recommendation 

**Authors**: Ruiyu Liu, Xiaohao Liu, Miaomiao Cai, Yunshan Ma, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2607.26720)  

**Abstract**: Real-world multimodal recommender systems often face incomplete modality observations, where items lack images, text, or other content features. Such incompleteness weakens item representations and degrades recommendation performance. Existing modality imputation methods estimate missing representations from available item content, but two challenges remain. First, they optimize the recovered representation itself without explicitly considering its relations with other modalities of the same item. The completed modalities may therefore form inconsistent cross-modal relations, causing Cross-modal Structural Distortion. Second, even structurally coherent recovered information may remain ineffective for personalized ranking. Recovered representations receive limited ranking-oriented guidance, while modality missingness disrupts the item neighborhoods required for preference propagation, resulting in a Preference Adaptation Gap. To address these challenges, we propose Calibrated Imputation for Incomplete Multimodal Recommendation (CaIRec), a two-stage framework. Structural Imputation Calibration (SIC) estimates missing-modality representations from shared information inferred from available modalities and calibrates their cross-modal organization through structural regularization and correspondence supervision from observed modality pairs. Preference-oriented Representation Calibration (PRC) performs recommendation-specific adaptation at both the representation and relation levels. It constructs pseudo-missing instances to align recovered representations with observed counterparts shaped by ranking supervision in the recommendation space. It further builds completion-aware item graphs by integrating completed content relations with collaborative evidence. Extensive experiments on three datasets under different modality-missing settings demonstrate the effectiveness and robustness of CaIRec. 

---
# WhisperRec: Latent Reasoning for Efficient Foundation Recommendation Models 

**Authors**: Hao Jiang, Peiru Du, Pengfei Yao, Mengting Li, Siyuan Lou, Kuo Cai, Sheng Yu, Qiang Luo, Jian Liang, Ruiming Tang, Fei Pan, Peng Jiang, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2607.26621)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities, motivating their adoption as backbones for foundation recommendation models (FRMs). Existing approaches typically enhance recommendation with explicit Chain-of-Thought (CoT) under the Think-then-Answer paradigm. However, generating lengthy rationales introduces substantial inference overhead, while fixed CoT templates struggle to model diverse, dynamic, and context-dependent user interests. We propose WhisperRec, an efficient latent reasoning framework for FRMs. WhisperRec compresses teacher-generated CoT into learnable latent reasoning tokens, enabling a Latent-Reason-then-Answer paradigm that performs reasoning in latent space without producing verbose rationales. This design retains decision-relevant reasoning information while avoiding the latency bottleneck of autoregressive rationale generation. Specifically, it first introduces Multi-View Adaptive CoT (MV-ACoT) to construct diverse, high-quality supervision from complementary perspectives on user interests. MV-ACoT also adapts reasoning complexity to each instance, applying lightweight analysis to clear cases and targeted multi-factor reasoning to challenging ones. Building on a pre-trained FRM, WhisperRec then employs a three-stage Latent Reasoning Alignment procedure to progressively internalize teacher CoT into latent representations. Finally, curriculum-based post-training activates latent-token reasoning for downstream recommendation while preserving standard recommendation capability. Experiments on an industrial-scale Kuaishou dataset and the public Kuaishou LLM-Rec benchmark show that WhisperRec consistently outperforms explicit-CoT methods and conventional baselines. Compared with explicit CoT Think and No-Think variants, WhisperRec improves SID@64 by 17.44% and 9.33%, respectively, and achieves over 10x higher online inference throughput. 

---
# ASARL: Autonomous Social-Aware Relevance Learning for QQ Search 

**Authors**: Tao Su, Jinjing Hu, Xiao Wang, Xingzhong Cao, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26593)  

**Abstract**: The rapid growth of online social platforms has transformed communication and information retrieval, giving rise to social search, where queries-titles are typically expressed in informal, community-specific language. While large language models provide strong general-purpose semantic understanding, their effectiveness in social search is constrained by contextual discrepancy, data scarcity, and behavior-driven dynamics. To address these challenges, we propose the Autonomous Social-Aware Relevance Learning (ASARL), a fully automated framework that integrates multi-agent data curation with staged model training. ASARL leverages a collaborative agent system: ReasonAgent generates interpretable relevance labels grounded in social attributes, CriticAgent validates and ensures logical consistency, and GenAgent augments long-tail data through synthetic query-title pairs. Building on the curated dataset, ASARL employs three-stage training: Social Context Training (SCT) to capture social language patterns, Preference-Guided Optimization (PGO) to align model predictions with behavioral signals, and Social Distillation (SD) to transfer these improvements into compact models for efficient deployment. Extensive offline and online experiments on the QQ search platform demonstrate significant improvements in both offline relevance metrics and online user engagement indicators, along with enhanced annotation efficiency. These results validate the effectiveness of combining autonomous, socially grounded data governance with preference-aligned training in practical search systems. 

---
# Multi-Decoder OneRec: Controllable Generative Retrieval for Multi-Objective Industrial Recommendation 

**Authors**: You Wang, Zhao Liu, Guoping Tang, Yiqing Yang, Shuo Su, Jing Liu, Naifu Zhou, Xiaoyou Zhou, Wei Jiang, Jian Liang, Xiao Lv, Ruiming Tang, Liyin Hong, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2607.26500)  

**Abstract**: Industrial recommender systems build candidate pools by assigning explicit quotas to objective-specific retrieval routes. This design offers quota control but increasingly fragments modeling, training, and serving as the route set grows. Semantic-ID-based generative retrieval provides a unified alternative, yet a single decoder entangles objective policies and limits candidate complementarity. We propose Multi-Decoder OneRec, a controllable framework that combines shared representations, isolated objective adaptation, and coordinated decoding. All objectives share a user-context module and the General Decoder, while each objective adds an isolated, parameter-efficient LoRA expert. During training, exposure-sample next-token prediction (NTP) updates the shared base, target-filtered NTP updates the event-based experts, and Kullback-Leibler (KL)-regularized policy optimization updates the Watch-time expert; gradient routing isolates these updates, and the General Decoder supplies a stop-gradient reference. At inference, explicit route quotas allocate the fixed budget and Multi-Decoder Constrained Beam Search reduces cross-route overlap. We publicly release Kwai26, a large-scale multi-objective benchmark with 1.31 billion raw item-level records, 31.85 million Item-ID entries, and 25.03 million items with valid Semantic IDs, together with predefined splits and an evaluation protocol. Under the same 512-item retrieval budget, Multi-Decoder OneRec improves over the single-decoder OneRec baseline by 1.69%-5.62% across four Recall@512 metrics. In a production A/B test, it yields relative gains of 0.37% in app usage time per device, 0.19% in Day-7 retained users, 0.19% in devices with at least one share, and 2.09% in new-content Cold-Start. These results show that generative retrieval can combine shared modeling with objective-specific control and complementary candidate generation. 

---
# NMKFR: A Robust Framework for Time-Aware Cold-Start Recommendation 

**Authors**: Chengzhi Liu, Ning Zeng, Zehui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2607.26429)  

**Abstract**: Item cold-start recommendation is difficult when new items have sparse early interactions and appear in recommendation environments that keep changing over time. Static content, early feedback, and temporal-state evidence are all useful, but their reliability varies across the item lifecycle. This work proposes a framework--Neural Memory Kalman Fusion Recommender (NMKFR), which combines a Titans-based semantic encoder with time-aware Kalman state tracking. The semantic branch extracts memory-enhanced item observations from text, while the temporal branch estimates latent states under irregular interaction intervals. The NMKFR further uses posterior covariance as an uncertainty signal to calibrate semantic memory retrieval and adaptive static-temporal fusion. Experiments on Amazon Video Games and MovieLens-32M evaluate NMKFR under time-aware and item cold-start protocols using sampled candidate ranking. Across the reported comparisons, ablations, diagnostics, and robustness analyses, NMKFR achieves the strongest retained results and exhibits bounded uncertainty-related internal behavior. These findings provide empirical evidence for posterior-covariance-guided semantic-temporal fusion under the evaluated offline settings. 

---
# PSG: Pair-Space Generation for Efficient Generative Reranking 

**Authors**: Chao Feng, Li Ma, Xiancheng Gao, Chenghao Zhang, Yuanhao Pu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26427)  

**Abstract**: Modern recommender systems adopt Generator-Evaluator (G-E) for list-wise reranking: a generator produces sequences from candidates and an evaluator scores them at sequence-level to filter out the optimal one for exposure. Auto-Regressive(AR), working as the backbone for generative recommendation, suffers two limitations. First, its complexity grows linearly with list length, forcing the system to generate fewer lists under rigorous latency constraints and thus limiting exploration. Second, teacher-forcing creates a train-test mismatch; cumulative errors worsen with length and degrade quality.
To address these problems, we propose Pair-Space Generation (PSG), a reformulation that elevates the generation atom from individual items to ordered item pairs. Given $n$ candidate items, PSG operates over pair vocabulary of size $n(n-1)$ per request, generates only $L/2$ tokens. Pair token representations are produced on-the-fly by a pretrained pair-token representation module optimized over large scale exposure logs, eliminating the data sparsity that would otherwise plague a quadratic sized vocabulary. We establish three theoretical guarantees: (i) PSG is bijective with item-space generation and induces an equivalent family of sequence distributions, thus incurring no loss of expressiveness; (ii) generation in pair-token space achieves approximately a $2\times$ to $4\times$ speedup theoretically under moderate settings and $1.83\times$ in the real industrial environmental settings; and (iii) under outcome-only rewards, the worst-case suboptimality of PSG is bounded by $O((L/2)^2 \bar{\epsilon})$, representing a nearly $4\times$ improvement over item-space generation. Beyond benchmark-based validation, PSG has also been deployed on Kuaishou, delivering a 0.178\% lift in per-user stay time on the platform, which serves over 400 million daily active users. 

---
# DIRECTOR: Dynamic Index-based Recommendation with Transport-Optimized Retrieval 

**Authors**: Yuanhao Pu, Chenghao Zhang, Chao Feng, Xiang Li, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2607.26418)  

**Abstract**: Reranking is a combinatorial decision problem that aims to select and order a high-utility slate from a request-specific candidate set. A major line of generative rerankers adopts autoregressive (AR) models, which construct the slate one position at a time to capture inter-position dependencies. However, under practical greedy or bounded-width decoding, prefix-based search may prematurely prune globally promising permutations and incurs inherently sequential latency, restricting the effective search space under a fixed serving budget. Non-autoregressive (NAR) alternatives alleviate this efficiency bottleneck through position-parallel prediction, but naive position-wise factorization treats different positions too independently, leading to insufficient cross-position coordination and potentially duplicate or conflicting item selections. To retain parallel efficiency while introducing global structural coordination, we propose Dynamic Index-based RECommendation with Transport-Optimized Retrieval (DIRECTOR), a transport-guided parallel reranking framework. DIRECTOR maps candidate items into a continuous latent space and generates request-conditioned dynamic retrieval indices for all target positions in parallel. During training, it uses entropy-regularized OT to provide conflict-aware supervision; at inference, it directly performs global hard matching on similarity matrix, producing duplicate-free slates without iterative transport. To further align the generator with an opaque list-wise evaluator that returns only a scalar utility, we introduce a prefix-anchored credit assignment mechanism that converts the global reward into position-specific training signals. Extensive offline and online experiments demonstrate that DIRECTOR consistently outperforms strong reranking baselines, achieving significant improvement in large-scale industrial recommendation scenarios. 

---
# Continuous Online Evaluation of Recommendation Strategies in Social Science Academic Search 

**Authors**: Mehmet Deniz Türkmen, Daniel Hienert  

**Link**: [PDF](https://arxiv.org/pdf/2607.26380)  

**Abstract**: Delivering relevant recommendations in academic search engines is a complex task due to the diversity of subject areas, information types, and user preferences. In this case study, we address these challenges by integrating and evaluating a range of recommendation systems within GESIS Search - a domain-specific search engine for the social sciences that provides researchers with access to research data, publications, variables, and measurement instruments. To support continuous, real-time evaluation of multiple recommendation strategies with actual platform users, we utilize the STELLA evaluation framework. We implement and compare a diverse set of algorithms, including traditional lexical similarity, semantic document similarity by using transformer-based embeddings, and session-based recommendations based on click paths from historical user sessions. Our results show that users prefer recommendations based on semantic similarity, which outperformed term-similarity and session-based methods. However, the performance of recommenders varies across categories within GESIS Search, suggesting that information-seeking behavior differs by information type. Overall, our study provides insights into how continuous evaluation can be incorporated to develop recommendations that better align with the preferences in academic search portals. 

---
# Embedding Items at Scale: Comparing GNN-Based and ID-Based Item Embeddings in the Yandex Ecosystem 

**Authors**: Sergei Makeev, Artem Matveev, Vladimir Baikalov, Kirill Khrylchenko  

**Link**: [PDF](https://arxiv.org/pdf/2607.26365)  

**Abstract**: Transformer-based sequential recommendation models, which process sequences of user-item interactions, rely heavily on the item embedding strategy. Existing approaches either use pretrained item embeddings or learn them end-to-end with the transformer. To the best of our knowledge, no prior work has compared these options from both cost and quality perspectives in a large-scale industrial setting. This paper is a case study that compares pretrained industrial graph neural network item embeddings with end-to-end trainable item embeddings across two mature production recommendation systems at Yandex: Yandex Market and Yandex Music. We additionally evaluate both approaches on a low-resource dataset sampled from Yandex Lavka production logs, for which both the data and code are publicly available for demonstration purposes. Our results show that a separate pretraining stage helps when training data is limited, but provides no worthwhile benefit for large-scale models trained on extensive datasets. 

---
# FinCacheServe: Dependency-Consistent Answer Reuse for Cost-Efficient RAG Serving over Mutable Enterprise Documents 

**Authors**: Lingteng Zeng, Yifan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.26076)  

**Abstract**: Retrieval-augmented generation services over mutable enterprise documents repeatedly execute semantically equivalent analysis requests. Answer reuse can remove GPU-bound generation work, yet response caches require dependency consistency when filings, evidence chunks, and tool outputs change. FinCacheServe treats each generated answer as a serving object indexed by enterprise intent and guarded by document versions, evidence fingerprints, tool fingerprints, model identity, and decoding configuration. A vLLM implementation evaluates SEC-derived financial-document workloads with Qwen2.5 models. On a 2,230-request hosted 7B trace, FinCacheServe skips 53.27% of LLM calls with zero observed dependency-stale outputs. Across three hosted 32B operator-suite seeds, it skips 53.31% of 544 requests, compared with 38.97% for versioned semantic caching and 22.43% for grounded-style reuse. Capacity, backend, and SLO replays show oracle-bounded cache management, 100k-entry transactional metadata behavior, and 44.30% lower estimated Wh per dependency-fresh 2s-SLO success than versioned semantic caching. 

---
# IDP AutoOpt: Agent-Driven Optimization of Document Processing Pipeline Configurations 

**Authors**: David Kaleko, Sergey Ivanov, Md Mofijul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2607.26075)  

**Abstract**: We present IDP AutoOpt, an autonomous LLM agent that discovers high-performing configurations for intelligent document processing (IDP) pipelines. Tuning IDP prompts, models, OCR settings, and schemas jointly currently costs domain specialists 20 to 80+ person-hours per document type and does not scale as enterprises add document classes. IDP AutoOpt runs a closed loop: it scores a configuration on a small labeled set, diagnoses field-level errors, generates targeted edits, and re-evaluates, guided by human-authored domain skills that encode production expertise. Across extraction, classification, and packet-splitting tasks deployed in healthcare, marketing-intelligence, and financial-services settings, IDP AutoOpt matches or exceeds human-expert accuracy at equal or lower cost (on an extraction benchmark, 90.2% vs 81.6% at 4.6 x lower per-page cost), cutting configuration time from weeks to under two hours. We further show that agent LLM capability has a hard threshold below which optimization fails, and that curated domain skills outperform raw source-code access, which can degrade performance when provided without structure. We also share practical lessons on context management and variance mitigation. Requiring only a configurable pipeline, a scoring function, and a small labeled set, the approach extends beyond IDP to other enterprise AI systems, such as RAG and multi-agent workflows, where configuration bottlenecks deployment. 

---
# Reproducibility in Recommender Systems: A Survey 

**Authors**: Alan Said, Alejandro Bellogin  

**Link**: [PDF](https://arxiv.org/pdf/2607.26074)  

**Abstract**: Reproducibility has become a cornerstone of credible recommender systems research, driven by growing concerns about the reliability and generalizability of experimental results. In response, the ACM RecSys conference introduced a dedicated Reproducibility Track in 2020 to encourage rigorous, transparent, and repeatable research. This paper presents a structured analysis of the track from 2020 to 2025, covering 51 accepted papers. We classify contributions by type and analyze common patterns in datasets, algorithms, frameworks, and evaluation practices, with the goal of understanding how reproducibility is operationalized in practice within the community.
Our findings reveal three main trends. First, the track has expanded in scope, evolving from a focus on reproduction and replication to include benchmarking, resources, and methodological contributions. Second, reproducibility papers exhibit a consistent methodological profile, relying on a limited set of datasets, algorithms, and evaluation protocols. Third, reproducibility in practice often involves extending prior experiments rather than strictly replicating them, with studies frequently introducing additional models or evaluation criteria. Overall, reproducibility work has improved transparency and documentation, but has had limited impact on methodological diversity, highlighting a gap between the conceptual definition of reproducibility and its implementation. 

---
# Guess Where You Go: Generative Next Point-of-Interest Recommendation in Amap 

**Authors**: Penglong Zhai, Bowen Zheng, Jie Li, Yifang Yuan, Yue Liu, Sicong Wang, Mingyang Yin, Tingting Hu, Shuaijun Guo, Fanyi Di, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26073)  

**Abstract**: Generative retrieval enables recommender systems to retrieve items by generating compact item identifiers, but scaling it to industrial scenarios remains challenging due to redundant or colliding token assignments and insufficient integration of heterogeneous item signals. These challenges are particularly critical for next Point-of-Interest (POI) recommendation, where models must represent structured spatial entities, capture sequential mobility patterns, and produce predictions consistent with real user behavior. We propose Gwhere, an end-to-end industrial framework that integrates semantic identifier (SID) generation with LLM-based generative next POI recommendation. Gwhere first learns discriminative POI SIDs through a contrastive residual-quantization tokenizer that aligns textual, visual, spatial, and collaborative signals. Based on these SIDs, Gwhere adapts LLMs to mobility scenarios via continued pretraining on enriched spatio-temporal corpora, supervised fine-tuning, and Exposure-Aware Kahneman-Tversky Optimization (EAKTO), a reinforcement learning objective for behavioral preference alignment. Experiments on public datasets and Amap's large-scale industrial dataset demonstrate the effectiveness of Gwhere. The system has been deployed in Amap's homepage service under high-concurrency and low-latency constraints. Long-term online A/B tests show improvements of 5.83% in P-CTR and 6.20% in U-CTR over the production baseline. The implementation is publicly available at this https URL. 

---
# IFCMemoryBench: Evaluating Long-Term Memory of LLM-Based Agents in BIM Information Retrieval 

**Authors**: Changyu Du, Alexander Vosseler, Filippo Mazza, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.26072)  

**Abstract**: Long-term memory is becoming a core capability of LLM-based agents, but existing evaluations largely test conversational recall in open-domain or persona-grounded settings. We argue that a stronger test is whether an agent can reuse information from prior sessions while acting over a live, structured, domain-specific environment. We study this problem in Building Information Modelling (BIM), a professional engineering workflow where agents must query large IFC models while also relying on project specifications, client decisions, and engineering conventions often discussed in conversation but absent from the model. We introduce IFCMemoryBench, a benchmark for evaluating long-term memory in LLM-based BIM information retrieval. IFCMemoryBench contains 143 multi-session tasks across 19 projects and 4,016 prior sessions, derived from incomplete-information questions in IFC-Bench v2. Each task seeds missing project context across earlier conversations and later asks a probe question that can be answered only by combining remembered context with live IFC queries. Our evaluation framework decomposes memory performance into ingestion, retrieval, and utilization, and measures both answer quality and memory quality with expert-validated LLM judges. We evaluate representative vector-, graph-, and file-based memory systems. The strongest system achieves only 32.4% answer accuracy under a deployment-realistic ingestion scope, and remains below 60% under oracle-filtered ingestion or a stronger probe agent. Analysis shows that current general-purpose memory systems often retrieve topically relevant context but store project knowledge as incomplete or fragmented facts. These results reveal a domain-transfer gap in agent memory and suggest that reliable professional agents require domain-aware memory representations linking conversations, project knowledge, and structured model entities. 

---
# GuidedRAG: Semantic Steering of Retrieval-Augmented Generation 

**Authors**: Matthijs Jansen op de Haar, Tobias Stähle, Lorenzo Gatti  

**Link**: [PDF](https://arxiv.org/pdf/2607.26071)  

**Abstract**: In this work, we propose GuidedRAG, a novel extension to traditional Retrieval-Augmented Generation (RAG) that introduces a dedicated selection stage and semantic steering during retrieval. In contrast to current state-of-the-art RAG approaches, which depend on increasingly complex retrieval and knowledge structures, GuidedRAG constrains the knowledge base using semantics before retrieval, aligning the retrieval space with user intent while substantially reducing the search space. Our evaluation shows that GuidedRAG improves retrieval relevance by 14.0-15.8%, mitigates a 19.7-27.4% loss in retrieval precision, and reduces retrieval overhead by orders of magnitude. Moreover, relevant chunks are consistently retrieved earlier in the ranking process, while alignment with user intent improves by 31.8-36.8%. We further show that GuidedRAG achieves full coverage across 15 diverse RAG variants, demonstrating generalizability across the literature. Together, these findings establish semantic steering and selections as a powerful and generalizable paradigm for improving the current state-of-the-art in RAG. 

---
# SimpleWikiSearch: A Clean Offline Wikipedia Environment for Agentic Search 

**Authors**: Guanming Xiong, Penghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26070)  

**Abstract**: Large language model (LLM)-based agentic search systems are often evaluated as if the underlying LLM were the only component that matters, yet their measured performance also depends on the surrounding search environment: the Wikipedia snapshot, preprocessing pipeline, chunking policy, retrieval backend, tool schema, observation format, and answer submission rule. These details are frequently under-specified, making it difficult to compare results or reproduce reported baselines. We present SimpleWikiSearch, whose corpus construction, retrieval stack, tool contract, and evaluation protocol are explicit and runnable. The environment starts from a full English Wikipedia dump, cleans and chunks the corpus, builds keyword and dense retrieval indexes, and exposes a minimal tool interface consisting of \texttt{search}, \texttt{open\_url}, and \texttt{submit\_answer}. We report baseline results on six QA datasets using open-source LLMs and provide a random-300 subset for comparisons with closed-source commercial models. SimpleWikiSearch provides a domain-specific agent harness and a controlled offline environment for reproducible agentic-search evaluation. Its contribution is this specified reference setup, rather than a new agent algorithm. Code and data will be available at: this https URL. 

---
# DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search 

**Authors**: Raphaël Sourty, Antoine Chaffin, Paulo Roberto Moura Junior, Amélie Chatelain  

**Link**: [PDF](https://arxiv.org/pdf/2607.27178)  

**Abstract**: State-of-the-art retrieval models increasingly rely on closed training data, creating a reproducibility gap. We present an open end-to-end recipe for training retrieval models and study how English supervision transfers to multilingual retrieval through translate-train. We first reconstruct and curate 665M English contrastive pre-training pairs from 1.4B pairs across 34 public sources and build 1.88M supervised fine-tuning pairs with mined hard negatives. Training yields two 149M-parameter models: DenseOn, a single-vector dense model, and LateOn, a ColBERT-style late-interaction model. They achieve 56.20 and 57.22 average nDCG@10 on BEIR, respectively, setting new state-of-the-art results for this size class. We then translate the validated English data into eight languages, yielding 2.8B pairs with cross-lingual samples, and train mDenseOn and mLateOn, two 307M-parameter models built on mmBERT-base. Despite sharing their backbone, data, and objectives, their representations behave differently: the dense model is strong on English and translated languages but degrades outside translate-train support, whereas the late-interaction model generalizes better to unseen languages and scripts. This suggests that token-level matching turns translate-train from a target-language expansion strategy into a multilingual generalization recipe. We publicly release the models, datasets, and training code. 

---
# Kairos: Numerically Robust News Recommendation under Item Cold-Start via Cholesky-based LinUCB 

**Authors**: Finn Hertsch  

**Link**: [PDF](https://arxiv.org/pdf/2607.26832)  

**Abstract**: Algorithmic news personalization in regional markets often fails because modern deep learning models require massive interaction data while real-world news has a short Time-to-Live (TTL < 48 h) and shallow article pools. This structural item cold-start deprives collaborative filtering of the data needed for robust modeling. This paper presents Project Kairos, a framework that bridges this data scarcity through a contextual online learning approach (LinUCB). To ensure numerical integrity for continuous operation, Kairos replaces error-prone Sherman-Morrison inversions with direct rank-1 updates of Cholesky factors. This preserves the positive definiteness of the covariance matrix even under ill-conditioned data scenarios. Simultaneously, Matryoshka Representation Learning (MRL) integration addresses inference latency. Empirical evaluations based on the Tagesschau API demonstrate that exploiting semantic redundancy in the feature space achieves a 4.85-fold efficiency gain without significantly compromising ranking precision. Kairos thus provides a blueprint for high-performance recommendation systems in resource- and data-constrained environments. 

---
# Scientific Knowledge Discovery in the Age of Large Language Models 

**Authors**: Eleni Adamidi, Serafeim Chatzopoulos, Thanasis Vergoulis  

**Link**: [PDF](https://arxiv.org/pdf/2607.26670)  

**Abstract**: The rapid growth of scholarly literature has made identifying relevant publications increasingly difficult, and conventional search systems still depend heavily on manually formulated queries and effortful manual inspection. Generative large language models (LLMs) offer a more flexible alternative, supporting literature retrieval and the screening of candidate studies against eligibility criteria. This chapter surveys 34 peer-reviewed papers applying generative LLMs to these two tasks, identified via a Boolean search over the OpenAIRE Graph (1,589 records screened to 34 inclusions). Reviewed studies are characterised by LLMs employed, model access and adaptation, prompting and architectural techniques, ground-truth sources, and evaluation metrics. 

---
# RAG-HAR+: Towards Cost-Efficient LLM-Based Human Activity Recognition for Edge Deployment 

**Authors**: Hansi Karunarathna, Nirhoshan Sivaroopan, Chamara Madarasingha, Anura Jayasumana, Kanchana Thilakarathna  

**Link**: [PDF](https://arxiv.org/pdf/2607.26631)  

**Abstract**: Human Activity Recognition (HAR) from wearable sensors supports applications in healthcare, rehabilitation, fitness tracking, and smart environments. Yet, existing deep learning approaches require dataset-specific training, large labeled corpora, and repeated adaptation to new sensor settings or activity taxonomies. Retrieval-Augmented Generation for Human Activity Recognition (RAG-HAR) addresses this by framing HAR as a training-free, retrieval-augmented task, in which statistical descriptions of sensor windows are used to retrieve similar labeled examples that guide LLM-based classification. We introduce RAG-HAR+, a retrieval-first and cost-optimized extension that strengthens retrieval while reducing dependence on LLM-based inference. RAG-HAR+ uses an offline Retrieval Designer Agent to design dataset-specific feature groups from a diverse pool of motion descriptors, enabling sensor windows to be compared using features better aligned with dataset-specific activity patterns. During inference, RAG-HAR+ uses majority voting over retrieved neighbors for samples with strong retrieval evidence and defers only uncertain cases to an LLM-based Ambiguity Resolver Agent. Across six HAR benchmarks, RAG-HAR+ maintains competitive or improved performance while reducing LLM usage, token consumption, and inference time. We further extend the RAG-HAR mobile prototype to demonstrate the practical feasibility of retrieval-first, LLM-assisted HAR in mobile sensing scenarios. 

---
# A Graph-Native Bitemporal Memory Store for Conversational AI Agents 

**Authors**: Alp Niksarli, Gopesh Baheti  

**Link**: [PDF](https://arxiv.org/pdf/2607.26520)  

**Abstract**: Conversational AI agents commonly lack persistent memory across sessions. The obvious fixes like injecting full chat histories into the context window, or delegating to a third-party memory service, either exhaust the model's context budget or send personal data through infrastructure the user does not control. We describe a memory store that avoids both problems: an agent-local Neo4j property graph augmented with HNSW vector indexes and a full bitemporal data model. Each memory is stored as an immutable identity node linked to versioned content nodes carrying two closed-open time intervals: valid time (when the fact was true in the world) and transaction time (when the database recorded it). This design supports point-in-time semantic retrieval without physically overwriting history. Semantic edges between related memories are maintained automatically at write time using cosine similarity over 1024-dimensional embeddings. We evaluate the system on LongMemEval, a 500-question benchmark spanning six question types designed to stress long-term memory. Across 60 sampled questions, the current-state semantic search path achieves 46.7% R@10 overall, rising to 80% on knowledge-update questions. The time-travel path yields 80% R@10 on knowledge-update but decreases recall on temporal-reasoning questions (50% to 37.5%), a consequence of post-filter dilution that points directly to a concrete design improvement. We discuss what these results reveal about the limits of pure retrieval for different question types and what each failure mode suggests for future work. 

---
# CMT-RAG: Complementary Memory Traces for Multi-turn Multi-hop RAG 

**Authors**: Lang Zhou, Yingjian Chen, Shuxuan Li, Kun-Yu Lin, Zhilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.26470)  

**Abstract**: Multi-turn information-seeking conversations require both multi-hop reasoning and long-range dependency tracking across turns. However, existing RAG systems typically represent conversational memory as raw dialogue history, rewritten queries, or unstructured summaries, making it difficult to recover the specific prior reasoning steps and evidence required for follow-up queries. Our key insight is to align conversational memory with retrieval by representing dialogue context as sub-question-level reasoning traces. Building on this insight, we introduce MuMu-QA, a benchmark for multi-turn multi-hop RAG with explicit cross-turn sub-question dependency annotations, and CMT-RAG, a complementary memory framework for this setting. At each turn, CMT-RAG employs a state-space trace generator, whose recurrent state serves as runtime memory, to incorporate recent conversational context and decompose the current query into structured trace drafts containing retrieval-oriented sub-questions and dependencies on earlier traces. It then grounds these drafts with retrieved evidence and stores them as persistent memory traces in a session-level DAG, enabling future turns to efficiently recover relevant prior reasoning and evidence. Experiments on MuMu-QA and corpus-level RAG benchmarks show that CMT-RAG consistently outperforms five categories of RAG baselines in answer accuracy. 

---
# RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning 

**Authors**: Pushkal Kumar, Tucker Nielson, Tanish Kolhe, Shubham Zala, Vincent Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26339)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems ground large language models (LLMs) in external corpora, but this reliance exposes them to corpus poisoning: maliciously injected passages that manipulate retrieved evidence. We introduce RAGuard, a layered defense against \emph{factual} corpus-poisoning attacks on RAG pipelines. The first layer adversarially fine-tunes a dense retriever on synthetic poisoned documents (fabricated facts, contradictions, and reasoning traps), teaching it to downrank malicious passages before generation. The second layer, the Zero-Knowledge Inference Patch ZKIP, is a label-free, black-box filter: for each retrieved document, it performs a leave-one-out decode and scores the document by the semantic shift and output-entropy change that its removal induces. ZKIP requires no poison labels, no ground-truth answers, and no access to model internals; it compares the model's own answers under counterfactual contexts. On poisoned Natural Questions at 5--30\% poison ratios, adversarial retriever training alone reduces but does not eliminate attack success, while ZKIP drives the measured attack success rate to 0.000 in every defended configuration, keeping Recall@5 within 0.03 of the clean-corpus baseline. Supervised analyses on both Natural Questions and BEIR (NFCorpus) confirm that the counterfactual signals ZKIP relies on carry learnable poison structure. The defense costs $k{+}1$ generator passes per query ($6\times$ for $k{=}5$); we analyze batching and early-stopping approximations that reduce this overhead. We also show that keyword-preserving poisons leave lexical retrievers such as BM25 essentially unaffected, an observation that delineates the boundary of the threat model. Code, datasets, and evaluation harnesses are released for reproducibility. 

---
