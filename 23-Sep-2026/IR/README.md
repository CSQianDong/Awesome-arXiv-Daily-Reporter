# Discovery-Driven Integration of Disjoint Tables via Text 

**Authors**: Md Ataur Rahman, Dimitris Sacharidis, Oscar Romero, Sergi Nadal  

**Link**: [PDF](https://arxiv.org/pdf/2609.26658)  

**Abstract**: Integrating heterogeneous datasets within data lakes is a critical challenge, particularly for semantically related tables that lack the explicit attributes needed to be joined. We study Discovery-Driven Integration, where the relevant sources and their missing relational structure must be discovered before integration. In this setting, unstructured text provides the evidence that connects otherwise disjoint tables. The fundamental challenge is to discover the relationships at a fine-grained level that connect individual rows from different tables through specific sentences. We formalize this task as Text-Mediated Join Path Discovery and propose a horizontal bidirectional cross-attention architecture called LOKI Latent-space Optimization for Knowledge Integration) that learns contextualized representations of table rows and sentences. Through a global table-text contrastive objective, fine-grained row-sentence associations emerge without explicit local supervision. Existing multi-modal discovery methods largely retrieve coarse-grained column-text associations, whereas integration systems assume supplied row-text links, schemas, or queries. LOKI instead transforms these implicit associations into explicit, interpretable join paths, organizes them into relation-consistent groups, and materializes them as typed integrated tables with sentence-level provenance. Comprehensive evaluations on real-world benchmarks demonstrate that LOKI consistently outperforms state-of-the-art multi-modal data discovery approaches, and materializes typed integrated tables with 0.982 macro typed-pair precision while being up to 40 times cheaper in LLM API cost than direct prompting. 

---
# When Does Permutation Instability Generalize? Independent-View Validation for Listwise LLM Reranking 

**Authors**: Wenzhang Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.26251)  

**Abstract**: Listwise language-model rerankers often disagree across equivalent candidate permutations. Finite instability diagnostics are therefore used to motivate additional sampling, aggregation, or selective computation. But an association with a validation statistic that reuses the probe views need not isolate predictive information about unseen permutations. Shared measurements can induce classical part-whole association. We study how this affects claims that a finite-view instability score predicts unseen permutations. We derive the exact finite-view decomposition and prospectively compare zero, one, and two reused views, including a fully disjoint four-view target.
The study covers two pinned 7B model families and two recommendation datasets, with controlled lists for signed offline analysis and untouched retriever lists for target-free replication. On the four controlled blocks, fully disjoint correlations are weak or heterogeneous (-0.061 to 0.281), whereas reusing both probe views yields 0.600 to 0.718; all paired contrasts are large (0.436 to 0.661) and Holm-significant. The overlap effect is positive in all four untouched-list blocks. Increasing the probe from two to four views clearly improves disjoint reliability in only one block. Moreover, the probe predicts aggregation-movement magnitude (Spearman rho = 0.142 to 0.426) but not stable signed target benefit, and 7 of 12 fixed-fraction probe-routing points are strictly dominated at measured cost. Thus, when the intended estimand is predictive information about unseen perturbation behavior, validation targets must be observation-disjoint from the probe to isolate that information; signed utility and cost-sensitive decisions remain separate questions. 

---
# Which Reranking Conclusions Survive the Answer Interface? A Prospective Finite-Orbit Audit 

**Authors**: Wenzhang Du  

**Link**: [PDF](https://arxiv.org/pdf/2609.26250)  

**Abstract**: Rerankers are increasingly evaluated through downstream language-model answers. This raises a retrieval-measurement question: if only the reader's answer interface changes, should we reach the same conclusion about BM25 versus BGE-v2-m3? We prospectively audit their claim-paired effect on RAGuard and FEVER with four readers. Retrieval policies, evidence, claims, and context depth remain fixed while semantic-to-label binding, A/B versus X/Y vocabulary, and option order form eight task-equivalent interfaces. We ask whether the estimated retrieval-policy effect, its ordering, or selection value changes. None of the six confirmatory settings showed statistically certified interface variation above the prespecified 0.015 materiality threshold, and none showed a certified reversal of the BM25-BGE ordering. Selector disagreement reaches 33.5% in one environment, yet none of eight environments establishes the prespecified material held-out value difference. These results do not support broad replicated instability, but they do not prove universal invariance: five settings remain too uncertain to satisfy the prespecified higher-order equivalence condition. They show why retrieval evaluations should separate policy level, interface stability, policy ordering, and selection value. When stability is unverified, a uniform average over the enumerated interfaces with explicit variation bounds avoids privileging one interface. 

---
# ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap 

**Authors**: Minhan Cho, Soyoung Park, Daejin Choi, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.26237)  

**Abstract**: We present the ABAI submission to COLIEE 2026 Task 1, case law retrieval, together with a controlled study of why it underperformed. The task suppresses the cited passages themselves, which removes much of the lexical overlap a retriever would rely on. Our pipeline answers this with four independently trained stages: multi-view BM25 over citation-context windows with reciprocal rank fusion, neural reranking, graph-based features from entity communities and a graph attention network, and a LightGBM meta-learner over 34 features. Our best run reached F1=0.177 on the official test set, against a cross-validated 0.311, and we attributed that gap to a recall ceiling, temporal distribution shift, and threshold miscalibration. We then tested all three. Under leakage-free protocols threshold transfer costs 0.007 F1, decision quality is flat across chronological quartiles, and the official test queries are not measurably farther from the training manifold than training queries are from each other, in two independent embedding spaces. Decomposing the misses instead splits them exactly evenly between candidates never retrieved and candidates retrieved but ranked below the cut. Measuring the remedies for each half, BM25 length-normalisation tuning, an event-triple view, and full-content dense fusion lift top-200 recall by three to seven points, and citation-graph features add 0.014 F1 over eight seeds once own-citation leakage is removed, while per-query cutoff rules, a zero-shot reranker swap, and a date filter do not help. We also document four evaluation artifacts, each of which reversed a result once the protocol was corrected. 

---
# A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data 

**Authors**: Robert Šamárek, Radek Martinek  

**Link**: [PDF](https://arxiv.org/pdf/2609.26218)  

**Abstract**: Structural graph analysis of the academic publishing network captures the topological relationships between entities but does not see the content of works. Building on our structural approach, this work complements it with a semantic layer and a parameterized structural-semantic fusion. We represent scientific documents by citation-informed vector embeddings (SPECTER2) and store them in an embedded vector database keyed by the stable OpenAlex ID, so that they connect directly to the graph layer. We define a modular late-fusion function that combines semantic similarity (cosine of embeddings) and structural similarity (bibliographic coupling) with a tunable weight alpha whose value is chosen according to the specific task. On the corpus of VSB - Technical University of Ostrava we show two things: citation-informed embeddings agree with the expert OpenAlex topical taxonomy better than a TF-IDF baseline, and in a recommendation use case the structural, semantic, and combined signals carry information in different regimes depending on the available data. Hybrid fusion here is not a universally better method but an explicit mechanism for steering complementary signals according to the task. We release the whole approach as an open-source extension of the apnet library with a reproducible workflow. 

---
# When Concealed Links Cannot Be Recovered: A Structural Identifiability Bound and Evaluation Pitfalls in Offshore Leak Networks 

**Authors**: Joseph Bingham  

**Link**: [PDF](https://arxiv.org/pdf/2609.26171)  

**Abstract**: Leaks such as the Panama and Paradise Papers expose large networks of offshore entities, and they invite an obvious question for network learning. Can the relations these structures are built to hide---above all, who beneficially owns what---be recovered from the public part of the leak by link prediction? We argue that the answer is mostly no, and that the analyses which suggest otherwise are measuring the wrong thing. Our main result is a distribution-free identifiability bound. For any recovery rule that respects graph isomorphism, and that covers every topological link-prediction score together with every message-passing graph neural network, a concealed endpoint left isolated in the observed graph is interchangeable with its structural twins, so its hidden edge cannot be recovered above chance. Isolation is only the sharpest case. In general the ceiling on recovery is set by the size of a node's structural-indistinguishability class, for which the Weisfeiler--Leman colour class is a computable stand-in, and node degree is at best a loose proxy. On the full ICIJ Offshore Leaks graph (814K entities and 84K labelled beneficial-owner edges) a classifier-free, degree-controlled probe reproduces an exact $0.5$ floor for isolated owners, who make up $20.6\%$ of all owners, and a trained graph neural network lands on the same floor. Recovery climbs only as structural distinctiveness grows, and the floor reappears in every one of the five leaks. Along the way we document five evaluation traps. Each one makes a bound that cannot be beaten look beaten, and we give a short rule that avoids them. The practical upshot is to redirect effort from the hidden principal, which is close to unrecoverable, toward the machinery of concealment, and to spell out why fusing external data helps far less than one would hope. 

---
# TailSpec-EASE: Knowledge-Graph-Regularized Linear Recommendation for Web Long-Tail Discovery 

**Authors**: Jianru Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.26143)  

**Abstract**: Recommender systems on Web platforms tend to over-serve popular items and neglect the long tail. Item-side knowledge graphs (KGs), often available as linked data or RDF-style Web resources, can help by connecting sparse items through shared semantic attributes. Many competitive KG-aware recommenders rely on graph neural architectures, whereas strong shallow linear models such as EASE-R typically ignore side information and may become infeasible in their global closed-form version. We introduce TailSpec-EASE, a lightweight recommender that injects a relation-aware spectral KG prior into a local closed-form reconstruction objective. The prior strength adapts to item popularity, giving stronger semantic guidance to long-tail items. Across four public benchmarks and a broad set of classical, linear, graph-CF, KG-aware neural, and score-level KG baselines, TailSpec-EASE attains a favorable trade-off between overall accuracy, long-tail performance, and training cost. It improves NDCG@20 by up to 24% over its counterpart without KG information. All tail-metric improvements over the no-KG counterpart are statistically significant under a paired bootstrap, and overall NDCG improves significantly on three of the four datasets. In a representative Amazon-book timing study, TailSpec-EASE trains in 37 seconds on CPU, compared with 2,584 seconds for a GPU-trained KGAT run and 15,800 seconds for CPU LightGCN, while attaining higher NDCG@20 and Tail Recall@20 on that dataset. It also remains feasible on catalogs where the global closed-form model runs out of memory. 

---
# Knowledge-as-Skill: A Structural Design for Autonomous Knowledge-Base Use by LLM Agents 

**Authors**: Jiangxu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.25991)  

**Abstract**: Retrieval-augmented generation (RAG) gives large language models (LLMs) access to external knowledge, but its conventional retrieve-concatenate-generate pipeline makes retrieval decisions on behalf of the model. As tool use and agent loops become more reliable, an agent can decide whether to retrieve, what to inspect, and when to stop. This shift exposes a new bottleneck: the agent may not know what a knowledge base contains. Traditional knowledge bases expose documents as anonymous text chunks with limited information about scope, purpose, provenance, or relations.
We propose Knowledge-as-Skill, an organization scheme that makes a knowledge base discoverable, navigable, and self-descriptive. It has three layers: a discovery layer centered on this http URL; a navigation layer with one this http URL per directory; and a knowledge layer containing documents with YAML frontmatter for topic, type, provenance, and lifecycle. The design follows the Open Knowledge Format (OKF) and the Skill protocol without modifying the agent framework. We also provide knowledge-as-skill, a pipeline for converting heterogeneous collections of PDFs, Word files, web exports, and notes into this structure.
In a preliminary evaluation on the WixQA enterprise customer-support benchmark, our setup obtains 0.889 Factuality and 0.816 Context Recall, compared with reported Corpus2Skill values of 0.767 and 0.708. It obtains slightly lower Faithfulness, lower Context Precision, and more interaction turns. Because the models, prompts, and knowledge-package construction differ, these results are directional cross-work evidence rather than a controlled comparison. 

---
# Robust Fusion of Semantic and Behavioural Signals for LLM Reranking in Personalised Search 

**Authors**: Aleksandr V. Petrov, Nathan Stein, Erik Lybecker, Emma Schüldt, Daniel Lazarovski, Hugues Bouchard, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2609.25825)  

**Abstract**: Personalised search must satisfy query intent while incorporating user context and historical interactions. LLM-based cross-encoders provide a single reranking interface, but injecting predictive behavioural statistics into their prompts can encourage shortcut learning: reliance on historical signals at the expense of semantic and user-context patterns that generalise to sparse or unseen searches.
We study this problem in the personalised search system of a large-scale audio streaming platform using Query Slice Stats (QSS), an interaction-derived behavioural feature summarising historical success for query-candidate pairs. Naive QSS injection improves ranking when the feature is available but reduces robustness when it is removed. We address this with deterministic dual-sample feature-dropout training, which presents each example once with QSS included and once with QSS removed.
Offline, QSS injection improves ranking quality by 13.3% when available. Dual-sample training preserves these gains while improving performance under QSS-removed evaluation by 4.0% relative to naive QSS training. In a live online test, both QSS-aware variants improve search success by roughly 2%. The aggregate test does not distinguish dual-sample from features-only training; the cold-start comparison is directionally consistent with the offline results.
Paired feature-present and feature-removed training can therefore reduce the tension between exploiting strong behavioural statistics and remaining robust when they are unavailable. 

---
# ReFilter: Bridging Embeddings and LLM Filtering for Similar Mobile App Retrieval 

**Authors**: Buthayna AlMulla, Maram Assi, Safwat Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2609.25306)  

**Abstract**: Retrieving similar mobile applications (apps) is essential for researchers, developers, and end-users. Researchers use similarity detection to study app ecosystems and trends, developers for competitor analysis, and end-users for focused app recommendations. Existing approaches rely on embedding-based retrieval, which captures semantic similarity but fails to identify functionally similar apps. To our knowledge, no prior work has applied large language model (LLM)-based filtering to this task, due to the high computational cost of evaluating large numbers of app pairs. To address this gap, we propose ReFilter, a hybrid framework that first Retrieves semantically related candidate apps using embeddings and then applies LLM-based contextual Filtering to identify true functionally similar apps with higher precision. This design balances efficiency and accuracy, achieving an F1-score of 90% for retrieving similar apps. By improving the relevance of app alternatives, ReFilter enables more accurate app comparisons and supports improved ecosystem understanding, competitor analysis, and recommendations. 

---
# GroundedGEO: Auditing the Evidence Gap in Generative Search Rankings 

**Authors**: Yihan Xia, Huiling Fan, Kangrong Zhong, Taotao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.25189)  

**Abstract**: Generative search systems rank products and services for consequential decisions, and publishers can cheaply make candidate text look relevant. Yet evidence status is not a text property but a claim-evidence relation: text-only rankers and defenses cannot separate honest detailed content from fabricated detail, creating an identifiability gap. We audit this gap with an evidence-paired benchmark (50 e-commerce queries, 1,950 cases) and a claim-level reranker, GroundedGEO, that penalizes query-relevant claims lacking support in a supplied packet. Matched rich variants control format and volume; packet twins add attestations at fixed text, while thinned packets withdraw them. On the frozen listwise ranker Qwen2.5-7B, unsupported-rich variants show significant normalized rank gain over clean candidates (+0.065 to +0.092 across claim profiles, Holm-corrected), while supported and neutral controls do not; the effect is model-dependent (marginal on MiMo-v2.5, absent on GLM-5.3-Flash). On a frozen pointwise scorer, oracle evidence labels cut the unsupported-rich top-3 rate from 0.65 to 0.43 (laundering from 0.61 to 0.39) at lambda=40 with zero false suppression; packet twins restore the original rates without changing text. Against a 370-claim human gold, all tested automatic judges fail the preregistered reliability gate, although the best local judge retains 79-100% of oracle suppression with zero measured false suppression on protected arms. Separately, stripping attestation coverage increases false suppression by 0.307. These diagnostic effects identify two limits on the evidence channel: label quality and packet coverage. They do not validate an automatic defense, and interpretation of the adverse human-gold arm remains pending adjudication. 

---
# SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue 

**Authors**: Haobo Zheng, Tan Tang, Yan Chen, Weijie Wang, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26780)  

**Abstract**: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, event, and time at query time. To reduce attribution and update errors during structured memory construction while enabling local deployment, we train Writer-R1 with SpeakerLevenshtein and speaker-conditioned GRPO. On GroupMemBench, SocialMemBench, and EverMemBench, SpeakerMem-R1 achieves binary accuracies of 47.9%, 69.2%, and 61.9%, respectively. On the publicly reported EverMemBench leaderboard from EverMind-AI, we achieves 62.33%, the best reported result among the latest state-of-the-art frameworks. It also achieves 70.85% on all 1,986 LoCoMo questions, which we use as a two-person long-term conversation boundary test. In a controlled evaluation of 305 questions, RL raises the SFT Writer's mean accuracy from 57.38% to 68.20%. We report both binary accuracy and token-F1, and ablations show that the verbatim and structured tracks, as well as person-level and group-level views, are complementary under the standardized evaluation interface. 

---
# CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval 

**Authors**: Daeyoung Roh, Donghee Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.26086)  

**Abstract**: An agentic retrieval system issues a sequence of search queries and must decide, at each step, whether the evidence collected so far is enough to stop. Delegating that decision to an LLM verifier or a prompt judge makes stopping reliable, but the verifier then reprocesses the growing evidence after every retrieval step, a substantial repeated cost. We show that most of these calls can be skipped without materially changing answer accuracy: a single threshold on a frozen sentence-embedding coverage margin detects the states in which the evidence is still plainly incomplete, and the verifier is called only on the ambiguous remainder, a gate we call CoVeR (Coverage-based Verifier Routing). Across three multi-hop QA benchmarks, with the evaluation protocol fixed before the full-scale run, the CoVeR-gated agent matches the answer accuracy of both the full-budget agent and the always-verify baseline within a fraction of an EM point. It cuts 62-68% of verifier calls, and 93% in a saturated regime. Routers built on evidence counts, lexical overlap, or BM25 relevance, alone or learned in combination, give weaker overall trade-offs, the gate transfers without re-tuning across deciders and agent scales, and its drafter distills into a 921k-parameter head atop the frozen encoder, leaving no LLM in the routing loop. The same signal cannot replace verification: matching a claim is far easier than deciding the claim is supported. 

---
# ARAFA: An LLM-Generated Arabic Fact-Checking Dataset 

**Authors**: Christophe Khalil, Shady Elbassuoni, Rida Assaf  

**Link**: [PDF](https://arxiv.org/pdf/2609.25833)  

**Abstract**: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a test sample from the dataset demonstrated strong inter-annotator agreement (kappa = 0.89) using Cohen's Kappa for supported claims and (kappa = 0.94) for refuted claims. Automatic validation based on a human-evaluated sample achieved 86% accuracy for supported claims and 88% for refuted ones. To showcase Arafa's value as a resource for automatic Arabic fact-checking, four open-source transformer-based models were fine-tuned using Arafa, with the top-performing model achieving a Macro F1-score of 77% on the test data. In addition to Arafa being the first large-scale dataset for Arabic fact-checking, our framework presents a scalable approach for developing similar resources for other low-resource languages. 

---
# From Offline Proxies to Online Decisions: A Layered Engagement Evaluation Framework for Conversational AI 

**Authors**: Xuanyi Li, Vaskar Nath, Hossein Amirkhani, Jay Li, Alex Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.25408)  

**Abstract**: Online A/B experiments are the decision standard for user engagement, but traffic and readout time limit how many conversational-AI changes can be tested. We ask whether an offline signal designed to be computable without treatment-arm user exposure agrees with the outcomes of those experiments. We contribute a reusable construction and diagnosis checklist that treats an offline proxy as a chain of three alignments: behavioral label to product outcome, learned classifier to candidate-assistant behavior, and aggregated offline signal to experiment effect. A companion evaluation protocol audits the whole composite by interval-aware decision agreement, which compares offline and online confidence intervals instead of point estimates, and by within-experiment ranking. The instantiation we evaluate comprises a fixed evaluation suite on which candidate behavior is scored, an engagement classifier trained to predict session/prompt level engagements, and a calibration layer mapping sample-level score differences to online model-level engagement deltas. We then report the audit: 489 paired offline-online contrasts (one candidate arm against its control) from 27 experiments on a deployed multi-turn assistant, spanning model checkpoints to system-prompt tuning. Our primary test uses the 113 contrasts from eight experiments that ran after the map was frozen: on these the composite reaches 81.1% F1, against 34.3% for the raw classifier score it is built on, and makes no wrong-direction calls where that raw score makes 31. Every offline prediction was computed before its experiment ran to prevent overfitting. The evidence supports using the composite to prioritize candidates before scarce experiment traffic is allocated---in our deployment of the experiment, selecting among training checkpoints and tuning system prompts. 

---
