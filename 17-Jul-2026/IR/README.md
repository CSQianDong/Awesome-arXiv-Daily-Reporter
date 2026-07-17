# Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search 

**Authors**: Debayan Mukhopadhyay, Utshab Kumar Ghosh, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.15253)  

**Abstract**: Retrieval systems are trained and evaluated on a static idea of usefulness: hand a document and a question to a reader model, see whether the answer improves, and score the document accordingly. The idea holds up when a document is read on its own. It breaks when a language model works as a search agent, issuing several queries and reasoning across turns, because a document can matter for what it lets the agent do next rather than for what it says about the current question.
We measure that gap rather than argue it. Using a ReAct style agent over HotpotQA, we replay 1000 development questions and, for every document the agent read, delete it and re-run the rest of the trajectory from that point. Comparing the original run against its counterfactual gives a Counterfactual Trajectory Utility (CTU) score from three deltas: final answer quality, next query retrieval quality, and turn count. Crossing CTU against Static RAG Utility (SRU) over 23,322 document observations, the two are close to statistically independent (Spearman rho = -0.026). Roughly a third of the documents the agent reads are causally load bearing while looking useless to a static reader; we call these bridge documents. The pattern survives when the reader based axis is swapped for a BM25 and cross encoder proxy, giving a bridge cell of 27.2% on an evenly spread axis.
A second experiment pins down the mechanism. Using the Observable Entity Relevance (OER) measure from prior work, entities that discriminate relevant from non-relevant candidates appear in the agent's next query 4.02 times more often than entities found only in non-relevant documents (6.1% vs 1.5%, n = 227,139). A bridge document earns its keep by handing the agent a discriminative entity that redirects the search. Static relevance and causal usefulness are different quantities in agentic retrieval, and optimizing the first does not deliver the second. 

---
# CoSimRec: Measuring Coordinated-Content Penetration in Recommender Feedback Loops 

**Authors**: Nan Li, Jiahong Shao, Jiuyang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15114)  

**Abstract**: Recommender systems increasingly shape which content reaches users, making it important to understand whether coordinated activity is amplified beyond the accounts that initiate it. Existing robustness evaluations largely focus on static target-rank changes and do not capture how coordinated interactions, recommendation, and user response evolve within a feedback loop. To address this gap, we propose CoSimRec, an offline agent-based evaluation framework that models coordinated accounts, dynamic ranking, non-bot responses, and ranking interventions in a shared closed-loop process. CoSimRec introduces the Algorithmic Penetration Rate (APR) metric family to measure target content's share of non-bot exposure and engagement, lift against matched no-attack baselines, and exposure gained per coordinated interaction. We evaluate CoSimRec on MIND, MovieLens, and LastFM using random, popularity-based, feedback-sensitive, MF, and BPR-MF recommenders, with ten-seed inference for the primary APR analysis and population-scale experiments of up to 1000 users. Random controls show no statistically supported positive penetration, whereas popularity-based and feedback-sensitive ranking produce significant positive APR-Lift in all six master-worker dataset--recommender settings, reaching 0.4505 on LastFM; synchronization-aware ranking reduces APR in every corresponding defense setting. 

---
# Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature 

**Authors**: Maximilian Kähler, Katja Konermann, Lisa Kluge, Markus Schumacher  

**Link**: [PDF](https://arxiv.org/pdf/2607.14882)  

**Abstract**: With a large controlled vocabulary as the label set, the task of automated subject indexing in a library can be understood as a multi-label classification task. If the set of subject terms is large, the problem fits the Extreme Multi-Label Classification (XMLC) objective. In this study, we apply a selection of specialised supervised XMLC methods to the test case of subject indexing contemporary German scientific literature, collected at the German National Library (DNB). We contrast these results by including a classical lexical matching baseline and three of our own recently developed LLM-based methods into the benchmark. Algorithms are evaluated and compared in several metrics. This includes binary relevance comparisons with previously indexed material, as well as graded relevance ratings by professional subject librarians. A challenge for all methods is to reliably make suggestions from the long tail of the subject vocabulary. We find that supervised XMLC algorithms relying on transformer-based dense features give best results in terms of overall binary relevance metrics. However, focusing on graded relevance and performance in the long tail of our subject vocabulary, the LLM-based generative methods give better results, making them a promising alternative for future productive use. 

---
# LLM-Based Re-Ranking for Real Estate Search 

**Authors**: Nkateko Ntimane, Rafel Guedes, Tiago Cunha, Pedro Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2607.14835)  

**Abstract**: QuintoAndar Group operates the leading housing marketplace in Latin America for both rentals and sales. The platform replaces traditionally paper-heavy workflows with a fully digital experience, making housing transactions faster and more accessible to tenants, buyers, and landlords in the region. Finding the ideal home in such a vast catalog is inherently difficult. At the same time, the widespread adoption of conversational assistants is reshaping user expectations: people increasingly want to express their needs through open, multi-turn dialog rather than rigid filter menus and faceted search. This shift is particularly pronounced in housing, where intent is multi-dimensional, context-dependent, and rarely reducible to a small set of structured constraints. To meet these expectations, we propose a Large Language Model (LLM) based re-ranker that augments a conversational recommendation system by reordering retrieved candidates according to the nuanced, context-rich intent expressed across the user's conversation. We additionally construct a large-scale offline evaluation dataset for conversational real-estate search, containing 960,000 query-item pairs constructed from both synthetic and production queries and annotated using an LLM-as-a-Judge framework with human validation. We validate our approach both offline, on this proprietary dataset, and online, through a production A/B test. Both evaluations show consistent improvements in ranking quality, including a statistically significant increase in production of +5.3% in click-through rate and +4.8% in scheduled visits, demonstrating the value of integrating conversational context into housing recommendations. 

---
# Impact of Expert-Following Strategies in Financial Asset Recommendation 

**Authors**: Ryuki Unno, Koshi Watanabe, Keigo Sakurai, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.14556)  

**Abstract**: Financial institutions hold rich transaction histories, yet delivering recommendations that simultaneously maximize investment returns and ensure preference alignment remains a significant challenge. Existing approaches, namely return-based and preference-based strategies, each optimize a single objective, resulting in a fundamental trade-off between profitability (ROI) and relevance (nDCG). In this paper, we propose the Expert-Following Strategies: a framework that identifies top-performing investors based on their historical ROI and recommends the assets they purchased, scored by ROI-weighted purchase frequency. Our experiments using real-world transaction histories show that our strategy achieves statistically significant improvement over the market-average baseline in both ROI and nDCG simultaneously across all four thresholds. 

---
# Long-History User Transformers for Real-Time Ad Ranking 

**Authors**: Viacheslav Ovchinnikov, Georgii Smirnov, Nikolai Savushkin, Veronika Ivanova, Maksim Kuzin  

**Link**: [PDF](https://arxiv.org/pdf/2607.14331)  

**Abstract**: Long interaction histories are among the most informative inputs for click-through rate (CTR) prediction, yet in online advertising they collide with a hard serving constraint: ads must be scored within a few hundred milliseconds to enter the auction, which rules out running a large sequence encoder at request time. We describe how a production advertising system resolves this conflict by decoupling history encoding from real-time inference. A high-capacity offline transformer asynchronously encodes the user's full cross-surface interaction history into a compact representation cached in a feature store, while a lightweight runtime model combines this cached representation with the user's most recent events and the request context at serving time. The offline encoder is pre-trained autoregressively on large-scale interaction logs with a dual objective - feedback prediction and next-item prediction - and the two-stage architecture is then fine-tuned for CTR prediction on the target advertising surface. Offline, the split design recovers 72-80% of the quality of a full-history runtime transformer that would be too expensive to deploy, and the cached representation is robust enough to staleness to permit inexpensive refresh policies. In production A/B experiments, the system improves the primary ranking metric by +2.77% in search advertising and +2.1% on the Yandex Advertising Network, with revenue gains of +2.26% and +0.43% respectively - without increasing serving latency. 

---
# Deep-learning Causal Retrieval Optimization for Efficient e-commerce Distribution in Pinterest 

**Authors**: Junpeng Hou, XianXing Zhang, Sai Xiao, Derek Cheng, Darren Reger, Olafur Gudmundsson, Mehdi Ben Ayed, Zhiqing Rao, Huizhong Duan  

**Link**: [PDF](https://arxiv.org/pdf/2607.14161)  

**Abstract**: Pinterest is where people turn inspiration into action as users browse ideas, then take steps toward realization, often by discovering shoppable content. To support this journey, we must distribute commerce content when it helps, not when it distracts. We frame this as a causal decision of triggering shopping candidate generators in early retrieval and deploy a production system at Pinterest that learns personalized and contextualized triggering policies. A deep multi-task model jointly predicts outcomes and uplift of multiple events, trained with a doubly-robust pseudo-outcome alongside calibrated outcome losses for stable, single-robust uplift learning. A randomized data logging supplies counterfactual coverage, and the model is evaluated by both regular and reverse metrics for full assessment. A linear-time offline replay is designed to select thresholds and forecast policy impact with extremely high consistency with online results. For productionization, the model runs in parallel with remote retrieval calls without end-to-end latency regression. At web scale, we cut shopping triggers by up to 85% while holding key shopping sessions neutral, improving important total sessions (+0.26%) and Pin saves (+1.10%), with significant infrastructure savings. By unifying deep causal learning with reliable offline replay and demonstrating production-grade deployment, this work provides a generally practical recipe for early-retrieval optimizations in modern cascading recommenders beyond shopping, aligning exploration and cost with user intent at scale. 

---
# SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration 

**Authors**: Yuyao Zhang, Junjie Gao, Zhengxian Wu, Jiaming Fan, Jin Zhang, Shihan Ma, Yao Yao, Weiran Qi, Chuyan Jin, Guiyu Ma, Xingzhong Xu, Kai Yang, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2607.15257)  

**Abstract**: Recent advances in Tool-Integrated Large Language Models have made web search a core capability of information-seeking agents. However, as interaction histories grow, agents increasingly struggle to track task progress. When search attempts fail to yield useful evidence, current single- and multi-agent systems can become trapped in repetitive loops, wasting search budgets and ultimately compromising the quality and completeness of the final output. We introduce SearchOS, a system-level multi-agent framework that turns fragile, implicit search progress into explicit, persistent, and shared state. First, we formulate open-domain information seeking as relational schema completion with grounded citations, where agents discover entities, populate attributes across linked tables, and anchor each value to source evidence. Then we design Search-Oriented Context Management (SOCM), which externalizes the evolving state into Frontier Task, an Evidence Graph, a Coverage Map, and Failure Memory. Built on SOCM, SearchOS applies a pipeline-parallel scheduling mechanism that overlaps the execution of sub-agents and continuously refills freed slots with tasks targeting unresolved coverage gaps to improve utilization and throughput. To schedule and control the execution of search agents, SearchOS introduces a Search Tool Middleware Harness that intercepts model and tool interactions to record grounded evidence and react to stalls or budget exhaustion, and provides a reusable hierarchical skill system comprising strategy and access skills to augment the agents' search process and avoid repeating failed search patterns across runs. On WideSearch and GISA, SearchOS leads all metrics among the evaluated single- and multi-agent baselines, paving the way toward robust information-seeking collaboration. 

---
# Towards Hierarchical Structure Understanding of Newspaper Images 

**Authors**: William Mocaër, Solène Tarride, Thomas Constum, Merveilles Agbeti-Messan, Tom Simon, Clément Chatelain, Stéphane Nicolas, Pierrick Tranouez, Sébastien Cretin, Thierry Paquet  

**Link**: [PDF](https://arxiv.org/pdf/2607.15082)  

**Abstract**: Understanding newspaper images remains a challenging task due to their complex, nested hierarchical structures and dense, heterogeneous layouts. In this paper, we explore two complementary approaches for newspaper structure understanding. First, we present a modular bottom-up pipeline that combines state-of-the-art open-source models: YOLO for layout detection, LayoutReader for reading order prediction, and a custom algorithm for article segmentation. This approach leverages existing robust components while maintaining flexibility and interpretability. Second, we introduce Tiramisu (Tiered Transformers for Hierarchical Structure Understanding), a novel end-to-end transformer-based architecture that explicitly models document hierarchy through an iterative tiered process. Tiramisu performs section and article separation, block localization, semantic categorization, and reading order prediction using highly parallelized attention mechanisms. Finally, we release Finlam La Liberté, a new dataset designed specifically for evaluating hierarchical information retrieval in historical newspapers. Experimental results demonstrate the effectiveness of both approaches in reconstructing complex newspaper hierarchies, with comparative analysis highlighting their respective strengths for scalable document digitization. The Tiramisu training code, including the synthetic newspaper generator, is available at this https URL. 

---
# Accelerating A/B-Tests with Counterfactual Estimation: Reducing Variance through Policy Overlap 

**Authors**: Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2607.14604)  

**Abstract**: Online controlled experiments are the gold standard for hypothesis testing in online platforms. Notwithstanding their ubiquity, they are notoriously expensive to run, and issues of variance hamper statistical power in assessing treatment effects. While standard variance reduction techniques leverage model-based control variates to reduce outcome noise, they remain agnostic to potential structural relationships between competing policies.
In this work, we identify a critical inefficiency in the standard A/B-testing protocol: when a treatment and control policy agree on an action, the resulting outcome contributes noise but no signal regarding the treatment effect -- unnecessarily inflating confidence intervals. We propose a novel experimental protocol that exploits this policy overlap to accelerate experimentation. The key insight is to frame the randomised treatment assignment mechanism as a meta-policy, and leverage $\Delta$-Off-Policy Estimation methods to obtain unbiased estimates for average treatment effects. We prove analytically that our approach recovers standard A/B-testing practices in the general case, but that its variance scales with the divergence between policies rather than raw outcome variance. Hence, we dominate the standard Difference-in-Means estimator whenever policies have common support, and the improvement is strict whenever the overlap region contributes non-zero residual variance. Empirical results corroborate these theoretical insights -- holding promise for significant impact on the real-world evaluation of recommender systems, information retrieval pipelines, and large language model interfaces. 

---
# SAGA: Schema-Aware Grounding for Agentic Text-to-SPARQL Generation 

**Authors**: Yiming Zhang, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2607.14494)  

**Abstract**: Complex knowledge base question answering (KBQA) is commonly approached through either information retrieval over a question-specific subgraph or semantic parsing into an executable logical form. We study the latter paradigm. Recent large language model agents make semantic parsing interactive: they alternate between reasoning, querying the knowledge base, and extending a partial SPARQL query. This interleaving reduces reliance on one-shot generation, but makes the quality of \emph{KB grounding} depend on what the interaction tools expose. Existing agents retrieve or prune candidate properties mainly through lexical relevance and instance-level observations, without systematically conditioning on entity types, property domains and ranges, or the expected answer type. We call this failure mode \emph{type-blind grounding}. It enlarges the grounding search space and often produces plausible-looking but semantically incompatible triple patterns that execute to empty results. We propose SAGA (\underline{S}chema-\underline{A}ware \underline{G}rounding for \underline{A}gentic Text-to-SPARQL Generation), a training-free framework that turns property exploration into a schema-constrained grounding operation. SAGA maintains a persistent bidirectional type state, filters known-incompatible property candidates at construction time, presents the remaining graph patterns in a compact schema-annotated format, and handles missing schema information permissively through empirical and trace-local evidence. Across nine benchmark settings over Wikidata and Freebase, SAGA achieves the highest F1 on all nine settings and the highest exact-match accuracy on eight, while reducing empty-result queries across all reported Wikidata settings. 

---
# DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA 

**Authors**: Brandon Michaels, Brendon Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2607.14400)  

**Abstract**: This paper describes DS@GT ARC's submission to the CLEF 2026 LongEval Task 4 on Retrieval-Augmented Generation (RAG). In this submission, we examine a divergence between traditional natural language evaluation metrics and citation integrity as applied to RAG QA systems. We evaluate a corrective pipeline using Corrective RAG (CRAG) and CiteFix against baseline and frontier model benchmark RAG QA scores. While frontier models maximized answer relevance and fluency scores, our RAGAs LLM-as-judge diagnostics indicate that frontier models would correctly identify relevant documents without using their context in answer generation. Conversely, by filtering chunks pre-generation and enforcing strict entailment of generated claims to the cited material post-generation, our corrective pipeline marginally improved citation faithfulness and answer grounding. We propose that evaluation of trustworthy RAG QA requires metrics that reward strict answer grounding. 

---
# Why Git Is the Memory Solution for the Agentic Development Lifecycle 

**Authors**: Frank Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.14390)  

**Abstract**: Coding agents now produce a growing share of a team's code, while the reasoning behind each change -- the alternatives weighed, the constraints discovered, the approaches rejected -- is trapped in assistant transcripts that vanish with the session. Memory for this setting, the agentic development lifecycle (ADLC), is usually posed as one retrieval problem and built as machinery: tiered stores, memory graphs, compiled wikis, model-judged admission. We argue memory should instead be git-bound -- built into the repository's version control, inheriting the guarantees the machinery struggles to construct: ground truth from commits, freshness from rebuild, verification from the merge, containment from review. On this ledger we solve two problems separately, then combine them. Seed supply is closed as an eight-corpus retrieval study under a pre-registered ship discipline: five imported ranking mechanisms rejected, two kept, and a best configuration of ~0.31 pooled MRR -- ~60x the raw-transcript grep floor, ~15x an honest parsed-turn floor. Answer assembly is where ranking stops helping: single-shot retrieval scores only 0.07-0.20 answer-sufficiency on real developer questions, and ungated episode injection measurably degrades good answers. A router dispatches breadth to a git-anchored structural map, pointed lookups to confidence-gated episodes, and rationale to decision synthesis, which reconstructs why-arcs no single session contains (0.83 sufficiency on a young ~50k-LOC production system). Routed, the system answers at 382-980 tokens per question -- three orders of magnitude below the recorded history. Because ground truth is mined from commit-session links rather than annotated, every result is replicable on any user's own history at zero labeling cost. The remaining constraint is capture. Code, benchmark, and paper source: this http URL. 

---
# ICAConfPubs: A Dataset and User Interface for ICA Conference Papers (2003-2018) 

**Authors**: Hongtao Hao, Xinyue Chen, Jiye Sun, Yanling Zhao, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14234)  

**Abstract**: This paper presents a comprehensive dataset of past ICA (The International Communication Association) annual conference papers from 2003 to 2018, encompassing 27,466 papers, 21,038 authors, and 4,935 sessions. We made the dataset publicly available in both CSV and JSON formats. Additionally, we developed an API to facilitate programmatic access, and an intuitive user interface to enable users to navigate and explore the data more easily. The web application, API documentation, downloadable data, and reproducible code to obtain and process the data are available at this https URL. 

---
# Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning 

**Authors**: Dingsu Wang, Filip Ryzner, Kelly He, Armando Ordorica, David Woo, Aditya Mantha, Liyao Lu, Usha Amrutha Nookala, Haoran Guo, Jiacong He, Olafur Gudmundsson, Matt Chun, Krystal Benitez, Dhruvil Deven Badani, Yijie Dylan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.14192)  

**Abstract**: As recommender systems mature in the past few years, their optimization objectives have evolved from a primary focusing on short-term behavioral signals to a broader emphasis on long-term user engagement and retention. However, directly optimizing retention is difficult because return signals are sparse, delayed, and only partially attributable to earlier recommendations. Prior work has addressed this challenge with sequential modeling and reinforcement learning, but these approaches typically require task specific reward engineering, substantial computational overhead, and surface specific implementations that are difficult to generalize. In this paper, we present a unified, model-agnostic downstream reward framework for optimizing long-term user value in large-scale recommendation systems. First, we formulate the downstream reward learning problem and develop an offline screening framework to identify session level behaviors that are both observable early and predictive of future retention. We then propose several model-agnostic downstream rewards signals derived from observed user action patterns across multiple sources. We further discuss the engineering effort to productionize the proposed rewards derivations and challenges we faced when adding them to our ranking models. Online A/B experiments demonstrate consistent improvements in engagement and retention-related metrics, and the framework has been deployed across multiple Pinterest surfaces, including Homefeed, Related Pins, Search, and Notifications. 

---
# A Temporal Machine Learning-Based Time-to-Event Model for Predicting ALS Progression and Healthcare Utilization 

**Authors**: Zongliang Yue, Qi Li, Terry Heiman-Patterson, Frank Bearoff, Zhaohui Qin, Huanmei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.14190)  

**Abstract**: Amyotrophic lateral sclerosis (ALS) is a progressive and heterogeneous neurodegenerative disease in which predicting clinically meaningful milestones, such as assistive device use, remains challenging. We developed a time-to-event, digital-twin-inspired framework that integrates longitudinal ALS Functional Rating Scale-Revised (ALSFRS-R) trajectories with survival modeling to support individualized prediction of functional decline and assistive device utilization. We constructed a harmonized longitudinal dataset by integrating diagnosis records, ALSFRS-R assessments, activities of daily living, and demographic information, followed by preprocessing to ensure data quality, temporal alignment, and cohort consistency. Correlation-based clustering identified coherent functional domains spanning bulbar, upper limb, axial, lower limb, and respiratory systems. Generalized additive mixed models characterized nonlinear, domain-specific functional decline across all domains. In addition, a temporal machine learning model was developed to predict longitudinal functional decline and capture stage-dependent disease progression. Cox proportional hazards modeling further identified lower limb function, particularly walking and stair climbing, as the strongest predictors of earlier wheelchair access. Building on these results, we implemented a digital twin-inspired temporal machine learning-based time-to-event (TTE) model that generates individualized survival curves and dynamically predicts wheelchair-free survival. This framework provides a scalable, interpretable, and clinically actionable approach for linking ALS progression with personalized decision support, with applications in proactive care planning, clinical trial stratification, and precision medicine. 

---
# Certified Domain Consistency for Multi-Domain Retrieval: Label-Free Per-Domain Contamination Control with Conformal Risk Guarantees 

**Authors**: Jayakumar Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2607.14157)  

**Abstract**: Retrieval over corpora that mix several domains often returns relevant but wrong-domain evidence that ranking metrics miss and that conformal risk control bounds only marginally, under-covering the worst domains. This work introduces C3R, a drop-in control layer that, from an inferred domain posterior and no query-time label, certifies a per-domain contamination budget where feasible and otherwise abstains rather than silently violating; on the hardest domains it guarantees a reduction, not a tight bound. The core is a two-split scheme built on risk-controlling prediction sets, whose finite-sample transfer bound crosses from the inferred to the true domain with fully estimable slack, supports heterogeneous budgets, and inverts for deployment. Population validity rests on this bound and a controlled simulation; across a thousand resampled calibrations the certificate never violates (a stability result) while marginal control violates the most-contaminated domain in every draw, and soft demotion retains more recall than the strongest calibrated cascade at equal certified contamination. The method replicates across open testbeds including an independent one from public federal regulations, and an LLM-judged downstream probe indicates wrong-authority grounding rises with contamination and falls under control. The layer is frozen-stack and reranker-agnostic. 

---
