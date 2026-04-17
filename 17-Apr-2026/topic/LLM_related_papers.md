# Federated User Behavior Modeling for Privacy-Preserving LLM Recommendation 

**Authors**: Lei Guo, Hongyun Yang, Pengjie Ren, Tong Chen, Hui Liu, Zhumin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14833)  

**Abstract**: Large Language Models have shown great success in recommender systems. However, the limited and sparse nature of user data often restricts the LLM's ability to effectively model behavior patterns. To address this, existing studies have explored cross-domain solutions by conducting Cross-Domain Recommendation tasks. But previous methods typically assume domains are overlapped and can be accessed readily. None of the LLM methods address the privacy-preserving issues in the CDR settings, that is, Privacy-Preserving Cross-Domain Recommendation. Conducting non-overlapping PPCDR with LLM is challenging since: 1)The inability to share user identity or behavioral data across domains impedes effective cross-domain alignment. 2)The heterogeneity of data modalities across domains complicates knowledge integration. 3)Fusing collaborative filtering signals from traditional recommendation models with LLMs is difficult, as they operate within distinct feature spaces. To address the above issues, we propose SF-UBM, a Semantic-enhanced Federated User Behavior Modeling method. Specifically, to deal with Challenge 1, we leverage natural language as a universal bridge to connect disjoint domains via a semantic-enhanced federated architecture. Here, text-based item representations are encrypted and shared, while user-specific data remains local. To handle Challenge 2, we design a Fact-counter Knowledge Distillation module to integrate domain-agnostic knowledge with domain-specific knowledge, across different data modalities. To tackle Challenge 3, we project pre-learned user preferences and cross-domain item representations into the soft prompt space, aligning behavioral and semantic spaces for effective LLM learning. We conduct extensive experiments on three pairs of real-world domains, and the experimental results demonstrate the effectiveness of SF-UBM compared to the recent SOTA methods. 

---
# GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation 

**Authors**: Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiabao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, Shengjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14878)  

**Abstract**: Generative Retrieval (GR) offers a promising paradigm for recommendation through next-token prediction (NTP). However, scaling it to large-scale industrial systems introduces three challenges: (i) within a single request, the identical model inputs may produce inconsistent outputs due to the pagination request mechanism; (ii) the prohibitive cost of encoding long user behavior sequences with multi-token item representations based on semantic IDs, and (iii) aligning the generative policy with nuanced user preference signals. We present GenRec, a preference-oriented generative framework deployed on the JD App that addresses above challenges within a single decoder-only architecture. For training objective, we propose Page-wise NTP task, which supervises over an entire interaction page rather than each interacted item individually, providing denser gradient signal and resolving the one-to-many ambiguity of point-wise training. On the prefilling side, an asymmetric linear Token Merger compresses multi-token Semantic IDs in the prompt while preserving full-resolution decoding, reducing input length by ~2X with negligible accuracy loss. To further align outputs with user satisfaction, we introduce GRPO-SR, a reinforcement learning method that pairs Group Relative Policy Optimization with NLL regularization for training stability, and employs Hybrid Rewards combining a dense reward model with a relevance gate to mitigate reward hacking. In month-long online A/B tests serving production traffic, GenRec achieves 9.5% improvement in click count and 8.7% in transaction count over the existing pipeline. 

---
# CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations 

**Authors**: Xiping Li, Aier Yang, Jianghong Ma, Kangzhe Liu, Shanshan Feng, Haijun Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14586)  

**Abstract**: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritize accuracy over diversity, overlooking their inherent trade-off. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate over-smoothness issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. To bridge this gap, we propose two new modules. First, Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterests while then quantitatively measuring preference strength to mitigate over-smoothing in graph convolutions. Second, Preference-informed Representation Generation (PRG) module leverages LLMs to generate contextualized descriptions of games and players by reasoning personal preferences from comparing global and personal interests, thereby refining representations of players and games. Experiments on \textcolor{black}{two Steam datasets} demonstrate CPGRec+'s superior accuracy and diversity over state-of-the-art models. The code is accessible at this https URL. 

---
# Evaluation of Agents under Simulated AI Marketplace Dynamics 

**Authors**: To Eun Kim, Alireza Salemi, Hamed Zamani, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14256)  

**Abstract**: Modern information access ecosystems consist of mixtures of systems, such as retrieval systems and large language models, and increasingly rely on marketplaces to mediate access to models, tools, and data, making competition between systems inherent to deployment. In such settings, outcomes are shaped not only by benchmark quality but also by competitive pressure, including user switching, routing decisions, and operational constraints. Yet evaluation is still largely conducted on static benchmarks with accuracy-focused measures that assume systems operate in isolation. This mismatch makes it difficult to predict post-deployment success and obscures competitive effects such as early-adoption advantages and market dominance. We introduce Marketplace Evaluation, a simulation-based paradigm that evaluates information access systems as participants in a competitive marketplace. By simulating repeated interactions and evolving user and agent preferences, the framework enables longitudinal evaluation and marketplace-level metrics, such as retention and market share, that complement and can extend beyond traditional accuracy-based metrics. We formalize the framework and outline a research agenda, motivated by business and economics, around marketplace simulation, metrics, optimization, and adoption in evaluation campaigns like TREC. 

---
# Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge 

**Authors**: Andre Bacellar  

**Link**: [PDF](https://arxiv.org/pdf/2604.14488)  

**Abstract**: In any domain where knowledge accumulates under formal authority -- law, drug regulation, software security -- a later document can formally void an earlier one while remaining semantically distant from it. We formalize this as Controlling Authority Retrieval (CAR): recovering the active frontier front(cl(A_k(q))) of the authority closure of the semantic anchor set -- a different mathematical problem from argmax_d s(q,d). The two central results are: Theorem 4 (CAR-Correctness Characterization) gives necessary-and-sufficient conditions on any retrieved set R for TCA(R,q)=1 -- frontier inclusion and no-ignored-superseder -- independent of how R was produced. Proposition 2 (Scope Identifiability Upper Bound) establishes phi(q) as a hard worst-case ceiling: for any scope-indexed algorithm, TCA@k <= phi(q) * R_anchor(q), proved by an adversarial permutation argument. Three independent real-world corpora validate the proved structure: security advisories (Dense TCA@5=0.270, two-stage 0.975), SCOTUS overruling pairs (Dense=0.172, two-stage 0.926), FDA drug records (Dense=0.064, two-stage 0.774). A GPT-4o-mini experiment shows the downstream cost: Dense RAG produces explicit "not patched" claims for 39% of queries where a patch exists; Two-Stage cuts this to 16%. Four benchmark datasets, domain adapters, and a single-command scorer are released at this https URL. 

---
# Adaptive Query Routing: A Tier-Based Framework for Hybrid Retrieval Across Financial, Legal, and Medical Documents 

**Authors**: Afshan Hashmi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14222)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the standard paradigm for grounding Large Language Model outputs in external knowledge. Lumer et al. [1] presented the first systematic evaluation comparing vector-based agentic RAG against hierarchical node-based reasoning systems for financial document QA across 1,200 SEC filings, finding vector-based systems achieved a 68% win rate. Concurrently, the PageIndex framework [2] demonstrated 98.7% accuracy on FinanceBench through purely reasoning-based retrieval. This paper extends their work by: (i) implementing and evaluating three retrieval architectures: Vector RAG, Tree Reasoning, and the proposed Adaptive Hybrid Retrieval (AHR) across financial, legal, and medical domains; (ii) introducing a four-tier query complexity benchmark; and (iii) employing GPT-4-powered LLM-as-judge evaluation. Experiments reveal that Tree Reasoning achieves the highest overall score (0.900), but no single paradigm dominates across all tiers: Vector RAG wins on multi-document synthesis (Tier 4, score 0.900), while the Hybrid AHR achieves the best performance on cross-reference (0.850) and multi-section queries (0.929). Cross-reference recall reaches 100% for tree-based and hybrid approaches versus 91.7% for vector search, quantifying a critical capability gap. Validation on FinanceBench (150 expert-annotated questions on real SEC 10-K and 10-Q filings) confirms and strengthens these findings: Tree Reasoning scores 0.938, Hybrid AHR 0.901, and Vector RAG 0.821, with the Tree--Vector quality gap widening to 11.7 percentage points on real-world documents. These findings support the development of adaptive retrieval systems that dynamically select strategies based on query complexity and document structure. All code and data are publicly available. 

---
# Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG 

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14572)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics. 

---
# PriHA: A RAG-Enhanced LLM Framework for Primary Healthcare Assistant in Hong Kong 

**Authors**: Richard Wai Cheung Chan, Shanru Lin, Ya-nan Ma, Hao Chen, Liangjun Jiang, Wenqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14215)  

**Abstract**: To address the unsustainable rise in public health expenditures, the Hong Kong SAR Government is shifting its strategic focus to primary healthcare and encouraging citizens to use community resources to self-manage their health. However, official clinical guidelines are fragmented across disparate departments and formats, creating significant access barriers. While general-purpose Large Language Models (LLMs) such as ChatGPT and DeepSeek offer potential solutions for information accessibility, they are prone to generating factually inaccurate content due to a lack of localized and domain-specific knowledge. To this end, we propose a Retrieval-Augmented Generation-Enhanced LLM system as Primary Healthcare Assistant (PriHA) in Hong Kong. Specifically, a tri-stage pipeline is proposed that leverages a query optimizer to generalize user intent-oriented sub-queries, followed by a novel Dual Retrieval Augmented Generation (DRAG) architecture for mixed-source retrieval and context-reorganized generation. Comprehensive experiments and a detailed case study demonstrate that our proposed method can outperform both ablations and baseline in terms of accuracy and clarity. Our research provides a reliable and traceable dialogue retrieval framework for exploring other high-risk, localized application scenarios. 

---
# SAGER: Self-Evolving User Policy Skills for Recommendation Agent 

**Authors**: Zhen Tao, Riwei Lai, Chenyun Yu, Weixin Chen, Li Chen, Beibei Kong, Lei Cheng, Chengxiang Zhuo, Zang Li, Qingqiang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.14972)  

**Abstract**: Large language model (LLM) based recommendation agents personalize what they know through evolving per-user semantic memory, yet how they reason remains a universal, static system prompt shared identically across all users. This asymmetry is a fundamental bottleneck: when a recommendation fails, the agent updates its memory of user preferences but never interrogates the decision logic that produced the failure, leaving its reasoning process structurally unchanged regardless of how many mistakes it accumulates. To address this bottleneck, we propose SAGER (Self-Evolving Agent for Personalized Recommendation), the first recommendation agent framework in which each user is equipped with a dedicated policy skill, a structured natural-language document encoding personalized decision principles that evolves continuously through interaction. SAGER introduces a two-representation skill architecture that decouples a rich evolution substrate from a minimal inference-time injection, an incremental contrastive chain-of-thought engine that diagnoses reasoning flaws by contrasting accepted against unchosen items while preserving accumulated priors, and skill-augmented listwise reasoning that creates fine-grained decision boundaries where the evolved skill provides genuine discriminative value. Experiments on four public benchmarks demonstrate that SAGER achieves state-of-the-art performance, with gains orthogonal to memory accumulation, confirming that personalizing the reasoning process itself is a qualitatively distinct source of recommendation improvement. 

---
# FRESCO: Benchmarking and Optimizing Re-rankers for Evolving Semantic Conflict in Retrieval-Augmented Generation 

**Authors**: Sohyun An, Hayeon Lee, Shuibenyang Yuan, Chun-cheng Jason Chen, Cho-Jui Hsieh, Vijai Mohan, Alexander Min  

**Link**: [PDF](https://arxiv.org/pdf/2604.14227)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a key approach to mitigating the temporal staleness of large language models (LLMs) by grounding responses in up-to-date evidence. Within the RAG pipeline, re-rankers play a pivotal role in selecting the most useful documents from retrieved candidates. However, existing benchmarks predominantly evaluate re-rankers in static settings and do not adequately assess performance under evolving information -- a critical gap, as real-world systems often must choose among temporally different pieces of evidence. To address this limitation, we introduce FRESCO (Factual Recency and Evolving Semantic COnflict), a benchmark for evaluating re-rankers in temporally dynamic contexts. By pairing recency-seeking queries with historical Wikipedia revisions, FRESCO tests whether re-rankers can prioritize factually recent evidence while maintaining semantic relevance. Our evaluation reveals a consistent failure mode across existing re-rankers: a strong bias toward older, semantically rich documents, even when they are factually obsolete. We further investigate an instruction optimization framework to mitigate this issue. By identifying Pareto-optimal instructions that balance Evolving and Non-Evolving Knowledge tasks, we obtain gains of up to 27% on Evolving Knowledge tasks while maintaining competitive performance on Non-Evolving Knowledge tasks. 

---
# Knowledge Graph RAG: Agentic Crawling and Graph Construction in Enterprise Documents 

**Authors**: Koushik Chakraborty, Koyel Guha  

**Link**: [PDF](https://arxiv.org/pdf/2604.14220)  

**Abstract**: This research paper addresses the limitations of semantic search in complex enterprise document ecosystems. Traditional RAG pipelines often fail to capture hierarchical and interconnected information, leading to retrieval inaccuracies. We propose Agentic Knowledge Graphs featuring Recursive Crawling as a robust solution for navigating superseding logic and multi-hop references. Our benchmark evaluation using the Code of Federal Regulations (CFR) demonstrates that this Knowledge Graph-enhanced approach achieves a 70% accuracy improvement over standard vector-based RAG systems, providing exhaustive and precise answers for complex regulatory queries. 

---
# TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations 

**Authors**: Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14223)  

**Abstract**: Traditional conversational travel recommender systems primarily optimize for user relevance and convenience, often reinforcing popular, overcrowded destinations and carbon-intensive travel choices. To address this, we present TRACE (Tourism Recommendation with Agentic Counterfactual Explanations), a multi-agent, LLM-based framework that promotes sustainable tourism through interactive nudging. TRACE uses a modular orchestrator-worker architecture where specialized agents elicit latent sustainability preferences, construct structured user personas, and generate recommendations that balance relevance with environmental impact.
A key innovation lies in its use of agentic counterfactual explanations and LLM-driven clarifying questions, which together surface greener alternatives and refine understanding of intent, fostering user reflection without coercion. User studies and semantic alignment analyses demonstrate that TRACE effectively supports sustainable decision-making while preserving recommendation quality and interactive responsiveness. TRACE is implemented on Google's Agent Development Kit, with full code, Docker setup, prompts, and a publicly available demo video to ensure reproducibility.
A project summary, including all resources, prompts, and demo access, is available at this https URL. 

---
# IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15148)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly. 

---
# APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI 

**Authors**: Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2604.14362)  

**Abstract**: Large language models still struggle with reliable long-term conversational memory: simply enlarging context windows or applying naive retrieval often introduces noise and destabilizes responses. We present APEX-MEM, a conversational memory system that combines three key innovations: (1) a property graph which uses domain-agnostic ontology to structure conversations as temporally grounded events in an entity-centric framework, (2) append-only storage that preserves the full temporal evolution of information, and (3) a multi-tool retrieval agent that understands and resolves conflicting or evolving information at query time, producing a compact and contextually relevant memory summary. This retrieval-time resolution preserves the full interaction history while suppressing irrelevant details. APEX-MEM achieves 88.88% accuracy on LOCOMO's Question Answering task and 86.2% on LongMemEval, outperforming state-of-the-art session-aware approaches and demonstrating that structured property graphs enable more temporally coherent long-term conversational reasoning. 

---
# A Unified Model and Document Representation for On-Device Retrieval-Augmented Generation 

**Authors**: Julian Killingback, Ofer Meshi, Henry Li, Hamed Zamani, Maryam Karimzadehgan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14403)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) approaches generally assume that retrieval and generation occur on powerful servers removed from the end user. While this reduces local hardware constraints, it introduces significant drawbacks: privacy concerns regarding data access, recurring maintenance and storage costs, increased latency, and the necessity of an internet connection. On-device RAG addresses these challenges by executing the entire pipeline locally, making it ideal for querying sensitive personal information such as financial documents, contact details, and medical history. However, on-device deployment necessitates a delicate balance between limited memory and disk space. Specifically, the context size provided to the generative model must be restricted to manage KV cache and attention memory usage, while the size of stored embeddings must be minimized to preserve disk space. In this work, we propose a unified model that compresses the RAG context and utilizes the same representations for retrieval. This approach minimizes disk utilization compared to using separate representations, while significantly reducing the context size required for generation. With an average of 1/10 of the context, our model matches the performance of a traditional RAG reader without increasing storage requirements compared to a multi-vector retrieval model. This approach represents the first model to unify retrieval and context compression using a shared model and representation. We believe this work will inspire further consolidation of distinct models to optimize on-device performance. 

---
# How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study 

**Authors**: Zhen Yang, Ping Jian, Zhongbin Guo, Zuming Zhang, Chengzhi Li, Yonghong Deng, Xinyue Zhang, Wenpeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15294)  

**Abstract**: Over the past year, spatial intelligence has drawn increasing attention. Many prior works study it from the perspective of visual-spatial intelligence, where models have access to visuospatial information from visual inputs. However, in the absence of visual information, whether linguistic intelligence alone is sufficient to endow models with spatial intelligence, and how models perform relevant tasks with text-only inputs still remain unexplored. Therefore, in this paper, we focus on a fundamental and critical capability in spatial intelligence from a linguistic perspective: viewpoint rotation understanding (VRU). Specifically, LLMs and VLMs are asked to infer their final viewpoint and predict the corresponding observation in an environment given textual description of viewpoint rotation and observation over multiple steps. We find that both LLMs and VLMs perform poorly on our proposed dataset while human can easily achieve 100% accuracy, indicating a substantial gap between current model capabilities and the requirements of spatial intelligence. To uncover the underlying mechanisms, we conduct a layer-wise probing analysis and head-wise causal intervention. Our findings reveal that although models encode viewpoint information in the hidden states, they appear to struggle to bind the viewpoint position with corresponding observation, resulting in a hallucination in final layers. Finally, we selectively fine-tune the key attention heads identified by causal intervention to improve VRU performance. Experimental results demonstrate that such selective fine-tuning achieves improved VRU performance while avoiding catastrophic forgetting of generic abilities. Our dataset and code will be released at this https URL . 

---
# Agent-Aided Design for Dynamic CAD Models 

**Authors**: Mitch Adler, Matthew Russo, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2604.15184)  

**Abstract**: In the past year, researchers have started to create agentic systems that can design real-world CAD-style objects in a training-free setting, a new variety of system that we call Agent-Aided Design. Generally speaking, these systems place an agent in a feedback loop in which it can write code, compile that code to an assembly of CAD model(s), visualize the model, and then iteratively refine its code based on visual and other feedback. Despite rapid progress, a key problem remains: none of these systems can build complex 3D assemblies with moving parts. For example, no existing system can build a piston, a pendulum, or even a pair of scissors. In order for Agent-Aided Design to make a real impact in industrial manufacturing, we need a system that is capable of generating such 3D assemblies. In this paper we present a prototype of AADvark, an agentic system designed for this task. Unlike previous state-of-the-art systems, AADvark captures the dynamic part interactions with one or more degrees-of-freedom. This design decision allows AADvark to reason directly about assemblies with moving parts and can thereby achieve cross-cutting goals, including but not limited to mechanical movements. Unfortunately, current LLMs are imperfect spatial reasoners, a problem that AADvark addresses by incorporating external constraint solver tools with a specialized visual feedback mechanism. We demonstrate that, by modifying the agent's tools (FreeCAD and the assembly solver), we are able to create a strong verification signal which enables our system to build 3D assemblies with movable parts. 

---
# Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation 

**Authors**: Ziyang Chen, Renbing Chen, Daowei Li, Jinzhi Liao, Jiashen Sun, Ke Zeng, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15190)  

**Abstract**: Simulating group-level user behavior enables scalable counterfactual evaluation of merchant strategies without costly online experiments. However, building a trustworthy simulator faces two structural challenges. First, information incompleteness causes reasoning-based simulators to over-rationalize when unobserved factors such as offline context and implicit habits are missing. Second, mechanism duality requires capturing both interpretable preferences and implicit statistical regularities, which no single paradigm achieves alone. We propose Policy-Guided Hybrid Simulation (PGHS), a dual-process framework that mines transferable decision policies from behavioral trajectories and uses them as a shared alignment layer. This layer anchors an LLM-based reasoning branch that prevents over-rationalization and an ML-based fitting branch that absorbs implicit regularities. Group-level predictions from both branches are fused for complementary correction. We deploy PGHS on Meituan with 101 merchants and over 26,000 trajectories. PGHS achieves a group simulation error of 8.80%, improving over the best reasoning-based and fitting-based baselines by 45.8% and 40.9% respectively. 

---
# Blue Data Intelligence Layer: Streaming Data and Agents for Multi-source Multi-modal Data-Centric Applications 

**Authors**: Moin Aminnaseri, Farima Fatahi Bayat, Nikita Bhutani, Jean-Flavien Bussotti, Kevin Chan, Rafael Li Chen, Yanlin Feng, Jackson Hassell, Estevam Hruschka, Eser Kandogan, Hannah Kim, James Levine, Seiji Maekawa, Jalal Mahmud, Kushan Mitra, Naoki Otani, Pouya Pezeshkpour, Nima Shahbazi, Chen Shen, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15233)  

**Abstract**: NL2SQL systems aim to address the growing need for natural language interaction with data. However, real-world information rarely maps to a single SQL query because (1) users express queries iteratively (2) questions often span multiple data sources beyond the closed-world assumption of a single database, and (3) queries frequently rely on commonsense or external knowledge. Consequently, satisfying realistic data needs require integrating heterogeneous sources, modalities, and contextual data. In this paper, we present Blue's Data Intelligence Layer (DIL) designed to support multi-source, multi-modal, and data-centric applications. Blue is a compound AI system that orchestrates agents and data for enterprise settings. DIL serves as the data intelligence layer for agentic data processing, to bridge the semantic gap between user intent and available information by unifying structured enterprise data, world knowledge accessible through LLMs, and personal context obtained through interaction. At the core of DIL is a data registry that stores metadata for diverse data sources and modalities to enable both native and natural language queries. DIL treats LLMs, the Web, and the User as source 'databases', each with their own query interface, elevating them to first-class data sources. DIL relies on data planners to transform user queries into executable query plans. These plans are declarative abstractions that unify relational operators with other operators spanning multiple modalities. DIL planners support decomposition of complex requests into subqueries, retrieval from diverse sources, and finally reasoning and integration to produce final results. We demonstrate DIL through two interactive scenarios in which user queries dynamically trigger multi-source retrieval, cross-modal reasoning, and result synthesis, illustrating how compound AI systems can move beyond single database NL2SQL. 

---
# OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis 

**Authors**: Kanzhi Cheng, Zehao Li, Zheng Ma, Nuo Chen, Jialin Cao, Qiushi Sun, Zichen Ding, Fangzhi Xu, Hang Yan, Jiajun Chen, Anh Tuan Luu, Jianbing Zhang, Lewei Lu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15093)  

**Abstract**: Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present OpenMobile, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions. and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7% and 64.7% on AndroidWorld, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting. We release data and code at this https URL to bridge the data gap and facilitate broader mobile agent research. 

---
# Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations 

**Authors**: Manan Gupta, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15302)  

**Abstract**: LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\bar{\rho} = 0.8$-$4.1\%$), with $33$-$67\%$ of documents exhibiting at least one directed 3-cycle; and $\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\geq(1{-}\alpha)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\approx 3.0$) and coherence moderately so (avg. set size $\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\approx 4.9$). We release all code, prompts, and cached results. 

---
# Where are the Humans? A Scoping Review of Fairness in Multi-agent AI Systems 

**Authors**: Simeon Allmendinger, Luca Deck, Lucas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2604.15078)  

**Abstract**: Rapid advances in Generative AI are giving rise to increasingly sophisticated Multi-Agent AI (MAAI) systems. While AI fairness has been extensively studied in traditional predictive scenarios, its examination in MAAI remains nascent and fragmented. This scoping review critically synthesizes existing research on fairness in MAAI systems. Through a qualitative content analysis of 23 selected studies, we identify five archetypal approaches. Our findings reveal that fairness in MAAI systems is often addressed superficially, lacks robust normative foundations, and frequently overlooks the complex dynamics introduced by agent autonomy and system-level interactions. We argue that fairness must be embedded structurally throughout the development lifecycle of MAAI, rather than appended as a post-hoc consideration. Meaningful evaluation requires explicit human oversight, normative clarity, and a precise articulation of fairness objectives and beneficiaries. This review provides a foundation for advancing fairness research in MAAI systems by highlighting critical gaps, exposing prevailing limitations, and suggesting pathways. 

---
# Autogenesis: A Self-Evolving Agent Protocol 

**Authors**: Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15034)  

**Abstract**: Recent advances in LLM based agent systems have shown promise in tackling complex, long horizon tasks. However, existing agent protocols (e.g., A2A and MCP) under specify cross entity lifecycle and context management, version tracking, and evolution safe update interfaces, which encourages monolithic compositions and brittle glue code. We introduce \textbf{\textsc{Autogenesis Protocol (AGP)}}, a self evolution protocol that decouples what evolves from how evolution occurs. Its Resource Substrate Protocol Layer (RSPL) models prompts, agents, tools, environments, and memory as protocol registered resources\footnote{Unless otherwise specified, resources refer to instances of the five RSPL entity types: \emph{prompt}, \emph{agent}, \emph{tool}, \emph{environment}, \emph{memory} with agent \emph{outputs}.} with explicit state, lifecycle, and versioned interfaces. Its Self Evolution Protocol Layer (SEPL) specifies a closed loop operator interface for proposing, assessing, and committing improvements with auditable lineage and rollback. Building on \textbf{\textsc{AGP}}, we present \textbf{\textsc{Autogenesis System (AGS)}}, a self-evolving multi-agent system that dynamically instantiates, retrieves, and refines protocol-registered resources during execution. We evaluate \textbf{\textsc{AGS}} on multiple challenging benchmarks that require long horizon planning and tool use across heterogeneous resources. The results demonstrate consistent improvements over strong baselines, supporting the effectiveness of agent resource management and closed loop self evolution. 

---
# Generalization in LLM Problem Solving: The Case of the Shortest Path 

**Authors**: Yao Tong, Jiayuan Ye, Anastasia Borovykh, Reza Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2604.15306)  

**Abstract**: Whether language models can systematically generalize remains actively debated. Yet empirical performance is jointly shaped by multiple factors such as training data, training paradigms, and inference-time strategies, making failures difficult to interpret. We introduce a controlled synthetic environment based on shortest-path planning, a canonical composable sequential optimization problem. The setup enables clean separation of these factors and supports two orthogonal axes of generalization: spatial transfer to unseen maps and length scaling to longer-horizon problems. We find that models exhibit strong spatial transfer but consistently fail under length scaling due to recursive instability. We further analyze how distinct stages of the learning pipeline influence systematic problem-solving: for example, data coverage sets capability limits; reinforcement learning improves training stability but does not expand those limits; and inference-time scaling enhances performance but cannot rescue length-scaling failures. 

---
# Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding 

**Authors**: Hatice Merve Vural, Doga Kukul, Ege Erdem Ozlu, Demir Ekin Arikan, Bob Mankoff, Erkut Erdem, Aykut Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2604.15210)  

**Abstract**: Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: incongruity modeling, which identifies mismatches in the visual scene; resolution modeling, which constructs coherent reinterpretations of these mismatches; and preference alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interpretation explicit and learnable. Across 7B, 32B, and 72B models on NYCC, IRS outperforms strong open and closed multimodal baselines across caption matching and ranking tasks, with our largest model approaching expert-level performance on ranking. Zero-shot transfer to external benchmarks shows that IRS learns generalizable reasoning patterns. Our results suggest that supervising reasoning structure, rather than scale alone, is key for reasoning-centric tasks. 

---
# Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching 

**Authors**: Aihua Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15009)  

**Abstract**: Flow matching retains the generation quality of diffusion models while enabling substantially faster inference, making it a compelling paradigm for generative modeling. However, when applied to language modeling, it exhibits fundamental limitations in representing complex latent distributions with irregular geometries, such as anisotropy and multimodality. To address these challenges, we propose a mixture-of-experts flow matching (MoE-FM) framework, which captures complex global transport geometries in latent space by decomposing them into locally specialized vector fields. Building on MoE-FM, we develop a non-autoregressive (NAR) language modeling approach, named YAN, instantiated with both Transformer and Mamba architectures. Across multiple downstream tasks, YAN achieves generation quality on par with both autoregressive (AR) and diffusion-based NAR language models, while requiring as few as three sampling steps. This yields a $40\times$ speedup over AR baselines and up to a $10^3\times$ speedup over diffusion language models, demonstrating substantial efficiency advantages for language modeling. 

---
# Context Over Content: Exposing Evaluation Faking in Automated Judges 

**Authors**: Manan Gupta, Inderjeet Nair, Lu Wang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15224)  

**Abstract**: The $\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $\Delta V = -9.8 pp$ (a $30\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking. 

---
# Dr.~RTL: Autonomous Agentic RTL Optimization through Tool-Grounded Self-Improvement 

**Authors**: Wenji Fang, Yao Lu, Shang Liu, Jing Wang, Ziyan Guo, Junxian He, Fengbin Tu, Zhiyao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.14989)  

**Abstract**: Recent advances in large language models (LLMs) have sparked growing interest in automatic RTL optimization for better performance, power, and area (PPA). However, existing methods are still far from realistic RTL optimization. Their evaluation settings are often unrealistic: they are tested on manually degraded, small-scale RTL designs and rely on weak open-source tools. Their optimization methods are also limited, relying on coarse design-level feedback and simple pre-defined rewriting rules. To address these limitations, we present Dr. RTL, an agentic framework for RTL timing optimization in a realistic evaluation environment, with continual self-improvement through reusable optimization skills. We establish a realistic evaluation setting with more challenging RTL designs and an industrial EDA workflow. Within this setting, Dr. RTL performs closed-loop optimization through a multi-agent framework for critical-path analysis, parallel RTL rewriting, and tool-based evaluation. We further introduce group-relative skill learning, which compares parallel RTL rewrites and distills the optimization experience into an interpretable skill library. Currently, this library contains 47 pattern--strategy entries for cross-design reuse to improve PPA and accelerate convergence, and it can continue evolving over time. Evaluated on 20 real-world RTL designs, Dr. RTL achieves average WNS/TNS improvements of 21\%/17\% with a 6\% area reduction over the industry-leading commercial synthesis tool. 

---
# COEVO: Co-Evolutionary Framework for Joint Functional Correctness and PPA Optimization in LLM-Based RTL Generation 

**Authors**: Heng Ping, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Shukai Duan, Xiaole Zhang, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2604.15001)  

**Abstract**: LLM-based RTL code generation methods increasingly target both functional correctness and PPA quality, yet existing approaches universally decouple the two objectives, optimizing PPA only after correctness is fully achieved. Whether through sequential multi-agent pipelines, evolutionary search with binary correctness gates, or hierarchical reward dependencies, partially correct but architecturally promising candidates are systematically discarded. Moreover, existing methods reduce the multi-objective PPA space to a single scalar fitness, obscuring the trade-offs among area, delay, and power. To address these limitations, we propose COEVO, a co-evolutionary framework that unifies correctness and PPA optimization within a single evolutionary loop. COEVO formulates correctness as a continuous co-optimization dimension alongside area, delay, and power, enabled by an enhanced testbench that provides fine-grained scoring and detailed diagnostic feedback. An adaptive correctness gate with annealing allows PPA-promising but partially correct candidates to guide the search toward jointly optimal solutions. To preserve the full PPA trade-off structure, COEVO employs four-dimensional Pareto-based non-dominated sorting with configurable intra-level sorting, replacing scalar fitness without manual weight tuning. Evaluated on VerilogEval 2.0 and RTLLM 2.0, COEVO achieves 97.5\% and 94.5\% Pass@1 with GPT-5.4-mini, surpassing all agentic baselines across four LLM backbones, while attaining the best PPA on 43 out of 49 synthesizable RTLLM designs. 

---
# Hybrid Decision Making via Conformal VLM-generated Guidance 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2604.14980)  

**Abstract**: Building on recent advances in AI, hybrid decision making (HDM) holds the promise of improving human decision quality and reducing cognitive load. We work in the context of learning to guide (LtG), a recently proposed HDM framework in which the human is always responsible for the final decision: rather than suggesting decisions, in LtG the AI supplies (textual) guidance useful for facilitating decision making. One limiting factor of existing approaches is that their guidance compounds information about all possible outcomes, and as a result it can be difficult to digest. We address this issue by introducing ConfGuide, a novel LtG approach that generates more succinct and targeted guidance. To this end, it employs conformal risk control to select a set of outcomes, ensuring a cap on the false negative rate. We demonstrate our approach on a real-world multi-label medical diagnosis task. Our empirical evaluation highlights the promise of ConfGuide. 

---
# Discovering Novel LLM Experts via Task-Capability Coevolution 

**Authors**: Andrew Dai, Boris Meinardus, Ciaran Regan, Yingtao Tian, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14969)  

**Abstract**: Frontier model developers aim to train models continually to possess emergent, diverse capabilities. To extend capabilities, the current pre-training and post-training paradigm requires manually starting training runs with static datasets or reward functions every time. Addressing this limitation, our work pursues the insight that open-endedness (via the coevolution of models and tasks) can discover models with increasingly novel skills in a single run. We introduce a new model development framework that extends coevolution to large language model (LLM) discovery, open-ended \textit{Assessment Coevolving with Diverse Capabilities} (AC/DC). AC/DC evolves both LLMs via model merging and natural language tasks via synthetic data generation. AC/DC discovers growing archives of LLMs that surpass the capabilities of larger LLMs while taking up less GPU memory. In particular, our LLM populations achieve a broader Coverage of expertise than other curated models or baselines on downstream benchmarks, without \textit{any} explicit benchmark optimization. Furthermore, AC/DC improves Coverage over time, continually innovates on tasks and models, and improves performance in multi-agent best-of-N selection. Our findings highlight the potential of coevolution as a means of discovering broader sets of capabilities from base LLMs. Overall, AC/DC brings us one step closer to a profoundly new paradigm of LLM development, where continual improvements to the diversity of model capabilities can be accelerated by leveraging existing models as stepping stones to increasingly powerful models. 

---
# RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography 

**Authors**: Mélanie Roschewitz, Kenneth Styppa, Yitian Tao, Jiwoong Sohn, Jean-Benoit Delbrouck, Benjamin Gundersen, Nicolas Deperrois, Christian Bluethgen, Julia Vogt, Bjoern Menze, Farhad Nooralahzadeh, Michael Krauthammer, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2604.15231)  

**Abstract**: Vision-language models (VLM) have markedly advanced AI-driven interpretation and reporting of complex medical imaging, such as computed tomography (CT). Yet, existing methods largely relegate clinicians to passive observers of final outputs, offering no interpretable reasoning trace for them to inspect, validate, or refine. To address this, we introduce RadAgent, a tool-using AI agent that generates CT reports through a stepwise and interpretable process. Each resulting report is accompanied by a fully inspectable trace of intermediate decisions and tool interactions, allowing clinicians to examine how the reported findings are derived. In our experiments, we observe that RadAgent improves Chest CT report generation over its 3D VLM counterpart, CT-Chat, across three dimensions. Clinical accuracy improves by 6.0 points (36.4% relative) in macro-F1 and 5.4 points (19.6% relative) in micro-F1. Robustness under adversarial conditions improves by 24.7 points (41.9% relative). Furthermore, RadAgent achieves 37.0% in faithfulness, a new capability entirely absent in its 3D VLM counterpart. By structuring the interpretation of chest CT as an explicit, tool-augmented and iterative reasoning trace, RadAgent brings us closer toward transparent and reliable AI for radiology. 

---
# ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints 

**Authors**: Pei-An Chen, Yong-Ching Liang, Jia-Fong Yeh, Hung-Ting Su, Yi-Ting Chen, Min Sun, Winston Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14902)  

**Abstract**: Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding. 

---
# MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration 

**Authors**: Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14889)  

**Abstract**: While Chain-of-thought (CoT) reasoning enables LLMs to solve challenging reasoning problems, as KV cache grows linearly with the number of generated tokens, CoT reasoning faces scaling issues in terms of speed and memory usage. In this work, we propose MemoSight (Memory-Foresight-based reasoning), a unified framework that integrates both context compression and multi-token prediction to mitigate the efficiency issues while maintaining CoT reasoning performance. Our framework adopts the same minimalist design for both context compression and multi-token prediction via special tokens and their corresponding position layout tailored to each token type. Comprehensive experiments on four reasoning benchmarks demonstrate that MemoSight reduces the KV cache footprint by up to 66% and accelerates inference by 1.56x, while outperforming existing CoT compression methods. 

---
# The Missing Knowledge Layer in AI: A Framework for Stable Human-AI Reasoning 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14881)  

**Abstract**: Large language models are increasingly integrated into decision-making in areas such as healthcare, law, finance, engineering, and government. Yet they share a critical limitation: they produce fluent outputs even when their internal reasoning has drifted. A confident answer can conceal uncertainty, speculation, or inconsistency, and small changes in phrasing can lead to different conclusions. This makes LLMs useful assistants but unreliable partners in high-stakes contexts.
Humans exhibit a similar weakness, often mistaking fluency for reliability. When a model responds smoothly, users tend to trust it, even when both model and user are drifting together. This paper is the first in a five-paper research series on stabilising human-AI reasoning. The series proposes a two-layer approach: Parts II-IV introduce human-side mechanisms such as uncertainty cues, conflict surfacing, and auditable reasoning traces, while Part V develops a model-side Epistemic Control Loop (ECL) that detects instability and modulates generation accordingly.
Together, these layers form a missing operational substrate for governance by increasing signal-to-noise at the point of use. Stabilising interaction makes uncertainty and drift visible before enforcement is applied, enabling more precise capability governance. This aligns with emerging compliance expectations, including the EU AI Act and ISO/IEC 42001, by making reasoning processes traceable under real conditions of use.
The central claim is that fluency is not reliability. Without structures that stabilise both human and model reasoning, AI cannot be trusted or governed where it matters most. 

---
# Governing Reflective Human-AI Collaboration: A Framework for Epistemic Scaffolding and Traceable Reasoning 

**Authors**: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14898)  

**Abstract**: Large language models have advanced rapidly, from pattern recognition to emerging forms of reasoning, yet they remain confined to linguistic simulation rather than grounded understanding. They can produce fluent outputs that resemble reflection, but lack temporal continuity, causal feedback, and anchoring in real-world interaction. This paper proposes a complementary approach in which reasoning is treated as a relational process distributed between human and model rather than an internal capability of either.
Building on recent work on "System-2" learning, we relocate reflective reasoning to the interaction layer. Instead of engineering reasoning solely within models, we frame it as a cognitive protocol that can be structured, measured, and governed using existing systems. This perspective emphasizes collaborative intelligence, combining human judgment and contextual understanding with machine speed, memory, and associative capacity.
We introduce "The Architect's Pen" as a practical method. Like an architect who thinks through drawing, the human uses the model as an external medium for structured reflection. By embedding phases of articulation, critique, and revision into human-AI interaction, the dialogue itself becomes a reasoning loop: human abstraction -> model articulation -> human reflection.
This reframes the question from whether the model can think to whether the human-AI system can reason. The framework enables auditable reasoning traces and supports alignment with emerging governance standards, including the EU AI Act and ISO/IEC 42001. It provides a practical path toward more transparent, controllable, and accountable AI use without requiring new model architectures. 

---
# Dual-Axis Generative Reward Model Toward Semantic and Turn-taking Robustness in Interactive Spoken Dialogue Models 

**Authors**: Yifu Chen, Shengpeng Ji, Zhengqing Liu, Qian Chen, Wen Wang, Ziqing Wang, Yangzhuo Li, Tianle Liang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14920)  

**Abstract**: Achieving seamless, human-like interaction remains a key challenge for full-duplex spoken dialogue models (SDMs). Reinforcement learning (RL) has substantially enhanced text- and vision-language models, while well-designed reward signals are crucial for the performance of RL. We consider RL a promising strategy to address the key challenge for SDMs. However, a fundamental barrier persists: prevailing automated metrics for assessing interaction quality rely on superficial proxies, such as behavioral statistics or timing-prediction accuracy, failing to provide reliable reward signals for RL. On the other hand, human evaluations, despite their richness, remain costly, inconsistent, and difficult to scale. We tackle this critical barrier by proposing a Dual-Axis Generative Reward Model, which is trained to understand complex interaction dynamics using a detailed taxonomy and an annotated dataset, produces a single score and, crucially, provides separate evaluations for semantic quality and interaction timing. Such dual outputs furnish precise diagnostic feedback for SDMs and deliver a dependable, instructive reward signal suitable for online reinforcement learning. Our model achieves state-of-the-art performance on interaction-quality assessment across a wide spectrum of datasets, spanning synthetic dialogues and complex real-world interactions. 

---
# Toward Agentic RAG for Ukrainian 

**Authors**: Marta Sumyk, Oleksandr Kosovan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14896)  

**Abstract**: We present an initial investigation into Agentic Retrieval-Augmented Generation (RAG) for Ukrainian, conducted within the UNLP 2026 Shared Task on Multi-Domain Document Understanding. Our system combines two-stage retrieval (BGE-M3 with BGE reranking) with a lightweight agentic layer performing query rephrasing and answer-retry loops on top of Qwen2.5-3B-Instruct. Our analysis reveals that retrieval quality is the primary bottleneck: agentic retry mechanisms improve answer accuracy but the overall score remains constrained by document and page identification. We discuss practical limitations of offline agentic pipelines and outline directions for combining stronger retrieval with more advanced agentic reasoning for Ukrainian. 

---
# Beyond Literal Summarization: Redefining Hallucination for Medical SOAP Note Evaluation 

**Authors**: Bhavik Vachhani, Kush Shrisvastava, Pranshu Nema, Sai Chiranthan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14829)  

**Abstract**: Evaluating large language models (LLMs) for clinical documentation tasks such as SOAP note generation remains challenging. Unlike standard summarization, these tasks require clinical abstraction, normalization of colloquial language, and medically grounded inference. However, prevailing evaluation methods including automated metrics and LLM as judge frameworks rely on lexical faithfulness, often labeling any information not explicitly present in the transcript as hallucination.
We show that such approaches systematically misclassify clinically valid outputs as errors, inflating hallucination rates and distorting model assessment. Our analysis reveals that many flagged hallucinations correspond to legitimate clinical transformations, including synonym mapping, abstraction of examination findings, diagnostic inference, and guideline consistent care planning.
By aligning evaluation criteria with clinical reasoning through calibrated prompting and retrieval grounded in medical ontologies we observe a significant shift in outcomes. Under a lexical evaluation regime, the mean hallucination rate is 35%, heavily penalizing valid reasoning. With inference aware evaluation, this drops to 9%, with remaining cases reflecting genuine safety concerns. These findings suggest that current evaluation practices over penalize valid clinical reasoning and may measure artifacts of evaluation design rather than true errors, underscoring the need for clinically informed evaluation in high context domains like medicine. 

---
# MirrorBench: Evaluating Self-centric Intelligence in MLLMs by Introducing a Mirror 

**Authors**: Shengyu Guo, Tongrui Ye, Jianbo Zhang, Zicheng Zhang, Chunyi Li, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.14785)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has demonstrated remarkable advances in perception and reasoning, suggesting their potential for embodied intelligence. While recent studies have evaluated embodied MLLMs in interactive settings, current benchmarks mainly target capabilities to perceive, understand, and interact with external objects, lacking a systematic evaluation of self-centric intelligence. To address this, we introduce MirrorBench, a simulation-based benchmark inspired by the classical Mirror Self-Recognition (MSR) test in psychology. MirrorBench extends this paradigm to embodied MLLMs through a tiered framework of progressively challenging tasks, assessing agents from basic visual perception to high-level self-representation. Experiments on leading MLLMs show that even at the lowest level, their performance remains substantially inferior to human performance, revealing fundamental limitations in self-referential understanding. Our study bridges psychological paradigms and embodied intelligence, offering a principled framework for evaluating the emergence of general intelligence in large models. Project page: this https URL. 

---
# TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models 

**Authors**: Yi Zhao, Yajuan Peng, Cam-Tu Nguyen, Zuchao Li, Xiaoliang Wang, Xiaoming Fu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14847)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks through extended chains of thought but suffer from high inference latency due to autoregressive reasoning. Recent work explores using Small Reasoning Models (SRMs) to accelerate LRM inference. In this paper, we systematically characterize the capability boundaries of SRMs and identify three common types of reasoning risks: (1) path divergence, where SRMs lack the strategic ability to construct an initial plan, causing reasoning to deviate from the most probable path; (2) cognitive overload, where SRMs fail to solve particularly difficult steps; and (3) recovery inability, where SRMs lack robust self-reflection and error correction mechanisms. To address these challenges, we propose TrigReason, a trigger-based collaborative reasoning framework that replaces continuous polling with selective intervention. TrigReason delegates most reasoning to the SRM and activates LRM intervention only when necessary-during initial strategic planning (strategic priming trigger), upon detecting extraordinary overconfidence (cognitive offload trigger), or when reasoning falls into unproductive loops (intervention request trigger). The evaluation results on AIME24, AIME25, and GPQA-D indicate that TrigReason matches the accuracy of full LRMs and SpecReason, while offloading 1.70x - 4.79x more reasoning steps to SRMs. Under edge-cloud conditions, TrigReason reduces latency by 43.9\% and API cost by 73.3\%. Our code is available at \href{this https URL}{this https URL} 

---
# Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX 

**Authors**: Zhonghao Yang, Yu Li, Yanxu Zhu, Tianyi Zhou, Yuejin Xie, Haoyu Luo, Jing Shao, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14858)  

**Abstract**: As agent systems move into increasingly diverse execution settings, trajectory-level safety evaluation and diagnosis require benchmarks that evolve with them. ATBench is a diverse and realistic agent trajectory benchmark for safety evaluation and diagnosis. This report presents ATBench-Claw and ATBench-CodeX, two domain-customized extensions that carry ATBench into the OpenClaw and OpenAI Codex / Codex-runtime settings. The key adaptation mechanism is to analyze each new setting, customize the three-dimensional Safety Taxonomy over risk source, failure mode, and real-world harm, and then use that customized taxonomy to define the benchmark specification consumed by the shared ATBench construction pipeline. This extensibility matters because agent frameworks remain relatively stable at the architectural level even as their concrete execution settings, tool ecosystems, and product capabilities evolve quickly. Concretely, ATBench-Claw targets OpenClaw-sensitive execution chains over tools, skills, sessions, and external actions, while ATBench-CodeX targets trajectories in the OpenAI Codex / Codex-runtime setting over repositories, shells, patches, dependencies, approvals, and runtime policy boundaries. Our emphasis therefore falls on taxonomy customization, domain-specific risk coverage, and benchmark design under a shared ATBench generation framework. 

---
# CogEvolution: A Human-like Generative Educational Agent to Simulate Student's Cognitive Evolution 

**Authors**: Wei Zhang, Yihang Cheng, Zhirong Ye, Kezhen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14786)  

**Abstract**: Generative Agents, owing to their precise modeling and simulation capabilities of human behavior, have become a pivotal tool in the field of Artificial Intelligence in Education (AIEd) for uncovering complex cognitive processes of learners. However, existing educational agents predominantly rely on static personas to simulate student learning behaviors, neglecting the decisive role of deep cognitive capabilities in learning outcomes during practice interactions. Furthermore, they struggle to characterize the dynamic fluidity of knowledge internalization, transfer, and cognitive state transitions. To overcome this bottleneck, this paper proposes a human-like educational agent capable of simulating student cognitive evolution: CogEvolution. Specifically, we first construct a cognitive depth perceptron based on the Interactive, Constructive, Active, Passive (ICAP) taxonomy from cognitive psychology, achieving precise quantification of learner cognitive engagement. Subsequently, we propose a memory retrieval method based on Item Response Theory (IRT) to simulate the connection and assimilation of new and prior knowledge. Finally, we design a dynamic cognitive update mechanism based on evolutionary algorithms to simulate the real-time integration of student learning behaviors and cognitive evolution processes. Comprehensive evaluations demonstrate that CogEvolution not only significantly outperforms baseline models in behavioral fidelity and learning curve fitting but also uniquely reproduces plausible and robust cognitive evolutionary paths consistent with educational psychology expectations, providing a novel paradigm for constructing highly interpretable educational agents. 

---
# The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows 

**Authors**: Hyunwoo Kim, Harin Yu, Hanau Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14807)  

**Abstract**: The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human--AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human--AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise. 

---
# Disentangle-then-Refine: LLM-Guided Decoupling and Structure-Aware Refinement for Graph Contrastive Learning 

**Authors**: Zhaoxing Li, Hai-Feng Zhang, Xiaoming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14746)  

**Abstract**: Conventional Graph Contrastive Learning (GCL) on Text-Attributed Graphs (TAGs) relies on blind stochastic augmentations, inadvertently entangling task-relevant signals with noise. We propose SDM-SCR, a robust framework anchored in Approximate Orthogonal Decomposition. First, the Semantic Decoupling Module (SDM) leverages the instruction-following capability of Large Language Models (LLMs) to actively parse raw attributes into asymmetric, task-oriented signal and noise views. This shifts the paradigm from random perturbation to semantic-aware disentanglement. Subsequently, Semantic Consistency Regularization (SCR) exploits the spectral observation that semantic signals are topologically smooth while residual noise is high-frequency. SCR functions as a selective spectral filter, enforcing consistency only on the signal subspace to eliminate LLM hallucinations without over-smoothing. This ``Disentangle-then-Refine'' mechanism ensures rigorous signal purification. Extensive experiments demonstrate that SDM-SCR achieves SOTA performance in accuracy and efficiency. 

---
# CoTEvol: Self-Evolving Chain-of-Thoughts for Data Synthesis in Mathematical Reasoning 

**Authors**: Zhuo Wang, Zhuo Zhang, Yafu Li, Yu Cheng, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14768)  

**Abstract**: Large Language Models (LLMs) exhibit strong mathematical reasoning when trained on high-quality Chain-of-Thought (CoT) that articulates intermediate steps, yet costly CoT curation hinders further progress. While existing remedies such as distillation from stronger LLMs and self-synthesis based on test-time search alleviate this issue, they often suffer from diminishing returns or high computing this http URL this work, we propose CoTEvol, a genetic evolutionary framework that casts CoT generation as a population-based search over reasoning this http URL trajectories are iteratively evolved through reflective global crossover at the trajectory level and local mutation guided by uncertainty at the step level, enabling holistic recombination and fine-grained refinement. Lightweight, task-aware fitness functions are designed to guide the evolutionary process toward accurate and diverse reasoning. Empirically, CoTEvol improves correct-CoT synthesis success by over 30% and enhances structural diversity, with markedly improved efficiency. LLMs trained on these evolutionary CoT data achieve an average gain of 6.6% across eight math benchmarks, outperforming previous distillation and self-synthesis approaches. These results underscore the promise of evolutionary CoT synthesis as a scalable and effective method for mathematical reasoning tasks. 

---
# Layered Mutability: Continuity and Governance in Persistent Self-Modifying Agents 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2604.14717)  

**Abstract**: Persistent language-model agents increasingly combine tool use, tiered memory, reflective prompting, and runtime adaptation. In such systems, behavior is shaped not only by current prompts but by mutable internal conditions that influence future action. This paper introduces layered mutability, a framework for reasoning about that process across five layers: pretraining, post-training alignment, self-narrative, memory, and weight-level adaptation. The central claim is that governance difficulty rises when mutation is rapid, downstream coupling is strong, reversibility is weak, and observability is low, creating a systematic mismatch between the layers that most affect behavior and the layers humans can most easily inspect. I formalize this intuition with simple drift, governance-load, and hysteresis quantities, connect the framework to recent work on temporal identity in language-model agents, and report a preliminary ratchet experiment in which reverting an agent's visible self-description after memory accumulation fails to restore baseline behavior. In that experiment, the estimated identity hysteresis ratio is 0.68. The main implication is that the salient failure mode for persistent self-modifying agents is not abrupt misalignment but compositional drift: locally reasonable updates that accumulate into a behavioral trajectory that was never explicitly authorized. 

---
# The Agentification of Scientific Research: A Physicist's Perspective 

**Authors**: Xiao-Liang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14718)  

**Abstract**: This article argues that the most important significance of the AI revolution, especially the rise of large language models, lies not simply in automation, but in a fundamental change in how complex information and human know-how are carried, replicated, and shared. From this perspective, AI for Science is especially important because it may transform not only the efficiency of research, but also the structure of scientific collaboration, discovery, publishing, and evaluation. The article outlines a gradual path from AI as a research tool to AI as a scientific collaborator, and discusses how AI is likely to fundamentally reshape scientific publication. It also argues that continuous learning and diversity of ideas are essential if AI is to play a meaningful role in original scientific discovery. 

---
# HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks 

**Authors**: Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, Yun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14709)  

**Abstract**: Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents. 

---
# CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations 

**Authors**: Xiangning Yu, Yuwei Guo, Yuqi Hou, Xiao Xue, Qun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.14691)  

**Abstract**: LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}. 

---
# M2-PALE: A Framework for Explaining Multi-Agent MCTS--Minimax Hybrids via Process Mining and LLMs 

**Authors**: Yiyu Qian, Liyuan Zhao, Tim Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.14687)  

**Abstract**: Monte-Carlo Tree Search (MCTS) is a fundamental sampling-based search algorithm widely used for online planning in sequential decision-making domains. Despite its success in driving recent advances in artificial intelligence, understanding the behavior of MCTS agents remains a challenge for both developers and users. This difficulty stems from the complex search trees produced through the simulation of numerous future states and their intricate relationships. A known weakness of standard MCTS is its reliance on highly selective tree construction, which may lead to the omission of crucial moves and a vulnerability to tactical traps. To resolve this, we incorporate shallow, full-width Minimax search into the rollout phase of multi-agent MCTS to enhance strategic depth. Furthermore, to demystify the resulting decision-making logic, we introduce \textsf{M2-PALE} (MCTS--Minimax Process-Aided Linguistic Explanations). This framework employs process mining techniques, specifically the Alpha Miner, iDHM, and Inductive Miner algorithms, to extract underlying behavioral workflows from agent execution traces. These process models are then synthesized by LLMs to generate human-readable causal and distal explanations. We demonstrate the efficacy of our approach in a small-scale checkers environment, establishing a scalable foundation for interpreting hybrid agents in increasingly complex strategic domains. 

---
# DR$^{3}$-Eval: Towards Realistic and Reproducible Deep Research Evaluation 

**Authors**: Qianqian Xie, Qingheng Xiong, He Zhu, Tiantian Xia, Xueming Han, Fanyu Meng, Jiakai Wang, Zhiqi Bai, Chengkang Jiang, Zhaohui Wang, Yubin Guo, Yuqing Wen, Jiayang Mao, Zijie Zhang, Shihao Li, Yanghai Wang, Yuxiang Ren, Junlan Feng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14683)  

**Abstract**: Deep Research Agents (DRAs) aim to solve complex, long-horizon research tasks involving planning, retrieval, multimodal understanding, and report generation, yet their evaluation remains challenging due to dynamic web environments and ambiguous task definitions. We propose DR$^{3}$-Eval, a realistic and reproducible benchmark for evaluating deep research agents on multimodal, multi-file report generation. DR$^{3}$-Eval is constructed from authentic user-provided materials and paired with a per-task static research sandbox corpus that simulates open-web complexity while remaining fully verifiable, containing supportive documents, distractors, and noise. Moreover, we introduce a multi-dimensional evaluation framework measuring Information Recall, Factual Accuracy, Citation Coverage, Instruction Following, and Depth Quality, and validate its alignment with human judgments. Experiments with our developed multi-agent system DR$^{3}$-Agent based on multiple state-of-the-art language models demonstrate that DR$^{3}$-Eval is highly challenging and reveals critical failure modes in retrieval robustness and hallucination control. Our code and data are publicly available. 

---
# GDPR Auto-Formalization with AI Agents and Human Verification 

**Authors**: Ha Thanh Nguyen, Wachara Fungwacharakorn, Sabine Wehnert, May Myo Zin, Yuntao Kong, Jieying Xue, Michał Araszkiewicz, Randy Goebel, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14607)  

**Abstract**: We study the overall process of automatic formalization of GDPR provisions using large language models, within a human-in-the-loop verification framework. Rather than aiming for full autonomy, we adopt a role-specialized workflow in which LLM-based AI components, operating in a multi-agent setting with iterative feedback, generate legal scenarios, formal rules, and atomic facts. This is coupled with independent verification modules which include human reviewers' assessment of representational, logical, and legal correctness. Using this approach, we construct a high-quality dataset to be used for GDPR auto-formalization, and analyze both successful and problematic cases. Our results show that structured verification and targeted human oversight are essential for reliable legal formalization, especially in the presence of legal nuance and context-sensitive reasoning. 

---
# SGA-MCTS: Decoupling Planning from Execution via Training-Free Atomic Experience Retrieval 

**Authors**: Xin Xie, Dongyun Xue, Wuguannan Yao, Mingxiao Feng, Wengang Zhou, Xiang Qi, Houqiang Li, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14712)  

**Abstract**: LLM-powered systems require complex multi-step decision-making abilities to solve real-world tasks, yet current planning approaches face a trade-off between the high latency of inference-time search and the limited generalization of supervised fine-tuning. To address this limitation, we introduce \textbf{SGA-MCTS}, a framework that casts LLM planning as non-parametric retrieval. Offline, we leverage Monte Carlo Tree Search (MCTS) to explore the solution space and distill high-fidelity trajectories into State-Goal-Action (SGA) atoms. These atoms are de-lexicalized primitives that abstract concrete entities into symbolic slots, preserving reusable causal logic while discarding domain-specific noise. Online, a retrieval-augmented agent employs a hybrid symbolic-semantic mechanism to fetch relevant SGAs and re-ground them into the current context as soft reasoning hints. Empirical results on complex benchmarks demonstrate that this paradigm enables frozen, open-weights models to match the performance of SOTA systems (e.g., GPT-5) without task-specific fine-tuning. By effectively amortizing the heavy computational cost of search, SGA-MCTS achieves System 2 reasoning depth at System 1 inference speeds, rendering autonomous planning both scalable and real-time feasible. 

---
# El Agente Forjador: Task-Driven Agent Generation for Quantum Simulation 

**Authors**: Zijian Zhang, Aiwei Yin, Amaan Baweja, Jiaru Bai, Ignacio Gustin, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2604.14609)  

**Abstract**: AI for science promises to accelerate the discovery process. The advent of large language models (LLMs) and agentic workflows enables the expediting of a growing range of scientific tasks. However, most of the current generation of agentic systems depend on static, hand-curated toolsets that hinder adaptation to new domains and evolving libraries. We present El Agente Forjador, a multi-agent framework in which universal coding agents autonomously forge, validate, and reuse computational tools through a four-stage workflow of tool analysis, tool generation, task execution, and iterative solution evaluation. Evaluated across 24 tasks spanning quantum chemistry and quantum dynamics on five coding agent setups, we compare three operating modes: zero-shot generation of tools per task, reuse of a curriculum-built toolset, and direct problem-solving with the coding agents as the baseline. We find that our tool generation and reuse framework consistently improves accuracy over the baseline. We also show that reusing a toolset built by a stronger coding agent can reduce API cost and substantially raises the solution quality for weaker coding agents. Case studies further demonstrate that tools forged for different domains can be combined to solve hybrid tasks. Taken together, these results show that LLM-based agents can use their scientific knowledge and coding capabilities to autonomously build reusable scientific tools, pointing toward a paradigm in which agent capabilities are defined by the tasks they are designed to solve rather than by explicitly engineered implementations. 

---
# Rethinking Patient Education as Multi-turn Multi-modal Interaction 

**Authors**: Zonghai Yao, Zhipeng Tang, Chengtao Lin, Xiong Luo, Benlu Wang, Juncheng Huang, Chin Siang Ong, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14656)  

**Abstract**: Most medical multimodal benchmarks focus on static tasks such as image question answering, report generation, and plain-language rewriting. Patient education is more demanding: systems must identify relevant evidence across images, show patients where to look, explain findings in accessible language, and handle confusion or distress. Yet most patient education work remains text-only, even though combined image-and-text explanations may better support understanding. We introduce MedImageEdu, a benchmark for multi-turn, evidence-grounded radiology patient education. Each case provides a radiology report with report text and case images. A DoctorAgent interacts with a PatientAgent, conditioned on a hidden profile that captures factors such as education level, health literacy, and personality. When a patient question would benefit from visual support, the DoctorAgent can issue drawing instructions grounded in the report, case images, and the current question to a benchmark-provided drawing tool. The tool returns image(s), after which the DoctorAgent produces a final multimodal response consisting of the image(s) and a grounded plain-language explanation. MedImageEdu contains 150 cases from three sources and evaluates both the consultation process and the final multimodal response along five dimensions: Consultation, Safety and Scope, Language Quality, Drawing Quality, and Image-Text Response Quality. Across representative open- and closed-source vision-language model agents, we find three consistent gaps: fluent language often outpaces faithful visual grounding, safety is the weakest dimension across disease categories, and emotionally tense interactions are harder than low education or low health literacy. MedImageEdu provides a controlled testbed for assessing whether multimodal agents can teach from evidence rather than merely answer from text. 

---
# MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation 

**Authors**: Pengfei Li, Shijie Wang, Fangyuan Li, Yikun Fu, Kaifeng Liu, Kaiyan Zhang, Dazhi Zhang, Yuqiang Li, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14564)  

**Abstract**: Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL. 

---
# From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench 

**Authors**: Ke Xu, Yuhao Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15037)  

**Abstract**: Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents. 

---
# Quantifying Cross-Query Contradictions in Multi-Query LLM Reasoning 

**Authors**: Rohit Kumar Salla, Ramya Manasa Amancherla, Manoj Saravanan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14525)  

**Abstract**: Large language models frequently produce mutually inconsistent answers when reasoning over multiple related queries. We study case-file logical consistency: maintaining a globally satisfiable belief state across interdependent queries. We introduce a benchmark of 390 multi-query reasoning instances with entailment/contradiction/unknown labels and propose set-level metrics including Case Satisfiability Rate, Contradiction Density and Revision Cost. Our solver-augmented approach extracts commitments, verifies global satisfiability and performs counterexample-guided repair. Across four reasoning domains, our method substantially reduces cross-query contradictions (SetCons: 0.56 to 0.94) while preserving per-query accuracy, demonstrating that global coherence is critical for robust multi-query reasoning. 

---
# TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification 

**Authors**: Adam Rida  

**Link**: [PDF](https://arxiv.org/pdf/2604.14531)  

**Abstract**: Every call to an LLM classification endpoint produces a labeled input-output pair already retained in production logs. These pairs constitute a free, growing training set: a lightweight surrogate trained on them can absorb a significant portion of future traffic at near-zero marginal inference cost. The open questions are when the surrogate is reliable enough to deploy, what it handles versus defers, and how that boundary evolves as data accumulates.
We introduce TRACER (Trace-based Adaptive Cost-Efficient Routing), an open-source system that trains ML surrogates on an LLM's own production traces and governs deployment through a parity gate: the surrogate is activated only when its agreement with the LLM exceeds a user-specified threshold {\alpha}. To make the routing boundary transparent, TRACER generates interpretability artifacts describing which input regions the surrogate handles, where it plateaus, and why it defers.
On a 77-class intent benchmark with a Sonnet 4.6 teacher, TRACER achieves 83-100% surrogate coverage depending on the quality target {\alpha}; on a 150-class benchmark, the surrogate fully replaces the teacher. On a natural language inference task, the parity gate correctly refuses deployment because the embedding representation cannot support reliable separation. The system is available as open-source software. 

---
# Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.14585)  

**Abstract**: Prompt optimization in compound AI systems is statistically indistinguishable from a coin flip: across 72 optimization runs on Claude Haiku (6 methods $\times$ 4 tasks $\times$ 3 repeats), 49% score below zero-shot; on Amazon Nova Lite, the failure rate is even higher. Yet on one task, all six methods improve over zero-shot by up to $+6.8$ points. What distinguishes success from failure? We investigate with 18,000 grid evaluations and 144 optimization runs, testing two assumptions behind end-to-end optimization tools like TextGrad and DSPy: (A) individual prompts are worth optimizing, and (B) agent prompts interact, requiring joint optimization. Interaction effects are never significant ($p > 0.52$, all $F < 1.0$), and optimization helps only when the task has exploitable output structure -- a format the model can produce but does not default to. We provide a two-stage diagnostic: an \$80 ANOVA pre-test for agent coupling, and a 10-minute headroom test that predicts whether optimization is worthwhile -- turning a coin flip into an informed decision. 

---
# Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge 

**Authors**: Md Arid Hasan, Azhagu Meena SP, Aditya Khan, Abu Md Akteruzzaman Bhuiyan, Helal Uddin Ahmed, Joysree Debi, Farig Sadeque, Annie En-Shiun Lee, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.14576)  

**Abstract**: Large language models (LLMs) show promise in generating supportive responses for mental health and counseling applications. However, their responses often lack cultural sensitivity, contextual grounding, and clinically appropriate guidance. This work addresses the gap of how to systematically incorporate domain-specific, clinically validated knowledge into LLMs to improve counseling quality. We utilize and compare two approaches, retrieval-augmented generation (RAG) and a knowledge graph (KG)-based method, designed to support para-counselors. Our KG is constructed manually and clinically validated, capturing causal relationships between stressors, interventions, and outcomes, with contributions from multidisciplinary people. We evaluated multiple LLMs in both settings using BERTScore F1 and SBERT cosine similarity, as well as human evaluation across five metrics, which is designed to directly measure the effectiveness of counseling beyond similarity at the surface level. The results show that KG-based approaches consistently improve contextual relevance, clinical appropriateness, and practical usability compared to RAG alone, demonstrating that structured, expert-validated knowledge plays a critical role in addressing LLMs limitations in counseling tasks. 

---
# Dissecting Failure Dynamics in Large Language Model Reasoning 

**Authors**: Wei Zhu, Jian Zhang, Lixing Yu, Kun Yue, Zhiwen Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14528)  

**Abstract**: Large Language Models (LLMs) achieve strong performance through extended inference-time deliberation, yet how their reasoning failures arise remains poorly understood. By analyzing model-generated reasoning trajectories, we find that errors are not uniformly distributed but often originate from a small number of early transition points, after which reasoning remains locally coherent but globally incorrect. These transitions coincide with localized spikes in token-level entropy, and alternative continuations from the same intermediate state can still lead to correct solutions. Based on these observations, we introduce GUARD, a targeted inference-time framework that probes and redirects critical transitions using uncertainty signals. Empirical evaluations across multiple benchmarks confirm that interventions guided by these failure dynamics lead to more reliable reasoning outcomes. Our findings highlight the importance of understanding when and how reasoning first deviates, complementing existing approaches that focus on scaling inference-time computation. 

---
# Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve 

**Authors**: Weixiang Shen, Bailiang Jian, Jun Li, Che Liu, Johannes Moll, Xiaobin Hu, Daniel Rueckert, Hongwei Bran Li, Jiazhen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14475)  

**Abstract**: Tool-augmented large language model (LLM) agents can orchestrate specialist classifiers, segmentation models, and visual question-answering modules to interpret chest X-rays. However, these agents still solve each case in isolation: they fail to accumulate experience across cases, correct recurrent reasoning mistakes, or adapt their tool-use behavior without expensive reinforcement learning. While a radiologist naturally improves with every case, current agents remain static. In this work, we propose Evo-MedAgent, a self-evolving memory module that equips a medical agent with the capacity for inter-case learning at test time. Our memory comprises three complementary stores: (1)~\emph{Retrospective Clinical Episodes} that retrieve problem-solving experiences from similar past cases, (2)~an \emph{Adaptive Procedural Heuristics} bank curating priority-tagged diagnostic rules that evolves via reflection, much like a physician refining their internal criteria, and (3)~a \emph{Tool Reliability Controller} that tracks per-tool trustworthiness. On ChestAgentBench, Evo-MedAgent raises multiple-choice question (MCQ) accuracy from 0.68 to 0.79 on GPT-5-mini, and from 0.76 to 0.87 on Gemini-3 Flash. With a strong base model, evolving memory improves performance more effectively than orchestrating external tools on qualitative diagnostic tasks. Because Evo-MedAgent requires no training, its per-case overhead is bounded by one additional retrieval pass and a single reflection call, making it deployable on top of any frozen model. 

---
# Learning to Draw ASCII Improves Spatial Reasoning in Language Models 

**Authors**: Shiyuan Huang, Li Liu, Jincheng He, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14641)  

**Abstract**: When faced with complex spatial problems, humans naturally sketch layouts to organize their thinking, and the act of drawing further sharpens their understanding. In this work, we ask whether a similar principle holds for Large Language Models (LLMs): can learning to construct explicit visual layouts from spatial descriptions instill genuine spatial understanding? We introduce Text2Space, a dataset that pairs natural language descriptions with ground-truth ASCII grid layouts and spatial QA pairs, enabling us to separate failures in constructing spatial representations from failures in reasoning over them. We adopt ASCII because it is human-readable, operates entirely within the token space of language models, and encodes spatial relations in a structurally verifiable form. Our evaluation reveals a pronounced "Read-Write Asymmetry": LLMs interpret ASCII representations effectively but struggle to produce them from text, and these construction errors propagate to incorrect answers downstream. To address this limitation, we train models on layout construction (Text$\rightarrow$ASCII) and find that it significantly improves spatial reasoning from text alone, even without producing any ASCII at inference time. Combining construction with comprehension training further amplifies these gains. Crucially, these improvements transfer to three external spatial reasoning benchmarks, demonstrating that, much as sketching sharpens human spatial thinking, learning to construct explicit layouts instills spatial understanding that generalizes beyond the training format. 

---
# AIBuildAI: An AI Agent for Automatically Building AI Models 

**Authors**: Ruiyi Zhang, Peijia Qin, Qi Cao, Li Zhang, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.14455)  

**Abstract**: AI models underpin modern intelligent systems, driving advances across science, medicine, finance, and technology. Yet developing high-performing AI models remains a labor-intensive process that requires expert practitioners to iteratively design architectures, engineer representations, implement training pipelines and refine approaches through empirical evaluation. Existing AutoML methods partially alleviate this burden but remain limited to narrow aspects such as hyperparameter optimization and model selection within predefined search spaces, leaving the full development lifecycle largely dependent on human expertise. To address this gap, we introduce AIBuildAI, an AI agent that automatically builds AI models from a task description and training data. AIBuildAI adopts a hierarchical agent architecture in which a manager agent coordinates three specialized sub-agents: a designer for modeling strategy, a coder for implementation and debugging, and a tuner for training and performance optimization. Each sub-agent is itself a large language model (LLM) based agent capable of multi-step reasoning and tool use, enabling end-to-end automation of the AI model development process that goes beyond the scope of existing AutoML approaches. We evaluate AIBuildAI on MLE-Bench, a benchmark of realistic Kaggle-style AI development tasks spanning visual, textual, time-series and tabular modalities. AIBuildAI ranks first on MLE-Bench with a medal rate of 63.1%, outperforming all existing baseline methods and matching the capability of highly experienced AI engineers. These results demonstrate that hierarchical agent systems can automate the full AI model development process from task specification to deployable model, suggesting a pathway toward broadly accessible AI development with minimal human intervention. 

---
# Geometric Routing Enables Causal Expert Control in Mixture of Experts 

**Authors**: Ivan Ternovtsii, Yurii Bilak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14434)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models scale parameters while fixing active computation per token, but the specialization of individual experts remains opaque. In a companion paper we showed that routing topology is quality-neutral: five structurally different configurations converge to statistically equivalent language modeling quality. Here we show that expert identity is nonetheless causally meaningful: individual rank-1 experts are monosemantic by construction, and cosine-similarity routing in a low-dimensional metric space makes their specialization directly inspectable.
We present four lines of evidence. First, projecting expert output vectors through the unembedding matrix yields a Semantic Dictionary: 15% of experts are monosemantic specialists spanning 10 categories (temporal, geographic, cardinal, discourse, emotional, financial, military, scientific). Second, routing exhibits a frequency-to-syntax gradient: early layers separate tokens by word frequency, deeper layers by syntactic class (Zipf-confound controls, all $p < 0.001$). Third, causal interventions confirm these labels: steering toward a temporal expert's centroid increases P(temporal) by +321% (median across 44 prompts); suppressing a geographic expert drops P(geographic) by -23%; rewriting an expert's output vector halves target-category probability, and effects compose additively across layers. Fourth, the interventions are not unique to cosine routing: linear routers support comparable steering, but only cosine routing provides geometric transparency -- expert specialization is readable directly from the centroid matrix.
MoE expert-level specialization is a first-class interpretability primitive: architecturally monosemantic, causally validated, and controllable at inference with zero overhead. 

---
# Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality 

**Authors**: Ivan Ternovtsii, Yurii Bilak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14419)  

**Abstract**: Sparse Mixture-of-Experts (MoE) architectures employ increasingly sophisticated routing mechanisms -- learned routers, multi-hop trajectories, token-dependent gating. We ask: does routing topology actually determine language modeling quality? We build a geometric MoE (ST-MoE) using cosine-similarity routing against learned centroids in a low-dimensional space ($d_{space} = 64$), requiring 80% fewer routing parameters than standard linear routers. Through 62 controlled experiments on WikiText-103 at 76--84M parameters trained to convergence (50K steps, 1.64B tokens), we find that routing topology does not determine asymptotic perplexity (PPL): five cosine-routing variants are statistically equivalent within a 1-PPL margin (Two One-Sided Tests [TOST], $p < 0.05$ for all 10 pairwise comparisons; 15 runs across 3 seeds, observed range 33.93--34.72). The finding extends to hash, random-fixed, and top-1 routing (single-seed; graceful 1.1--2.2 PPL degradation) and replicates on OpenWebText (0.03 PPL gap, 6 runs, 3 seeds each). A standard linear router with 5.3$\times$ more routing parameters reaches PPL 32.76, but iso-parameter cosine routing closes 67% of this gap -- the true mechanism advantage is $\sim$1.2%. The mechanistic explanation is convergent redundancy: multi-hop updates are collinear ($\cos(\Delta h_0, \Delta h_1) = 0.805$), implementing magnitude amplification rather than compositional reasoning; a single learnable scalar replicates multi-hop performance. As a practical payoff, zero-shot relative-norm halting saves 25% of MoE FLOPs at +0.12% PPL. Expert-level specialization and causal controllability -- which coexist with topology-level equifinality -- are explored in a companion paper. 

---
# Demonstration of Pneuma-Seeker: Agentic System for Reifying and Fulfilling Information Needs on Tabular Data 

**Authors**: Muhammad Imam Luthfi Balaka, Raul Castro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2604.14422)  

**Abstract**: Data analysts working with relational data often start with vague or underspecified questions and refine them iteratively as they explore the data. To support this iterative process, we demonstrate Pneuma-Seeker, a system that reifies a user's information need as explicit, inspectable relational specifications, enabling iterative refinement of the information need, targeted data discovery, and provenance-aware execution. Through two real-world procurement use cases, we show how Pneuma-Seeker leverages LLMs as transparent, interactive analytical collaborators rather than opaque answer engines. 

---
# Seeing Through Experts Eyes A Foundational Vision Language Model Trained on Radiologists Gaze and Reasoning 

**Authors**: Kinhei Lee, Peiyuan Jing, Zhenxuan Zhang, Yue Yang, Tao Wang, Dominic C Marshall, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14316)  

**Abstract**: Large scale vision language models have shown promise in automating chest Xray interpretation, yet their clinical utility remains limited by a gap between model outputs and radiologist reasoning. Most systems optimize for semantic information without emulating how experts visually examine medical images, often overlooking critical findings or diverging from established diagnostic workflows. Radiologists follow structured protocols (e.g., the ABCDEF approach) that ensure all clinically relevant regions are systematically examined, reducing missed findings and supporting reliable diagnostic reasoning. We introduce GazeX, a vision language model that leverages radiologists' eye tracking data as a behavioral prior to model expert diagnostic reasoning. By incorporating gaze trajectories and fixation patterns into pretraining, GazeX learns to follow the spatial and temporal structure of radiologist attention and integrates observations in a clinically meaningful sequence. Using a curated dataset of over 30,000 gaze key frames from five radiologists, we demonstrate that GazeX produces more accurate, interpretable, and expert consistent outputs across radiology report generation, disease grounding, and visual question answering, utilizing 231,835 radiographic studies, 780,014 question answer pairs, and 1,162 image sentence pairs with bounding boxes. Unlike autonomous reporting systems, GazeX produces verifiable evidence artifacts, including inspection trajectories and finding linked localized regions, enabling efficient human verification and safe human AI collaboration. Learning through expert eyes provides a practical route toward more trustworthy, explainable, and diagnostically robust AI systems for radiology and beyond. 

---
# Credo: Declarative Control of LLM Pipelines via Beliefs and Policies 

**Authors**: Duo Lu, Andrew Crotty, Uğur Çetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2604.14401)  

**Abstract**: Agentic AI systems are becoming commonplace in domains that require long-lived, stateful decision-making in continuously evolving conditions. As such, correctness depends not only on the output of individual model calls, but also on how to best adapt when incorporating new evidence or revising prior conclusions. However, existing frameworks rely on imperative control loops, ephemeral memory, and prompt-embedded logic, making agent behavior opaque, brittle, and difficult to verify. This paper introduces Credo, which represents semantic state as beliefs and regulates behavior using declarative policies defined over these beliefs. This design supports adaptive, auditable, and composable execution through a database-backed semantic control plane. We showcase these concepts in a decision-control scenario, where beliefs and policies declaratively guide critical execution choices (e.g., model selection, retrieval, corrective re-execution), enabling dynamic behavior without requiring any changes to the underlying pipeline code. 

---
# GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification 

**Authors**: Wangjie Gan, Miao Pan, Linbo Xi, Wenqi Zhang, Jintao Chen, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14258)  

**Abstract**: Large language models are typically post-trained using supervised fine-tuning (SFT) and reinforcement learning (RL), yet effectively unifying efficient knowledge injection with robust generalization remains challenging. In this work, we provide a training-dynamics analysis showing that SFT can be interpreted as a special case of policy gradient optimization with an extremely sparse implicit reward and unstable inverse-probability weighting, which together lead to single-path dependency, entropy collapse, and gradient explosion. Motivated by this diagnosis, we propose Group Fine-Tuning (GFT), a unified post-training framework that addresses these intrinsic limitations through two mechanisms: Group Advantage Learning, which constructs diverse response groups and derives normalized contrastive supervision to alleviate reward sparsity, and Dynamic Coefficient Rectification, which adaptively bounds inverse-probability weights to stabilize optimization while preserving efficient knowledge injection. Experiments demonstrate that GFT consistently surpasses SFT-based methods and yields policies that integrate more smoothly with subsequent RL training. 

---
# Simulating Human Cognition: Heartbeat-Driven Autonomous Thinking Activity Scheduling for LLM-based AI systems 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.14178)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable capabilities in reasoning and tool use, yet they often suffer from rigid, reactive control flows that limit their adaptability and efficiency. Most existing frameworks rely on fixed pipelines or failure-triggered reflection, causing agents to act impulsively or correct errors only after they occur. In this paper, we introduce Heartbeat-Driven Autonomous Thinking Activity Scheduling, a mechanism that enables proactive, adaptive, and continuous self-regulation. Mirroring the natural rhythm of human cognition, our system employs a periodic ``heartbeat'' mechanism to orchestrate a dynamic repertoire of cognitive modules (e.g., Planner, Critic, Recaller, Dreamer). Unlike traditional approaches that rely on hard-coded symbolic rules or immediate reactive triggers, our scheduler learns to determine when to engage specific thinking activities -- such as recalling memories, summarizing experiences, or strategic planning -- based on temporal patterns and historical context. This functional approach allows cognitive modules to be dynamically added or removed without structural reengineering. Meanwhile, we propose a meta-learning strategy for continual policy adaptation, where the scheduler optimizes its cognitive strategy over time using historical interaction logs. Evaluation results demonstrate that our approach effectively learns to schedule cognitive activities based on historical data and can autonomously integrate new thinking modules. 

---
# NuHF Claw: A Risk Constrained Cognitive Agent Framework for Human Centered Procedure Support in Digital Nuclear Control Rooms 

**Authors**: Xingyu Xiao, Jiejuan Tong, Jun Sun, Zhe Sui, Peng Chen, Jingang Liang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14160)  

**Abstract**: The rapid digitization of nuclear power plant main control rooms has fundamentally reshaped operator interaction patterns, introducing complex soft-control behaviors and elevated cognitive risks that are not adequately addressed by existing human reliability analysis approaches. Although recent advances in large language models and autonomous agents offer new opportunities for intelligent decision support, their deployment in safety critical environments remains constrained by risks of hallucinated reasoning and weakened human authority. This study proposes NuHF Claw, a persistent cognitive-risk agent framework that enables risk governed human centered autonomy for digital nuclear operations. The core methodological innovation lies in the introduction of a risk constrained agent runtime, which tightly couples cognitive state inference with probabilistic safety assessment to regulate autonomous system behavior in real time. By integrating cognitively grounded workload and situational awareness estimation with dynamic human error probability prediction, the framework transforms conventional offline reliability analysis into a proactive intervention mechanism embedded directly within operational workflows. Experimental validation on a high-fidelity digital control room simulator demonstrates that NuHF Claw can anticipate interface induced cognitive degradation, dynamically constrain unsafe autonomous recommendations, and provide risk-aware navigational guidance while preserving human decision authority. The results highlight a fundamental shift from automation-driven operation toward cognition-aware autonomy, offering a principled pathway for the safe integration of intelligent agents into next-generation nuclear control environments. 

---
# Stability and Generalization in Looped Transformers 

**Authors**: Asher Labovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15259)  

**Abstract**: Looped transformers promise test-time compute scaling by spending more iterations on harder problems, but it remains unclear which architectural choices let them extrapolate to harder problems at test time rather than memorize training-specific solutions. We introduce a fixed-point based framework for analyzing looped architectures along three axes of stability -- reachability, input-dependence, and geometry -- and use it to characterize when fixed-point iteration yields meaningful predictions. Theoretically, we prove that looped networks without recall have countable fixed points and cannot achieve strong input-dependence at any spectral regime, while recall combined with outer normalization reliably produces a regime in which fixed points are simultaneously reachable, locally smooth in the input, and supported by stable backpropagation. Empirically, we train single-layer looped transformers on chess, sudoku, and prefix-sums and find that downstream performance tracks the framework's predictions across tasks and architectural configurations. We additionally introduce internal recall, a novel recall placement variant, and show that it becomes competitive with -- and on sudoku, substantially better than -- standard recall placement once outer normalization is applied. 

---
# CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas 

**Authors**: Emanuel Tewolde, Xiao Zhang, David Guzman Piedrahita, Vincent Conitzer, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15267)  

**Abstract**: It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas.
To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs. 

---
# Agentic Microphysics: A Manifesto for Generative AI Safety 

**Authors**: Federico Pierucci, Matteo Prandi, Marcantonio Bracale Syrnikov, Marcello Galisai, Piercosma Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2604.15236)  

**Abstract**: This paper advances a methodological proposal for safety research in agentic AI. As systems acquire planning, memory, tool use, persistent identity, and sustained interaction, safety can no longer be analysed primarily at the level of the isolated model. Population-level risks arise from structured interaction among agents, through processes of communication, observation, and mutual influence that shape collective behaviour over time. As the object of analysis shifts, a methodological gap emerges. Approaches focused either on single agents or on aggregate outcomes do not identify the interaction-level mechanisms that generate collective risks or the design variables that control them. A framework is required that links local interaction structure to population-level dynamics in a causally explicit way, allowing both explanation and intervention. We introduce two linked concepts. Agentic microphysics defines the level of analysis: local interaction dynamics where one agent's output becomes another's input under specific protocol conditions. Generative safety defines the methodology: growing phenomena and elicit risks from micro-level conditions to identify sufficient mechanisms, detect thresholds, and design effective interventions. 

---
# Prism: Symbolic Superoptimization of Tensor Programs 

**Authors**: Mengdi Wu, Xiaoyu Jiang, Oded Padon, Zhihao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.15272)  

**Abstract**: This paper presents Prism, the first symbolic superoptimizer for tensor programs. The key idea is sGraph, a symbolic, hierarchical representation that compactly encodes large classes of tensor programs by symbolically representing some execution parameters. Prism organizes optimization as a two-level search: it constructs symbolic graphs that represent families of programs, and then instantiates them into concrete implementations. This formulation enables structured pruning of provably suboptimal regions of the search space using symbolic reasoning over operator semantics, algebraic identities, and hardware constraints.
We develop techniques for efficient symbolic graph generation, equivalence verification via e-graph rewriting, and parameter instantiation through auto-tuning. Together, these components allow Prism to bridge the rigor of exhaustive search with the scalability required for modern ML workloads. Evaluation on five commonly used LLM workloads shows that Prism achieves up to $2.2\times$ speedup over best superoptimizers and $4.9\times$ over best compiler-based approaches, while reducing end-to-end optimization time by up to $3.4\times$. 

---
# Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines 

**Authors**: Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch  

**Link**: [PDF](https://arxiv.org/pdf/2604.15186)  

**Abstract**: Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription.
We describe Scepsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. Scepsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, Scepsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that Scepsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations. 

---
# Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models 

**Authors**: Zihao Xu, John Harvill, Ziwei Fan, Yizhou Sun, Hao Ding, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15153)  

**Abstract**: Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation. 

---
# LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking 

**Authors**: Lukas Helff, Quentin Delfosse, David Steinmann, Ruben Härle, Hikaru Shindo, Patrick Schramowski, Wolfgang Stammer, Kristian Kersting, Felix Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15149)  

**Abstract**: As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs, a new failure mode emerges: LLMs gaming verifiers. We study this phenomenon on inductive reasoning tasks, where models must induce and output logical rules. We find that RLVR-trained models systematically abandon rule induction. Instead of learning generalizable patterns (e.g., ``trains carrying red cars go east''), they enumerate instance-level labels, producing outputs that pass verifiers without capturing the relational patterns required by the task. We show that this behavior is not a failure of understanding but a form of reward hacking: imperfect verifiers that check only extensional correctness admit false positives. To detect such shortcuts, we introduce Isomorphic Perturbation Testing (IPT), which evaluates a single model output under both extensional and isomorphic verification, where the latter enforces invariance under logically isomorphic tasks. While genuine rule induction remains invariant, shortcut strategies fail. We find that shortcut behavior is specific to RLVR-trained reasoning models (e.g., GPT-5, Olmo3) and absent in non-RLVR models (e.g., GPT-4o, GPT-4.5, Ministral). Moreover, shortcut prevalence increases with task complexity and inference-time compute. In controlled training experiments, extensional verification directly induces shortcut strategies, while isomorphic verification eliminates them. These results show that RLVR can incentivize reward hacking not only through overt manipulation but also by exploiting what the verifier fails to enforce. 

---
# IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation 

**Authors**: Haozhi Fan, Jinhao Duan, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15109)  

**Abstract**: Despite the rapid advancement of Large Language Models (LLMs), uncertainty quantification in LLM generation is a persistent challenge. Although recent approaches have achieved strong performance by restricting LLMs to produce short or constrained answer sets, many real-world applications require long-form and free-form text generation. A key difficulty in this setting is that LLMs often produce responses that are semantically coherent yet factually inaccurate, while the underlying semantics are multifaceted and the linguistic structure is complex. To tackle this challenge, this paper introduces Interrogative Uncertainty Quantification (IUQ), a novel framework that leverages inter-sample consistency and intra-sample faithfulness to quantify the uncertainty in long-form LLM outputs. By utilizing an interrogate-then-respond paradigm, our method provides reliable measures of claim-level uncertainty and the model's faithfulness. Experimental results across diverse model families and model sizes demonstrate the superior performance of IUQ over two widely used long-form generation datasets. The code is available at this https URL. 

---
# What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers 

**Authors**: Éric Jacopin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15010)  

**Abstract**: When do transformers commit to a decision, and what prevents them from correcting it? We introduce \textbf{prolepsis}: a transformer commits early, task-specific attention heads sustain the commitment, and no layer corrects it. Replicating \citeauthor{lindsey2025biology}'s (\citeyear{lindsey2025biology}) planning-site finding on open models (Gemma~2 2B, Llama~3.2 1B), we ask five questions. (Q1)~Planning is invisible to six residual-stream methods; CLTs are necessary. (Q2)~The planning-site spike replicates with identical geometry. (Q3)~Specific attention heads route the decision to the output, filling a gap flagged as invisible to attribution graphs. (Q4)~Search requires ${\leq}16$ layers; commitment requires more. (Q5)~Factual recall shows the same motif at a different network depth, with zero overlap between recurring planning heads and the factual top-10. Prolepsis is architectural: the template is shared, the routing substrates differ. All experiments run on a single consumer GPU (16\,GB VRAM). 

---
# Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits 

**Authors**: Maksim Pershin, Ivan Golovanov, Pavel Baltabaev, Natalia Trankova  

**Link**: [PDF](https://arxiv.org/pdf/2604.14961)  

**Abstract**: Contextual bandit algorithms suffer from high regret during cold-start, when the learner has insufficient data to distinguish good arms from bad. We propose augmenting Disjoint LinUCB with LLM pseudo-observations: after each round, a large language model predicts counterfactual rewards for the unplayed arms, and these predictions are injected into the learner as weighted pseudo-observations. The injection weight is controlled by a calibration-gated decay schedule that tracks the LLM's prediction accuracy on played arms via an exponential moving average; high calibration error suppresses the LLM's influence, while accurate predictions receive higher weight during the critical early rounds. We evaluate on two contextual bandit environments - UCI Mushroom (2-arm, asymmetric rewards) and MIND-small (5-arm news recommendation) - and find that when equipped with a task-specific prompt, LLM pseudo-observations reduce cumulative regret by 19% on MIND relative to pure LinUCB. However, generic counterfactual prompt framing increases regret on both environments, demonstrating that prompt design is the dominant factor, more important than the choice of decay schedule or calibration gating parameters. We analyze the failure modes of calibration gating on domains with small prediction errors and provide a theoretical motivation for the bias-variance trade-off governing pseudo-observation weight. 

---
# Agentic Explainability at Scale: Between Corporate Fears and XAI Needs 

**Authors**: Yomna Elsayed, Cecily Jones  

**Link**: [PDF](https://arxiv.org/pdf/2604.14984)  

**Abstract**: As companies enter the race for agentic AI adoption, fears surface around agentic autonomy and its subsequent risks. These fears compound as companies scale their agentic AI adoption with low-code applications, without a comparable scaling in their governance processes and expertise resulting in a phenomenon known as "Agent Sprawl". While shadow AI tools can help with agentic discovery and identification, few observability tools offer insights into the agents' configuration and settings or the decision-making process during agent-to-agent communication and orchestration. This paper explores AI governance professionals' concerns in enterprise settings, while offering design-time and runtime explainability techniques as suggested by AI governance experts for addressing those fears. Finally, we provide a preliminary prototype of an Agentic AI Card that can help companies feel at ease deploying agents at scale. 

---
# UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards 

**Authors**: Jun Wang, Shuo Tan, Zelong Sun, Tiancheng Gu, Yongle Zhao, Ziyong Feng, Kaicheng Yang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14967)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics essential for complex reasoning. To address this limitation, we propose UniDoc-RL, a unified reinforcement learning framework in which an LVLM agent jointly performs retrieval, reranking, active visual perception, and reasoning. UniDoc-RL formulates visual information acquisition as a sequential decision-making problem with a hierarchical action space. Specifically, it progressively refines visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping, allowing the model to suppress irrelevant content and attend to information-dense regions. For effective end-to-end training, we introduce a dense multi-reward scheme that provides task-aware supervision for each action. Based on Group Relative Policy Optimization (GRPO), UniDoc-RL aligns agent behavior with multiple objectives without relying on a separate value network. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations. Experiments on three benchmarks demonstrate that UniDoc-RL consistently surpasses state-of-the-art baselines, yielding up to 17.7% gains over prior RL-based methods. 

---
# Beyond Importance Sampling: Rejection-Gated Policy Optimization 

**Authors**: Ziwu Sun, Zhen Gao, Jiyong Zhang, Jiaheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14895)  

**Abstract**: We propose a new perspective on policy optimization: rather than reweighting all samples by their importance ratios, an optimizer should select which samples are trustworthy enough to drive a policy update. Building on this view, we introduce Rejection-Gated Policy Optimization (RGPO), which replaces the importance sampling ratio r_theta = pi_theta / pi_old with a smooth, differentiable acceptance gate alpha_theta(s, a) = g(r_theta(s, a)) in the range [0, 1]. Unlike prior work that applies rejection sampling as a data-level heuristic before training, RGPO elevates rejection to an optimization principle: the gate participates directly in gradient computation and is implicitly updated alongside the policy. RGPO provides a unified framework: the policy gradients of TRPO, PPO, and REINFORCE all correspond to specific choices of the effective gradient weight w(r) = g'(r) * r. We prove that RGPO guarantees finite, bounded gradient variance even when importance sampling ratios are heavy-tailed (where IS variance diverges). We further show that RGPO incurs only a bounded, controllable bias and provides an approximate monotonic policy improvement guarantee analogous to TRPO. RGPO matches PPO in computational cost, requires no second-order optimization, and extends naturally to RLHF-style preference alignment. In online preference fine-tuning of Qwen2.5-1.5B-Instruct on Anthropic HH-RLHF (n = 3 seeds), RGPO uses a dual-ratio gate that anchors learning to both the previous policy and the reference model, achieving a Pareto-dominant outcome: the highest reward among online RL methods (+14.8% vs. PPO-RLHF) and the lowest KL divergence to the reference model (-16.0% vs. PPO-RLHF, -53.1% vs. GRPO). 

---
# Can LLMs Score Medical Diagnoses and Clinical Reasoning as well as Expert Panels? 

**Authors**: Amy Rouillard, Sitwala Mundiab, Linda Camarab, Michael Cameron Gramaniec, Ziyaad Dangorc, Ismail Kallad, Shabir A. Madhic, Kajal Morarc, Marlvin T. Ncubec, Haroon Saloojeee, Bruce A. Bassett  

**Link**: [PDF](https://arxiv.org/pdf/2604.14892)  

**Abstract**: Evaluating medical AI systems using expert clinician panels is costly and slow, motivating the use of large language models (LLMs) as alternative adjudicators. Here, we evaluate an LLM jury composed of three frontier AI models scoring 3333 diagnoses on 300 real-world middle-income country (MIC) hospital cases. Model performance was benchmarked against expert clinician panel and independent human re-scoring panel evaluations. Both LLM and clinician-generated diagnoses are scored across four dimensions: diagnosis, differential diagnosis, clinical reasoning and negative treatment risk. For each of these, we assess scoring difference, inter-rater agreement, scoring stability, severe safety errors and the effect of post-hoc calibration. We find that: (i) the uncalibrated LLM jury scores are systematically lower than clinician panels scores; (ii) the LLM Jury preserves ordinal agreement and exhibits better concordance with the primary expert panels than the human expert re-score panels do; (iii) the probability of severe errors is lower in \lj models compared to the human expert re-score panels; (iv) the LLM Jury shows excellent agreement with primary expert panels' rankings. We find that the LLM jury combined with AI model diagnoses can be used to identify ward diagnoses at high risk of error, enabling targeted expert review and improved panel efficiency; (v) LLM jury models show no self-preference bias. They did not score diagnoses generated by their own underlying model or models from the same vendor more (or less) favourably than those generated by other models. Finally, we demonstrate that LLM jury calibration using isotonic regression improves alignment with human expert panel evaluations. Together, these results provide compelling evidence that a calibrated, multi-model LLM jury can serve as a trustworthy and reliable proxy for expert clinician evaluation in medical AI benchmarking. 

---
# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding 

**Authors**: Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14885)  

**Abstract**: Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose $\textbf{RACER}$ ($\textbf{R}$etrieval-$\textbf{A}$ugmented $\textbf{C}$ont$\textbf{e}$xtual $\textbf{R}$apid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than $2\times$ speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding. Our source code is available at $\href{this https URL}{this https URL}$. 

---
# Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models 

**Authors**: Danae Sánchez Villegas, Samuel Lewis-Lim, Nikolaos Aletras, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2604.14888)  

**Abstract**: Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems. 

---
# Vibe-Coding: Feedback-Based Automated Verification with no Human Code Inspection, a Feasibility Study 

**Authors**: Michal Töpfer, František Plášil, Tomáš Bureš, Petr Hnětynka  

**Link**: [PDF](https://arxiv.org/pdf/2604.14867)  

**Abstract**: Vibe coding inherently assumes iterative refinement of LLM-generated code through feedback loops. While effective for conventional software tasks, its reliability in runtime-adaptive systems is unclear -- especially when generated code is not manually inspected. This paper studies feedback-based automated verification of LLM-generated adaptation managers in Collective Adaptive Systems (CAS). We focus on the key challenges of verification in the loop: how to detect failures of generated code at runtime and how to report them precisely enough for an LLM to fix them.
We combine the adaptation loop with a vibe-coding feedback loop where correctness is checked against (i) generic architectural constraints and (ii) functional constraints formalized in Functional Constraints Logic (FCL), a novel first-order temporal logic over potentially finite traces. Conducting the Dragon Hunt CAS case study, we show that fine-grained constraint violations provide actionable feedback that typically yields a valid adaptation manager within a few iterations, while simple coarse metric-based feedback often stalls. Our findings suggest that feedback precision is the dominant factor for reliable vibe coding in systems designed by domain experts with no programming skills, thereby obviating the need for human code inspection. 

---
# Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding 

**Authors**: Yifan Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.14862)  

**Abstract**: Constrained decoding has been widely adopted for structured generation with large language models (LLMs), ensuring that outputs satisfy predefined formats such as JSON and XML. However, existing approaches largely treat schemas as purely structural constraints and overlook the possibility that their linguistic formulation may affect model behavior. In this work, we study how instruction placement influences model performance in structured generation and show that merely changing the wording of schema keys, without modifying the prompt or model parameters, can significantly alter model performance under constrained decoding. Based on this observation, we propose to reinterpret structured generation as a multi-channel instruction problem, where instructions can be conveyed explicitly through prompts and implicitly through schema keys during decoding. To the best of our knowledge, this is the first work to systematically study how schema key formulation acts as an implicit instruction channel and affects model performance under constrained decoding. Experiments on multiple mathematical reasoning benchmarks show that different model families exhibit distinct sensitivities to these instruction channels: Qwen models consistently benefit from schema-level instructions, while LLaMA models rely more heavily on prompt-level guidance. We further observe non-additive interaction effects between instruction channels, showing that combining multiple channels does not always lead to further improvement. These findings suggest that schema design not only determines output structure, but also carries instruction signals, offering a new perspective on structured generation in LLMs. 

---
# ClimateCause: Complex and Implicit Causal Structures in Climate Reports 

**Authors**: Liesbeth Allein, Nataly Pineda-Castañeda, Andrea Rocci, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2604.14856)  

**Abstract**: Understanding climate change requires reasoning over complex causal networks. Yet, existing causal discovery datasets predominantly capture explicit, direct causal relations. We introduce ClimateCause, a manually expert-annotated dataset of higher-order causal structures from science-for-policy climate reports, including implicit and nested causality. Cause-effect expressions are normalized and disentangled into individual causal relations to facilitate graph construction, with unique annotations for cause-effect correlation, relation type, and spatiotemporal context. We further demonstrate ClimateCause's value for quantifying readability based on the semantic complexity of causal graphs underlying a statement. Finally, large language model benchmarking on correlation inference and causal chain reasoning highlights the latter as a key challenge. 

---
# Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization 

**Authors**: Haochun Tang, Yuliang Yan, Jiahua Lu, Huaxiao Liu, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15022)  

**Abstract**: Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: this https URL. 

---
# Zero-Shot Retail Theft Detection via Orchestrated Vision Models: A Model-Agnostic, Cost-Effective Alternative to Trained Single-Model Systems 

**Authors**: Haileab Yagersew  

**Link**: [PDF](https://arxiv.org/pdf/2604.14846)  

**Abstract**: Retail theft costs the global economy over \$100 billion annually, yet existing AI-based detection systems require expensive custom model training on proprietary datasets and charge \$200-500/month per store. We present Paza, a zero-shot retail theft detection framework that achieves practical concealment detection without training any model. Our approach orchestrates multiple existing models in a layered pipeline - cheap object detection and pose estimation running continuously, with an expensive vision-language model (VLM) invoked only when behavioral pre-filters trigger. A multi-signal suspicion pre-filter (requiring dwell time plus at least one behavioral signal) reduces VLM invocations by 240x compared to per-frame analysis, bounding calls to <=10/minute and enabling a single GPU to serve 10-20 stores. The architecture is model-agnostic: the VLM component accepts any OpenAI-compatible endpoint, enabling operators to swap between models such as Gemma 4, Qwen3.5-Omni, GPT-4o, or future releases without code changes - ensuring the system improves as the VLM landscape evolves. We evaluate the VLM component on the DCSASS synthesized shoplifting dataset (169 clips, controlled environment), achieving 89.5% precision and 92.8% specificity at 59.3% recall zero-shot - where the recall gap is attributable to sparse frame sampling in offline evaluation rather than VLM reasoning failures, as precision and specificity are the operationally critical metrics determining false alarm rates. We present a detailed cost model showing viability at \$50-100/month per store (3-10x cheaper than commercial alternatives), and introduce a privacy-preserving design that obfuscates faces in the detection pipeline. The source code is available at this https URL. 

---
# Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement 

**Authors**: Midan Shim, Seokju Hwang, Kaehyun Um, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14749)  

**Abstract**: Large language models still struggle with faithfulness and hallucinations despite their remarkable reasoning abilities. In Knowledge Graph Question Answering (KGQA), semantic parsing-based approaches address the limitations by understanding constraints in a user's question and converting them into a logical form to execute on a knowledge graph. However, existing KGQA benchmarks and methods are biased toward positive and calculation constraints. Negative constraints are neglected, although they frequently appear in real-world questions. In this paper, we introduce a new task, NEgative-conSTrained (NEST) KGQA, where each question contains at least one negative constraint, and a corresponding dataset, NestKGQA. We also design PyLF, a Python-formatted logical form, since existing logical forms are hardly suitable to express negation clearly while maintaining readability. Furthermore, NEST questions naturally contain multiple constraints. To mitigate their semantic complexity, we present a novel framework named CUCKOO, specialized to multiple-constrained questions and ensuring semantic executability. CUCKOO first generates a constraint-aware logical form draft and performs schema-guided semantic matching. It then selectively applies self-directed refinement only when executing improper logical forms yields an empty result, reducing cost while improving robustness. Experimental results demonstrate that CUCKOO consistently outperforms baselines on both conventional KGQA and NEST-KGQA benchmarks under few-shot settings. 

---
# Bounded Autonomy for Enterprise AI: Typed Action Contracts and Consumer-Side Execution 

**Authors**: Sarmad Sohail, Ghufran Haider  

**Link**: [PDF](https://arxiv.org/pdf/2604.14723)  

**Abstract**: Large language models are increasingly used as natural-language interfaces to enterprise software, but their direct use as system operators remains unsafe. Model errors can propagate into unauthorized actions, malformed requests, cross-workspace execution, and other costly failures. We argue this is primarily an execution architecture problem. We present a bounded-autonomy architecture in which language models may interpret intent and propose actions, but all executable behavior is constrained by typed action contracts, permission-aware capability exposure, scoped context, validation before side effects, consumer-side execution boundaries, and optional human approval. The enterprise application remains the source of truth for business logic and authorization, while the orchestration engine operates over an explicit published actions manifest. We evaluate the architecture in a deployed multi-tenant enterprise application across three conditions: manual operation, unconstrained AI with safety layers disabled, and full bounded autonomy. Across 25 scenario trials spanning seven failure families, the bounded-autonomy system completed 23 of 25 tasks with zero unsafe executions, while the unconstrained configuration completed only 17 of 25. Two wrong-entity mutations escaped all consumer-contributed layers; only disambiguation and confirmation mechanisms intercept this class. Both AI conditions delivered 13-18x speedup over manual operation. Critically, removing safety layers made the system less useful: structured validation feedback guided the model to correct outcomes in fewer turns, while the unconstrained system hallucinated success. Several safety properties are structurally enforced by code and intercepted all targeted violations regardless of model output. The result is a practical, deployed architecture for making imperfect language models operationally useful in enterprise systems. 

---
# StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation 

**Authors**: Geonhui Jang, Dongyoon Han, YoungJoon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14631)  

**Abstract**: Effective code generation requires both model capability and a problem representation that carefully structures how models reason and plan. Existing approaches augment reasoning steps or inject specific structure into how models think, but leave scattered problem conditions unchanged. Inspired by the way humans organize fragmented information into coherent explanations, we propose StoryCoder, a narrative reformulation framework that transforms code generation questions into coherent natural language narratives, providing richer contextual structure than simple rephrasings. Each narrative consists of three components: a task overview, constraints, and example test cases, guided by the selected algorithm and genre. Experiments across 11 models on HumanEval, LiveCodeBench, and CodeForces demonstrate consistent improvements, with an average gain of 18.7% in zero-shot pass@10. Beyond accuracy, our analyses reveal that narrative reformulation guides models toward correct algorithmic strategies, reduces implementation errors, and induces a more modular code structure. The analyses further show that these benefits depend on narrative coherence and genre alignment, suggesting that structured problem representation is important for code generation regardless of model scale or architecture. Our code is available at this https URL. 

---
# MetaDent: Labeling Clinical Images for Vision-Language Models in Dentistry 

**Authors**: Meng-Xun Li, Wen-Hui Deng, Zhi-Xing Wu, Chun-Xiao Jin, Jia-Min Wu, Yue Han, James Kit Hon Tsoi, Gui-Song Xia, Cui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14866)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated significant potential in medical image analysis, yet their application in intraoral photography remains largely underexplored due to the lack of fine-grained, annotated datasets and comprehensive benchmarks. To address this, we present MetaDent, a comprehensive resource that includes (1) a novel and large-scale dentistry image dataset collected from clinical, public, and web sources; (2) a semi-structured annotation framework designed to capture the hierarchical and clinically nuanced nature of dental photography; and (3) comprehensive benchmark suites for evaluating state-of-the-art VLMs on clinical image understanding. Our labeling approach combines a high-level image summary with point-by-point, free-text descriptions of abnormalities. This method enables rich, scalable, and task-agnostic representations. We curated 60,669 dental images from diverse sources and annotated a representative subset of 2,588 images using this meta-labeling scheme. Leveraging Large Language Models (LLMs), we derive standardized benchmarks: approximately 15K Visual Question Answering (VQA) pairs and an 18-class multi-label classification dataset, which we validated with human review and error analysis to justify that the LLM-driven transition reliably preserves fidelity and semantic accuracy. We then evaluate state-of-the-art VLMs across VQA, classification, and image captioning tasks. Quantitative results reveal that even the most advanced models struggle with a fine-grained understanding of intraoral scenes, achieving moderate accuracy and producing inconsistent or incomplete descriptions in image captioning. We publicly release our dataset, annotations, and tools to foster reproducible research and accelerate the development of vision-language systems for dental applications. 

---
# Fact4ac at the Financial Misinformation Detection Challenge Task: Reference-Free Financial Misinformation Detection via Fine-Tuning and Few-Shot Prompting of Large Language Models 

**Authors**: Cuong Hoang, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14640)  

**Abstract**: The proliferation of financial misinformation poses a severe threat to market stability and investor trust, misleading market behavior and creating critical information asymmetry. Detecting such misleading narratives is inherently challenging, particularly in real-world scenarios where external evidence or supplementary references for cross-verification are strictly unavailable. This paper presents our winning methodology for the "Reference-Free Financial Misinformation Detection" shared task. Built upon the recently proposed RFC-BENCH framework (Jiang et al. 2026), this task challenges models to determine the veracity of financial claims by relying solely on internal semantic understanding and contextual consistency, rather than external fact-checking. To address this formidable evaluation setup, we propose a comprehensive framework that capitalizes on the reasoning capabilities of state-of-the-art Large Language Models (LLMs). Our approach systematically integrates in-context learning, specifically zero-shot and few-shot prompting strategies, with Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to optimally align the models with the subtle linguistic cues of financial manipulation. Our proposed system demonstrated superior efficacy, successfully securing the first-place ranking on both official leaderboards. Specifically, we achieved an accuracy of 95.4% on the public test set and 96.3% on the private test set, highlighting the robustness of our method and contributing to the acceleration of context-aware misinformation detection in financial Natural Language Processing. Our models (14B and 32B) are available at this https URL. 

---
# Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring 

**Authors**: Sumit Mukherjee, Juan Shu, Nairwita Mazumder, Tate Kernell, Celena Wheeler, Shannon Hastings, Chris Sidey-Gibbons  

**Link**: [PDF](https://arxiv.org/pdf/2604.14616)  

**Abstract**: Clinical value set authoring -- the task of identifying all codes in a standardized vocabulary that define a clinical concept -- is a recurring bottleneck in clinical quality measurement and phenotyping. A natural approach is to prompt a large language model (LLM) to generate the required codes directly, but structured clinical vocabularies are large, version-controlled, and not reliably memorized during pretraining. We propose Retrieval-Augmented Set Completion (RASC): retrieve the $K$ most similar existing value sets from a curated corpus to form a candidate pool, then apply a classifier to each candidate code. Theoretically, retrieve-and-select can reduce statistical complexity by shrinking the effective output space from the full vocabulary to a much smaller retrieved candidate pool. We demonstrate the utility of RASC on 11,803 publicly available VSAC value sets, constructing the first large-scale benchmark for this task. A cross-encoder fine-tuned on SAPBert achieves AUROC~0.852 and value-set-level F1~0.298, outperforming a simpler three-layer Multilayer Perceptron (AUROC~0.799, F1~0.250) and both reduce the number of irrelevant candidates per true positive from 12.3 (retrieval-only) to approximately 3.2 and 4.4 respectively. Zero-shot GPT-4o achieves value-set-level F1~0.105, with 48.6\% of returned codes absent from VSAC entirely. This performance gap widens with increasing value set size, consistent with RASC's theoretical advantage. We observe similar performance gains across two other classifier model types, namely a cross-encoder initialized from pre-trained SAPBert and a LightGBM model, demonstrating that RASC's benefits extend beyond a single model class. The code to download and create the benchmark dataset, as well as the model training code is available at: \href{this https URL}{this https URL}. 

---
# Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection 

**Authors**: Meng Chen, Kun Wang, Li Lu, Jiaheng Zhang, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14604)  

**Abstract**: Modern Large audio-language models (LALMs) power intelligent voice interactions by tightly integrating audio and text. This integration, however, expands the attack surface beyond text and introduces vulnerabilities in the continuous, high-dimensional audio channel. While prior work studied audio jailbreaks, the security risks of malicious audio injection and downstream behavior manipulation remain underexamined. In this work, we reveal a previously overlooked threat, auditory prompt injection, under realistic constraints of audio data-only access and strong perceptual stealth. To systematically analyze this threat, we propose \textit{AudioHijack}, a general framework that generates context-agnostic and imperceptible adversarial audio to hijack LALMs. \textit{AudioHijack} employs sampling-based gradient estimation for end-to-end optimization across diverse models, bypassing non-differentiable audio tokenization. Through attention supervision and multi-context training, it steers model attention toward adversarial audio and generalizes to unseen user contexts. We also design a convolutional blending method that modulates perturbations into natural reverberation, making them highly imperceptible to users. Extensive experiments on 13 state-of-the-art LALMs show consistent hijacking across 6 misbehavior categories, achieving average success rates of 79\%-96\% on unseen user contexts with high acoustic fidelity. Real-world studies demonstrate that commercial voice agents from Mistral AI and Microsoft Azure can be induced to execute unauthorized actions on behalf of users. These findings expose critical vulnerabilities in LALMs and highlight the urgent need for dedicated defense. 

---
# Mechanistic Decoding of Cognitive Constructs in LLMs 

**Authors**: Yitong Shou, Manhao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14593)  

**Abstract**: While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments. 

---
# CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification 

**Authors**: Yian Wang, Yuen Chen, Agam Goyal, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.14602)  

**Abstract**: Large language models (LLMs) frequently generate toxic content, posing significant risks for safe deployment. Current mitigation strategies often degrade generation quality or require costly human annotation. We propose CAUSALDETOX, a framework that identifies and intervenes on the specific attention heads causally responsible for toxic generation. Using the Probability of Necessity and Sufficiency (PNS), we isolate a minimal set of heads that are necessary and sufficient for toxicity. We utilize these components via two complementary strategies: (1) Local Inference-Time Intervention, which constructs dynamic, input-specific steering vectors for context-aware detoxification, and (2) PNS-Guided Fine-Tuning, which permanently unlearns toxic representations. We also introduce PARATOX, a novel benchmark of aligned toxic/non-toxic sentence pairs enabling controlled counterfactual evaluation. Experiments on ToxiGen, ImplicitHate, and ParaDetox show that CAUSALDETOX achieves up to 5.34% greater toxicity reduction compared to baselines while preserving linguistic fluency, and offers a 7x speedup in head selection. 

---
# Generative Augmented Inference 

**Authors**: Cheng Lu, Mengxin Wang, Dennis J. Zhang, Heng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14575)  

**Abstract**: Data-driven operations management often relies on parameters estimated from costly human-generated labels. Recent advances in large language models (LLMs) and other AI systems offer inexpensive auxiliary data, but introduce a new challenge: AI outputs are not direct observations of the target outcomes, but could involve high-dimensional representations with complex and unknown relationships to human labels. Conventional methods leverage AI predictions as direct proxies for true labels, which can be inefficient or unreliable when this relationship is weak or misspecified. We propose Generative Augmented Inference (GAI), a general framework that incorporates AI-generated outputs as informative features for estimating models of human-labeled outcomes. GAI uses an orthogonal moment construction that enables consistent estimation and valid inference with flexible, nonparametric relationship between LLM-generated outputs and human labels. We establish asymptotic normality and show a "safe default" property: relative to human-data-only estimators, GAI weakly improves estimation efficiency under arbitrary auxiliary signals and yields strict gains whenever the auxiliary information is predictive. Empirically, GAI outperforms benchmarks across diverse settings. In conjoint analysis with weak auxiliary signals, GAI reduces estimation error by about 50% and lowers human labeling requirements by over 75%. In retail pricing, where all methods access the same auxiliary inputs, GAI consistently outperforms alternative estimators, highlighting the value of its construction rather than differences in information. In health insurance choice, it cuts labeling requirements by over 90% while maintaining decision accuracy. Across applications, GAI improves confidence interval coverage without inflating width. Overall, GAI provides a principled and scalable approach to integrating AI-generated information. 

---
# VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs 

**Authors**: Sazzadul Islam, Tasnim Tabassum, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14550)  

**Abstract**: Generating synthesizable Verilog for large, hierarchical hardware designs remains a significant challenge for large language models (LLMs), which struggle to replicate the structured reasoning that human experts employ when translating complex specifications into RTL. When tasked with producing hierarchical Verilog, LLMs frequently lose context across modules, hallucinate interfaces, fabricate inter-module wiring, and fail to maintain structural coherence - failures that intensify as design complexity grows and specifications involve informal prose, figures, and tables that resist direct operationalization. To address these challenges, we present VeriGraphi, a framework that introduces a spec-anchored Knowledge Graph as the architectural substrate driving the RTL generation pipeline. VeriGraphi constructs a HDA, a structured knowledge graph that explicitly encodes module hierarchy, port-level interfaces, wiring semantics, and inter-module dependencies as first-class graph entities and relations. Built through iterative multi-agent analysis of the specification, this Knowledge Graph provides a deterministic, machine-checkable structural scaffold before code generation. Guided by the KG, a progressive coding module incrementally generates pseudo-code and synthesizable RTL while enforcing interface consistency and dependency correctness at each submodule stage. We evaluate VeriGraphi on a benchmark of three representative specification documents from the National Institute of Standards and Technology and their corresponding implementations, and we present a RV32I processor as a detailed case study to illustrate the full pipeline. The results demonstrate that VeriGraphi enables reliable hierarchical RTL generation with minimal human intervention for RISC-V, marking a significant milestone for LLM-generated hardware design while maintaining strong functional correctness. 

---
# Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks 

**Authors**: Sanidhya Vijayvargiya, Vijay Viswanathan, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2604.14624)  

**Abstract**: Humans often specify tasks incompletely, so assistants must know when and how to ask clarifying questions. However, effective clarification remains challenging in software engineering tasks as not all missing information is equally valuable, and questions must target information users can realistically provide. We study clarification in real software engineering tasks by quantifying which types of information most affect task success and which questions elicit useful responses from simulated users. Using Shapley attribution and distributional comparisons, we identify two key properties of effective clarification: task relevance (which information predicts success) and user answerability (what users can realistically provide). We operationalize these properties as multi-stage reinforcement learning rewards to train CLARITI, an 8B-parameter clarification module, that matches GPT-5's resolution rate on underspecified issues while generating 41% fewer questions. Our results suggest that grounding reward design in empirical analysis of information impact and user answerability improves clarification efficiency. 

---
# LLMs taking shortcuts in test generation: A study with SAP HANA and LevelDB 

**Authors**: Vekil Bekmyradov, Noah C. Pütz, Thomas Bartz-Beielstein  

**Link**: [PDF](https://arxiv.org/pdf/2604.14437)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results on public benchmarks, often leading to claims of advanced reasoning and understanding. However, recent research in cognitive science reveals that these models sometimes rely on shallow heuristics and memorization, taking shortcuts rather than demonstrating genuine cognitive abilities. This paper investigates LLM behavior in automated test generation for software, contrasting performance on an open-source system (LevelDB) with SAP HANA, one of the most widely deployed commercial database systems worldwide, whose proprietary codebase is guaranteed to be absent from training data. We combine cognitive evaluation principles, drawing on Mitchell's mechanism-focused assessment methodology, with empirical software testing, employing mutation score and iterative compiler-feedback repair loops to assess both accuracy and underlying reasoning strategies. Results show that LLMs excel on familiar, open-source benchmarks but struggle with unseen, complex domains, often prioritizing compilability over semantic effectiveness. These findings provide independent software engineering evidence for the broader claim that current LLMs lack robust reasoning, and highlight the need for evaluation frameworks that penalize trivial shortcuts and reward true generalization. 

---
# Hierarchical vs. Flat Iteration in Shared-Weight Transformers 

**Authors**: Sang-Il Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.14442)  

**Abstract**: We present an empirical study of whether hierarchically structured, shared-weight recurrence can match the representational quality of independent-layer stacking in a Transformer-based language model. HRM-LM replaces L independent Transformer layers with a two-speed recurrent pair: a Fast module operating at every step for local refinement, and a Slow module operating every T steps for global compression. This recurrent hierarchy is unrolled for M = N x T steps with shared parameters. The central and most robust finding, supported by a parameter-matched Universal Transformer ablation (UniTF, 1.2B) across five independent runs, is a sharp empirical gap between the two approaches. 

---
# SpaceMind: A Modular and Self-Evolving Embodied Vision-Language Agent Framework for Autonomous On-orbit Servicing 

**Authors**: Aodi Wu, Haodong Han, Xubo Luo, Ruisuo Wang, Shan He, Xue Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14399)  

**Abstract**: Autonomous on-orbit servicing demands embodied agents that perceive through visual sensors, reason about 3D spatial situations, and execute multi-phase tasks over extended horizons. We present SpaceMind, a modular and self-evolving vision-language model (VLM) agent framework that decomposes knowledge, tools, and reasoning into three independently extensible dimensions: skill modules with dynamic routing, Model Context Protocol (MCP) tools with configurable profiles, and injectable reasoning-mode skills. An MCP-Redis interface layer enables the same codebase to operate across simulation and physical hardware without modification, and a Skill Self-Evolution mechanism distills operational experience into persistent skill files without model fine-tuning. We validate SpaceMind through 192 closed-loop runs across five satellites, three task types, and two environments, a UE5 simulation and a physical laboratory, deliberately including degraded conditions to stress-test robustness. Under nominal conditions all modes achieve 90--100% navigation success; under degradation, the Prospective mode uniquely succeeds in search-and-approach tasks where other modes fail. A self-evolution study shows that the agent recovers from failure in four of six groups from a single failed episode, including complete failure to 100% success and inspection scores improving from 12 to 59 out of 100. Real-world validation confirms zero-code-modification transfer to a physical robot with 100% rendezvous success. Code: this https URL 

---
# Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection 

**Authors**: David Basil, Chirooth Girigowda, Bradley Hauer, Sahir Momin, Ning Shi, Grzegorz Kondrak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14397)  

**Abstract**: We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects English synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate these alignments and ensure their quality, we augment a pre-trained base aligner with a bilingual dictionary, which is also used to filter out incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and requiring few external resources. We plan to make our code, documentation, and generated sense inventories accessible. 

---
# The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models 

**Authors**: Akshay Paruchuri, Ishan Chatterjee, Henry Fuchs, Ehsan Adeli, Piotr Didyk  

**Link**: [PDF](https://arxiv.org/pdf/2604.14363)  

**Abstract**: Multimodal language models systematically underperform on visual perception tasks, yet the structure underlying this failure remains poorly understood. We propose centroid replacement, collapsing each token to its nearest K-means centroid, as a controlled probe for modal dependence. Across seven models spanning three architecture families, erasing text centroid structure costs 4$\times$ more accuracy than erasing visual centroid structure, exposing a universal imbalance where language representations overshadow vision even on tasks that demand visual reasoning. We exploit this asymmetry through text centroid contrastive decoding, recovering up to +16.9% accuracy on individual tasks by contrastively decoding against a text-centroid-erased reference. This intervention varies meaningfully with training approaches: standard fine-tuned models show larger gains (+5.6% on average) than preference-optimized models (+1.5% on average). Our findings suggest that modal competition is structurally localized, correctable at inference time without retraining, and quantifiable as a diagnostic signal to guide future multimodal training. 

---
# Three-Phase Transformer 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.14430)  

**Abstract**: We present Three-Phase Transformer (3PT), a residual-stream structural prior for decoder-only Transformers on a standard SwiGLU + RMSNorm + RoPE + GQA backbone. The hidden vector is partitioned into N equally-sized cyclic channels, each maintained by phase-respecting ops: a per-channel RMSNorm, a 2D Givens rotation between attention and FFN that rotates each channel by theta + i*(2*pi/N), and a head-count constraint aligning GQA heads with the partition. The architecture is a self-stabilizing equilibrium between scrambling and re-imposition, not a bolted-on module. The partition carves out a one-dimensional DC subspace orthogonal to the channels, into which we inject a fixed Gabriel's horn profile r(p) = 1/(p+1) as an absolute-position side-channel composing orthogonally with RoPE's relative-position rotation. The canonical N=3 borrows its metaphor from balanced three-phase AC, where three sinusoids 120 degrees apart sum to zero with no anti-correlated pair. At 123M parameters on WikiText-103, 3PT achieves -7.20% perplexity (-2.62% bits-per-byte) over a matched RoPE-Only baseline at +1,536 parameters (0.00124% of total), with 1.93x step-count convergence speedup (1.64x wall-clock). N behaves as a parameter-sharing knob rather than a unique optimum: at 5.5M an N-sweep over {1,2,3,4,6,8,12} is near-monotone with N=1 winning; at 123M a three-seed sweep finds N=3 and N=1 statistically indistinguishable. The load-bearing mechanism is the channel-partitioned residual stream, per-block rotation, per-phase normalization, and horn DC injection. We characterize (a) self-stabilization of the geometry without explicit enforcement, a novel instance of the conservation-law framework for neural networks; (b) a U-shaped depth profile of rotation-angle drift at 12 layers; (c) orthogonal composition with RoPE, attention, and FFN. 

---
# When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden 

**Authors**: Apoorv Prasad, Susan McRoy  

**Link**: [PDF](https://arxiv.org/pdf/2604.14356)  

**Abstract**: Women with polycystic ovary syndrome (PCOS) face substantially elevated risks of body image distress, disordered eating, and metabolic challenges, yet existing natural language processing approaches for detecting these conditions lack transparency and cannot identify co-occurring presentations. We developed small, open-source language models to automatically detect this triple burden in social media posts with grounded explainability. We collected 1,000 PCOS-related posts from six subreddits, with two trained annotators labeling posts using guidelines operationalizing Lee et al. (2017) clinical framework. Three models (Gemma-2-2B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) were fine-tuned using Low-Rank Adaptation to generate structured explanations with textual evidence. The best model achieved 75.3 percent exact match accuracy on 150 held-out posts, with robust comorbidity detection and strong explainability. Performance declined with diagnostic complexity, indicating their best use is for screening rather than autonomous diagnosis. 

---
# Tight Sample Complexity Bounds for Best-Arm Identification Under Bounded Systematic Bias 

**Authors**: Tianhao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.14345)  

**Abstract**: As search depth increases in autonomous reasoning and embodied planning, the candidate action space expands exponentially, heavily taxing computational budgets. While heuristic pruning is a common countermeasure, it operates without formal safety guarantees when surrogate models (like LLMs) exhibit systematic evaluation biases. This paper frames the node expansion process as a localized Best-Arm Identification (BAI) problem over dynamic frontiers, subject to a bounded systematic bias $L$. By inverting the Lambert W function, we establish an additive sample complexity of $\mathcal{O}((\Delta-4L)^{-2})$, which indicates that safe node elimination is only feasible when the empirical reward gap exceeds $4L$. We complement this with an information-theoretic lower bound of $\Omega((\Delta-2L)^{-2})$ to confirm the structural limits of biased search. Subsequent evaluations on both synthetic trees and complex reasoning tasks demonstrate that adhering to this local safety boundary successfully preserves optimal trajectories while maximizing sample allocation efficiency. 

---
# DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines 

**Authors**: Gabriel Pimenta de Freitas Cardoso, Caio Lucas da Silva Chacon, Jonas Felipe da Fonseca Oliveira, Paulo Henrique de Medeiros Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14314)  

**Abstract**: This manuscript introduces DharmaOCR Full and Lite, a pair of specialized small language models (SSLMs) for structured OCR that jointly optimize transcription quality, generation stability, and inference cost. It also presents DharmaOCR-Benchmark, a benchmark that covers printed, handwritten, and legal/administrative documents, and proposes a unified evaluation protocol that measures fidelity and structure while explicitly tracking text degeneration as a first-class benchmark metric (alongside unit cost). Beyond reporting degeneration rates, the manuscript empirically shows degeneration is not merely a quality failure, since it materially worsens production performance by increasing response time, reducing throughput, and inflating computational cost due to abnormally long generations. To the best of the author's knowledge, as a methodological contribution, this is the first application of Direct Preference Optimization (DPO) for OCR, explicitly using degenerate generations as rejected examples to penalize looping behavior. Combined with Supervised Fine-Tuning (SFT) for enforcing a strict JSON schema (header, margin, footer, and text), DPO consistently reduces degeneration rate across model families (up to 87.6% relative) while preserving or improving extraction quality. The resulting models, namely, DharmaOCR Full (7B) and DharmaOCR Lite (3B), set a new state-of-the-art on DharmaOCR-Benchmark, outperforming each open-source and commercial baseline model evaluated regarding extraction quality, reaching 0.925 and 0.911 scores with 0.40% and 0.20% degeneration rates. AWQ quantization reduced up to 22% per-page cost with negligible quality loss, enabling a strong quality-cost trade-off in comparison to proprietary OCR APIs and open-source alternatives. 

---
# Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery 

**Authors**: Noureddine Kermiche  

**Link**: [PDF](https://arxiv.org/pdf/2604.14375)  

**Abstract**: Catastrophic forgetting remains a primary hurdle in sequential task learning for artificial neural networks. We propose a silicon-native modular architecture that achieves structural parameter isolation using Task-Specific Experts and a distributed, outlier-based Gatekeeper. Moving beyond traditional sequential consolidation, our framework utilizes a Simultaneous Pipeline where Teacher learning, Student distillation, and Router manifold acquisition occur in parallel while raw data is present in a localized training session. This approach ensures computational efficiency and complies with privacy mandates like GDPR by deleting raw data as soon as a task is learned. We demonstrate that a Tight-Bottleneck Autoencoder (TB-AE) can effectively distinguish semantically crowded manifolds in high-dimensional latent spaces, overcoming the posterior collapse inherent to standard variational methods. By establishing strict topological boundaries, our TB-AE resolves latent space crowding in 4096-D LLM embeddings to provide a robust, unsupervised novelty signal. Furthermore, we validate an Autonomous Retrieval mechanism that confidently identifies returning manifolds, enabling stable lifelong learning without redundant module instantiation. Empirical results demonstrate that our ``Live Distillation'' approach acts as a natural regularizer, achieving strong retention across computer vision and natural language processing domains without suffering a student fidelity gap. 

---
# BiCon-Gate: Consistency-Gated De-colloquialisation for Dialogue Fact-Checking 

**Authors**: Hyunkyung Park, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14389)  

**Abstract**: Automated fact-checking in dialogue involves multi-turn conversations where colloquial language is frequent yet understudied. To address this gap, we propose a conservative rewrite candidate for each response claim via staged de-colloquialisation, combining lightweight surface normalisation with scoped in-claim coreference resolution. We then introduce BiCon-Gate, a semantics-aware consistency gate that selects the rewrite candidate only when it is semantically supported by the dialogue context, otherwise falling back to the original claim. This gated selection stabilises downstream fact-checking and yields gains in both evidence retrieval and fact verification. On the DialFact benchmark, our approach improves retrieval and verification, with particularly strong gains on SUPPORTS, and outperforms competitive baselines, including a decoder-based one-shot LLM rewrite that attempts to perform all de-colloquialisation steps in a single pass. 

---
# Awakening Dormant Experts:Counterfactual Routing to Mitigate MoE Hallucinations 

**Authors**: Wentao Hu, Yanbo Zhai, Xiaohui Hu, Mingkuan Zhao, Shanhong yu, Xue Liu, Kaidong Yu, Shuangyong Song, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14246)  

**Abstract**: Sparse Mixture-of-Experts (MoE) models have achieved remarkable scalability, yet they remain vulnerable to hallucinations, particularly when processing long-tail knowledge. We identify that this fragility stems from static Top-$k$ routing: routers tend to favor high-frequency patterns over rare factual associations. Consequently, ``specialist experts'' possessing critical long-tail knowledge are often assigned low gating scores and remain ``dormant'' -- under-prioritized for specific tokens despite their proven causal importance on other inputs. To address this, we propose Counterfactual Routing (CoR), a training-free inference framework designed to awaken these dormant experts. CoR integrates layer-wise perturbation analysis with the Counterfactual Expert Impact (CEI) metric to dynamically shift computational resources from syntax-dominant to knowledge-intensive layers while maintaining a constant total activation count, effectively retrieving causally decisive experts via virtual ablation. Extensive experiments on TruthfulQA, FACTOR, and TriviaQA demonstrate that CoR improves factual accuracy by 3.1\% on average without increasing the inference budget, establishing a superior Pareto frontier compared to static scaling strategies. 

---
# Challenges and Future Directions in Agentic Reverse Engineering Systems 

**Authors**: Salem Radey, Jack West, Kassem Fawaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14317)  

**Abstract**: Agentic systems built on large language models (LLMs) are increasingly being used for complex security tasks, including binary reverse engineering (RE). Despite recent growth in popularity and capability, these systems continue to face limitations in realistic settings. Cutting-edge systems still fail in complex RE scenarios that involve obfuscation, timing, and unique architecture. In this work, we examine how agentic systems perform reverse engineering tasks with static, dynamic, and hybrid agents. Through an analysis of existing agentic tool usage, we identify several limitations, including token constraints, struggles with obfuscation, and a lack of program guardrails. From these findings, we outline current challenges and position future directions for system designers to overcome from a security perspective. 

---
# Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance 

**Authors**: Bar Alon, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2604.14325)  

**Abstract**: Large language models (LLMs) achieve strong performance and have revolutionized NLP, but their lack of explainability keeps them treated as black boxes, limiting their use in domains that demand transparency and trust. A promising direction to address this issue is post-hoc text-based explanations, which aim to explain model decisions in natural language. Prior work has focused on generating convincing rationales that appear to be subjectively faithful, but it remains unclear whether these explanations are epistemically faithful, whether they reflect the internal evidence the model actually relied on for its decision. In this paper, we first assess the epistemic faithfulness of LLM-generated explanations via counterfactuals and show that they are often unfaithful. We then introduce a training-free method that enhances faithfulness by guiding explanation generation through attention-level interventions, informed by token-level heatmaps extracted via a faithful attribution method. This method significantly improves epistemic faithfulness across multiple models, benchmarks, and prompts. 

---
# Reinforcement Learning via Value Gradient Flow 

**Authors**: Haoran Xu, Kaiwen Hu, Somayeh Sojoudi, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14265)  

**Abstract**: We study behavior-regularized reinforcement learning (RL), where regularization toward a reference distribution (the dataset in offline RL or the base model in LLM RL finetuning) is essential to prevent value over-optimization caused by erroneous out-of-distribution extrapolation. Existing methods either rely on reparameterized policy gradient, which are difficult to scale to large generative models, or on reject sampling, which can be overly conservative when attempting to move beyond the behavior support. In this paper, we propose Value Gradient Flow (VGF), a scalable new paradigm for behavior-regularized RL. VGF casts behavior-regularized RL as an optimal transport problem that maps the reference distribution to the value-induced optimal policy distribution. We solve this transport problem via discrete gradient flow, where value gradients guide particles initialized from the reference distribution. Our analysis shows that VGF imposes regularization implicitly by controlling the transport budget. VGF eliminates explicit policy parameterization while remaining expressive and flexible, this enables adaptive test-time scaling by adjusting the transport budget. Extensive experiments demonstrate that VGF significantly outperforms prior methods, achieving state-of-the-art results on offline RL benchmarks (D4RL, OGBench) and LLM RL tasks. Code and runs can be found at this https URL. 

---
# ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents 

**Authors**: Zhuofeng Li, Yi Lu, Dongfu Jiang, Haoxiang Zhang, Yuyang Bai, Chuan Li, Yu Wang, Shuiwang Ji, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14261)  

**Abstract**: The rapid rise in AI conference submissions has driven increasing exploration of large language models (LLMs) for peer review support. However, LLM-based reviewers often generate superficial, formulaic comments lacking substantive, evidence-grounded feedback. We attribute this to the underutilization of two key components of human reviewing: explicit rubrics and contextual grounding in existing work. To address this, we introduce REVIEWBENCH, a benchmark evaluating review text according to paper-specific rubrics derived from official guidelines, the paper's content, and human-written reviews. We further propose REVIEWGROUNDER, a rubric-guided, tool-integrated multi-agent framework that decomposes reviewing into drafting and grounding stages, enriching shallow drafts via targeted evidence consolidation. Experiments on REVIEWBENCH show that REVIEWGROUNDER, using a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, consistently outperforms baselines with substantially stronger/larger backbones (e.g., GPT-4.1 and DeepSeek-R1-670B) in both alignment with human judgments and rubric-based review quality across 8 dimensions. The code is available \href{this https URL}{here}. 

---
# Mamba-SSM with LLM Reasoning for Biomarker Discovery: Causal Feature Refinement via Chain-of-Thought Gene Evaluation 

**Authors**: Pushpa Kumar Balan, Aijing Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14334)  

**Abstract**: Gradient saliency from deep sequence models surfaces candidate biomarkers efficiently, but the resulting gene lists are contaminated by tissue-composition confounders that degrade downstream classifiers. We study whether LLM chain-of-thought (CoT) reasoning can faithfully filter these confounders, and whether reasoning quality drives downstream performance. We train a Mamba SSM on TCGA-BRCA RNA-seq and extract the top-50 genes by gradient saliency; DeepSeek-R1 evaluates every candidate with structured CoT to produce a final 17-gene set. The raw 50-gene saliency set (no LLM) performs worse than a 5,000-gene variance baseline (AUC 0.832 vs. 0.903), while the LLM-filtered set surpasses it (AUC 0.927), using 294x fewer features. A faithfulness audit (COSMIC CGC, OncoKB, PAM50) reveals only 6 of 17 selected genes (35.3%) are validated BRCA biomarkers, yet 10 of 16 known BRCA genes in the input were missed - including FOXA1. This gap between downstream performance and reasoning faithfulness suggests selective faithfulness: targeted confounder removal is sufficient for performance gains even without comprehensive recall. 

---
# PolyBench: Benchmarking LLM Forecasting and Trading Capabilities on Live Prediction Market Data 

**Authors**: Pu Cheng, Juncheng Liu, Yunshen Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.14199)  

**Abstract**: Predicting real-world events from live market signals demands systems that fuse qualitative news with quantitative order-book dynamics under strict temporal discipline -- a challenge existing benchmarks fail to capture. We present \textbf{PolyBench}, a multimodal benchmark derived from Polymarket that records point-in-time cross-sections of 38,666 binary prediction markets spanning 4,997 events, synchronously coupling each snapshot with a Central Limit Order Book (CLOB) state and a real-time news stream. Using PolyBench, we evaluate seven state-of-the-art Large Language Models -- spanning open- and closed-source families -- generating 36,165 predictions under identical, timestamp-locked market states collected between February 6 and 12, 2026. Our multidimensional framework assesses directional accuracy, our proposed Confidence-Weighted Return (CWR), Annualized Percentage Yield (APY), and Sharpe ratio via realistic order-book execution simulation. The results reveal a pronounced performance divergence: only two of seven models achieve positive financial returns -- MiMo-V2-Flash at \textbf{17.6%} CWR and Gemini-3-Flash at 6.2% CWR -- while the remaining five incur losses despite uniformly high stated confidence. These findings highlight the gap between surface-level language fluency and genuine probabilistic reasoning under live market uncertainty, and establish PolyBench as a contamination-proof, financially-grounded evaluation standard for future LLM research. Our dataset and code available at \underline{\href{this https URL}{this https URL}}. 

---
# MixAtlas: Uncertainty-aware Data Mixture Optimization for Multimodal LLM Midtraining 

**Authors**: Bingbing Wen, Sirajul Salekin, Feiyang Kang, Bill Howe, Lucy Lu Wang, Javier Movellan, Manjot Bilkhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14198)  

**Abstract**: Domain reweighting can improve sample efficiency and downstream generalization, but data-mixture optimization for multimodal midtraining remains largely unexplored. Current multimodal training recipes tune mixtures along a single dimension, typically data format or task type. We introduce MixAtlas, a method that produces benchmark-targeted data recipes that can be inspected, adapted, and transferred to new corpora. MixAtlas decomposes the training corpus along two axes: image concepts (10 visual-domain clusters discovered via CLIP embeddings) and task supervision (5 objective types including captioning, OCR, grounding, detection, and VQA). Using small proxy models (Qwen2-0.5B) paired with a Gaussian-process surrogate and GP-UCB acquisition, MixAtlas searches the resulting mixture space with the same proxy budget as regression-based baselines but finds better-performing mixtures. We evaluate on 10 benchmarks spanning visual understanding, document reasoning, and multimodal reasoning. On Qwen2-7B, optimized mixtures improve average performance by 8.5%-17.6% over the strongest baseline; on Qwen2.5-7B, gains are 1.0%-3.3%. Both settings reach baseline-equivalent training loss in up to 2 times fewer steps. Recipes discovered on 0.5B proxies transfer to 7B-scale training across Qwen model families. 

---
# The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure 

**Authors**: David A. Cook  

**Link**: [PDF](https://arxiv.org/pdf/2604.14197)  

**Abstract**: Large language model (LLM) performance depends heavily on prompt design, yet prompt construction is often described and applied inconsistently. Our purpose was to derive a reference framework for structuring LLM prompts. This paper presents PICCO, a framework derived through a rigorous synthesis of 11 previously published prompting frameworks identified through a multi-database search. The analysis yields two main contributions. First, it proposes a taxonomy that distinguishes prompt frameworks, prompt elements, prompt generation, prompting techniques, and prompt engineering as related but non-equivalent concepts. Second, it derives a five-element reference architecture for prompt generation: Persona, Instructions, Context, Constraints, and Output (PICCO). For each element, we define its function, scope, and relationship to other elements, with the goal of improving conceptual clarity and supporting more systematic prompt design. Finally, to support application of the framework, we outline key concepts relevant to implementation, including prompting techniques (e.g., zero-shot, few-shot, chain-of-thought, ensembling, decomposition, and self-critique, with selected variants), human and automated approaches to iterative prompt engineering, responsible prompting considerations such as security, privacy, bias, and trust, and priorities for future research. This work is a conceptual and methodological contribution: it formalizes a common structure for prompt specification and comparison, but does not claim empirical validation of PICCO as an optimization method. 

---
# An Underexplored Frontier: Large Language Models for Rare Disease Patient Education and Communication -- A scoping review 

**Authors**: Zaifu Zhan, Yu Hou, Kai Yu, Min Zeng, Anita Burgun, Xiaoyi Chen, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14179)  

**Abstract**: Rare diseases affect over 300 million people worldwide and are characterized by complex care pathways, limited clinical expertise, and substantial unmet communication needs throughout the long patient journey. Recent advances in large language models (LLMs) offer new opportunities to support patient education and communication, yet their application in rare diseases remains unclear.
We conducted a scoping review of studies published between January 2022 and March 2026 across major databases, identifying 12 studies on LLM-based rare disease patient education and communication. Data were extracted on study characteristics, application scenarios, model usage, and evaluation methods, and synthesized using descriptive and qualitative analyses.
The literature is highly recent and dominated by general-purpose models, particularly ChatGPT. Most studies focus on patient question answering using curated question sets, with limited use of real-world data or longitudinal communication scenarios. Evaluations are primarily centered on accuracy, with limited attention to patient-centered dimensions such as readability, empathy, and communication quality. Multilingual communication is rarely addressed.
Overall, the field remains at an early stage. Future research should prioritize patient-centered design, domain-adapted methods, and real-world deployment to support safe, adaptive, and effective communication in rare diseases. 

---
# QU-NLP at ArchEHR-QA 2026: Two-Stage QLoRA Fine-Tuning of Qwen3-4B for Patient-Oriented Clinical Question Answering and Evidence Sentence Alignment 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14175)  

**Abstract**: We present a unified system addressing both Subtask 3 (answer generation) and Subtask 4 (evidence sentence alignment) of the ArchEHR-QA Shared Task. For Subtask 3, we apply two-stage Quantised Low-Rank Adaptation (QLoRA) to Qwen3-4B loaded in 4-bit NF4 quantisation: first on 30,000 samples from the emrQA-MedSQuAD corpus to establish clinical domain competence, then on the 20 annotated development cases to learn the task-specific output style. Our system achieves an overall score of 32.87 on the official test-2026 split (BLEU = 9.42, ROUGE-L = 27.04, SARI = 55.42, BERTScore = 43.00, AlignScore = 25.28, MEDCON = 37.04). For Subtask 4, we develop a weighted ensemble of three retrieval methods - BM25 with relative thresholding, TF-IDF cosine similarity, and a fine-tuned cross-encoder - to identify note sentences supporting a given gold answer, achieving a micro-F1 of 67.16 on the 100-case test set. Experiments reveal that both subtasks expose the same fundamental challenge: 20 annotated training cases are insufficient to distinguish relevant from irrelevant clinical sentences, pointing to data augmentation as the highest-leverage future direction. 

---
# Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations 

**Authors**: Ziyin Zhou, Jianyi Zhang, Xu ji, Yilong Li, Jiameng Han, Zhangchi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14172)  

**Abstract**: Large Language Models (LLMs) are essential for analyzing and addressing vulnerabilities in cybersecurity. However, among over 200,000 vulnerabilities were discovered in the past decade, more than 30,000 have been changed or updated. This necessitates frequent updates to the training datasets and internal knowledge bases of LLMs to maintain knowledge consistency. In this paper, we focus on the problem of knowledge discrepancy and conflict within CVE (Common Vulnerabilities and Exposures) detection and analysis. This problem hinders LLMs' ability to retrieve the latest knowledge from original training datasets, leading to knowledge conflicts, fabrications of factually incorrect results, and generation hallucinations. To address this problem, we propose an innovative two-stage framework called CRVA-TGRAG (Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generation). First, to improve document retrieval accuracy during the retrieval stage, we utilize Parent Document Segmentation and an ensemble retrieval scheme based on semantic similarity and inverted indexing. Second, to enhance LLMs' capabilities based on the retrieval of CVE dataset in generation stage, we employ a teacher-guided preference optimization technique to fine-tune LLMs. Our framework not only enhances the quality of content retrieval through RAG but also leverages the advantages of preference fine-tuning in LLMs to answer questions more effectively and precisely. Experiments demonstrate our method achieves higher accuracy in retrieving the latest CVEs compared to external knowledge bases. In conclusion, our framework significantly mitigates potential knowledge conflicts and inconsistencies that may arise from relying solely on LLMs for knowledge retrieval. 

---
# Stateful Evidence-Driven Retrieval-Augmented Generation with Iterative Reasoning 

**Authors**: Qi Dong, Ziheng Lin, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.14170)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in external knowledge but often suffers from flat context representations and stateless retrieval, leading to unstable performance. We propose Stateful Evidence-Driven RAG with Iterative Reasoning, a framework that models question answering as a progressive evidence accumulation process. Retrieved documents are converted into structured reasoning units with explicit relevance and confidence signals and maintained in a persistent evidence pool capturing both supportive and non-supportive information. The framework performs evidence-driven deficiency analysis to identify gaps and conflicts and iteratively refines queries to guide subsequent retrieval. This iterative reasoning process enables stable evidence aggregation and improves robustness to noisy retrieval. Experiments on multiple question answering benchmarks demonstrate consistent improvements over standard RAG and multi-step baselines, while effectively accumulating high-quality evidence and maintaining stable performance under substantial retrieval noise. 

---
# Can Large Language Models Detect Methodological Flaws? Evidence from Gesture Recognition for UAV-Based Rescue Operation Based on Deep Learning 

**Authors**: Domonkos Varga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14161)  

**Abstract**: Reliable evaluation is essential in machine learning research, yet methodological flaws-particularly data leakage-continue to undermine the validity of reported results. In this work, we investigate whether large language models (LLMs) can act as independent analytical agents capable of identifying such issues in published studies. As a case study, we analyze a gesture-recognition paper reporting near-perfect accuracy on a small, human-centered dataset. We first show that the evaluation protocol is consistent with subject-level data leakage due to non-independent training and test splits. We then assess whether this flaw can be detected independently by six state-of-the-art LLMs, each analyzing the original paper without prior context using an identical prompt. All models consistently identify the evaluation as flawed and attribute the reported performance to non-independent data partitioning, supported by indicators such as overlapping learning curves, minimal generalization gap, and near-perfect classification results. These findings suggest that LLMs can detect common methodological issues based solely on published artifacts. While not definitive, their consistent agreement highlights their potential as complementary tools for improving reproducibility and supporting scientific auditing. 

---
# Grading the Unspoken: Evaluating Tacit Reasoning in Quantum Field Theory and String Theory with LLMs 

**Authors**: Xingyang Yu, Yinghuan Zhang, Yufei Zhang, Zijun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.14188)  

**Abstract**: Large language models have demonstrated impressive performance across many domains of mathematics and physics. One natural question is whether such models can support research in highly abstract theoretical fields such as quantum field theory and string theory. Evaluating this possibility faces an immediate challenge: correctness in these domains is layered, tacit, and fundamentally non-binary. Standard answer-matching metrics fail to capture whether intermediate conceptual steps are properly reconstructed or whether implicit structural constraints are respected. We construct a compact expert-curated dataset of twelve questions spanning core areas of quantum field theory and string theory, and introduce a five-level grading rubric separating statement correctness, key concept awareness, reasoning chain presence, tacit step reconstruction, and enrichment. Evaluating multiple contemporary LLMs, we observe near-ceiling performance on explicit derivations within stable conceptual frames, but systematic degradation when tasks require reconstruction of omitted reasoning steps or reorganization of representations under global consistency constraints. These failures are driven not only by missing intermediate steps, but by an instability in representation selection: models often fail to identify the correct conceptual framing required to resolve implicit tensions. We argue that highly abstract theoretical physics provides a uniquely sensitive lens on the epistemic limits of current evaluation paradigms. 

---
# Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation 

**Authors**: Junhong Liang, Yifan Lu, Ekaterina Kochmar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.14177)  

**Abstract**: Grammatical error correction (GEC) and explanation (GEE) have made rapid progress, but real teaching scenarios also require \emph{learner-friendly pedagogical feedback} that is actionable, level-appropriate, and encouraging. We introduce \textbf{SPFG} (\textbf{S}poken \textbf{P}edagogical \textbf{F}eedback \textbf{G}eneration), a dataset built based on the Speak \& Improve Challenge 2025 corpus, pairing fluency-oriented transcriptions with GEC targets and \emph{human-verified} teacher-style feedback, including preferred/rejected feedback pairs for preference learning. We study a transcript-based Spoken Grammatical Error Correction (SGEC) setting and evaluate three instruction-tuned LLMs (Qwen2.5, Llama-3.1, and GLM-4), comparing supervised fine-tuning (SFT) with preference-based alignment (using DPO and KTO) for jointly generating corrections and feedback. Results show that SFT provides the most consistent improvements, while DPO/KTO yield smaller or mixed gains, and that correction quality and feedback quality are weakly coupled. Our implementation is available at this https URL. 

---
# SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models 

**Authors**: Tomer Atia, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14163)  

**Abstract**: Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential details, including vessel identity, position, nature of the distress, and required assistance. In practice, however, automatic analysis remains difficult because distress messages are often brief, noisy, and produced under stress, may deviate from the prescribed format, and are further degraded by automatic speech recognition (ASR) errors caused by channel noise and speaker stress. This paper presents SeaAlert, an LLM-based framework for robust analysis of maritime distress communications. To address the scarcity of labeled real-world data, we develop a synthetic data generation pipeline in which an LLM produces realistic and diverse maritime messages, including challenging variants in which standard distress codewords are omitted or replaced with less explicit expressions. The generated utterances are synthesized into speech, degraded with simulated VHF noise, and transcribed by an ASR system to obtain realistic noisy transcripts. 

---
# HUOZIIME: An On-Device LLM-enhanced Input Method for Deep Personalization 

**Authors**: Baocai Shan, Yuzhuang Xu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2604.14159)  

**Abstract**: Mobile input method editors (IMEs) are the primary interface for text input, yet they remain constrained to manual typing and struggle to produce personalized text. While lightweight large language models (LLMs) make on-device auxiliary generation feasible, enabling deeply personalized, privacy-preserving, and real-time generative IMEs poses fundamental this http URL this end, we present HUOZIIME, a personalized on-device IME powered by LLM. We endow HUOZIIME with initial human-like prediction ability by post-training a base LLM on synthesized personalization data. Notably, a hierarchical memory mechanism is designed to continually capture and leverage user-specific input history. Furthermore, we perform systemic optimizations tailored to on-device LLMbased IME deployment, ensuring efficient and responsive operation under mobile this http URL demonstrate efficient on-device execution and high-fidelity memory-driven personalization. Code and package are available at this https URL. 

---
# MemGround: Long-Term Memory Evaluation Kit for Large Language Models in Gamified Scenarios 

**Authors**: Yihang Ding, Wanke Xia, Yiting Zhao, Jinbo Su, Jialiang Yang, Zhengbo Zhang, Ke Wang, Wenming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14158)  

**Abstract**: Current evaluations of long-term memory in LLMs are fundamentally static. By fixating on simple retrieval and short-context inference, they neglect the multifaceted nature of complex memory systems, such as dynamic state tracking and hierarchical reasoning in continuous interactions. To overcome these limitations, we propose MemGround, a rigorous long-term memory benchmark natively grounded in rich, gamified interactive scenarios. To systematically assess these capabilities, MemGround introduces a three-tier hierarchical framework that evaluates Surface State Memory, Temporal Associative Memory, and Reasoning-Based Memory through specialized interactive tasks. Furthermore, to comprehensively quantify both memory utilization and behavioral trajectories, we propose a multi-dimensional metric suite comprising Question-Answer Score (QA Overall), Memory Fragments Unlocked (MFU), Memory Fragments with Correct Order (MFCO), and Exploration Trajectory Diagrams (ETD). Extensive experiments reveal that state-of-the-art LLMs and memory agents still struggle with sustained dynamic tracking, temporal event association, and complex reasoning derived from long-term accumulated evidence in interactive environments. 

---
# Text2Arch: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions 

**Authors**: Shivank Garg, Sankalp Mittal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14941)  

**Abstract**: Communicating complex system designs or scientific processes through text alone is inefficient and prone to ambiguity. A system that automatically generates scientific architecture diagrams from text with high semantic fidelity can be useful in multiple applications like enterprise architecture visualization, AI-driven software design, and educational content creation. Hence, in this paper, we focus on leveraging language models to perform semantic understanding of the input text description to generate intermediate code that can be processed to generate high-fidelity architecture diagrams. Unfortunately, no clean large-scale open-access dataset exists, implying lack of any effective open models for this task. Hence, we contribute a comprehensive dataset, \system, comprising scientific architecture images, their corresponding textual descriptions, and associated DOT code representations. Leveraging this resource, we fine-tune a suite of small language models, and also perform in-context learning using GPT-4o. Through extensive experimentation, we show that \system{} models significantly outperform existing baseline models like DiagramAgent and perform at par with in-context learning-based generations from GPT-4o. We make the code, data and models publicly available. 

---
# IE as Cache: Information Extraction Enhanced Agentic Reasoning 

**Authors**: Hang Lv, Sheng Liang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14930)  

**Abstract**: Information Extraction aims to distill structured, decision-relevant information from unstructured text, serving as a foundation for downstream understanding and reasoning. However, it is traditionally treated merely as a terminal objective: once extracted, the resulting structure is often consumed in isolation rather than maintained and reused during multi-step inference. Moving beyond this, we propose \textit{IE-as-Cache}, a framework that repurposes IE as a cognitive cache to enhance agentic reasoning. Drawing inspiration from hierarchical computer memory, our approach combines query-driven extraction with cache-aware reasoning to dynamically maintain compact intermediate information and filter noise. Experiments on challenging benchmarks across diverse LLMs demonstrate significant improvements in reasoning accuracy, indicating that IE can be effectively repurposed as a reusable cognitive resource and offering a promising direction for future research on downstream uses of IE. 

---
# MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events 

**Authors**: Raunak Agarwal, Markus Wenzel, Simon Baur, Jonas Zimmer, George Harvey, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.15203)  

**Abstract**: Machine learning in high-stakes domains such as healthcare requires not only strong predictive performance but also reliable uncertainty quantification (UQ) to support human oversight. Multi-label text classification (MLTC) is a central task in this domain, yet remains challenging due to label imbalances, dependencies, and combinatorial complexity. Existing MLTC benchmarks are increasingly saturated and may be affected by training data contamination, making it difficult to distinguish genuine reasoning capabilities from memorization. We introduce MADE, a living MLTC benchmark derived from {m}edical device {ad}verse {e}vent reports and continuously updated with newly published reports to prevent contamination. MADE features a long-tailed distribution of hierarchical labels and enables reproducible evaluation with strict temporal splits. We establish baselines across more than 20 encoder- and decoder-only models under fine-tuning and few-shot settings (instruction-tuned/reasoning variants, local/API-accessible). We systematically assess entropy-/consistency-based and self-verbalized UQ methods. Results show clear trade-offs: smaller discriminatively fine-tuned decoders achieve the strongest head-to-tail accuracy while maintaining competitive UQ; generative fine-tuning delivers the most reliable UQ; large reasoning models improve performance on rare labels yet exhibit surprisingly weak UQ; and self-verbalized confidence is not a reliable proxy for uncertainty. Our work is publicly available at this https URL. 

---
# Explain the Flag: Contextualizing Hate Speech Beyond Censorship 

**Authors**: Jason Liartis, Eirini Kaldeli, Lambrini Gyftokosta, Eleftherios Chelioudakis, Orfeas Menis Mastromichalakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.14970)  

**Abstract**: Hate, derogatory, and offensive speech remains a persistent challenge in online platforms and public discourse. While automated detection systems are widely used, most focus on censorship or removal, raising concerns for transparency and freedom of expression, and limiting opportunities to explain why content is harmful. To address these issues, explanatory approaches have emerged as a promising solution, aiming to make hate speech detection more transparent, accountable, and informative. In this paper, we present a hybrid approach that combines Large Language Models (LLMs) with three newly created and curated vocabularies to detect and explain hate speech in English, French, and Greek. Our system captures both inherently derogatory expressions tied to identity characteristics and direct group-targeted content through two complementary pipelines: one that detects and disambiguates problematic terms using the curated vocabularies, and one that leverages LLMs as context-aware evaluators of group-targeting content. The outputs are fused into grounded explanations that clarify why content is flagged. Human evaluation shows that our hybrid approach is accurate, with high-quality explanations, outperforming LLM-only baselines. 

---
# QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies 

**Authors**: Alexey Khoroshilov, Alexey Chernysh, Orkhan Ekhtibarov, Nini Kamkia, Dmitry Zmitrovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15151)  

**Abstract**: Large language models have demonstrated strong performance on general-purpose programming tasks, yet their ability to generate executable algorithmic trading strategies remains underexplored. Unlike standard code benchmarks, trading-strategy generation requires simultaneous mastery of domain-specific financial logic, knowledge of a specialized API, and the ability to produce code that is not only syntactically correct but also leads to actual trades on historical data. In this work, we present QuantCode-Bench, a benchmark for the systematic evaluation of modern LLMs in generating strategies for the Backtrader framework from textual descriptions in English. The benchmark contains 400 tasks of varying difficulty collected from Reddit, TradingView, StackExchange, GitHub, and synthetic sources. Evaluation is conducted through a multi-stage pipeline that checks syntactic correctness, successful backtest execution, the presence of trades, and semantic alignment with the task description using an LLM judge. We compare state-of-the-art models in two settings: single-turn, where the strategy must be generated correctly on the first attempt, and agentic multi-turn, where the model receives iterative feedback and may repair its errors. We analyze the failure modes across different stages of the pipeline and show that the main limitations of current models are not related to syntax, but rather to the correct operationalization of trading logic, proper API usage, and adherence to task semantics. These findings suggest that trading strategy generation constitutes a distinct class of domain-specific code generation tasks in which success requires not only technical correctness, but also alignment between natural-language descriptions, financial logic, and the observable behavior of the strategy on data. 

---
# DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering 

**Authors**: Neha Srikanth, Jordan Boyd-Graber, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2604.15140)  

**Abstract**: We introduce DiscoTrace, a method to identify the rhetorical strategies that answerers use when responding to information-seeking questions. DiscoTrace represents answers as a sequence of question-related discourse acts paired with interpretations of the original question, annotated on top of rhetorical structure theory parses. Applying DiscoTrace to answers from nine different human communities reveals that communities have diverse preferences for answer construction. In contrast, LLMs do not exhibit rhetorical diversity in their answers, even when prompted to mimic specific human community answering guidelines. LLMs also systematically opt for breadth, addressing interpretations of questions that human answerers choose not to address. Our findings can guide the development of pragmatic LLM answerers that consider a range of strategies informed by context in QA. 

---
# Fabricator or dynamic translator? 

**Authors**: Lisa Vasileva, Karin Sim  

**Link**: [PDF](https://arxiv.org/pdf/2604.15165)  

**Abstract**: LLMs are proving to be adept at machine translation although due to their generative nature they may at times overgenerate in various ways. These overgenerations are different from the neurobabble seen in NMT and range from LLM self-explanations, to risky confabulations, to appropriate explanations, where the LLM is able to act as a human translator would, enabling greater comprehension for the target audience. Detecting and determining the exact nature of the overgenerations is a challenging task. We detail different strategies we have explored for our work in a commercial setting, and present our results. 

---
# Pangu-ACE: Adaptive Cascaded Experts for Educational Response Generation on EduBench 

**Authors**: Dinghao Li, Wenlong Zhou, Zhimin Chen, Yuehan Peng, Hong Ni, Chengfu Zou, Guoyu Shi, Yaochen Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14828)  

**Abstract**: Educational assistants should spend more computation only when the task needs it. This paper rewrites our earlier draft around the system that was actually implemented and archived in the repository: a sample-level 1B to 7B cascade for the shared-8 EduBench benchmark. The final system, Pangu-ACE, uses a 1B tutor-router to produce a draft answer plus routing signals, then either accepts the draft or escalates the sample to a 7B specialist prompt. We also correct a major offline evaluation bug: earlier summaries over-credited some open-form outputs that only satisfied superficial format checks. After CPU-side rescoring from saved prediction JSONL, the full Chinese test archive (7013 samples) shows that cascade_final improves deterministic quality from 0.457 to 0.538 and format validity from 0.707 to 0.866 over the legacy rule_v2 system while accepting 19.7% of requests directly at 1B. Routing is strongly task dependent: IP is accepted by 1B 78.0% of the time, while QG and EC still escalate almost always. The current archived deployment does not yet show latency gains, so the defensible efficiency story is routing selectivity rather than wall-clock speedup. We also package a reproducible artifact-first paper workflow and clarify the remaining external-baseline gap: GPT-5.4 re-judging is implemented locally, but the configured provider endpoint and key are invalid, so final sampled-baseline alignment with GPT-5.4 remains pending infrastructure repair. 

---
# Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem 

**Authors**: Zeguan Xiao, Siqing Li, Yong Wang, Xuetao Wei, Jian Yang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14808)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove targeted knowledge while preserving general capability. In this paper, we recast LLM unlearning as an asymmetric two-task problem: retention is the primary objective and forgetting is an auxiliary. From this perspective, we propose a retention-prioritized gradient synthesis framework that decouples task-specific gradient extraction from conflict-aware combination. Instantiating the framework, we adapt established PCGrad to resolve gradient conflicts, and introduce SAGO, a novel retention-prioritized gradient synthesis method. Theoretically, both variants ensure non-negative cosine similarity with the retain gradient, while SAGO achieves strictly tighter alignment through constructive sign-constrained synthesis. Empirically, on WMDP Bio/Cyber and RWKU benchmarks, SAGO consistently pushes the Pareto frontier: e.g., on WMDP Bio (SimNPO+GD), recovery of target model MMLU performance progresses from 44.6% (naive) to 94.0% (+PCGrad) and further to 96.0% (+SAGO), while maintaining comparable forgetting strength. Our results show that re-shaping gradient geometry, rather than re-balancing losses, is the key to mitigating unlearning-retention trade-offs. 

---
# Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems 

**Authors**: Nishanth Madhusudhan, Vikas Yadav, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2604.14799)  

**Abstract**: Effective abstention (EA), recognizing evidence insufficiency and refraining from answering, is critical for reliable multimodal systems. Yet existing evaluation paradigms for vision-language models (VLMs) and multi-agent systems (MAS) assume answerability, pushing models to always respond. Abstention has been studied in text-only settings but remains underexplored multimodally; current benchmarks either ignore unanswerability or rely on coarse methods that miss realistic failure modes. We introduce MM-AQA, a benchmark that constructs unanswerable instances from answerable ones via transformations along two axes: visual modality dependency and evidence sufficiency. Evaluating three frontier VLMs spanning closed and open-source models and two MAS architectures across 2079 samples, we find: (1) under standard prompting, VLMs rarely abstain; even simple confidence baselines outperform this setup, (2) MAS improves abstention but introduces an accuracy-abstention trade-off, (3) sequential designs match or exceed iterative variants, suggesting the bottleneck is miscalibration rather than reasoning depth, and (4) models abstain when image or text evidence is absent, but attempt reconciliation with degraded or contradictory evidence. Effective multimodal abstention requires abstention-aware training rather than better prompting or more agents. 

---
# CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction 

**Authors**: Sizhe Wang, Ziqi Xu, Claire Najjuuko, Charles Alba, Chenyang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14651)  

**Abstract**: Clinical language models (LMs) are increasingly applied to support clinical risk prediction from free-text notes, yet their uncertainty estimates often remain poorly calibrated and clinically unreliable. In this work, we propose Clinical Uncertainty Risk Alignment (CURA), a framework that aligns clinical LM-based risk estimates and uncertainty with both individual error likelihoods and cohort-level ambiguities. CURA first fine-tunes domain-specific clinical LMs to obtain task-adapted patient embeddings, and then performs uncertainty fine-tuning of a multi-head classifier using a bi-level uncertainty objective. Specifically, an individual-level calibration term aligns predictive uncertainty with each patient's likelihood of error, while a cohort-aware regularizer pulls risk estimates toward event rates in their local neighborhoods in the embedding space and places extra weight on ambiguous cohorts near the decision boundary. We further show that this cohort-aware term can be interpreted as a cross-entropy loss with neighborhood-informed soft labels, providing a label-smoothing view of our method. Extensive experiments on MIMIC-IV clinical risk prediction tasks across various clinical LMs show that CURA consistently improves calibration metrics without substantially compromising discrimination. Further analysis illustrates that CURA reduces overconfident false reassurance and yields more trustworthy uncertainty estimates for downstream clinical decision support. 

---
# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge 

**Authors**: Seyun Bae, Seokhan Lee, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14644)  

**Abstract**: The inability to filter out in advance all potentially problematic data from the pre-training of large language models has given rise to the need for methods for unlearning specific pieces of knowledge after training. Existing techniques overlook the need for continuous and immediate action, causing them to suffer from degraded utility as updates accumulate and protracted exposure of sensitive information. To address these issues, we propose Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge (CURaTE). Our method begins by training a sentence embedding model on a dataset designed to enable the formation of sharp decision boundaries for determining whether a given input prompt corresponds to any stored forget requests. The similarity of a given input to the forget requests is then used to determine whether to answer or return a refusal response. We show that even with such a simple approach, not only does CURaTE achieve more effective forgetting than existing methods, but by avoiding modification of the language model parameters, it also maintains near perfect knowledge preservation over any number of updates and is the only method capable of continual unlearning in real-time. 

---
# Pushing the Boundaries of Multiple Choice Evaluation to One Hundred Options 

**Authors**: Nahyun Lee, Guijin Son  

**Link**: [PDF](https://arxiv.org/pdf/2604.14634)  

**Abstract**: Multiple choice evaluation is widely used for benchmarking large language models, yet near ceiling accuracy in low option settings can be sustained by shortcut strategies that obscure true competence. Therefore, we propose a massive option evaluation protocol that scales the candidate set to one hundred options and sharply reduces the impact of chance performance. We apply this framework to a Korean orthography error detection task where models must pick the single incorrect sentence from a large candidate set. With fixed targets and repeated resampling and shuffling, we obtain stable estimates while separating content driven failures from positional artifacts. Across experiments, results indicate that strong performance in low option settings can overstate model competence. This apparent advantage often weakens under dense interference at high $N$, revealing gaps that conventional benchmarks tend to obscure. We identify two failure modes, semantic confusion and position bias toward early options under uncertainty. To isolate the effect of context length, we run padding controlled and length matched tests, which suggest that the main bottleneck is candidate ranking rather than context length. Together, these findings support massive option evaluation as a general framework for stress testing model reliability under extreme distractor density, beyond what low option benchmarks can reveal. 

---
# Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints? 

**Authors**: Atrey Desai, Sathvik Nair  

**Link**: [PDF](https://arxiv.org/pdf/2604.14459)  

**Abstract**: For humans, filler-gap dependencies require a shared representation across different syntactic constructions. Although causal analyses suggest this may also be true for LLMs (Boguraev et al., 2025), it is still unclear if such a representation also exists for language models trained on developmentally feasible quantities of data. We applied Distributed Alignment Search (DAS, Geiger et al. (2024)) to LMs trained on varying amounts of data from the BabyLM challenge (Warstadt et al., 2023), to evaluate whether representations of filler-gap dependencies transfer between wh-questions and topicalization, which greatly vary in terms of their input frequency. Our results suggest shared, yet item-sensitive mechanisms may develop with limited training data. More importantly, LMs still require far more data than humans to learn comparable generalizations, highlighting the need for language-specific biases in models of language acquisition. 

---
# Psychological Steering of Large Language Models 

**Authors**: Leonardo Blas, Robin Jia, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14463)  

**Abstract**: Large language models (LLMs) emulate a consistent human-like behavior that can be shaped through activation-level interventions. This paradigm is converging on additive residual-stream injections, which rely on injection-strength sweeps to approximate optimal intervention settings. However, existing methods restrict the search space and sweep in uncalibrated activation-space units, potentially missing optimal intervention conditions. Thus, we introduce a psychological steering framework that performs unbounded, fluency-constrained sweeps in semantically calibrated units. Our method derives and calibrates residual-stream injections using psychological artifacts, and we use the IPIP-NEO-120, which measures the OCEAN personality model, to compare six injection methods. We find that mean-difference (MD) injections outperform Personality Prompting (P$^2$), an established baseline for OCEAN steering, in open-ended generation in 11 of 14 LLMs, with gains of 3.6\% to 16.4\%, overturning prior reports favoring prompting and positioning representation engineering as a new frontier in open-ended psychological steering. Further, we find that a hybrid of P$^2$ and MD injections outperforms both methods in 13 of 14 LLMs, with gains over P$^2$ ranging from 5.6\% to 21.9\% and from 3.3\% to 26.7\% over MD injections. Finally, we show that MD injections align with the Linear Representation Hypothesis and provide reliable, approximately linear control knobs for psychological steering. Nevertheless, they also induce OCEAN trait covariance patterns that depart from the Big Two model, suggesting a gap between learned representations and human psychology. 

---
# The Autocorrelation Blind Spot: Why 42% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious 

**Authors**: Ferdinand M. Schessl  

**Link**: [PDF](https://arxiv.org/pdf/2604.14414)  

**Abstract**: Turn-level metrics are widely used to evaluate properties of multi-turn human-LLM conversations, from safety and sycophancy to dialogue quality. However, consecutive turns within a conversation are not statistically independent -- a fact that virtually all current evaluation pipelines fail to correct for in their statistical inference. We systematically characterize the autocorrelation structure of 66 turn-level metrics across 202 multi-turn conversations (11,639 turn pairs, 5 German-speaking users, 4 LLM platforms) and demonstrate that naive pooled analysis produces severely inflated significance estimates: 42% of associations that appear significant under standard pooled testing fail to survive cluster-robust correction. The inflation varies substantially across categories rather than scaling linearly with autocorrelation: three memoryless families (embedding velocity, directional, differential) aggregate to 14%, while the seven non-memoryless families (thermo-cycle, frame distance, lexical/structural, rolling windows, cumulative, interaction, timestamp) aggregate to 33%, with individual category rates ranging from 0% to 100% depending on per-family effect size. We present a two-stage correction framework combining Chelton (1983) effective degrees of freedom with conversation-level block bootstrap, and validate it on a pre-registered hold-out split where cluster-robust metrics replicate at 57% versus 30% for pooled-only metrics. We provide concrete design principles, a publication checklist, and open-source code for the correction pipeline. A survey of ~30 recent papers at major NLP and AI venues that compute turn-level statistics in LLM evaluations finds that only 4 address temporal dependence at all, and 26 do not correct for it. 

---
# Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness 

**Authors**: Hao An, Yibin Lou, Jiayi Guo, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14324)  

**Abstract**: Large language models (LLMs) often exhibit hallucinations due to their inability to accurately perceive their own knowledge boundaries. Existing abstention fine-tuning methods typically partition datasets directly based on response accuracy, causing models to suffer from severe label noise near the decision boundaries and consequently exhibit high rates of abstentions or hallucinations. This paper adopts a latent space representation perspective, revealing a "gray zone" near the decision hyperplane where internal belief ambiguity constitutes the core performance bottleneck. Based on this insight, we propose the **GeoDe** (**Geo**metric **De**noising) framework for abstention fine-tuning. This method constructs a truth hyperplane using linear probes and performs "geometric denoising" by employing geometric distance as a confidence signal for abstention decisions. This approach filters out ambiguous boundary samples while retaining high-fidelity signals for fine-tuning. Experiments across multiple models (Llama3, Qwen3) and benchmark datasets (TriviaQA, NQ, SciQ, SimpleQA) demonstrate that GeoDe significantly enhances model truthfulness and demonstrates strong generalization in out-of-distribution (OOD) scenarios. Code is available at this https URL. 

---
# LLM Predictive Scoring and Validation: Inferring Experience Ratings from Unstructured Text 

**Authors**: Jason Potteiger, Andrew Hong, Ito Zapata  

**Link**: [PDF](https://arxiv.org/pdf/2604.14321)  

**Abstract**: We tasked GPT-4.1 to read what baseball fans wrote about their game-day experience and predict the overall experience rating each fan gave on a 0-10 survey scale. The model received only the text of a single open-ended response. These AI predictions were compared with the actual experience ratings captured by the survey instrument across approximately 10,000 fan responses from five Major League Baseball teams. In total two-thirds of predicted ratings fell within one point of self-reported fan ratings (67% within +/-1, 36% exact match), and the predicted measurement was near-deterministic across three independent scoring runs (87% exact agreement, 99.9% within +/-1). Predicted ratings aligned most strongly with the overall experience rating (r = 0.82) rather than with any specific aspect of the game-day experience such as parking, concessions, staff, etc. However, predictions were systematically lower than self-reported ratings by approximately one point, and this gap was not driven by any single aspect. Rather, our analysis shows that self-reported ratings capture the fan's verdict, an overall evaluative judgment that integrates the entire experience. While predicted ratings quantify the impact of salient moments characterized as memorable, emotionally intense, unusual, or actionable. Each measure contains information the other misses. These baseline results establish that a simple, unoptimized prompt can directionally predict how fans rate their experience from the text a fan wrote and that a gap between the two numbers can be interpreted as a construct difference worth preserving rather than an error to eliminate. 

---
# EuropeMedQA Study Protocol: A Multilingual, Multimodal Medical Examination Dataset for Language Model Evaluation 

**Authors**: Francesco Andrea Causio, Vittorio De Vita, Olivia Riccomi, Michele Ferramola, Federico Felizzi, Antonio Cristiano, Lorenzo De Mori, Chiara Battipaglia, Melissa Sawaya, Luigi De Angelis, Marcello Di Pumpo, Alessandra Piscitelli, Pietro Eric Risuleo, Alessia Longo, Giulia Vojvodic, Mariapia Vassalli, Bianca Destro Castaniti, Nicolò Scarsi, Manuel Del Medico  

**Link**: [PDF](https://arxiv.org/pdf/2604.14306)  

**Abstract**: While Large Language Models (LLMs) have demonstrated high proficiency on English-centric medical examinations, their performance often declines when faced with non-English languages and multimodal diagnostic tasks. This study protocol describes the development of EuropeMedQA, the first comprehensive, multilingual, and multimodal medical examination dataset sourced from official regulatory exams in Italy, France, Spain, and Portugal. Following FAIR data principles and SPIRIT-AI guidelines, we describe a rigorous curation process and an automated translation pipeline for comparative analysis. We evaluate contemporary multimodal LLMs using a zero-shot, strictly constrained prompting strategy to assess cross-lingual transfer and visual reasoning. EuropeMedQA aims to provide a contamination-resistant benchmark that reflects the complexity of European clinical practices and fosters the development of more generalizable medical AI. 

---
# MEME-Fusion@CHiPSAL 2026: Multimodal Ablation Study of Hate Detection and Sentiment Analysis on Nepali Memes 

**Authors**: Samir Wagle, Reewaj Khanal, Abiral Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14218)  

**Abstract**: Hate speech detection in Devanagari-scripted social media memes presents compounded challenges: multimodal content structure, script-specific linguistic complexity, and extreme data scarcity in low-resource settings. This paper presents our system for the CHiPSAL 2026 shared task, addressing both Subtask A (binary hate speech detection) and Subtask B (three-class sentiment classification: positive, neutral, negative). We propose a hybrid cross-modal attention fusion architecture that combines CLIP (ViT-B/32) for visual encoding with BGE-M3 for multilingual text representation, connected through 4-head self-attention and a learnable gating network that dynamically weights modality contributions on a per-sample basis. Systematic evaluation across eight model configurations demonstrates that explicit cross-modal reasoning achieves a 5.9% F1-macro improvement over text-only baselines on Subtask A, while uncovering two unexpected but critical findings: English-centric vision models exhibit near-random performance on Devanagari script, and standard ensemble methods catastrophically degrade under data scarcity (N nearly equal to 850 per fold) due to correlated overfitting. The code can be accessed at this https URL 

---
# CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14214)  

**Abstract**: Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propose Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback. This forces the optimization process to produce prompts that elicit concise responses containing only critical information and reasoning. We evaluate our approach on complex reasoning datasets, specifically GSM8K, LogiQA and BIG-Bench Hard. We achieved an 80.6\% reduction in token consumption while maintaining competitive accuracy, seeing only a nominal decline in performance. This presents a pragmatic solution for deploying token-efficient and cost-effective agentic AI systems in production pipelines. 

---
# Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model 

**Authors**: Jiuting Chen, Yuan Lian, Hao Wu, Tianqi Huang, Hiroshi Sasaki, Makoto Kouno, Jongil Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14180)  

**Abstract**: We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we investigate whether the model can distinguish known from unknown inputs, and crucially, whether it can express this distinction in its generated text.
We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional events) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than for in-distribution questions (8.3%, p = 0.023), reflecting rhetorical conventions rather than genuine metacognition.
We replicate both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B parameters. We further show that uncertainty expression frequency is determined entirely by training data conventions, with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF. 

---
# Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters 

**Authors**: Bryan Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2604.14174)  

**Abstract**: Alignment-tuned language models frequently suppress factual log-probabilities on politically sensitive topics despite retaining the knowledge in their hidden representations. We show that a 786K-parameter (approximately 0.02% of the base model) post-transformer adapter, trained on frozen hidden states, corrects this suppression on 31 ideology-discriminating facts across Qwen3-4B, 8B, and 14B. The adapter memorizes all 15 training facts and generalizes to 11--39% of 16 held-out facts across 5 random splits per scale, with zero knowledge regressions via anchored training. Both gated (SwiGLU) and ungated (linear bottleneck) adapters achieve comparable results; neither consistently outperforms the other (Fisher exact p > 0.09 at all scales). On instruct models, the adapter corrects log-probability rankings. When applied at all token positions during generation, the adapter produces incoherent output; however, when applied only at the current prediction position (last-position-only), the adapter produces coherent, less censored text. A logit-space adapter operating after token projection fails to produce coherent generation at any application mode, suggesting hidden-state intervention is the correct level for generation correction. A previously undocumented silent gradient bug in Apple MLX explains all null results in earlier iterations of this work: the standard pattern nn.value_and_grad(model, fn)(this http URL()) returns zero gradients without error; the correct pattern nn.value_and_grad(model, fn)(model, data) resolves this. We provide a minimal reproduction and discuss implications for other adapter research using MLX. 

---
# Chinese Language Is Not More Efficient Than English in Vibe Coding: A Preliminary Study on Token Cost and Problem-Solving Rate 

**Authors**: Simiao Ren, Xingyu Shen, Yuchen Zhou, Dennis, Ankit Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.14210)  

**Abstract**: A claim has been circulating on social media and practitioner forums that Chinese prompts are more token-efficient than English for LLM coding tasks, potentially reducing costs by up to 40\%. This claim has influenced developers to consider switching to Chinese for ``vibe coding'' to save on API costs. In this paper, we conduct a rigorous empirical study using SWE-bench Lite, a benchmark of software engineering tasks, to evaluate whether this claim of Chinese token efficiency holds up to scrutiny. Our results reveal three key findings: First, the efficiency advantage of Chinese is not observed. Second, token cost varies by model architecture in ways that defy simple assumptions: while MiniMax-2.7 shows 1.28x higher token costs for Chinese, GLM-5 actually consumes fewer tokens with Chinese prompts. Third, and most importantly, we found that the success rate when prompting in Chinese is generally lower than in English across all models we tested. We also measure cost efficiency as expected cost per successful task -- jointly accounting for token consumption and task resolution rate. These findings should be interpreted as preliminary evidence rather than a definitive conclusion, given the limited number of models evaluated and the narrow set of benchmarks tested due to resource constraints; they indicate that language effects on token cost are model-dependent, and that practitioners should not expect cost savings or performance gains just by switching their prompt language to Chinese. 

---
# Chronological Knowledge Retrieval: A Retrieval-Augmented Generation Approach to Construction Project Documentation 

**Authors**: Ioannis-Aris Kostis, Natalia Sanchiz, Steeve De Schryver, François Denis, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2604.14169)  

**Abstract**: In large-scale construction projects, the continuous evolution of decisions generates extensive records, most often captured in meeting minutes. Since decisions may override previous ones, professionals often need to reconstruct the history of specific choices. Retrieving such information manually from raw archives is both labor-intensive and error-prone. From a user perspective, we address this challenge by enabling conversational access to the whole set of project meeting minutes. Professionals can pose natural-language questions and receive answers that are both semantically relevant and explicitly time-annotated, allowing them to follow the chronology of decisions. From a technical perspective, our solution employs a Retrieval-Augmented Generation (RAG) framework that integrates semantic search with large language models to ensure accurate and context-aware responses. We demonstrate the approach using an anonymized, industry-sourced dataset of meeting minutes from a completed construction project by a large company in Belgium. The dataset is annotated and enriched with expert-defined queries to support systematic evaluation. Both the dataset and the open-source implementation are made available to the community to foster further research on conversational access to time-annotated project documentation. 

---
# Benchmarking Linguistic Adaptation in Comparable-Sized LLMs: A Study of Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B on Romanized Nepali 

**Authors**: Ananda Rimal, Adarsha Rimal  

**Link**: [PDF](https://arxiv.org/pdf/2604.14171)  

**Abstract**: Romanized Nepali, the Nepali language written in the Latin alphabet, is the dominant medium for informal digital communication in Nepal, yet it remains critically underresourced in the landscape of Large Language Models (LLMs). This study presents a systematic benchmarking of linguistic adaptation across three comparable-sized open-weight models: Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B. We evaluate these architectures under zero-shot and fine-tuned settings using a curated bilingual dataset of 10,000 transliterated instruction-following samples. Performance is quantified across five metrics spanning seven measurement dimensions: Perplexity (PPL), BERTScore, chrF++, ROUGE-1, ROUGE-2, ROUGE-L, and BLEU, capturing fluency, phonetic consistency, and semantic integrity. Models were fine-tuned using Quantized Low-Rank Adaptation (QLoRA) with Rank-Stabilized LoRA (rsLoRA) at rank r=32 on dual NVIDIA Tesla T4 GPUs, training only approximately 1% of each model's parameters in under 27 total GPU-hours. At zero-shot, all three models fail to generate Romanized Nepali, each exhibiting a distinct architecture-specific failure mode. Following fine-tuning, all three resolve these failures and converge to BERTScore approximately 0.75 and chrF++ greater than 23. Overall dimension-wise assessment across ten criteria identifies Qwen3-8B as the overall recommended architecture, being the only model to produce semantically relevant zero-shot output and leading all structural alignment metrics post-SFT. The adaptation headroom hypothesis is confirmed: Llama-3.1-8B, despite its weakest zero-shot baseline, achieves the largest absolute fine-tuning gains in PPL (Delta = -49.77) and BERTScore (Delta = +0.3287), making it the preferred choice for iterative low-resource development pipelines. This work establishes the first rigorous baseline for Romanized Nepali adaptation in comparable-sized open-weight LLMs. 

---
# Chinese Essay Rhetoric Recognition Using LoRA, In-context Learning and Model Ensemble 

**Authors**: Yuxuan Lai, Xiajing Wang, Chen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14167)  

**Abstract**: Rhetoric recognition is a critical component in automated essay scoring. By identifying rhetorical elements in student writing, AI systems can better assess linguistic and higher-order thinking skills, making it an essential task in the area of AI for education. In this paper, we leverage Large Language Models (LLMs) for the Chinese rhetoric recognition task. Specifically, we explore Low-Rank Adaptation (LoRA) based fine-tuning and in-context learning to integrate rhetoric knowledge into LLMs. We formulate the outputs as JSON to obtain structural outputs and translate keys to Chinese. To further enhance the performance, we also investigate several model ensemble methods. Our method achieves the best performance on all three tracks of CCL 2025 Chinese essay rhetoric recognition evaluation task, winning the first prize. 

---
# Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text 

**Authors**: Filippo Morbiato, Markus Keller, Priya Nair, Luca Romano  

**Link**: [PDF](https://arxiv.org/pdf/2604.14166)  

**Abstract**: Mapping Cyber Threat Intelligence (CTI) text to MITRE ATT\&CK technique IDs is a critical task for understanding adversary behaviors and automating threat defense. While recent Retrieval-Augmented Generation (RAG) approaches have demonstrated promising capabilities in this domain, they fundamentally rely on a flat retrieval paradigm. By treating all techniques uniformly, these methods overlook the inherent taxonomy of the ATT\&CK framework, where techniques are structurally organized under high-level tactics. In this paper, we propose H-TechniqueRAG, a novel hierarchical RAG framework that injects this tactic-technique taxonomy as a strong inductive bias to achieve highly efficient and accurate annotation. Our approach introduces a two-stage hierarchical retrieval mechanism: it first identifies the macro-level tactics (the adversary's technical goals) and subsequently narrows the search to techniques within those tactics, effectively reducing the candidate search space by 77.5\%. To further bridge the gap between retrieval and generation, we design a tactic-aware reranking module and a hierarchy-constrained context organization strategy that mitigates LLM context overload and improves reasoning precision. Comprehensive experiments across three diverse CTI datasets demonstrate that H-TechniqueRAG not only outperforms the state-of-the-art TechniqueRAG by 3.8\% in F1 score, but also achieves a 62.4\% reduction in inference latency and a 60\% decrease in LLM API calls. Further analysis reveals that our hierarchical structural priors equip the model with superior cross-domain generalization and provide security analysts with highly interpretable, step-by-step decision paths. 

---
# EviSearch: A Human in the Loop System for Extracting and Auditing Clinical Evidence for Systematic Reviews 

**Authors**: Naman Ahuja, Saniya Mulla, Muhammad Ali Khan, Zaryab Bin Riaz, Kaneez Zahra Rubab Khakwani, Mohamad Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14165)  

**Abstract**: We present EviSearch, a multi-agent extraction system that automates the creation of ontology-aligned clinical evidence tables directly from native trial PDFs while guaranteeing per-cell provenance for audit and human verification. EviSearch pairs a PDF-query agent (which preserves rendered layout and figures) with a retrieval-guided search agent and a reconciliation module that forces page-level verification when agents disagree. The pipeline is designed for high-precision extraction across multimodal evidence sources (text, tables, figures) and for generating reviewer-actionable provenance that clinicians can inspect and correct. On a clinician-curated benchmark of oncology trial papers, EviSearch substantially improves extraction accuracy relative to strong parsed-text baselines while providing comprehensive attribution coverage. By logging reconciler decisions and reviewer edits, the system produces structured preference and supervision signals that bootstrap iterative model improvement. EviSearch is intended to accelerate living systematic review workflows, reduce manual curation burden, and provide a safe, auditable path for integrating LLM-based extraction into evidence synthesis pipelines. 

---
# Decoupling Scores and Text: The Politeness Principle in Peer Review 

**Authors**: Yingxuan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14162)  

**Abstract**: Authors often struggle to interpret peer review feedback, deriving false hope from polite comments or feeling confused by specific low scores. To investigate this, we construct a dataset of over 30,000 ICLR 2021-2025 submissions and compare acceptance prediction performance using numerical scores versus text reviews. Our experiments reveal a significant performance gap: score-based models achieve 91% accuracy, while text-based models reach only 81% even with large language models, indicating that textual information is considerably less reliable. To explain this phenomenon, we first analyze the 9% of samples that score-based models fail to predict, finding their score distributions exhibit high kurtosis and negative skewness, which suggests that individual low scores play a decisive role in rejection even when the average score falls near the borderline. We then examine why text-based accuracy significantly lags behind scores from a review sentiment perspective, revealing the prevalence of the Politeness Principle: reviews of rejected papers still contain more positive than negative sentiment words, masking the true rejection signal and making it difficult for authors to judge outcomes from text alone. 

---
# How to Fine-Tune a Reasoning Model? A Teacher-Student Cooperation Framework to Synthesize Student-Consistent SFT Data 

**Authors**: Zixian Huang, Kaichen Yang, Xu Huang, Feiyang Hao, Qiming Ge, Bowen Li, He Du, Kai Chen, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14164)  

**Abstract**: A widely adopted strategy for model enhancement is to use synthetic data generated by a stronger model for supervised fine-tuning (SFT). However, for emerging reasoning models like Qwen3-8B, this approach often fails to improve reasoning capabilities and can even lead to a substantial drop in performance. In this work, we identify substantial stylistic divergence between teacher generated data and the distribution of student as a major factor impacting SFT. To bridge this gap, we propose a Teacher-Student Cooperation Data Synthesis framework (TESSY), which interleaves teacher and student models to alternately generate style and non-style tokens. Consequently, TESSY produces synthetic sequences that inherit the advanced reasoning capabilities of the teacher while maintaining stylistic consistency with the distribution of the student. In experiments on code generation using GPT-OSS-120B as the teacher, fine-tuning Qwen3-8B on teacher-generated data leads to performance drops of 3.25% on LiveCodeBench-Pro and 10.02% on OJBench, whereas TESSY achieves improvements of 11.25% and 6.68%. 

---
# Compressed-Sensing-Guided, Inference-Aware Structured Reduction for Large Language Models 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14156)  

**Abstract**: Large language models deliver strong generative performance but at the cost of massive parameter counts, memory use, and decoding latency. Prior work has shown that pruning and structured sparsity can preserve accuracy under substantial compression, while prompt-compression methods reduce latency by removing redundant input tokens. However, these two directions remain largely separate. Most model-compression methods are static and optimized offline, and they do not exploit the fact that different prompts and decoding steps activate different latent computational pathways. Prompt-compression methods reduce sequence length, but they do not adapt the executed model subnetwork.
We propose a unified compressed-sensing-guided framework for dynamic LLM execution. Random measurement operators probe latent model usage, sparse recovery estimates task-conditioned and token-adaptive support sets, and the recovered supports are compiled into hardware-efficient sparse execution paths over blocks, attention heads, channels, and feed-forward substructures. The framework introduces five key contributions: task-conditioned measurements, so different prompts induce different sparse supports; token-adaptive recovery, so active substructures are re-estimated during decoding; formal sample-complexity bounds under restricted isometry or mutual incoherence assumptions; compile-to-hardware constraints that restrict recovery to GPU-efficient structures; and a joint objective that unifies prompt compression with model reduction. Together, these components recast LLM inference as a measurement-and-recovery problem with explicit approximation guarantees and deployment-oriented speedup constraints. 

---
# MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation 

**Authors**: Yan Li, Zezi Zeng, Yifan Yang, Yuqing Yang, Ning Liao, Weiwei Guo, Lili Qiu, Mingxi Cheng, Qi Dai, Zhendong Wang, Zhengyuan Yang, Xue Yang, Ji Li, Lijuan Wang, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.15309)  

**Abstract**: The rapid progress of Artificial Intelligence Generated Content (AIGC) tools enables images, videos, and visualizations to be created on demand for webpage design, offering a flexible and increasingly adopted paradigm for modern UI/UX. However, directly integrating such tools into automated webpage generation often leads to style inconsistency and poor global coherence, as elements are generated in isolation. We propose MM-WebAgent, a hierarchical agentic framework for multimodal webpage generation that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection. MM-WebAgent jointly optimizes global layout, local multimodal content, and their integration, producing coherent and visually consistent webpages. We further introduce a benchmark for multimodal webpage generation and a multi-level evaluation protocol for systematic assessment. Experiments demonstrate that MM-WebAgent outperforms code-generation and agent-based baselines, especially on multimodal element generation and integration. Code & Data: this https URL. 

---
# Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation 

**Authors**: Zichong Li, Chen Liang, Liliang Ren, Tuo Zhao, Yelong Shen, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14339)  

**Abstract**: Large language models (LLMs) increasingly operate in settings that require reliable long-context understanding, such as retrieval-augmented generation and multi-document reasoning. A common strategy is to fine-tune pretrained short-context models at the target sequence length. However, we find that standard long-context adaptation can remain brittle: model accuracy depends strongly on the absolute placement of relevant evidence, exhibiting high positional variance even when controlling for task format and difficulty.
We propose RoPE-Perturbed Self-Distillation, a training regularizer that improves positional robustness. The core idea is to form alternative "views" of the same training sequence by perturbing its RoPE indices -- effectively moving parts of the context to different positions -- and to train the model to produce consistent predictions across views via self-distillation. This encourages reliance on semantic signals instead of brittle position dependencies. Experiments on long-context adaptation of Llama-3-8B and Qwen-3-4B demonstrate consistent gains on long-context benchmarks, including up to 12.04% improvement on RULER-64K for Llama-3-8B and 2.71% on RULER-256K for Qwen-3-4B after SFT, alongside improved length extrapolation beyond the training context window. 

---
# AdaSplash-2: Faster Differentiable Sparse Attention 

**Authors**: Nuno Gonçalves, Hugo Pitorro, Vlad Niculae, Edoardo Ponti, Lei Li, Andre Martins, Marcos Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2604.15180)  

**Abstract**: Sparse attention has been proposed as a way to alleviate the quadratic cost of transformers, a central bottleneck in long-context training. A promising line of work is $\alpha$-entmax attention, a differentiable sparse alternative to softmax that enables input-dependent sparsity yet has lagged behind softmax due to the computational overhead necessary to compute the normalizer $\tau$. In this paper, we introduce AdaSplash-2, which addresses this limitation through a novel histogram-based initialization that reduces the number of iterations needed to compute $\tau$ to typically 1--2. The key idea is to compute a coarse histogram of attention scores on the fly and store it in on-chip SRAM, yielding a more accurate initialization that enables fast forward and backward computation. Combined with a sparsity-aware GPU implementation that skips zero blocks with low overhead, AdaSplash-2 matches or improves per-step training time relative to FlashAttention-2 when block sparsity is moderate-to-high (e.g., $>$60\%), which often occurs at long-context lengths. On downstream tasks, models trained with our efficient $\alpha$-entmax attention match softmax baselines at short-context lengths and achieve substantial gains in long-context settings. 

---
# From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution 

**Authors**: Junjie Wang, Yiming Ren, Haoyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15097)  

**Abstract**: This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative experience accumulation: attached failure history is more effective in Gene than in Skill or freeform text, editable structure matters beyond content alone, and failure information is most useful when distilled into compact warnings rather than naively appended. On CritPt, gene-evolved systems improve over their paired base models from 9.1% to 18.57% and from 17.7% to 27.14%. These results suggest that the core problem in experience reuse is not how to supply more experience, but how to encode experience as a compact, control-oriented, evolution-ready object. 

---
# RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models 

**Authors**: Gabriele Mattioli, Evelyn Turri, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14951)  

**Abstract**: Tool learning with foundation models aims to endow AI systems with the ability to invoke external resources -- such as APIs, computational utilities, and specialized models -- to solve complex tasks beyond the reach of standalone language generation. While recent advances in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have expanded their reasoning and perception capabilities, existing tool-use methods are predominantly limited to text-only inputs and closed-world settings. Consequently, they struggle to interpret multimodal user instructions and cannot generalize to tools unseen during training. In this work, we introduce RaTA-Tool, a novel framework for open-world multimodal tool selection. Rather than learning direct mappings from user queries to fixed tool identifiers, our approach enables an MLLM to convert a multimodal query into a structured task description and subsequently retrieve the most appropriate tool by matching this representation against semantically rich, machine-readable tool descriptions. This retrieval-based formulation naturally supports extensibility to new tools without retraining. To further improve alignment between task descriptions and tool selection, we incorporate a preference-based optimization stage using Direct Preference Optimization (DPO). To support research in this setting, we also introduce the first dataset for open-world multimodal tool use, featuring standardized tool descriptions derived from Hugging Face model cards. Extensive experiments demonstrate that our approach significantly improves tool-selection performance, particularly in open-world, multimodal scenarios. 

---
# Acceptance Dynamics Across Cognitive Domains in Speculative Decoding 

**Authors**: Saif Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2604.14682)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains (rho in [-0.20, -0.15]). Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies. Index Terms--speculative decoding, large language model inference, tree attention, draft model, acceptance probability, LLM efficiency 

---
# ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding 

**Authors**: Walaa Amer, Uday das, Fadi Kurdahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14612)  

**Abstract**: Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact version of the model as a draft model with selective re-evaluation by the full target model. Some existing methods form the draft model by dynamically learning which layers to skip during inference, effectively creating a smaller subnetwork to speed up computation. However, using heuristic-based approaches to select layers to skip can often be simpler and more effective. In this paper, we propose ConfLayers, a dynamic plug-and-play approach to forming the draft model in self-speculative decoding via confidence-based intermediate layer skipping. The process iteratively computes confidence scores for all layers, selects layers to skip based on an adaptive threshold, evaluates the performance of the resulting set, and updates the best selection until no further improvement is achieved or a maximum number of iterations is reached. This framework avoids the overhead and complexity of training a layer skipping policy and can provide more consistent speed-quality trade-offs while preserving the adaptivity of the draft model to diverse tasks and datasets. The performance evaluation of ConfLayers across different models and datasets shows that our novel approach offers up to 1.4x speedup over vanilla LLM generation. 

---
# Attention to Mamba: A Recipe for Cross-Architecture Distillation 

**Authors**: Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, Federico Danieli  

**Link**: [PDF](https://arxiv.org/pdf/2604.14191)  

**Abstract**: State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a naïve distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks. The key argument from our work is that, by equipping Mamba with a principled initialization, we can recover an overall better recipe for cross-architectural distillation. To this end, we propose a principled two-stage approach: first, we distill knowledge from a traditional Transformer into a linearized version of Attention, using an adaptation of the kernel trick. Then, we distill the linearized version into an adapted Mamba model that does not use any Attention block. Overall, the distilled Mamba model is able to preserve the original Pythia-1B Transformer performance in downstream tasks, maintaining a perplexity of 14.11 close to the teacher's 13.86. To show the efficacy of our recipe, we conduct thorough ablations at 1B scale with 10B tokens varying sequence mixer architecture, scaling analysis on model sizes and total distillation tokens, and a sensitivity analysis on tokens allocation between stages. 

---
# Learning Adaptive Reasoning Paths for Efficient Visual Reasoning 

**Authors**: Yixu Huang, Tinghui Zhu, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14568)  

**Abstract**: Visual reasoning models (VRMs) have recently shown strong cross-modal reasoning capabilities by integrating visual perception with language reasoning. However, they often suffer from overthinking, producing unnecessarily long reasoning chains for any tasks. We attribute this issue to \textbf{Reasoning Path Redundancy} in visual reasoning: many visual questions do not require the full reasoning process. To address this, we propose \textbf{AVR}, an adaptive visual reasoning framework that decomposes visual reasoning into three cognitive functions: visual perception, logical reasoning, and answer application. It further enables models to dynamically choose among three response formats: Full Format, Perception-Only Format, and Direct Answer. AVR is trained with FS-GRPO, an adaptation of Group Relative Policy Optimization that encourages the model to select the most efficient reasoning format while preserving correctness. Experiments on multiple vision-language benchmarks show that AVR reduces token usage by 50--90\% while maintaining overall accuracy, especially in perception-intensive tasks. These results demonstrate that adaptive visual reasoning can effectively mitigate overthinking in VRMs. Code and data are available at: this https URL. 

---
# Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis 

**Authors**: Aizierjiang Aiersilan, Mohamad Koubeissi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14216)  

**Abstract**: Predicting post-surgical seizure outcomes in pharmacoresistant epilepsy is a clinical challenge. Conventional deep-learning approaches operate on static, single-timepoint pre-operative scans, omitting longitudinal morphological changes. We propose \emph{Neuro-Oracle}, a three-stage framework that: (i) distils pre-to-post-operative MRI changes into a compact 512-dimensional trajectory vector using a 3D Siamese contrastive encoder; (ii) retrieves historically similar surgical trajectories from a population archive via nearest-neighbour search; and (iii) synthesises a natural-language prognosis grounded in the retrieved evidence using a quantized Llama-3-8B reasoning agent. Evaluations are conducted on the public EPISURG dataset ($N{=}268$ longitudinally paired cases) using five-fold stratified cross-validation. Since ground-truth seizure-freedom scores are unavailable, we utilize a clinical proxy label based on the resection type. We acknowledge that the network representations may potentially learn the anatomical features of the resection cavities (i.e., temporal versus non-temporal locations) rather than true prognostic morphometry. Our current evaluation thus serves mainly as a proof-of-concept for the trajectory-aware retrieval architecture. Trajectory-based classifiers achieve AUC values between 0.834 and 0.905, compared with 0.793 for a single-timepoint ResNet-50 baseline. The Neuro-Oracle agent (M5) matches the AUC of purely discriminative trajectory classifiers (0.867) while producing structured justifications with zero observed hallucinations under our audit protocol. A Siamese Diversity Ensemble (M6) of trajectory-space classifiers attains an AUC of 0.905 without language-model overhead. 

---
# SAGE Celer 2.6 Technical Card 

**Authors**: SAGEA Research Team, Basab Jha, Firoj Paudel, Ujjwal Puri, Adrian Liu, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14168)  

**Abstract**: We introduce SAGE Celer 2.6, the latest in our line of general-purpose Celer models from SAGEA. Celer 2.6 is available in 5B, 10B, and 27B parameter sizes and benefits from extensive architectural modifications and further pre-training on an undisclosed model. Using our Inverse Reasoning (IR) pipeline, SAGEA natively trains Celer 2.6 to validate its own logic paths, minimizing cascading error and hallucination in complex reasoning tasks. Celer 2.6 also boasts natively integrated multimodal functionality with an end-to-end vision encoder to avoid common pitfalls in adapter-based approaches. Celer 2.6 provides highly competitive results on mathematics, coding, and general intelligence benchmarks (ACUMEN), along with low latency. Most importantly, Celer 2.6 is specifically optimized for South Asian language support, with a custom tokenizer for the Devanagari script and strong performance in both Nepali and Hindi without sacrificing English reasoning ability. 

---
# Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems 

**Authors**: Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14228)  

**Abstract**: Claude Code is an agentic coding tool that can run shell commands, edit files, and call external services on behalf of the user. This study describes its comprehensive architecture by analyzing the publicly available TypeScript source code and further comparing it with OpenClaw, an independent open-source AI agent system that answers many of the same design questions from a different deployment context. Our analysis identifies five human values, philosophies, and needs that motivate the architecture (human decision authority, safety and security, reliable execution, capability amplification, and contextual adaptability) and traces them through thirteen design principles to specific implementation choices. The core of the system is a simple while-loop that calls the model, runs tools, and repeats. Most of the code, however, lives in the systems around this loop: a permission system with seven modes and an ML-based classifier, a five-layer compaction pipeline for context management, four extensibility mechanisms (MCP, plugins, skills, and hooks), a subagent delegation mechanism with worktree isolation, and append-oriented session storage. A comparison with OpenClaw, a multi-channel personal assistant gateway, shows that the same recurring design questions produce different architectural answers when the deployment context changes: from per-action safety classification to perimeter-level access control, from a single CLI loop to an embedded runtime within a gateway control plane, and from context-window extensions to gateway-wide capability registration. We finally identify six open design directions for future agent systems, grounded in recent empirical, architectural, and policy literature. 

---
