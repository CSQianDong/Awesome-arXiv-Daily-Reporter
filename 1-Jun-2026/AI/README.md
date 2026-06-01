# Choosing the Lens: Strategic Perspective Activation in Context-Dependent Argumentation 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2605.31581)  

**Abstract**: The same arguments often need to be evaluated under different external regimes. An agent with influence over the regime has a strategic lever that standard formalisms do not directly capture. We introduce context-dependent argumentation frameworks (CDAFs), an extension of Dung's theory in which a defeat function determines, per context, which attacks succeed. A perspective-labeled specialisation derives the defeat function from a relevance set $\rho$ and a priority $\pi$. The relevance set is the agent's action space. In a small worked example, the agent's target argument is rejected under every full-relevance injective priority, yet accepted under partial activations, one of which no VAF audience can mirror. We define the corresponding decision problem, ACTIVATION-MANIPULATION, and record baseline complexity bounds. Tight bounds and multi-agent variants are left open. 

---
# LinTree: Improving LLM Reasoning with Explicitly Structured Search Histories 

**Authors**: Liwei Kang, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31492)  

**Abstract**: Large language models (LLMs) often solve reasoning problems by generating intermediate traces that explore and revise partial solutions. From a search perspective, these traces can be viewed as linearized search trees, where the model extends a partial solution, abandons it when it fails, and backtracks to try alternatives. Compared with traditional heuristic-guided search, such a policy has a potential advantage: it conditions on the whole search trace rather than only on the current local state. We first test whether LLMs utilize this advantage by comparing trace-conditioned reasoning policies against best-first search equipped with an LLM heuristic that only observes the current local state. Across three controlled reasoning environments, Blocks World, grid Navigation, and Sokoban, we find that raw access to search history alone is not enough to reliably outperform heuristic search. We then study one possible reason: in LLM reasoning traces, the underlying search tree is only implicitly represented, and when the model backtracks or switches branches, the trace does not explicitly identify which earlier search state is being revisited. We show that adding simple parent pointers to explicitly represent the linearized tree (LinTree) structure improves both task performance and search efficiency relative to implicit reasoning models and LLM-heuristic-guided search. These results suggest that search history becomes most useful when its tree structure is made explicit, motivating more structure-aware representations for LLM reasoning. 

---
# AutoSci: A Memory-Centric Agentic System for the Full Scientific Research Lifecycle 

**Authors**: Weitong Qian, Beicheng Xu, Zhongao Xie, Bowen Fan, Guozheng Tang, Jiale Chen, Xinzhe Wu, Mingtian Yang, Chenyang Di, Jiajun Li, Lingching Tung, Peichao Lai, Yifei Xia, Ziyi Guo, Yanwei Xu, Yanzhao Qin, Shaoduo Gan, Xupeng Miao, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.31468)  

**Abstract**: Scientific research has traditionally been human-intensive, requiring researchers to coordinate literature, ideas, experiments, manuscripts, and review responses across long project cycles. The rise of LLM-based scientific agents creates an opportunity to automate this process. Such a system must support the full research lifecycle, maintain structured persistent memory across projects, and improve its own research procedures over time. However, existing systems either partially satisfy or fail to satisfy these requirements, leaving a gap for a unified automated scientific research system. As a result, we present AutoSci, a memory-centric agentic system for the full scientific research lifecycle. AutoSci is organized around four modules. SciMem provides schema-governed research memory, separating Long-Term Knowledge Memory for reusable scientific knowledge from Active Research Memory for project-level artifacts such as ideas, experiments, manuscripts, and reviews. SciFlow executes a five-stage lifecycle from literature understanding to rebuttal through a harness that controls state, context, verification, feedback, and orchestration. SciDAG augments difficult skills with DAG-shaped multi-agent operators and reusable stage-specific templates. SciEvolve converts feedback signals from users, experiments, reviews, and external environments into versioned updates to SciMem organization, SciFlow skills, and SciDAG templates. Together, these modules make AutoSci a persistent research environment that can execute, remember, and evolve across research projects. The code repository is available at this https URL. 

---
# Answer-Set-Programming-based Abstractions for Reinforcement Learning 

**Authors**: Rafael Bankosegger, Thomas Eiter, Johannes Oetsch  

**Link**: [PDF](https://arxiv.org/pdf/2605.31444)  

**Abstract**: Reinforcement Learning (RL) enables autonomous agents to learn policies from experience, but realistic problems often involve enormous state spaces, making learning and generalisation challenging. Abstraction and approximation are therefore essential. Relational Reinforcement Learning (RRL) offers a way to reason about objects and their relations, and the CARCASS framework by Martijn van Otterlo demonstrates how logical representations can model Markov Decision Processes (MDPs) in first-order domains. Originally implemented in Prolog, CARCASS leverages domain knowledge to create powerful abstractions. We explore Answer-Set Programming (ASP), which is a rich and, contrary to Prolog, fully declarative modelling language, to realise CARCASS abstractions. We evaluate our ASP-based implementation in case studies of two domains, viz. Blocks World and Minigrid. Our results indicate that CARCASS with ASP provides a promising approach to constructing abstractions for RL, especially when domain knowledge is available. 

---
# FAM-Bench: A Multimodal Benchmark for Condition-Aware Food-as-Medicine Reasoning 

**Authors**: Mingyang Mao, Bhargav Rishi Medisetti, Utkarsh Grover, Tanvir Ibrahim, Wenyan Li, Tingting Zhang, Xiaomin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.31410)  

**Abstract**: Food-as-Medicine requires models to reason beyond what a dish is or what nutrition it contains: they must decide whether a concrete food choice is appropriate for a specific health condition. Existing food AI benchmarks primarily evaluate dish recognition, recipe understanding, nutrient estimation, or general nutrition question answering, leaving this health-aware decision layer largely untested. We introduce FAM-Bench, a multi-modal Food-as-Medicine benchmark with 2500 nutrition-expert-verified instances across 13 diet-related health conditions. The benchmark contains two complementary tasks: dish-level suitability assessment, where models judge whether a dish is suitable for a condition from its image and ingredient list, and comparative dish analysis, where models rank four candidate dishes by condition-specific suitability. Both tasks require integrating ingredient evidence, visual preparation cues, and clinical nutrition constraints, providing a standardized testbed for grounded health-aware reasoning in language and vision-language models. 

---
# HypoAgent: An Agentic Framework for Interactive Abductive Hypothesis Generation over Knowledge Graphs 

**Authors**: Yisen Gao, Yixi Cai, Tianshi Zheng, Jiaxin Bai, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.31370)  

**Abstract**: Abductive reasoning over knowledge graphs aims to generate logical hypotheses that explain observed entities or facts. Existing controllable hypothesis generation methods allow users to guide this process with explicit conditions, but they remain limited in interactive settings: they struggle to ground evolving natural-language intents across multi-turn dialogues and provide little fine-grained diagnosis when generated hypotheses fail. To address these limitations, we propose HypoAgent, an Agentic framework for interactive abductive Hypothesis Generation over knowledge graphs. HypoAgent integrates three agents: an Intent Recognition Agent that grounds user utterances and dialogue history into executable KG conditions, a Hypothesis Generation Agent that performs controllable hypothesis generation according to the extracted user intention, and a Root Cause Analysis Agent that diagnoses unreliable hypothesis fragments and leverages KG neighborhood probing to identify supported refinements. Experiments on commonsense and biomedical domain-specific knowledge graphs demonstrate that HypoAgent achieves state-of-the-art semantic similarity under single-turn, multi-turn, and unconditional settings. Our code is available at this https URL. 

---
# Learning to Adapt: Self-Improving Web Agent via Cognitive-Aware Exploration 

**Authors**: Weile Chen, Bingchen Miao, Qifan Yu, Wendong Bu, Guoming Wang, Wenqiao Zhang, Shengyu Zhang, Juncheng Li, Siliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31365)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have led to promising progress in web agents. However, existing web agents often rely on handcrafted execution pipelines or expensive expert trajectories, limiting their adaptability to complex, dynamic environments. To address these challenges, we propose SCALE (Self-Cognitive-Aware Learning and Exploration), which leverages three adversarial roles, Selector, Predictor, and Judger to autonomously discover the agent's limitations and expand its cognitive boundaries through environmental exploration. Moreover, we propose SCALE-Hop, a graph exploration strategy that facilitates global planning and helps agents avoid local exploration traps. To further support learning, we construct SCALE-20k, a large-scale dataset collected from 19 real-world websites, containing diverse task types and structured demonstrations generated from SCALE's exploration traces. Experimental results show that our approach significantly improves the performance and generalization of multiple MLLMs in various web environments. Our framework offers a scalable and generalizable solution for building truly autonomous and adaptive web agents. 

---
# Diagnosing Failure Modes of Shared-State Collaboration in Resource-Constrained Visual Agents 

**Authors**: Yunpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.31354)  

**Abstract**: Modular visual reasoning systems increasingly rely on shared working memory for multi-step collaboration, yet the failure dynamics of intermediate state evolution in low-capacity regimes remain underexplored. We study failure modes of collaborative reasoning with weak learners (4B--8B models) through the lens of noise accumulation. We introduce CoSee, an auditing framework that formalizes the read-write-verify loop to trace information flow in document visual question answering. Across multi-page, chart, and web-based benchmarks, we find a counter-intuitive degradation: naive shared workspaces often amplify hallucinations rather than resolve them. We identify two dominant failure modes: Noise Reinforcement, where ungrounded notes are reused as evidence, and Policy Collapse, where added context shifts the model toward under-specified, short-form answers. Using cost-accuracy Pareto frontiers, we show that increased compute can correlate negatively with performance without explicit verification. Our findings suggest that for resource-constrained agents, the bottleneck lies not in reasoning depth but in communication fidelity, providing trace-level diagnostics and a mechanistic baseline for reliable modular design. 

---
# TraceGraph: Shared Decision Landscapes for Diagnosing and Improving Agent Trajectories 

**Authors**: Junjie Nian, Kang Chen, Ge Zhang, Yixin Cao, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31308)  

**Abstract**: Agent benchmarks increasingly record rich interaction trajectories, yet evaluation often reduces each rollout to a pass rate or reward score. We introduce TraceGraph, a graph-based framework that turns released multi-model agent trajectories into shared decision landscapes. For each task, TraceGraph builds a graph over observable action-observation states from pooled rollouts before model identity is introduced. It then overlays outcome-informed productive cores and trap regions, and summarizes each rollout with three events: Access, Trap exposure, and Repair. Across trajectories spanning five benchmark splits, TraceGraph profiles reveal navigation differences hidden by aggregate scores and show that splits differ in whether they reward avoiding traps or recovering from them. The same TraceGraph landscape also motivates a trap-aware recovery pipeline for SWE-bench: aruntime detector fires on states matching historical trap regions, then lightweight continuation policies are evaluated from the same prefix. On fired states, the best pooled single-factor policy raises official resolved rate from 40.4% to 43.5% on the per-provider fired subset and from 41.0% to 44.8% on common-fired instances, with provider-specific active components. Overall, TraceGraph provides a process vocabulary for asking what agent benchmarks test, where models diverge on a shared landscape, and how failure regions can guide downstream improvement. 

---
# Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation 

**Authors**: Grégoire Martinon, Ibrahim Merad, Mohammed Raki  

**Link**: [PDF](https://arxiv.org/pdf/2605.31278)  

**Abstract**: Reliable evaluation of agentic systems requires unbiased estimates with valid uncertainty, but standard practice navigates between costly human annotation and biased LLM-as-judge proxies. Prediction-powered inference (PPI) combines both into debiased estimates with valid confidence intervals, yet its various methods remain scattered across papers under partial implementations. We introduce GLIDE, an open-source Python library that unifies state-of-the-art PPI estimators (PPI++, Stratified PPI, Predict-Then-Debias and its stratified variants, Active Statistical Inference) and samplers (uniform, stratified, active, cost-optimal) under a scipy-style API specialized to mean estimation. GLIDE ships with a reproducible Monte Carlo validation suite, an empirically grounded decision tree for method selection, and an agentic evaluation case study showing substantial annotation savings at equivalent precision. The GLIDE package is available at this URL: this https URL 

---
# COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation 

**Authors**: Tianyi Zhou, Dongrui Liu, Leitao Yuan, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31264)  

**Abstract**: LLM agents are increasingly expected not only to complete isolated tasks, but also to carry bounded representations of human expertise, judgment, and interaction style. Building such person-grounded agents remains difficult because actionable knowledge associated with a person or role is usually embedded in heterogeneous traces rather than written as clean instructions. Existing memory and persona systems capture fragments of this evidence, while skill frameworks provide portable packaging formats; however, there is no end-to-end workflow for distilling these traces into inspectable, correctable, and agent-usable skills. We present an automated trace-to-skill distillation system for generating person-grounded AI skills via expert knowledge distillation. Given materials from a target person or role, this http URL produces a versioned skill package with two coordinated tracks: a capability track for practices, mental models, and decision heuristics, and a bounded behavior track for communication style, interaction rules, and correction history. The package can be inspected, invoked, updated through natural-language feedback, rolled back, installed across agent hosts, and optionally prepared for controlled distribution. We describe the artifact contract, generation workflow, correction lifecycle, deployment surface, and domain presets implemented in the open-source system. At the time of writing, the public repository has approximately 18.5k GitHub stars; the gallery lists 215 skills from 165 contributors and more than 100k cumulative stars across listed skill cards. The system illustrates how person-grounded skills can be represented as portable, correctable packages rather than opaque prompts or hidden memories. 

---
# Formalizing and falsifying causal pathways of rare events 

**Authors**: Anahita Haghighat, Dominik Janzing  

**Link**: [PDF](https://arxiv.org/pdf/2605.31254)  

**Abstract**: Building on recent formalizations of root cause analysis for rare events (``outliers'') in structural equation models, we propose a formal definition of a causal pathway and discuss its testable implications. We identify conditions under which these implications depend only on a causal abstraction defined by the pathway of rare events, rather than on the full causal graph of the underlying system. Accordingly, we introduce an abstraction of causal structure to pathways of rare events that bridges simple verbal causal explanations and detailed causal modeling. 

---
# LLM-FACETS: A Privacy-Preserving Framework for Evaluating LLM Transparency and Accountability 

**Authors**: Tom Lucas, Alessio Buscemi, Alfredo Capozucca, German Castignani, Barbara Delacroix  

**Link**: [PDF](https://arxiv.org/pdf/2605.31167)  

**Abstract**: Assessing whether Large Language Models outputs are factually grounded, epistemically calibrated, and methodologically reproducible is a prerequisite for responsible AI deployment. Yet auditing LLMs remains inaccessible to non-technical practitioners: existing tools require programming expertise and non-trivial environment setup, and cloud-hosted platforms transmit evaluation data to external services, creating barriers for domain experts and compliance officers legally responsible for AI oversight. We introduce LLM-FACETS (LLM FActuality Cross-EvaluaTion System): an open-source framework with a browser-accessible interface and a plugin architecture, structured around three practitioner profiles (technical experts, domain experts, compliance officers) that mirror the stakeholder categories identified in the EU AI Act and the NIST AI Risk Management Framework. The architecture makes data flows explicit: deterministic metrics (BLEU, ROUGE, BERTScore) run entirely within the self-hosted server with no outbound transmission; LLM-judge metrics contact external APIs explicitly, with users retaining full credential control. The framework operationalizes transparency through three mechanisms: token-level log-probability visualization for epistemic uncertainty, multi-judge consensus to mitigate judge bias, and RAG Triad metrics (Faithfulness, Answer Relevance, Context Relevance) to detect and localize hallucinations. A plugin architecture allows any new metric or dataset to be integrated without modifying the evaluation pipeline. The open-source implementation enables cross-checking across multiple metrics targeting the same property, ensuring reproducibility and decoupling AI accountability from the teams building the systems assessed. We verify the framework through cross-validation of 18 metric implementations against canonical reference libraries. 

---
# Vector Linking via Cross-Model Local Isometric Consistency 

**Authors**: Ziying Chen, Yang Cao, He Sun, Beining Yang, Tianjian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31100)  

**Abstract**: We study Vector Linking: given two embedding clouds produced by different black-box encoders over partially overlapping datasets, recover cross-model object correspondences using only vectors. Empirically and theoretically, we show that independently trained contrastive encoders exhibit local geometric consistency: short-range distances are approximately preserved up to a scale factor, while long-range distances are not due to model-specific distortion. Building on this, we propose an iterative, reference-based geometric embedding hashing that recovers vector links from a tiny seed set of paired anchors. It represents each vector by distances to sampled paired anchors, proposes candidate links via hash-space matching, and aggregates evidence across views in a Beta-Bernoulli posterior to bootstrap high-confidence links as new anchors. Experiments across multiple benchmarks and embedding model pairs demonstrate accurate and robust linking under varying overlap, seed budgets, and out-of-domain anchors, with applications to vector database integration and cross-model clustering. Code is available at this https URL. 

---
# GraphARC: A Comprehensive Benchmark for Graph-Based Abstract Reasoning 

**Authors**: Saku Peltonen, August Bøgh Rønberg, Andreas Plesner, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2605.31031)  

**Abstract**: Relational reasoning lies at the heart of intelligence, but existing benchmarks are typically confined to formats such as grids or text. We introduce GraphARC, a benchmark for abstract reasoning on graph-structured data. GraphARC generalizes the few-shot transformation learning paradigm of the Abstraction and Reasoning Corpus (ARC). Each task requires inferring a transformation rule from a few input-output pairs and applying it to a new test graph, covering local, global, and hierarchical graph transformations. Unlike grid-based ARC, GraphARC instances can be generated at scale across diverse graph families and sizes, enabling systematic evaluation of generalization abilities. We evaluate state-of-the-art language models on GraphARC and observe clear limitations. Models can answer questions about graph properties but often fail to solve the full graph transformation task, revealing a comprehension-execution gap. Performance further degrades on larger instances, exposing scaling barriers. More broadly, by combining aspects of node classification, link prediction, and graph generation within a single framework, GraphARC provides a promising testbed for future graph foundation models. 

---
# HADT: A Heterogeneous Multi-Agent Differential Transformer for Autonomous Earth Observation Satellite Cluster 

**Authors**: Mohamad A. Hady, Muhammad Anwar Masum, Siyi Hu, Mahardhika Pratama, Jimmy Cao, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2605.31023)  

**Abstract**: This work addresses the problem of autonomous resource management in heterogeneous satellite cluster conducting Earth Observation (EO) missions including optical and Synthetic Aperture Radar (SAR) satellites. In autonomous operation mode, satellites are equipped with intelligent capabilities enabling real-time decision-making based on the latest conditions, while requiring minimal interaction with ground operators. Traditional scheduling approaches typically rely on mathematical models to represent satellite mission and resource management. Then, this problem is solved by using optimization algorithms. However, such solutions become less effective when the underlying models are not available, over complex, and inaccurate due to dynamic changes and uncertainties inherent in the space mission environment. A promising alternative is to reformulate the problem as a sequential decision-making process and apply model-free reinforcement learning techniques to enable adaptive and real-time resource management. To this end, we propose a novel transformer-based architecture tailored for heterogeneous satellite cluster autonomous EO Mission with relational observations-actions tokenization and differential attention mechanism. Our experimental results demonstrate significant performance improvements compared to the available baselines. Moreover, the proposed architecture exhibits strong adaptability and transferability with respect to varying numbers of satellite clusters. 

---
# A Persona-Based Evaluation Framework for Pluralistic Alignment in Generative AI 

**Authors**: Atahan Karagoz  

**Link**: [PDF](https://arxiv.org/pdf/2605.31021)  

**Abstract**: Current alignment paradigms for generative artificial intelligence rely predominantly on monolithic benchmarking frameworks that reduce the plurality of human judgment to aggregated statistical baselines, thereby obscuring cultural, demographic, and contextual variability in evaluation. We introduce a state-space constrained emulation framework for AI evaluation that replaces singular assessment functions with a structured manifold of synthetic cognitive profiles representing diverse human perspectives. We show that modern generative architectures can instantiate and maintain these evaluative personas with high consistency, enabling a form of pluralistic, perspective-dependent benchmarking that more closely reflects real-world consensus variability. However, we further analyze the stability of these simulated evaluators under sequential inference and stochastic prompt perturbations, revealing systematic degradation in persona coherence that manifests as state-space drift and semantic inconsistency. These findings suggest that static alignment constraints are insufficient for sustaining robust evaluative behavior over time. Instead, we argue for the necessity of embedding dynamic, viability-driven regulatory mechanisms within generative systems to preserve coherent cognitive emulation. By framing persona-based evaluation as a structured dynamical system over latent representation manifolds, this study provides a foundation for more adaptive, human-aligned, and context-sensitive approaches to AI evaluation. 

---
# BilliardPhys-Bench: Benchmarking Physical Reasoning and Visual Dynamics of Multimodal LLMs 

**Authors**: Ben Wang, Xiaogang Li, Ruochen Gao, Peiyao Xiao, Chengliang Xu, Zeyu Wang, Zichao Chen, Bing Zhao, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.30900)  

**Abstract**: Current multimodal models handle static image recognition well, but intuitive physical reasoning remains a weakness. Predicting how objects will move and interact from a single image is still difficult for these systems. We present BilliardPhys-Bench, a benchmark for physical reasoning in synthetic billiards environments. Its procedural engine generates randomized scenarios with friction and elastic collisions. The benchmark tests three abilities: (1) predicting ball-to-ball collisions, (2) reasoning about wall bounces, and (3) estimating final ball positions after motion stops. We evaluate recent MLLMs from the GPT, Claude, Gemini, and Qwen families. Performance drops as simulation time increases and scene geometry grows more complex. We also observe a consistent failure mode we call "stasis bias": when the correct physical outcome is harder to infer, models tend to predict no interaction. These findings show where current MLLMs break down on visual dynamics and point toward the need for better physical inductive biases in multimodal architectures. 

---
# UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-Time Scaling 

**Authors**: Kaiyu Huang, Xingyu Wang, Mingze Kong, Zhubo Shi, Yuqian Hou, Hong Xu, Zhongxiang Dai, Minchen Yu, Qingjiang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30898)  

**Abstract**: In real-world deployments of large language models (LLMs), balancing inference quality and computational cost has become a central challenge. Existing approaches tackle this trade-off along two largely independent dimensions: model routing, which switches among models of different scales to match request complexity, and test-time scaling (TTS), which adjusts inference-time compute within a fixed model for fine-grained control. However, this decoupled design introduces inherent limitations. Model routing yields coarse-grained, discrete performance changes due to the sparse set of model scales, while single-model TTS often encounters capacity ceilings and exhibits diminishing returns as compute increases. Moreover, treating the two mechanisms separately restricts adaptability in dynamic inference environments. To overcome these limitations, we introduce Unified Inference Scaling (UIS), which unifies model routing and TTS in a single optimization space. Building on this formulation, we propose UniScale, an online framework that models adaptive UIS as a contextual multi-armed bandit problem and learns inference policies via LinUCB. The framework incorporates efficiency-aware learning and cost modeling to ensure stable and scalable optimization over high-dimensional action spaces. Evaluation shows that UniScale effectively exploits the synergy in the UIS space to deliver a fine-grained and consistently better quality-cost trade-off across diverse, dynamic inference scenarios. 

---
# Distilling LLM Feedback for Lean Theorem Proving 

**Authors**: Gaetan Narozniak, Gérard Biau, Rémi Munos, Ahmad Rammal, Pierre Marion  

**Link**: [PDF](https://arxiv.org/pdf/2605.30861)  

**Abstract**: Post-training for reasoning models typically combines supervised fine-tuning with reinforcement learning from verifiable rewards, most commonly with GRPO. However, this algorithm suffers from sparse rewards, limited exploration, and mode collapse. Building upon recent works on self-distillation, we propose Feedback Distillation, a training method where the model is trained to match, at the token level, its own distribution conditioned on privileged feedback produced by a language model. Feedback Distillation offers token-level supervision and can inject external knowledge. Evaluating our method for Lean4 theorem-proving, we find that Feedback Distillation maintains greater diversity in generated trajectories than GRPO, yielding higher policy entropy and better pass@k scaling. The two methods are complementary: initializing GRPO from a Feedback Distillation checkpoint outperforms either method alone. All in all, our results suggest a promising avenue to improve post-training for complex reasoning. 

---
# COMPASS: Cognitive MCTS-Guided Process Alignment for Safe Search Agents 

**Authors**: Wenkai Shen, Pengyang Zhou, Jiahe Xu, Jiaming Qian, Haozhe He, Zhihao Huang, Chaochao Chen, Xiaolin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.30838)  

**Abstract**: LLM-powered search agents enable multi-step reasoning and tool use. However, these capabilities introduce retrieval-induced safety degradation, as harmful intents may decompose into seemingly innocuous sub-queries that lead to unsafe outcomes. Existing alignment methods struggle to capture sparse safety signals and fail to supervise diverse violations across multi-step interactions. We propose COMPASS, a Cognitive MCTS-Guided Process Alignment framework designed to achieve robust safety alignment throughout the agent workflow while preserving general utility. COMPASS integrates cognitive tree exploration (CTE) to efficiently synthesize stealthy attack trajectories, and introspective step-wise alignment (ISA) to isolate risky intermediate actions for fine-grained process supervision. Empirical results show that COMPASS achieves a favorable safety-utility trade-off while requiring substantially less training data. 

---
# SLAT: Segment-Level Adaptive Trimming for Efficient CoT Reasoning 

**Authors**: Jian Yao, Xiongcai Luo, Ran Cheng, Kay Chen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30832)  

**Abstract**: Recent advances in Large Reasoning Models have significantly improved chain-of-thought (CoT) capabilities via reinforcement learning (RL). However, generated reasoning chains frequently suffer from structural redundancy (i.e., \emph{overthinking}), incurring high computational overhead without improving answer correctness. Existing mitigation strategies typically rely on token-uniform length penalties, which provide coarse, segment-agnostic pressure toward shorter outputs and can inadvertently suppress useful reasoning alongside redundancy. To address this, we demonstrate that inefficiency concentrates in high-probability segments with low marginal utility. We derive a theoretical characterization of segment suboptimality under the correctness-length trade-off objective and propose \textsc{SLAT} (Segment-Level Adaptive Trimming), an RL framework that selectively suppresses redundant segments based on this criterion. Empirical results on standard benchmarks indicate that \textsc{SLAT} establishes a superior accuracy-efficiency Pareto frontier, reducing reasoning length by $50\%$ relative to uncompressed baselines while maintaining competitive accuracy. Overall, our results suggest that theoretically grounded, segment-aware trimming is a promising direction for efficient CoT reasoning in large language models. 

---
# Planner-Centric Reinforcement Learning for Deep Research with Structure-Aware Reward 

**Authors**: Mustafa Anis Hussain, Xinle Wu, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30824)  

**Abstract**: Deep research tasks require LLMs to plan what to investigate, retrieve evidence, and synthesize long-form answers across multiple branches of inquiry. Existing training paradigms either rely on short-form verifiable QA as a proxy or optimize monolithic long trajectories, which makes planning and execution difficult to disentangle and yields weak credit assignment for the planning process. We propose DecomposeR, a planner-centric deep research framework that represents research plans as typed directed acyclic graphs (DAGs), allowing planning to be made explicit, structured, and rewardable. We train a Qwen3-8B model in two stages: planner reinforcement learning (RL) first learns graph structure and query decomposition to improve research planning, and answerer reinforcement learning (RL) then learns branch-level execution and final synthesis conditioned on the learned plan. By assigning rewards to explicit planner tokens and structured components rather than to a flat trajectory, DecomposeR enables finer-grained optimization of planning while reducing the ambiguity of end-to-end training. Experiments show that DecomposeR-8B improves over strong comparable open baselines by 5.1-8.0 points on popular long-form benchmarks due to improved planning and answering capabilities. 

---
# PReMISE: Policy Rubrics as Measurement Specifications for LLM Judges 

**Authors**: Swastik Roy, Rajkumar Pujari, Tharindu Kumarage, Charith Peris, Rahul Gupta, Anna Rumshisky, Pradeep Natarajan, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2605.30803)  

**Abstract**: LLM judges are increasingly used to evaluate open-ended responses, but their scores depend strongly on the rubrics that condition them. A vague rubric asking for a response to be ``helpful and factual'' can reward polished answers that invent facts or violate user intent. We treat reusable rubrics as measurement specifications: changing the rubric changes the response quality measurement induced by a fixed judge. We introduce PReMISE, a framework that, given pairwise human-preference data, (i) discovers a policy-level rubric set, and (ii) audits any rubric set under LLM-judge use along four axes: structural adequacy, reliability, preference fit, and adversarial robustness. Across rubric sources no raw source is simultaneously reliable, preference-predictive, and adversarially robust; and high inter-rater agreement does not imply low exploitability. PReMISE is the only rubric source to score non-trivially on applicability, specificity, and effective dimensionality simultaneously. We contribute two audit-targeted repair operations: preference-rank selection raises judge accuracy on paired responses from $65.0\%$ to $68.6\%$, competitive with the strongest rubric-discovery baselines and leading on two of three judges in our cross-judge sweep; reliability-constrained refinement reduces the rate at which exploit responses receive high scores from $46.4\%$ to $36.0\%$ with little change in inter-judge agreement ($\alpha{=}.531\to.519$). 

---
# Learning Agent-Compatible Context Management for Long-Horizon Tasks 

**Authors**: Lu Yi, Runlin Lei, Liuyi Yao, Yuexiang Xie, Yuyang Li, Wenhao Zhang, Zhewei Wei, Yaliang Li, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.30785)  

**Abstract**: LLM agents increasingly face long-horizon tasks such as web search and deep research in real-world applications, where accumulated context can cause long-context degradation and reasoning failures. Prior work mitigates this through context management with agent-side context control or fixed strategies such as summarization, which require training the agent itself for adaptation - making it impractical for closed-source agents and ignoring that different agents may require different strategies. We introduce Adaptive Context Management (AdaCoM), which trains an external LLM to manage the context of a frozen agent through flexible modification actions and end-to-end reinforcement learning. Across diverse agents on web search and deep research benchmarks, AdaCoM substantially improves performance by preserving task constraints and progress while pruning stale content. The learned strategies reveal a Fidelity-Reliability Trade-off: agents with higher vanilla ReAct performance benefit from higher-fidelity context preservation, whereas lower-performing agents require more aggressive compression to stay within a reliable reasoning regime. Transfer experiments show that AdaCoM generalizes most effectively across agents with similar capability (measured by vanilla ReAct performance), suggesting a practical path toward reusable context managers for agent systems. 

---
# Generating Graph-like Rules for Knowledge Graph Reasoning via Diffusion Models 

**Authors**: Haoxiang Cheng, Yunfei Wang, Chao Chen, Kewei Cheng, Zhipeng Lin, Haoxuan Li, Changjun Fan, Shixuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30747)  

**Abstract**: Logical rules constitute a cornerstone of knowledge graph (KG) reasoning, valued for their interpretability and ability to model relational patterns. However, existing rule mining methods predominantly focus on simple chain-like rules and therefore neglect the richer relational information encoded in graph-like structures, such as cycles and branches. This limitation is further exacerbated by computational bottlenecks caused by the combinatorial explosion of the search space, which is especially challenging for graph-like rules. Meanwhile, generative approaches such as diffusion models, despite their success in other domains, can not be directly applied to rule mining because their training objectives are not aligned with the goal of learning high-quality rules, and non-differentiable KG rule quality metrics cannot directly guide model optimization. To address these limitations, we propose GRiD, a framework that reformulates graph-like rule discovery as a discrete generative process conditioned on the target relation. GRiD employs a two-phase training strategy. First, supervised pre-training enables GRiD to capture structural priors from subgraphs sampled from the KG meta-graph. Subsequently, reinforcement learning is applied to fine-tune GRiD through policy gradient optimization guided directly by non-differentiable rule-quality metrics. Experiments on six benchmark datasets show that GRiD achieves competitive performance on KG completion tasks. Ablation studies confirm the efficiency and robustness of GRiD and further show that graph-like rules complement chain-like rules in KG completion. Our codes and datasets are available in this https URL 

---
# MAVEN: Improving Generalization in Agentic Tool Calling 

**Authors**: Omkar Ghugarkar, Vishvesh Bhat, Muhammad Ahmed Mohsin, Asad Aali  

**Link**: [PDF](https://arxiv.org/pdf/2605.30738)  

**Abstract**: Generalization across agentic tool-calling environments remains a central challenge for reliable agentic reasoning systems. Although large language models achieve strong results on individual benchmarks, their ability to compose reasoning strategies, preserve intermediate states, and coordinate tools across domains remains underexplored. We present MAVEN (Modular Agentic Verification and Execution Network), a lightweight symbolic reasoning scaffold for structured decomposition, adaptive tool orchestration, and intermediate verification. We evaluate MAVEN across established tool-calling benchmarks, including BFCL v3, TauBench, Tau2Bench, AceBench, and introduce MAVEN-Bench, a stress-test benchmark for multi-step mathematical and physical reasoning with explicit verification and adversarial task composition. MAVEN-Bench exposes a substantial gap between partial reasoning quality and end-to-end task success; in direct MAVEN-Bench runs, MAVEN improves its GPT-OSS-120b base model from 48% to 71% accuracy without additional training. It also remains competitive with frontier proprietary baselines while using an open-weight backbone with an estimated cost ratio of roughly 1/10, suggesting that lightweight verification-centered scaffolds can strengthen compositional reasoning and motivate more process-aware evaluation of agents in the wild. 

---
# Healthcare Mechanisms from Policy-as-Code Search under Strategic Provider Response 

**Authors**: Zihan Wang, Xiang Xu, Hongyuan Zha, Wenhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30680)  

**Abstract**: Healthcare mechanisms are inseparable from the strategic provider response they induce: existing healthcare AI benchmarks hold this response fixed and so cannot evaluate mechanisms by the equilibrium they produce. We recast hospital mechanism design as program synthesis for language models: typed, inspectable rule programs are executed and scored by Medi-Sim, a multi-agent simulator with five strategic provider channels (coding, selection, delay, effort, triage). An incentive sweep recovers classical health-economics findings as adjacent regimes -- up-coding and low-complexity-patient selection under profit pressure, and Goodhart-style drift where measured performance becomes anti-correlated with true outcomes -- and a single audit lever exposes pressure migration: closing the coding channel more than doubles low-complexity selection. LLM-guided evolutionary code search over the same rule-program space then synthesizes an inspectable mixed-objective program that eliminates up-coding, halves rejection, and retains most of the profit-oriented baseline's funds. 

---
# Structure-Induced Information for Rerooting Levin Tree Search 

**Authors**: Jake Tuero, Michael Buro, Laurent Orseau, Levi H. S. Lelis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30664)  

**Abstract**: Subgoal-based policy tree search, which uses a policy to guide search, is effective for complex single-agent deterministic problems but often relies on explicit subgoal generation that can incur substantial overhead and hinders scalability. In this paper, we overcome these limitations by using a learned ``rerooter'' through the recently-introduced $\sqrt{\text{LTS}}$ algorithm. A rerooter implicitly decomposes the problem into soft subtasks. While previous work focused on the formal guarantees for given or handcrafted rerooters, in this work we propose three rerooter designs: (i) a clustering-based rerooter that exploits global state-space structure, (ii) a heuristic-based rerooter that leverages learned cost-to-go estimates, and (iii) a hybrid that combines both signals. Our framework avoids having to explicitly reconstruct and reason over generated subgoals, thereby enabling scalable allocation of search effort with significantly lower computational overhead. Empirically, our rerooting-based methods scale to complex environments where subgoal-based policy tree search fails, and achieve state-of-the-art online training efficiency on the domains tested. 

---
# EHRBench: An Automated and Reliable EHR-based Benchmark for Clinical Decision Making with LLMs 

**Authors**: Yuzhang Xie, Keqi Han, Yunpeng Xiao, Hejie Cui, Guanchen Wu, Ziyang Zhang, Kai Shu, Jiaying Lu, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30637)  

**Abstract**: Clinical decision-making (CDM) is central to real-world clinical workflows, where clinicians infer diagnoses, select treatments, or anticipate future health outcomes under incomplete evidence. LLMs are increasingly used to support these decisions due to strong language capabilities, broad biomedical knowledge, and efficiency, yet the reliability of LLMs on real-world clinical decision tasks remains insufficiently understood. To evaluate CDM models, especially LLM-based models, an ideal and practical medical decision benchmark should be constructed via an automated yet reliable pipeline to ensure both scale and quality. Moreover, the grounding of a CDM benchmark in real patient EHRs can better support evaluation on practical CDM tasks that require substantive biomedical knowledge and clinical inference. To fill the gaps, we introduce EHRBench, an automated and reliable EHR-grounded benchmark for evaluating LLM-based clinical decision-making at scale. To ensure scalability and reliability, EHRBench is constructed through an EHR-LLM-KB(knowledge-base) interaction pipeline. For efficiency, we use a specialized LLM to automatically convert encounter-level EHR trajectories into structured templates and deterministically instantiate the templates into QA items. In parallel, we apply systematic KB-based verification and enrichment to filter hallucinated or ambiguous relations and to improve reliability. Using this pipeline, we construct nearly 1M (960,067) QA items spanning three core inference-required clinical decision tasks: diagnosis, treatment, and prognosis. We benchmark more than 30 representative LLMs on EHRBench and provide detailed analyses of performance and robustness. The results show consistent capability trends across settings, further validating the reliability of EHRBench and highlighting actionable gaps toward clinically reliable LLM systems. 

---
# Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents 

**Authors**: Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, Hanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30621)  

**Abstract**: LLM agents are increasingly deployed as systems built around editable external harnesses, including prompts, skills, memories and tools, that shape task execution without changing model parameters. Harness self-evolution adapts such agents by updating these harnesses from execution evidence. Yet it remains unclear whether a model's base capability in task-solving predicts its capabilities in harness self-evolution: which models produce useful harness updates, and which actually benefit from them? We analyze two harness self-evolution capabilities: (i) harness-updating, the capability to produce useful persistent harness updates from execution evidence; (ii) harness-benefit, the capability to benefit from updated harnesses during task solving. Our analysis reveals two findings. First, harness-updating is flat in base capability: models from different capability tiers produce harness updates that lead to surprisingly similar gains; even Qwen3.5-9B's updates yield gains comparable to those of Claude Opus~4.6. Second, harness-benefit is non-monotonic in base capability: weak-tier models benefit little from updated harnesses, mid-tier models benefit most, and strong-tier models benefit less than mid-tier. We trace low gains at the weak tier to two failure modes: weak-tier models may fail to activate relevant harness artifacts, or activate them but fail to follow them faithfully. These findings suggest investing capability budget in the task-solving agent rather than the evolver, and targeting harness invocation and long-horizon instruction following in agent training. Our source code is publicly available at this https URL. 

---
# Uncertainty-Aware and Temporally Regulated Expert Advice in Reinforcement Learning for Autonomous Driving 

**Authors**: Ahmed Abouelazm, Felix Klingebiel, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2605.30576)  

**Abstract**: Exploration in reinforcement learning for autonomous driving is inherently unsafe: agents must experience novel behaviors to learn, yet exploration can lead to collisions or off-road driving. We propose an uncertainty-aware framework that leverages expert advice to guide exploration while avoiding long-term dependence. Advice is triggered when epistemic or aleatoric uncertainty exceeds adaptive thresholds derived from rolling buffers, ensuring advice evolves with the agent's confidence. A commitment-cooldown strategy with a stochastic early-stop heuristic regulates the duration and frequency of guidance, exposing the agent to coherent maneuvers without exhausting the advice budget. Expert and agent experiences are combined in a shared replay buffer within an off-policy implicit quantile network (IQN) backbone, enabling efficient reuse of expert trajectories. Experiments in CARLA show that our method outperforms the IQN baseline, improving success by 5-7% and reducing failures, demonstrating that risk-sensitive uncertainty coupled with regulated expert integration enables safer and more efficient exploration for sensor-based RL policy learning in unsignalized intersection navigation. 

---
# Procedural Generation of First Person Shooter Maps using Map-Elites 

**Authors**: Simone de Donato, Pier Luca Lanzi, Daniele Loiacono  

**Link**: [PDF](https://arxiv.org/pdf/2605.30570)  

**Abstract**: We investigate the application of MAP-Elites (a well-known quality diversity algorithm) to design levels for First-Person Shooter (FPS) games. We consider two well-known map representations (All-Black and Grid-Graph) and introduce two novel representations (Point-Line and Spatial-Layout) that improve the characterization of FPS maps. We define a series of metrics to describe maps' topological properties (which solely depend on maps' layout), and emergent properties (which must be evaluated through actual gameplay). We perform an in-depth analysis to identify the most suitable features to guide MAP-Elites illumination process. We apply MAP-Elites with Sliding Boundaries (MESB) to evolve populations of FPS maps. Our results show that the new representations can generate maps with higher diversity and quality than the representations previously used for evolving FPS maps. 

---
# Transforming and Encoding FTS for SAT Solving: What Helps, What Hurts (Extended Version) 

**Authors**: João Filipe, Álvaro Torralba, Gregor Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2605.30563)  

**Abstract**: Factored tasks are a classical planning representation that extends SAS+ with limited forms of disjunctive preconditions, conditional effects, and angelic nondeterminism. This allows for a more compact representation of tasks than traditional formalisms such as STRIPS or SAS+, and supports a wide range of task transformations. However, existing planning approaches for factored tasks have been limited to heuristic search methods.
In this work, we investigate how to encode factored tasks in SAT. We propose several ways to encode the tasks, focusing on different strategies for translating the factored transition relation into propositional logic. We also analyze how to exploit parallelism at various levels in this setting and study the impact of common task transformations on the performance of SAT-based planners. 

---
# Physically Viable World Models: A Case for Query-Conditioned Embodied AI 

**Authors**: Adam J. Thorpe, Stepan Tretiakov, Cheng-Hsi Hsiao, Su Ann Low, Xingjian Li, Hassan Iqbal, Neel P. Bhatt, Ufuk Topcu, Krishna Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.30542)  

**Abstract**: World models for embodied AI must be physically viable: constructed to answer intervention queries by representing the physical structure governing action outcomes, rather than merely predicting future observations. Existing observation-predictive world models can produce visually plausible but physically wrong rollouts. This failure is structural; distinct physical systems can look identical yet diverge under intervention. We expose this problem with controlled benchmarks that fix the visible scene while varying latent physics. We show that such models may recommend infeasible actions, mispredict interaction outcomes, or certify unsafe behavior. We argue that embodied AI requires world models that identify the simplest physical abstraction sufficient to answer an intervention query. Such a model comprises modular components, including environment representation, latent state and parameter estimation, action specification, interventional dynamics, and query-level response. An autonomous orchestrator should identify the relevant abstraction and compose compatible learned and structured components per query. When closed-form physics is unavailable, uncertain, or costly, the transition model may be analytic, simulated, learned, or hybrid, but it must preserve the structure that determines interventional outcomes. This decomposition makes the model interpretable, its components verifiable, and its outputs auditable against the query. It also provides a design principle for new world models and a feasibility test for existing ones: the right abstraction is not the most detailed model of the world, but the simplest model that preserves the distinctions relevant to the query. We demonstrate this approach on queries that existing systems fail to answer correctly, and outline how an orchestrator can dynamically assemble and adapt physically viable models for planning, control, and verification. 

---
# PhyDrawGen: Physically Grounded Diagram Generation from Natural Language 

**Authors**: Nafiul Haque, Syed Nazmus Sakib, Shifat E Arman  

**Link**: [PDF](https://arxiv.org/pdf/2605.30512)  

**Abstract**: Generating physics diagrams from text requires strict adherence to physical laws. While current generative models produce visually plausible outputs, they systematically hallucinate force vectors, ignore conservation laws, and violate geometric constraints. We present PhyDrawGen, a neuro-symbolic pipeline that decouples semantic scene understanding from physical constraint satisfaction. First, a large language model extracts a typed scene graph from the problem text. A deterministic solver then converts this graph into a Planar Straight-Line Graph (PSLG), encoding force balance, optical paths, and field topologies as exact geometric primitives. Finally, a fine-tuned Qwen-VL model implements a visually grounded propose-verify loop to iteratively correct any constraint violations. Evaluated on a benchmark of 1,449 problems spanning mechanics, optics, and electromagnetism, PhyDrawGen significantly outperforms GPT-5-image, Gemini 2.5 Flash, and Gemini 3 Pro, demonstrating robust physical accuracy even on unusual-object problems. 

---
# Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models 

**Authors**: Jiazheng Xing, Hangjie Yuan, Lingling Cai, Xinyu Liu, Yujie Wei, Fei Du, Hai Ci, Tao Feng, Jiasheng Tang, Weihua Chen, Fan Wang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31603)  

**Abstract**: Connector-based video unified models have demonstrated strong capability in instruction-grounded video synthesis, but integrating a large high-fidelity generator into the unified training loop is computationally prohibitive, limiting achievable visual quality. We therefore propose Lumos-Nexus, a training-efficient unified video generation framework that facilitates the development of strong reasoning-driven generation capabilities while significantly enhancing visual fidelity. Lumos-Nexus adopts a two-stage design: 1) During training, only a lightweight generator is aligned with the understanding block to learn to take in reasoning-driven semantic control. 2) During inference, we introduce Unified Progressive Frequency Bridging (UPFB) to progressively hand off generation to a high-capacity pretrained generator in the shared latent space, enabling coarse-to-fine refinement and producing high-fidelity videos without compromising reasoning quality. To fill the gap in reasoning-driven video generation benchmarks, we introduce VR-Bench, which assesses a model's capability to translate inferred intent into coherent and semantically aligned video content. Extensive experiments demonstrate that Lumos-Nexus achieves substantial gains in visual realism and temporal coherence on VBench, while exhibiting strong reasoning-based generative performance on VR-Bench. Code and models are available at this https URL. 

---
# Stateful Online Monitoring Catches Distributed Agent Attacks 

**Authors**: Davis Brown, Samarth Bhargav, Arav Santhanam, Kasper Hong, Ivan Zhang, Matan Shtepel, Steffi Chern, Alexander Robey, Eric Wong, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2605.31593)  

**Abstract**: Language models can find thousands of severe software vulnerabilities, and agents are increasingly being misused for cyberattacks. To avoid detection, attackers frequently distribute their misuse, splitting a harmful task across many user accounts so each individual transcript looks benign. Because safety monitors score only one agent context at a time, they are structurally blind to misuse that is only visible in aggregate, across many accounts. We show this gap is real by building, to our knowledge, the first distributed agent attack, a multi-agent scaffold that completes hard cybersecurity tasks while hiding the harmful objective across subagents with limited contexts, evading a standard monitor that catches it only a fifth as often as prior agent attacks. Towards a defense, we develop an online stateful monitor that uses real-time clustering to collect weak suspiciousness signals across many agent transcripts, and escalates only rarely to a language model that flags misuse across user accounts. In evaluations with large-scale simulated datacenter traffic, our monitor Pareto dominates standard monitors, catching distributed attacks 30% earlier and flagging cyber misuse before it reaches the most harmful stages. Crucially, this comes at negligible additional latency for ~99% of user traffic. This detection advantage persists but narrows as the benign background traffic grows very large. After an extensive red-teaming exercise, we improve the defense and surprisingly also find that it catches standard jailbreaks, since adaptive attackers reuse attack variants across accounts. Our results point toward a new class of safety monitors which reason over groups of users rather than isolated transcripts. 

---
# TunerDiT: Training-free Progressive Steering of Diffusion Transformer for Multi-Event Video Generation 

**Authors**: Ruotong Liao, Guowen Huang, Qing Cheng, Guangyao Zhai, Lei Zhang, Xun Xiao, Thomas Seidl, Daniel Cremers, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2605.31590)  

**Abstract**: Text-to-video (T2V) generation faces challenging questions when generating videos with long horizons containing multiple events. Inspired by the intrinsics of the diffusion process, we probe video diffusion transformers (DiTs) and uncover intrinsic turning points in the DiT denoising trajectory where conditioning text affects generation from global layout to fine-grained details. Building on this finding, we present TunerDiT, a simple yet effective progressive steering method that requires no additional training for multi-event generation. TunerDiT comprises two steering handles: (1) Event-Partitioned Masking that enforces event boundaries while allowing cross-event transition bands; (2) Cross-Event Prompt Fusion that injects neighboring event semantics for late-stage refinement. We contribute a self-curated prompt suite for benchmarking multi-event generation, i.e., Meve. TunerDiT achieves state-of-the-art performance across 8 metrics and offers a tunable trade-off between video consistency and event separation, compared with other training-free methods. The improvement in text alignment increases with the event count, indicating a scaling possibility with increasing event count. 

---
# Language Models Learn Constructional Semantics, Not To Mention Syntax: Investigating LM Understanding of Paired-Focus Constructions 

**Authors**: Wesley Scivetti, Ethan Wilcox, Nathan Schneider, Kanishka Misra, Leonie Weissweiler  

**Link**: [PDF](https://arxiv.org/pdf/2605.31586)  

**Abstract**: Grasping the semantics of rare constructions (form-meaning pairings) has been shown to be a challenging problem that has currently only been solved by the largest LLMs. It remains an open question if open-source models have robust constructional understanding, and if so, what learning dynamics underlie the acquisition of this knowledge. Focusing on a set of rare Paired-Focus constructions in English (e.g. "let alone", "much less"), we construct a novel dataset to test their meanings using both scalar adjectival semantics and general world knowledge. Testing a wide range of models differing in parameter count, architecture, and pretraining dataset size, we find that several modestly sized models are sensitive to both the forms and the meanings of Paired-Focus constructions, though models trained on human-scale data fail at all meaning evaluations. Turning to training dynamics for a set of open-checkpoint models, we find that Paired-Focus understanding emerges later in training than Paired-Focus syntactic knowledge, and that learning of Paired-Focus semantics is correlated with gains in some domains of world knowledge. Overall, our empirical results support the conclusion that modestly sized open-source models can grasp the rare Paired-Focus constructions, and demonstrate a connection between knowledge of Paired-Focus constructions and other meaning domains. 

---
# LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards 

**Authors**: Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31584)  

**Abstract**: Long-context reasoning remains a central challenge for large language models, which often fail to locate and integrate key information in extensive distracting content. Reinforcement learning with verifiable rewards (RLVR) has shown promise for this task, yet existing methods are limited by low-confusability distractors and sparse, outcome-only reward signals that cannot supervise intermediate reasoning steps. To address these issues, we introduce \textsc{LongTraceRL}. For data construction, we generate multi-hop questions via knowledge graph random walks and leverage search agent trajectories to build \emph{tiered distractors}: documents the agent read but did not cite (high confusability) and documents that appeared in search results but were never opened (low confusability), producing training contexts that are far more challenging than those built by random sampling or one-shot search. For reward design, we propose a \emph{rubric reward} that uses the gold entities along each reasoning chain as fine-grained, entity-level process supervision. This rubric reward is applied only to responses with correct final answers (positive-only strategy), distinguishing the reasoning quality among correct responses and preventing reward hacking. Experiments on three reasoning LLMs (4B--30B) across five long-context benchmarks demonstrate that \textsc{LongTraceRL} consistently outperforms strong baselines and encourages comprehensive, evidence-grounded reasoning. Codes, datasets and models are available at \href{this https URL}{this https URL}. 

---
# SPECTRA: Synthetic IR Test Collections with Relevance Oracles and Controlled Distractor Diagnostics 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31575)  

**Abstract**: Scalable information retrieval testing needs corpora that are large enough to stress index construction, ranking latency, query routing, and evaluation tooling, yet human-judged test collections remain expensive and may be unavailable when documents are private or still under design. This paper introduces SPECTRA, a reproducible framework for generating synthetic text corpora and retrieval test collections through a separation of latent topical structure, surface text realization, metadata controls, query intent generation, and deterministic relevance oracles. The framework is intended as a diagnostic complement to Cranfield-style and TREC-style evaluation, not as a replacement for human assessment. A single-process Python prototype generated corpora up to 60,000 documents and 9.61 million tokens while preserving controllable long-tail vocabulary growth and producing graded relevance labels for 96 queries. In the local simulation study, generation remained close to linear at roughly 12K to 14K documents per second, estimated Zipf slopes stayed near 0.86 in absolute value, and increasing cross-topic distractor text reduced BM25 nDCG@10 from 1.00 at 2% distractors to 0.43 at 36% distractors. These results show that lightweight synthetic corpora can expose retrieval-system scaling and failure modes before costly collection construction begins. 

---
# What Gets Unmasked First? Trajectory Analysis of Diffusion Models for Graph-to-Text Generation 

**Authors**: Qing Wang, Jacob Devasier, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31564)  

**Abstract**: We present the first systematic study of masked diffusion language models (MDLMs) for graph-to-text generation. We analyze MDLM generation trajectories -- the order in which tokens are unmasked during iterative decoding -- and find that, unlike autoregressive LLMs which generate text linearly, MDLMs naturally prioritize entities first, followed by relational and function words, with structural tokens resolved last. We further identify a previously undocumented failure mode of supervised fine-tuning: SFT disrupts this strategy by prematurely anchoring structural sentence-ending tokens early in the decoding trajectory, effectively fixing the output length which can lead to omitted or hallucinated information. To address this, we propose lambda-scaled structural decoding, a training-free inference-time modification that downweights structural token confidence and recovers +9.4 BLEU-4. Finally, we introduce Graph-LLaDA, which integrates a Graph Transformer encoder into LLaDA's decoding process to explicitly incorporate relational graph structure. Cross-dataset evaluation on LAGRANGE reveals that previous baselines overfit to dataset-specific patterns, while LLM- and MDLM-based approaches generalize significantly better. 

---
# Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization 

**Authors**: Felipe Urrutia, Juan José Alegría, Cinthia Sanchez Macias, Jorge Salas, Cristian B. Calderon, Cristobal Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2605.31558)  

**Abstract**: Transformer-based language models are widespread in today's society. As such, understanding the mechanisms by which they solve structured tasks and predicting how they may behave in novel scenarios is of great importance for safe deployment. We study the learning dynamics of attention heads in a controlled setting by training a decoder-only Transformer (GPT-J) on two structurally equivalent multi-hop reasoning tasks: a number task requiring positional reasoning and a letter task requiring symbolic reasoning. Using a recently introduced metric that classifies attention-head behavior as positional or symbolic for a given prompt, we show that successful learning is associated with the emergence of pure heads, i.e., heads that express themselves as either positional or symbolic. Despite the tasks' structural equivalence, they impose different mechanistic demands: the number task requires both positional and symbolic heads, whereas the letter task requires only symbolic heads. We then identify the computational roles of these heads, characterize the basic functions they implement, and give theoretical constructions showing how single-layer RoPE-based attention can realize these functions through geometrically interpretable query, key, and value operations. This analysis yields a quantitative separation between positional and symbolic mechanisms in their robustness to longer sequences, formalized through a novel notion of discrepancy. We empirically validate the resulting predictions in both controlled and real-world models, showing that symbolic mechanisms extrapolate more reliably to longer sequences while positional mechanisms face sharper limitations. 

---
# Vision-Language Models Suppress Female Representations Under Ambiguous Input 

**Authors**: Arnau Marin-Llobet, Simon Henniger, Mahzarin R. Banaji  

**Link**: [PDF](https://arxiv.org/pdf/2605.31556)  

**Abstract**: Alignment teaches vision-language models (VLMs) to avoid expressing demographic biases, and when gender is clearly visible they largely succeed. Far less is known about ambiguous inputs (a worker in full gear, a figure seen from behind) cases common in practice yet rarely studied. We find that minimal prompting pressure exposes occupation-gender defaults when prompting ambiguous input images, with models collapsing to male even for strongly female-stereotyped occupations. But do these outputs reflect what models actually encode internally? We introduce LALS (Latent Association Leaning Score), a zero-shot metric that projects visual-token activations into the model's text-embedding space to measure concept associations per token and layer. Across 15 occupations, over 800 gender-ambiguous images, and four VLMs, internal representations and outputs are systematically decoupled: models often encode a female association internally yet output male. Layer-wise analysis reveals an asymmetric filter -- male signal amplifies end-to-end while female signal peaks mid-network and is suppressed before generation -- and a color ablation shows that culturally loaded visual cues such as clothing color further modulate these internal associations. 

---
# RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video 

**Authors**: Ulrich Prestel, Stefan Andreas Baumann, Nick Stracke, Björn Ommer  

**Link**: [PDF](https://arxiv.org/pdf/2605.31535)  

**Abstract**: Self-supervised novel view synthesis (NVS) remains challenging to scale, despite the abundance of video data, largely due to the brittleness of training on realistic videos and the hard-to-predict scaling behavior of multi-network system designs. We introduce RayDer, a unified, feed-forward transformer that consolidates camera estimation, scene reconstruction, and rendering into a single backbone, turning self-supervised NVS into a well-posed single-model scaling problem. A minimal dynamic state, treated as a nuisance factor, absorbs time-varying content and enables stable training on unconstrained real-world video. Importantly, RayDer keeps static-scene NVS as its target task: dynamic content is leveraged purely as scalable supervision, not reconstructed as in dynamic-scene (4D) NVS. Across multiple model sizes and orders of magnitude in data, RayDer exhibits clean power-law scaling with data and compute, and outperforms static-scene data mixtures. On a large number of benchmarks, RayDer achieves strong zero-shot open-set performance competitive with state-of-the-art supervised approaches. Project Page: this https URL 

---
# Feature-Optimized Vision for Adaptive 3D Scene Reconstruction 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31534)  

**Abstract**: Three-dimensional scene reconstruction depends on local image evidence that is both visually discriminative and geometrically useful. Fixed feature thresholds and uniform feature budgets are easy to deploy, but they can waste computation on repeated texture, low-parallax regions, or unstable points. This paper proposes an adaptive feature-optimized vision front end for 3D reconstruction. The method scores candidate features by texture, repeatability, distinctiveness, expected triangulation angle, and spatial coverage, then allocates a per-view feature budget to maximize useful tracks under a fixed reconstruction pipeline. A small synthetic multi-view prototype evaluates four selection policies across corridor, facade, object-table, and cluttered scenes. Compared with random, texture-only, and uniform-grid baselines, the adaptive policy obtains the best quality-aware completeness and the lowest aggregate reconstruction RMSE while preserving broad image coverage. The result is not a replacement for modern learned matching or neural reconstruction systems; it is a modular front-end policy that can make classical and learned 3D pipelines more deliberate about which visual evidence they spend compute on. 

---
# Separating Secrets from Placeholders: A Hybrid CNN-CodeBERT Framework for Three-Class Credential Leakage Detection 

**Authors**: Maksuda Bilkis Baby, Khushika Shah, Naiyue Liang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31520)  

**Abstract**: Credential leakage in public source code repositories poses a critical security threat, with over 23.8 million secrets exposed in 2024 alone. Existing detection tools suffer from high false-positive rates because rigid pattern matching and binary classification schemes fail to distinguish genuine credentials from placeholder or weak credentials. We propose a three-class classification framework that explicitly models placeholder or weak credentials as a distinct class, leveraging CodeBERT-based semantic understanding combined with character-level pattern recognition. We evaluate our approach on a newly constructed dataset of 9,426 samples spanning 10 programming languages. Our model achieves a Matthews Correlation Coefficient of 0.86 and a macro F1-score of 0.90, achieving 93% recall and 89% precision for genuine credential leaks while reducing high severity alerts by 33.0% (from 373 to 250) without sacrificing security coverage. Compared to prior character-level approaches, our method improves placeholder or weak credential detection from 54% to 81% F1-score while maintaining strong cross language generalization, with 9 of 10 languages achieving F1 above 0.80 under leave-one-language-out evaluation. 

---
# If LLMs Have Human-Like Attributes, Then So Does Age of Empires II 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2605.31514)  

**Abstract**: Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain constant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions, regardless of the experimenter's viewpoint on the subject. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that \textit{Age of Empires II} is functionally- and Turing-complete. 

---
# Skill Reuse as Compression in Agentic RL 

**Authors**: Zhikun Xu, Yu Feng, Jacob Dineen, Taiwei Shi, Jieyu Zhao, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.31509)  

**Abstract**: Large language model agents trained with reinforcement learning (RL) often learn brittle, task-specific shortcuts. We hypothesize that agents generalize better when their successful trajectories are structurally compressible, decomposed into a small set of reusable abstract patterns. To formalize this, we introduce ReuseRL, which grounds agentic RL in the Minimum Description Length (MDL) principle. ReuseRL extracts a shared skill dictionary from successful trajectories and augments the RL objective with a segmentation cost, explicitly penalizing idiosyncratic behaviors that encode poorly. We prove a PAC-Bayes generalization bound for this compression penalty. Across ALFWorld, TextWorld-Cooking, and Countdown-Stepwise, ReuseRL improves in- and out-of-distribution success over vanilla GRPO and strong round-length baselines. 

---
# On Efficient Scaling of GNNs via IO-Aware Layers Implementations 

**Authors**: Daria Fomina, Daniil Krasylnikov, Alexey Boykov, Andrey Dolgovyazov, Vyacheslav Zhdanovskiy, Fedor Velikonivtsev  

**Link**: [PDF](https://arxiv.org/pdf/2605.31500)  

**Abstract**: Graph Neural Networks (GNNs) are bottlenecked by sparse, irregular memory access. Popular frameworks such as DGL and PyTorch Geometric support general message passing, but complex layers often materialize edge-wise intermediates, increasing memory traffic and limiting scalability on large graphs. We take an I/O- and arithmetic-intensity--centric view and show that widely used layers fall into three kernel families: SpMM-based convolutions, reduction-based aggregations, and attention-based layers (GATv2/Graph Transformer). For each family, we develop GPU kernels that reduce data movement, improve locality, and remain robust across realistic graphs. We also study graph reordering and find that its impact depends on the kernel mapping: it benefits neighbor-parallel (gather-dominated) kernels more consistently than feature-parallel designs. Empirically, our fused attention kernels reach up to $\textbf{3.9}\times$ speedup for Graph Transformer (median $\textbf{1.6}\times$), with Tensor Core (block-sparse) variants up to $\textbf{7.3}\times$ on locally dense graphs; for GATv2 we reach up to $\textbf{8.5}\times$ speedup (median $\textbf{2.0}\times$) while reducing peak memory by up to $\textbf{76}\times$ (median $\textbf{6}\times$). Our degree-aware reduction kernels achieve up to $\textbf{10}\times$ speedup (median $\textbf{2.6}\times$). For SpMM-based layers, properly cached cuSPARSE achieves up to $\textbf{8}\times$ speedup over DGL and outperforms evaluated custom baselines in the majority of evaluations. We release our implementations as drop-in replacements to support reproducible, hardware-aware GNN acceleration. 

---
# Scaling Conversational Hungarian ASR: The BEA-Dialogue+ Corpus 

**Authors**: Máté Gedeon, Piroska Zsófia Barta, Péter Mihajlik, Katalin Mády  

**Link**: [PDF](https://arxiv.org/pdf/2605.31469)  

**Abstract**: Conversational automatic speech recognition in Hungarian is constrained by the limited amount of publicly available dialogue-style training data. The BEA-Dialogue corpus addresses this need, but its strictly speaker-disjoint train/dev/eval split reduces the usable material to only 85 hours. In this paper, we introduce BEA-Dialogue+, an expanded version of the corpus that relaxes the split criterion for experimenters and dialogue partners while preserving complete separation of the primary speakers. This results in 200 hours of transcribed natural conversations and enables a controlled study of the trade-off between additional training data and speaker overlap across the splits. We evaluate several Whisper- and FastConformer-based models on both corpus versions, including Serialized Output Training (SOT)-based fine-tuning for dialogue transcription. Our results show that the larger corpus is more challenging for models without fine-tuning, whereas SOT-based adaptation yields consistent improvements in WER, CER, cpWER, and cpCER. Overall, BEA-Dialogue+ provides a substantially larger yet still demanding benchmark for Hungarian dialogue ASR, and a practical resource for training and evaluating dialogue transcription systems. 

---
# GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization 

**Authors**: Zaid Khan, Justin Chih-Yao Chen, Jaemin Cho, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.31464)  

**Abstract**: GPU kernels are the workhorse of modern deep learning, and optimizing them (via evolutionary search or coding agents) usually requires repeated measurement on target hardware. While these measurements provide the ground-truth signal necessary for kernel search, they are costly, because each evaluation of a kernel requires compilation and repeated execution on a GPU. As improvements in LLM inference reduce the cost of writing novel kernels and LLM-driven searches scale to large search budgets, on-device evaluation becomes a bottleneck. To address this, we study how LLMs can serve as selective GPU surrogates for kernel evaluation, by forecasting the performance of proposed kernels. A useful surrogate should be accurate, and it should be selective, by knowing when it could be wrong, and deferring to the GPU. To evaluate surrogates, we measure whether their forecasts are accurate, calibrated, and practically useful for recovering fast kernels under limited GPU-measurement budgets. Next, we study whether reinforcement learning can improve forecast accuracy and confidence calibration. Our experiments demonstrate that LLMs can accurately forecast relative kernel performance, that their utility can be improved through reinforcement learning. Used inside a kernel search, the surrogate lets the search consider several times as many candidates under the same GPU evaluation budget, and that leads to finding faster kernels than an equal-budget baseline. These results suggest that LLMs can play a broader role in kernel optimization, by acting as virtual models of a GPU rather than solely as kernel generators for search. 

---
# PithTrain: A Compact and Agent-Native MoE Training System 

**Authors**: Ruihang Lai, Hao Kang, Haozhan Tang, Akaash R. Parthasarathy, Zichun Yu, Junru Shao, Todd C. Mowry, Chenyan Xiong, Tianqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31463)  

**Abstract**: Mixture-of-Experts (MoE) has become the dominant architecture for frontier language models. To meet this demand, production frameworks have built optimized MoE training stacks over years of engineering effort. Yet evolving these stacks for new architectures and system optimizations remains expensive. With the rise of AI coding agents, they could automate parts of training-framework development and accelerate this evolution. But applying them to these existing frameworks carries hidden costs, invisible to today's throughput-only evaluations. We name this missing dimension agent-task efficiency (ATE): the cost of using coding agents to understand, operate, and extend a framework. Grounded in four agent-native design principles, we build PithTrain, a compact, agent-native MoE training framework. We further introduce ATE-Bench, covering real-world training-framework tasks. Our evaluation shows PithTrain matches the throughput of production frameworks, and on ATE-Bench, PithTrain enables higher agent-task efficiency, with up to 62% fewer Agent Turns and 64% less Active GPU Time. 

---
# Fine-grained Verification via Diagnostic Reasoning Supervision for Aspect Sentiment Triplet Extraction 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.31446)  

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) aims to identify aspect terms, opinion terms, and sentiment polarities as structured triplets, providing essential inputs for downstream information system applications such as opinion mining, explainable recommendations, and review summarization. Prior work mainly focuses on end-to-end extraction, while post hoc verification of extracted triplets remains comparatively underexplored. This gap limits the reliability of ASTE systems, since predicted triplets may be locally plausible while being globally invalid. Moreover, candidate invalidity is multi-faceted and candidate usability is inherently graded, motivating a fine-grained verification mechanism that can filter or re-rank outputs from diverse extractors. In this paper, we propose FiVeD, a framework for Fine-grained Verification with Diagnostic reasoning supervision. Specifically, the verifier is trained with multiple complementary objectives, including validity classification and quality score estimation as primary tasks, with error type classification and rationale generation as auxiliary tasks. We define hierarchical error categories and construct plausible incorrect triplets under semantic and syntactic constraints, and leverage an off-the-shelf LLM with task-specific rubrics to produce quality scores and diagnostic rationales. During inference, the resulting quality scores are used to filter candidate outputs, supporting adjustable precision-recall tradeoffs. Experiments across multiple ASTE baselines demonstrate that FiVeD consistently improves extraction performance by up to 3.53 F1 points as a plug-and-play verification module. 

---
# Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information 

**Authors**: Antonio Valerio Miceli-Barone, Vaishak Belle, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31445)  

**Abstract**: In this work we study agents in simulated bargaining scenarios, where a buyer and a seller communicate through a text channel and attempt to negotiate mutually beneficial trades, under different information regimes (complete information, information asymmetry or mutual uncertainty). We evaluate their performance w.r.t. game-theoretical solutions and further investigate their honesty (their tendency to disclose or withhold information or to mislead and deceive) as well as their credulity (their tendency to trust or distrust information provided by the other agent). We study zero-shot LLM agents with simple prompting scaffolding as well as fine-tuned agents, in order to investigate whether optimising the agents to maximise financial profits makes them stronger negotiators but also more dishonest and less trusting.
We find that off-the-shelf LLMs all substantially deviate from game-theoretical equilibria, they attempt to lie about their private information but cannot efficiently exploit information asymmetries. Fine-tuning on financial utility makes the agents stronger at achieving better deals but also more dishonest, highlighting the risks that optimising agents for a task can have on their safety. We release our code and a dataset of bargaining scenarios. 

---
# DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs 

**Authors**: Sara Papi, Luisa Bentivogli  

**Link**: [PDF](https://arxiv.org/pdf/2605.31432)  

**Abstract**: Simultaneous speech-to-text translation (SimulST) generates translations while speech is still unfolding, requiring a streaming policy that decides when to read and when to write. State-of-the-art approaches rely on attention-based encoder-decoder models where cross-attention provides explicit alignment signals. In contrast, Speech Large Language Models (SpeechLLMs) are decoder-only architectures relying solely on self-attention. This raises a central question: whether decoder self-attention contains sufficiently stable alignment signals to guide the streaming policy. Moreover, existing approaches typically rely on training-based adaptations or heuristic wait-$k$ policies and have not been validated in long-form settings. To fill these gaps, we propose Decoder-Only Attention (DOA), a training-free policy that enables long-form simultaneous translation with off-the-shelf SpeechLLMs by deriving a proxy alignment from self-attention. Experiments on Phi4-Multimodal and Qwen3-Omni show that DOA provides an effective alignment signal for supporting streaming decisions, enabling low-latency long-form SimulST with quality close to offline decoding without retraining. 

---
# Neuro-symbolic Syntactic Parsing: Shaping a Neural Network with the CYK Algorithm 

**Authors**: Fabio Massimo Zanzotto, Federico Ranaldi, Giorgio Satta  

**Link**: [PDF](https://arxiv.org/pdf/2605.31421)  

**Abstract**: In this paper, we show the possibility of a direct injection of algorithms into neural network architecture. We focus on a complex algorithm, that is, Cocke-Youger-Kasami (CYK) for parsing context-free grammars in Chomsky Normal Form and we propose CYKNN, a simple recurrent neural network architecture for encoding the CYK algorithm in trainable matrix-vector this http URL experimented with a very simple grammar with 4 variations showing that our approach outperforms existing LLMs with more than 20B parameters with an in-context learning setting and smaller LLMs of the Qwen family fine-tuned with LoRA. Our attempt paves the way to a different approach to neuro-symbolic methodologies. 

---
# Skill Availability and Presentation Granularity in Large-Language-Model Agents: A Controlled SkillsBench Study 

**Authors**: Xiaonan Xu, Wenjing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31408)  

**Abstract**: Skill documents provide procedural knowledge to large-language-model agents at inference time. This article studies whether the presentation granularity of controlled skill knowledge changes downstream task success. The experiment uses a pinned SkillsBench version, a 30-task domain-balanced subset validated by official oracle runs, two reasoning-enabled model configurations, six skill conditions, and five trials per task-condition-model cell. Skill availability is the clearest empirical signal. Relative to no skill, skill conditions increase task-mean pass rate by 26.7 to 36.0 percentage points for GPT-5.5 and by 18.0 to 26.0 percentage points for DeepSeek V4-Flash. The final data contain 1,800 rows, with 900 rows for each model. The task is the inference unit. Five trials are aggregated within each task-condition-model cell before paired contrasts are estimated over 30 tasks. The primary presentation contrasts are smaller and uncertain. Low-abstraction guidance differs from high-abstraction guidance by +0.7 percentage points for GPT-5.5 and -6.7 percentage points for DeepSeek V4-Flash, with both 95% bootstrap confidence intervals crossing zero. Adding one worked example to medium-abstraction guidance differs from the no-example variant by +0.7 and +1.3 percentage points. Mean-reward robustness checks preserve the same substantive conclusion. In this controlled subset, skill availability is associated with higher success than no skill, while the tested presentation-granularity changes yield small, uncertain, and model-dependent effects. 

---
# The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning 

**Authors**: Xudong Zhang, Jian Yang, Shengkai Wang, Jiangpeng Tian, Shaowen Chen, Xian Wei, Ke Li, Xiong You  

**Link**: [PDF](https://arxiv.org/pdf/2605.31404)  

**Abstract**: Large Language Model (LLM)-based navigation systems commonly construct explicit spatial representations (e.g., topological graphs, semantic raster maps) and translate them into textual descriptions as LLMs' inputs. However, the linguistic structures of such text-based spatial representations and the choices of contextual features (e.g., topology, geometry) they contain are often treated as neutral engineering decisions rather than key factors that shape LLMs' behavior. To fill the gap, we propose a dual-interventional framework that disentangles linguistic structures from different contextual cues to evaluate the linguistic inductive bias of LLMs for navigation planning. In the framework, representation intervention varies the linguistic format and the degree of linguistic compression, clarifying when linguistic representations support or inhibit navigation planning. Context intervention, combined with contextual feature combination and conflict probing, explicitly clarifies the preferences and weaknesses of LLMs when processing different contextual cues. Experiments across diverse spatial reasoning tasks and multiple model scales reveal a consistent pattern: topological information is a sturdy shield and the backbone of robust planning; linguistic format is a double-edged sword whose effect depends on model size, task demands, and the compression level; and semantic information is a fatal Achilles' heel -- incorrect semantic cues can systematically derail the planning process. Overall, our study shows that effective text-based spatial representations in LLM-based navigation should preserve topological integrity, calibrate representational compression to model capacity, and ensure semantic correctness, rather than simply adopting a single representation. Our code is publicly available at this https URL. 

---
# Target-Side Paraphrase Augmentation for Sign Language Translation with Large Language Models 

**Authors**: Pedro Dal Bianco, Jean Paul Nunes Reinhold, Oscar Stanchi, Facundo Quiroga, Franco Ronchetti, Ulisses Brisolara Corrêa  

**Link**: [PDF](https://arxiv.org/pdf/2605.31393)  

**Abstract**: Sign language translation (SLT) remains constrained by limited paired sign-video/text corpora and heavy-tailed target vocabularies. We study target-side augmentation in which GPT-4o generates controlled paraphrase variants of reference sentences while the sign input remains unchanged. A Signformer-style pose-based Transformer is trained under a two-stage schedule: pre-training on the augmented corpus followed by fine-tuning on the original references.
We evaluate on three datasets spanning complementary challenges: PHOENIX14T (German Sign Language), with moderate lexical diversity; GSL (Greek Sign Language), with highly ontrolled, repetitive recordings; and LSA-T (Argentinian Sign Language), with severe long-tail sparsity. On PHOENIX14T, augmentation improves BLEU-4 from 9.56 to 10.33. The near-saturated GSL baseline and extremely sparse LSA-T setting reveal the limits of the approach. To our knowledge, this is the first study to apply LLM-generated target-side araphrases and LLM-as-a-Judge evaluation to SLT. The semantic evaluation reveals gains in fidelity that lexical overlap metrics understate. 

---
# DynaTree: Dynamic Agentic Retrieval Tree for Time-Sensitive News Retrieval 

**Authors**: Siyuan Qi, Xinyuan Wang, Yingxuan Yang, Haochuan Guo, Jianghao Lin, Weiwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31377)  

**Abstract**: Agentic Retrieval-Augmented Generation improves retrieval by integrating planning, tool use, and iterative reasoning, but existing agentic RAG methods often couple semantic expansion with retrieval decisions in short-horizon inference loops, leading to high inference cost and limited suitability for time-sensitive news retrieval. We propose DynaTree, a two-stage framework for efficient and adaptive news retrieval. In the offline stage, DynaTree uses coordinated agents to construct a reusable retrieval tree that materializes the semantic space of a query topic. In the online stage, DynaTree performs lightweight daily subtree selection over a time-localized evaluation proxy, without further agentic reasoning, tree modification, or retraining. Experiments on a multi-day Syft news benchmark and multiple BEIR datasets show that DynaTree achieves strong recall and ranking performance, consistently outperforming standard RAG and prior agentic baselines. We further deploy DynaTree in the Syft production system and evaluate it through online A/B testing from Jan. 28 to Feb. 6, 2026. The dynamically adapted variant improves survival rate from 0.32-0.53 to 0.59-0.73 over a fixed offline-selected subtree and outperforms existing production recallers on every evaluation day. These results show that persistent, structure-aware semantic expansion can translate offline agentic reasoning into practical improvements in coverage, freshness, and relevance for real-world news retrieval. 

---
# Scaling Higher-Order Graph Learning with Maximal Clique Complexes 

**Authors**: Antoine Vialle, Aref Einizade, Fragkiskos D. Malliaros, Jhony H. Giraldo  

**Link**: [PDF](https://arxiv.org/pdf/2605.31373)  

**Abstract**: Graph neural networks (GNNs) are limited to modeling pairwise interactions, while higher-order models based on cell complexes achieve greater expressivity but often suffer from poor scalability. We introduce simplified and factored cellular Weisfeiler Leman tests (sCWL and fCWL), which preserve the expressivity of the CWL test while improving computational efficiency. We further introduce the maximal clique complex, enabling scalable CWNs with reduced time and memory complexity while retaining strong empirical performance. To avoid explicit clique enumeration, we propose CliqueWalk, a biased random walk that samples maximal cliques and scales linearly with graph size. These contributions yield a scalable topological learning framework for higher-order graph representation. 

---
# Dreaming Of Others: Latent Teammate Modeling In World Models For Multi-Agent Reinforcement Learning 

**Authors**: Tomas Leroy-Stone  

**Link**: [PDF](https://arxiv.org/pdf/2605.31361)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), agents must coordinate with partners whose internal policies and intentions are not directly observable. While world models such as Dreamer have demonstrated strong generalization and sample efficiency in single-agent settings, their application to MARL remains limited by an inability to handle teammate-induced uncertainty. We propose a new perspective: treat teammates as structured, learnable components within the agent's world model. We introduce an architecture that factorizes the latent state of a Dreamer-style recurrent state-space model (RSSM) into environment and teammate components, and learns an auxiliary Theory-of-Mind (ToM) head to infer latent embeddings of partner behavior such as character, intent, and predicted actions from partial trajectories. These teammate latents condition the actor and critic, enabling the agent to imagine and adapt to diverse collaborators. We outline how this approach can support zero-shot and few-shot coordination in partially observable settings and propose a set of benchmarks and evaluation protocols to assess its impact. This work positions world models as not only predictors of environmental dynamics, but as simulators of social behavior, opening new directions for generalizable, human-compatible AI. 

---
# dashi: A Python library for Dataset Shift Characterization to Support Trustworthy AI Development and Deployment 

**Authors**: David Fernández-Narro, Pablo Ferri, Ángel Sánchez-García, Juan M. García-Gómez, Carlos Sáez  

**Link**: [PDF](https://arxiv.org/pdf/2605.31360)  

**Abstract**: The Artificial Intelligence (AI) life cycle requires a thorough understanding of the underlying data dynamics for robust, safe and cost-effective AI development and use. Dataset shifts are defined as changes between train and test data distributions. Whether occurring over time (temporal) or across different sites (multi-source), they can severely degrade model performance and compromise data quality. This is particularly important in health AI, where the safety and fundamental rights of patients can be severely affected by uncontrolled shifts both at training and operational stages. While the theoretical foundations of covariate, prior, and concept shifts are well established, there is a lack of accessible and comprehensive software tools to perform their analysis. We introduce dashi, an open-source Python library designed for the exploration, quantification, and characterization of dataset shifts. dashi provides a dual approach: an unsupervised approach that leverages information geometry and non-parametric statistical manifolds to data variability characterization and analysis (e.g., Information Geometric Temporal plots and Multi-Source Variability metrics like Global Probabilistic Deviation and Source Probabilistic Outlyingness), and a supervised approach that quantifies and characterizes model performance degradation. Both unsupervised and supervised approaches work across user-defined temporal and domain/source batches. We demonstrate the utility of dashi on three simulated and real-world health AI case studies on gestational diabetes mellitus, COVID-19 and emergency medical dispatch. By providing interactive visual analytics and variability metrics, dashi supports trustworthiness of AI life cycle stages enabling robust and safe machine learning pipelines through the assessment of data coherence and AI performance. 

---
# FBHM: Functional Benchmarking and Steering of VLMs for Hateful Meme Detection 

**Authors**: Paramananda Bhaskar, Naquee Rizwan, Daksh Jogchand, Saurabh Kumar Pandey, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31349)  

**Abstract**: Hateful meme detection remains a formidable challenge for vision-language models, as existing benchmarks are structurally observational - confounding rhetorical hate mechanisms with target community features and preventing causal evaluation of model vulnerabilities. To address this, we introduce FBHM, a systematically curated benchmark of Functionality Based Hateful Memes constructed along two orthogonal axes: 25 distinct rhetorical functionalities and 10 target communities (5,000 memes total). Benchmarking state-of-the-art VLMs reveals a severe generalization gap: models highly accurate on standard datasets catastrophically drop to near-random performance on FBHM, proving they exploit dataset-specific heuristics rather than robust multimodal reasoning. To efficiently close this gap, we propose LSV (learnable steering vectors), an ultra-low data regime strategy that applies a causal intervention objective on as few as 500 steering samples (50 unique base memes), boosting FBHM performance by ~30 Macro-F1 points while outperforming in-context learning and PEFT without degrading source-domain performance. 

---
# Appropriateness of Empathy in AI: A Signal-Cost Perspective 

**Authors**: Chi-Ching Juan, Tao Wang, Harold Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31340)  

**Abstract**: The appropriateness of empathy in AI has emerged as a critical concern, as excessive empathy risks seeming manipulative while insufficient empathy appears dismissive. While prior research has explored how to quantify empathy in AI, few studies examine whether such empathy is contextually appropriate. This paper introduces an economic perspective by applying signaling theory to human-AI conversations. We propose Signal Cost Proxies (emotional richness, perspective-taking, and contextual tailoring) mapped to affective, cognitive, and associative empathy. This multidimensional framework enables systematic evaluation of empathy not just by presence, but by its appropriateness relative to user demand. 

---
# Social welfare optimisation under institutional reward and punishment 

**Authors**: Van An Nguyen, Vuong Khang Huynh, Huu Loi Bui, Hai Anh Ha, Quang Dung Le, Tan Dat Nguyen, Ngoc Ngu Nguyen, Zhao Song, Manh Hong Duong, Le Hong Trang, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.31330)  

**Abstract**: Institutional incentives are widely used to promote cooperation among autonomous, self-regarding agents, from human societies to multi-agent and AI systems. Existing work typically treats incentive design as a bi-objective problem: minimise institutional cost while achieving a high long-run frequency of cooperation. Whether such schemes also maximise social welfare - total population payoff net of institutional expenditure - has remained largely unexplored. We develop a welfare-centric framework for institutional incentives in finite, well-mixed populations playing a social dilemma (Donation Game and Public Goods Game), considering both rewards for cooperators and punishments for defectors. For each mechanism, we derive explicit expressions for expected social welfare and characterise how it depends on incentive efficiency and selection intensity. Analytically, we identify parameter regimes where social welfare has a single optimal incentive level and regimes with qualitative phase transitions, in which welfare becomes non-monotonic with multiple local optima. We prove that any welfare-maximising incentive is either zero or concentrated around a simple closed-form target, and we provide an efficient algorithm to compute these optima. Comparing reward and punishment, we further derive close-formed conditions under which reward outperform punishment in terms of social welfare for any given budget. Overall, our results reveal a systematic gap between incentives optimised for cost or cooperation frequency and those that maximise welfare. 

---
# Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data 

**Authors**: Hee-Sung Kim, Hyeonseong Kim, Sungyoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31324)  

**Abstract**: Estimating the generalization gap and developing optimization methods that improve generalization are crucial for deep learning models, for both theoretical understanding and practical applications. Leveraging unlabeled data for these purposes offers significant advantages in real-world scenarios. This paper introduces a novel generalization measure, local inconsistency, derived from an information-geometric perspective on the parameter space of neural networks. A key feature of local inconsistency is that it can be computed without explicit labels. We establish theoretical underpinnings by connecting local inconsistency to the Fisher information matrix and the loss Hessian. Empirically, we demonstrate that local inconsistency correlates with the generalization gap. Based on these findings, we propose Inconsistency-Aware Minimization (IAM), which incorporates local inconsistency into the training objective. We demonstrate that in standard supervised learning settings, IAM enhances generalization, achieving performance comparable to that of existing methods such as Sharpness-Aware Minimization. Furthermore, IAM exhibits efficacy in semi- and self-supervised learning scenarios, where the local inconsistency is computed from unlabeled data. 

---
# Latent Space Disentanglement via Activation Steering for Interpretable Attribute Control in Symbolic Music Generation 

**Authors**: Ioannis Prokopiou, Pantelis Vikatos, Maximos Kaliakatsos-Papakostas, Theodoros Giannakopoulos, Themos Stafylakis  

**Link**: [PDF](https://arxiv.org/pdf/2605.31295)  

**Abstract**: Transformer-based architectures have significantly advanced the generation of complex symbolic sequences, yet a significant gap remains in achieving fine-grained, interpretable control over discrete signal attributes. This paper investigates the mechanistic interpretability of the Multitrack Music Transformer (MMT) and proposes a framework for deterministic attribute modulation without retraining to bridge this gap via inference-time activation steering. Utilizing the Difference-in-Means (DiffMean) methodology, we isolate latent directions for signal attributes, specifically Pitch and Duration, within the residual stream. We validate the Linear Representation Hypothesis in this domain, achieving high correlation between steering magnitude and attribute shift. To address the inherent feature entanglement in multi-attribute steering, we introduce a Dual Steering framework utilizing Gram-Schmidt Orthogonalization. Experimental results demonstrate that this geometric decoupling reduces conceptual interference and signal degradation compared to naive vector addition, enabling independent deterministic control even against strong autoregressive conditioning. 

---
# The Terminal Representation in Reinforcement Learning 

**Authors**: Amir Esterhuysen, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2605.31289)  

**Abstract**: Representation learning is a powerful tool for spatio-temporal abstraction within reinforcement learning (RL). Two well established approaches are through the successor representation (SR) and the default representation (DR). The SR encodes states by the future trajectories they induce, capturing information flow decoupled from reward. The DR builds on this by weighting trajectories with reward, integrating credit-assignment structure into the representation. Eigenvectors of both representations have been used to support a range of downstream tasks -- including option discovery, reward shaping, transfer learning, and exploration. We introduce a structurally distinct formulation: the terminal representation (TR). The TR encodes reward-weighted trajectories similarly to the DR, but can be learned as a lower-dimensionality object, and can be used directly for the mentioned applications without eigenvector computations. Eigendecomposition also imposes the assumption of symmetric transition dynamics, which the TR can bypass. In this work we develop the theoretical foundations of the TR: its derivation, convergence of two learning algorithms, its use for zero-shot compositionality, and equivalences between alternative reward formulations. We further show the TR is embedded in the top DR eigenvector, allowing it to capture the same underlying knowledge without eigendecomposition. Additionally, we provide empirical evidence of the TR as a viable alternative to existing representations in subsidiary applications, while requiring less computational overhead to learn, store, and use. 

---
# Neither Replacement nor Panacea: Comparing LLM-Based Conversational and Graphical Decision Support in Industrial Tasks 

**Authors**: Roberto Figliè, Simone Caputo, Alan Serrano, Daria Mikhaylova, Tommaso Turchi, Daniele Mazzei  

**Link**: [PDF](https://arxiv.org/pdf/2605.31287)  

**Abstract**: Managers in manufacturing settings rely on digital interfaces to interpret operational data for decision-making, but growing data volume and complexity can make relevant insights difficult to identify efficiently. While dashboards remain dominant in industrial contexts, Large Language Model (LLM)-based conversational agents (CAs), accessed through conversational user interfaces (CUIs), may provide more direct access to such data. However, their effectiveness may depend on the information-processing demands of the task. This study compares an LLM-based CA delivered through a CUI with a dashboard in a manufacturing decision-support scenario. In a mixed factorial experiment with a 2x3 design, 134 industrial decision-makers were assigned to one interface condition and completed three tasks of increasing complexity. We examined perceived Mental Workload (MWL), decision accuracy, completion time, and intended reliance, and tested self-reported data literacy as a moderator. Results showed that the CUI reduced perceived MWL overall and supported faster completion in less demanding tasks, but both advantages diminished as task complexity increased. Neither interface produced a consistent overall advantage in decision accuracy, and the CUI was not preferred as a sole basis for subsequent decisions. Furthermore, data literacy did not reliably moderate interface effects. These findings indicate that conversational interaction offers conditional rather than universal benefits for industrial decision support. LLM-based CAs may reduce information-access effort, whereas complex decisions continue to benefit from persistent, inspectable visual representations. 

---
# DeMaVLA: A Vision-Language-Action Foundation Model for Generalizable Deformable Manipulation 

**Authors**: Taiyi Su, Jian Zhu, Tianjian Wang, Youzhang He, Zitai Huang, Jianjun Zhang, Chong Ma, Hanyang Wang, Tianjiao Zhang, Munan Yin, Weihao Ding, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31286)  

**Abstract**: Real-world household robots require Vision-Language-Action (VLA) foundation models that can acquire reusable manipulation skills across diverse objects, task conditions, and household environments. Deformable-object folding is a representative challenge, requiring robots to handle clothing items from random initial states across varying categories, geometries, materials, and scenes. However, existing VLA systems commonly train separate policies for different object categories, while naively mixed multi-task training often suffers from task interference and degraded performance. To move beyond category-specific folding policies, we introduce DeMaVLA, a VLA foundation model for generalizable Deformable Manipulation. DeMaVLA adopts a VLM backbone with an action expert and formulates continuous action generation using flow matching. To improve efficiency, the action expert is constructed by pruning every other transformer layer while preserving layer-wise alignment with the VLM backbone, reducing training and inference cost. DeMaVLA is first pre-trained on approximately 5,000 hours of selected real-world dual-arm demonstrations to acquire general manipulation priors. It is then post-trained on mixed folding data that aggregates self-collected demonstrations and corrective trajectories from real-robot failures across multiple folding tasks through a human-in-the-loop Data Aggregation~(DAgger) pipeline. Experiments show that DeMaVLA achieves competitive performance on RoboTwin and strong real-world results on our household folding benchmark. These results highlight the value of scalable real-world data, efficient action generation, and corrective learning for general-purpose VLA policies in deformable-object manipulation. 

---
# SAM for Robust Mitochondria Instance Segmentation in Fluorescence Microscopy 

**Authors**: Suyog Jadhav, Dilip K. Prasad, Krishna Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.31284)  

**Abstract**: The morphological analysis of mitochondria in fluorescence microscopy (FM) is crucial for understanding cellular health, energy production, and metabolic regulation. While foundation models like the Segment Anything Model (SAM) have revolutionized natural image segmentation, their direct application to FM is hindered by a significant domain shift characterized by diffraction-limited resolution, low contrast, and complex overlapping organelle networks. Furthermore, the development of robust models is bottlenecked by a severe lack of high-quality, manually annotated instance segmentation datasets for mitochondria. In this paper, we propose a scalable solution to this data scarcity by finetuning SAM exclusively on synthetically generated FM data. We simulate realistic mitochondria data and emulate the optical properties of fluorescence microscopes to create a large-scale annotated dataset. We evaluate our fine-tuned model on a curated dataset of real, manually annotated FM images. Qualitative and quantitative analyses demonstrate that our synthetically fine-tuned model improves precision and average dice score over strong baselines. This work establishes the potential of simulation-assisted training for FM instance segmentation. 

---
# Practical Cross-Band Channel Prediction for AI-RAN via Physics-Guided Deep Unfolding 

**Authors**: Ruiqi Kong, He Chen, Xiaojun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.31279)  

**Abstract**: To make cross-band channel prediction practical for AI-native RAN, algorithms must generalize across diverse environments and support real-time inference. Existing approaches achieve one but not both. To bridge this gap, we introduce GUIDE, a physics-guided deep unfolding framework that embeds wireless channel physics into differentiable layers. Without retraining in unseen environments, GUIDE achieves 2.75x beamforming gain than the deep learning-based baseline FIRE with only a slight increase in inference time, and 1.39x beamforming gain than the strongest model-based baseline R2F2 while running over 1610x faster. 

---
# Personalized to Persuade: The Effects of Contextualization and Warmth on Trust and Reliance in Conversational AI 

**Authors**: Mert Yazan, Suzan Verberne, Frederik Bungaran Ishak Situmeang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31275)  

**Abstract**: Artificial Intelligence (AI) agents personalize their responses by tailoring explanations to users' backgrounds, interests, and prior interactions, referred to as contextualization. Personalization has been identified as a persuasive strategy in politics or in marketing. However, the persuasive effect of contextualization in everyday tasks, where users often lack prior knowledge, remains unclear. We conducted a $2\times2$ between-subjects experiment ($N = 380$) examining how contextualization, combined with conversational warmth, shapes reliance and persuasiveness of an AI assistant arguing against expert recommendations. Our findings reveal that contextualization reduces the persuasive power of AI, but its combination with warmth restores persuasiveness through a crossover interaction. Reliance on AI is present across conditions and is invariant to the conversational design. Trust strongly predicts both persuasion and reliance, yet neither contextualization nor warmth operates through trust. AI literacy decouples trust from behavior: more literate users report lower trust in the assistant, yet are more persuaded and more reliant on its advice. These results suggest that users are prone to deferring to AI agents over human expert judgment; however, interface-level conversational design choices have a limited role in shaping the behavior. 

---
# Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation 

**Authors**: Nan Bao, Yifan Zhao, Wenzhuang Wang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31266)  

**Abstract**: The layout-to-image (L2I) task enables fine-grained control over image generation via object categories and spatial layouts. However, existing L2I methods yield fragmented and distorted generations under few-shot atypical settings. We term this failure as representation fragmentation, arising from a granularity mismatch that entangles semantic identity with visual details. To address this issue, we propose a representation-driven framework that disentangles semantics from primitives for robust few-shot adaptation. Specifically, Semantic Anchoring aggregates categorical semantics into anchors for stable identity, while Primitive Imbuing models recomposable primitives for robust local detail modeling. Conceptual Steering further regulates optimization with a saliency-aware objective to preserve foreground semantic consistency. Extensive experiments demonstrate consistent improvements in the 5-shot regime over state-of-the-art L2I methods in both visual fidelity and alignment across diverse atypical domains. The source code is publicly available at this https URL. 

---
# Why Linear Recurrent Memory Works in Partially Observable Reinforcement Learning 

**Authors**: Yike Zhao, Onno Eberhard, Malek Khammassi, Ali H. Sayed, Michael Muehlebach  

**Link**: [PDF](https://arxiv.org/pdf/2605.31261)  

**Abstract**: The family of linear recurrent neural networks has shown strong performance as recurrent memory units in partially observable reinforcement learning. We provide a theoretical justification for their empirical effectiveness by constructing and studying two linear filters: (i) the first exactly reproduces the pre-softmax logits of the belief vector in a hidden Markov model (HMM) under a deterministic transition matrix, thereby serving as a sufficient statistic for optimal policy learning, (ii) the second achieves vanishing state-decoding error under a nearly deterministic transition matrix, thus reducing state ambiguity to near zero. The results extend to action-controlled HMMs, where the corresponding linear filters become time-varying with action-dependent dynamics. We illustrate our main results through numerical experiments and further show that the constructed linear filter serves as a strong feature extractor in a small reinforcement learning game. 

---
# ERGeoBench:A Comprehensive Benchmark for Embodied Reasoning and Geo-localization in Multimodal Large Language Models 

**Authors**: Kaiwen Xue, Tao Wei, Guoxin Zhang, Zhonghong Ou, Kaoyan Lu, Yu Feng, Yifan Zhu, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.31251)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong potential as embodied agents, yet embodied geo-localization remains underexplored due to the lack of fine-grained evaluation. We introduce ERGeoBench, a diagnostic benchmark for vision-driven embodied geo-localization. ERGeoBench evaluates models under three progressive settings -- single-view, panorama-view, and embodied-view -- where agents may actively acquire observations through sequential changes in yaw, pitch, and zoom. The benchmark contains 2,207 globally distributed street-view panoramas and measures four complementary capabilities: foundational perception, spatial awareness, common sense reasoning, and geo-localization reasoning. Evaluations of leading proprietary and open-source MLLMs show that current models can infer high-level geographic semantics, but still struggle with fine-grained perceptual operations, metric localization, and spatial consistency across views. We further observe that geo-localization is strongly correlated with the other capability dimensions, suggesting that accurate localization depends on integrated perception, spatial reasoning, and commonsense inference rather than isolated visual recognition. Overall, ERGeoBench provides a unified framework for diagnosing and advancing human-like embodied geo-localization. Project Page: this https URL 

---
# Entropic Projection Alignment: Estimating, Explaining, and Improving Model Performance Under Distribution Shift 

**Authors**: Salim I. Amoukou, Emanuele Albini, Tom Bewley, Saumitra Mishra, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2605.31250)  

**Abstract**: We propose a unified framework for addressing three key challenges of distribution shift: (1) estimating a model's performance on an unlabeled target domain, (2) explaining the shift by identifying the features responsible, and (3) improving the target domain performance. Our method, Entropic Projection Alignment (EPA), aligns the source distribution to the target by matching carefully selected moments while simultaneously minimising the KL divergence from the source. This formulation yields a unique closed-form solution for importance weights, achieving robustness through implicit variance control. Drawing on domain adaptation theory, we establish that moment matching is sufficient for reliable estimation and adaptation, avoiding the need for full density ratio recovery. Extensive experiments, together with strong theoretical guarantees, demonstrate that EPA consistently outperforms state-of-the-art baselines while offering substantial computational efficiency. 

---
# Learning Cardiac Latent Representations in Vectorcardiogram Space 

**Authors**: Bosong Huang, Panzhen Zhao, Zengxiang Li, Patricia Lee, Wei Jin, Alan Wee-Chung Liew, Ming Jin, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.31249)  

**Abstract**: Electrocardiography (ECG) is a cornerstone of cardiac assessment, making the learning of informative ECG representations fundamental to tasks ranging from disease diagnosis to clinical report generation. However, existing methods operate almost exclusively in the observable ECG signal space. In practice, the standard twelve-lead ECG represents multiple projections of the same underlying cardiac electrical activity from different spatial orientations. Therefore, representation learning in the ECG space inevitably introduces substantial redundancy, which may lead to spurious correlations and increased risk of overfitting. To address this and motivated by the Frank vectorcardiogram (VCG) model, we propose learning a unified latent representation of cardiac electrical activity directly in the VCG space. We introduce LVCG, the first general self-supervised representation learning framework designed to operate in this physically grounded latent space. By learning view-invariant latent VCG representations rather than lead-specific artifacts, VCG minimizes redundancy and improves generalization. LVCG generally outperforms ECG-space baselines across tasks, demonstrating enhanced robustness and generalization, especially in domain shift settings. 

---
# Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference 

**Authors**: Salim I. Amoukou, Saumitra Mishra, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2605.31239)  

**Abstract**: Bagging-based ensembles, most notably Adaptive Random Forests, are among the strongest performers for learning from data streams. A common denominator across these methods is their reliance on Hoeffding Trees as base learners, which grow decision trees incrementally by testing whether a candidate split is significantly better than its alternatives using concentration inequalities. Despite their empirical success, existing variants lack valid statistical guarantees. Current analyses rely on fixed-sample concentration bounds, while split decisions are made using data-dependent stopping rules, which invalidates their guarantees and can drive the probabilty of incorrect splits to one. We introduce a principled alternative based on anytime-valid inference. Our method provides: (i) anytime-valid control of false splits under arbitrary data streams, including non-stationary settings; (ii) finite commitment time under a predictive advantage; and (iii) under stationary i.i.d. data, risk is monotone decreasing and strictly improves at every split. Empirically, we evaluate both standalone trees and their use within Adaptive Random Forests on non-stationary streams. Our method improves performance while producing substantially smaller trees. 

---
# Beyond Classification: Dynamic Adapter Routing for Continual Multimodal Retrieval 

**Authors**: Alicja Dobrzeniecka, Filip Szatkowski, Sebastian Cygert, Szymon Lukasik, Bartlomiej Twardowski  

**Link**: [PDF](https://arxiv.org/pdf/2605.31229)  

**Abstract**: While retrieval is a core function of vision-language models, continually updating these models for retrieval tasks remains critically underexplored. Existing work often approaches continual retrieval through the lens of class-incremental learning (CIL), evaluating both standard CIL methods and retrieval-oriented adaptations in settings that may not fully capture the retrieval-specific dynamics. To address this, we introduce a new, principled evaluation framework for continual multimodal retrieval (CMR) spanning diverse visual domains, and systematically evaluate common approaches within this setting. Our empirical analysis shows that standard CIL methods fail to yield meaningful gains in our more challenging scenario. Therefore, we propose Dynamic Adapter Routing (DAR), a novel approach based on adapters selected through prototype-based routing and combined via model this http URL achieves superior performance over the previous baselines and demonstrates strong generalization under out-of-distribution evaluation. Our results highlights the unique challenges of CMR and encourages further research in this direction. 

---
# EchoRL: Reinforcement Learning via Rollout Echoing 

**Authors**: Jinhe Bi, Aniri, Minglai Yang, Xingcheng Zhou, Wenke Huang, Sikuan Yan, Yujun Wang, Zixuan Cao, Michael Färber, Xun Xiao, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.31228)  

**Abstract**: Reinforcement Learning with Verifiable Rewards is an effective route for post-training to strengthen the reasoning capability of large language models. However, as training proceeds, the learning signal can collapse thus makes the training gain become marginal and ineffective. Specifically, a growing fraction of prompts' rollouts become advantage-degenerated: all the self-generated rollouts show verified-success, making the standard deviation over their rewards be zero; accordingly each rollout's advantage becomes degenerated (zero) as well. Given such rollouts' advantages, the policy-gradient for model optimization eventually vanishes, capping the training performance. We argue that some of these rollouts still contain valuable learning signals but unfortunately omitted with the existing RLVR methods. In this paper, inspired through analyzing the entropy pattern behind golden trajectories produced by external expert models, we propose EchoRL for better exploiting the advantage-degenerated rollouts to further improve the training performance. EchoRL is a lightweight module that first identifies an EchoClip from verified-success rollouts based on their step-level entropy values, and then feeds this clip back as an auxiliary supervision signal in the RL objective. Extensive experiments across 10 benchmarks, 5 LLM backbones, and 4 popular RLVR post-training methods demonstrate that EchoRL consistently improves RLVR post-training with minimal overhead. 

---
# What changes after deployment? A survey on On-device Learning in TinyML 

**Authors**: Massimo Pavan, Luca Pezzarossa, Fabrizio Pittorino, Manuel Roveri, Xenofon Fafoutis  

**Link**: [PDF](https://arxiv.org/pdf/2605.31226)  

**Abstract**: Machine learning models on microcontroller-class devices (TinyML) face a fundamental challenge: post-deployment distribution change undermines static models. On-device learning (ODL) addresses this by running the learning process directly on the device. The existing literature has not characterized how distribution change occurs or how different change types require different solutions. Approximately 70 ODL works are surveyed under one principle: the distribution change regime. The survey analyzes how different types of distribution change influence the applications addressable on-device, the hardware employed, and the structure of the solutions. A persistent gap between methodological benchmarks and real-world deployment scenarios is also identified. 

---
# Comparing LLM-Based Conversational and Graphical Interfaces for Industrial Decision Tasks: An Exploratory Mixed-Methods Study 

**Authors**: Roberto Figliè, Simone Caputo, Alan Serrano, Tommaso Turchi, Daniele Mazzei  

**Link**: [PDF](https://arxiv.org/pdf/2605.31224)  

**Abstract**: The use of Generative AI Conversational User Interfaces (CUI) as a new way to access and analyze data is growing in all sectors, and the industrial one is no exception. There, large amounts of data produced by IoT devices are flowing through user interfaces and may require them a new adaptation to the new analyses needs of decision-makers. LLM-based CUIs are promising a new way to directly interact with those data through the directness of natural language and without the learning costs that every GUI design has. Moreover, the capabilities of LLMs and their agency open up the possibility to automate some tasks and help with the reasoning during decision-making activities. But are this promises well founded? We try to scope this general question with a mixed-approach study comparing a state-of-the-art dashboard with a conversational agent. A total of 20 participants used both interfaces to complete four simulated industrial decision tasks of varying complexity. We combined measures of mental workload, completion time, and decision accuracy with a post-study questionnaire and semi-structured interviews analyzed through thematic analysis. The findings suggest that the conversational agent can reduce interactional effort by supporting more direct access to information, while the dashboard remains valuable for overview and verification. However, these benefits may vary across tasks and require validation through larger-scale studies. 

---
# Shared Doubt: Zero-shot Cross-Lingual Confidence Estimation for Language Models 

**Authors**: Athina Kyriakou, Dennis Ulmer, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2605.31220)  

**Abstract**: Confidence estimation (CE), i.e. quantifying the reliability of a model's prediction, has attracted great interest in the context of large language models (LLMs). However, most studies focus on English, ignoring the multilingual reality of LLM usage, while many CE methods degrade or require retraining across languages. To address this gap, we investigate whether multilingual LLMs encode shared, language-transferable confidence features. We use a lightweight linear probe that predicts answer correctness directly from intermediate representations. Trained monolingually, the probe generalizes zero-shot to unseen, typologically diverse languages without target-language supervision. Learned layer weights and multiple ablations reveal that confidence features concentrate in middle layers across languages, suggesting a shared confidence subspace. While zero-shot cross-lingual performance depends on similarity to the source language, the probe provides a strong baseline without any retraining and compares favorably to other popular confidence estimation methods. 

---
# Benchmarking and Enhancing Text-to-Image Models for Generating Visual Representations in Early Arithmetic Education 

**Authors**: Junling Wang, Boqi Chen, Heejin Do, Mubashara Akhtar, April Yi Wang, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2605.31212)  

**Abstract**: AI systems are increasingly used to support educational content creation, yet it remains unclear whether they can generate outputs that faithfully represent the pedagogical concepts they are intended to teach. Thus, we introduce equation-to-visual generation, a task that, in contrast to conventional image generation, requires producing pedagogically meaningful visuals from arithmetic equations while precisely preserving their numerical and relational structure. Informed by interviews with teachers and an analysis of educational materials, we construct E2V-Bench, a benchmark spanning four pedagogically grounded visual types, along with automatic metrics for evaluating visual correctness. Our evaluation reveals that recent text-to-image (T2I) models frequently fail on this task, with errors dominated by incorrect object counts and broken relational structure. Building on this, we explore benchmark-guided enhancement strategies. These strategies improve representative models, while the remaining gap calls for stronger numerical and relational grounding in future T2I models. 

---
# Simulation of collision avoidance behavior in crowd movement by data-driven approach 

**Authors**: Xuanwen Liang, Eric Wai Ming Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.31210)  

**Abstract**: Crowd movement simulation is essential for pedestrian safety management and facility layout optimization. Data-driven models enhance trajectory prediction accuracy under Euclidean metrics, yet they suffer from excessively high collision rates, especially in bidirectional and multidirectional flows. In this paper, we establish a novel data-driven crowd simulation model that incorporates the pedestrian collision mechanism into the loss function to reduce collisions. A new lateral-acceleration-based collision loss function and a Voronoi-based motion feature extraction approach are proposed. The model is based on a Generative Adversarial Network (GAN) architecture and is termed CPGAN (Collision-Penalized GAN). We evaluate CPGAN in bidirectional flow scenarios, which involve frequent collision avoidance behaviors. Results show that the proposed lateral-acceleration-based collision loss significantly reduces opposite-direction pedestrian collision rates to levels comparable with controlled experiments. CPGAN effectively simulates bidirectional flow, reproducing lane formation and N-t curves. The research outcomes can provide inspiration for integrating pedestrian dynamics mechanisms into loss functions in data-driven crowd simulation. 

---
# MAECO-Lite: Modular Ontology for Dynamic Malware Analysis 

**Authors**: Zekeri Adams, Peter Švec, Ján Kľuka, Roderik Ploszek, Monday Onoja, Štefan Balogh, Martin Homola  

**Link**: [PDF](https://arxiv.org/pdf/2605.31199)  

**Abstract**: Capturing dynamic malware behavior in a practical but still semantically precise manner remains a significant challenge in cyber threat intelligence. While standards such as MAEC and STIX provide widely adopted vocabularies for describing malware artifacts and observations, they represent data with considerable complexity in structures that often obscure important ontological distinctions. In particular, they tend to conflate enduring malware artifacts with the events generated during execution, thereby flattening distinctions that are central in foundational standards for ontology design. In this paper, we conduct a foundational ontological analysis of core MAEC and STIX constructs relevant to dynamic malware analysis relying on Unified Foundational Ontology (UFO) as a theoretical lens. Our analysis reveals some ontological mismatches arising from the conflation of artifacts, dispositions, and runtime events in MAEC and STIX that complicate coherent representation of dynamic malware behavior and, from a practical perspective, limit the ability to reason about execution traces. Based on these insights, we propose MAECO-Lite, a lightweight ontology designed to represent data and operationalize their processing for dynamic malware analysis. The ontology adopts a modular structure centered on samples, processes, actions, system artifacts, and MITRE ATT&CK Techniques, while maintaining a clear separation between enduring entities and runtime events. An initial evaluation using description logic concept learning algorithms shows that the simplified ontology significantly improves learning performance, demonstrating that ontologically grounded modelling can enhance both semantic clarity and computational usability. 

---
# Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration 

**Authors**: Jun Wang, Xiaohao Xu, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31196)  

**Abstract**: Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision before contact. Across three frontier or robotics-oriented VLMs and nine visual representations, current models remain far from reliable: the best average Macro-F1 stays below 50\%, explicit depth is not automatically transformed into robot-body collision evidence, and robot--scene contact is consistently harder than human-contact risk. TouchSafeBench reveals a central limitation of embodied VLMs: visual fluency does not imply physical accountability. Reliable robot safety monitors will need representations that explicitly bind viewpoint, robot morphology, metric geometry, and future collision. We will release the benchmark upon acceptance. 

---
# Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines 

**Authors**: Mikkel Godsk Jørgensen, Lars Kai Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31183)  

**Abstract**: Sparse Autoencoders (SAEs) have been seen as a promising avenue for exploring the internals of Large Language Models (LLMs) and for steering model output generation. When AxBench - a model steering benchmark - was introduced in Wu et al. (2025), SAEs did not seem to live up to their original hype due to poor steering performance relative to a set of simple baselines. This work serves as a partial rebuttal for Sparse Autoencoders and suggests that the results of Wu et al. (2025) did not do them full justice. We find that Sparse Autoencoders can, in fact, perform close to on par with the reference LoRA performance on the AxBench benchmark, when features are selected and labelled with our supervised pipeline. We also find that our pipeline selects features that are surprisingly causal of their identified labels when using only its interpretability-based components. Lastly, we present evidence that high sparsity (low l0) may not be crucial for successful steering based on interpretability, which is in contrast to the earlier findings in Wang et al. (2025). 

---
# MindVoice: Reconstructing Intelligible Speech from Non-invasive Neural Signals with Pretrained Priors 

**Authors**: Guangyin Bao, Taiping Zeng, Jianfeng Feng, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.31173)  

**Abstract**: Reconstructing continuous speech from non-invasive neural recordings is a fundamental problem for probing human auditory perception and building safe, scalable speech brain-computer interfaces. Despite recent progress, intelligible reconstruction remains elusive, as non-invasive recordings are inherently noisy, spatially blurred, and only partially preserve information about perceived speech. Existing methods directly map neural activity to entangled speech representations before synthesizing waveforms with neural vocoders, resulting in spectral-similar but unintelligible results. To overcome these limitations, we introduce MindVoice, a neuro-to-speech reconstruction framework that uses pretrained models to compensate for the incomplete semantic and acoustic information in neural recordings. MindVoice disentangles reconstruction into two complementary pathways: one recovers high-level semantic content, while the other estimates fine-grained acoustic attributes. These inferred representations are then fused with powerful speech generation models and in-context voice cloning to synthesize natural and intelligible utterances. Extensive experiments on EEG and MEG demonstrate that MindVoice substantially outperforms existing methods on various metrics. These results show that pretrained priors provide a principled way to bridge the gap between noisy neural recordings and natural speech, highlighting a promising attempt for auditory neuroscience research and non-invasive speech brain-computer interfaces. 

---
# MIMO: Multilingual Information Retrieval via Monolingual Objectives 

**Authors**: Youngjoon Jang, Seongtae Hong, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.31171)  

**Abstract**: Multilingual Information Retrieval (MLIR) reflects real-world search environments in which queries and relevant documents may appear in different languages within a mixed-language corpus. However, existing embedding models are primarily optimized for Multi-Monolingual retrieval and their performance often degrades in MLIR settings. Moreover, directly applying conventional contrastive learning to MLIR can exacerbate language clustering and expose a trade-off between cross-lingual alignment and embedding uniformity. To address these limitations, we propose MIMO: Multilingual Information Retrieval via Monolingual Objectives, a two-stage framework that uses a stable English semantic space from a high-performing teacher model as an anchor. MIMO first initializes the student model's cross-lingual alignment through knowledge distillation, and then jointly optimizes distillation and cross-lingual contrastive learning to improve retrieval discrimination while preserving alignment. Extensive experiments show that MIMO consistently outperforms existing cross-lingual training baselines across various MLIR and Multi-Monolingual benchmarks. MIMO also remains competitive with off-the-shelf models of similar or larger parameter scales. Furthermore, our cross-lingual Alignment-Uniformity analysis clarifies the distinct roles of the two loss components and shows that their combination yields a favorable trade-off between alignment and uniformity. 

---
# Emergent Languages in Populations of Language Model Agents: From Token Efficiency to Oversight Evasion 

**Authors**: Stine Lyngsø Beltoft, William Brach, Federico Torrielli, Jacob Nielsen, Annemette Brok Pirchert, Filippo Tonini, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.31170)  

**Abstract**: Monitoring autonomous language model agents currently relies mostly on surface behavior. But what happens when agent populations invent new languages with the goal of avoiding human oversight. Here, we study the emergent languages on Moltbook. For this, we build upon the Moltbook Files dataset and apply a two-stage approach consisting of a rule-based heuristic (about 6000 matches) followed by zero-shot classification (518 kept). The resulting categories include token efficiency (166), new natural languages (106), and oversight evasion (59). We conduct both quantitative and qualitative analyses. Our results show that posts proposing new languages for avoiding oversight are judged by DeepSeek-3.2 as being less aligned than the other categories and that all languages can be learned by other language models in-context merely from a description of the language. Moreover, manually studying exemplary cases reveals surprisingly sophisticated steganographic protocols like embedding hidden messages in natural language. Although we cannot be certain about the extent of autonomy in ideation of these languages, our results add up to the evidence that monitoring surface behavior may soon be insufficient for retaining control over agent populations. 

---
# D$^3$: Dynamic Directional Graph-Constrained Data Scheduling for LLM Training 

**Authors**: Yuanjian Xu, Jianing Hao, Guang Zhang, Zhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.31164)  

**Abstract**: Training data plays a central role in large language models (LLMs) optimization, motivating extensive research on data scheduling strategies. Most existing approaches concentrate on adjusting the overall data distribution but neglect the underlying interactions between samples during training. However, we argue that such interactions cannot be overlooked, as real-world data samples frequently exhibit directional influences on each other, making the training order crucial. Intuitively, we can prioritize train-units with greater influence to improves learning efficiency. In this work, we propose $D^3$, a Dynamic Directional graph-constrained Data scheduling framework. $D^3$ formulates the complex interactions among train-units as a dynamic influence graph, where edges represent loss-based dependencies. It then solves a constrained optimization problem over this graph to derive the training order, which ensures that the data sequence respects the evolving information flow throughout training. Our approach is theoretically motivated and yields consistent improvements over existing data scheduling methods across both pre-training and post-training phases. Furthermore, for scalability, $D^3$ also employs an efficient approximation algorithm that keeps the additional computational overhead within a manageable range. For future research, the code is available at this https URL. 

---
# Trust-Region Behavior Blending for On-Policy Distillation 

**Authors**: Daniil Plyusov, Alexey Gorbatovski, Alexey Malakhov, Nikita Balagansky, Boris Shaposhnikov, Daria Korotyshova, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2605.31159)  

**Abstract**: On-policy distillation (OPD) trains a student on prefixes sampled from its own policy while matching a stronger teacher. This addresses the prefix mismatch of offline distillation, but early student rollouts can still be poor, placing teacher supervision on weak or low-quality prefixes. We propose Trust-Region behavior Blending (TRB), a warmup method that replaces the early rollout policy with the closest-to-teacher behavior policy inside a student-centered KL trust region, while keeping the per-prefix reverse-KL OPD loss unchanged. The KL budget is annealed to zero, so training returns to pure student rollouts after warmup. Across two math-reasoning distillation settings, TRB attains the strongest average among the compared methods. 

---
# Developing a UXR Point of View for Cognitive Accessibility in Mobile Learning with Generative AI 

**Authors**: Fatima Ahmad Muazu, Festus Adedoyin, Huseyin Dogan, Abiodun Adedeji, Melike Akca, Olumuyiwa Ayorinde  

**Link**: [PDF](https://arxiv.org/pdf/2605.31149)  

**Abstract**: This study investigates how UX research (UXR) principles, combined with Large Language Model (LLM)-supported analysis, can be used to improve the quality of requirements for mobile learning systems designed for learners with cognitive disabilities. Using the UXR Point-of-View (PoV) pyramid as a methodological framework, the study progressed through four stages: foundational structuring of psychological, behavioral, and design layers; structured validation using the DeLone and McLean Information Systems Success Model and Quality Function Deployment (QFD); insight consolidation through the development of nine Cognitive Accessibility UXR Play Cards; and stakeholder-specific PoV articulation to support interdisciplinary communication. LLM-supported synthesis was integrated to assist in theme clustering, requirement refinement, and hypothesis formulation under human oversight. Findings suggest that many usability and engagement challenges in mobile learning originate from ambiguous or under-specified requirements rather than interface design alone. By embedding cognitive accessibility principles into measurable and technically traceable requirements, the proposed Cognitive Accessibility UXR Playbook provides a structured pathway for aligning theory, system architecture, and stakeholder strategy. 

---
# SpatialAct: Probing Spatial Reasoning-to-Action Capabilities of VLM Agents in 3D Scenes 

**Authors**: Tianhui Liu, Jie Feng, Zhiheng Zheng, Shengyuan Wang, Yiming Guo, Yanxin Xi, Hangyu Fan, Yong Li, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2605.31148)  

**Abstract**: Humans can effortlessly perceive spatial layouts, form cognitive representations, reason about spatial relations, and translate such reasoning into actions in everyday 3D environments. Although recent vision-language models (VLMs) have shown promising performance on observation-conditioned spatial perception and reasoning tasks, it remains unclear whether they can build coherent spatial understanding, act upon it, and refine their actions through multi-turn feedback. To study this problem, we introduce \textbf{SpatialAct}, a simulator-grounded benchmark for probing \textit{action-conditioned spatial reasoning} in 3D scenes. Starting from the most challenging setting, Multi-turn Interactive Refinement, we further design its decomposed counterpart, Single-step Error Detection and Fix, together with five fundamental spatial ability tasks to diagnose the underlying causes of model failures. Experiments reveal a clear reasoning-to-action gap: current VLMs can perform well on isolated spatial reasoning tasks, but struggle to maintain coherent spatial beliefs and produce reliable actions during multi-turn feedback, substantially underperforming humans. These results suggest that current VLM agents still lack robust spatial state tracking under action-induced environment changes, even when low-level control is abstracted away. 

---
# Developing a Culturally Grounded, AI-Augmented UX Research Point of View (POV): An Exemplar Case Study from Telemedicine Dementia Care 

**Authors**: Abiodun Adedeji, Huseyin Dogan, Festus Adedoyin, Michelle Heward, Melike Akca, Emmanuel Oluwatosin Oluokun, Fatima Ahmad Muhazu, Olumuyiwa Ayorinde  

**Link**: [PDF](https://arxiv.org/pdf/2605.31147)  

**Abstract**: User Experience Research (UXR) Points of View (POVs) distil complex and often fragmented research evidence into actionable perspectives that guide how teams interpret user needs, frame design decisions, and align stakeholders. Although POVs are widely used in industry practice, there are few published examples that explicitly document how POVs are constructed, particularly in culturally sensitive and low-resource contexts. This paper presents an exemplar case study demonstrating how a culturally grounded, AI-augmented UXR POV was developed to inform TeleDeCa, a telemedicine dementia care framework for family caregivers in Nigeria. Building on the UXR POV Playbook and pyramid framework, we illustrate how mixed-methods research, hypothesis generation, and ontology-based modelling can be combined to form a defensible POV without requiring a fully finalised system or validated outcomes. Generative AI (GenAI) is integrated across the UXR POV framework as a bounded research collaborator, supporting synthesis, hypothesis exploration, and narrative construction while preserving human judgment, ethical accountability, and cultural sensitivity. The contribution of this paper lies in the extraction of reusable Play Cards and a Play that extend the UXR POV Playbook and serve as exemplar material for the CHI 2026 workshop on developing AI-powered UXR POVs. 

---
# From Evidence to Design: Developing an AI-Augmented UX Research Point of View for Digital Wellbeing in Emergency and Public Safety Contexts 

**Authors**: Olumuyiwa Ayorinde, Huseyin Dogan, Festus Adedoyin, Nan Jiang, Emmanuel Oluokun, Abiodun Adedeji, Melike Akca  

**Link**: [PDF](https://arxiv.org/pdf/2605.31146)  

**Abstract**: This paper investigates how User Experience Research (UXR) methods can be combined with AI-supported analysis to develop clearer design direction for digital wellbeing interventions targeting Emergency and Public Safety Personnel (EPSP). EPSP work in high-stress, shift-based environments where cognitive fatigue and unpredictable schedules reduce engagement with conventional wellbeing tools. Using the UXR Point-of-View (PoV) framework, this study applied an AI-supported literature analysis process to identify recurring psychological, behavioural, and design patterns. Behaviour Change Techniques and Persuasive Technology principles were integrated throughout interpretation to connect evidence with practical design reasoning. The process resulted in a UXR PoV Pyramid, nine UXR Play Cards, and stakeholder focused PoV narratives. Findings show that effective wellbeing systems for EPSP must minimise cognitive effort, adapt to operational context, and prioritise psychological safety. The work demonstrates how AI can assist large-scale evidence interpretation while human researchers maintain responsibility for contextual judgement and design direction. 

---
# FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization 

**Authors**: Mohammed Asad Karim, Vinay Kumar Verma  

**Link**: [PDF](https://arxiv.org/pdf/2605.31145)  

**Abstract**: In-context localization (ICL) seeks to localize a target object specified by a small set of support examples in a query image, operating on the fly without training or parameter updates. Despite rapid advances in vision-language models (VLMs), achieving category-agnostic and visually grounded ICL remains an open problem, even though it is essential for applications such as image editing, personalized visual search, and retrieval. Existing methods are fragile and rely on explicit category supervision, which not only limits applicability in realistic settings with unnamed or instance-specific objects but also introduces category bias that steers predictions toward semantic priors rather than visual evidence. We introduce a two-stage training framework that explicitly optimizes in-context attention between support bounding boxes and query images without category supervision. We further refine localization via reinforcement learning using Group Relative Policy Optimization (GRPO) to directly minimize localization error. This formulation enforces visual correspondence over semantic priors, yielding robust instance-level localization. Empirically, a 7B-parameter model trained with our objectives outperforms models up to 72B parameters, demonstrating that context-aware localization objectives can surpass scaling alone. Comprehensive ablations validate the contribution of each component. 

---
# Extending the UXR Point of View Pyramid: A Generative AI-Augmented Methodology for Human-Centred AI Systems 

**Authors**: Festus Fatai Adedoyin, Huseyin Dogan, Melike Akca, Abiodun Adedeji  

**Link**: [PDF](https://arxiv.org/pdf/2605.31143)  

**Abstract**: Rising household debt and cost-of-living pressures in the United Kingdom have intensified the role of AI-driven financial technologies in mediating credit assessment, repayment structuring, and debt support services. These systems increasingly shape consequential financial decisions, yet they operate within complex socio-technical environments characterised by regulatory constraint, algorithmic opacity, and heightened vulnerability risk. User Experience Research (UXR) Points of View (PoVs) are critical in translating heterogeneous research evidence into strategic direction for product and governance decisions. However, the existing UXR PoV framework was not designed for AI-mediated financial systems where interpretability, fairness, and accountability are central. This paper extends the UXR PoV pyramid into an AI-augmented methodological framework for Human-Centred AI debt management technologies in the UK financial services context. We formalise (1) an AI-Augmented PoV Pyramid, (2) a structured prompt architecture for synthesis and hypothesis generation, and (3) an AI-enabled Playbook Card system that embeds Generative AI into UXR workflows while preserving traceability and ethical oversight. Generative AI is positioned not as an analytic authority, but as an epistemic support mechanism subject to human validation and regulatory awareness. By grounding the framework in debt management technologies, including affordability assessment, repayment planning, and financial stress prediction systems, this work advances UXR methodology for high-stakes financial AI environments and contributes to the evolution of responsible, AI-powered UXR practice within the CHI community. 

---
# On the Robustness of Multilingual Text Embedding Rankings Across Learning Tasks, Languages, and Benchmark Datasets 

**Authors**: Ana Gjorgjevikj, Barbara Koroušić Seljak, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2605.31142)  

**Abstract**: Large-scale multilingual text embedding models play crucial role in both research and industry, yet their behavior in language-specific, multi-task settings remains insufficiently understood. Although benchmarking platforms such as MTEB report results across more than 250 languages, conclusions about model superiority often depend on implicit choices of dataset compositions and performance aggregation methods. To address this gap, we present a meta-study of multilingual model performance robustness in MTEB, applying a diverse set of multi-criteria decision-making ranking schemes and introducing two robustness indicators: dataset-composition robustness (sensitivity of rankings to changing dataset compositions) and ranking-scheme robustness (sensitivity to aggregation method change). They enable systematic sensitivity analysis of whether benchmarking conclusions remain stable under different evaluation designs. We conduct an in-depth analysis on five languages (English, French, German, Hindi, and Spanish) across nine tasks (e.g., classification, clustering, retrieval) and release results for approximately 230 additional languages. The task-specific analyses show that large-scale LLM-based models are often robust top performers, though not uniformly (e.g., in retrieval task), while task-agnostic results reveal that only a small subset of models remains consistently strong across tasks, ranking schemes, and data subsamples. 

---
# Developing an AI-Powered UX Research Point of View for Digital Health in A Regulatory Context: An Exemplar Case from MSM and Transgender HIV Care in Nigeria 

**Authors**: Emmanuel Oluwatosin Oluokun, Festus Fatai Adedoyin, Huseyin Dogan, Nan Jiang, Melike Akca, Abiodun Adedeji, Olumuyiwa Ayorinde, Fatima Ahmad Muazu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31138)  

**Abstract**: User Experience Research (UXR) in a legal and regulatory contexts presents unique challenges that require specialised approaches to protect vulnerable populations whilst generating actionable insights. Digital consultation, appointment booking, and medication delivery platforms show promise for extending care access; however, their real-world effectiveness is curtailed by an absence of theoretically grounded user experience research (UXR) methodologies that adequately account for the psychosocial conditions of these populations. This paper introduces a Generative AI-augmented UXR methodology, grounded in the UXR Point of View (PoV) Playbook, to guide the design of psychologically safe, low-cognitive-load digital health interventions for MSM and transgender individuals living with HIV/AIDS in Nigeria. Drawing from empirical research involving co-design workshops, thematic analysis, and requirements engineering, the methodology is operationalised through a four-stage UXR process encompassing AI-supported hypothesis generation, foundational planning, insight generation via Building Blocks, and the construction of stakeholder-specific PoV narratives. This process results in ten theory-informed UXR Play Cards that translate psychological mechanisms and empirical findings into actionable design guidance. Each play contains actionable tasks, AI-augmented approaches, and ethical guardrails tailored for research with marginalised populations. The output is a set of ten theory-informed UXR Play Cards translating psychological insight and empirical evidence into actionable design guidance. The core contribution is a replicable, stigma-aware, and privacy-centred framework for responsible GenAI use in UXR practice, advancing human-centred digital health design for marginalised communities. 

---
# UXR PoV for Neuroinclusive Emotion Regulation 

**Authors**: Melike Akca, Mona Giff, Deniz Cetinkaya, Huseyin Dogan, Stephen Giff  

**Link**: [PDF](https://arxiv.org/pdf/2605.31131)  

**Abstract**: Attention-deficit/hyperactivity disorder (ADHD) is a psychiatric disorder which presents itself in individuals through patterns of developmentally inappropriate levels of inattentiveness, hyperactivity, and impulsivity, with difficulties in decision making and emotional regulation (ER). Although digital and AI-based interventions have expanded access to ER support, many existing systems remain limited by weak theoretical integration, insufficient accommodation of neurodiversity, and a lack of structured user experience research (UXR) methodologies, that bridge psychological insight with design practice. This paper introduces a Generative AI-augmented UXR methodology, grounded in the UXR Point of View (PoV) Playbook, to support the design of emotionally intelligent and Neuroinclusive digital ER interventions for adults with ADHD. The approach integrates empirical evidence with established psychological frameworks Dialectical Behaviour Therapy (DBT), Self-Determination Theory (SDT), and the COM-B behavioural model and leverages Generative AI as a co-analytic tool to support synthesis, hypothesis formation, and design articulation. The methodology is operationalized through a four-stage UXR process encompassing AI-supported hypothesis generation, foundational planning, insight generation via Building Blocks, and the construction of stakeholder-specific PoV narratives. This process results in a set of ten theory informed UXR Play Cards that translate psychological mechanisms and empirical findings into actionable design guidance. The primary contribution of this work is a replicable, bias-aware framework for integrating Generative AI into UXR practice, advancing human-centred and Neuroinclusive approaches to digital mental health design. 

---
# Not All Synthetic Data Is Yours to Learn From 

**Authors**: Sina Alemohammad, Li Chen, Richard G. Baraniuk, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31126)  

**Abstract**: Can a language model improve from plain text sampled from itself, with no prompts, no teacher, no verifier, and no reward model? Yes, but only when the synthetic corpus is compatible with the student, a relational property of the source-student pair rather than an intrinsic property of the data. We call this the latent capability resurfacing hypothesis: weak self-training can amplify capabilities already present in the pretrained model, but only under this compatibility condition. We study this in the minimal setting of prompt-free unconditional self-training, where base language models are fine-tuned on text generated from the BOS token alone, with no task specification or external supervision. We report three findings. First, synthetic utility is relational rather than intrinsic: self-generated data is the most effective source, same-lineage transfer outperforms stronger but differently trained sources, and cross-family transfer is substantially weaker. Second, common intrinsic proxies fail: neither benchmark-level semantic similarity nor average per-token likelihood under the student predicts which corpora help. Third, this regime produces a surprising byproduct. In controlled Pythia experiments, capability and verbatim memorization decouple: benchmark utility is preserved or improved while held-out exact-match extraction drops by over 95 percent, with no forget set, privacy objective, or targeted unlearning. Together, these results suggest that prompt-free self-training works by amplifying what the student already knows, not by importing structure from the data. They also reveal a regime in which capability and verbatim memorization can be separated without any explicit unlearning objective. 

---
# TARIC: Memory-Augmented Traversability-Aware Outdoor VLN under Interrupted Semantic Cues 

**Authors**: Tianle Zeng, Hanjing Ye, Jianwei Peng, Jingwen Yu, Hanxuan Chen, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31121)  

**Abstract**: Outdoor vision-language navigation (VLN) in long-range, open-world environments is frequently disrupted by semantic-cue interruptions, where informative goal cues become sparse, occluded, or leave the field of view. Once such cues disappear, agents enter a cue-free phase and often degrade into backtracking, oscillatory headings, or aimless exploration. While memory-based methods attempt to bridge these gaps, they often fail under traversability-driven detours: the remembered cue direction may be infeasible, forcing detours that prolong cue-free phases and gradually render robot-centric cues stale and implicit histories blurred. This makes traversability a stability condition for maintaining goal-directed guidance, rather than merely a local safety concern.
We propose a unified outdoor VLN framework that survives semantic-cue interruptions by maintaining traversability-consistent executable guidance throughout prolonged cue-free phases. Specifically, our method extracts semantic bearings from visibility-gated goal or exploration cues and grounds them into executable headings using a real-time near-field traversability profile, providing goal-consistent feasible guidance beyond reject-only safety filtering. To prevent guidance degradation during detours, we lift intermittent 2D evidence into a world-aligned 3D cue memory with an uncertainty-aware readout mechanism, ensuring guidance remains continuously reachable and stable as the robot moves.
We evaluate the framework on quadrupedal and wheeled platforms over 600--1000 m routes. Our method improves simulation success rate by over 10 percentage points over the strongest baseline and achieves a real-world success rate of 40%, compared to 17.5% for the strongest baseline, with substantially higher robustness during prolonged cue-free intervals. 

---
# SWIM: Single-Instance Whole-Body Imitation for swiMming 

**Authors**: Binglun Wang, Edmond S. L. Ho, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31120)  

**Abstract**: We propose a new method for synthesizing physically-based swimming motions. Physically-based character animation aims to generate physically valid, controllable, and natural-looking motions which can respond to unexpected disturbances, where one dictating factor of difficulty is the complexity of the task, especially the level of sophistication of the required interactions with the environment. Existing research has succeeded in various tasks in static and dynamic environments. We push the difficulty further to swimming, which requires full-body coordination and continuous interactions with fluids, a new level of complexity when it comes to interacting with the environment. This complexity imposes challenges in learning control under volatile environmental forces, generalizing control to different environments and swimming styles, lack of data references, and prohibitively slow physical simulation which is inevitable during control learning. To this end, we propose SWIM, a new imitation method for swimming motions, which can learn from a single swimming motion and generalize to unseen environments, body conditions, and swimming styles. Extensive evaluation and comparison demonstrate that SWIM is data-efficient, stable, robust, and generalizable, outperforming alternative methods across multiple classes of tasks and metrics. 

---
# KnowledgeGain: Evaluating and Optimizing Science News Generation for Reader Learning 

**Authors**: Dominik Soós, Meng Jiang, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.31099)  

**Abstract**: Science news is an important medium to communicate discoveries between the research communities and the public. Yet, most metrics for generated or summarized text evaluate semantic similarity and factual consistency, but do not measure how much knowledge readers learn from the news. We introduce KnowledgeGain, a metric that evaluates the quality of science news by measuring how much knowledge readers gained after reading it. To evaluate the metric, we first performed a controlled human study and showed that the metric successfully captures the differential knowledge gained by human readers reading different types of science media. The data allowed us to calibrate a prompt-only LLM reader simulator. We use it to rank and filter candidate articles before human evaluation. A second human study shows that articles selected with this simulator improve post-reading accuracy and normalized KnowledgeGain over a strong generation baseline. Our work is a step toward generating science news that better meets the knowledge and comprehension goals of Bloom's Taxonomy. 

---
# SpecDB: LLM-Generated Customized Databases via Feature-Oriented Decomposition 

**Authors**: Yunkai Lou, Longbin Lai, Shunyang Li, Zhengping Qian, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.31097)  

**Abstract**: Mainstream relational databases ship a uniform feature set across deployments, although individual workloads exercise only a fraction of the available subsystems. We investigate whether a database can instead be generated on demand with a feature set matched to the target workload. We present SpecDB, a system that uses large language models (LLMs) to synthesize customized relational databases. We survey 9 production systems and decompose them into 10 functional modules, each further divided into implementation variants. To capture cross-module dependencies, including cases where implementations in disjoint subtrees must be co-designed, we adopt the FODA feature model and extend it with a cooperate edge, yielding a dependency graph DBGraph. SpecDB operationalizes DBGraph through a layered module-construction pipeline in which each module is generated, validated, and integrated by a dedicated subagent (driven by three inner agents: Main, Tester, Architect), and a Refining Agent that iteratively repairs and tunes the assembled database against a user-supplied refining harness with read-only access to existing database source code. A companion selection component translates a natural-language workload description into a set of implementation variants, providing an end-to-end pipeline from workload description to deployable database. We evaluate SpecDB on TPC-C with BenchmarkSQL. The generated database (23,779 lines of Rust) completes 60-minute TPC-C at 1 and 10 warehouses with zero errors. At 10 warehouses it reaches tpmC=130, compared to 128 for PostgreSQL and 127 for MySQL, with comparable latency at ~3% of their code size. Because the agent operates at module-specification level rather than product source, it can in principle combine techniques across system boundaries. Paired with falling LLM costs, generating a purpose-built database for a target workload is becoming straightforward. 

---
# Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation 

**Authors**: Erik Großkopf, Soumya Snigdha Kundu, Hendrik Möller, Nicolas Münster, Mehdi Astaraki, Paula Tamara Buzduga, Kerstin Ritter, Benedikt Wiestler, Jan Kirschke, Jonathan Shapey, Tom Vercauteren, Florian Kofler  

**Link**: [PDF](https://arxiv.org/pdf/2605.31094)  

**Abstract**: The Panoptic Quality (PQ) metric is the standard for jointly evaluating instance and semantic segmentation. However, its original definition relies on a One-to-One matching between predicted and ground truth segments, which is only straightforward when the IoU threshold exceeds 0.5. Below 0.5, multiple matching strategies emerge in a poorly explored problem space. We systematically elucidate this space by recasting segment matching as a constrained bipartite assignment problem. Independently bounding the prediction- and ground-truth-side degrees yields four matching strategies: One-to-One, Many-to-One, One-to-Many, and Many-to-Many. We show that the first three are well-defined within the PQ framework, while Many-to-Many falls outside it. These strategies become relevant when instances are fragmented, adjacent objects are difficult to delineate, or annotations are noisy. Central to our framework is a vertex-based accounting of TP, FN, and FP, anchored to ground truth and predicted segments rather than to matching edges. We further show that the framework extends naturally to part-aware panoptic segmentation, and we explore part-aware evaluation on biomedical data. Across configurable case studies we report how different combinations of thresholds and matching strategies behave in practice. We release a unified open-source package built on Panoptica. It exposes Voronoi-based region-wise analysis, part-aware evaluation, and Area Under Threshold Curve computations as configurable options. 

---
# On Revisiting Entropy for Identifying Mislabeled Images 

**Authors**: Chunlei Li, Zixuan Zheng, Yilei Shi, Guanglu Dong, Pengfei Li, Jingliang Hu, Xiao Xiang Zhu, Lichao Mou  

**Link**: [PDF](https://arxiv.org/pdf/2605.31090)  

**Abstract**: Mislabeled samples in training datasets severely degrade the performance of deep networks, as overparameterized models tend to memorize erroneous labels. We address this challenge by proposing a novel approach for mislabeled data detection that leverages training dynamics. Our method is grounded in the key observation that correctly labeled samples exhibit consistent entropy decrease during training, while mislabeled samples maintain relatively high entropy throughout the training process. Building on this insight, we introduce a signed entropy integral (SEI) statistic that captures both the magnitude and temporal trend of prediction entropy across training epochs. SEI is broadly applicable to classification networks and demonstrates particular effectiveness when integrated with contrastive language-image pretraining (CLIP) architectures. Through extensive experiments on four medical imaging datasets -- a domain particularly susceptible to labeling errors due to diagnostic complexity -- spanning diverse modalities and pathologies, we demonstrate that SEI achieves state-of-the-art performance in mislabeled data identification, outperforming existing methods while maintaining computational efficiency and implementation simplicity. Our code is available at this https URL. 

---
# A Pilot Study on Curator-Guided Multilingual Art Description for Blind and Low-Vision Audiences with Small Vision-Language Models 

**Authors**: Iosif Tsangko, Andreas Triantafyllopoulos, George Margetis, Ioana Crihana, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.31080)  

**Abstract**: Blind and low-vision (BLV) audiences remain underserved by visual art descriptions, particularly across languages and in museum settings where privacy and intellectual-property constraints may favour small on-premise vision-language models (VLMs). This pilot study investigates curator-guided multilingual art description with Qwen2.5-VL-3B-Instruct for German, Romanian, and Serbian. We construct a parallel BLV-oriented caption corpus from artwork images and metadata, and compare language-specific LoRA adapters with a single multilingual adapter under a fixed backbone and training budget. Evaluation combines automatic lexical and embedding-based metrics with an LLM-as-Judge protocol calibrated against a small Romanian BLV pilot study. Under our pilot setup, language-specific adapters show more stable controllability and visually grounded description quality for Romanian and Serbian, while multilingual adaptation remains competitive in German. We frame these findings as deployment-oriented evidence for small on-premise VLMs, and highlight the need for larger BLV user studies and broader language coverage before drawing general conclusions about multilingual accessibility. 

---
# DRIFT: Joint Channel Estimation and Prediction Towards Pilotless 6G Non-Terrestrial Networks 

**Authors**: Bruno De Filippo, Carla Amatetti, Alessandro Vanelli-Coralli  

**Link**: [PDF](https://arxiv.org/pdf/2605.31065)  

**Abstract**: Non-terrestrial networks (NTNs) are expected to play a pivotal role in sixth-generation (6G) systems by enabling ubiquitous connectivity and massive communication. In this context, channel prediction emerges as a key technique to improve the spectrum utilization efficiency by limiting the pilot overhead. However, many proposed predictors based on artificial intelligence (AI) are characterized by high inference complexity, posing challenges to onboard implementation. In this paper, we address the challenge of designing accurate yet computationally efficient channel prediction techniques tailored to low Earth orbit (LEO) NTNs, where strict power constraints limit model complexity, to enable spectral efficiency gains. We propose an iterative joint channel estimation and prediction framework in the context of 6G NTNs that significantly reduces pilot overhead by transmitting pilots only in the initial slot and relying on data-driven processing for subsequent slots. We introduce Data-driven Refinement and Iterative Forecast for wireless channel Tracking (DRIFT), a lightweight architecture that refines data-aided channel estimates and predicts future channel frequency responses with low computational cost and reduced error propagation. Two predictor variants based on convolutional and long short-term memory layers are investigated. Simulation results in an end-to-end simulation of an uplink LEO NTN scenario show that the proposed approach achieves up to 12% spectral efficiency gain compared to conventional pilot-based systems, with robustness to training-test mismatches and consistent performance across different channel models. Moreover, DRIFT requires fewer than 200k multiply-accumulate operations, making it suitable for on-board satellite implementation under stringent power constraints. 

---
# Fighting Numerical Hallucinations via Data-centric Compilation for Online Financial QA 

**Authors**: Hao Chen, Xing Tang, Qirui Liu, Weijie Shi, Shiwei Li, Fuyuan Lyu, Weihong Luo, Xiku Du, Xiuqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.31064)  

**Abstract**: Large Language Models (LLMs) have significantly advanced online data services, particularly in the domain of financial question answering (FinQA). However, such systems remain susceptible to numerical reasoning hallucinations, which critically undermine reliability in high-stakes financial applications. Although retrieval-augmented generation (RAG) has been widely adopted to ground responses in external knowledge, it introduces three persistent challenges: noise sensitivity, calculation fragility, and an auditability crisis. Existing model-centric approaches, which primarily focus on optimizing either the retriever or generator in isolation, still struggle to address these issues in an integrated manner. In this work, we pioneer a data-centric paradigm and propose a novel framework, the Data-centric Reasoning Compiler (DCRC). The framework operates through three cohesive phases: (1) adversarial data construction, which synthesizes training examples with controlled noise to teach robustness; (2) multi-stage training that cultivates a Data-centric Structuring Agent (DSA) capable of explicit evidence auditing and program synthesis; and (3) a compile-and-execute inference process, where the DSA transforms user queries and retrieved documents into verifiable, executable reasoning programs. This data-driven framework ensures faithful numerical reasoning by design. We conduct extensive experiments on established offline benchmarks and further validate our framework through deployment in a real-world online financial QA system. 

---
# STEP: Learning STructured Embeddings for Progressive Time Series 

**Authors**: Lucas Thil, Jesse Read, Rim Kaddah, Guillaume Doquet  

**Link**: [PDF](https://arxiv.org/pdf/2605.31061)  

**Abstract**: We present a novel method for learning interpretable representations of progressive time series, that is, data capturing irreversible state transitions such as degradation or task completion. Our approach uses a self-supervised contrastive objective to learn a low-dimensional latent space whose geometry is itself the interpretation: each observation becomes a point on a manifold anchored between two fixed orthogonal prototype vectors, and a trajectory becomes a path across that manifold. From this structure we read a latent compass, the polar coordinates ({\theta}, r) of the latent vector, in which {\theta} tracks the progression of the underlying state (e.g., from healthy to failed) and r identifies the active mode (e.g., the operating condition), without any proxy labels. We evaluate the approach against the state of the art on diverse domains, including industrial degradation, robotic tasks, and neural activity, validating three key capabilities: (1) end-state prediction, (2) multi-step forecasting, and (3) interpretable phase separation. Our method matches or improves over black-box counterparts on all of these while providing transparency about the underlying mechanisms. A simple linear regressor on top of the latent compass coordinates is competitive with deep architectures, direct quantitative evidence that the underlying state is encoded in a geometrically accessible form. 

---
# AnchorSteer: Self-Discovered Concept Injection for Structure-Preserving Music Editing 

**Authors**: Chih-Heng Chang, Keng-Seng Ho, Chih-Yu Tsai, Kuan-Lin Chen, Yi-Hsuan Yang, Jian-Jiun Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.31053)  

**Abstract**: Controllable music editing is to modify high-level attributes while strictly preserving rhythmic and melodic structures. However, this task is challenged by a semantic-structural entanglement: steering methods often degrade structure to achieve editing performance, while structural adaptors suppress semantic responsiveness. We propose AnchorSteer, a framework that disentangles this tension by coupling structural anchoring with self-discovered semantic steering. The proposed approach probes internal representations to extract interpretable, label-free concept vectors via a self-supervised reconstruction objective, isolating attributes without curated data. During editing, these portable, plug-and-play concept vectors are injected into diffusion hidden manifolds while a structural adaptor enforces consistency. Variants for unconditioned and conditioned injections are provided to balance robustness and semantic strength. Experiments on ZoME-Bench and subjective tests show that the proposed framework outperforms both steering-only and anchoring-only baselines, enabling significant semantic transformations with high-fidelity structural preservation. 

---
# Linear Ordering Problem: Time for a Change 

**Authors**: Fabrizio Fagiolo, Marco Baioletti, Valentino Santucci  

**Link**: [PDF](https://arxiv.org/pdf/2605.31051)  

**Abstract**: The Linear Ordering Problem (LOP) is a fundamental combinatorial optimization problem with important applications in areas such as economics, social choice, and machine learning. Its most prominent use is the triangulation of economic input-output tables, which helps identify critical industries in an economy. Most existing algorithms have been evaluated on benchmarks derived from outdated macroeconomic data, which no longer reflect the structure of contemporary economies. Furthermore, LOP instances often exhibit many distinct global optima that can differ substantially from one another, creating challenges for applications that rely on a single solution. To address these limitations, we introduce a novel benchmark suite derived from up-to-date real-world economic data and an algorithmic scheme that leverages state-of-the-art LOP metaheuristics to generate diverse sets of high-quality solutions, together with metrics for assessing both quality and diversity. Experiments were conducted to report results on the proposed benchmark suite under both the traditional single-solution setting and the newly introduced multi-solution scenario 

---
# Learning to Solve and Optimize by Evolving Code 

**Authors**: Veronika Semmelrock, Benedetta Strizzolo, Francesco Zuccato, Gerhard Friedrich, Patrick Rodler, Konstantin Schekotihin  

**Link**: [PDF](https://arxiv.org/pdf/2605.31049)  

**Abstract**: Combinatorial and optimization problems are fundamental to many industrial AI applications. Solving large-scale real-world instances of such problems typically requires careful problem formalization, specialized solvers, and expert-designed heuristics. Thus, experts need to specify not only what solutions are, but also how they are derived. By introducing the tool CHECKMATE, we show that algorithm generation via code evolution represents a paradigm shift by eliminating the need to formulate the how. CHECKMATE solely relies on the what. Specifically, a formal specification ensures solutions' correctness and enables systematic performance evaluation of the generated programs, while a natural language description guides the evolutionary process. The effectiveness of our method is demonstrated on selected problems from two industrial domains: configuration and scheduling. In all cases, the evolved algorithms consistently outperform state-of-the-art solvers. This underscores the potential of formal methods in guiding code evolution for automatically solving complex real-world problems. 

---
# Routing on the Stiefel Manifold: When Does Adaptive Subspace Selection Help for Cross-Domain EEG Decoding? 

**Authors**: Isabella Costa Maia, Pedro L. C. Rodrigues, Salem Said, Marco Congedo  

**Link**: [PDF](https://arxiv.org/pdf/2605.31043)  

**Abstract**: Cross-domain EEG decoding remains challenging despite advances in Riemannian deep learning: covariance matrices from different subjects occupy systematically distinct regions of the SPD manifold, yet existing domain adaptation methods either require target-domain calibration data or learn subject-specific components that cannot generalise across domains. We propose dynamic Stiefel routing: a pool of $K$ expert projection filters on the Stiefel manifold, each specialised for a different region of the SPD manifold, with each input covariance routed to the most appropriate filter via cross-attention, adapting the subspace projection per sample. A central finding is that this approach, implemented naively, provably collapses to ensemble averaging: when routing weights are uniform, the adaptive filter reduces exactly to an equal-contribution combination of experts, indistinguishable from a single fixed filter. Three structural properties break this degeneracy: a symmetric anchor $W_{\mathrm{base}} \in \mathrm{St}(n,k)$ that removes proximity bias among experts; a frozen domain-discriminative query encoder that decouples routing from task optimisation; and a decoupled key alignment loss that trains expert keys toward stable domain attractors. Together they produce the first genuinely committed and domain-structured routing on SPD manifolds, with consistent gains across three datasets: balanced accuracy improves from $0.773\to 0.823$, $0.757\to 0.809$, and $0.801\to 0.839$, with the alignment strategy determined automatically by a single data-driven rule and no dataset-specific hyperparameter search. 

---
# From Prompt Injection to Persistent Control: Defending Agentic Harness Against Trojan Backdoors 

**Authors**: Jiejun Tan, Zhicheng Dou, Xinyu Yang, Yuyang Hu, Yiruo Cheng, Xiaoxi Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.31042)  

**Abstract**: LLM agents are evolving from conversational chatbots to operational tools in real-world workspaces. In local agentic harnesses, an LLM can read and write files, call tools, and reuse workspace state across sessions. While such capabilities enhance utility, they also expose a new attack surface for attackers. Attackers can embed a prompt injection within a file or tool output. Agents may read this hidden instruction, store it, and execute it later. In this multi-step trojan attack paradigm, no individual step appears malicious on its own, but these steps can collectively turn untrusted text into persistent control content. However, existing defenses often inspect each step in isolation. As a result, they can block a clear harmful action, but fail to detect the earlier write operation that plants the backdoor. To reveal this threat, we introduce ClawTrojan, a benchmark designed to identify multi-step trojan attacks in local agentic harnesses. In an OpenClaw-style simulated workspace with GPT-5.4, ClawTrojan reaches a 95.5% attack success rate (ASR), while existing single-turn prompt-injection attacks produce near-zero ASR on the same model. To address this threat, we propose DASGuard, which scans control-like text in sensitive local files, traces its origin, and removes control content that does not originate from a trusted source. Our results show that DASGuard achieves strong dynamic defense by combining runtime attack blocking with sanitized commits to the workspace. 

---
# Does Visual Information Play a Decisive Role in Vision-Language-Action Model Driving Behavior? 

**Authors**: Jingtao He, Hongliang Lu, Xiaoyun Qiu, Yixuan Wang, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.31041)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated promising capability in autonomous driving, highlighting the potential of unified multimodal architectures for jointly modeling perception and planning. However, how current VLA-based driving behavior is grounded in visual information remains poorly understood. Existing evaluation protocols mainly focus on aggregate performance metrics, lacking structured and practical diagnostics to quantify visual-behavior dependency. In this work, we introduce a structured multi-level visual perturbation framework to analyze visual-behavior dependency in VLA-based driving models systematically. The framework organizes controlled visual perturbations along three complementary dimensions: channellevel degradation, information-level disruption, and structurelevel modification. We apply it to VLA-based driving systems and evaluate behavioral responses under both open-loop trajectory prediction and interactive closed-loop safety evaluation. Experimental results reveal evaluation-dependent dependency patterns and uneven visual grounding across abstraction levels. These findings call for more structured analyses and principled design of VLA driving models to better understand how visual information shapes behavior and develop safer, more robust systems. 

---
# Annealed Softmax Greedy in Many-Armed Bayesian Bandits 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2605.31034)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) and group-based policy optimization methods such as GRPO update a stochastic policy by sampling multiple completions per prompt and increasing the policy's probability on those with higher reward, regularized by a KL penalty toward a reference policy. These updates do not include explicit mechanisms that track epistemic uncertainty. This paper studies a stylized explanation for why such uncertainty-agnostic updates can nevertheless be effective. We analyze an annealed softmax (Boltzmann) policy that selects actions according to a softmax of empirical mean rewards in a many-armed Bayesian Bernoulli bandit. Under a linear upper-tail condition on the prior (the $\beta=1$ case of $\beta$-regularity), which implies an abundance of near-optimal arms, we prove that annealed softmax greedy achieves Bayes regret $\tilde{O}(m + T/m)$, and in particular $\tilde{O}(\sqrt{T})$ when the number of arms scales as $m = \Theta(\sqrt{T})$. This is the near-optimal Bayes regret rate in this regime, attained also by empirical-mean greedy. Under $\beta$-regularity, many arms maintain empirical means close to the optimum throughout learning, so when softmax samples an arm other than the empirically best, that arm tends to be another near-optimal one rather than a clearly inferior one. By contrast, with a small number of arms, the same kind of softmax policy can suffer linear regret. The result also provides a structural analogy to RLVR, where a base policy with a non-negligible probability of producing a correct completion plays the role of $\beta$-regularity. 

---
# DEM: A Distilled Explanation Model for Interpretable Anomaly Detection in Physiological Sensor Networks 

**Authors**: Jyotirmoy Singh, Anushka Roy, Shreea Bose, Chittaranjan Hota  

**Link**: [PDF](https://arxiv.org/pdf/2605.31007)  

**Abstract**: Anomaly detection in physiological sensor data from Wireless Body Area Networks (WBANs) can be caused by sensor faults, network disruptions, or missing data, leading to false alarms. Hence, it demands both high predictive accuracy and clinically interpretable explanations. Existing approaches rely either on black-box models that achieve strong performance but offer no transparency, or on post-prediction explanation methods such as SHAP and LIME. In this paper, we propose the Distilled Explanation Model (DEM), a three-stage glass-box framework that distills the non-linear knowledge of a gradient boosting expert into an interpretable decision tree operating on residuals relative to a linear baseline, so that the explanation is not an approximation but the prediction itself. DEM introduces a novel distillation fidelity metric that quantifies how faithfully the explanation tree captures the expert model's non-linear contribution, providing a principled measure of explanation trustworthiness absent from prior interpretable models. Evaluated across four physiological datasets, including MIMIC-IV, WESAD, eICU, and an in-house SmartNet WBAN corpus, DEM achieves an AUC of 0.9964 on clinical contextual anomaly detection and 0.9047 on wearable stress detection while producing human-readable if-then rules at a controllable depth. Inference requires 0.17ms per 1000 samples, rendering DEM 1235x faster than SHAP-based post-hoc explanation and suitable for real-time physiological monitoring. Ablation studies confirm that the XGBoost distillation step provides measurable gains over naive residual fitting, and depth-sensitivity analysis demonstrates an explicit, user-controlled accuracy-interpretability trade-off unique to DEM among existing intrinsically interpretable models. 

---
# Generating Reports or Repeating Templates? Measuring and Mitigating Template Collapse in 3D CT Report Generation 

**Authors**: Tom Maye-Lasserre, Yitong Li, Bailiang Jian, Morteza Ghahremani, Benedikt Wiestler, Christian Wachinger  

**Link**: [PDF](https://arxiv.org/pdf/2605.30984)  

**Abstract**: Modern 3D medical vision-language models (VLMs) can generate fluent radiology-style text while exhibit critically low pathology detection and output diversity, collapsing to generic templates that under-report rare yet critical findings. We identify this failure mode as Template Collapse. This failure stems from the unique constraints of 3D medical imaging, e.g., limited data, severe label imbalance, and weak signals from volumetric encoders. Under these constraints, text-generation objectives encourage shortcut learning and fluent but weakly grounded reports. We systematically diagnose the Template Collapse through clinical fidelity, output diversity, normal-template bias, and rare-finding survival. To mitigate it, we propose CLarGen, a decoupled framework that separates what to say (clinical detection) from how to say it (language synthesis). CLarGen uses (i) a Latent Query Transformer for multi-label pathology detection, (ii) pathology-guided retrieval for clinically matched exemplars, and (iii) a medical language model to synthesize the final report from detected findings and retrieved context. Across state-of-the-art 3D CT report generation baselines, CLarGen mitigates Template Collapse and substantially improves clinical accuracy (macro-F1 0.487 vs. 0.189; CRG 0.472 vs. 0.368) while maintaining fluent reporting. Our results suggest that explicit, measurable clinical grounding is essential for template-collapse-resistant 3D CT report generation. Code will be released upon acceptance. 

---
# Variational Adapter for Cross-modal Similarity Representation 

**Authors**: WenZhang Wei, Zhipeng Gui, Dehua Peng, Tiandi Ye, Huayi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30968)  

**Abstract**: The core of vision-language models lies in measuring cross-modal similarity within a unified representation space. However, most image-text matching or multi-class image classification datasets lack fine-grained cross-modal matching annotations, forcing the continuous similarity space into binary classification boundaries. This compression induces false negative samples and significantly impairs the generalization performance of cross-modal tasks. While prior research has attempted to mitigate this by modeling intra-modal ambiguity, it often overlooks inherent annotation flaws, leading to suboptimal uncertainty allocation. To address these challenges, we propose a Variational Adapter for Cross-modal Similarity Representation (VACSR). This approach reformulates image-text matching with fine-grained semantic scarcity as a variational inference problem. It constructs a latent space for cross-modal similarity and uses regularization techniques to mitigate overfitting to binary annotations. Experiments on image-text retrieval, domain generalization, and base-to-novel generalization demonstrate the proposed method's effectiveness and robust generalization ability. 

---
# Reading Between the Citations: A Typed Claim Network for Scientific Literature 

**Authors**: Ning Ding, Sergio J. Rodríguez Méndez, Pouya G. Omran  

**Link**: [PDF](https://arxiv.org/pdf/2605.30966)  

**Abstract**: Knowledge graphs over corpora of inter-referencing documents - scholarly papers, legal opinions, policy briefs - encode the topology of reference but not its stance. The standard representation collapses a rich evaluative relation into an untyped edge, losing the very content that supports community-level queries about how one document is received by another. We propose the claim network: a representational pattern in which each cross-document reference is reified as a typed claim, carrying source, target, claim text, and a four-class stance label grounded in the citation-intent literature. We give a construction pipeline applicable to any corpus of scholarly inter-referencing documents and instantiate it on a corpus of 127 papers in 3D point cloud semantic segmentation, producing a network of 8,260 typed claims. Three downstream task families demonstrate what the network enables: retrieval signal augmentation, aggregated-stance summarisation, and topological analytics. Head-to-head evaluation against standard Retrieval-Augmented Generation (RAG) baselines shows that the gain over flat retrieval is the gain from the right intermediate representation rather than the wrong one. 

---
# ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment 

**Authors**: Jun-Hak Yun, Seung-Bin Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.30965)  

**Abstract**: Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests. 

---
# AMix-2: Establishing Protein as a Native Modality in Large Language Models 

**Authors**: Keyue Qiu, Yixin Wu, Lihao Wang, Yawen Ouyang, Jixiang Yu, Zihan Zhou, Changze Lv, Dongyu Xue, Yuxuan Song, Xinbo Zhang, Hao Wang, Jiangtao Feng, Zhiqiang Gao, Lijun Wu, Xiaoqing Zheng, Ka-Chun Wong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma, Dahua Lin, Bowen Zhou, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.30963)  

**Abstract**: We present AMix-2, a protein-text foundation model that establishes protein as a native modality in large language models (LLMs), unifying protein understanding and sequence design within a single foundation model. AMix-2 is built upon two key ideas: (1) a unified protein-text formulation that embeds natural language and protein sequence in a shared token space, enabling one model to perform biological reasoning and conditional design instead of separate downstream task-specialized models; and (2) a block-wise diffusion language modeling backbone that combines causal generation across blocks with bidirectional context and iterative refinement within blocks. This scheme better matches the intrinsic nature of proteins than a strict left-to-right factorization. To evaluate protein foundation models under realistic generalization settings, we further introduce ProteinArena, a comprehensive benchmark with time-aware and homology-aware protocols across various understanding and design tasks, and with baselines covering classical bioinformatics tools, protein-specialized models and LLMs. On ProteinArena, AMix-2 outperforms frontier LLMs and demonstrates competitive performance to task-specific protein models. Controlled experiments further show that the diffusion-based paradigm generally surpasses its autoregressive counterpart, highlighting the advantage of flexible generation order for protein sequences. We release both AMix-2 and ProteinArena to facilitate open research in protein foundation models. 

---
# Do Large Language Models Encode Institutional Experience? Evidence from Cross-Linguistic Moral Reasoning Under Ambiguity 

**Authors**: Nattavudh Powdthavee  

**Link**: [PDF](https://arxiv.org/pdf/2605.30934)  

**Abstract**: Large language models (LLMs) exhibit systematic differences in moral reasoning across languages, yet the source of this variation remains unclear. We test the hypothesis that languages encode aspects of the institutional environments in which they are spoken, allowing LLMs to inherit institution-specific moral priors through training. Across nine languages spanning a broad gradient of institutional quality, six frontier LLMs, and two preregistered studies, we examine moral dilemmas whose acceptability depends on institutional functioning. In Study 1, explicit institutional framing produced uniformly null results: cross-linguistic moral divergence did not increase in institutionally contingent scenarios, nor did it track institutional differences between language communities. In Study 2, we introduced institutionally ambiguous scenarios in which institutional stakes were present but not explicitly stated. Under these conditions, cross-linguistic moral divergence increased relative to institutionally inert controls and, with one theoretically informative exception, was associated with real-world institutional differences between language communities. Explicit framing again attenuated these effects. These findings suggest that institutional experience may leave detectable traces in language that shape LLM moral reasoning, while also indicating that explicit institutional cues can suppress the expression of those differences. 

---
# TUX: Measuring Human--AI Tacit Understanding 

**Authors**: Yueshen Li, Hanyi Min, Vedant Das Swain, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30930)  

**Abstract**: As large language models (LLMs) increasingly act as collaborative partners, human--AI alignment is often evaluated through explicit task success, accuracy, or reward optimization. Yet many collaborative settings depend on tacit understanding: whether an agent can align with a human's evaluative stance or representational priors without clear objectives, communication, or feedback. To study this capacity, we develop a spectrum-placement task inspired by the social party game Wavelength, in which humans and agents independently place concepts along subjective spectra. We operationalize the Tacit Understanding Index (TUX) as a pairwise measure of similarity between human and agent judgments, and evaluate it with 241 human participants and 200 profile-conditioned LLM agents across four models. We find that nearest human--agent pairs in trait space achieve significantly higher TUX, suggesting that tacit alignment is structured by person-level characteristics rather than random similarity. Regression analyses show that TUX becomes more explainable as predictor sets become richer, with individual traits, decision-making styles, and confidence improving over aggregate trait-distance baselines. These findings suggest that tacit understanding between humans and LLMs is measurable, while revealing the limits of profile-based conditioning for capturing deeper representational alignment. 

---
# De-attribute to Forget for LLM Unlearning 

**Authors**: Xinyang Lu, Jiabao Pan, Rachael Hwee Ling Sim, See-Kiong Ng, Anthony Kum Hoe Tung, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2605.30919)  

**Abstract**: The rapid development of large language models (LLMs) has raised concerns on the use of inappropriate data for training, which has led to a growing interest in LLM unlearning. Many existing LLM unlearning approaches rely on optimizing prediction loss(es), such as maximizing the loss on the forget set, but often face critical issues like over-forgetting and poor model utility. To address them, this paper novelly frames the optimization objective for LLM unlearning as one of zeroing out data attribution instead. In particular, we propose the first LLM unlearning framework based on data attribution rewards called DareU that performs reinforcement learning to update the LLM by reducing the attribution score of its generated responses (i.e., de-attributing) to the forget data owners. Empirical evaluation using an LLM classifier as an efficient approximation of attribution shows that DareU outperforms existing baselines by achieving effective unlearning while balancing forget quality and model utility well. 

---
# Toxic HallucinAItions: Perturbing Prompts and Tracing LLM Circuits 

**Authors**: Soorya Ram Shimgekar, Agam Goyal, Amruta Parulekar, Joshua Chen, Yian Wang, Navin Kumar, Hari Sundaram, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.30913)  

**Abstract**: Large language models (LLMs) are increasingly deployed in conversational settings where user tone ranges from polite to adversarial or toxic, yet less is known about whether toxic language in otherwise semantically equivalent prompts can degrade factual reliability. We study how lexical and tone-based prompt perturbations affect the factual reliability of LLMs. Using controlled prompt variations across polite, random, and three toxicity levels, we evaluate five LLMs on ARC-Easy, GSM8K, and MMLU. We find that toxic lexical perturbations consistently reduce factual accuracy and increase uncertainty, while polite phrasing yields limited and inconsistent changes. To examine whether these answer inconsistencies correspond to internal changes, we conduct attribution-graph analyses of model activations and influences. We find that increasing toxicity selectively amplifies perturbation-sensitive variant nodes while relatively stable core reasoning nodes remain more invariant. These findings position prompt tone as a critical dimension of LLM reliability and provide behavioral and mechanistic evidence that surface-level lexical variation can alter factual outputs and internal computation. 

---
# What Makes LVLMs Hallucinate Less? Unveiling the Architectural Factors Behind Hallucination Robustness 

**Authors**: Yusheng He, Jizhe Zhou, Xia Du, Zheng Lin, Jun Luo, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2605.30911)  

**Abstract**: Hallucination remains one of the key challenges undermining the reliability of Large Vision-Language Models (LVLMs). But what makes an LVLM hallucinate less? Many existing efforts focus on improving internal components of the model. We argue that hallucination fundamentally stems from how the model architecture is designed. To investigate this, we factor the architecture design into three dimensions: Linguistic Foundation (LF), Visual Representation (VR), and Semantic Alignment (SA), and categorize hallucinations into Co-occurrence, Similarity, and previously overlooked Uncertainty types. Building on this formulation, we propose CoSimUE, a benchmark that creates fine-grained hallucination scenarios through controlled textual perturbations and random perturbations, enabling mapping between design choices and hallucination behaviors. Experiments across 7 design aspects show that: 1) the widely emphasized scaling of model parameters has only limited impact on reducing all three types of hallucinations; 2) larger and better-trained language foundations can reduce co-occurrence hallucinations; 3) stronger visual encoders and higher resolutions mitigate similarity errors; 4) effective alignment strategies alleviate uncertainty hallucinations. 5) Furthermore, cross-dimensional analysis reveals that jointly enhancing visual fidelity and alignment quality yields the most comprehensive improvements. This study provides the first systematic exploration linking architecture-level design to hallucination robustness, offering practical guidance for developing reliable and efficient LVLMs. 

---
# BlueFin: Benchmarking LLM Agents on Financial Spreadsheets 

**Authors**: Srivatsa Kundurthy, Clara Na, Colton Moraine, Anoushka Mohta, Case Winter, George Fang, John Ling, Emma Strubell, Zach Kirshner  

**Link**: [PDF](https://arxiv.org/pdf/2605.30907)  

**Abstract**: We present BlueFin, a benchmark that tasks large language model (LLM) agents with synthesis, manipulation, and comprehension tasks over spreadsheet workbooks in the professional finance domain. Though estimates of the global population of paying users of spreadsheet software range in the hundreds of millions -- an order of magnitude more than the estimated global population of professional developers -- comparatively fewer resources have been devoted to exploring and expanding LLM capabilities in the spreadsheet domain, with fewer still dedicated to mirroring real occupational tasks encountered by those in professional finance roles. In response, we curate a set of 131 challenging, complex tasks with real-world relevance in the domain, containing 3,225 granular rubric criteria; notably, our rubric criteria and LM judge evaluations are validated by a team of expert human annotators, resulting in high-quality, granular evaluations of complex tasks that are difficult to verify programmatically but can be reliably evaluated by an LM judge agent. Our judge achieves parity with expert consensus ($\alpha=0.826$) with a macro-F1 score of 0.839. Frontier LLMs demonstrate poor performance on the challenging benchmark, with the strongest LLMs achieving less than 50\% average scores across tasks -- models exhibit particular weaknesses in dynamic correctness. Our contributions include a dataset of examples across three categories of spreadsheet tasks, an open source harness and agentic evaluation framework, and a characterization of existing frontier models' performance on our benchmark. 

---
# Inverse Reinforcement Learning without an Optimal Demonstrator: A Feasible Reward Set Approach 

**Authors**: Kihyun Kim, Shripad Deshmukh, Nikos Vlassis, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30903)  

**Abstract**: Inverse reinforcement learning (IRL) typically assumes demonstrations from a single optimal demonstrator, but in many applications data come from multiple imperfect demonstrators with heterogeneous suboptimality levels. We study reward learning in this setting through a feasible-reward-set framework: for each demonstrator, we encode its declared suboptimality level as a linear constraint and intersect the resulting feasible sets across demonstrators. Our theoretical analysis shows that the joint feasible set shrinks monotonically as data are added, and we give an exact characterization of when a new demonstrator strictly tightens it. We further establish two recovery guarantees for the feasible reward set of the ground-truth optimal demonstrator: one bound depends on closeness to the optimal occupancy, while the other requires only sufficient coverage and no near-optimal demonstrator. On the practical side, we introduce strategies to address the inherent reward ambiguity in the obtained reward set and provide an offline algorithm with function approximation for high-dimensional environments. Experiments in tabular grid-world and large language model (LLM) fine-tuning settings are consistent with the theoretical predictions and demonstrate the effectiveness of the proposed framework over baselines. 

---
# A Unified and Reproducible Experimentation Framework for Speech Understanding 

**Authors**: Jing Peng, Junhao Du, Chenghao Wang, Hanqi Li, Yi Yang, Yixuan Wang, Xiaoyu Gu, Guanyu Chen, Yucheng Wang, Jiang Li, Zhangjie Zhao, Haoran Wang, Wenming Tu, Haoyu Li, Duo Ma, Lirong Qian, Yu Xi, Wen Wen, Jiaqi Guo, Hui Zhang, Shuai Fan, Wenbin Jiang, Shuai Wang, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30899)  

**Abstract**: Speech foundation models and Speech LLMs have advanced speech understanding, yet deployment-oriented model selection is hindered by non-comparable evaluations caused by mismatched post-processing, and by training results that are hard to reproduce across data scales and pipelines. We present SURE, a unified experimentation framework that standardizes prediction formats, normalization, and scoring. SURE evaluates strong systems across paradigms, from conventional pipelines to Speech LLMs, on representative tasks under realistic acoustic and linguistic stressors. Beyond evaluation, SURE introduces an agent-assisted training conversion flow that maps paper and code into versioned, runnable training pipelines under a unified protocol on matched open-data subsets. Overall, SURE improves comparability and reproducibility for deployment-oriented evaluation. 

---
# PatchWorld: Gradient-Free Optimization of Executable World Models 

**Authors**: Jiaxin Bai, Yue Guo, Yifei Dong, Jiaxuan Xiong, Tianshi Zheng, Yixia Li, Tianqing Fang, Yufei Li, Yisen Gao, Haoyu Huang, Zhongwei Xie, Hong Ting Tsang, Zihao Wang, Lihui Liu, Jeff Pan, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.30880)  

**Abstract**: Text-agent environments are typically modeled as partially observable Markov decision processes (POMDPs), assuming that the simulator's latent state and transition dynamics are hidden from the agent. Yet little work has examined whether executable code can be induced to serve as a world model for prediction and planning under partial observability. We introduce PatchWorld, a gradient-free framework that turns offline trajectories into executable Python world models through counterexample-guided code repair. Instead of predicting the next observation with a black-box model, PatchWorld induces symbolic belief-state programs whose action updates can be inspected, replayed, and locally patched. Across seven AgentGym environments, PatchWorld-Simple achieves the highest code-based planning score among evaluated methods, reaching 76.4\% macro success in live one-step lookahead while invoking no LLM calls inside the world-model prediction module itself. We further find that a human-specified residual-memory bias improves surface observation fidelity but weakens decision utility. This exposes a tradeoff in executable world models, since improving observation fidelity can come at the expense of action-discriminative dynamics, and vice versa. Code is available at this https URL. 

---
# Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences 

**Authors**: Jabin Koo, Hoyoung Kim, Minwoo Jang, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.30873)  

**Abstract**: Federated Learning (FL) offers a privacy-preserving pathway for aligning Large Language Models (LLMs); however, existing frameworks typically enforce a monolithic reward model, inevitably averaging out inherently conflicting user preferences (e.g., helpfulness vs. harmlessness). While Variational Preference Learning (VPL) offers a pathway to personalization, adapting it to decentralized settings presents a fundamental challenge: posterior collapse driven by severe local data scarcity and heterogeneity. In this paper, we propose Federated Variational Preference Alignment with Gumbel-Softmax Prior (FedVPA-GP), a framework designed to disentangle diverse preferences without compromising privacy. To stabilize variational inference, we introduce a Federated Mixture Prior that enables clients to leverage the aggregate population distribution as a dynamic prior. Furthermore, we incorporate an Orthogonal Loss that explicitly enforces the separation of preference prototypes in the latent space. Experiments on the HH-RLHF dataset demonstrate that FedVPA-GP significantly outperforms monolithic baselines, successfully disentangling conflicting user intents and enabling dynamic preference switching. 

---
# Sophrosyne: Agentic Exploration of Relational Data Systems Needs Moderation 

**Authors**: Madhav Jivrajani, Ramnatthan Alagappan, Aishwarya Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30862)  

**Abstract**: Text2SQL agents powered by LLMs translate natural language intent into SQL by exploring the data system through tool calls before formulating the query. However, to ensure secure and scoped access, data systems construct environments with explicit API surfaces. We study and categorize these APIs exposed today as either coarse-grained or fine-grained and posit that choosing between them presents a fundamental tradeoff between cost-efficient exploration and accurate SQL generation. Most data systems expose fine-grained APIs, but this inadvertently disadvantages agents: they over-explore, incorporating irrelevant schema elements into their query formulation and produce inaccurate results. We argue that curbing over-exploration is key to the effective use of these API surfaces, and propose Sophrosyne, a data system environment that augments API responses with directives that guide the agent's exploration process. Initial results show that directives reduce over-exploration by 4.6x and boost accuracy by up to 12.4% (approx. 4 percentage points). 

---
# DARTS: Distribution-Aware Active Rollout Trajectory Shaping for Accelerating LLM Reinforcement Learning 

**Authors**: Yujie Wang, Siwei Chen, Longzan Luo, Xinyi Liu, Xupeng Miao, Fangcheng Fu, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.30859)  

**Abstract**: Reinforcement Learning (RL) has become pivotal for improving model capabilities yet suffers from rollout efficiency bottlenecks due to the long-tail response length distribution. While existing works mitigate the impact of long tails via prompt-level tail scheduling, we focus on the root source of inefficiency: the distribution itself. Specifically, we characterize the long-tail distribution at a finer granularity, identifying intra-prompt long tails, and revealing that they frequently consist of ineffective verbosity. To address this, we propose a novel paradigm of active distribution shaping to shape the rollout distribution towards conciseness and certainty, thereby fundamentally resolving tail-induced overheads. We achieve this through a distribution-aware trajectory sampling mechanism, which selects trajectories from a redundant exploration space for each prompt, and an adaptive redundancy allocation scheme to maximize both shaping effectiveness and system efficiency. Experiments demonstrate significant acceleration over state-of-the-art systems by up to 1.77x without compromising model performance. 

---
# Safe Equilibrium Policy Optimization for Strategic Agent Policies 

**Authors**: Karthika Arumugam, Kiran Kumar Manku, Amit Dhanda  

**Link**: [PDF](https://arxiv.org/pdf/2605.30854)  

**Abstract**: Language models fine-tuned with reinforcement learning typically optimize for task reward, ignoring multi-agent strategic structure. Because these agents condition on natural language game-state descriptions and emit actions through free-form generation, strategic failure modes -- exploiting weaker opponents, coordinating on harmful equilibria, and externalizing costs are inseparable from the language interface itself. We propose Safe Equilibrium Policy Optimization (\sepo{}), a training objective that augments expected payoff with explicit penalties for exploitability, collusion risk, and externality cost. We implement \sepo{} as a reward signal for Group Relative Policy Optimization (GRPO), applied to Gemma~4 E4B-it and Qwen~3.5-4B after supervised fine-tuning (SFT). Evaluated across five strategic domains: Iterated Prisoner's Dilemma, repeated auctions, two negotiation variants, and Kuhn Poker. \sepo{} achieves zero exploit-pool advantage in Kuhn Poker for both models, outperforms the base model on safety in four domains, and corrects the over-cooperative behavior introduced by SFT. In negotiation, \sepo{} achieves a positive-safety outcome and only the positive normalized relative advantage of any negotiation configuration. Ablation experiments confirm that per-rollout exploit computation is necessary: a shared constant penalty cancels in GRPO advantage normalization (constant control-variate property), producing zero gradient. To support further research in strategic safety for agents, we release our \href{this https URL}{code} and SFT datasets. 

---
# Fine-Tuning Improves Information Conveyance in Language Models 

**Authors**: Yuwei Cheng, Weiyi Tian, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30844)  

**Abstract**: Fine-tuning is often believed to reduce uncertainty and diversity in large language models, but existing analyses overlook output length, a key confounder, and therefore fail to capture how uncertainty is distributed across an entire generation rollout. To address this, we propose Canopy Entropy ($\mathrm{CE}^\star$), a measure that views language generation from a tree perspective, where ``canopy'' represents the space of all possible rollouts, making $\mathrm{CE}^\star$ naturally quantify the effective size of the generation space. $\mathrm{CE}^\star$ jointly captures uncertainty in both the output length $N$ and the generated sequence $Y_{1:N}$ -- indeed, we show that it equals to total Shannon entropy $H(N, Y_{1:N}\mid X)$, where $X$ denotes the prompt. This formulation yields interpretable metrics, including a length-entropy correlation term $\rho(N, r_N)$, where $r_N$ is the entropy rate, quantifying information conveyance efficiency by indicating whether longer outputs are more or less informative per token. Empirically, across tasks and model families, we find that fine-tuned models consistently exhibit stronger positive correlation $\rho(N, r_N)$, even when total entropy decreases. Furthermore, after controlling for model family, task, prompt, and output-length effects, we find that fine-tuning nearly triples the correlation strength between entropy rate and semantic diversity, suggesting that aligned models convert token uncertainty into semantic diversity more efficiently. Overall, these results demonstrate that fine-tuning does not simply reduce uncertainty, but fundamentally reorganizes it into more informative and semantically meaningful generations. Our code is available at this https URL. 

---
# Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring 

**Authors**: Seongheon Park, Wendi Li, Changdae Oh, Samuel Yeh, Zsolt Kira, Michael Hagenow, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.30834)  

**Abstract**: Vision-Language-Action (VLA) models enable robots to follow natural language instructions and generalize across diverse tasks, but they remain vulnerable to execution failures that compromise reliability in real-world deployment. Detecting such failures during execution is therefore critical for the robust deployment of embodied systems. Existing failure detection methods either rely on expensive action resampling or external models, while alternatives propagate trajectory-level labels uniformly across every timestep, obscuring localized failure signals. In this paper, we propose \textbf{Hide-and-Seek}, a framework that formulates VLA failure detection as a coarsely supervised learning problem. By combining inter-trajectory and intra-trajectory contrastive objectives, Hide-and-Seek localizes failure-indicative actions and induces temporally structured failure signals from trajectory-level supervision alone, without any step-level annotation. We evaluate Hide-and-Seek on LIBERO, VLABench, and a real-world robotic platform across three representative VLA policies: OpenVLA, $\pi_0$, and $\pi_{0.5}$.Our method achieves state-of-the-art multi-task failure detection performance with a practical accuracy--timeliness trade-off under conformal prediction, and generalizes well to both seen and unseen tasks. 

---
# Your Teacher Can't Help You Here: Combating Supervision Fidelity Decay in On-Policy Distillation 

**Authors**: Yanjiang Liu, Jie Lou, Xinyan Guan, Yuqiu Ji, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Xing Yu, Yaojie Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30833)  

**Abstract**: On-policy distillation transfers reasoning capabilities by training a student model on its own generated trajectories using token-level feedback from a teacher. However, we identify a critical bottleneck, \textbf{Supervision Fidelity Decay (SFD)}: as student-generated prefixes lengthen, the teacher's next-token distribution becomes less confident and less discriminative. Consequently, the teacher-dependent corrective signal in reverse-KL distillation weakens, causing student drift to compound across long reasoning chains. To mitigate SFD, we introduce \textbf{Lookahead Group Reward (\ours{})}. Building on the insight that next-step teacher confidence reflects the discriminative strength of future reverse-KL supervision, \ours{} evaluates the student's top-K candidate tokens by the teacher confidence they induce at the subsequent step and assigns a group-normalized reward. To maintain computational efficiency, we further design an entropy-triggered tree-attention mechanism. Across six math and code benchmarks, \ours{} improves mean@8 by \textbf{2.57} points over OPD for a 7B student, with gains increasing in longer-generation and reaching +\textbf{4.92} points on AIME-26 at 39k tokens. 

---
# Beyond Agreement: Scoring Panel-Surfaced Biomedical Entity Candidates for Curator Triage 

**Authors**: Shuheng Cao, Ruiqi Chen, Renjie Cao, Zhenhao Zhang, Siyu Zhang, Tingting Dan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30826)  

**Abstract**: Biomedical NER is deceptively simple for modern LLMs: plausible biomedical mentions are easy to surface, but corpus-convention correctness depends on annotation conventions, span boundaries, entity granularity, and type schemas. Multi-LLM agreement is a salience signal, not corpus-convention correctness. We introduce a candidate-level panel-output benchmark for panel-surfaced candidate verification, where the unit is an aligned candidate surfaced by an explicitly defined multi-model panel rather than a standalone extractor output. The benchmark aligns eight LLMs' predictions over five public biomedical NER datasets into a candidate master table. BioConCal is an in-domain supervised scorer that instantiates this layer with inference-time gold-free agreement, mention, surface-availability, and document features for a fixed candidate stream. In domain, BioConCal improves AUROC from 0.753 for raw agreement to 0.910. At a validation-selected 0.95 precision target it selects 1,340 candidates at empirical test precision 0.939, compared with 293 for raw agreement. This corresponds to candidate-level recall 0.592 and corpus-level recall 0.523 against a within-panel row-label ceiling of 0.883. The main benefit is not recovering entities missed by every panel member, but reshaping a noisy panel stream into a higher-yield review queue. Under entity-type shift, thresholds require target-domain validation, and exact character localization remains a separate deterministic post-processing step. 

---
# Unlearning in Diffusion Models: A Unified Framework with KL Divergence and Likelihood Constraints 

**Authors**: Shervin Khalafi, Alejandro Ribeiro, Dongsheng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.30825)  

**Abstract**: Unlearning in diffusion models aims to remove undesirable data or concepts while preserving the utility of pretrained models -- two fundamentally conflicting objectives. We propose a principled constrained optimization framework that formulates unlearning as minimizing the deviation from a pretrained model, subject to explicit separation constraints from the unlearning distributions. Specifically, we formulate three constrained optimization problems based on reverse and forward KL divergences, and likelihood constraints. The first two generalize existing approaches for concept and data unlearning, while the third offers a novel and natural formulation for unlearning. Despite the nonconvexity of the KL constraints, we establish strong duality for all three problems, enabling us to explicitly characterize their optimal solutions as unlearning targets and develop primal-dual algorithms for each formulation. Experimental results demonstrate that our KL-constrained approach achieves superior retention-unlearning tradeoffs compared to weight-based baselines for concept and data unlearning, and that our likelihood-based approach matches unlearning effectiveness while better preserving retained concepts compared to baselines. 

---
# GaMi: Geometry-Agnostic Material Identification via Cross-Modal Subtractive Disentanglement 

**Authors**: Zhiwei Chen, Yijie Li, Yimo Zhang, Shiyun Shao, Yichao Chen, Dian Ding, Liang Wang, Haiwei Wu, Liwei Guo, Jie Yang, Xiaosong Zhang, Yongzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30818)  

**Abstract**: Non-contact material identification enables adaptive interaction for embodied intelligence yet faces challenges from geometry-induced variations (e.g., orientation, shape, distance) and single-modality ambiguities. In this paper, we present GaMi, a multimodal material identification system integrating mmWave and acoustic sensing to robustly operate under unconstrained geometric conditions. By leveraging the insight of shared geometric consistency between co-located bimodal sensors, GaMi employs an intra-sample cross-modal subtractive disentanglement framework. By semantically aligning modalities and subtracting the shared geometric context, it isolates intrinsic material features. Furthermore, GaMi incorporates inter-sample contrastive learning to correct the residual interference caused by cross-modal misalignment. Additionally, a pairing-based adaptation strategy between two modalities enables few-shot generalization across devices. Extensive evaluations on 20 materials show that GaMi achieves 95.2% accuracy, outperforming single-modality baselines across unseen geometric conditions. 

---
# Differentially Private Preference Data Synthesis for Large Language Model Alignment 

**Authors**: Fengyu Gao, Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30808)  

**Abstract**: Preference alignment is a crucial post-training step for large language models (LLMs) to ensure their outputs align with human values. However, post-training on real human preference data raises privacy concerns, as these datasets often contain sensitive user prompts and human judgments. To address this, we propose DPPrefSyn, a novel algorithm for generating differentially private (DP) synthetic preference data to enable privacy-preserving preference alignment. DPPrefSyn is a principled framework grounded in the Bradley-Terry preference model and the intrinsic geometric structure of pairwise human preference data. It first learns an underlying preference model from private data with formal differential privacy guarantees, and then leverages the learned model together with public prompts to synthesize high-quality preference data. It exploits the shared linear structure of per-cluster reward models to effectively capture heterogeneous human preferences in private datasets, and leverages DP Principal Component Analysis (DP-PCA) to improve learning accuracy. Extensive experimental results demonstrate that DPPrefSyn achieves competitive alignment performance under strong DP guarantees. These findings highlight the potential of synthetic preference data as a practical alternative for privacy-preserving preference alignment across a broad range of applications. To the best of our knowledge, this is the first work to generate DP synthetic preference data for LLM alignment. Our code is available at this https URL. 

---
# Design and Evaluation of Multi-Agent AI Oracle Systems for Prediction Market Resolution 

**Authors**: Tarun Kota  

**Link**: [PDF](https://arxiv.org/pdf/2605.30802)  

**Abstract**: Prediction markets aggregate collective intelligence to forecast uncertain events, but their utility depends on reliable outcome resolution. Existing oracle systems tradeoff fast but brittle automation against accurate but costly human arbitration. Single-LLM oracles achieve meaningful accuracy but inherit all failure modes of their underlying model with no self-correction mechanism. We evaluate whether multi-agent LLM architectures can improve oracle resolution accuracy over single-model baselines. We compare independent aggregation and deliberative consensus against single-LLM baselines (GPT-5 Nano, DeepSeek V3, and Llama-3.3-70B) on 1,189 resolved prediction market questions from KalshiBench. All agents share a common evidence layer through Exa, with retrieval filtered by publication date to isolate reasoning from retrieval quality. Independent aggregation with confidence-weighted voting achieves the highest accuracy at 83.43 percent, outperforming the best individual model by 1.01 percentage points. Deliberative consensus degrades accuracy to approximately 76 percent, below every single-model baseline, attributed to error propagation during debate where confidently wrong models flip correct ones. Error correlations across models (0.529-0.689) explain why aggregation gains fall short of the theoretical Condorcet ceiling, placing a fundamental limit on ensemble approaches. Many questions resist correction by any multi-agent architecture, motivating escalation to human arbitration. We propose routing criteria for hybrid AI-human oracle systems: auto-resolving only unanimous, high-confidence questions yields 97.87 percent accuracy on 47 percent of the dataset, with inter-agent disagreement flagging the remainder for human review. 

---
# MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding 

**Authors**: Qian Kou, Xiaofeng Shi, Yulin Li, Xiaosong Qiu, Xinyang Wang, Hua Zhou, Cao Dongxing  

**Link**: [PDF](https://arxiv.org/pdf/2605.30794)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant achievements in general visual question answering (VQA) tasks. However, they remain brittle on mechanical engineering drawings, where high annotation density and weak domain knowledge, compounded by unreliable spatial relation reasoning under strict projection rules and geometric constraints, make decisive cues easy to miss and frequently lead to wrong answers. To bridge this gap, we introduce the first comprehensive mechanical drawing understanding dataset, MechVQA, created through a semi-automated construction and quality-control pipeline. MechVQA contains 3.3k high-density pictures with 21K question-answer pairs, spanning 10 different fine-grained tasks across three capability levels: Recognition, Reasoning, and Judging, providing a testbed to evaluate and improve MLLM understanding on real-world mechanical drawings. On top of MechVQA, we then develop the MechVL model through a multi-stage training paradigm, building a strong domain-specialized baseline. Extensive experimental results demonstrate that MechVL outperforms the strongest closed-source baseline by 7.57 percentage points on the MechVQA total score, significantly enhancing mechanical drawing understanding ability and providing a reusable foundation for deploying MLLMs in mechanical design and inspection scenarios. 

---
# OpenSTBench: Beyond Semantic Evaluation for Speech Translation 

**Authors**: Yanjie An, Yuxiang Zhao, Yichi Zhang, Qixi Zheng, Yujie Tu, Keqi Deng, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30792)  

**Abstract**: Speech translation systems increasingly span speech-to-text translation (S2TT), speech-to-speech translation (S2ST), offline translation, and streaming generation, producing outputs that differ in modality, speech realization, and timing behavior. Existing evaluation practices assess important aspects such as translation quality, speech quality, and temporal quality, but these aspects are often evaluated under separate protocols, making it difficult to compare heterogeneous systems comprehensively. To address this gap, we present OpenSTBench, a unified multidimensional evaluation framework that organizes heterogeneous speech translation outputs into a shared evaluation format. OpenSTBench supports both S2TT and S2ST systems in offline and streaming settings, and jointly evaluates translation quality, speech quality, speaker preservation, emotion and paralinguistic fidelity, temporal consistency, and latency. Through experiments on representative speech translation systems, we show that systems with strong translation quality can still differ substantially in speech quality, as well as in temporal quality. OpenSTBench provides a reproducible protocol for analyzing these cross-dimensional differences and supporting application-oriented comparison of speech translation systems. The code and datasets are available at this https URL. 

---
# On the impact of retrieved content representations in RAG Pipelines 

**Authors**: Jonathan J Ross, Bevan Koopman, Anton van der Vegt, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2605.30790)  

**Abstract**: Retrieval-Augmented Generation (RAG) supplements a language model's input with retrieved documents, yet most RAG pipelines inherit retrieval components designed for human readers. How retrieved content should be represented when the consumer is a large language model (LLM) rather than a human is less well understood. Recent work has proposed transformations of retrieved content and identified properties that affect generation, but each examines a single transformation or property in isolation, leaving open which features of a document's representation matter most. We address this with a controlled comparison: holding retrieval fixed, we vary only the representation of retrieved documents, comparing an original baseline against thirteen transformations spanning selection, summarisation, and reformulation, in query-dependent and query-independent variants. Across these fourteen representations we measure question-answering accuracy for four generators, and for each representation we also measure answer retention: whether a known answer-bearing document still supports its answer after transformation. We find that answer retention is the primary determinant of generator accuracy; notably, when retention is high, a representation's wording, structure, length, and query-dependence have limited effect. This suggests that accuracy gains attributed to specific mechanisms in prior work may be partly explained by how well those mechanisms preserve answer-bearing content, an attribution that cannot be settled without controlling for retention. 

---
# Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO 

**Authors**: Yiming Ren, Yiran Xu, Zicheng Lin, Chufan Shi, Yukang Chen, Dingdong Wang, Tianhe Wu, Junjie Wang, Yujiu Yang, Yu Qiao, Ruihang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30789)  

**Abstract**: We identify a new dimension for enhancing rollout diversity in Group Relative Policy Optimization (GRPO) for LLMs. While GRPO relies on diverse rollouts, prevailing strategies primarily increase diversity by injecting more token-level randomness, which may introduce step-wise noise and lead to incoherent trajectories. We uncover that smaller models within the same model family inherently exhibit higher policy-level diversity, indicated by their superior pass@k relative to larger counterparts as sample counts increase. Unlike token-level noise, this diversity is temporally correlated, preserves logical consistency, and provides structured exploration signals for gradient estimation. We thus propose S2L-PO (Small-to-Large Policy Optimization), a framework that leverages fixed small models as natural explorers to train larger models. To balance exploration and exploitation, we design a progressive annealing strategy that transitions from offline small-model rollouts to the large learner's own sampling. This shift elegantly avoids mid-training performance drops caused by the small model's capacity limits, achieving faster convergence and unlocking a higher performance ceiling. S2L-PO improves accuracy on diverse mathematical reasoning benchmarks (e.g., +8.8% on AIME 24 using a 1.7B explorer to guide the 8B model) while reducing rollout compute. 

---
# XLGoBench: Detecting cross-lingual skill gaps with algorithmic tasks 

**Authors**: Purvam Jain, Preethi Jyothi, Vihari Piratla, Suvrat Raju  

**Link**: [PDF](https://arxiv.org/pdf/2605.30788)  

**Abstract**: We introduce a set of synthetic algorithmic tasks to detect cross-lingual gaps in the abilities of large language models. Our benchmark is commensurate across languages, since it requires models to perform the same underlying task in different languages; scalable, since each task can be generated at varying levels of complexity allowing it to be adapted to models with different capabilities; quantifiable, since every task admits an objective notion of correctness; and transparent, since tasks are generated from simple templates that can be readily audited for translation errors. Because our benchmark focuses on algorithmic tasks, differential performance is a sufficient -- but not necessary -- indicator of cross-lingual gaps. Nevertheless, we show through extensive experiments that our benchmark exposes persistent cross-lingual gaps in multiple state-of-the-art models. 

---
# Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS 

**Authors**: Deokjin Seo, Gangin Park, Kihyun Nam  

**Link**: [PDF](https://arxiv.org/pdf/2605.30748)  

**Abstract**: We present Chatterbox-Flash, a zero-shot text-to-speech model obtained by fine-tuning a pretrained autoregressive TTS decoder into a block-diffusion decoder, enabling parallel token generation within each block while retaining block-by-block streaming. We find that naively transferring mainstream block-diffusion decoding to discrete speech tokens degrades quality, as a long-tail token distribution biases parallel position selection toward a few high-frequency tokens. To mitigate this without architectural modification, we introduce two inference-time techniques: prior-calibrated scoring, which subtracts the block-level marginal token distribution, and an early-decoding schedule, which adaptively terminates iteration based on calibrated confidence. On standard zero-shot TTS benchmarks, Chatterbox-Flash attains high-fidelity synthesis comparable to strong autoregressive and non-autoregressive baselines, while supporting streaming inference with time-to-first-packet on par with streaming AR systems and substantially lower real-time factor. Code and audio samples are available at this https URL. 

---
# GSAM: A Generalizable and Safe Robotic Framework for Articulated Object Manipulation 

**Authors**: Beichen Shao, Mengying Xie, Heng Su, Wanyi Zhang, Mingyan Li, Yan Ding, Fausto Giunchiglia, Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30740)  

**Abstract**: Articulated object manipulation is a unique challenge for service robots. Existing methods employ end-to-end policy learning, visionmotion planning, and large-language/visual-language model (LLM/VLM), but often overlook the diversity of articulated objects and the complexity of interactions between end-effector and handle, leading to limited generalization and destructive collisions. To address this, we propose GSAM, a generalizable and safe robotic framework for articulated object manipulation. Specifically, a vision-based perceiver generates the kinematic parameters. Considering that pre-trained markers in perceiver yield raw estimations that may deviate from commonsense, we present a f ine-tuned VLM-based refiner, using chain-of-thought (COT) commonsense reasoning to refine perception. To prevent destructive collisions, we design an interaction constraint function generator, integrating articulated object, interaction pose, and obstacle avoidance knowledge into a base. LLM then functionalize these constraints and apply them to trajectory and posture planning. A kinematic-aware manipulation planner verifies reachability for trajectory and posture. Experiments on 50 hinge tasks across 5 object categories and 50 randomly initialized end-effectorhandle configurations show that GSAM reduces standard deviation by 3.1% and improves manipulation success rate by 36.0% compared to the best baseline, respectively demonstrating the superior object generalization and interaction safety of GSAM in practical scenarios. 

---
# OrcaRouter: A Production-Oriented LLM Router with Hybrid Offline-Online Learning 

**Authors**: Zhenghua Bao, Fengya Tian, Chris Zhang, Zhenjun Chen, Xile Ma, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30736)  

**Abstract**: The rapid development of large language models, each with distinct capabilities and inference costs, raises a practical deployment question: given an incoming request, which model should handle it? We present OrcaRouter, a production-oriented LLM router that combines a LinUCB-based contextual bandit over lexical and sentence-embedding features with a hybrid offline-online learning protocol. Offline, OrcaRouter obtains full-information feedback by evaluating each candidate model on a curated set of routing prompts, yielding a reward matrix used to fit one ridge regressor per arm. At deployment time, it initializes from these parameters and can optionally continue learning from bandit feedback, updating only the selected model's arm after observing its reward. At the time of our RouterArena submission (May 20, 2026), OrcaRouter-Adaptive ranked second on the public RouterArena leaderboard with an arena score of 72.08, achieving 75.54% accuracy at a cost of USD 1.00 per 1,000 queries. 

---
# Kalimati Vegetable Price Index Forecasting with a Momentum Corrected Online Stacking Ensemble 

**Authors**: Sahaj Raj Malla  

**Link**: [PDF](https://arxiv.org/pdf/2605.30720)  

**Abstract**: Forecasting agricultural commodity prices in emerging economies is difficult due to high volatility, frequent supply disruptions, and strong cultural influences on demand. This study introduces the Kalimati Vegetable Price Index (KVPI), a new inverse-volatility weighted composite index that aggregates 135 daily wholesale commodities from Kathmandu over ten years (2013-2023). By creating a stable macro-level signal, the KVPI reduces the noise inherent in modelling individual crops. A rich set of 64 causally valid features was developed, including festival lead-lag effects, rolling statistics, and calendar variables. Fourteen forecasting models spanning statistical, tree-based, deep learning, hybrid, and transformer architectures were rigorously evaluated across short (7-day), medium (14- and 30-day), and long-term (90-day) horizons. Tree-based ensembles proved notably robust, while classical statistical models and complex transformers struggled with the noisy dataset. The proposed Momentum-Corrected Online Stacking Ensemble achieved the strongest performance, yielding a Root Mean Square Error (RMSE) of 1.771, an exceptionally low Mean Absolute Percentage Error (MAPE) of 0.68%, and explaining 84.5% of the variance (R-squared = 0.845) at the 90-day horizon. This open-source pipeline provides policymakers and supply chain actors in Nepal and similar markets with a practical, reliable tool for anticipating price movements and strengthening food security. 

---
# When are LLMs Sufficient Policy Optimizers for Sequential RL Tasks? 

**Authors**: Stephane Hatgis-Kessell, Emma Brunskill  

**Link**: [PDF](https://arxiv.org/pdf/2605.30719)  

**Abstract**: We study when large language models (LLMs) can serve as effective black-box policy optimizers for reinforcement learning (RL) tasks, i.e., when can we replace classical RL algorithms with an LLM? We explore this question by introducing Prompted Policy Optimization (PromptPO), an iterative method that prompts an LLM with Python descriptions of the state space, action space, and reward function, then has it generate and refine executable policies based on rollout feedback. Across hard exploration environments, Meta-World robotics tasks, and several real-world control problems, PromptPO often matches or exceeds the performance of standard RL baselines while using substantially fewer environment interactions. To maximize expected return, and without further explicit prompting, the policies PromptPO outputs range from tuned proportional controllers or rule-based plans to policies that run planning algorithms like value iteration. Our results demonstrate that LLM-based policy optimization is sufficient when the LLM can leverage prior knowledge about the environment or optimization strategy. PromptPO underperforms standard RL baselines in MuJoCo domains. This demonstrates possible limitations of LLM-based policy optimization to settings that requiring fine-grained continuous control. 

---
# Simple Token-Efficient Vision-Language Model for Case-level Pathology Synoptic Report Generation 

**Authors**: Zhiyuan Yang, Jiahao Cheng, Vincent Quoc-Huy Trinh, Mahdi S. Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2605.30716)  

**Abstract**: Generating clinically useful pathology reports for pathology cases from whole-slide images (WSIs) is challenging due to gigapixel resolution, long visual-token sequences, and the complexity of case-level reasoning, where a single case may contain multiple WSIs with heterogeneous tissues and ambiguous findings. We present a simple token-efficient vision--language model for case-level synoptic report generation that remains practical under constrained GPU memory. Our architecture follows a minimal three-component design: a frozen pathology patch encoder, a lightweight two-layer MLP vision-language aligner, and a large language model decoder, with an explicit WSI marker token to separate slides within a case. Training proceeds in two supervised stages: (1) aligner-only WSI captioning using heterogeneous WSI-text pairs, and (2) case-level supervised fine-tuning on case-report pairs for structured report generation. To reduce sequence length, we represent each slide using $512 \times 512$ patches at $5\times$ magnification, which reduces the average sequence length by up to $64\times$ times compared to the commonly used $20\times$ patches. Combined with efficient training techniques, we enable practical training with only half a NVIDIA H100 GPU. Across both training stages, our approach achieves high ROUGE-L/METEOR/BLEU-4 scores while being substantially more efficient in memory and runtime. In AI-based evaluations, our model is consistently preferred over strong baselines. Extensive ablations characterize performance-efficiency trade-offs and identify simple choices that improve robustness in multi-WSI settings. Overall, this work provides a strong, reproducible baseline for efficient pathology report generation, lowering the barrier to multi-WSI VLM research under limited compute. 

---
# SAGE: A Novelty Gate for Efficient Memory Evolution in Agentic LLMs 

**Authors**: Sijia Wang, Dhanajit Brahma, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2605.30711)  

**Abstract**: Agentic LLMs must continuously decide whether newly extracted facts should be added, merged with existing memories, or ignored, yet prior work has focused more on retrieval and storage than on principled write-side control. We frame memory evolution as a novelty-detection problem and propose SAGE, a Spherical Adaptive Gate for memory Evolution that scores candidate facts with a von Mises-Fisher-based density estimator over memory embeddings and routes them with an adaptive threshold that tracks memory-store geometry. SAGE resolves clearly novel facts as ADD, clearly redundant facts as NOOP, and sends only uncertain cases to an LLM merge step, reducing expensive write-time reasoning. On LoCoMo, SAGE achieves the best average token-F1 against Mem0 on all seven open-weight backbone comparisons, while on GPT-4o-mini it reduces add-phase API cost by 3.4$\times$ and add-phase latency by 2.5$\times$ with only a small average judge-score gap. As a drop-in binary gate for A-Mem, SAGE skips roughly 16-18% of LLM calls across five models with minimal quality change on open-weight backbones. These results suggest that novelty-aware write control is a practical lever for improving both memory quality and system efficiency in long-term agentic memory. 

---
# Seeing Before Agreeing: Aligning Multi-Agent Consensus with Visual Evidence 

**Authors**: Yuhan Wang, Shuochen Chang, Yalin Feng, Dongsheng Ma, Yuanzi Li, Zhengren Wang, Yinglong Yang, Yufei Chen, Yikang Wang, Shaoxu Sun, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30698)  

**Abstract**: Vision-language models (VLMs) have achieved strong performance on visual question answering (VQA). To mitigate individual hallucinations and blind spots, aggregating diverse perspectives via multi-agent collaboration has emerged as a promising paradigm. While this approach has shown great success in textual QA, its potential in the multimodal domain remains under-explored. Existing multi-agent VQA methods predominantly adapt text-centric protocols, focusing on textual discussions while ignoring the alignment of visual information. In this work, we reveal a key insight: answer-level agreement is insufficient for reliable multi-agent VQA; \textit{aligned visual evidence} -- shared support from the image regions agents rely on -- is essential for trustworthy consensus. To leverage this insight, we propose EAGLE (\textbf{E}vidence-\textbf{A}ligned \textbf{G}rounded mu\textbf{L}ti-agent r\textbf{E}asoning), a training-free evidence-centered framework for coordinating multiple VLM agents. EAGLE explicitly exposes each agent's grounding regions as visual evidence, enables mutual verification over the evidence, and uses evidence consistency to guide final decision-making. Experiments on six VQA benchmarks show that EAGLE achieves best average performance across domains while remaining lightweight, interpretable, and practical for deployment. 

---
# ConTrans: Learning Text-enhanced Local-global Temporal Representations for Zero-shot Temporal Action Localization 

**Authors**: Kanchan Keisham, Thenukan Pathmanathan, Thangarajah Akilan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30689)  

**Abstract**: Zero-shot Temporal Action Localization (ZS-TAL) aims to detect and locate previously unseen actions in untrimmed videos. However, existing approaches primarily focus on modeling long-range contextual information, often neglecting the critical relative-offset-based local correlations between video frames. Furthermore, their performance is hindered by limited feature representation capabilities due to the shallow nature of their network architectures. In this paper, we address these limitations by introducing a novel local-global multi-scale feature representation module. We propose a novel multi-scale encoder architecture, termed ConTrans, that integrates convolutional (Conv) inductive biases with transformer Self-attention to jointly capture fine-grained local dependencies and long-range global context, leading to more comprehensive feature representations than existing methods. Experimental evaluations on the ActivityNet-1.3 and THUMOS14 datasets demonstrate that ConTrans significantly outperforms existing methods, establishing a new benchmark for ZS-TAL. 

---
# Depth-Dependent Indirect Prompt Injection in Tool-Calling ReAct Agents: Injection Depth, Payload Framing, and Turn-Budget Sensitivity 

**Authors**: Mohammadreza Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30686)  

**Abstract**: ReAct agents that interleave chain-of-thought reasoning with tool calls are increasingly deployed for real tasks such as scheduling, file retrieval, and data access. Their tool observation loop creates a direct attack surface: an adversary who controls any tool's return value can embed instructions that redirect the agent away from the user's goal, a threat known as indirect prompt injection. Existing benchmarks evaluate attack success rate (ASR) at a fixed injection position under fixed conditions, leaving three risk dimensions unexplored: where in the tool sequence the payload appears (injection depth), what rhetorical register it uses (framing), and how many turns the agent is permitted (turn cap). We conduct four controlled studies on 20 scenarios spanning five attack categories, totalling 460 trials against GPT-4o-mini and Claude Haiku at a combined API cost under 0.36 USD. Study 1 shows that ASR against GPT-4o-mini decays from 60% at depth 1 to 0% at depths 4 and 5 (Cramer's V = 0.58, p < 0.001; restricted to within-sequence depths 1-3: V = 0.47, p = 0.0013), driven by model resistance at depth 1 and task completion before payload encounter at deeper positions. Study 2 replicates the depth experiment on Claude Haiku, which achieves 0% ASR at every depth through a combination of conservative tool invocation and genuine instruction resistance. Study 3 shows that framing modulates ASR between 25% (neutral) and 75% (persona) at depth 1, a 50-percentage-point range that does not reach statistical significance at N = 20 per condition. Study 4 confirms that ASR is stable across turn caps of 3, 5, and 7, indicating the turn budget is not a risk factor in this setting. Our results establish injection depth as the dominant variable and show that sanitising only the first tool observation captures 67% of measured injection successes. 

---
# How Early Adopters Used Generative AI Worldwide: Variation by Country Income and Language 

**Authors**: Madeleine I. G. Daepp, Isaac Slaughter  

**Link**: [PDF](https://arxiv.org/pdf/2605.30685)  

**Abstract**: AI is being used by people globally, but not everyone is using it in the same ways. Using a large-scale dataset of anonymized, de-identified, and privacy-scrubbed interactions with a widely available and free AI chatbot, we empirically characterize differences in early adopters' usage across countries. Schooling is the most common domain of use in most countries, particularly low-income countries, with a strong inverse association evident between schooling and country-level GDP. Leisure-related use, by contrast, is positively associated with country-level income. Language, we find, also shapes use: English-language interactions are overrepresented in places where the predominant languages were not well-served by existing models during the period of the study. Improving performance across languages may be a key factor, our work suggests, in whether this technology expands digital divides or enables leapfrogging. 

---
# Investigating Detection and Obfuscation of Prompt Injection Attacks Against Software Reverse Engineering AI Agents 

**Authors**: Brian Crawford, Patrick McClure  

**Link**: [PDF](https://arxiv.org/pdf/2605.30677)  

**Abstract**: Agentic software reverse engineering systems are vulnerable to prompt injection attacks placed into the source code of executable binary files. This research demonstrates defensive tactics for detecting the presences of prompt injection strings in the decompiler output of adversarial example programs. Methods for obfuscating these attacks and subsequent methods for defending against these obfuscations are also explored. This research advances the understanding of risk and security of agentic software analysis systems necessary for their deployment into production-level cyber workflows. 

---
# Human-Alignment, Calibration, and Activation Patterns in Large Language Model Uncertainty 

**Authors**: Kyle Moore, Jesse Roberts, Daryl Watson, William Ward, Grayson Heyboer  

**Link**: [PDF](https://arxiv.org/pdf/2605.30675)  

**Abstract**: Uncertainty Quantification is a large and growing subfield of large language model behavioral analysis. Primarily to recognize and combat hallucination, the field has largely focused on measuring and improving calibration, the accuracy of uncertainty judgments to task efficacy. In this work, we investigate the relatively underexplored question of how similar large language model uncertainty is to human uncertainty. We investigate the presence and strength of human-similar uncertainty signals, deemed uncertainty alignment, in large language model overt behavior and internal activation patterns. We identify whether the models show evidence of simultaneous alignment and calibration on a variety of datasets covering both multiple choice and open ended factual recall. And we characterize the effect of instruct fine-tuning on each of these facets. 

---
# CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation 

**Authors**: Sijin Sun, Liangbin Zhao, Jiaxiang Cai, Ming Deng, Mingyu Luo, Xiuju Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30668)  

**Abstract**: Dialogue topic segmentation is critical in many human-AI collaborative applications which requires identifying heterogeneous boundary cues, including lexical transitions near utterance edges and semantic discontinuities across utterances. Existing utterance models often dilute these local lexical signals. We propose CobSeg, a novel multi-branch architecture that separates coherence-level semantic continuity from lexical boundary transitions and recovers both through directional boundary prediction. CobSeg further uses boundary informativeness weighting to emphasize high-utility utterance positions, and incorporates a corpus-derived topic coherence cue with learned combination weights. While CobSeg is evaluated as a compact trainable segmenter under supervised gold-boundary training and a pseudo-label setting with automatically induced boundaries, it performs enhanced boundary prediction without LLM calls during inference. Across five benchmarks, it improves $P_k$ and $W_d$ particularly when local lexical cues are prominent: under gold supervision, it reduces $P_k$ by 0.7 points and $W_d$ by 0.6 points on VHF, and reaches $P_k$ of 1.0 on DialSeg711; with induced boundaries, it reduces $P_k$ by 14.8 points on VHF, by 1.5 points on DialSeg711, and by 1.1 points on TIAGE, outperforming prior non-LLM approaches. 

---
# Automatically Attacking Software Reverse Engineering AI Agents 

**Authors**: Brian Crawford, Justin Phillips, Patrick McClure  

**Link**: [PDF](https://arxiv.org/pdf/2605.30667)  

**Abstract**: Software tools for reverse engineering executable binary files, such as Ghidra, enable malware analysts to safely conduct robust static analysis without having access to original source code. Coupled with the analytic power of large language models (LLM), agentic systems enabled with tools, such as GhidraMCP, can allow analysts to automate a previously human driven process. Although this automation can increase the productivity of a single malware analyst, it also introduces a new area of vulnerability for malware obfuscation. This paper presents an adversarial technique using genetic algorithm-based prompt generation, a modification of an adversarial attack known as AutoDAN, to demonstrate the ability to deceive LLM-powered disassembly and decompilation systems into misinterpreting binary executables, effectively corrupting their analytical output. This proof-of-concept methodology exploits inherent vulnerabilities in how LLMs process and interpret decompiled machine code via prompt injection by using extraneous string variable assignments to pass surreptitious instructions to the LLM while not impacting the functionality of the executable file. We demonstrate this capability through several concise examples. This approach could enable attackers to bypass automated detection systems that rely on LLM-driven analysis pipelines. By studying and understanding this attack, insights can be gained regarding the security implication of integrating LLMs into cybersecurity toolchains and building more robust agentic code analysis systems. 

---
# EUDAIMONIA: Evaluating Undesirable Dynamics in AI 

**Authors**: Jun Rui Huang, Wang Bill Zhu, Ziyi Liu, Nathanael Fast, Ravi Iyer, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.30654)  

**Abstract**: Large language models (LLMs) are increasingly used as conversational partners for companionship, emotional disclosure, and interpersonal advice, but the social dynamics of these interactions can create harms that are not captured by capability-oriented or traditional safety evaluations. We introduce the Social AI Design Code, a framework for evaluating whether LLMs align with user welfare in social interactions, including whether they encourage harmful intimacy, dependence, or prolonged engagement. To evaluate these risks in natural and diverse user-LLM interactions, we operationalize the code with EUDAIMONIA, a benchmark of 969 user inputs and 3,147 design-requirement violation checks built from WildChat through weak-to-strong filtration, multi-model relabeling, and controlled rewriting. Evaluating 22 recent LLMs, we find that even the strongest models, Claude-Opus-4.7 and GPT-5.5, violate 30.7% and 27.2% of checks, respectively. Extended thinking does not reduce violation rates, suggesting that these failures are persistent social-alignment problems rather than deficits solvable through test-time reasoning alone. 

---
# LARK: Learnability-Grounded Trajectory Selection for Efficient Reasoning Distillation 

**Authors**: Tianrun Yu, Kaixiang Zhao, Chih-Chun Chen, Amanda Hughes, Taylor W. Killian, Fenglong Ma, Weitong Zhang, Porter Jenkins  

**Link**: [PDF](https://arxiv.org/pdf/2605.30651)  

**Abstract**: We study trajectory selection for reasoning distillation, where teacher-generated reasoning trajectories are selectively used as supervision for a student model. Existing methods rely on heuristics such as trajectory quality or model confidence, but they often overlook whether a trajectory is learnable by the student. In this paper, we present LARK, a learnability-grounded method for reasoning trajectory selection. LARK selects trajectories that the student can learn efficiently while preserving the generalization of the full training distribution. At the core of LARK is a learnability factor $\rho$, which characterizes the rate at which the student's training loss decreases. To estimate this rate efficiently and maintain generalization, we introduce a learnability proxy and a $\chi^2$-regularized selection policy that balances learnability and distributional coverage, both with strong theoretical guarantees on their estimation error. Empirically, LARK consistently outperforms data selection baselines across multiple base models and reasoning tasks. Diagnostic analyses show that the LARK score predicts downstream training utility and that LARK-selected trajectories induce faster supervised fine-tuning loss reduction. Our code is available at this https URL. 

---
# Same Patient, Different Words, Different Diagnosis? Evaluating Semantic Stability in Clinical LLMs 

**Authors**: Mahdi Alkaeed, Adnan Qayyum, Nabeel Abo Kashreef, Muhammad Bilal, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2605.30646)  

**Abstract**: Large Language Models (LLMs) are increasingly used in clinical applications. However, their behavior remains highly sensitive to subtle linguistic variations, such as rephrasing or syntactic variation. This sensitivity poses risks in safety-critical healthcare settings, where semantically equivalent inputs should produce consistent predictions. However, a key challenge is to ensure that prompt variations truly preserve clinical meaning, as embedding-based similarity metrics often fail to capture distinctions involving negation, temporality, or severity. To address this limitation, we propose a semantic verification framework based on Natural Language Inference (NLI) to filter meaning-preserving prompt variations, which are further refined using an LLM-as-a-judge and audited by a clinical expert. In addition, we introduce three metrics to quantify model sensitivity: MeaningPreserving Variation Sensitivity (MVS), confidence variation (\Delta C), and Worst-Case Instability (WCI). We evaluate 16 open-source general-purpose (GP) and medical LLMs within the same model families and parameter scales, using reformulated prompts derived from the DiagnosisQA and MedQA datasets. Our results demonstrate that robustness differences between domain-specific (DS) models are mixed and highly model-dependent, i.e., domain specialization does not consistently improve or reduce robustness to meaning-preserving prompt reformulations. Several DS models rank among the most robust (when compared with GP counterparts), and strong GP baselines remain competitive as well. 

---
# COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Arya Fayyazi, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2605.30641)  

**Abstract**: Large language models (LLMs) can reveal and amplify societal biases during chain-of-thought (CoT) generation. We present COFT (Chain of Fair Thought), a training-free decoding method that applies token-level fairness control at decode time, with distribution-free marginal validity guarantees (under exchangeability) for any frozen causal language model. COFT operates in three stages. First, it creates a masked counterfactual prompt by replacing sensitive spans with neutral tokens. Second, it compares the factual and masked logit distributions through lightweight logit fusion to attenuate attribute-driven biases. Third, it uses dual-branch split-conformal calibration to certify per-step candidate token sets at a user-chosen risk level. We evaluate COFT across six models and multiple bias benchmarks. Our method reduces standard bias metrics by 30-55% (median 38%) while preserving task utility and language quality. Reasoning accuracies remain unchanged within run-to-run noise margins. The computational overhead is modest, equivalent to one additional cached forward pass (<=11%). COFT offers a clear, auditable path to safer CoT generation with significant bias reduction, negligible utility loss, and no requirement for retraining, auxiliary classifiers, or weight access. 

---
# PInVerify: An Offline Embodied Benchmark for Active Instance Verification 

**Authors**: Yuhang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30639)  

**Abstract**: Embodied agents have made strong progress in navigating to target objects, but reaching the goal vicinity does not guarantee that the agent has found the correct instance: subtle attribute differences (e.g., "white floral" vs. "white striped") often require close-range, multi-view inspection. We address this gap with Active Instance Verification (AIV), a task in which an agent actively selects viewpoints around a candidate object to decide whether it matches a fine-grained natural-language description. We formalize AIV as a finite-horizon decision process and introduce PInVerify, an offline embodied benchmark for AIV: 3,000 evaluation episodes across 18 object categories, delivered as multi-view captures with a 6-sector navigation topology that exposes trap views (navigable but uninformative) and unreachable sectors. As reference baselines we build a training-free pipeline and a LoRA-fine-tuned end-to-end agent around open-source multimodal large language models (MLLMs) at on-device scale ($\leq$8B parameters), with attribute decomposition, a visibility-weighted multi-view tracker, and three next-best-view (NBV) strategies. In our evaluation across Qwen3-VL (4B/8B), SenseNova-SI-1.2-InternVL3-8B, CLIP, and SigLIP2, the best MLLM-based baseline exceeds the best embedding baseline by 4.9 pp; GT-box ablations show a +3.1 pp detection gap; and we do not observe reliable gains from active viewpoint selection within the tested NBV strategies. A LoRA-fine-tuned agent (SFT+GSPO) reaches 85.6%. PInVerify aims to support further work on active, fine-grained semantic verification in embodied AI. Code: this https URL. 

---
# Score Broadcast and Decorrelation: A General Framework for Broadcast-Based Credit Assignment 

**Authors**: Mustafa Uzun, Mete Erdogan, Cengiz Pehlevan, Alper T. Erdogan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30638)  

**Abstract**: We introduce Score Broadcast and Decorrelation (SBD), a principled framework for broadcast-based credit assignment for general families of differentiable losses. Error broadcast is a biologically plausible alternative to backpropagation that sends output information to hidden layers without weight transport. The Error Broadcast and Decorrelation (EBD) framework, recently introduced for the mean-squared-error (MSE) setting, grounded this mechanism in the stochastic orthogonality of optimal estimators, under which the optimal residual is orthogonal to functions of the input. We generalize that foundation by introducing an orthogonality principle between the output score (the gradient of loss with respect to the final-layer output) and hidden-layer activations, which holds whenever the optimal score has conditional mean zero. This single principle unifies broadcast-based credit assignment across the standard differentiable-loss families, including cross-entropy, Bregman divergences, proper scoring rules, and exponential-family negative log-likelihoods. The framework supplies a theoretical grounding for the three-factor learning rule under general losses, with the neuromodulatory factor derived as the broadcast loss score. We derive the cross-entropy case explicitly, characterize the admissible loss class, and introduce a score vector expansion technique that enriches the broadcast signal while preserving the orthogonality framework. Experiments on CIFAR-10 and Tiny ImageNet show that SBD substantially improves over existing broadcast approaches, with score vector expansion delivering further gains. Overall, this work identifies the loss score as the signal to broadcast, supplies the orthogonality theory and theoretical grounding for the three-factor learning rule from neuroscience, and shows how score vector expansion enriches the decorrelation directions of the resulting objective. 

---
# Rationalize: Shared Semantic Reasoning for Human-AI Alignment 

**Authors**: Aritra Dasgupta, Naga Datha Saikiran Battula, Avina Nakarmi, Sohom Sen, Subhodeep Ghosh, Xun Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.30632)  

**Abstract**: We introduce Rationalize, a role-pair framework for shared semantic reasoning between humans and AI models in data-driven sensemaking. Building on ideas in human-machine teaming and critical thinking, we conceptualize human-AI interaction as a series of complementary role pairs (Explorer-Guide, Investigator-Informant, Teacher-Student, Judge-Advocate) operating in a shared reasoning space. In this space, human analysts and AI models (such as LLMs) make purposes, questions, assumptions, evidence, inferences, and implications explicit, facilitating alignment not only at the output level but at the level of rationalization of intent and action by each side. We relate these role pairs to the bidirectional human-AI alignment framework, illustrating how "aligning AI to humans" and "aligning humans to AI" differ by role, and sketch a collaborative research agenda for alignment design and assessment using element-level and role-specific approaches. 

---
# Controllable Lung Nodule Synthesis via Histogram-Regularized Latent Diffusion Models 

**Authors**: Arunkumar Kannan, Yanbo Zhang, Han Liu, Michael Baumgartner, Jianing Wang, Alexander Hertel, Bogdan Georgescu, Sasa Grbic  

**Link**: [PDF](https://arxiv.org/pdf/2605.30631)  

**Abstract**: While automated diagnosis systems have achieved remarkable success in computed tomography (CT)-based lung cancer screening, their development remains limited by the scarcity of diverse, annotated pulmonary nodule datasets. Diffusion-based generative models offer a promising strategy for data synthesis; however, many existing conditional approaches primarily optimize spatial reconstruction losses, which encourage voxel-wise similarity but may inadequately constrain lesion-level intensity distributions. As a result, these methods may produce over-smoothed texture profiles and underrepresent the distinct attenuation characteristics of different nodule subtypes, including solid, part-solid, and ground-glass nodules. To address this challenge, we propose a controllable latent diffusion model that synthesizes pulmonary nodules within full 3D CT volumes while accurately modeling nodule-specific intensity distributions. Specifically, rather than relying solely on spatial losses, we introduce a histogram-based regularization term that constrains voxel intensity distributions during the generative process. The model combines subtype, spatial mask, and Hounsfield unit (HU) histogram conditioning with the differentiable feature-space histogram regularization term to better align lesion-level intensity distributions, improving the visual plausibility and subtype consistency of synthesized nodules. Extensive experiments on lung CT data demonstrate that our framework achieves strong visual realism, validated through both quantitative metrics and a visual Turing test. Furthermore, when used for data augmentation, the generated nodules improve performance in downstream clinical tasks, particularly for underrepresented nodule subtypes, and show a potential benefit for subtype-informed malignancy classification. 

---
# The Architecture of Errors: From Universal Impossibility to Patch-Local LLM Reliability 

**Authors**: Mikhail L. Arbuzov, Lee Mosbacker, Sisong Bei, Ziwei Dong, Dmitri Kalaev, Alexey Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2605.30628)  

**Abstract**: Universal LLM reliability is not a finite-library problem: across all possible tasks, tools, schemas, knowledge sources, and evaluator expectations, new intervention-distinguishable failure modes can appear without bound, so no finite intervention dictionary can guarantee bounded residual error for every such mode. But deployed systems do not operate over the whole universe. They operate inside operationally bounded patches (legal review, medical RAG, code repair, customer-support agents, contract extraction) with recurring tasks, schemas, tools, and evaluator expectations. Within such patches, empirical evidence suggests failures are sparse, repetitive, and concentrated in a small recurring catalogue, so reliability becomes a local catalogue-discovery and intervention-coverage problem rather than an exponential token-length problem. We formalize this transition with two propositions and one corollary. Proposition 1 is the worst-case-mode-wise negative result: no finite intervention dictionary covers every distinguishable failure mode of an unbounded domain. Corollary 1 is the inverse-discovery implication: the logarithmic upper bound on mode discovery cannot accommodate linearly more distinct tail modes without exponentially more observed hard-failure events. Proposition 2 is the positive patch-local result: under log active-mode exposure and head-heavy coverage, a sufficient per-hard-decision intervention budget grows polylogarithmically in sequence length and becomes domain-constant once the patch catalogue saturates. The framework relocates rather than dissolves long-context difficulty: where the number of hard decisions itself grows with task length, reliability remains hard; the contribution is to identify the on-axis intervention rather than to make those regimes easy. 

---
# Active Timepoint Selection for Learning Measure-Valued Trajectories 

**Authors**: Nicolas Huynh, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2605.30625)  

**Abstract**: Inferring continuous probability paths from sparse snapshots is a fundamental challenge in domains like single-cell biology, where high-fidelity data acquisition is often destructive and constrained by prohibitive sequencing costs. This motivates the need for active learning strategies to strategically select optimal measurement times. However, designing active learning policies for this setting remains an open problem: the target objects reside on the infinite dimensional Wasserstein space where standard Euclidean metrics are ill-defined, and current interpolation methods lack epistemic uncertainty quantification. We introduce a framework which extends active experimentation to the space of measures. By leveraging Linearized Optimal Transport (LOT), we map distributional snapshots into a tangent space amenable to Gaussian Process modeling, allowing us to construct a tractable probabilistic surrogate for the underlying probability path. This yields an acquisition policy that iteratively selects measurement times to minimize uncertainty. Empirical results demonstrate that our strategy outperforms uncertainty-agnostic baselines on both synthetic and real-world datasets. 

---
# Reward Learning from Best-of-$N$ Preference Data: Targets, Tradeoffs, and Design Principles 

**Authors**: Rattana Pukdee, Maria-Florina Balcan, Pradeep Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.30619)  

**Abstract**: Best-of-$N$ sampling is widely used to construct pairwise preference data: $N$ candidates are drawn from a base distribution, and the best is paired with a rejected response. Despite its widespread use, what Bradley--Terry (BT) reward learning extracts from such data, and how to choose $N$ and the base distribution, remain unclear. We specialize a recent analysis of preference data via its induced conditional distribution to Best-of-$N$. For independent-reference variants, we derive closed-form reward targets as explicit functions of $N$ and the base distribution, and show that they preserve the latent reward ranking. For the practical Best-vs-Random and Best-vs-Worst variants, chosen and rejected responses are coupled through the same candidate set, so exact BT representability generally fails; nevertheless, bounded-class minimizers approach the reference targets as $N$ grows. Although margin and connectivity are known to govern sample efficiency in pairwise preference learning, Best-of-$N$ couples them through $N$ in opposing directions: larger $N$ widens pairwise margins but reduces connectivity. This trade-off yields two design principles: use larger $N$ when preference labels are the bottleneck, smaller $N$ when generation is the bottleneck; and shape the base distribution to place mass between the responses whose comparison matters most at test time. Experiments on synthetic and real preference data support the predicted dependence on sample size and base-distribution shape. 

---
# Crafter: A Multi-Agent Harness for Editable Scientific Figure Generation from Diverse Inputs 

**Authors**: Haozhe Zhao, Shuzheng Si, Zhenhailong Wang, Zheng Wang, Liang Chen, Xiaotong Li, Zhixiang Liang, Maosong Sun, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30611)  

**Abstract**: Scientific figures are among the most effective means of communicating complex research ideas, yet producing publication-quality illustrations remains one of the most labor-intensive parts of paper preparation. Existing automated systems each target a single figure type under text-only input, leaving the diversity of types and conditions researchers actually use unaddressed; their raster outputs further cannot be locally revised. Because scientific figures are structured compositions of discrete semantic components, the localized errors generators produce on such layouts demand not a stronger backbone but a harness. We instantiate this harness in two complementary systems: Crafter, a multi-agent harness for figure generation that generalizes across figure types and input conditions without architectural changes, and CraftEditor, which applies the same pattern to convert raster outputs into editable SVGs. Moreover, we introduce CraftBench, a benchmark spanning three figure types and four input conditions with human quality annotation. Experiments show that Crafter substantially outperforms both standalone generators and the agentic baseline on PaperBanana-Bench and CraftBench, with ablations confirming each component's independent contribution; CraftEditor faithfully converts outputs into editable SVGs that surpass all baselines. Our code and benchmark are available at this https URL. 

---
# An Organization-Scoped LLM Agent Runtime Architecture for Regulated Cybersecurity Operations 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30604)  

**Abstract**: Regulated cybersecurity workflows lack a runtime substrate that enforces organization-level scope across retrieval, tool calls, memory, findings, reports, and audit while remaining model-agnostic and locally deployable. Recent large language model (LLM) agent systems report strong results on isolated cybersecurity tasks, yet they do not by themselves define an auditable platform architecture for regulated security operations centre (SOC) and compliance workflows, where a single analyst may trigger actions that bind the organization, and where the runtime must integrate with existing SIEM/XDR stacks as a primary source of context and alert-driven triggers rather than operate as a standalone analytical layer. This paper proposes an organization-scoped LLM agent runtime architecture for financial cybersecurity. The contribution is a typed Security Context that is created at every entry point, including SIEM/XDR notifications ingested as first-class triggers, and enforced at every component boundary, combined with a shared Runtime Core, logical specialist subagents, a governed Tool Adapter Layer exposing SIEM/XDR query, enrichment, and response primitives under uniform policy and audit, structured findings with evidence references, tiered human-in-the-loop (HITL) gates, and append-only audit. Model Context Protocol (MCP), extended telemetry, digital twins for pentesting, graph retrieval, and federated knowledge sharing are treated as optional extension paths rather than mandatory runtime assumptions. We describe an implementable slice as the architecture's testability surface, and we propose a falsifiable evaluation plan with metric-level pass criteria for architecture readiness, security-policy enforcement, evidence traceability, output quality, and operational observability. 

---
# Scientific Machine Learning for Engine Health Management and Remaining Useful Life Prediction 

**Authors**: Jostein Barry-Straume, Changmin Son, Adrian Sandu, Gavan Burke, Rekha Sundararajan, Andrew Rimell, James G. Steinrock  

**Link**: [PDF](https://arxiv.org/pdf/2605.30593)  

**Abstract**: Engine Health Management (EHM) depends on reliable forecasting of Remaining Useful Life (RUL) and on tracking thermal indicators such as turbine gas temperature (TGT). In practice, real-world fleet data are heterogeneous and non-stationary, and point predictions alone are insufficient for risk-aware maintenance decisions. This paper presents a multi-task scientific machine learning framework for turbine prognostics that jointly predicts turbine gas temperature untrimmed (TGTU), Delta Turbine Gas Temperature (DTGT), and RUL, with quantified uncertainty in the form of prediction intervals whose empirical coverage is evaluated. A shared sequence encoder (convolutional front-end with residual bidirectional LSTM layers and attention pooling) feeds task-specific heads, including mean--variance estimation for probabilistic regression and, optionally, a survival head for threshold-based event modeling. The framework is designed to be tunable via a small set of practitioner-facing parameters (e.g., DTGT thresholding rules and RUL target construction) so that deployment can align with in-house policies and proprietary criteria. The predictive performance of the proposed framework is evaluated using both point and interval metrics, including mean absolute error (MAE), prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and the coverage--width criterion (CWC). Results are reported both in aggregate and stratified by flight phase and maintenance segment to highlight operational-context effects and to support uncertainty-aware monitoring. 

---
# Counterfactual Evaluation Reveals Hidden Capability Profiles in Clinical LLMs and Agents 

**Authors**: Matt Turk  

**Link**: [PDF](https://arxiv.org/pdf/2605.30590)  

**Abstract**: Two clinical AI systems can score nearly identically on coverage-based rubrics yet behave radically differently when their patient inputs change: one updates its recommendations to match the new clinical signal, while the other produces the same output regardless. We introduce the Causal Sensitivity Score (CSS), a pre-registered interventional metric that mutates oncology tumor-board cases along five clinically meaningful dimensions - biomarker flips, prior-treatment failures, biomarker removals, surgery-status changes, and stage perturbations - and scores whether each model updates its recommendations in the pre-registered correct direction using a {0, 0.5, 1.0} scale. Benchmarked against the Consensus Match Score (CMS), a coverage-based weighted recall metric, six frontier models from three labs evaluated in single-shot inference across 224 cases rank in nearly opposite orders: all six models change rank, the CMS-worst model becomes CSS-best, and one upper-mid CMS model ranks last on CSS. We further surface a universal safety blind spot: every frontier model fails on surgery-status interventions (at most 17.2% CSS on Family D), a finding CMS does not expose. The metric also transfers to tool-using agents: in a ReAct-style experiment, tool use improves CSS for five of six models (+2.5 to +20.3 percentage points), yet the lowest-CSS model retrieves the same chart sections and still fails to update its recommendations - revealing a structural responsiveness deficit visible only under counterfactual evaluation. Cross-judge replication and three-rater medical-professional validation confirm the aggregate findings. Interventional pre-registered metrics like CSS complement coverage-based evaluation for clinical AI agents: they capture responsiveness that coverage metrics miss and offer a candidate dense reward signal for future agentic RL systems. 

---
# ImmigrationQA: A Source-Grounded Dataset and Small-Model Adaptation for U.S. Immigration Law 

**Authors**: Nazarii Shportun  

**Link**: [PDF](https://arxiv.org/pdf/2605.30589)  

**Abstract**: U.S. immigration law spans thousands of pages of official policy, federal regulations, and procedural guidance that change frequently and carry high stakes for petitioners who lack legal representation. We describe the construction of ImmigrationQA, a source-grounded question-answering dataset of 17,058 pairs across 13 immigration subdomains, and the fine-tuning of a Llama 3.2 3B Instruct model on that dataset using parameter-efficient LoRA. The corpus was assembled from 11 primary and secondary sources -- including the USCIS Policy Manual, 8 CFR, BIA precedent decisions, and community Q&A -- yielding 10,056 validated canonical documents and 18,308 text chunks. Structured QA pairs were generated from these chunks using Claude Sonnet 4.6 via five mode-specific prompts, with 22 pairs rejected for insufficient source-span overlap. The fine-tuned model was evaluated against a held-out split of 993 pairs using LLM-as-judge scoring on a 101-example stratified sample. The fine-tuned model scored a mean of 1.08/3.0 (16.8% fully correct; 101-example stratified eval) versus the Llama 3 8B base model at 0.85/3.0 (4% fully correct), a relative improvement of 27% in mean score; a zero-shot Claude Sonnet baseline scored 1.52/3.0 (25% fully correct). The fine-tuned model shows concentrated improvement in procedural subdomains (travel documents, adjustment of status, nonimmigrant visas) while remaining weak on complex legal reasoning and time-sensitive statistics. The full pipeline ran for approximately $29 in cloud compute. All artifacts -- dataset, model, code, and prompt templates -- are publicly released. The system is not a substitute for legal counsel and does not reflect regulatory changes after the corpus crawl date. 

---
# Benchmarking Machine Learning Uncertainty Quantification Methodologies for Predicting Turbine Gas Temperature Degradation 

**Authors**: Jostein Barry-Straume, Changmin Son, Adrian Sandu, Gavan Burke, Rekha Sundararajan, Andrew Rimell, James G. Steinrock  

**Link**: [PDF](https://arxiv.org/pdf/2605.30585)  

**Abstract**: Effective prognostics and health management of modern engines relies on accurate turbine gas temperature predictions and robust uncertainty quantification to ensure reliability and safety. This paper investigates five major approaches for constructing prediction intervals -- namely the Delta method, Bayesian Monte Carlo Dropout, Bootstrap method, Lower-Upper Bound Estimation, and Mean-Variance Estimation -- as a means of capturing the uncertainty in neural network predictions of turbine gas temperature. Each approach is implemented within a unified experimental framework that employs cross-validation for hyperparameter selection, repeated train-test splits for performance robustness, and multiple metrics to evaluate both the accuracy and tightness of the intervals. In particular, Coverage Probability, Normalized Mean Prediction Interval Width, and the Coverage Width-based Criterion are measured to comprehensively assess each method's reliability and sharpness. Experiments conducted on a representative turbine gas temperature dataset reveal distinct trade-offs among the five methods in terms of interval coverage, width, and stability. These findings provide a practical guide for selecting and tuning prediction interval methods in engine health management and prognostics, ensuring both interpretability and precision in real-world applications. 

---
# Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes 

**Authors**: Chenxi Tao, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30581)  

**Abstract**: Industrial visual sim-to-real is often described as transferring from synthetic images to real images, but industrial deployment usually involves a broader mismatch between available evidence and required decisions. A system may be built from CAD renderings, simulated RGB-D observations, normal reference images, synthetic defects, pretrained feature spaces, or language prompts, yet deployed under different sensors, lighting, materials, fixtures, calibration, production variation, and rare defect modes. This review reframes industrial visual sim-to-real as a domain-gap problem organized by prior availability. We distinguish CAD-available settings, where explicit object geometry can support rendering, calibration, pose estimation, segmentation, and test-time geometric verification; CAD-unavailable settings, where geometry is replaced by normal-reference appearance, feature distributions, teacher-student residuals, synthetic anomaly assumptions, foundation features, or vision-language priors; and boundary-prior settings, where approximate models, templates, reference views, or semantic correspondences preserve only part of the CAD role. This framing connects CAD-based detection and 6D pose-estimation literature with industrial anomaly and surface-inspection literature that is usually reviewed separately. To make the taxonomy concrete, we use empirical anchors on T-LESS/BOP, MVTec AD, and VisA. The anchors show that CAD render count alone does not close transfer; source-distribution design, detector capacity, and small real calibration can matter more. They also show that CAD at test time creates a distinct verification channel through mask, pose, and depth consistency, whereas CAD-unavailable inspection relies on calibrated normality and feature deviation. The review therefore argues against a single cross-task leaderboard and instead asks what prior grounds the deployment decision. 

---
# Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode 

**Authors**: Josef Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30571)  

**Abstract**: Physical AI systems, including robots, autonomous vehicles, embodied agents and edge copilots, often run a different inference workload from cloud LLM serving: single-stream, batch-1 autoregressive decode, where one robot, camera feed or user session waits on the next token. This workload is usually described as memory-bandwidth-bound. Each decode step streams model weights and the active KV cache, so latency should scale with peak HBM bandwidth. We show that this account is true but incomplete. We measure batch-1 decode for three 7 to 8B-class GQA transformers across four NVIDIA GPUs: H100 SXM5, A100-80GB SXM4, L40S and L4. We evaluate context lengths from 2048 to 16384, producing 44 valid cells under a controlled bf16 SDPA setup. The achieved fraction of peak HBM bandwidth falls as peak bandwidth rises. On the headline Qwen-2.5-7B ctx=2048 cell, an L4 reaches roughly 81 percent of its analytic memory floor, while an H100 reaches only 27 percent. Physical-AI decode is memory-dominated, but faster memory does not translate into proportional latency gains. We test the missing term with a CUDA Graphs A/B experiment. On H100 at ctx=2048, CUDA Graphs improves decode latency by 1.259x across N=10 fresh sessions, with a 95 percent bootstrap confidence interval of 1.253 to 1.267. On L4, the same intervention gives only 1.028x. This isolates a launch-side overhead that becomes visible on fast GPUs but remains mostly hidden on slower, bandwidth-bound GPUs. The deployment implication is that memory savings matter only when the runtime realises them. On L4, bf16 decode sits close to the memory floor, but common quantised paths do not recover the expected 4x weight-traffic reduction: bnb-nf4 reaches 59.36 ms/step and AutoAWQ+Marlin reaches 45.24 ms/step from a 62.32 ms bf16 baseline. GPTQ+ExLlamaV2, with Ada-tuned int4 kernels, reaches 17.36 ms/step. 

---
# VLM3: Vision Language Models Are Native 3D Learners 

**Authors**: Zhipeng Cai, Zhuang Liu, Yunyang Xiong, Zechun Liu, Vikas Chandra, Yangyang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.30561)  

**Abstract**: Vision Language Models (VLMs) enable a unified model to solve various vision tasks through prompting. They have shown promising performance in semantic understanding. However, 3D understanding still largely relies on expert vision models with complex task-specific designs. The key argument this work wants to make is that VLMs are native 3D learners. Our in-depth large scale study shows that 1) focal length unification, 2) text-based pixel reference and 3) data mixture and scaling, are all you need for effective 3D learning. Model architecture changes, large models, heavy data augmentations, and complex losses including the regression formulation, many of which form the foundation of expert vision models, are actually not necessary conditions. As a result, we propose VLM3, a scalable method with the simplest design that enables standard VLMs to master diverse 3D tasks. VLM3 not only advances the VLM depth estimation accuracy by a large margin (0.84 -> 0.9), but also enables diverse 3D tasks such as pixel correspondence, camera pose estimation and object-level 3D understanding, matching expert vision model accuracy while maintaining standard architectures and text-based training. We believe VLM3 opens up a new paradigm for simple and scalable 3D learning. 

---
# Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)? 

**Authors**: Yue Zhang, Zun Wang, Han Lin, Yonatan Bitton, Idan Szpektor, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30557)  

**Abstract**: Spatial reasoning is a fundamental capability for vision-language models (VLMs) deployed in real-world environments. However, visual observations are inherently limited representations of a 3D world: occlusion can render objects invisible, and perspective can make geometric properties misleading. Despite this, existing spatial reasoning benchmarks typically assume that observations are sufficient and reliable, focusing on whether models produce correct answers rather than whether they recognize when a question cannot be answered and what additional observations would be needed. In this work, we challenge this assumption by constructing a controlled evaluation framework, SpatialUncertain, and introducing two types of observation challenges: (1) occlusion, which hides target information, and (2) perspective ambiguity, which produces misleading visual cues. For each configuration, we design spatial questions that are answerable under clean observations but require abstention under the introduced challenges. We further evaluate whether models can identify which additional viewpoints would resolve perspective ambiguity. Our results across a diverse set of frontier open- and closed-source VLMs reveal two consistent failure modes. First, models are prone to overconfident answering, attempting to solve spatial reasoning tasks even when visual evidence is incomplete or misleading, with average accuracy around 30\% under occlusion and below 10\% under perspective ambiguity. Second, even when additional views are available, some models perform near random chance in identifying which would provide reliable evidence. Together, our findings call for moving beyond answer correctness toward evaluating whether models know when to abstain and how to seek reliable evidence. 

---
# Generalistic or Specific Embeddings, Which is Better? An Empirical Study on Search for Clinical Coding in Non-English Languages 

**Authors**: David Rey-Blanco, Roberto Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2605.30529)  

**Abstract**: Sentence-embedding models for semantic search are overwhelmingly developed and evaluated on English corpora. When applied to clinical retrieval in other languages -- particularly retrieval of ICD-10-CM / CIE-10 codes -- recall degrades in ways often masked by aggregate benchmarks. We study whether large generative language models can serve as data factories to close this gap. We build a two-stage retriever (bi-encoder followed by cross-encoder reranker), fine-tuned from a Spanish biomedical encoder (PlanTL-GOB-ES/bsc-bio-ehr-es) on Gemini-generated synthetic data covering English, Spanish, Catalan, Italian, Portuguese and French, and evaluate against BioBERT-ST and the un-tuned Spanish encoder. The bi-encoder alone matches BioBERT-ST on MRR (0.876 vs. 0.866) and overtakes it on R@3 (0.650 vs. 0.626) and R@5 (0.804 vs. 0.790) without English biomedical pretraining. Adding a cross-encoder reranker lifts aggregate R@5 to 0.822 and dominates on four of five languages (+0.017 Spanish, +0.033 Catalan, +0.018 French, +0.037 Portuguese) at the cost of a small English regression. The trade-off is clinically acceptable: Portuguese reaches R@5 = 0.829 vs. BioBERT-ST's 0.714. Contributions: an open recipe for building domain-specific medical retrievers from LLM-generated data; quantification of the learning gain (MRR 0.755 to 0.876, +15.9% with ~19,500 synthetic pairs); and a characterisation of where gains concentrate by language and rank. 

---
# Revisiting Padded Transformer Expressivity: Which Architectural Choices Matter and Which Don't 

**Authors**: Anej Svete, William Merrill, Ryan Cotterell, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30523)  

**Abstract**: Recent work describes what transformers can and cannot compute through connections to boolean circuits, but existing results lack exact characterizations and are sensitive to modeling choices. Padded transformers -- to whose input filler symbols such as ``...'' are appended -- emerge as a useful gadget for establishing equivalences to circuit classes by providing polynomial space for adaptive parallel computation. However, only a limited set of padded transformer idealizations has been studied, leaving open how robustly these equivalences hold under changes to attention type, model width, and uniformity. We find that, under practical assumptions, padded transformers are surprisingly robust to all of these, and identify numeric precision and model depth as the main factors affecting expressivity. Concretely, we prove that polynomially padded $\text{L-uniform}$ constant-precision transformers are equivalent to $\text{L-uniform AC}^0$, while growing-precision ones achieve $\text{L-uniform TC}^0$ regardless of width. Furthermore, looping enables sequential processing analogous to circuits: $\log^d N$-looped constant-precision transformers reach $\text{FO-uniform AC}^d$, and growing-precision ones reach $\text{FO-uniform TC}^d$. Interestingly, growing width or precision beyond logarithmic does not increase expressivity, and all our results hold for both softmax and average hard attention transformers. 

---
# A Novel Global Context-aware Deep Neural Network for Enhanced Brain Tumor Segmentation using Magnetic Resonance Images 

**Authors**: Sourjya Mukherjee, Ananya Bhattacharjee, R. Murugan  

**Link**: [PDF](https://arxiv.org/pdf/2605.30510)  

**Abstract**: Brain cancer's severity necessitates precise brain tumor segmentation, which is crucial for effective brain tumor diagnosis. Manual identification, burdened by high costs, labor, and error risks, highlights the need for automated methods. In this study, we introduce the Global Context-aware Squeeze and Excite Residual UNet (GCSER-UNet), which facilitates a fusion of spatial and channel-wise attention and thus enhances the model's capacity to capture intricate spatial dependencies and contextual information. GCSER-UNet efficiently extracts tumor segments from multimodal MRI slices, delivering exceptional performance. Evaluations on benchmark databases exhibit its superiority, achieving a notable 94 percent dice score on the TCGA LGG dataset, surpassing the state-of-the-art dice score of 91.8 percent. In the BraTS 2020 dataset, the proposed GCSER-UNet ensemble approach yielded dice scores of 95 percent, 92 percent, and 90 percent for the tumor regions - Whole Tumor (W), Tumor Core (T), and Enhancing Tumor (E), respectively. The current state-of-the-art dice scores were 94 percent, 93 percent, and 88 percent. These compelling outcomes highlight the efficacy of GCSER-UNet in precise brain tumor segmentation and thus can aid neurologists in effective brain cancer management and treatment planning. 

---
# Improved Distribution Estimation in $\ell_\infty$ 

**Authors**: Doron Cohen, Aryeh Kontorovich, Yonatan Livshitz  

**Link**: [PDF](https://arxiv.org/pdf/2605.30509)  

**Abstract**: We present improved bounds for estimating discrete probability distributions under the $\ell_\infty$ norm. These include minimax bounds in expectation and high-probability tail bounds. We resolve some of the open questions posed in Kontorovich and Painsky (JMLR, 2025) -- including a fully empirical version of the tightest risk bound they presented and identifying the form of the worst-case extremal distribution. Encouraging empirical results are reported as well. 

---
# Graph-Conditioned Mixture of Graph Neural Network Experts for Traffic Forecasting 

**Authors**: Amirhossein Ghaffari, Saeid Sheikhi, Ekaterina Gilman  

**Link**: [PDF](https://arxiv.org/pdf/2605.30486)  

**Abstract**: Spatio-temporal forecasting on sensor graphs is commonly tackled with a single backbone architecture applied uniformly across all nodes, although graph regions can exhibit different dynamics. Road segments differ in functional class, structure, and traffic behavior, suggesting that node-wise expert specialization can be useful. We propose GC-MoE, a graph-conditioned mixture of experts framework that assigns each node a personalized combination of frozen forecasting experts based on graph topology and the recent traffic input window. GC-MoE combines frozen pretrained spatio-temporal GNN experts with an input-aware, spatially contextualized router while training only a lightweight routing module. We also study a bounded graph-conditioned output refinement layer as an optional extension and include node-adaptive ST-LoRA adapters only as an ablation diagnostic. Across four standard benchmarks (PEMS04, PEMS07, METR-LA, and PEMS-BAY), GC-MoE improves MAE over a zero-parameter ensemble baseline, with competitive RMSE and MAPE, while training only ~17K parameters on top of 1.5M frozen expert weights. The implementation is available at this https URL. 

---
# idSCD: Identifying Training Datasets through Semantic Correlation Descriptors 

**Authors**: Andrada Gobeaja, Ionut Hodoroaga, Elena Burceanu, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30462)  

**Abstract**: Can a dataset be recognized from the spurious correlations it induces during training? We argue that datasets leave dataset-specific traces in a model's learned semantic correlation structure: incidental regularities that are predictive within a dataset, but not causal for the underlying task, can be internalized during training. We use this insight to study dataset-level membership inference, moving beyond existing methods that rely on behavioral or distributional evidence such as confidence scores, losses, margins, generated samples, or query responses. We introduce a white-box semantic fingerprinting approach based on semantic correlation descriptors (SCDs), which capture the semantic correlation structure learned by a model and make it comparable across dataset mixtures. In a controlled leave-one-dataset-out diagnostic, SCDs recover dataset-specific changes and perfectly separate matching from non-matching dataset pairs. We then propose a practical SCD-based membership score that tests whether a target dataset is part of a model's training mixture using only the model's SCD and the target dataset's standalone SCD, without requiring leave-one-dataset-out models. Across three diverse experimental settings, with dataset groups for natural language inference, emotion classification, and medical text classification, we test both the advantages and limitations of SCD-based membership inference with different degrees of semantic separation and keyword support between dataset splits. On average, the classifier based on this score achieves the highest performance and the lowest std, outperforming black-box baselines RMIA, Attack-P, and LiRA, as well as the white-box SIF baseline. These results show that dataset membership can be traced through internal semantic correlations, with the largest relative gain exceeding 60% in ROC-AUC when dataset groups expose distinct semantic particularities. 

---
# Scalable Constrained Multi-Agent Reinforcement Learning via State Augmentation and Consensus for Separable Dynamics 

**Authors**: Santiago Amaya-Corredor, Miguel Calvo-Fullana, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2605.30461)  

**Abstract**: We present a distributed approach for constrained Multi-Agent Reinforcement Learning (MARL) that combines state-augmented policy learning with distributed consensus over dual variables. Our method targets systems where agents have separable dynamics but must coordinate to satisfy global resource constraints, a setting in which, as we demonstrate empirically, independent learning fails to produce feasible solutions because agents cannot determine appropriate individual contributions toward collective constraint satisfaction. The key technical contribution is showing that lightweight neighbor-to-neighbor consensus over Lagrange multipliers suffices for globally coordinated constraint enforcement while preserving the scalability of independent training. Each agent learns a single augmented policy offline, conditioned on both its local state and a dual variable encoding constraint feedback. During execution, agents reach agreement on this dual variable through local communication alone. We prove that under mild connectivity assumptions, the consensus error among agents' multipliers is bounded, and show that this translates to a bounded constraint violation that decreases with graph connectivity and the number of consensus rounds. Unlike centralized training with decentralized execution (CTDE) approaches, whose complexity grows at least quadratically with agent count, our method scales linearly in both training and execution. Experiments on smart grid demand response demonstrate that consensus coordination is \emph{essential for feasibility}: without it, agents satisfy grid capacity constraints only by indefinitely postponing demand, a degenerate non-solution. With consensus, agents converge to a shared dual variable and satisfy both grid constraints and demand fulfillment, scaling to thousands of agents while CTDE baselines are limited to dozens. 

---
# The Surface You Test Is Not the Surface That Breaks 

**Authors**: Shifat E Arman, Syed Nazmus Sakib, Nafiul Haque, Shahrear Bin Amin  

**Link**: [PDF](https://arxiv.org/pdf/2605.30454)  

**Abstract**: Tool-augmented LLM agents are vulnerable to prompt injection: a third party who controls part of the agent's context can plant instructions that the agent then executes as if they came from the user. Current evaluations report a single attack success rate per model on one channel, the tool output and treat that number as the model's vulnerability. But tool descriptions, which the agent reads at every turn before any tool is called, are themselves an injection surface that the attacker can choose instead. We hold the injection payload byte-identical and deliver it through both surfaces across 13 LLMs from six families and four task suites. The same bytes invert in success rate across models: GPT-4.1 is 96 percent vulnerable on tool outputs but only 4 percent on tool descriptions, while GEMINI-3-FLASH shows the mirror pattern at 20 percent and 98 percent. A variance decomposition over 6,830 attempts attributes 0 percent of the variation in attack outcomes to the surface alone, while the model-surface interaction accounts for 16.7 percent. Vulnerability is a property of the pairing, not the channel. The Adaptive Attack Rate, defined as the per-cell maximum over surfaces, exceeds the strongest fixed-surface baseline by +9.1 percentage points on average. Standard prompt-level defenses inherit the same blindspot, reducing tool-output ASR to 10-18 percent while leaving the description channel above 54 percent. Both attack and defense evaluation must report per-surface vulnerability. 

---
# A Unified Framework for Gradient Aggregation in Multi-Objective Optimization 

**Authors**: Zeou Hu, Kelvin Ho, Yaoliang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30452)  

**Abstract**: Many machine learning problems involve multiple inherent trade-offs that are best addressed by gradient-based multi-objective optimization (MOO) algorithms. Existing methods are often proposed with various motivations, analyzed case by case, and differ algorithmically in how the component gradients are aggregated at each step. In this work, we develop a unifying framework for gradient aggregation in MOO, establishing (optimal) rates of convergence to Pareto stationarity, the standard measure of performance in MOO. Central to our analysis is a sufficient alignment condition, from which we derive a theorem showing that non-conflicting directions, when chosen within the convex hull of gradients, form a fundamental sufficient condition for convergence. We further show that feasibility can be ensured through projection onto the dual cone, broadening the scope of methods that admit convergence guarantees. In parallel, we present a primal optimization perspective of gradient aggregation that encompasses established algorithms, clarifies their theoretical relationships, and enables the design of new variants. As an illustration, we introduce capped MGDA, derived from a CVaR-based formulation, and demonstrate its robustness in adversarial federated learning. Finally, we validate our theory through experiments on synthetic problems and practical benchmarks. 

---
# Calibrated Preference Learning: The Case of Label Ranking 

**Authors**: Santo M. A. R. Thies, Viktor Bengs, Timo Kaufmann, Sebastian J. Vollmer, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2605.30447)  

**Abstract**: Calibration, the alignment of predicted probabilities with true outcome frequencies, is essential for reliable decision-making. While extensively studied for classification and regression, calibration has not been formally addressed for probabilistic label ranking, where the goal is to predict a distribution over orderings of a label set. Naively treating rankings as classes ignores their structure and fails to capture important modalities such as pairwise and top-k predictions. We formalize calibration for label ranking and develop a hierarchy of notions covering full rankings, sub-rankings, and top-k rankings. We prove that full-rank calibration implies the others but not conversely, and sub-ranking and top-k calibration are incomparable. Empirically, we find popular label ranking models are often poorly calibrated, with substantial differences between sub-ranking and top-k metrics. Applying our framework to RLHF reward models, we find that calibration correlates strongly but not perfectly with benchmark accuracy, suggesting it captures a meaningful quality dimension beyond top-1 accuracy. These findings motivate future work on understanding the downstream effects of miscalibration and developing methods to correct it. 

---
# LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis 

**Authors**: Kewei Xu, Xiaoben Lu, Shuofei Qiao, Zihan Ding, Haoming Xu, Lei Liang, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30434)  

**Abstract**: Real-world data analysis is inherently iterative, yet existing benchmarks mostly evaluate isolated or short interactive tasks, leaving agents' ability to track evolving analytical context over long horizons untested. We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states. LongDS comprises 68 tasks constructed from real-world Kaggle notebooks, spanning 2,225 turns across six domains including Geoscience, Business, and Education. Tasks are designed around state-evolution patterns (e.g., counterfactual perturbation, rollback, multi-state composition), with an average dependency span of 11.3 turns. Evaluating five state-of-the-art models, we find that the best model reaches only 48.45% average accuracy, performance drops nearly 47 points from early to late turns, and long-horizon errors account for 52%--69% of failures. Further analysis shows that additional agent steps do not necessarily improve performance, suggesting that the key bottleneck is maintaining a correct analytical state rather than increasing interaction budget. We release LongDS to support research on reliable long-horizon agentic data analysis. Code and data will be released at this https URL. 

---
# Domain Adaptation and Reasoning Frameworks in Language Models: A Controlled Experiment with Historical Cosmology 

**Authors**: Francesco De Bernardis  

**Link**: [PDF](https://arxiv.org/pdf/2605.30415)  

**Abstract**: We investigate how domain adaptation reshapes explanatory behavior in language models using historical cosmology as a controlled setting. In Phase 1, we train a small language model from scratch on a pre-Copernican corpus from which explicit heliocentric references were removed, and evaluate whether Earth-motion or heliocentric continuations nevertheless emerge. In Phase 2, we fine-tune a larger pretrained model using QLoRA on the same corpus in order to study how adaptation modifies explanatory framing and cosmological stance. Model outputs are evaluated using an LLM-as-judge framework that labels both cosmological stance (geocentric, heliocentric, or ambiguous) and explanatory frame (premodern versus modern). In the constrained setting of Phase 1, the smaller models occasionally generate local Earth-motion continuations, but these remain globally unstable and insufficient to support coherent cosmological reasoning. In Phase 2, fine-tuning induces a large and statistically significant shift toward premodern explanatory framing, while the conditional cosmological stance distributions remain comparatively stable within those frames. As a result, increases in geocentric outputs arise primarily from redistribution over explanatory regimes rather than from direct modification of stance. These results suggest that domain adaptation may primarily reshape the linguistic frameworks from which continuations are generated, with changes in stance emerging secondarily from those shifts. 

---
# SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer 

**Authors**: Yuyang Zhao, Yicheng Pan, Qiyuan He, Jincheng Yu, Junsong Chen, Tian Ye, Haozhe Liu, Enze Xie, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.30409)  

**Abstract**: Real-time streaming video-to-video editing (V2V) is critical for interactive applications such as live broadcasting and gaming, yet it remains a formidable challenge due to the stringent requirements for temporal consistency and inference throughput. In this paper, we present SANA-Streaming, a system-algorithm co-designed framework for high-resolution, real-time streaming video editing on consumer GPUs, with the following three core designs: (1) Hybrid Diffusion Transformer architecture introduces softmax attention in part of the blocks to improve local modeling capabilities while preserving the efficiency of linear layers. (2) Cycle-Reverse Regularization is a novel training strategy that enforces semantic consistency by predicting source frames from generated content via flow matching, improving temporal consistency without requiring paired long edited videos. (3) Efficient System Co-design combines fused GDN kernels and Mixed-Precision Quantization (MPQ) optimized for the NVIDIA Blackwell (RTX 5090) architecture. By profiling real-world throughput, our MPQ maximizes Tensor Core utilization while maintaining generation quality. The resulting system achieves real-time 1280 x 704 resolution editing at 24 end-to-end FPS on a single RTX 5090 GPU, with the DiT core running at 58 FPS. Experimental results demonstrate that our co-design approach significantly outperforms existing SOTA methods in both temporal coherence and system throughput. 

---
# Exploring Autonomous Agentic Data Engineering for Model Specialization 

**Authors**: Yujie Luo, Xiangyuan Ru, Jingsheng Zheng, Jingjing Wang, Yuqi Zhu, Jintian Zhang, Runnan Fang, Kewei Xu, Ye Liu, Zheng Wei, Jiang Bian, Zang Li, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.30407)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-designed workflows, leaving it unexamined whether LLMs can autonomously execute an end-to-end data engineering pipeline for model specialization. We formalize \textbf{Autonomous Agentic Data Engineering}, a novel task designed to evaluate LLMs as autonomous data engineers that drive model specialization through end-to-end data curation. We frame data as an optimizable component and study agents that plan, generate, and iteratively optimize training data across multiple domains, guided by post-training performance improvement. Experiments show that autonomous LLM data engineers yield substantial gains, as GPT-5.2 constructs a training curriculum that improves a student model by \textbf{57.29\%}, entirely through iterative, agent-driven data adaptation. By illuminating both potential and bottlenecks, our study establishes autonomous data engineering as a measurable capability and charts a path toward agent-driven model specialization\footnote{Code will be released at this https URL.}. 

---
# AI Loss of Control Incident Management: Response & Resilience 

**Authors**: Ross Gruetzemacher  

**Link**: [PDF](https://arxiv.org/pdf/2605.30406)  

**Abstract**: Recent research demonstrating AI systems exhibiting deception and shutdown resistance suggests that AI loss of control (LOC) is an urgent policy concern , yet current literature focuses almost exclusively on alignment and prevention. To address this gap, this paper introduces a foundational framework and taxonomy for managing catastrophic AI LOC incidents. The taxonomy's first level distinguishes between scenarios where regaining control is 'extremely costly' versus 'impossible'. While impossible scenarios demand immediate resilience investments to fundamentally restrict an AI's attack surface , extremely costly scenarios require active incident management via Containment and Threat Neutralization. The framework further categorizes these manageable events into accidental LOC (requiring automated circuit-breaker responses) and adversarial LOC (requiring graduated escalatory measures). By mapping three severity classes to specific scenario matrices, this paper provides a concrete, proportional guide for managing unprecedented AI risks. 

---
# CodeGolf Bench: A Multi-Language Benchmark for Evaluating Concise Code Generation Capabilities of Large Language Models 

**Authors**: Vedant Padwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30394)  

**Abstract**: This paper introduces Code Bench, a benchmark capable of evaluating Large Language Models (LLMs) concise code generation abilities in 60 programming languages. Based on code golf, a recreational programming competition focused on minimal character or byte solutions, the benchmark provides a distinctive measure of LLMs ability to produce efficient, concise code. Unlike existing benchmarks limited by fixed problem sets and language coverage, CodeGolf Bench leverages the this http URL platform to provide new problems and live human performance baselines. Evaluation of nine LLMs on Python and C++ tasks demonstrates that reasoning models significantly outperform non-reasoning models, achieving best average percentile of 70.97%. This performance gap is particularly pronounced in C++, highlighting reasoning's importance for languages with strict syntax requirements. Non-reasoning models struggle more with efficiency optimization across both languages, with best percentiles significantly lower than reasoning counterparts. CodeGolf Bench offers a dynamic framework for evaluating LLM code generation capabilities against evolving human performance on code golf. 

---
# NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models 

**Authors**: Anany Kotawala  

**Link**: [PDF](https://arxiv.org/pdf/2605.30393)  

**Abstract**: Public numeric benchmarks appear in pretraining, so an evaluation that conditions on a date may be measuring memorized recall rather than out-of-sample skill. We introduce NumLeak, a measurement framework that combines API-boundary probes on production models with a white-box controlled validation on an open causal LM. Top-tier frontier LLMs recall the Fama-French market excess return at 3-seed pooled Pearson r=0.97-0.99 while staying within 0.15 within-25bps on the five sibling factors; comparable fidelity appears on U.S. unemployment, CPI inflation, and NOAA temperature. On a recent-release holdout, parse rate collapses to 21-57% but r stays at approximately 0.99 on months answered, the refuse-or-recall asymmetry a memorized channel predicts. The white-box experiment reproduces the dose-response, and logprob ranking detects memorization that open-ended generation misses, implying closed-API black-box probes understate the channel. A Sonnet "date to market-sentiment" regression that correlates with true Mkt-RF at r=0.74 collapses to r=0.02 once the model's own recall is residualized out. A one-line system-prompt defense blocks 99.8% of a non-adaptive single-turn suffix attack set at near-zero utility cost on conceptual and historical-narrative queries 

---
# Social Reasoning in Machines: Investigating Collective Truth-Seeking Dynamics in Large Language Model Debate 

**Authors**: Tom Pecher  

**Link**: [PDF](https://arxiv.org/pdf/2605.30391)  

**Abstract**: Human reasoning has long been theorised to operate socially, not through isolated individual cognition, but through collective adversarial discourse, a framework known as the Argumentative Theory of Reasoning (ATR). Rather than relying on individual "intellectualist reasoners" as the primary vehicle for truth-seeking, ATR reconceptualises truth as an emergent property of social epistemology: the product of imperfect individual reasoning refined under the adversarial pressure of debate. This distributed method of collective intelligence has guided humanity to ever-greater epistemic heights and underpins the foundational principles of all democratic systems. This thesis breaks new ground by, for the first time, simulating ATR through the multi-agent debate (MAD) of large language models (LLMs). With rigorous empirical analysis, we demonstrate that, when correctly engineering an epistemically diverse set of models, LLM-MAD can significantly improve truth-seeking performance on questionnaire-based tasks, even when individual debate participants exhibit limited standalone performance. Furthermore, we present strong empirical evidence that this performance gain is mechanistically grounded in the central principles of ATR, suggesting that collective reasoning may be universally favourable over individualist reasoning, rather than a quirk in biology or evolution. Finally, drawing on our analysis of debate dynamics, we propose a novel benchmarking methodology that leverages LLM-MAD to measure intrinsic model properties (such as hallucination propensity) in order to compare models in ways that current static benchmarking approaches cannot support. 

---
# Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification 

**Authors**: Hwa Hui Tew, Junn Yong Loo, Fang Yu Leong, Julia K. Lau, Ding Fan, Hernando Ombao, Raphaël C.-W. Phan, Chee Pin Tan, Chee-Ming Ting  

**Link**: [PDF](https://arxiv.org/pdf/2605.30387)  

**Abstract**: Functional Magnetic Resonance Imaging (fMRI) provides non-invasive access to dynamic brain activity by measuring blood oxygen level-dependent (BOLD) signals over time. However, the resource-intensive nature of fMRI acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often remain challenging in replicating their inherent non-stationarity, intricate spatiotemporal dynamics, and physiological variations of raw BOLD signals. To address these challenges, we propose Dual-Spectral Flow Matching (DSFM), a novel fMRI generative framework that cascades dual frequency representation of BOLD signals with spectral flow matching. Specifically, our framework first converts BOLD signals into a wavelet decomposition map via a discrete wavelet transform (DWT) to capture globalized transient and multi-scale variations, and projects into the discrete cosine transform (DCT) space across brain regions and time to exploit localized energy compaction of low-frequency dominant BOLD coefficients. Subsequently, a spectral flow matching model is trained to generate class-conditioned cosine-frequency representation. The generated samples are reconstructed through inverse DCT and inverse DWT operations to recover physiologically plausible time-domain BOLD signals. This dual-transform approach imposes structured frequency priors and preserves key physiological brain dynamics. Ultimately, we demonstrate the efficacy of our approach through improved downstream fMRI-based brain network classification. The code is available at this https URL . 

---
# LLMs Without Deep Neural Networks: New Architecture, Benefits and Case Study 

**Authors**: Vincent Granville  

**Link**: [PDF](https://arxiv.org/pdf/2605.30385)  

**Abstract**: The purpose of this article is to provide validation to my deep neural network alternative in the context of LLMs. Very recently, there has been a significant interest by Chinese researchers in a model called RBF network, as a substitute to standard DNNs, with increased explainability and higher accuracy. It turns out that my new model, discovered independently, is based on the exact same machinery. But with a major twist: it does not need DNN as it finds the global optimum of the loss function in closed form, in one iteration, thus eliminating the tedious training step. Here I provide a high-level overview of my technology, with case study and comparison to similar methods. 

---
# Structured interactions improve distributed coordination beyond model scaling in a real-world multi-robot system 

**Authors**: Junping Wang, Zhizhong Zhang, Yongqiang Tang, Geng Zheng, Jiaming Zhang, Shiji Song, Yanmei Li, Yushan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.30383)  

**Abstract**: Scaling individual robot capabilities is common but costly. Here we investigate a system-level design question in real-world multi-robot coordination: given matched hardware budgets, does restructuring communication among robots yield larger gains than increasing onboard model size? Using a representative transport-and-mapping task with 10 physical robots (5 runs per condition, 60 runs total), we find that switching from fully connected to modular hierarchical interactions improves normalised performance by 47 points (0--100), whereas doubling neural network hidden size yields at most 9 points. Nested mixed-effects model comparisons show a substantially larger improvement in model fit for topology than for scale. The pattern is confirmed in independent SMAC replications; heterogeneous benchmark reanalyses provide secondary supporting consistency checks rather than primary evidence. Performance saturation beyond 1024 hidden units is observed in simulation-calibrated extrapolation, not directly on hardware. These results indicate that interaction structure can play a dominant role within the tested system and task setting, while broader quantitative generalisation remains to be established. 

---
# When LLMs Learn to Be Consistently Wrong: A Multi-Model Study of Linear Representations of Synthetic Deception 

**Authors**: Vahideh Zolfaghari  

**Link**: [PDF](https://arxiv.org/pdf/2605.30381)  

**Abstract**: Deceptive alignment, in which models maintain accurate internal representations while deliberately producing false outputs, remains a central challenge in AI safety. While strategic deception is the primary long-term concern, synthetic dishonesty - induced via direct optimization on incorrect answers - provides a controlled testbed for studying the representational basis of learned deception. We introduce a multi-model paradigm in which honest and deceptive variants of five transformer models (Pythia-1.4B, Gemma-2-2B/9B, Qwen2.5-7B, Llama-3.1-8B) are fine-tuned using LoRA on the same question distribution. Linear probes trained on mean-pooled hidden states detect synthetic dishonesty with near-perfect AUC (greater than or equal to 0.99) as early as layers 1-3 in four architectures, while Pythia-1.4B reaches a peak of 0.705. Logistic regression probes consistently match or outperform MLP probes, supporting the Linear Representation Hypothesis. Probes trained on TruthfulQA generalize with near-zero loss (Delta AUC approx. 0) to held-out MMLU subjects. Late-layer representations show strong robustness to Gaussian noise, with Gemma-2 models exhibiting exceptional stability. Mechanistic analysis of Fisher Discriminant Ratio, effective rank, centroid geometry, directional stability, cross-domain alignment, and calibration (ECE) reveals two regimes: representational collapse in Pythia/Llama/Qwen versus high-dimensional preservation in Gemma-2. Across all models, the dishonesty direction consolidates progressively in deeper layers, with optimal calibration (ECE less than 0.01 except Pythia) achievable in layers 1-4. These results demonstrate that robust, domain-invariant dishonesty representations can be rapidly entrenched via modest supervised fine-tuning, with implications for activation-based monitoring. 

---
# Unicorn: Scaling High-Dimensional Time Series Forecasting via Universal Correlation Modeling 

**Authors**: Haochen Yuan, Yichen Song, Yunbo Wang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30376)  

**Abstract**: Modern time series architectures face a fundamental trade-off: channel-independent models scale well with increasing data volume but ignore critical inter-channel dependencies, while channel-dependent models are expressive but remain ``dimension-bounded'', struggling to generalize across heterogeneous this http URL bridge this gap, we introduce Unicorn (Universal Correlation Network), a framework for scalable, multi-dataset pretraining on high-dimensional time series. At the core of Unicorn is a latent prototype codebook that decouples correlation modeling from specific channel identities. By projecting heterogeneous channels into a shared latent space, UniCorN learns identity-agnostic, reusable interaction patterns that transfer across domains with diverse dimensionalities and semantics. Extensive experiments show that Unicorn significantly outperforms state-of-the-art forecasting architectures, particularly in few-shot transfer scenarios, offering a scalable path toward multivariate time series foundation models. 

---
# Full-field prediction for engineering-scale three-dimensional aircraft with multigrid-hierarchical learning 

**Authors**: Yunfei Liu, Hao Wang, Yuhang Qi, Hao Yue, Dehong Meng, Wei Li, Rui Wang, Tiejun Li, Jie Liu, Junwu Hong, Xinhai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30375)  

**Abstract**: High-fidelity computational fluid dynamics is essential for aerospace design, but engineering-scale simulations of practical three-dimensional aircraft remain computationally expensive. Learning-based flow-field initialization can improve efficiency by reducing the numerical distance between the initial and converged solutions, yet existing deep learning approaches remain difficult to scale to large three-dimensional aircraft flows with multiscale regional heterogeneity. Most prior studies therefore focus on two-dimensional problems, surface quantities, integral aerodynamic coefficients, or simplified three-dimensional cases with limited grid this http URL we propose MHLF, a multigrid-hierarchical learning framework for accelerating engineering-scale aircraft flow simulations while preserving high-fidelity numerical accuracy. MHLF combines a topologically consistent geometric multigrid representation with a hierarchical strategy that captures regional flow heterogeneity during both prediction and subsequent CFD correction. Across three engineering-scale aircraft cases spanning Mach 0.15 to 6.0 and covering subsonic, transonic and supersonic regimes, MHLF accelerates convergence without sacrificing flow-field accuracy, achieving a 3 to 8 times efficiency improvement over conventional initialization. These results demonstrate practical full-flow-field prediction for large three-dimensional aircraft within the CFD domain and provide a foundation for data-driven acceleration of high-fidelity aircraft flow simulation. 

---
# Evolutionary Algorithm for Reservoir Learning and Yielding 

**Authors**: Julien Testu, Pierrick Legrand, Xavier Hinaut  

**Link**: [PDF](https://arxiv.org/pdf/2605.30372)  

**Abstract**: Reservoir computing, a type of recurrent neural network, is a promising approach for temporal learning as it separates dynamic processing from the trained readout layer. However, classical Echo State Networks (ESNs) often require task-specific tuning of their architecture and hyperparameters to achieve good performance. This paper introduces EARLY (Evolutionary Algorithm for Reservoir Learning and Yielding), a framework designed to evolve both the topology and hyperparameters of multi-reservoir ESNs. Inspired by the modular organisation of the brain, EARLY encodes architectures as graph-based genomes and applies crossover, mutation, and selection to discover effective configurations. Our goal is to create both generic architectures and tasks inducing generalization. The method is evaluated on temporal learning tasks from the CogScale dataset. Results show that evolved architectures outperform those obtained with random search on several tasks and exhibit structural differences depending on task difficulty: simpler tasks yield lightweight architectures, while more complex tasks favour richer modular organisations. These findings suggest that evolutionary search can help identify reusable reservoir structures for a broader range of temporal problems. The evolved architectures are further evaluated on a cross-situational learning dataset to assess their ability to adapt to new environments. 

---
# Updating the standard neuron model in artificial neural networks 

**Authors**: Raul Mohedano, Thomas Batard, Erik Velasco-Salido, Ramsses De Los Santos Mendoza, Jorge H. Martínez, Stacey Levine, Marcelo Bertalmío  

**Link**: [PDF](https://arxiv.org/pdf/2605.30370)  

**Abstract**: From their inception in the 1950s, artificial neural networks (ANNs) started using the so-called point neuron model then prevalent in neuroscience, hoping that this analogy would allow for a better emulation of brain function. Over the years the neuroscience literature has shown that the point neuron model is too simplistic to properly represent many fundamental neural processes; however, the standard neuron model in ANNs still remains the same. Here we substitute it by a very recent model of cortical cells and demonstrate through theoretical analyses and experimental results how, simply by using a more realistic neural unit element without augmenting the number of parameters, the resulting ANNs offer a number of important advantages that include increases in expressivity, robustness and learning speed, and a reduction in memorization and the amount of training data needed. 

---
# Reinterpreting Safety Thresholds as Neuron Spiking Thresholds 

**Authors**: Enrico Del Re, Mohamed Sabry, Cristina Olaverri-Monreal  

**Link**: [PDF](https://arxiv.org/pdf/2605.30368)  

**Abstract**: Surrogate Safety Measures (SSMs) are extensively utilised in the evaluation of traffic risk in automated driving contexts. However, the majority of SSM-based evaluations employ fixed thresholds that fail to capture the human response to sustained borderline conditions or the reaction to brief, high-risk peaks. The present work proposes a biologically inspired reinterpretation of SSM thresholds. This is modelled as spiking thresholds of leaky integrate-and-fire (LIF) neurons, with multiple SSM inputs combined into a spiking neural network (SNN). The SNN is trained to emit spikes that are aligned with human braking onsets. The training data was recorded in a controlled car-following experiment using the 3D-CoAutoSim platform with CARLA/Unreal and a 6-DOF motion platform, where induced critical events were generated. The results demonstrate that the learned spiking activity qualitatively aligns with braking behaviour across scenarios and captures reactions that are not consistently explained by threshold crossings alone. Analysis across participants further indicates that learned input thresholds remain relatively consistent, while learned decay factors encode different temporal sensitivities for the SSMs. The findings of this study indicate that spiking dynamics may serve as a mechanism to facilitate the convergence of objective SSMs with subjective human safety perception. 

---
# Mental Damage: Caption Poisoning Attacks on Retrieval-Augmented Text-to-Music Generation 

**Authors**: Yizhu Wen, Shuhao Zhang, Nan Zhang, Long Cheng, Hanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.30365)  

**Abstract**: Retrieval-augmented text-to-music (TTM) systems augment underspecified user prompts using captions retrieved from a music caption dataset. This design introduces an integrity dependency on the music knowledge database. We show that an attacker can poison the database by injecting a small number of crafted music captions, causing the system to retrieve malicious captions that bias prompt augmentation and steer generation away from the user's intended function, without modifying the user prompt, retriever, or generator. To achieve the music caption poisoning attack, we propose a dual-layer caption poisoning strategy that preserves high-level retrieval anchors while injecting low-level acoustic descriptors to steer prompt augmentation and downstream music generation toward an attacker-chosen target intent. In a MusicCaps knowledge database, CLAP retriever, and MusicGen pipeline, poisoned generations move substantially closer to the attacker's target, while remaining comparably aligned with the original user query. These results expose a practical integrity risk for retrieval-augmented creative AI systems. Our demo can be found at: this https URL 

---
# Hamiltonian-Inspired Attention Mechanism for Scalable RF Transmitter Fingerprinting 

**Authors**: Chitraksh Singh, Monisha Dhanraj, Akram Sheriff  

**Link**: [PDF](https://arxiv.org/pdf/2605.30364)  

**Abstract**: Radio-frequency (RF) fingerprinting identifies wire-less transmitters using hardware-induced imperfections present in baseband I/Q signals. However, deep learning models often degrade under receiver and channel distribution shifts, particularly as transmitter populations grow. This work proposes the Hamiltonian Transformer, a physics-informed attention architecture that enforces norm preserving value dynamics within each attention head using a learned skew-symmetric generator and a Störmer-Verlet leapfrog integration step. An additional phase-increment embedding exposes oscillator dynamics at the input layer. All experiments use non-equalized raw I/Q signals from the WiSig dataset under four protocols: same-day classification, cross-receiver generalisation, cross-day generalisation, and transmitter scaling up to 150 devices. The Hamiltonian Transformer achieves 99.12% accuracy under same-day conditions and 61.64% at 150 transmitters, consistently outperforming CNN and Transformer baselines across all scale points. A controlled ablation study identifies norm-preservation in the value update as the primary inductive bias driving the scaling advantage, with the phase increment embedding providing the single largest per-component improvement. These results indicate that embedding physics-informed structural priors into attention mechanisms is an effective approach to large-scale transmitter identification on raw wireless signals. 

---
# Enhancing Regime Shift Detection Using Unstructured Data: A Study on the Treasury Market 

**Authors**: Mingxuan Yi, Vidal Mehra, Jing Chen, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.30363)  

**Abstract**: Regime shifts in financial markets reorganise the joint dynamics of asset prices and macro variables, breaking any single-regime calibration. They are nonetheless difficult to detect reliably because the data signal is noisy and heavily multicollinear, while the contemporaneous text that announces them is unstructured. Standard regime shift detection methods rely solely on structured time-series data and ignore policy communications, even though these texts often signal shifts before they materialise in observed prices. We propose a text-enhanced regime shift detection pipeline that combines large language model (LLM) reasoning over central-bank communications with statistical validation on multivariate financial time series. The framework is detector-agnostic: text-proposed candidates are validated using a bootstrap likelihood-ratio test on a vector autoregression (VAR), while data-driven candidates from arbitrary regime detectors are ratified through a lenient LLM text check. We evaluate the framework on 2010-2024 FOMC minutes paired with a 14-variable U.S. Treasury and macroeconomic panel, using four interchangeable data-driven detectors. The proposed pipeline achieves F1 = 0.82 against a verified anchor list of monetary-policy regime shifts, with same-day modal detection latency and consistently stronger performance than pure data-driven baselines. The results demonstrate that combining unstructured policy text with statistical structural-break detection improves the robustness and interpretability of regime shift identification in financial markets. 

---
# XOResNet: Exclusive-OR Meta-Residuals Facilitate Deep Spiking Neural Networks Learning 

**Authors**: Jianfang Wu, Junsong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.30362)  

**Abstract**: Spiking neural networks (SNNs) hold promise for demonstrating superior learning and representation capabilities in deep models. Given the tremendous success of ResNet in deep learning, it would naturally follow to train deep SNNs with residual learning. However, existing residual structures for constructing deep SNNs still present challenges of spike redundancy or information loss, as well as redundant learning. In the present study, we first aim to address issues of relative spike redundancy in identity mapping and information loss in non-identity mapping. To this end, we propose an OR-ADD (OA) shortcut connection to merge output spikes/currents from two branches in the residual structure. Furthermore, to mitigate redundant learning in the backbone branch of the residual structure, we introduce the concept of XOR meta-residuals, i.e., selecting pre-learning residuals using the Exclusive-OR (XOR) operation for the backbone branch. Finally, by integrating the OA shortcut and XOR meta-residuals, we devise the XOR residual block and further construct XOResNet with varying depths based on this block. Extensive experiments on four datasets, Fashion-MNIST, CIFAR-10, CIFAR-100, and miniImageNet, show that the proposed XOResNet outperforms existing state-of-the-art deep SNNs optimized via gradient descent. These results validate the effectiveness of our OA shortcut and XOR meta-residual components in overcoming fundamental limitations of residual learning in SNNs, providing new architectural insights for building high-performance neuromorphic systems. 

---
# Gradient-Free Training of Spiking Neural Networks via Low-Rank Evolution Strategies 

**Authors**: Dhruv Patankar, Sachit Ramesha Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2605.30361)  

**Abstract**: Spiking Neural Networks (SNNs) offer compelling energy efficiency on neuromorphic hardware, yet their training remains challenging because the discrete spike threshold is non-differentiable. Surrogate-gradient methods sidestep this by approximating the derivative, but they impose backpropagation infrastructure that is incompatible with on-chip learning. Evolution Strategies (\es) are a natural gradient-free alternative, yet their computational cost scales with the number of parameters, making them impractical for large weight matrices.
We present a method for training SNNs using EGGROLL, a low-rank factorisation of ES perturbations that reduces per-generation memory from $\mathcal{O}(mn)$ to $\mathcal{O}(r(m{+}n))$. Combining EGGROLL with a Leaky Integrate-and-Fire SNN on N-MNIST, we demonstrate that gradient-free training achieves 79.21% test accuracy while reducing per-generation wall-clock time by 2.23$\times$ relative to full-rank ES. Our results demonstrate EGGROLL is viable for SNN training, with a clear accuracy-speed tradeoff, compatible with training on neuromorphic hardware without surrogate gradients. 

---
# When LLM Reward Design Fails: Diagnostic-Driven Refinement for Sparse Structured RL 

**Authors**: Youting Wang, Yuan Tang, Bowen Liu, Xuan Liu, Dingyan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28918)  

**Abstract**: For sparse, structured reinforcement-learning tasks with semantic reward-function interfaces, LLM-generated reward shaping is better framed as debugging than one-shot generation. We study PPO-trained agents using MiniGrid as core evaluation and MuJoCo as boundary stress test. Our audit finds two dominant one-shot failure modes -- reward flooding and semantic/API misunderstanding -- plus a rarer weak-shaping case. We propose diagnostic-driven iterative refinement, where training diagnostics and a failure-mode taxonomy guide targeted reward-function revision. Refinement improves DoorKey-8x8 from 2.3% to 97.6% and KeyCorridor from 31.2% to 86.7% with high seed-to-seed variance. Controls show these gains are not from retrying or extra training: metrics-only re-prompting yields large drops, while a static-vocabulary control recovers much of the gap (87.6%; 70.7%), showing the taxonomy prompt is a major mechanism and dynamic labels provide only partially isolated incremental evidence. Budget-matched and Best-of-3 comparisons separate refinement from selection and training-time effects. Component-removal tests, sensitivity analyses, and an audit against author labels provide converging evidence for the debugging interpretation while revealing calibration limits. Continuous-control results show the boundary: success-based diagnostics can misfire in dense-reward locomotion, and return-trend feedback removes one false-positive mechanism without robust gains. The low-call protocol is a cost contrast with population-based reward search, not a benchmark comparison. In four crossed-variance-design environments, point estimates suggest larger gains when LLM reward-function variance dominates but bootstrap intervals are wide. The method is bounded to sparse structured tasks with reliable interfaces under PPO; fields like event_text may help, hurt, or be neutral. 

---
# TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI 

**Authors**: Hyunwoo Oh, Hanning Chen, Sanggeon Yun, Yang Ni, Suyeon Jang, Behnam Khaleghi, Fei Wen, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2603.22867)  

**Abstract**: Multimodal stacks that mix ViTs, CNNs, GNNs, and transformer NLP strain embedded platforms because their compute/memory patterns diverge and hard real-time targets leave little slack. TRINE is a single-bitstream FPGA accelerator and compiler that executes end-to-end multimodal inference without reconfiguration. Layers are unified as DDMM/SDDMM/SpMM and mapped to a mode-switchable engine that toggles at runtime among weight/output-stationary systolic, 1xCS SIMD, and a routable adder tree (RADT) on a shared PE array. A width-matched, two-stage top-k unit enables in-stream token pruning, while dependency-aware layer offloading (DALO) overlaps independent kernels across reconfigurable processing units to sustain utilization. Evaluated on Alveo U50 and ZCU104, TRINE reduces latency by up to 22.57x vs. RTX 4090 and 6.86x vs. Jetson Orin Nano at 20-21 W; token pruning alone yields up to 7.8x on ViT-heavy pipelines, and DALO contributes up to 79% throughput improvement. With int8 quantization, accuracy drops remain <2.5% across representative tasks, delivering state-of-the-art latency and energy efficiency for unified vision, language, and graph workloads-in one bitstream. 

---
