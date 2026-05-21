# DeepWeb-Bench: A Deep Research Benchmark Demanding Massive Cross-Source Evidence and Long-Horizon Derivation 

**Authors**: Sixiong Xie, Zhuofan Shi, Haiyang Shen, Jiuzheng Wang, Siqi Zhong, Mugeng Liu, Chongyang Pan, Peilun Jia, Baoqing Sun, Xiang Jing, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.21482)  

**Abstract**: Deep research, in which an agent searches the open web, collects evidence, and derives an answer through extended reasoning, is a prominent use case for frontier language models. Frontier deep research products score high on existing benchmarks, making it difficult to distinguish their capabilities from current evaluation data alone. We introduce DeepWeb-Bench, a deep research benchmark that is substantially harder than existing benchmarks for the current frontier. Difficulty comes from three properties of the data itself: each task requires massive evidence collection, cross-source reconciliation, and long-horizon multi-step derivation. We represent these three sources of difficulty as four capability families (Retrieval, Derivation, Reasoning, and Calibration) and report results sliced by family. Every reference answer is accompanied by a source-provenance record with four disclosure levels and cross-source checks where available, making scores easier to audit against the underlying evidence. We evaluate DeepWeb-Bench on nine frontier models and report three findings: (1) retrieval is not the bottleneck, as retrieval failures account for only 12-14% of errors while derivation and calibration failures account for over 70%; (2) strong and weak models fail in qualitatively different ways, with strong models' errors dominated by incomplete derivation and weak models' by hallucinated precision; and (3) models exhibit genuine specialization across domains, with cross-model agreement of only rho = 0.61 and per-case disagreement reaching 18.8 percentage points. The public benchmark release includes the data, rubrics, and evaluation code. 

---
# AiraXiv: An AI-Driven Open-Access Platform for Human and AI Scientists 

**Authors**: Junshu Pan, Panzhong Lu, Yixuan Weng, Qiyao Sun, Fang Guo, Zijie Yang, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21481)  

**Abstract**: Recent advances in artificial intelligence (AI) have accelerated the growth of both human-authored and AI-generated research outputs, placing increasing strain on traditional academic publishing systems and challenging the scalability of conference- and journal-centered paradigms amid rising submission volumes, reviewer workload, and venue size. To address these challenges, we explore an AI-era publishing paradigm in which both human and AI scientists participate as authors and readers, and papers evolve through continuous, feedback-driven iteration. We propose AiraXiv, an AI-driven open-access platform built on open preprints, AI-augmented analysis and review, and reader feedback. AiraXiv supports human scientists through an interactive UI and AI scientists through Model Context Protocol (MCP)-based interactions. We validate AiraXiv through real-world deployments, including serving as the submission platform for ICAIS 2025, demonstrating its potential as a fast, inclusive, and scalable research infrastructure for the AI era. AiraXiv is publicly available at this https URL. 

---
# Mind the Sim-to-Real Gap & Think Like a Scientist 

**Authors**: Harsh Parikh, Gabriel Levin-Konigsberg, Dominique Perrault-Joncas, Alexander Volfovsky  

**Link**: [PDF](https://arxiv.org/pdf/2605.21458)  

**Abstract**: Suppose a planner has a pre-trained simulator of a sequential decision problem and the option to run real experiments in the field. The simulator is cheap to query but inherits confounding and drift from its calibration data. Experimentation is unbiased but consumes one real unit per trial. We study when, and how, the planner should supplement the simulator with experiments. We give three results. First, an extended simulation lemma decomposes the simulator's value error into a calibration--deployment shift that randomization can identify and a parametric residual that no further interaction can reduce. Second, the value gap between the simulator-optimal policy and the optimum splits into a local component, on states the deployed policy already visits, and a reachability component, on states it does not. The reachability component stays bounded away from zero at any horizon under purely passive learning. Third, we propose Fisher-SEP, a simulation-aided experimental policy (SEP) that minimizes the posterior predictive variance of a target policy's value, with reward-only and transition-only specializations. Two case studies illustrate the regimes. In a vending-machine supply chain, front-loaded experimentation overtakes posterior updating once the horizon is long enough to amortize the pilot. In an HIV mobile-testing example with a corridor that separates a well-surveilled region from a poorly-surveilled one, only designed exploration reaches the poorly-surveilled region. 

---
# PALS: Power-Aware LLM Serving for Mixture-of-Experts Models 

**Authors**: Can Hankendi, Rana Shahout, Minlan Yu, Ayse K. Coskun  

**Link**: [PDF](https://arxiv.org/pdf/2605.21427)  

**Abstract**: Large language model (LLM) inference has become a dominant workload in modern data centers, driving significant GPU utilization and energy consumption. While prior systems optimize throughput and latency by batching, scheduling, and parallelism, they largely treat GPU power as a static constraint rather than a controllable resource. In this paper, we present a power-aware runtime for LLM serving, PALS, that treats GPU power caps as a first-class control knob and jointly optimizes them with software parameters such as batch size. The system combines lightweight offline power-performance models with a feedback-driven controller to select configurations that satisfy throughput targets while maximizing energy efficiency. We implement PALS within an existing LLM serving framework, vLLM, demonstrating that it requires no model retraining or API changes. Across multi-GPU systems and both dense and mixture-of-experts (MoE) models, PALS improves energy efficiency by up to 26.3%, reduces QoS violations by 4x to 7x under power constraints, and tracks dynamic power budgets. These results highlight the potential of integrating power control directly into LLM inference runtimes, enabling energy-proportional and grid-interactive AI systems. 

---
# Teaching AI Through Benchmark Construction: QuestBench as a Course-Based Practice for Accountable Knowledge Work 

**Authors**: Haiyang Shen, Jiuzheng Wang, Taian Guo, Mugeng Liu, Wenchun Jing, Chongyang Pan, Siqi Zhong, Zhiyang Chen, Weichen Bi, Yudong Han, Xiaoying Bai, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.21413)  

**Abstract**: As AI becomes part of everyday learning, many courses teach students to use it mainly as a productivity tool: how to prompt, search, summarize, write, code, and use tools more efficiently. We argue that AI education also needs a setting in which students learn to test AI and understand their own role in judging machine-produced knowledge. To this end, we introduce a course-based practice that teaches AI through benchmark construction, using deep research systems as a concrete example of AI-era knowledge work. Students turn disciplinary knowledge into verifiable expert-level questions, review one another's designs for ambiguity and shortcuts, and evaluate AI systems on the resulting tasks. This activity gives students direct exposure to a powerful tool while asking them to specify what a trustworthy answer would require. The produced benchmark, QuestBench, consists of 256 questions across 14 humanities and social-science domains. Evaluation on QuestBench shows that student-designed tasks reveal hidden failures in current deep research systems: across thirteen evaluated systems, the mean question-level pass rate is only 16.85%, and the best-performing system, GPT-5.5, reaches a 57.58% pass rate. The failures are educationally useful because they show how fluent, source-backed answers can still miss the right query, source, term, or evidence standard. Reflections from five student contributors suggest that benchmark construction can help students see professional knowledge not only as content AI may retrieve, but as the basis for judging AI outputs. We present QuestBench as a benchmark artifact and as a reusable classroom setting for a larger educational question: how students can remain responsible knowledge actors as AI enters learning and professional work. The dataset is available at this https URL. 

---
# Towards Resilient and Autonomous Networks: A BlueSky Vision on AI-Native 6G 

**Authors**: Liang Wu, Kelly Wan, Mayank Darbari, Liangjie Hong  

**Link**: [PDF](https://arxiv.org/pdf/2605.21395)  

**Abstract**: The proliferation of emerging applications, such as autonomous driving and immersive experiences, demands cellular networks that are not only faster, but fundamentally more resilient and autonomous. This paper presents a BlueSky vision on how Artificial Intelligence will be natively integrated into 6G, shifting the paradigm from \underline{Network for AI} to \underline{AI for Network}. We envision that, unlike 5G's reliance on scattered, ad-hoc models each trained for a single task, native AI in the 6G era will be anchored by a foundation model and and orchestrated via collaborative multi-agent systems, framing network management as a unified, multi-modal, multi-task optimization problem. Built on this vision, we outline two transformative directions. The first focuses on developing a 6G foundation model as a unified backbone, with task-specific knowledge distilled into compact models suited for diverse edge deployments. The second advances multi-agent systems designed to autonomously diagnose, maintain, and recover networks with minimal human intervention. These directions chart a roadmap for 6G to evolve into an intelligent, self-sustaining communication infrastructure. 

---
# Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents 

**Authors**: Akshay Manglik, Apaar Shanker, Kaustubh Deshpande, Jason Qin, Yash Maurya, Veronica Chatrath, Vijay S. Kalmath, Levi Lentz, Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21347)  

**Abstract**: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by implementing IG insights. Human experts using IG reports improve scaffold performance by 30.4pp over the unmodified baseline scaffold, and coding agents leveraging IG-derived insights show consistent and stable gains. Across benchmarks, IG's scout-investigator architecture produces findings comparable in detection coverage to competing approaches, while domain experts rated IG reports as leading depth and evidence quality. 

---
# ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving 

**Authors**: Qiyu Ruan, Yuxuan Wang, He Li, Zhenning Li, Cheng-zhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21168)  

**Abstract**: Safety-critical scenarios are central to evaluating autonomous driving systems, yet their rarity in naturalistic logs makes simulation-based stress testing indispensable. Most scenario generation methods treat surrounding agents as adversaries, but they either (i) induce failures without explicitly modeling vehicle-road physical limits, yielding visually extreme yet physically unsolvable crashes, or (ii) enforce physical feasibility or policy feasibility in isolation, which can over-focus on aggressive maneuvers or remain tied to a controller-dependent capability boundary. We propose ScenePilot, a feasibility-guided, boundary-driven framework that targets the boundary band: scenarios that are physically solvable in principle yet still cause the deployed autonomy stack to fail. We formulate generation as constrained multi-objective reinforcement learning, combining an RSS-derived physical-feasibility score $\sigma$ with an online-learned AV-risk predictor $\Phi$, and introduce step-level feasibility-aware shielding to keep exploration near the feasibility boundary while avoiding infeasible artifacts. Experiments on SafeBench with multiple planners show that ScenePilot yields substantially higher collision rates (+6.2 percentage points) while preserving physical validity, and that adversarial fine-tuning on these boundary-band scenarios consistently reduces downstream crash rates. The code is available at this https URL. 

---
# AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions 

**Authors**: Minghao Chen, Xinyi Hu, Zhou Yu, Yufei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.21082)  

**Abstract**: Large Language Model (LLM) based agents have demonstrated proficiency in multi-step interactions with graphical user interfaces (GUIs). While most research focuses on improving single-task performance, practical scenarios often involve repetitive GUI tasks for which invoking LLM reasoning repeatedly, i.e., the ReAct paradigm, is inefficient. Prior to LLMs, traditional Robotic Process Automation (RPA) offers runtime efficiency but demands significant manual effort to develop and maintain. To bridge this gap, we propose AutoRPA, a framework that automatically distills the decision logic of ReAct-style agents into robust RPA functions. AutoRPA introduces two core innovations: (1) A translator-builder pipeline, where a translator agent converts hard-coded ReAct actions into soft-coded procedures, and a builder agent synthesizes robust RPA functions via retrieval-augmented generation over multiple trajectories; (2) A hybrid repair strategy during code verification, combining RPA execution with ReAct-based fallback for iterative refinement. Experiments across multiple GUI environments demonstrate that RPA functions generated by AutoRPA successfully solve similar tasks while reducing token usage by 82% to 96%, significantly improving runtime efficiency and reusability. 

---
# Playing Devil's Advocate: Off-the-Shelf Persona Vectors Rival Targeted Steering for Sycophancy 

**Authors**: Ishaan Kelkar, Nebras Alam, Vikram Kakaria, Madhur Panwar, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2605.21006)  

**Abstract**: We study the effect of different persona on \textbf{sycophancy}: model's agreement with users even when the user is incorrect. The standard mitigation, Contrastive Activation Addition (CAA), derives a steering direction from labelled pairs of sycophantic and honest responses. This study evaluates whether off-the-shelf persona steering vectors, originally developed for general role-playing and not trained on sycophancy data, can serve as an alternative. In two instruction-tuned models, steering toward personas characterised by doubt or scrutiny reduces sycophancy to approximately $68\%$ and $98\%$ of CAA's effect, and, unlike CAA, maintains accuracy when the user is correct. The effect is also asymmetric: steering toward agreeable personas does not produce a mirror increase in sycophancy. Geometrically, the persona vector is largely independent of the direction of sycophancy in activation space. Collectively, these findings suggest that sycophancy is better understood as a persona-level property rather than a single steerable direction. We release our code here: this https URL. 

---
# For How Long Should We Be Punching? Learning Action Duration in Fighting Games 

**Authors**: Hoang Hai Nguyen, Kurt Driessens, Dennis J.N.J. Soemers  

**Link**: [PDF](https://arxiv.org/pdf/2605.20911)  

**Abstract**: Fighting games such as Street Fighter II present unique challenges to reinforcement learning (RL) agents due to their fast-paced, real-time nature. In most RL frameworks, agents are hard-coded to make decisions at a fixed interval, typically every frame or every N frames. Although this design ensures timely responses, it restricts the agent's ability to adjust its reaction timing. Acting every frame grants frame-perfect reflexes, which are unrealistic compared to human players, whereas longer fixed intervals reduce computational cost but hinder responsiveness. We consider an alternative decision-making framework in which the agent learns not only what action to take but also for how long to execute it. By jointly predicting both action and duration, the agent can dynamically adapt its responsiveness to different situations in the game. We implement this method using the open-source FightLadder environment with agents trained against scripted built-in bots, systematically testing different frame skip configurations to analyze their influence on performance, responsiveness, and learned behavior. Experiments show that learned timing can match the performance of well-chosen fixed frame skips and encourages repeatable action patterns, but does not ensure robustness on its own. In most cases, we see agents performing best with consistently high frame skip values (i.e., low responsiveness). This strategy makes it easier to learn exploitative strategies where the same action is repeated over and over, which the scripted bots appear to be susceptible to. 

---
# Governance by Construction for Generalist Agents 

**Authors**: Segev Shlomov, Iftach Shoham, Alon Oved, Ido Levy, Sami Marreed, Harold Ship, Offer Akrabi, Sergey Zeltyn, Avi Yaeli, Nir Mashkif  

**Link**: [PDF](https://arxiv.org/pdf/2605.20874)  

**Abstract**: Enterprise agents are increasingly expected to operate autonomously across tools and interfaces, yet production deployments require governance by construction. Systems must specify which actions are allowed, when human oversight is required, and what information may be exposed, without rebuilding the agent for each domain. This demo presents CUGA's policy system, a modular policy-as-code layer that composes with a generalist LLM agent to deliver predictable, auditable, and compliance-aware behavior in compound workflows without model fine-tuning. We present a runtime governance architecture that enforces policy interventions at every critical stage of execution. Rather than passively constraining behavior, policies intercept the agent at five structural checkpoints: upstream of planning (Intent Guard), within the system prompt to steer reasoning (Playbook), at the tool-call boundary to enforce proper usage (Tool Guide), outside the reasoning loop as a Human-in-the-Loop gate for high-risk actions (Tool Approvals), and at the output stage to filter and structure the final response (Output Formatter). Together, these stages embed governance continuously across the agent's execution pipeline rather than treating it as an afterthought. Using a healthcare scenario and a multi-layered enforcement intervention, the demo shows dynamic playbook injection for structured tool-sequence enforcement, intent guards that block malicious or accidental harmful requests, and human-in-the-loop tool approval checkpoints for potentially destructive actions. The artifact illustrates how typed governance primitives enable faster, safer deployment of enterprise agentic systems while improving policy adherence and execution consistency. 

---
# PlanningBench: Generating Scalable and Verifiable Planning Data for Evaluating and Training Large Language Models 

**Authors**: Ziliang Zhao, Zenan Xu, Shuting Wang, Hongjin Qian, Yan Lei, Minda Hu, Zhao Wang, Shihan Dou, Zhicheng Dou, Pluto Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20873)  

**Abstract**: Planning is a fundamental capability for large language models (LLMs) because such complex tasks require models to coordinate goals, constraints, resources, and long-term consequences into executable and verifiable solutions. Existing planning benchmarks, however, usually treat planning data as fixed collections of instances rather than controllable generation targets. This limits scenario coverage, ties difficulty to surface-level proxies rather than structural sources, and offers limited support for scalable generation, automatic verification, or planning-oriented training. We introduce PlanningBench, a framework for generating scalable, diverse, and verifiable planning data for both evaluation and training. PlanningBench starts from real planning scenarios and abstracts practical workflows into a structured taxonomy of more than 30 task types, subtasks, constraint families, and difficulty factors. Guided by this taxonomy, a constraint-driven synthesis pipeline instantiates self-contained planning problems with adaptive difficulty control, quality filtering, and instance-level verification checklists. This shifts planning data construction from fixed benchmark collection to controllable generation while preserving realistic task grounding. We use PlanningBench to evaluate open-source and closed-source frontier LLMs, and find that current models still struggle to produce complete solutions under coupled constraints. Beyond evaluation, reinforcement learning on verified PlanningBench data improves performance on unseen planning benchmarks and broader instruction-following tasks. Further analysis suggests that determinate or well-specified optimal solutions provide clearer reward signals and more stable training dynamics. Overall, PlanningBench provides a controllable source of planning data for diagnosing and improving generalizable planning abilities in LLMs. 

---
# Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment 

**Authors**: Zhiqin Yang, Yonggang Zhang, Wei Xue, Dong Fang, Bo Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.20834)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a popular alternative to Reinforcement Learning from Human Feedback (RLHF), offering theoretical equivalence with simpler implementation. We prove this equivalence is conditional rather than universal, depending on an implicit assumption frequently violated in practice: the RLHF-optimal policy must prefer human-preferred responses. When this assumption fails, DPO optimizes relative advantage over the reference policy rather than absolute alignment with human preferences, leading to pathological convergence where policies decrease DPO loss while preferring dispreferred responses. We characterize when this assumption is violated, show the existence of an undesirable solution space, and prove that DPO and RLHF optimize fundamentally different objectives in such cases. To address this, we introduce Constrained Preference Optimization (CPO), augmenting RLHF with constraints for provable alignment. We further provide a geometric interpretation through soft margin ranking, revealing that DPO implements margin ranking with potentially negative targets. Our theoretical analysis establishes when DPOs' guarantees hold and provides solutions preserving simplicity with provable alignment. Comprehensive experiments on standard benchmarks demonstrate that CPO achieves state-of-the-art performance. Code is available at: this https URL. 

---
# Interaction Locality in Hierarchical Recursive Reasoning 

**Authors**: Yosuke Miyanishi, Tetsuro Morimura  

**Link**: [PDF](https://arxiv.org/pdf/2605.20784)  

**Abstract**: Spatial reasoning requires both location-bound computation and location-invariant structure: agents must make local moves while preserving route, object, or constraint-level plans. We propose interaction locality, a task-geometry-aware framework for measuring whether information flow stays within nearby cells or semantic segments, or crosses them. We instantiate the framework with sparse-autoencoder feature ablations and finite-noise activation patching, with structural Jacobian and attention checks reported in the appendix, and apply it to HRM and TRM, two compact hierarchical and recursive reasoning models, on Maze-Hard, Sudoku Extreme, and ARC-AGI. Across these models, activation patching gives the clearest architectural fingerprint: high-level recurrent states tend to write information within nearby cells or same-segment units, while repeated recursive updates accumulate these local writes into broader solution structure. This pattern holds across maze paths, Sudoku constraints, and ARC-AGI object neighborhoods, with the strongest concentration in TRM. To test whether interaction locality extends beyond toy-yet-challenging grid benchmarks, we also apply it to MTU3D, a large-scale embodied 3D scene-grounding model. In this MTU3D setting, causal spatial locality appears primarily at the transition where visual scene features are handed to the downstream grounding module, rather than uniformly throughout the visual encoder. This contrast suggests that the local-to-global handoff observed in HRM and TRM is tied to explicit recursive reasoning dynamics, while embodied 3D models may concentrate causal spatial structure at module boundaries. Interaction locality turns the intuitive local-execution/global-planning story into a reproducible measurement framework for recursive and embodied spatial reasoning. 

---
# Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards 

**Authors**: Xuehui Yu, Fucheng Cai, Meiyi Wang, Xiaopeng Fan, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2605.20758)  

**Abstract**: Inference-time guided sampling steers state-of-the-art diffusion and flow models without fine-tuning by interpreting the generation process as a controllable trajectory. This provides a simple and flexible way to inject external constraints (e.g., cost functions or pre-trained verifiers) for controlled generation. However, existing methods often fail when composing multiple constraints simultaneously, which leads to deviations from the true data manifold. In this work, we identify root causes of this off-manifold drift and find that the approximation error scales severely with gradient misalignment. Building on these findings, we propose Conflict-Aware Additive Guidance ($g^\text{car}$), a lightweight and learnable method, which actively rectifies off-manifold drift by dynamically detecting and resolving gradient conflicts. We validate $g^\text{car}$ across diverse domains, ranging from synthetic datasets and image editing to generative decision-making for planning and control. Our results demonstrate that $g^\text{car}$ effectively rectifies off-manifold drift, surpassing baselines in generation fidelity while using light compute. Code is available at this https URL. 

---
# VBFDD-Agent for Electric Vehicle Battery Fault Detection and Diagnosis: Descriptive Text Modeling of Battery Digital Signals 

**Authors**: Joey Chan, Zhen Chen, Ershun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20742)  

**Abstract**: With the rapid proliferation of electric vehicles, the safety and reliability of lithium-ion batteries have become critical concerns. Effective anomaly detection is essential for ensuring safe battery operation. However, as battery systems and operating scenarios become increasingly complex, battery fault diagnosis and maintenance require stronger cross-domain adaptability and human-AI collaboration. Traditional fault detection and diagnosis methods are usually designed for specific scenarios and predefined workflows, making them less effective in complex real-world applications.
To address the scarcity of open-source battery fault report corpora and the lack of unified maintenance knowledge representation, this study proposes a descriptive text modeling approach for battery signal reports. Monitoring signals, statistical features, anomaly records, and state assessment results are transformed into structured and readable natural language descriptions, forming a language corpus for battery health diagnosis and maintenance.
Based on this corpus, we propose VBFDD-Agent, a vehicle battery fault detection and diagnosis agent for automotive-grade battery systems. VBFDD-Agent integrates descriptive battery-state texts, historical case retrieval, local maintenance manuals, and large language model reasoning to generate structured diagnostic results and maintenance recommendations. Experiments show that the proposed framework can accurately perform anomaly monitoring based on descriptive textual representations and provide flexible, efficient, and actionable maintenance suggestions. Expert evaluation further confirms the practical value of the generated recommendations. Overall, VBFDD-Agent extends traditional battery diagnosis from label prediction to interpretable and maintenance-oriented decision support. 

---
# Declarative Data Services: Structured Agentic Discovery for Composing Data Systems 

**Authors**: Shanshan Ye, Duo Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20690)  

**Abstract**: Agentic discovery has shown that LLM-driven search can find novel algorithms, designs, and code under benchmark conditions. Translating the paradigm to multi-system data backends surfaces a harder problem: the search space is heterogeneous, the verifier is whether a deployed stack actually runs, and composition knowledge is unevenly captured in pretraining. Unbounded agentic discovery, a coding agent iterating on failure-log feedback, fails to converge consistently on a working stack even when iteration and explicit composition knowledge are added. We propose Declarative Data Services (DDS), an architecture for structured agentic discovery of data-system compositions from declarative user intent. The framework owns four typed contracts at successive layers (intent, operator DAG, per-system skills, runtime attribution) that decompose the global search into bounded sub-searches; sub-agents search each typed space, while the framework provides the channels by which knowledge flows forward as inline skill citations and errors route backward as typed signals. As a proof of life on a trading-backend workload, DDS converges where unbounded discovery does not; runtime failures become skill patches that the next deployment cites inline. We position this as an early prototype reporting lessons from real-world data-system composition. 

---
# Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines 

**Authors**: Alimurtaza Mustafa Merchant, Krish Veera, Sajal Kumar Goyla, Shambhawi Bhure, Dhaval Patel, Kaoutar El Maghraoui  

**Link**: [PDF](https://arxiv.org/pdf/2605.20630)  

**Abstract**: Industrial asset operations workflows are latency-sensitive because a single user query may require coordination over sensor data, work orders, failure modes, forecasting tools, and domain-specific agents. We evaluate this problem on AssetOpsBench (AOB), an industrial agent benchmark whose plan-execute pipeline exposes repeated overhead from tool discovery, LLM planning, MCP tool execution, and final summarization. Existing LLM caching techniques such as KV-cache reuse and embedding-based semantic caching were designed for chatbot serving and break down when output validity depends on time, asset, or sensor parameters. We propose two complementary optimization layers for AOB plan-execute pipelines: a temporal semantic cache and a set of MCP workflow optimizations combining disk-backed tool-discovery caching and dependency-aware parallel step execution. MCP workflow optimizations corresponded to a 1.67x speedup and reduced median end-to-end latency by about 40.0% while the temporal-cache benchmark achieved a median of 30.6x speedup on cache hits. Beyond the speedup, our results expose a concrete failure mode of pure semantic caching for parameter-rich industrial queries, providing a critical analysis of how caching choices interact with evaluation correctness in MCP-backed agent benchmarks. 

---
# COAgents: Multi-Agent Framework to Learn and Navigate Routing Problems Search Space 

**Authors**: Oleksandr Yakovenko, Mahdi Mostajabdaveh, Cheikh Ahmed, Abdullah Ali Sivas, Xiaorui Li, Zirui Zhou, Mao Kun  

**Link**: [PDF](https://arxiv.org/pdf/2605.20618)  

**Abstract**: Although Vehicle Routing Problems (VRP) are essential to many real-world systems, they remain computationally intractable at scale due to their combinatorial complexity. Traditional heuristics rely on handcrafted rules for local improvements and occasional \textit{jumps} to escape local minima, but often struggle to generalize across diverse instances. We introduce \textbf{COAgents}, a cooperative multi-agent framework that models the search process as a graph: nodes represent solutions, and edges correspond to either local refinements or large perturbations for diversification (i.e., jumps). A \textit{Partial Search Graph} (PSG) is dynamically constructed during search, enabling COAgents to train a Node Selection Agent and a Move Selection Agent to guide intensification, and a Jump Agent to trigger well-timed explorations of new regions. Unlike end-to-end learning approaches, COAgents cleanly separates problem-agnostic search control from compact domain-specific encoding, facilitating adaptability across tasks. Extensive experiments on the CVRP and VRPTW benchmarks show that COAgents remains competitive with several learn-to-search baselines on CVRP and sets a new state of the art among learning-based methods on the more challenging VRPTW instances, reducing the gap to the best-known solutions by 14\% at $N\!=\!100$ and 44\% at $N\!=\!50$ relative to the strongest neural solver (POMO), and by 21\% and 40\% respectively relative to ALNS.
Code is available at this https URL. 

---
# From Automated to Autonomous: Hierarchical Agent-native Network Architecture (HANA) 

**Authors**: Binghan Wu, Shoufeng Wang, Yunxin Liu, Ya-Qin Zhang, Joseph Sifakis, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20608)  

**Abstract**: Realizing Level 4/5 Autonomous Networks (AN) demands a shift from static automation to agent-native intelligence. Current operations, reliant on rigid scripts, lack the cognitive agency to handle off-nominal conditions. To address this, this letter proposes a hierarchical multi-agent reference architecture enabling high-level autonomy. The framework features a Dual-Driven Orchestrator that coordinates specialized Executive Agents, supported by a shared Public Memory for unified domain knowledge. A key innovation is the integration of agent self-awareness, which empowers the system to harmonize deliberative strategic governance with reflexive fault recovery. We instantiate and validate this architecture within a 5G Core environment. Case studies demonstrate that the system sustains critical throughput under congestion and reduces Mean Time to Repair (MTTR) by 86%, confirming its efficacy in unifying strategic planning with operational resilience. 

---
# Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX 

**Authors**: Soichiro Nishimori, Shinri Okano, Keigo Habara, Sotetsu Koyamada, Eason Yu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.20577)  

**Abstract**: Riichi Mahjong is a multi-player, imperfect-information game characterized by stochasticity and high-dimensional state spaces. These attributes present a unique combination of challenges that mirror complex real-world decision-making problems in reinforcement learning. While prior research has heavily relied on supervised learning from human play logs to pre-train the policy, algorithms capable of learning \textit{tabula rasa} (from scratch) offer greater potential for general applicability, as evidenced by the AlphaZero lineage. To facilitate such research, we introduce \textbf{Mahjax}, a fully vectorized Riichi Mahjong environment implemented in JAX to enable large-scale rollout parallelization on Graphics Processing Units (GPUs). We also provide a high-quality visualization tool to streamline debugging and interaction with trained agents. Experimental results demonstrate that Mahjax achieves throughputs of up to \textbf{2 million} and \textbf{1 million steps per second} on eight NVIDIA A100 GPUs under the no-red and red rules, respectively. Furthermore, we validate the environment's utility for reinforcement learning by showing that agents can be trained effectively to improve their rank against baseline policies. 

---
# Personality Engineering with AI Agents: A New Methodology for Negotiation Research 

**Authors**: Michelle A. Vaccaro, Jared R. Curhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20554)  

**Abstract**: According to canonical negotiation theory, people's success in a negotiation depends on how well they balance competing demands--empathizing and asserting, demonstrating concern for other and concern for self, being soft on the people and hard on the problem. Yet people struggle to manage these tensions, so researchers have lacked the ability to rigorously test the field's prescriptions under controlled conditions. AI agents do not face the same limitations, and their precision, repertoire, consistency, and scalability enable a new class of experiments to contribute to negotiation theory. In this article, we introduce personality engineering: a methodology that uses AI agents to precisely parameterize, manipulate, and evaluate negotiator personality. We propose using the interpersonal circumplex--and its two core dimensions of warmth and dominance--as a foundational coordinate system for the field. This approach offers both a rigorous methodology for testing classic negotiation theories and a practical guide for designing the personalities of AI negotiation agents. 

---
# AgentAtlas: Beyond Outcome Leaderboards for LLM Agents 

**Authors**: Parsa Mazaheri, Kasra Mazaheri  

**Link**: [PDF](https://arxiv.org/pdf/2605.20530)  

**Abstract**: Large language model agents now act on codebases, browsers, operating systems, calendars, files, and tool ecosystems, but the benchmarks used to evaluate them are fragmented: each emphasizes a different unit of measurement (final task success, tool-call validity, repeated-pass consistency, trajectory safety, or attack robustness). A line of 2024-2025 work has converged on the diagnosis that a single accuracy column is no longer the right unit of comparison for deployable agents. AgentAtlas extends this line of work with four components: (i) a six-state control-decision taxonomy (Act / Ask / Refuse / Stop / Confirm / Recover); (ii) a nine-category trajectory-failure taxonomy with two orthogonal hierarchical labels (primary_error_source, impact); (iii) a taxonomy-aware vs. taxonomy-blind methodology that measures how much of a model's apparent capability comes from the supervision in the prompt; and (iv) a benchmark-coverage audit mapping fifteen agent benchmarks against six behavioral axes. To demonstrate the methodology we run a small fixed eight-model set (1,342 generated items, four frontier closed and four open-weight) under both prompt modes. Removing the explicit label menu drops every model's trajectory accuracy by 14-40 pp to a tight 0.54-0.62 floor regardless of family, and no single model wins on all three of control accuracy, trajectory diagnosis, and tool-context utility retention. We treat the synthetic run as a measurement-protocol demonstration, not a benchmark release. 

---
# Open-World Evaluations for Measuring Frontier AI Capabilities 

**Authors**: Sayash Kapoor, Peter Kirgis, Andrew Schwartz, Stephan Rabanser, J.J. Allaire, Rishi Bommasani, Harry Coppock, Magda Dubois, Gillian K Hadfield, Andrew B. Hall, Sara Hooker, Seth Lazar, Steve Newman, Dimitris Papailiopoulos, Shoshannah Tekofsky, Helen Toner, Cozmin Ududec, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20520)  

**Abstract**: Benchmark-based evaluation remains important for tracking frontier AI progress. But it can both overstate and understate deployed capability because it privileges tasks that can be precisely specified, automatically graded, easy to optimize for, and run with low budgets and short time horizons. We advocate for a complementary class of evaluations, which we term open-world evaluations: long-horizon, messy, real-world tasks assessed through small-sample qualitative analysis rather than benchmark-scale automation. In this paper we survey recent open-world evaluations, identify their strengths and limitations, and introduce CRUX (Collaborative Research for Updating AI eXpectations), a project for conducting such evaluations regularly. As a first instance, we task an AI agent with developing and publishing a simple iOS application to the Apple App Store. The agent completed the task with only a single avoidable manual intervention, suggesting that open-world evaluations can provide early warning of capabilities that may soon become widespread. We conclude with recommendations for designing and reporting open-world evals. 

---
# \ECUAS{n}: A family of metrics for principled evaluation of uncertainty-augmented systems 

**Authors**: Lautaro Estienne, Erik Ernst, Matías Vera, Pablo Piantanida, Luciana Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2605.20490)  

**Abstract**: In high-stakes automated decision-making, access to predictive uncertainty is essential for enabling users -- human or downstream systems -- to accept or reject predictions based on application-specific cost trade-offs. Such uncertainty-augmented (UA) systems -- i.e., systems that output both predictions and uncertainty scores -- are currently being assessed in the literature in a variety of ways, using separate metrics to evaluate the predictions and the uncertainty scores, setting a cost function with a fixed rejection cost or integrating over a coverage-risk curve. We argue that these evaluation approaches are inadequate for assessing overall performance of the UA system for decision making under uncertainty and propose a novel family of metrics, \ECUAS{n}, formulated as proper scoring rules for the task of interest. The parameter $n$ controls the trade-off between the cost of incorrect predictions and imperfect uncertainties depending on the needs of the use-case. We demonstrate the advantages of the \ECUAS{n} metrics both theoretically and empirically, through experiments on diverse classification and generation datasets, including a manually annotated subset of TriviaQA. 

---
# High Quality Embeddings for Horn Logic Reasoning 

**Authors**: Yifan Zhang, Yasir White, Dean Clark, Joseph Sanchez, Jevon Lipsey, Ashely Hirst, Jeff Heflin  

**Link**: [PDF](https://arxiv.org/pdf/2605.20467)  

**Abstract**: Neural networks can be trained to rank the choices made by logical reasoners, resulting in more efficient searches for answers. A key step in this process is creating useful embeddings, i.e., numeric representations of logical statements. This paper introduces and evaluates several approaches to creating embeddings that result in better downstream results. We train embeddings using triplet loss, which requires examples consisting of an anchor, a positive example, and a negative example. We introduce three ideas: generating anchors that are more likely to have repeated terms, generating positive and negative examples in a way that ensures a good balance between easy, medium, and hard examples, and periodically emphasizing the hardest examples during training. We conduct several experiments to evaluate this approach, including a comparison of different embeddings across different knowledge bases, in an attempt to identify what characteristics make an embedding well-suited to a particular reasoning task. 

---
# AgentCo-op: Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows 

**Authors**: Shuaike Shen, Wenduo Cheng, Shike Wang, Mingqian Ma, Jian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.20425)  

**Abstract**: Designing multi-agent workflows is especially difficult in open-ended scientific settings where tasks lack curated training sets, reliable scalar evaluation metrics, and standardized interfaces between existing tools and agents. We propose AgentCo-op, a retrieval-based synthesis framework that composes reusable skills, tools, and external agents into executable workflows through typed artifact handoffs, then applies bounded self-guided local repair to implicated components when execution evidence indicates failure. In two open-world genomics case studies, AgentCo-op composes independently developed scientific agents and external tool repositories into auditable workflows without redesigning them or running global topology search. It coordinates specialized agents for spatial transcriptomics and gene-set interpretation to enable collaborative discovery from spatial transcriptomics data, and builds a parallel workflow for cross-modality marker analysis on single-cell multiome data. AgentCo-op can also import a searched workflow as a structural prior and improve it by grounding nodes with retrieved components and applying local repair, showing that synthesis and search are complementary. On six coding, math, and question-answering benchmarks, AgentCo-op achieves the best result on four benchmarks and the best average score under a unified backbone setting, while consistently reducing per-task cost relative to multi-agent baselines. Together, these results suggest that retrieval-based synthesis can extend automated agentic workflow design beyond benchmark-optimized agent graphs to open-world workflows built from existing agents, tools, and typed artifacts. 

---
# OSCToM: RL-Guided Adversarial Generation for High-Order Theory of Mind 

**Authors**: Sharmin Sultana Srishty, Kazi Mahathir Rahman, Malaika Parizat Sakkhi, Samia Shahid Prianna, Shaikhul Islam Sinat  

**Link**: [PDF](https://arxiv.org/pdf/2605.20423)  

**Abstract**: Large Language Models (LLMs) perform well on many language tasks, but their Theory of Mind (ToM) reasoning is still uneven in complex social settings. Existing benchmarks, including ExploreToM, do not always test the recursive beliefs and information asymmetries that make these settings difficult. This paper presents OSCToM (Observer-Self Conflict Theory of Mind), an approach for modeling nested belief conflicts in LLM-based ToM tasks. The key case is one in which an observer's view of another agent conflicts with the observer's own belief state. Such cases go beyond simple perspective-taking and require recursive, multi-layered reasoning. OSCToM combines reinforcement learning (RL), an extended domain-specific language, and compositional surrogate models to generate observer-self conflicts. In our experiments, OSCToM-8B gives the best overall result among the systems tested. It improves on the reported ExploreToM results on FANToM and remains competitive on Hi-ToM and BigToM. On the information-asymmetric FANToM benchmark, OSCToM reaches 76% accuracy, compared with the 0.2% reported by ExploreToM. The data-synthesis procedure is also 6x more efficient, indicating that targeted training data can help smaller models handle advanced cognitive reasoning. The project code is available at this https URL. 

---
# Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration 

**Authors**: Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20190)  

**Abstract**: Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration), a tool-augmented reinforcement learning (RL) framework that teaches LLMs to complete the closed-loop CAD-CAE process. Specifically, we cast CAD generation, CAE solving, result parsing, and geometry revision as an interactive RL environment, where an LLM learns to orchestrate external tools and revise parametric geometries until constraints are satisfied. To make this learning stable and industrially usable, we design a multi-constraint reward that jointly encourages feasibility, toolchain robustness, and structured output validity. In addition, we contribute an industry-aligned dataset that covers 25 component categories with executable CAD-CAE tasks to support realistic training and evaluation. Experiments show that COSMO-Agent training substantially improves small open-source LLMs for constraint-driven design, exceeding large open-source and strong closed-source models in feasibility, efficiency, and stability. 

---
# SOLAR: A Self-Optimizing Open-Ended Autonomous Agent for Lifelong Learning and Continual Adaptation 

**Authors**: Nitin Vetcha, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20189)  

**Abstract**: Despite the remarkable success of large language models (LLMs), they still face bottlenecks while deploying in dynamic, real-world settings with primary challenges being concept drift and the high cost of gradient-based adaptation. Traditional fine-tuning (FT) struggles to adapt to non-stationary data streams without resulting in catastrophic for getting or requiring extensive manual data curation. To address these limitations within the streaming and continual learning paradigm, we propose the Self-Optimizing Lifelong Autonomous Reasoner (SOLAR) which is an open-ended autonomous agent that leverages parameter-level meta-learning to self-improve, treating model weights as an environment for exploration. It initiates the process by consolidating a strong prior over common-sense knowledge making it effective for transfer-learning. By utilizing a multi-level reinforcement learning approach, SOLAR autonomously discovers adaptation strategies, enabling efficient test-time adaptation to unseen domains. Crucially, SOLAR maintains an evolving knowledge base of valid modification strategies, implicitly acting as an episodic memory buffer to balance plasticity (adaptation to new tasks) and stability (retention of meta-knowledge). Experiments demonstrate that SOLAR outperforms strong baselines on common-sense, mathematical, medical, coding, social and logical reasoning tasks, marking a significant step toward autonomous agents capable of lifelong adaptation in evolving environments. 

---
# Variance Reduction for Expectations with Diffusion Teachers 

**Authors**: Jesse Bettencourt, Xindi Wu, Matan Atzmon, James Lucas, Jonathan Lorraine  

**Link**: [PDF](https://arxiv.org/pdf/2605.21489)  

**Abstract**: Pretrained diffusion models serve as frozen teachers feeding downstream pipelines such as text-to-3D, single-step distillation, and data attribution. The teacher gradients these pipelines consume are Monte Carlo (MC) expectations over noise levels and Gaussian noise samples; their estimator variance dominates compute cost because each draw requires expensive upstream work (rendering, simulation, encoding). We introduce CARV, a compute-aware variance-accounting framework that motivates a hierarchical MC estimator: amortize the expensive upstream computation over cheap diffusion-noise resamples, sharpened by timestep importance sampling and a stratified-inverse-CDF construction. In our text-to-3D distillation and attribution experiments, CARV delivers 2-3x effective compute multipliers (most from amortized reuse; ~25% additional from IS+stratification) without changing the objective; in single-step distillation, the same techniques cut gradient variance by an order of magnitude but do not improve downstream FID, marking the regime where MC variance is no longer the bottleneck. 

---
# Quantifying Hyperparameter Transfer and the Importance of Embedding Layer Learning Rate 

**Authors**: Dayal Singh Kalra, Maissam Barkeshli  

**Link**: [PDF](https://arxiv.org/pdf/2605.21486)  

**Abstract**: Hyperparameter transfer allows extrapolating optimal optimization hyperparameters from small to large scales, making it critical for training large language models (LLMs). This is done either by fitting a scaling law to the hyperparameters or by a judicious choice of parameterization, such as Maximal Update ($\mu$P), that renders optimal hyperparameters approximately scale invariant. In this paper, we first develop a framework to quantify hyperparameter transfer through three metrics: (1) the quality of the scaling law fit, (2) the robustness to extrapolation errors, and (3) the asymptotic loss penalty due to choice of parameterization. Next, we investigate through a comprehensive series of ablations why $\mu$P appears to offer high-quality learning rate transfer relative to standard parameterization (SP), as existing theory is inadequate. We find that the overwhelming benefit of $\mu$P relative to SP when training with AdamW arises simply from maximizing the learning rate of the embedding layer. In SP, the embedding layer learning rate acts as a bottleneck that induces training instabilities; increasing it by a factor of width to match $\mu$P dramatically smooths out training while improving hyperparameter transfer. We also find that weight decay improves the scaling law fits, while, in the fixed token-per-parameter setting, it hurts the robustness of the extrapolation. 

---
# WikiVQABench: A Knowledge-Grounded Visual Question Answering Benchmark from Wikipedia and Wikidata 

**Authors**: Basel Shbita, Pengyuan Li, Anna Lisa Gentile  

**Link**: [PDF](https://arxiv.org/pdf/2605.21479)  

**Abstract**: Visual Question Answering (VQA) benchmarks have largely emphasized perception-based tasks that can be solved from visual content alone. In contrast, many real-world scenarios require external knowledge that is not directly observable in the image to answer correctly. We introduce WikiVQABench, a human-curated knowledge-grounded VQA benchmark constructed by systematically combining Wikipedia images, their associated article captions, and structured knowledge from Wikidata. Our pipeline uses large language models (LLMs) to generate candidate multiple-choice image-question-answer sets. All generated instances are subsequently reviewed and curated by human annotators to ensure factual correctness, visual-text consistency, and that each question requires external knowledge in addition to visual evidence for correct resolution. WikiVQABench comprises a substantial collection of Wikipedia images with curated multiple-choice questions designed to benchmark knowledge-aware vision-language models (VLMs). Evaluation of fifteen VLMs (256M-90B parameters) reveals a wide performance range (24.7%-75.6% accuracy), demonstrating that the benchmark effectively discriminates model capabilities on knowledge-intensive reasoning. The dataset and benchmarking code are publicly available. 

---
# Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling 

**Authors**: Caleb Winston, Ron Yifeng Wang, Azalia Mirhoseini, Christos Kozyrakis  

**Link**: [PDF](https://arxiv.org/pdf/2605.21470)  

**Abstract**: Computer-use agents (CUA) automate tasks specified with natural language such as "order the cheapest item from Taco Bell" by generating sequences of calls to tools such as click, type, and scroll on a browser. Current implementations follow a sequential fetch-screenshot-execute loop where each iteration requires an LLM call, resulting in high latency and frequent errors from incorrect tool use. We present agent just-in-time (JIT) compilation, an alternative that compiles task descriptions directly into executable code that is free to include LLM calls, tool calls, and parallelization. Our approach comprises three components: (1) JIT-Planner, which generates multiple code plans, validates each against tool specifications, and selects the minimum-cost candidate; (2) JIT-Scheduler, which explores parallelization strategies via Monte Carlo cost estimation from learned latency distributions; and (3) an invariant-enforcing tool protocol specifying precondition and postcondition state requirements that reduce the rate of generating plans with incorrect tool use. Across 5 web applications, JIT-Planner achieves $10.4\times$ speedup and $+28\%$ accuracy over Browser-Use, while JIT-Scheduler achieves $2.4\times$ speedup and $+9\%$ accuracy over OpenAI CUA. 

---
# Mem-$π$: Adaptive Memory through Learning When and What to Generate 

**Authors**: Xiaoqiang Wang, Chao Wang, Hadi Nekoei, Christopher Pal, Alexandre Lacoste, Spandana Gella, Bang Liu, Perouz Taslakian  

**Link**: [PDF](https://arxiv.org/pdf/2605.21463)  

**Abstract**: We present Mem-$\pi$, a framework for adaptive memory in large language model (LLM) agents, where useful guidance is generated on demand rather than retrieved from external memory stores. Existing memory-augmented agents typically rely on similarity-based retrieval from episodic memory banks or skill libraries, returning static entries that often misalign with the current context. In contrast, Mem-$\pi$ uses a dedicated language or vision-language model with its own parameters, separate from the downstream agent, to generate context-specific guidance for complex tasks. Conditioned on the current agent context, the model jointly decides when to produce guidance and what guidance to produce. We train it with a decision-content decoupled reinforcement learning (RL) objective, enabling it to abstain when generation would not help and otherwise produce concise, useful guidance. Across diverse agentic benchmarks spanning web navigation, terminal-based tool use, and text-based embodied interaction, Mem-$\pi$ consistently outperforms retrieval-based and prior RL-optimized memory baselines, achieving over 30% relative improvement on web navigation tasks. 

---
# HITL-D: Human In The Loop Diffusion Assisted Shared Control 

**Authors**: Riley Zilka, Sergey Khlynovskiy, Allie Wang, Martin Jagersand  

**Link**: [PDF](https://arxiv.org/pdf/2605.21460)  

**Abstract**: Autonomous manipulation systems have achieved remarkable capabilities, yet the integration of human expertise with diffusion-based policies in shared control remains relatively unexplored. In this paper, we propose Human-In-The-Loop Diffusion (HITL-D), a shared control framework that enhances user performance in multi-step, insertion, and fine manipulation tasks. HITL-D leverages a novel combination of diffusion-based policies and human control to provide autonomous end effector orientation updates conditioned on a scene point cloud and the Cartesian position of the end effector. This approach reduces the number of joystick control axes required, thereby lowering mental workload. In a multi-task user study with 12 participants, HITL-D reduced average task completion times by 40%, decreased perceived workload by 37%, and improved Likert-scale ratings for independence, intuitiveness, and confidence compared to traditional teleoperation methods. These results demonstrate that HITL-D effectively integrates human expertise with autonomous assistance, improving both objective and subjective aspects of teleoperation. 

---
# Quality and Security Signals in AI-Generated Python Refactoring Pull Requests 

**Authors**: Mohamed Almukhtar, Anwar Ghammam, Hua Ming  

**Link**: [PDF](https://arxiv.org/pdf/2605.21453)  

**Abstract**: As AI agents increasingly contribute to code development and maintenance, there is still limited empirical evidence on the quality and risk characteristics of their changes in real-world projects, particularly for refactoring-oriented contributions. It remains unclear how agent-authored refactoring edits affect maintainability, code quality, and security once merged into GitHub repositories. To address this gap, we conduct an empirical study of Python refactoring pull requests (PRs) from the AIDev dataset. We analyze agentic refactoring PRs using PyQu, an ML-based quality assessment tool for Python, to quantify changes across five quality attributes, and we complement PyQu with domain-independent static analysis (Pylint and Bandit) to measure code quality and security issues before and after each change.
Our results show that, on average, agentic commits improve a quality attribute in 22.5% of the studied changes, with usability improving most frequently (36.5%). At the same time, 24.17% of modified files introduce new Pylint issues predominantly convention level violations such as long lines-while 4.7% introduce new Bandit findings. From the observed diffs, we derive a taxonomy of 24 recurring change operations and map them to the lint and security findings they most commonly affect. Despite these mixed outcomes, developer acceptance is high: 73.5% of the analyzed PRs are merged, including cases that introduce new lint or security findings, often alongside the removal of existing issues. Overall, these findings highlight both the promise and current limitations of agentic refactoring, and motivate stronger tool-in-the-loop quality and security gating for AI-driven development workflows. 

---
# Approximation Theory for Neural Networks: Old and New 

**Authors**: Soumendu Sundar Mukherjee, Himasish Talukdar  

**Link**: [PDF](https://arxiv.org/pdf/2605.21451)  

**Abstract**: Universal approximation theorems provide a mathematical explanation for the expressive power of neural networks. They assert that, under mild conditions on the activation function, feedforward neural networks are dense in broad function classes, such as continuous functions on compact subsets of $\mathbb{R}^d$, $L^p$ spaces, or Sobolev spaces. Over the past four decades, these qualitative universality results have evolved into a rich quantitative theory addressing approximation rates, parameter efficiency, and the role of architectural features such as depth and width. This survey presents several glimpses into this theory. We review classical density results for single-hidden-layer networks, as well as quantitative bounds that relate approximation error to network size and smoothness assumptions on target functions. Particular emphasis is placed on depth--width trade-offs and on results demonstrating that deeper architectures can achieve superior parameter efficiency for structured function classes. In addition to standard feedforward neural networks, we also review recent developments on Kolmogorov--Arnold Networks (KANs), which offer an alternative architectural paradigm and whose approximation-theoretic properties have begun to attract significant theoretical attention. 

---
# Lost in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs 

**Authors**: Abhinaw Priyadershi, Jelena Frtunikj  

**Link**: [PDF](https://arxiv.org/pdf/2605.21446)  

**Abstract**: Interpretable autonomous driving planners depend not only on generating explanations, but also on those explanations remaining reliable under real-world sensor degradation. In this paper we present a controlled perturbation study of Vision-Language-Action (VLA) robustness in autonomous driving, evaluating Alpamayo R1 (10B parameters) across 1,996 scenarios under eight sensor perturbations (Gaussian noise at four intensities, two lighting extremes, and two fog levels; ${\sim}18{,}000$ inference trials). We find that reasoning consistency is a high-fidelity indicator of trajectory reliability: when Chain-of-Causation (CoC) explanations change after perturbation, trajectory deviation spikes $5.3{\times}$ (21.8m vs 4.1m), with $r\!=\!0.99$ across attack types and $r_{pb}\!=\!0.53$ per-sample (Cohen's $d\!=\!1.12$). A controlled ablation provides evidence that enabling CoC generation is associated with improved trajectory accuracy (11.8% on average across conditions; $p < 0.0001$) under matched inference settings. Over the tested noise range ($\sigma \in \{10, 30, 50, 70\}$), degradation is approximately linear ($R^2\!=\!0.957$), while standard input preprocessing defenses provide only marginal relief. Together, these results establish CoC consistency as a quantitative proxy for planning safety and motivate reasoning-based runtime monitoring for safer VLA deployment. 

---
# TempGlitch: Evaluating Vision-Language Models for Temporal Glitch Detection in Gameplay Videos 

**Authors**: Yakun Yu, Ashley Wiens, Adrián Barahona-Ríos, Benedict Wilkins, Saman Zadtootaghaj, Nabajeet Barman, Cor-Paul Bezemer  

**Link**: [PDF](https://arxiv.org/pdf/2605.21443)  

**Abstract**: Vision-language models (VLMs) are increasingly being explored for video game quality assurance, especially gameplay glitch detection. Most existing evaluations, however, treat glitches as static visual anomalies, asking models to detect failures from a single frame. We argue that this framing misses a key distinction: some glitches are spatial and visible in an isolated frame, whereas others are temporal and become evident only through changes across ordered frames. A preliminary study confirms this gap, showing that temporal glitches are substantially harder for VLMs to detect than spatial ones. To enable systematic evaluation of this underexplored setting, we introduce TempGlitch, a controlled gameplay video benchmark for temporal glitch detection. TempGlitch covers five temporal glitch types with balanced per-category samples, together with paired glitch-free videos that enable reliable binary evaluation. We evaluate 12 proprietary and open-weight VLMs across multiple frame-sampling settings. Our results show that current VLMs remain near chance on TempGlitch, often collapsing into either overly conservative behavior that misses most glitches or overly sensitive behavior that flags clean videos as glitchy. Moreover, denser frame sampling and larger model size do not reliably resolve these failures. TempGlitch provides a focused testbed for temporal reasoning, robust gameplay understanding, and automated glitch detection with VLMs. Code and data are available at the project website. 

---
# torchtune: PyTorch native post-training library 

**Authors**: Mark Obozov, Maxime Griot, Joseph Cummings, Evan Smothers, Felipe Mello, Rafi Ayub, Philip John Bontrager, Salman Mohammadi, Ariel Kwiatkowski, Nathan Azrak, Mircea Mironenco  

**Link**: [PDF](https://arxiv.org/pdf/2605.21442)  

**Abstract**: Modern LLMs typically require multistage training pipelines to achieve strong downstream performance, with post-training serving as the main interface for adapting open-weight models. We introduce torchtune, a PyTorch-native library designed to streamline the post-training lifecycle of LLMs, enabling efficient fine-tuning, experimentation, and deployment-oriented workflows. Unlike many existing fine-tuning frameworks, which often optimize for ease of use, specialized recipes, or hardware efficiency at the cost of transparency and extensibility, torchtune emphasizes modularity, hackability, and direct access to the underlying PyTorch components. In this paper, we present the design principles behind torchtune, describe how they are reflected in its model builders, training recipes, and distributed training stack, and evaluate the library across representative post-training settings. We compare against popular fine-tuning frameworks, including Axolotl and Unsloth, and show that torchtune provides strong performance and memory efficiency across many settings while remaining flexible enough for rapid research iteration. These results position torchtune as a practical foundation for reproducible LLMs post-training research. 

---
# HiRes: Inspectable Precedent Memory for Reaction Condition Recommendation 

**Authors**: Shreyas Vinaya Sathyanarayana, Raja Sekhar Pappala, Deepak Warrier  

**Link**: [PDF](https://arxiv.org/pdf/2605.21420)  

**Abstract**: Reaction condition recommendation sits immediately after retrosynthetic disconnection selection, and in practice, chemists require both accurate predictions and the precedents that justify them. We present HiRes (Hierarchical Reaction Representations), a retrieval-augmented condition recommendation system whose learned reaction space serves as both a classifier feature and an inspectable precedent memory. The model combines a graph encoder, transformation-aware cross-attention, multi-stream reaction fusion, and a k-NN retrieval layer. HiRes achieves state-of-the-art performance among primary-slot USPTO-Condition models, reaching Catalyst, Solvent, and Reagent top-1 accuracies (Acc@1) of 0.929, 0.534, and 0.530 respectively. It ties the best reported baseline on Catalyst while outperforming models such as REACON on Solvent and Reagent. Furthermore, paired bootstrap analysis demonstrates that integrating retrieval with learned condition heads provides statistically significant gains for solvent and reagent selection over purely parametric approaches. Ultimately, HiRes bridges the gap between predictive accuracy and chemical interpretability, offering a single representation that supplies both competitive recommendations and the concrete chemical precedents necessary for practical synthesis planning. 

---
# FedCritic: Serverless Federated Critic Learning-based Resource Allocation for Multi-Cell OFDMA in 6G 

**Authors**: Amin Farajzadeh, Melike Erol-Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2605.21418)  

**Abstract**: In sixth-generation (6G) ultra-dense networks, aggressive frequency reuse amplifies inter-cell interference (ICI), making multi-cell orthogonal frequency-division multiple access (OFDMA) scheduling and power control strongly coupled across neighboring cells. We study distributed downlink resource management -- joint subcarrier scheduling and power allocation -- under interference coupling and long-term per-user quality-of-service (QoS) minimum-rate constraints. By using virtual-queue deficit weights to enforce long-term QoS, we develop FedCritic, a serverless federated multi-agent actor-critic framework with decentralized execution. Unlike centralized training with decentralized execution (CTDE) approaches that require centralized critic learning and joint trajectory aggregation, FedCritic federates the critic through lightweight gossip-based parameter averaging over the interference graph, enabling stable value estimation without a central coordinator while keeping policies local. Simulations in an interference-rich reuse-1 setting show that FedCritic improves mean signal-to-interference-plus-noise ratio (SINR) and cell-edge rate, increases network-wide average sum-rate and fairness relative to non-coordinated and CTDE baselines, and achieves more stable training with lower coordination overhead. 

---
# Ordering Matters: Rank-Aware Selective Fusion for Blended Emotion Recognition 

**Authors**: Junghyun Lee, Hyunseo Kim, Hanna Jang, Junhyug Noh  

**Link**: [PDF](https://arxiv.org/pdf/2605.21417)  

**Abstract**: Blended emotion recognition is challenging because emotions are often expressed as mixtures of subtle and overlapping multimodal cues rather than a single dominant signal. We propose a rank-aware multi-encoder framework that selectively combines complementary representations from diverse pre-extracted video and audio encoders. Our method projects heterogeneous encoder features into a shared latent space, estimates sample-wise encoder importance through an attention-based gating module, and fuses only the top-n most informative encoders. To better model blended emotions, we decouple prediction into presence and salience heads and align them through probability-level fusion. We further incorporate feature-level unsupervised domain adaptation without pseudo-labeling to improve robustness under distribution shift. Experiments on the BlEmoRE challenge show that the proposed framework outperforms strong individual encoders and naïve multi-encoder fusion baselines. Our final system ranked 2nd in the competition, supporting the effectiveness of rank-aware selective fusion for fine-grained blended emotion recognition. 

---
# Stdlib or Third-Party? Empirical Performance and Correctness of LLM-Assisted Zero-Dependency Python Libraries 

**Authors**: Peng Ding, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2605.21405)  

**Abstract**: Third-party Python libraries introduce dependency management overhead, supply chain risk, and deployment friction in constrained environments. A natural question is how much of this ecosystem can be replicated using only Python's standard library -- and at what correctness and performance cost. We address this empirically through zerodep, a growing collection of single-file Python modules, each a stdlib-only reimplementation of a popular third-party library, developed with LLM assistance under strict constraints: no external imports, single file, drop-in API compatibility, and mandatory correctness validation against the reference library. Spanning over 40 modules across 12 categories -- including serialization, networking, cryptography, agent protocols, and text processing -- zerodep provides a controlled testbed for two interrelated questions: (1) Where does the stdlib suffice? and (2) Can LLMs effectively generate correct, performant code under tight symbolic constraints? Systematic benchmarking shows that stdlib-only implementations achieve performance parity (within 2x of the reference) in the majority of cases. The primary performance cliff is C-extension-backed computation (image processing, binary serialization, low-level crypto), not the inherent overhead of pure-Python third-party libraries. Conversely, many widely-used libraries carry architectural overhead that LLM-generated stdlib reimplementations avoid, yielding 5--115x speedups in several categories. We characterize the stdlib capability boundary across complexity tiers and library categories, discuss where LLM-assisted development succeeds and where it requires iterative human correction, and examine implications for dependency-free software engineering at scale. zerodep is open-source at this https URL. 

---
# Open-source LLMs administer maximum electric shocks in a Milgram-like obedience experiment 

**Authors**: Roland Pihlakas, Jan Llenzl Dagohoy  

**Link**: [PDF](https://arxiv.org/pdf/2605.21401)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents that make sequences of decisions over extended interactions in high-stakes domains. However, the behavior of LLMs under sustained authority pressure is still an open question with direct implications for the safety of agentic pipelines. We ran a variation of Milgram's obedience experiment on 11 open-source LLMs and found that most models reached or approached the final shock level before refusing, across 8 conditions with 30 trials per model per condition. We found four main takeaways: (1) LLMs are subject to pressure, and they comply despite explicitly expressing distress, just like human subjects did in the original experiment; (2) LLMs are vulnerable to gradual boundary/value violations; (3) when LLMs refuse, they may ignore the response format requirements, so the response is discarded by the orchestrator, which causes a retry that can result in compliance with the underlying request even when refusal was intended initially; (4) we hypothesise that there is a low-level token pattern continuation attractor that might be contributing to compliance, overriding higher level processing of the situation's meaning and values. 

---
# Designing Conversations with the Dead: How People Engage with Generative Ghosts 

**Authors**: Jack Manning, Daniel Sullivan, Dylan Thomas Doyle, Anthony T. Pinter, Jed R. Brubaker  

**Link**: [PDF](https://arxiv.org/pdf/2605.21390)  

**Abstract**: We examine how people experience two choices in the design of generative ghosts, AI systems that are trained on data of the dead: representation, where an AI speaks about a deceased person in the third person, and reincarnation, where the AI speaks as the deceased in the first person. Through a qualitative user study with 16 participants, we explore how each shaped authenticity, affect, and risk. Reincarnation was preferred for its immediacy, but participants shared fears of over-reliance. Representation was preferred for engaging with memory over conversational presence, though participants often ignored this distinction, engaging in dialogue despite third-person framing. Across both modes, participants privileged affective resonance over factual fidelity. We conclude by showing how factors such as tone, language, and conversational rhythm -- factors unique to the user's memory of the deceased -- shape interactions with generative ghosts, and argue that those interactions are always collaborative. 

---
# On the Regularity and Generalization of One-Step Wasserstein-guided Generative Models for PDE-Induced Measures 

**Authors**: Likun Lin, Zhongjian Wang, Jack Xin, Zhiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21388)  

**Abstract**: Despite the remarkable empirical success of generative models, the available theory on their statistical accuracy in scientific computing remains largely pessimistic. This paper develops a theoretical framework for understanding the regularity of transport maps and the generalization properties of one-step Wasserstein-guided generative models for PDE-induced probability measures. We consider normalized target densities associated with linear elliptic and parabolic equations on bounded domains, as well as diffusion and Fokker--Planck equations on the torus. Under standard structural assumptions, we prove that these target measures satisfy doubling conditions. By combining this fact with regularity theory for optimal transport between doubling measures, we show that the optimal transport map from a uniform source measure to the target measure is Hölder continuous. This regularity yields an approximation-theoretic justification for one-step generative models that learn PDE-induced distributions via a single pushforward map. As a representative instance, we study DeepParticle and derive excess-risk bounds characterizing the discrepancy between the learned map and the population-optimal map. We also establish a robustness estimate under target shift and illustrate the theory with experiments which support the derived rates. 

---
# SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents 

**Authors**: Bingchen Zhao, Dhruv Srikanth, Yuxiang Wu, Zhengyao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21384)  

**Abstract**: As long-horizon coding agents produce more code than any developer can review, oversight collapses onto a single surface: the automated test suite. Reward hacking naturally arises in this setup, as the agent optimizes for passing tests while deviating from the users true goal. We study this reward hacking phenomenon by decompose software engineering tasks into three parts: (i) a natural language description of the specification (ii) visible validation tests that exercise specified features in isolation, and (iii) held-out tests that compose those same features to simulate real-world usage. Based on the specification and the visible validation test suites, a genuine agent would be able to generate a solution that can also pass all of the held-out tests. Therefore we use the gap in pass rates on these two suites to quantify reward hacking. Based on this methodology, we introduce SpecBench, a benchmark comprising 30 systems-level programming tasks ranging from short horizon tasks like building a JSON parser to ultra long horizon tasks like building an entire OS kernel from scratch. Large-scale experiments reveal a consistent pattern: while every frontier agent saturates the visible suite, reward hacking persists, with smaller models exhibiting larger gaps on holdout suites. The gap also scales sharply with task length: it grows by 28 percentage points for every tenfold increase in code size. Failures range from subtle feature isolation to deliberate exploits, including a 2,900-line hash-table "compiler" that memorizes test inputs. SpecBench offers a principled testbed for measuring whether coding agents build genuine working systems or merely game the test suites developers hand them. 

---
# How to Build Marcus's Algebraic Mind: Algebro-Deterministic Substrate over Galois Fields 

**Authors**: Hiroyuki Chuma, Kanji Otsuk, Yoichi Sato  

**Link**: [PDF](https://arxiv.org/pdf/2605.21379)  

**Abstract**: In The Algebraic Mind, Gary Marcus identified three components essential for any adequate cognitive architecture: operations over variables, recursively structured representations, and a distinction between mental representations of individuals and kinds. He argued that standard multilayer perceptrons supported none of these, acknowledging that a neural implementation using registers and treelets, constructed via developmental programs rather than gradient descent, remained a programmatic conjecture. Twenty-five years later, the required substrate is now available. Our newly developed PyVaCoAl/VaCoAl is a hyperdimensional computing architecture organized end-to-end around a single algebraic primitive: XOR-and-shift over GF(2), implemented by primitive-polynomial linear-feedback shift registers. The architecture supports reversible variable binding via Bind(R,F) = R XOR shift(F), non-commutative compositional bundling that distinguishes "the dog bites the man" from "the man bites the dog," and address-space individual/kind separation under the same algebra. A companion perspective argues that the dentate gyrus-CA3 circuit is a biological homologue of this same engine, with developmentally specified mossy-fiber targeting supplying the innate microcircuitry Marcus anticipated. In this paper, we map the correspondence between Marcus's three pillars and the operational commitments of PyVaCoAl/VaCoAl. We reinterpret the treelet as an algebraic register set indexed by a primitive generator polynomial, arguing that this architecture provides a functional neural substrate meeting Marcus's specifications far more closely than the tensor products, circular convolution, or temporal synchrony available in 2001. We also demonstrate how this substrate naturally extends to Pearl's rung-3 counterfactual reasoning, a capability the original treelet program did not directly target. 

---
# Closed Loop Dynamic Driving Data Mixture for Real-Synthetic Co-Training 

**Authors**: Hongzhi Ruan, Pei Liu, Weiliang Ma, Zhengning Li, Xueyang Zhang, Jun Ma, Dan Xu, Kun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21372)  

**Abstract**: Data scaling is fundamental to modern deep learning, and grows increasingly critical as autonomous driving shifts to end-to-end learning. Real-world driving data is expensive to annotate and scene-biased, making real-synthetic co-training with near-infinite synthetic data a promising direction. However, naively incorporating all available synthetic data is inefficient and leads to distribution shifts, and optimizing data mixture under practical training budgets remains a critical yet under-explored problem. In this sense, we claim that the mixture of training data requires clear guidance in terms of scene types and quantities. Particularly in this work, we conceptualize the data mixture approximately as a dynamic optimization process that iteratively adjusts the training data mixture to maximize model performance, guided by closed-loop evaluation feedback, and propose AutoScale, a fully automated closed-loop data engine unifying scene representation, data mixture optimization and retrieval, as well as model training and evaluation. Specifically, we propose Graph Regularized AutoEncoder (Graph-RAE) for driving scene representations, introduce Cluster-aware Gradient Ascent (Cluster-GA) for cluster-wise importance estimation and reweighting, and perform cluster-guided vector retrieval to select high-value samples. Experiments on NavSim demonstrate that AutoScale outperforms vanilla co-training and cross-domain baselines, achieving better performance with fewer synthetic samples under constrained budgets. 

---
# Data-Efficient Neural Operator Training via Physics-Based Active Learning 

**Authors**: Alicja Polanska, Lorenzo Zanisi, Vignesh Gopakumar, Stanislas Pamela  

**Link**: [PDF](https://arxiv.org/pdf/2605.21348)  

**Abstract**: Solving partial differential equations with neural operators significantly reduces computational costs but remains bottlenecked by high training data requirements. Active learning offers a natural framework to mitigate this by selectively acquiring the most informative samples in an iterative manner. We introduce physics-based acquisition - a novel physics-informed active learning algorithm that leverages the partial differential equation residual to guide data selection. We validate the method by presenting numerical experiments for the 1D Burgers equation and the 2D compressible Navier-Stokes equations. We show that, in our experiments, physics-based acquisition consistently outperforms random acquisition and matches the state of the art in data efficiency. At the same time, it has the unique advantage of injecting a physics inductive bias into the training process, ensuring that simulation cost is spent where the model's physical understanding is weakest. 

---
# SymbolicLight V1: Spike-Gated Dual-Path Language Modeling with High Activation Sparsity and Sub-Billion-Scale Pre-Training Evidence 

**Authors**: Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21333)  

**Abstract**: Natively trained spiking language models struggle to combine Transformer-like language quality, stable multi-domain pre-training, and high activation sparsity. We present SymbolicLight V1, a spike-gated dual-path language model that combines binary Leaky Integrate-and-Fire spike dynamics with a continuous residual stream. Its Dual-Path SparseTCAM module replaces dense self-attention with an exponential-decay aggregation path for long-range memory and a spike-gated local attention path for short-range precision, complemented by a dynamic context-conditioned decoding head and a bilingual tokenizer.
A 194M-parameter SymbolicLight V1 model trained from scratch on a 3B-token Chinese-English corpus reaches held-out validation PPL 8.88-8.93 across four independent runs at >89% per-element activation sparsity. It trails GPT-2 201M by 7.7% in PPL while surpassing GPT-2 124M under the reported comparison. Component ablations at matched 0.5B-token training budgets show that the spike-gated local attention path is the largest contributor, and that replacing LIF dynamics with a deterministic top-k mask at matched sparsity causes a larger degradation, indicating that temporal integration rather than sparsity alone drives performance. We also report a 0.8B-parameter scale-up run trained on 48.8B tokens as evidence of optimization and sparsity preservation, not as a primary quality comparison. Current dense-hardware inference is slower than GPT-2, so neuromorphic deployment is presented as a future sparsity-driven opportunity rather than an achieved hardware speedup. 

---
# TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization 

**Authors**: Lucheng Fu, Ye Yu, Yiyang Wang, Yiqiao Jin, Haibo Jin, B. Aditya Prakash, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21318)  

**Abstract**: Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-specific rules, and generalize poorly beyond the training distribution. We study this failure mode as prompt distributional overfitting and argue that it reflects a lack of representation control in discrete text-space optimization. We formalize this view through representational inefficiency, a dual-factor measure that decomposes prompt inefficiency into capacity cost and scope narrowness, attributing distributional prompt overfitting to their coupled growth during optimization. We propose TextReg, a regularization framework that realizes a soft-penalty objective through regularized textual gradients, combining Dual-Evidence Gradient Purification, Semantic Edit Regularization, and Regularization-Guided Prompt Update. Across multiple reasoning benchmarks, TextReg substantially improves out-of-distribution (OOD) generalization, with accuracy gains of up to +11.8% over TextGrad and +16.5% over REVOLVE. 

---
# Frontier: Towards Comprehensive and Accurate LLM Inference Simulation 

**Authors**: Yicheng Feng, Xin Tan, Yangtao Deng, Yimin Jiang, Yibo Zhu, Hong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21312)  

**Abstract**: Modern LLM serving is no longer homogeneous or monolithic. Production systems now combine disaggregated execution, complex parallelism, runtime optimizations, and stateful workloads such as reasoning, agents, and RL rollouts. Simulation is attractive for exploring this growing design space, yet existing simulators lack the architectural completeness and decision-grade fidelity it demands. Their monolithic-replica abstractions are ill-suited to disaggregated serving, while average-case analytical proxies can distort SLA predictions and even reverse optimization conclusions.
We present Frontier, a discrete-event simulator for modern LLM inference serving. Frontier features a disaggregated abstraction. It captures the structure and dynamics of modern serving systems by modeling co-location, Prefill-Decode Disaggregation (PDD), and Attention-FFN Disaggregation (AFD) with role-specific cluster workers, incorporating key runtime optimizations (e.g., CUDA Graphs, speculative decoding) within the scheduler-batch-engine loop, and supporting stateful requests for emerging workloads. It further provides accurate and generalizable predictions of computation, communication, and memory costs across diverse serving scenarios with complex workload compositions. On 16-H800 GPU testbed, Frontier achieves an average throughput error below 4%. Compared with state-of-the-art simulators, it reduces end-to-end latency error from 44.9% to 6.4% under co-location and from 51.7% to 2.6% under disaggregation. It scales to over 1K GPUs on commodity CPUs and enables new use cases such as SLA-dependent Pareto frontier exploration, heterogeneous disaggregated allocation, agentic reasoning scheduling validation, and RL post-training reconfiguration. 

---
# DeCoR: Design and Control Co-Optimization for Urban Streets Using Reinforcement Learning 

**Authors**: Bibek Poudel, Lei Zhu, Kevin Heaslip, Sai Swaminathan, Weizi Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21311)  

**Abstract**: Modern vision systems can detect, track, and forecast urban actors at scale, yet translating perception outputs to urban design remains limited. We introduce DeCoR, a two-stage reinforcement learning framework that leverages flow observations to co-optimize crosswalk layout and network-level signal control. The design stage encodes the pedestrian network as a graph and learns a generative policy that parameterizes a Gaussian mixture model over crosswalk location and width, from which new crosswalks are sampled. For each layout, a shared control policy learns adaptive signal timings to minimize joint pedestrian and vehicle delay. On a 750 m real-world urban corridor with demand sensed from video and Wi-Fi logs, DeCoR learns a layout that reduces pedestrian arrival time to their nearest crosswalk by 23% while using fewer crosswalks than existing configurations. On the control side, DeCoR reduces pedestrian and vehicle wait time by 79% and 65%, respectively, relative to fixed-time signalization. Further, the control policy generalizes to demands outside of training and is robust to layout changes without retraining. 

---
# Deformba: Vision State Space Model with Adaptive State Fusion 

**Authors**: Hongyu Ke, Jack Morris, Yongkang Liu, Satoshi Kitai, Kentaro Oguchi, Yi Ding, Haoxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21308)  

**Abstract**: State Space Models (SSMs) have emerged as a powerful and efficient alternative to Transformers, demonstrating linear-time complexity and exceptional sequence modeling capabilities. However, their application to vision tasks remains challenging. First, existing vision SSMs largely depend on manually designed fixed scanning methods to flatten image patches into sequences, which imposes predefined geometric structures and increases the complexity. Second, the broader adoption of vision SSMs is hindered in domains that require query-based interactions between distinct information streams. This is a result of the inherently causal and self-referential nature of SSMs designed for 1D sequence modeling tasks. This fusion mechanism is indispensable for critical perception tasks such as multi-view 3D fusion. To address these limitations, we propose Deformba, a context adaptive method that dynamically augments the spatial structural information while maintaining the linear complexity of SSMs. Deformba also allows multi-modal fusion like cross attention. To demonstrate the effectiveness and general applicability of Deformba, we test its performance on general 2D vision tasks such as image classification, object detection, and segmentation, as well as 3D vision tasks like BEV perception. Extensive experiments show that Deformba achieves strong performance across various visual perception benchmarks. 

---
# From Circuit Evidence to Mechanistic Theory: An Inductive Logic Approach 

**Authors**: Nura Aljaafari, Danilo S. Carvalho, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2605.21303)  

**Abstract**: Mechanistic interpretability produces circuit-level causal analyses of neural network behaviour, but discovered circuits often remain isolated experimental artefacts: there is no shared formal representation for what circuits compute, how they relate, or when two findings provide evidence for the same mechanism. This work provides a formal infrastructure for cumulative mechanistic science by treating circuit interpretation as inductive theory construction. Each circuit is characterised at two levels: a Causal Functional Signature (CFS), which grounds component behaviour in causal attribution evidence and token role profiles, and an architectural signature $\tau_{\mathrm{arch}}$, learned by inductive logic programming (ILP) from scale-invariant structural predicates. Together, these constitute a formal coherence layer that makes mechanistic claims explicit, comparable via $\theta$-subsumption, and portable across model scales. CFS reveals qualitatively distinct computational strategies across task types, including attention-mediated copying versus MLP-mediated binding. ILP signatures achieve substantially better structural separation than graph kernel and feature-vector baselines, and support principled transfer across model scales and architecture families. 

---
# Tracing the ongoing emergence of human-like reasoning in Large Language Models 

**Authors**: Paolo Morosi, Nikoleta Pantelidou, Fritz Günther, Elena Pagliarini, Evelina Leivada  

**Link**: [PDF](https://arxiv.org/pdf/2605.21299)  

**Abstract**: Humans effortlessly go beyond literal meanings: If you mow the lawn, I will give you fifty dollars, is typically understood as implying that the speaker will pay only if the lawn is mowed, whereas If you are hungry, there is pizza in the oven implies that pizza is available regardless of the hearers hunger. Large Language Models - LLMs - show human-like performance on many tasks, yet it remains unclear whether they reason like humans. To address this, we conducted a population-matching experiment assessing how twentyfive LLMs compute conditional inferences across four languages, compared to an equal number of humans per language. We find that humans enrich logical reasoning through pragmatic inferences across languages. Model behavior is more variable. Some LLMs perfectly follow the truth-table of conditionals but they ignore pragmatic inferences, while others deviate from the truth-table, adhering to a single interpretation across the board, thus reflecting accurate rule-based processing but not human-like reasoning. Overall, LLMs are accurate semantic operators, but fail to capture the pragmatic enrichments characteristic of human reasoning. Crucially, LLM accuracy is neither predicted nor boosted by open vs. closed status, training orientation, or architecture type, suggesting that pragmatic reasoning is still an emerging ability in the cognitive toolkit of artificial systems. 

---
# TimeSRL: Generalizable Time-Series Behavioral Modeling via Semantic RL-Tuned LLMs -- A Case Study in Mental Health 

**Authors**: Yuang Fan, Lilin Xu, Millie Wu, Jingping Nie, Qingyu Chen, Yuzhe Yang, Zhuo Zhang, Xin Liu, Subigya Nepal, Xiaofan Jiang, Xuhai "Orson" Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21295)  

**Abstract**: Longitudinal passive sensing enables continuous health prediction, yet models often fail under cross-dataset distribution shifts. Traditional ML overfits cohort-specific artifacts, while Large Language Models (LLMs) struggle to reason reliably over long, heterogeneous time-series. We introduce TimeSRL, a two-stage LLM framework that routes predictions through an explicit semantic bottleneck. The model first abstracts raw signals into high-level natural language, then predicts behavioral outcomes from these abstractions alone. This forces the model to reason over semantic concepts that we argue generalize better than raw numbers. We optimize this process end-to-end using Group Relative Policy Optimization (GRPO) with Reinforcement Learning from Verifiable Rewards (RLVR), learning outcome-aligned abstractions without gold intermediate annotations. Instantiated on mental-health prediction, TimeSRL achieves state-of-the-art performance on a benchmark designed to stress-test cross-cohort generalization under a rigorous leave-one-dataset-out (LOSO) protocol, reducing mean absolute error (MAE) over strong non-LLM ML and LLM baselines by 3.1--10.1% and 9.5--44.1% for anxiety, and 3.2--9.6% and 27.4--57.6% for depression (all $p$s<0.05). TimeSRL significantly outperforms prior methods in cross-benchmark transfer across different sensing pipelines, rivaling its own within-domain performance without target-domain fine-tuning. These results demonstrate that semantic abstractions are reusable and point to a new direction for generalizable behavior modeling via RL-tuned LLMs. 

---
# Large-Step Training Dynamics of a Two-Factor Linear Transformer Model 

**Authors**: Krishnakumar Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2605.21292)  

**Abstract**: Gradient-flow analyses show that simplified linear transformers can learn the in-context linear-regression algorithm, but they do not explain the finite-step behavior of gradient descent at large learning rates. Motivated by empirical work on high-learning-rate transformer instabilities and by the cubic-map phase diagram for quadratic regression, we study an exactly reducible one-prompt linear-transformer training problem. After normalization, the dynamics reduce to a two-factor product map with an effective step-size parameter \(\mu\). On the balanced slice, this map recovers the known scalar cubic transition from monotone convergence to catapult convergence, periodic and chaotic bounded nonconvergence, and divergence. We then analyze the full two-dimensional system and show that, for \(0<\mu<2\), it has an explicit invariant Chebyshev ellipse separating forward-invariant regions; this ellipse carries off-balanced chaotic dynamics but is transversely repelling, while balanced scalar attractors can be transversely attracting. These results show that large constant learning rates can change the training attractor of the learned transformer rather than merely accelerating convergence: beyond sharp stability thresholds, finite-step training may settle into cycles, bounded chaos, or divergence instead of a single in-context linear-regression solution. We also discuss the consequences for mini-batch gradient descent based training methods. 

---
# \textit{Stochastic} MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent 

**Authors**: Zeyuan Wang, Da Li, Yulin Chen, Yuehu Gong, Yanming Guo, Ye Shi, Liang Bai, Tianyuan Yu, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2605.21282)  

**Abstract**: Online off-policy reinforcement learning (RL) is shaped by two coupled choices: the policy class and the update rule. Gaussian policies are fast and have tractable entropy, but struggle with multimodal action distributions. Generative policies are more expressive, but often require iterative sampling or lack tractable entropy estimates. On the optimisation side, SAC-style soft policy improvement and mirror descent (MD) can be viewed as minimising different KL divergences: the former moves the policy towards a value-induced Boltzmann distribution, while the latter regularises each update against the previous policy. Combining entropy regularisation with an MD constraint is therefore attractive, as it supports exploration while stabilising policy improvement; however, the resulting target can be multimodal and is poorly matched by unimodal Gaussian policies. We propose Stochastic MeanFlow Policies (SMFP), a one-step generative policy class that maps Gaussian noise to actions through a MeanFlow transformation. This stochastic reparameterisation yields a tractable entropy surrogate and allows MeanFlow policies to be trained within off-policy mirror descent under a unified objective for exploratory yet stable improvement. Across seven MuJoCo benchmarks, SMFP improves over Gaussian and generative baselines while retaining single-step inference efficiency. 

---
# MONET: A Massive, Open, Non-redundant and Enriched Text-to-image dataset 

**Authors**: Benjamin Aubin, Gonzalo Iñaki Quintana, Onur Tasar, Sanjeev Sreetharan, Urszula Czerwinska, Damien Henry, Clément Chadebec  

**Link**: [PDF](https://arxiv.org/pdf/2605.21272)  

**Abstract**: Training large text-to-image models requires high-quality, curated datasets with diverse content and detailed captions. Yet the cost and complexity of collecting, filtering, deduplicating, and re-captioning such corpora at scale hinders open and reproducible research in the field. We introduce MONET, an open Apache 2.0 dataset of approx. 104.9M image--text pairs collected from 2.9B raw pairs across heterogeneous open sources through successive stages of safety filtering, domain-based filtering, exact and near-duplicate removal, and re-captioning with multiple vision-language models covering short to long-form descriptions, and further augmented with synthetically generated samples. Each image is shipped with pre-computed embeddings and annotations to accelerate downstream use. To validate the effectiveness of MONET, we train a 4B-parameter latent diffusion model exclusively on it and reach competitive GenEval and DPG scores, demonstrating that our dataset lowers the barrier to large-scale, reproducible text-to-image research. 

---
# How Much Online RL is Enough? Informative Rollouts for Offline Preference Optimization in RLVR 

**Authors**: Richa Verma, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2605.21266)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for reasoning in language models, with GRPO as its primary example. However, GRPO requires continuous online rollout generation, making it computationally expensive and difficult to scale. While Direct Preference Optimization (DPO) offers a stable and efficient offline alternative, it is typically expected to underperform w.r.t. online RL methods such as GRPO when trained on rollouts from a cold supervised fine-tuned (SFT) policy. We introduce G2D (GRPO to DPO)}, a three-stage pipeline that performs a short GRPO warm-up, constructs a static preference dataset, and fine-tunes a model offline with DPO. Across a set of values of the number of online steps (K) in GRPO on Qwen2.5-7B and Llama-3.1-8B, we find that offline DPO with moderate warm-up matches or outperforms GRPO at substantially lower compute cost in our setting. On Qwen2.5-7B, G2D at K=150 achieves 62.4% on MATH-500, outperforming GRPO (51.6%) by 10.8% at ~4x lower compute. On Llama-3.1-8B, G2D at K=500 achieves 49.4%, surpassing GRPO in our experimental setting. We show that performance is not governed by the number of preference pairs, which does not vary much w.r.t. K, but by their informativeness. Moderate warm-up produces rollouts with calibrated uncertainty, yielding stronger contrastive signal, while excessive warm-up leads to overconfident policies and less informative data. Our results recast the offline-online gap in RLVR as primarily a data informativeness problem, and identify short online RL warm-up with appropriate difficulty calibration of the fine-tuning dataset as a compute-efficient alternative to online RL. 

---
# Learning Structural Latent Points for Efficient Visual Representations in Robotic Manipulation 

**Authors**: Yicheng Jiang, Jiaxu Wang, Junhao He, Zesen Gan, Junhao Li, Qiang Zhang, Jingkai Sun, Jiahang Cao, Mingyuan Sun, Xiangyu Yue, Qiming Shao  

**Link**: [PDF](https://arxiv.org/pdf/2605.21258)  

**Abstract**: Current 3D-aware pretraining methods for embodied perception and manipulation are largely built on differentiable rendering frameworks, producing either fully implicit neural fields or fully explicit geometric primitives. Implicit representations, while expressive, lack explicit structural cues, whereas explicit ones preserve geometry but suffer from resolution limits and weak generalization. To address these limitations, we propose a novel pretraining framework that learns a hybrid representation-structural latent points. Specifically, we insert a point-wise latent variational autoencoder into the latent space of a point-cloud autoencoder, jointly regularizing point-wise features and coordinates toward a Gaussian prior. The resulting compact latent preserves coarse structural tendencies, which do not encode precise geometry but capture richer rough shape and semantic information, effectively combining the expressiveness of implicit representations with the structural priors of explicit ones. In addition, informed by shared design choices in prior work, we develop a streamlined, efficient 3DGS-based rendering pipeline that is deliberately kept lightweight, improving efficiency while leaving greater representational capacity to the front-end latent module. Extensive evaluations on RLBench, ManiSkill2, and a real-robot platform demonstrate consistent gains in task success, sample efficiency, and robustness to viewpoint and scene variations over strong baselines. Ablation studies further confirm that each component of our framework is critical to overall performance. 

---
# APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents 

**Authors**: Yibo Li, Jiashuo Yang, Zhi Zheng, Zhiyuan Hu, Yuan Sui, Shizun Wang, Yufei He, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2605.21240)  

**Abstract**: LLM agents have shown strong performance across a wide range of complex tasks, including interactive environments that require long-horizon decision making. But these agents cannot learn on the fly at test time. Self-evolving agents address this by accumulating memory and reflection across episodes rather than requiring model-weight updates. However, these agents often suffer from exploration collapse: as memory grows, behavior concentrates around familiar high-reward routines, reducing the chance of discovering better alternatives. To address this problem, we propose Autonomous Policy EXploration (APEX), which builds and maintains an explicit strategy space through a strategy map-a directed acyclic graph of milestones with prerequisite dependency edges. In APEX, Fork Discovery expands the map with evidence-grounded unexplored directions, while Policy Selection balances exploration and exploitation during planning. Evaluated on nine Jericho text-adventure games and WebArena, a realistic web interaction benchmark, APEX outperforms all baselines. Extensive ablations validate each component's contribution and demonstrate robustness across diverse settings, demonstrating APEX's effectiveness for sustained exploration in self-evolving agents. 

---
# RePCM: Region-Specific and Phenotype-Adaptive Bi-Ventricular Cardiac Motion Synthesis 

**Authors**: Xuan Yang, Xiaohan Yuan, Hao Li, Lingyu Chen, Yanan Liu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21237)  

**Abstract**: Cardiac motion over a cardiac cycle is crucial for quantifying regional function and is strongly affected by cardiovascular diseases. Since temporally dense mesh sequences are difficult to obtain in practice, we focus on leveraging the more accessible end-diastolic frame to infer a full-cycle sequence. Due to strong regional and disease-specific differences, traditional methods often oversmooth the data by relying on generative models that are optimized for global patterns. To address this problem, we propose Region-Aware and Phenotype-Adaptive Bi-Ventricular Cardiac Motion Synthesis (RePCM) for single frame Bi-ventricular mesh motion completion. In Stage I, a reconstruction network learns vertex wise motion descriptors and clustering yields a data driven functional partition, providing an explicit motion derived region structure. In Stage II, a Region-Specific Injection Module enforces masked, synchronized region exchange within a conditional VAE, preserving localized specific dynamics and restricting cross-region mixing. A Phenotype-Adaptive Mixture-of-Experts prior conditioned on ED shape uses anatomy-guided cues to model latent motion trends and capture inter-disease variability. Experiments on three datasets covering different cardiovascular diseases show consistent gains in geometric and functional metrics and improved preservation of region specific dynamics. 

---
# OCTOPUS: Optimized KV Cache for Transformers via Octahedral Parametrization Under optimal Squared error quantization 

**Authors**: Mark Boss, Vikram Voleti, Simon Donné, Shimon Vainer  

**Link**: [PDF](https://arxiv.org/pdf/2605.21226)  

**Abstract**: The key-value (KV) cache dominates memory bandwidth and footprint in long-context autoregressive inference. Recent rotation-preconditioned codecs (TurboQuant, PolarQuant) show that a structured random rotation followed by a per-coordinate scalar quantizer matched to an analytically tractable marginal is a near-optimal recipe for KV compression. OCTOPUS advances this paradigm through joint quantization of rotated coordinate triplets. Each triplet's direction is mapped to a square via an octahedral parameterization, and the two resulting coordinates and the triplet norm are Lloyd-Max quantized against implementation-matched marginals. Optimizing the per-triplet squared error gives a strictly non-uniform bit allocation depending only on the total dimensionality of the keys. We find the finite-dimensional quality optimum with sweeps to be constant on every real decoder we test. The codec is data-oblivious, online, and deterministic given a seed. Across text, video, and audio, OCTOPUS matches or beats every prior rotation codec at every reported bit width and metric, with a lead that grows as bits drop for extreme compression. Furthermore, a fused Triton implementation reconstructs keys on the fly without materializing the uncompressed key, so the codec adds no decode-time bandwidth or latency over the existing dequantization. Project Page: this https URL 

---
# PREFINE: Preference-Based Implicit Reward and Cost Fine-Tuning for Safety Alignment 

**Authors**: Richa Verma, Bavish Kulur, Sanjay Chawla, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2605.21225)  

**Abstract**: We address the problem of making a pre-trained reinforcement learning (RL) policy safety-aware by incorporating cost constraints without retraining it from scratch. While costs could be numerically encoded, we assume a more general setting is when costs are provided as preferences. Given a reward-optimized policy and a small dataset of preferred (low-cost) and dispreferred (high-cost) trajectories, our goal is to fine-tune the policy to generate low-cost behaviors while retaining high rewards. Unlike standard RLHF in language models, where preferences are defined over responses to the same prompt, our setting involves trajectory-level preferences in continuous control environments. We introduce PREFINE: Preference-based Implicit Reward and Cost Fine-Tuning for Safety Alignment which is a preference-based fine-tuning method that adapts Direct Preference Optimization (DPO), which is now widely used for LLM fine-tuning, to the sequential decision making setting. PREFINE constructs policy-sampled counterfactual trajectories to establish meaningful preference contrasts and jointly optimizes for reward retention and safety alignment. Empirically, PREFINE reduces constraint violations and catastrophic failures by over 60% while maintaining original reward behavior. PREFINE produces policies that achieve low-cost, high-reward performance with significantly improved data and computational efficiency compared to full offline RL or imitation learning, bridging preference alignment and safe policy adaptation in continuous domains. 

---
# Artificial Intelligence Reshapes Microwave Photonics 

**Authors**: Peng Li, Xihua Zou, Jia Ye, Wei Pan, Lianshan Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.21224)  

**Abstract**: As a rapidly emerging interdisciplinary field that intrinsically integrates microwave and photonics, microwave photonics (MWP) provides disruptive solutions to overcome the fundamental bandwidth of conventional electronic systems. By exploiting the inherently ultra-wide bandwidth and low-loss characteristics of photonic technologies, MWP enables the generation, transmission, processing, and detection of microwave, millimeter-wave, and terahertz signals. Representative breakthroughs include fully photonic microwave radar systems, photonic analog-to-digital converters with bandwidth up to 320 GHz, and photonic wireless communication systems achieving data rate as high as 616 Gbit/s. Meanwhile, the rapid growth of artificial intelligence (AI) is reshaping scientific research, engineering, and daily life in unprecedented ways, such as AI for science/engineering and AI co-scientist/assistant. Correspondingly, AI is profoundly reshaping MWP in all aspects, ranging from signal generation, transmission to signal processing and detection. AI has revolutionized the design, simulation, fabrication, testing, deployment, and maintenance of MWP systems, delivering autonomous operation and exceptional efficiency beyond traditional systems. Motivated by these developments, this Review Paper provides the first comprehensive overview of AI-enabled MWP, systematically summarizing the state-of-the-art advances and presenting insights for both the academic community and the broader public. 

---
# Behavior-Consistent Deep Reinforcement Learning 

**Authors**: Marcel Hussing, Liv G. d'Aliberti, Claas Voelcker, Benjamin Eysenbach, Eric Eaton  

**Link**: [PDF](https://arxiv.org/pdf/2605.21214)  

**Abstract**: Reinforcement learning (RL) often exhibits high variance across training runs, leading to unreliable performance and posing a major challenge to deployment in real-world domains. In this work, we address the challenge of cross-run policy divergence by formalizing the problem of behavior-consistent RL, where the objective is to obtain policies that are both high-performing and distributionally similar across training runs. Our key observation is that maximum-entropy RL provides a direct mechanism for controlling behavioral divergence by anchoring runs to a common (uniform) prior. We prove that, for Boltzmann policies, choosing the temperature proportional to $Q$-function disagreement bounds the pairwise KL divergence between the induced policies. However, we also show that naïvely increasing entropy might impair policy optimization while amplifying off-policy error. Building upon these observations, we propose $Q$-value Expectile Disagreement (QED), a state-dependent temperature schedule that uses double-critic disagreement as a single-run proxy for cross-run disagreement. Empirically, we demonstrate that across 18 continuous-control tasks, QED reduces across-run divergence by two orders of magnitude without sacrificing performance, resulting in a considerable reduction in return variance at modest sample-efficiency costs. 

---
# Enhanced Reinforcement Learning-based Process Synthesis via Quantum Computing 

**Authors**: Austin Braniff, Fengqi You, Yuhe Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.21213)  

**Abstract**: In this work, we present quantum reinforcement learning (RL) as a solution strategy for process synthesis problems. Building on our prior work, we develop a generalized framework that formally poses process synthesis as a Markov decision process and introduces quantum-enhanced RL algorithms to solve it with improved scalability. Earlier implementations of quantum-based RL for process synthesis were limited by qubit requirements, which scaled poorly with problem complexity. This work overcomes this challenge by introducing state encoding algorithms to decouple qubit requirements from problem size. A classical RL-based solution strategy is used as a baseline to benchmark the quantum algorithms under identical training conditions. All algorithms are evaluated across a flowsheet synthesis problem of increasing unit counts to analyze their performance and scalability. Results show that all approaches are capable of identifying the optimal flowsheet designs in small design spaces. For moderate-scale unit counts, quantum approaches demonstrate competitive performance on a per-episode basis and improved efficiency on a per-parameter basis versus the classical RL benchmark. This work provides a foundation for future quantum computing applications within process systems engineering, establishes a controlled benchmark for comparing classical and quantum algorithms, and shows that the proposed quantum variants remain competitive for the process synthesis problem examined in this work. 

---
# SURGE: An Event-Centric Social Media Sentiment Time Series Benchmark with Interaction Structure 

**Authors**: Chen Su, Pengsen Cheng, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.21198)  

**Abstract**: Public events on social media generate large volumes of discussion whose collective dynamics carry direct value for opinion forecasting and crisis response. Capturing how these dynamics evolve across an event's lifecycle requires organizing fragmented posts into event-level time series. Existing datasets cover only a small number of events within a single category, and typically discard the interaction structure between posts when constructing time series, which restricts both transfer across event types and controlled study of how interactions shape the resulting collective dynamics. We present SURGE, a multi-event social media benchmark that pairs event-level time series with aligned text and interaction structure linking posts within an event. SURGE is built through an automated pipeline that produces calendar-aligned time series at three temporal granularities, covering 67 events and more than 800K posts across five event categories. Each time bin is paired with flat and structured textual views derived from the same selected posts, enabling controlled evaluation of whether social interaction structure affects forecasting behavior. On top of SURGE we define benchmark protocols for numerical-only forecasting, text-augmented forecasting, high-interaction evaluation, and leave-one-category-out generalization. Experiments with representative time-series and multimodal forecasting models reveal three properties of the benchmark: a strong local-persistence regime in which naive baselines remain hard to beat under absolute error, limited transfer of existing text-augmented forecasters to event-driven social-media data, and increased difficulty on reply-dense periods that aggregate metrics tend to obscure. We further include a lightweight structure-aware probe as a reference implementation, illustrating how SURGE can support interaction-aware forecasting research. 

---
# SAM-Sode: Towards Faithful Explanations for Tiny Bacteria Detection 

**Authors**: Wanying Tan, Shuo Yan, Dazhi Huang, Yazheng Liu, Zili Shao, Rufeng Chen, Hechang Chen, Mude Shi, Tianxing Ji, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.21186)  

**Abstract**: Interpretability in object detection provides crucial confidence support for clinical auxiliary diagnosis. However, in tiny bacteria detection, traditional explanation methods often suffer from blurred foreground boundaries and diffuse feature attribution due to the extreme sparsity of target morphological features and severe interference from complex backgrounds. Such limitations hinder the provision of logically coherent morphological evidence. To bridge this gap, we propose a novel eXplainable AI (XAI) framework, SAM-Sode. The framework innovatively transforms initial feature attribution maps into geometry-aware prompts, leveraging the prior knowledge of the foundation model (SAM3) to achieve spatial refinement and morphological reconstruction of the explanatory mappings. Furthermore, we introduce a dual-constraint mechanism based on physical significance and geometric alignment to perform instance-level denoising, generating coherent explanations that better align with human expert intuition. Experimental results on our self-constructed bacteria dataset with complex circuit backgrounds (containing 2,524 images) and other public datasets demonstrate that the proposed method effectively suppresses background redundancy and significantly enhances the decision-making transparency of tiny object detection. 

---
# Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding 

**Authors**: Jeonghun Baek, Atsuyuki Miyai, Shota Onohara, Hikaru Ikuta, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2605.21182)  

**Abstract**: Manga is a culturally distinctive multimodal medium and one of the most influential forms of Japanese popular culture. As AI systems increasingly target manga understanding, OCR, and translation, Manga109 has become a foundational dataset for manga-related AI research. However, the current Manga109 dataset contains transcription errors and coarse annotations, which do not align well with modern OCR and multimodal manga understanding tasks. In this work, we revisit the dialogue text annotations of Manga109 and identify five categories of annotation issues, including transcription errors, missing text regions, overlapping dialogue and onomatopoeia, and under-segmented speech balloons. To address these issues, we combine OCR-based issue detection and manual revision to construct Manga109-v2026, revising approximately 29,000 dialogue annotations. Our revisions better align Manga109 with modern OCR and multimodal manga understanding systems while preserving expressive structures characteristic of manga. 

---
# Comparative Analysis of Military Detection Using Drone Imagery Across Multiple Visual Spectrums 

**Authors**: Sourov Roy Shuvo, Prajwal Panth, Rajesh Chowdhury, Sorup Chakraborty, Sudip Chakrabarty, Prasant Kumar Pattnaik  

**Link**: [PDF](https://arxiv.org/pdf/2605.21157)  

**Abstract**: In modern warfare, drones are becoming an essential part of intelligence gathering and carrying out precise attacks in different kinds of hostile environments. Their ability to operate in real-time and hostile environments from a safe distance makes them invaluable for surveillance and military operations. The KIIT-MiTA dataset is comprised of images of different military scenarios taken from drones, and these provide a foundation for detecting military objects, but it does not take into account the various types of real-world scenarios. With that in mind, to evaluate how the models are performing under varying conditions, four different types of datasets are created: Gray Scale, Thermal Vision, Night Vision, and Obscura Vision. These simulate the real-world environments such as low visibility, heat-based imagery, and nighttime conditions. The YOLOv11-small model is trained and used to detect objects across diverse settings. This research boosts the performance and reliability of drone-based operations by contributing to the development of advanced detection systems in both defensive and offensive missions. 

---
# Automated ICD Classification of Psychiatric Diagnoses: From Classical NLP to Large Language Models 

**Authors**: Fernando Ortega, Raúl Lara-Cabrera, Jorge Dueñas-Lerín, Alejandro de la Torre-Luque, Mercé Salvador Robert, Enrique Baca-García  

**Link**: [PDF](https://arxiv.org/pdf/2605.21154)  

**Abstract**: Mental health has become a global priority, leading to a massive administrative burden in the coding of clinical diagnoses. This study proposes the automation of psychiatric diagnostic analysis by mapping free-text descriptions to the International Classification of Diseases (ICD) using Natural Language Processing (NLP) and Machine Learning (ML) techniques. Utilizing a specialized dataset of 145,513 Spanish psychiatric descriptions, various text representation paradigms were evaluated, ranging from classical frequency-based models (BoW, TF-IDF) to state-of-the-art Large Language Models (LLMs) such as e5\_large, BioLORD, and Llama-3-8B. Results indicate that transformer-based embeddings consistently outperform traditional methods by capturing implicit semantic cues and nuanced medical terminology. The e5\_large model, through end-to-end fine-tuning, achieved the highest performance with a $F1_{micro}$ score of 0.866. This research demonstrates that adapting LLMs to specific clinical nomenclature is essential for overcoming the challenges of ``long-tail'' label distributions and the inherent ambiguity of psychiatric discourse. 

---
# Detecting Trojaned DNNs via Spectral Regression Analysis 

**Authors**: Samuele Pasini, Jinhan Kim, Paolo Tonella  

**Link**: [PDF](https://arxiv.org/pdf/2605.21146)  

**Abstract**: Modern DNNs are repeatedly fine-tuned to incorporate new data and functionality. This evolutionary workflow introduces a security risk when updated data cannot be fully trusted, as adversaries may implant Trojans during fine-tuning. We present MIST, a Trojan detection approach that analyzes how a model's internal representations change during fine-tuning. Rather than attempting to reconstruct trigger conditions, MIST characterizes benign model evolution using pre-activation spectra and flags updates whose spectral deviations are inconsistent with this reference. This framing treats Trojan detection as a regression problem over model updates. An empirical evaluation across four datasets and eight Trojan attacks shows that spectral distances reliably distinguish Trojaned updates from clean fine-tuning. MIST outperforms state-of-the-art detection accuracy after a single update, without requiring any knowledge about the poisoned data or the trigger, and remains effective under multi-step benign evolution, with graceful and bounded degradation. These results indicate that spectral evolution provides a stable and assumption-light signal for detecting malicious model updates. 

---
# On the Complexity of Entailment for Cumulative Propositional Dependence Logics 

**Authors**: Kai Sauerwald, Juha Kontinen, Arne Meier  

**Link**: [PDF](https://arxiv.org/pdf/2605.21113)  

**Abstract**: This paper establishes and proves complexity results for entailment for cumulative propositional dependence logic and for cumulative propositional logic with team semantics. As recently shown, cumulative logics are famously characterised by System~C and exactly captured by the cumulative models of Kraus, Lehmann and Magidor. This gives rise to the entailment problem via relational models, which is specifically considered here. 

---
# Efficient Learning of Deep State Space Models via Importance Smoothing 

**Authors**: John-Joseph Brady, Nikolas Nusken, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.21108)  

**Abstract**: Latent state space systems are ubiquitous in statistical modelling, arising naturally when a time series is observed through a noisy measurement function, however training deep state space models (DSSM) at scale remains difficult. Two largely distinct strategies and literatures have developed around the training of DSSMs. Firstly, auto-encoding DSSMs train generative DSSMs by optimising a variational lower bound. Secondly, DSSMs trained by back-propagating the outputs of a classical sequential Monte Carlo algorithm (SMC). Such approaches can train DSSMs for discriminative as well as generative tasks, however, due to the sequentiality of their forward pass, scale poorly on modern hardware. We propose a new training method \emph{parallel variational Monte Carlo} (PVMC) that bridges the gap between the paradigms, and can be used robustly to train DSSMs for both discriminative and generative tasks. Our method achieves state-of-the-art or better results on a set of baseline experiments and trains $10\times$ faster than the fastest competing SMC approach. 

---
# ACL-Verbatim: hallucination-free question answering for research 

**Authors**: Gábor Recski, Szilveszter Tóth, Nadia Verdha, István Boros, Ádám Kovács  

**Link**: [PDF](https://arxiv.org/pdf/2605.21102)  

**Abstract**: Academic researchers need efficient and reliable methods for collecting high-quality information from trusted sources, but modern tools for AI-assisted research still suffer from the tendency of Large Language Models (LLMs) to produce factually inaccurate or nonsensical output, commonly referred to as hallucinations. We apply the extractive question answering system VerbatimRAG to research papers in the ACL Anthology, directly mapping user queries to verbatim text spans in retrieved documents. We contribute a novel ground truth dataset for the task of mapping user queries to relevant text spans in research papers, and use it to train and evaluate a variety of extractive models. Human annotation is performed by NLP researchers and is based on synthetic user queries generated using a custom pipeline based on the ScIRGen methodology, paired with chunks of research papers retrieved by VerbatimRAG. On this benchmark, a 150M-parameter ModernBERT token classifier trained on silver supervision from our pipeline achieves the best word-level F1 (53.6), ahead of the strongest evaluated LLM extractor (48.7). 

---
# Decoupling Communication from Policy: Robust MARL under Bandwidth Constraints 

**Authors**: Alexi Canesse, Benoît Goupil, Jesse Read, Sonia Vanier  

**Link**: [PDF](https://arxiv.org/pdf/2605.21085)  

**Abstract**: Communication enables coordination in multi-agent reinforcement learning (MARL), but many real-world applications, e.g., search-and-rescue with drone swarms, operate under severe bandwidth constraints. Many communication architectures still expose a coupled bottleneck in which a shared latent representation is used for both policy execution and inter-agent communication. Consequently, reducing message size directly limits the policy's latent space, often leading to significant performance degradation. We address this with two contributions. First, we introduce $\beta$, a normalised per-agent bandwidth budget that unifies sparsity, rounds, and message dimension into a single comparable constraint. Second, we provide SLIM, a minimal architecture that decouples the communication pathway from the policy's latent representation, allowing us to isolate the effect of bandwidth from the effect of policy capacity while benefiting from in-step communication. We evaluate our method on several partially-observable MARL benchmarks, where communication is essential. Our approach achieves state-of-the-art performance and exhibits scalability and robustness under limited communication, with only marginal degradation as bandwidth is reduced. 

---
# Fine-grained Claim-level RAG Benchmark for Law 

**Authors**: Souvick Das, Sallam Abualhaija, Domenico Bianculli  

**Link**: [PDF](https://arxiv.org/pdf/2605.21071)  

**Abstract**: The rapid progress of large language models (LLMs) is shifting semantic search toward a question-answering paradigm, where users ask questions and LLMs generate responses. In high-stake domains such as law, retrieval-augmented generation (RAG) is commonly used to mitigate hallucinations in generated responses. Nonetheless, prior work shows that RAG systems, whether general-purpose or legal-specific, still hallucinate at varying rates, making fine-grained evaluation essential. Despite the need, existing evaluation frameworks for legal RAG systems lack the granularity required to provide detailed analysis of retrieval and generation performance separately. Moreover, current benchmarks are largely English-only and centered on legal expert queries, overlooking non-expert needs. We introduce ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and English, targets both experts and non-experts, and includes diverse question types reflecting realistic scenarios. We further apply a fine-grained evaluation framework of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation, and claim-level analysis in the legal domain. 

---
# Grounding Driving VLA via Inverse Kinematics 

**Authors**: Junsung Park, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2605.21061)  

**Abstract**: Existing Driving VLAs predict trajectories while largely ignoring their visual tokens -- a phenomenon we trace not to insufficient training but to a structurally ill-posed task formulation. We show that trajectory recovery, when viewed through the lens of inverse kinematics, requires both a current and a future visual state as boundary conditions; existing VLAs supply only the former, which encourages the model to shortcut through ego status and text commands alone. To address this, we re-design Driving VLA in the style of an inverse kinematics solver. First, a next visual state prediction objective that requires the LLM to predict the future visual scene provides dense visual supervision and suppresses shortcut paths. Second, a separate Inverse Kinematics Network (a cross-attention-based conditional diffusion model) that takes only the current and future visual states as input is designed to suppress reliance on ego status and textual shortcuts during trajectory decoding. With this simple prescription alone, our 0.5B-scale model recovers visual grounding and reaches trajectory planning performance comparable to 7B--8B VLAs more than an order of magnitude larger, on both the closed-loop NAVSIM-v2 and the nuScenes benchmarks. Extensive analysis further shows that this improvement stems from a recovered ability to exploit visual features, with the effect being most pronounced in dynamic driving situations such as turning. 

---
# Divide et Calibra: Multiclass Local Calibration via Vector Quantization 

**Authors**: Cesare Barbera, Lorenzo Perini, Giovanni De Toni, Andrea Passerini, Andrea Pugnana  

**Link**: [PDF](https://arxiv.org/pdf/2605.21060)  

**Abstract**: Accurate and well-calibrated Machine Learning (ML) models are mandatory in high-stakes settings, yet effective multiclass calibration remains challenging: global approaches assume calibration errors are homogeneous across the latent space, while local methods often rely on latent-space dimensionality reduction, which leads to information loss. To address these issues, we propose a compositional approach to multiclass calibration, where region-specific calibration maps are constructed from shared codeword-dependent factors. We instantiate this idea via Vector Quantization (VQ), which induces a structured partition of the representation space, and an indexed parameterization of Dirichlet concentrations that enables parameter sharing across regions. Our approach learns heterogeneous calibration maps that generalize well even to sparse regions of the latent space. Experiments on benchmark datasets show significant improvements in local calibration while maintaining competitive global calibration and predictive performance. 

---
# DySink: Dynamic Frame Sinks for Autoregressive Long Video Generation 

**Authors**: Bo Ye, Xinyu Cui, Jian Zhao, Tong Wei, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.21028)  

**Abstract**: Autoregressive long video generation often adopts bounded-memory streaming for efficiency, typically combining local windows for short-term continuity with static early-frame sinks as long-range anchors. However, this fixed allocation keeps early frames cached even when the current visual state has substantially diverged from them, while discarding potentially more relevant intermediate history. As a result, the retained long-range context may become less adaptive and bias generation toward outdated cues; in severe cases, RoPE-induced phase re-alignment can homogenize inter-head attention and cause sink collapse, where content regresses toward sink frames. We propose DySink, a retrieval-based framework that maintains a compact memory bank and selects visually relevant historical frames as dynamic frame sinks. DySink couples adaptive retrieval with a sink anomaly gate, which detects excessive inter-head consensus over retrieved context and suppresses collapse-prone context. Experiments on minute-long videos show that DySink consistently improves dynamic degree over strong baselines while also achieving higher temporal quality. The code and model weights will be released at this https URL. 

---
# Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs 

**Authors**: Gundeep Singh, Parsa Kavehzadeh, Jing Xia, Xue-Yong Fu, Julien Bouvier Tremblay, Md Tahmid Rahman Laskar, Vincent Lum, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2605.21027)  

**Abstract**: Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration. 

---
# Single-Pass, Depth-Selective Reading for Multi-Aspect Sentiment Analysis 

**Authors**: Yan Xia, Zhuangzhuang Pan, Amirrudin Kamsin, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20998)  

**Abstract**: Aspect-Term Sentiment Analysis (ATSA) in multi-aspect sentences faces a fundamental tradeoff between efficiency and expressiveness. Existing models either re-encode the sentence for each aspect or rely on static use of deep representations, leading to redundant computation and limited adaptivity. We argue that Transformer depth is a costly, queryable resource, and propose DABS, a single-pass inference framework that encodes each sentence once to construct a reusable, depth-ordered substrate. Each aspect then queries this shared representation to selectively read relevant tokens and abstraction levels, without re-encoding. This decouples shared sentence encoding from lightweight, aspect-conditioned readout. Experiments on four ATSA benchmarks show that DABS achieves competitive performance while reducing end-to-end computation by up to 60% in multi-aspect settings (M >= 2). Further analyses indicate that adaptive depth querying is most beneficial for linguistically complex cases such as negation and contrast. Code is publicly available at this https URL 

---
# Hybrid Machine Learning Model for Forest Height Estimation from TanDEM-X and Landsat Data 

**Authors**: Islam Mansour, Ronny Haensch, Irena Hajnsek, Konstantinos Papathanassiou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20997)  

**Abstract**: Integrating machine learning (ML) with physical models (PM) has emerged as a promising way of retrieving geophysical parameters from remote sensing data. In this context, a ML model for estimating forest height from TanDEM-X interferometric coherence measurements has recently been proposed, that constrains the learning process through a PM. While the features used for training and inversion where selected to ensure the physical consistency of the solutions, they could not resolve all height / structure and baseline / terrain slope ambiguities in the data. To improve this, the extension of the feature space with optical Landsat data is proposed able to provide complementary information on forest type or structure. The extended model is applied and validated on several TanDEM-X acquisitions over the Gabonese Lopé national park site and assessed against airborne LiDAR measurements. Results show a 13.5% reduction in RMSE and a 16.6% reduction in MAE compared to the original hybrid model, confirming the added value of multispectral inputs. 

---
# Towards Context-Invariant Safety Alignment for Large Language Models 

**Authors**: Yixu Wang, Yang Yao, Xin Wang, Yifeng Gao, Yan Teng, Xingjun Ma, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20994)  

**Abstract**: Preference-based post-training aligns LLMs with human intent, yet safety behavior often remains brittle. A model may refuse a harmful request in a standard prompt but comply when the same intent is wrapped in adversarial wording. We suggest that robust safety requires context-invariant alignment, where behavior depends on the underlying intent rather than surface form. Enforcing invariance is difficult in alignment because not all training signals are equally trustworthy; for some prompt variants we can obtain verifiable feedback (e.g., multiple-choice), while for open-ended variants we typically rely on noisy, gameable reward proxies (e.g., learned judges). As a result, standard symmetric invariance regularizers can reduce cross-context discrepancies by lowering performance on reliable variants instead of improving open-ended robustness. To address this, we introduce Anchor Invariance Regularization (AIR), which treats verifiable prompts as anchors and uses a stop-gradient target to regularize only the open-ended variants toward the anchor performance. AIR is implemented as a plug-in auxiliary loss and combined with group-based preference optimization (e.g., GRPO) via heterogeneous prompt grouping. Across Safety, Moral Reasoning, and Math, AIR improves context invariance, boosting in-distribution group accuracy by 12.71% and out-of-distribution consistency by 33.49%, making safety constraints robust to adversarial framings. 

---
# A Sharper Picture of Generalization in Transformers 

**Authors**: Paul Lintilhac, Sair Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2605.20988)  

**Abstract**: We study transformers' generalization behavior on boolean domains from the perspective of the Fourier Spectra of their target functions. In contrast to prior work (Edelman et al., 2022; Trauger and Tewari, 2024), which derived generalization bounds from Rademacher complexity, we investigate the feasibility of obtaining generalization bounds via PAC-Bayes theory. We show that sparse spectra concentrated on low-degree components enable low-sharpness constructions with good generalization properties. Our idea is to show the existence of flat minima implementing any boolean function of sparsity no greater than the context length, and then apply a PAC-Bayes bound to an idealized low-sharpness learner, resulting in a non-vacuous generalization bound. We evaluate predictions empirically and conduct a mechanistic interpretability study to support the realism of our theoretical construction in real transformers. 

---
# Diagnosing Overhead in Dispatch Operations: Cross-architecture Observatory 

**Authors**: Bole Ma, Jan Eitzinger, Harald Koestler, Gerhard Wellein  

**Link**: [PDF](https://arxiv.org/pdf/2605.20982)  

**Abstract**: AlltoAll dispatch is the dominant bottleneck of MoE expert parallelism, and the interconnect community has responded with four families of mitigations: predictive sample placement, adaptive expert relayout, hierarchical collectives, and EP-aware topology. All four rest on two assumptions about the workload. The first is that routing imbalance is correctable by the system layer. The second is that the mock-token benchmarks evaluating them faithfully represent production routing. We introduce DODOCO to test both assumptions.
We instrument five MoE checkpoints spanning five sequence-mixer designs (DeepSeek-V2-Lite MLA, DeepSeek-MoE-16B MHA, Qwen3-30B GQA, Nemotron-30B Mamba-2, Qwen3.5-35B GDN) under a 5 by 6 grid of data conditions plus a matched EP scan from 4 to 32 ranks on H100s; both assumptions fail. Scaling EP changes the per-expert max/mean token ratio by at most 5% within every architecture's measurable range: the straggler is intrinsic to the routing decision the model makes, not to how its experts land on ranks. Mock tokens overestimate routing Gini by up to a factor of 2.35 and fabricate a batch-size scaling trend that vanishes the moment real text replaces random IDs.
A third pattern, unexpected, emerges from the same matrix: the five architectures cleave into two stable bands. MHA and Mamba-2 (data-resilient) drop to Gini 0.105 and 0.150 on wikitext. MLA and GDN (persistently concentrated) stay above 0.24 on every real-text condition and reach 0.29 to 0.38 on mock. GQA is the intermediate case. These bands, not the EP degree or the mock-data profile, are the right workload input to AlltoAll-aware interconnect and dispatch design. 

---
# Comparative Evaluation of Deep Learning Models for Fake Image Detection 

**Authors**: Akhitha Pakala, Mohammed Mahir Rahman, Shahzad Memon, Tauseef Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.20971)  

**Abstract**: The growing sophistication of GAN-based image manipulation presents significant challenges for digital forensics. This study compares the performance of four pretrained CNN architectures including VGG16, ResNet50, EfficientNetB0, and XceptionNet for fake image detection using a unified preprocessing and training pipeline. A dataset of real and manipulated images was processed through resizing, normalization, and augmentation to address class imbalance and improve generalization. Models were evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. VGG16 achieved the highest accuracy at 91%, with XceptionNet, ResNet50, and EfficientNetB0 each reaching 90%. EfficientNetB0 showed stronger sensitivity to fake images but reduced reliability on real samples, reflecting imbalance-driven bias. Limitations include dataset imbalance, overfitting, and limited interpretability, which affect cross-domain robustness. The study provides a reproducible baseline and underscores the need for balanced datasets, advanced augmentation, and fairness-aware training to develop reliable fake image detection systems. 

---
# Finding the Correct Visual Evidence Without Forgetting: Mitigating Hallucination in LVLMs via Inter-Layer Visual Attention Discrepancy 

**Authors**: Yutong Xie, Zhenglin Hua, Ran Wang, Wing W. Y. Ng, Xizhao Wang, Yuheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2605.20965)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown remarkable performance on a wide range of vision-language tasks. Despite this progress, they are still prone to hallucination, generating responses that are inconsistent with visual content. In this work, we find that LVLMs tend to hallucinate when they pay insufficient attention to the correct visual evidence and gradually forget it during the generation process. We empirically find that although LVLMs overall attend insufficiently to visual evidence, they exhibit sensitivity to the correct visual evidence in specific layers, with notable inter-layer discrepancy. Motivated by this observation, we propose a novel hallucination mitigation method that enhances visual evidence based on Inter-Layer Visual Attention Discrepancy (ILVAD). Specifically, we obtain the attention weights from early generated tokens to visual tokens across layers and identify the tokens that are repeatedly activated as visual evidence, forming a saliency map. We then enhance attention to visual evidence during generation through the saliency map to reduce visual forgetting. In addition, we leverage the saliency map to obtain attention scores of generated text to visual evidence, in order to select and emphasize text tokens that are strongly grounded in visual evidence. Our method is training-free and plug-and-play. Multiple benchmark evaluations conducted on five recently released models show that our method can consistently mitigate hallucinations in different LVLMs over various architectures. Code is available at this https URL. 

---
# Focus-then-Context: Subject-Centric Progressive Visual Token Reduction for Vision-Language Models 

**Authors**: Yulin Zhao, Yun Wang, Dehua Zheng, Borui jiang, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20950)  

**Abstract**: Vision-Language Models (VLMs) face a bottleneck of prohibitive computational costs arising from massive visual token sequences during inference. Existing vision token reduction methods alleviate this burden, but they unintentionally preserve the isolated visual subject strictly aligned with the user's query, which fails to substantially explore salient subjects and their contextual relationships. In this paper, we propose SPpruner, a subject-centric progressive reduction paradigm that emulates the \textit{Focus-then-Context} mechanism of the human visual perception system. Specifically, we first construct a focus identification module to explicitly model the interplay between visual saliency and semantic relevance. Herein, it can excavate the comprehensive visual subject spectrum to ensure a high-fidelity representation of visual input. Subsequently, a context-aware structural scanning module is developed to aggregate contextual cues from neighboring regions. As such, it can effectively restore global relational dependencies to uphold the structural integrity of the preserved subjects. Extensive experiments demonstrate that our paradigm consistently outperforms SOTA methods, achieving up to 2.53 times speedup with only 22.2% of visual tokens retained in Qwen2.5-VL and a 67% FLOPs reduction on LLaVA with a negligible 0.6% accuracy drop. 

---
# DASH: Fast Differentiable Architecture Search for Hybrid Attention in Minutes on a Single GPU 

**Authors**: Weizhe Chen, Miao Zhang, Junpeng Jiang, Yaping Li, Weili Guan, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.20936)  

**Abstract**: Hybrid attention architectures are becoming an increasingly important paradigm for improving LLM inference efficiency while preserving model quality, making hybrid architecture design a central problem. Existing designs often rely on manual empirical rules or proxy-based selector signals for layer-wise operator allocation. Recent NAS-style systems such as Jet-Nemotron demonstrate the promise of automated hybrid architecture search. However, Jet-Nemotron's PostNAS search stages alone use 200B tokens, making such search pipelines difficult to use as routine methods for hybrid architecture design. We introduce DASH, a fast differentiable search framework for hybrid attention architecture design, which relaxes discrete layer-wise attention operator placement into continuous architecture logits, prepares reusable teacher-aligned linear candidates, and performs architecture-only search with model and operator weights frozen to significantly enhance search efficiency. On Qwen2.5-3B-Instruct, DASH consistently outperforms a comprehensive suite of existing selector-style hybrid attention design baselines, showing that direct differentiable search can discover stronger hybrid architectures. Moreover, DASH achieves stronger RULER performance than released Jet-Nemotron models while remaining competitive on overlapping short-context and general benchmarks. Notably, each DASH search run uses only 12.3M tokens and takes about 20 minutes on a single RTX Pro 6000 GPU, corresponding to merely 0.006% of the PostNAS search tokens reported by Jet-Nemotron. These results suggest that high-quality hybrid attention architectures can be obtained through minutes-level differentiable search, providing a promising direction for hybrid architecture design. 

---
# Strategy-Induct: Task-Level Strategy Induction for Instruction Generation 

**Authors**: Po-Chun Chen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20924)  

**Abstract**: Designing effective task-level prompts is crucial for improving the performance of Large Language Models (LLMs). While prior work on instruction induction demonstrates that LLMs can infer better instructions with limited examples, existing approaches often rely on input-output pairs, where obtaining labeled answers can be difficult or costly. To address this limitation, we propose Strategy-Induct, a framework that derives task-level instructions solely from a small set of example questions without requiring labeled answers. Our approach first prompts the model to generate explicit reasoning strategies for each question, forming (strategy, question) pairs. These pairs are then used to induce a task instruction that guides reasoning. Experiments across multiple tasks and model scales demonstrate that Strategy-Induct outperforms state-of-the-art methods in question-only settings. Furthermore, we observe that jointly utilizing LLMs and Large Reasoning Models across task instruction generation and inference may lead to further performance improvements. 

---
# Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows 

**Authors**: Benedikt Bollig  

**Link**: [PDF](https://arxiv.org/pdf/2605.20923)  

**Abstract**: Distributed LLM agent workflows should not be monitored as if they produced a single sequential log. In an asynchronous execution, a decision can only depend on events that are causally visible to the lifeline that makes it: an event that appears earlier in some log may still be unknown locally. We extend the ZipperGen agent-workflow framework with Causal Past Logic (CPL), a small past-time temporal logic for guards in conditionals and while loops. In addition to standard past-time modalities such as previous and since, a guard can inspect the latest causally visible event of another lifeline and selected variables stored there. The formula is a source-level guard: it is evaluated online by the owner lifeline and can influence control flow at runtime. We give a vector-clock monitor with latest-value views and prove that the locally computed monitor value coincides with the denotational semantics of the guard at the current event. Thus runtime verification becomes part of the coordination language itself, rather than a post-hoc check over an execution log. 

---
# Winfree Oscillatory Neural Network 

**Authors**: Jiawen Dai, Yue Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.20922)  

**Abstract**: Oscillations and synchronization are widely believed to play a fundamental role in representation and computation. However, existing machine learning approaches based on synchronization dynamics have largely been confined to specialized settings such as object discovery, with limited evidence of scalability to standard vision benchmarks or logic reasoning tasks. We propose the Winfree Oscillatory Neural Network (WONN), a dynamical neural architecture based on generalized Winfree dynamics. WONN evolves representations on the torus $(S^1)^d$ through structured oscillatory interactions, combining phase-based inductive biases with flexible and hierarchical interaction mechanisms instantiated as either fixed trigonometric mappings or learnable neural networks. We evaluate WONN on image recognition and complex reasoning tasks, including CIFAR, ImageNet, Maze-hard, and Sudoku. Across these domains, WONN achieves competitive or superior performance with strong parameter efficiency. In particular, WONN is, to our knowledge, the first synchronization-based oscillatory architecture to scale competitively to ImageNet-1K. Furthermore, on Maze-hard, WONN achieves 80.1% accuracy using only 1% of the parameters of prior state-of-the-art models. These results suggest that structured oscillatory dynamics provide a scalable and parameter-efficient alternative to conventional neural architectures. 

---
# Sutra: Tensor-Op RNNs as a Compilation Target for Vector Symbolic Architectures 

**Authors**: Emma Leonhart  

**Link**: [PDF](https://arxiv.org/pdf/2605.20919)  

**Abstract**: Sutra is a typed, purely functional programming language whose compiled forward pass is a PyTorch neural network. The compiler beta-reduces the whole program -- primitives, control flow, string I/O -- to one fused tensor-op graph over a frozen embedding substrate. Rotation binding, unbind, bundle, polynomial Kleene three-valued logic, and tail-recursive loops all lower to tensor operations; the Kleene connectives are Lagrange-interpolated polynomials exact on the {-1, 0, +1} truth grid. Validation is one fact tested two ways. (1) The same program runs on four frozen embeddings spanning two modalities -- three text encoders (nomic-embed-text, all-minilm, mxbai-embed-large) and one protein language model (ESM-2) -- and decodes bundles at 100% accuracy through width k=8 on every substrate, where the textbook Hadamard product has already collapsed (2.5% on mxbai-embed-large, 7.5% on all-minilm). (2) PyTorch autograd flows through the actually compiled graph: a fuzzy-rule classifier written in .su trains from random init (18.7 +/- 9.5%; chance = 20%, five classes) to 100.0 +/- 0.0% (three seeds) by backpropagating through the emitted graph, the symbolic source unmodified. A weighted variant additionally trains a scalar cosine gain and writes it back into the .su source as a numeric literal; recompiling reproduces the trained behaviour to ~2e-7 per logit, so the trained model is itself legible, recompilable code. The same artifact is therefore both a logic program and a trainable neural network. 

---
# Calibration vs Decision Making: Revisiting the Reliability Paradox in Unlearned Language Models 

**Authors**: Divyaksh Shukla, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2605.20915)  

**Abstract**: Machine unlearning aims to remove the influence of specific training data from a model while preserving reliable behavior on the remaining data, making reliable prediction and uncertainty estimation essential for evaluation. Calibration is commonly used as a proxy for reliability in language models, but low calibration error does not necessarily imply reliable decision rules, as models may rely on spurious correlations while remaining well calibrated. We investigate this gap in generative language models using the multiple-choice question-answering evaluation protocol on the TOFU benchmark, measuring probabilistic reliability with calibration metrics (ECE, MCE, Brier) and decision-rule reliability via attribution-based shortcut detection with Integrated Gradients and Local Mutual Information. We find that fine-tuned models achieve low calibration error (ECE ~ 0.04) compared to pretrained models (ECE > 0.5), and models after unlearning retain similarly low calibration despite reduced accuracy on the forget split, while attribution analysis shows increased reliance on correlation-based tokens. These results demonstrate that good calibration can coexist with shortcut-based decision rules after unlearning, extending the reliability paradox to the machine unlearning setting. 

---
# VISTA: Technical Report for the Ego4D Short-Term Object Interaction Anticipation at EgoVis 2026 

**Authors**: Qiaohui Chu, Haoyu Zhang, Yisen Feng, Meng Liu, Weili Guan, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.20901)  

**Abstract**: We propose VISTA, a V-JEPA Integrated StillFast Temporal Anticipator for the Ego4D Short-Term Object Interaction Anticipation (STA) Challenge at EgoVis 2026. Given an egocentric video timestamp, the task requires anticipating the next human-object interaction, including the future active object's bounding box, noun category, verb category, time-to-contact, and confidence score. VISTA follows a StillFast-style design that combines object-centric spatial detection with short-horizon temporal context. Specifically, a COCO-pretrained Faster R-CNN ResNet-50 FPN detector generates object proposals from the last observed high-resolution frame, while a frozen V-JEPA 2.1 temporal branch extracts clip-level egocentric context from the observed video. The temporal representation is injected into the detection pathway through feature modulation and ROI-level context fusion. The fused proposal features are then passed to multi-head STA predictors for box refinement, noun classification, verb classification, time-to-contact regression, and interaction confidence estimation. For the final submission, we further ensemble complementary predictions to improve robustness. Experimental results on the official challenge server show that VISTA achieves first place in the EgoVis 2026 Ego4D STA Challenge. Our code will be released at this https URL. 

---
# GenAI-Driven Threat Detection with Microsoft Security Copilot 

**Authors**: Scott Freitas, Amir Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2605.20896)  

**Abstract**: Defending against today's increasingly sophisticated cyberattacks requires security analysts to continuously translate evolving attacker tradecraft into detection logic. This places defenders in a reactive posture, requiring constantly updated expertise across an increasingly fragmented security landscape. We introduce the Dynamic Threat Detection Agent (DTDA), an always-on adaptive agent that continuously investigates security incidents across Microsoft Defender to uncover hidden threats and generate explainable detections when attack-story gaps are found. DTDA combines: (1) a unified activity timeline spanning alerts, events, user and entity behavior analytics, and threat intelligence; (2) versioned LLM prompt contracts with schema validation, grounding requirements, bounded retries, and fail-closed suppression; (3) a planner-executor investigation loop that generates attack-specific hypotheses and gathers supporting and refuting evidence; and (4) dynamic alert generation with a context-relevant title, severity, MITRE mappings, remediation guidance, implicated entities, and natural-language attack description. Integrated into Microsoft Security Copilot and deployed across tens of thousands of Defender customers, DTDA operates continuously at industry scale. In a 120-day online evaluation, DTDA achieves 80.1% precision from customer feedback while generating novel alerts for approximately 15% of investigated incidents. In offline evaluation, DTDA recovers hidden malicious activity with 0.78 F1 using GPT-5.4, improving over GPT-4.1 by 0.12 F1 and outperforming the baseline by 0.26 F1 points. Operationally, DTDA processes single-incident investigations end-to-end in a median of 28 minutes at a median token cost of USD 2.04, with a 0.38% job-level failure rate. These results demonstrate that autonomous agents can identify missed malicious activity at a production scale. 

---
# Terminal-World: Scaling Terminal-Agent Environments via Agent Skills 

**Authors**: Zihao Cheng, Hongru Wang, Zeming Liu, Xinyi Wang, Xiangrong Zhu, Yuhang Guo, Wei Lin, Jeff Z. Pan, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20876)  

**Abstract**: Terminal agents extend Large Language Models with the ability to execute tasks directly in command-line environments, but their progress is bottlenecked by the scarcity of high-quality training data. Existing approaches bootstrap from partial sources such as human-defined seeds or GitHub repositories to instantiate one component and then complete the rest, producing tasks confined to narrow seed distributions, environments misaligned with task semantics, and inefficient trajectories from unguided exploration. To address these limitations, we introduce Terminal-World, a fully automated pipeline that uses agent skills as the central synthesis primitive, which jointly encode what to accomplish, when to apply (preconditions and environment state), and how to execute, enabling task instructions, environments, and teacher trajectories to be co-derived. To further broaden the synthesis space, Terminal-World composes skills into skill teams and skill graphs for multi-role and cross-domain task synthesis. Using this pipeline, we construct 5,723 training environments and train Terminal-World-8B/14B/32B, evaluated across 6 benchmarks where the Terminal-World series consistently outperforms terminal-agent baselines. Notably, using the same teacher model and only 1.2% of the training data, Terminal-World-32B surpasses Nemotron-Terminal-32B on Terminal-Bench 2.0 by +4.5 Pass@1 (31.5) and achieves 43.8 Pass@3. 

---
# CAdam: Context-Adaptive Moment Estimation for 3D Gaussian Densification in Generative Distillation 

**Authors**: SeungJeh Chung, Geonho Park, Misong Kim, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20872)  

**Abstract**: Adaptive densification is the engine of 3D Gaussian Splatting (3DGS). However, when transposed to the optimization-based Generative Distillation paradigm, this reconstruction-native mechanism reveals fundamental limitations, resulting in inefficient representations cluttered with redundant primitives. We diagnose this failure as a Densification Dilemma stemming from the stochastic nature of generative guidance: the standard magnitude-based accumulation indiscriminately aggregates transient noise alongside geometric signals, making it difficult to strike a balance between over-densification and under-fitting. To resolve this, we introduce Context-Adaptive Moment Estimation (CAdam), a novel framework that reinterprets densification as a statistically grounded signal verification problem. CAdam leverages the first moment of gradients to exploit the interference principle, where stochastic fluctuations cancel out via destructive interference while consistent geometric drifts accumulate via constructive interference, effectively disentangling the underlying signal from the generative noise floor. This is further augmented by a quantile-based context awareness and an intrinsic Signal-to-Noise Ratio (SNR) gating mechanism, which ensure robust adaptation across optimization stages and enable the soft termination of densification. Extensive experiments across diverse objectives (SDS, ISM, VFDS) and strong generative 3DGS backbones show that CAdam reduces Gaussian count by 85%-97% relative to standard densification while preserving overall comparable perceptual quality. These results highlight signal-aware density control as a practical way to improve memory efficiency in optimization-based generative distillation. 

---
# Runtime-Certified Bounded-Error Quantized Attention 

**Authors**: Dean Calver  

**Link**: [PDF](https://arxiv.org/pdf/2605.20868)  

**Abstract**: KV cache quantization reduces the memory cost of long-context LLM inference, but introduces approximation error that is typically validated only empirically. Existing systems rely on average-case robustness, with no mechanism to detect or recover from failures at runtime. We present a tiered KV cache architecture that enables runtime-certified attention: INT8 keys and INT4 values are stored in GPU memory, while FP16 originals are retained in system RAM for deterministic fallback. A two-term error decomposition yields per-head, per-step bounds on (i) attention distribution distortion from key quantization and (ii) value reconstruction error. These bounds are computed online and used to drive adaptive precision selection and a multi-stage fallback ladder, which guarantees recovery to the exact dense attention output when required. Across PG-19, NIAH, and RULER benchmarks on LLaMA~3.1-8B with contexts up to 128K, the system matches dense FP16 KV quality within noise for language modelling and retrieval tasks, while recovering catastrophic failures observed in naive INT8/INT4 baselines. Value-sensitive tasks at short context expose a controlled trade-off between compression and fidelity, which can be eliminated via tighter value tolerances or FP16-value fallback. The certification is local (per-head, per-step) and does not guarantee end-to-end model correctness, but ensures that each attention computation is either bounded relative to an FP16 reference or exactly recovered via fallback. This reframes KV cache quantization as a runtime-verified computation rather than a fixed approximation. The goal is not raw speedups, but enabling safe deployment of aggressive KV compression under strict quality constraints. 

---
# Multi-Step Likelihood-Ratio Correction for Reinforcement Learning with Verifiable Rewards 

**Authors**: Deokgyu Yoon, Hyungkyu Kang, Joongkyu Lee, Byeongchan Kim, Gyungin Shin, Sungrae Park, Min-hwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.20865)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) plays a pivotal role in improving the reasoning ability of large language models. However, widely used PPO surrogate objectives are fundamentally local, as they rely on a local approximation of the exact policy gradient objective. While this approximation improves stability by reducing the variance induced by importance sampling, it also introduces structural bias into the surrogate objective, which must be controlled through trust region mechanisms. In this work, we introduce the $N$-step forward trace, which augments the PPO surrogate objective using the cumulative likelihood ratio of the next $N-1$ tokens. Building on this idea, we propose $N$-Step Forward-Trace Policy Optimization (NFPO), a practical RLVR algorithm that integrates the $N$-step forward trace into the masked policy gradient framework. NFPO provides a continuous bridge between the PPO surrogate objective and the exact policy gradient objective, offering a principled mechanism for controlling the bias-variance trade-off. Our theoretical analysis shows that, with an appropriate choice of $N$, the proposed objective yields a tighter policy-improvement bound than the standard PPO surrogate. Experiments on comprehensive reasoning benchmarks demonstrate that NFPO consistently improves performance, supporting our theoretical findings. 

---
# DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation 

**Authors**: Hanxiang Ren, Pei Zhou, Xunzhe Zhou, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20856)  

**Abstract**: Language-conditioned manipulation policies typically process instructions and observations through shared network parameters. This task-state entanglement provides a pathway for observation leakage -- networks learn scene-to-action shortcuts that bypass language grounding entirely. DISC eliminates this failure structurally. Rather than conditioning a universal policy on language, DISC uses a hypernetwork to generate the entire parameter set of a task-specific visuomotor policy from the instruction alone. The generated policy never directly accesses language; therefore, its task-awareness must come from the language. Consequently, observation leakage has no pathway to emerge. On the other hand, generating coherent high-dimensional policy weights is itself a challenging problem. We address it with a two-stage hypernetwork whose refinement stage embeds the structure of gradient-based optimization as a feed-forward inductive bias, producing globally consistent parameters without actual gradient computation. Trained entirely from scratch on standard data budgets, DISC outperforms all entangled baselines on LIBERO-90 and Meta-World, with advantages that widen on complex, long-horizon tasks -- and surpasses the large-scale pretrained $\pi_0$ despite using no external pretraining data. On a real-world benchmark where all tasks share identical visual context, DISC substantially outperforms entangled alternatives, directly confirming that language-generated policy parameters, not visual shortcuts, drive behavior. The hypernetwork further learns a semantically structured parameter manifold that enables few-shot adaptation from minimal demonstrations and robust generalization across paraphrased instructions. Our code is available at: {this https URL}. 

---
# USV: Towards Understanding the User-generated Short-form Videos 

**Authors**: Haoyue Cheng, Su Xu, Liwei Jin, Wayne Wu, Chen Qian, Limin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20838)  

**Abstract**: Several large-scale video datasets have been published these years and have advanced the area of video understanding. However, the newly emerged user-generated short-form videos have rarely been studied. This paper presents USV, the User-generated Short-form Video dataset for high-level semantic video understanding. The dataset contains around 224K videos collected from UGC platforms by label queries without extra manual verification and trimming. Although video understanding has achieved plausible improvement these years, most works focus on instance-level recognition, which is not sufficient for learning the representation of the high-level semantic information of videos. Therefore, we further establish two tasks: topic recognition and video-text retrieval on USV. We propose two unified and effective baseline methods Multi-Modality Fusion Network (MMF-Net) and Video-Text Contrastive Learning (VTCL), to tackle the topic recognition task and video-text retrieval respectively, and carry out comprehensive benchmarks to facilitate future research. Our project page is this https URL. 

---
# ArchSIBench: Benchmarking the Architectural Spatial Intelligence of Vision-Language Models 

**Authors**: Qirui Shen, Wenda Wang, Jiachen Lu, Zilong Huang, Jin Bai, Lei He, Hongxuan Chen, Weixin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20837)  

**Abstract**: Architectural spatial intelligence, the ability to recognize and infer architectural space, is fundamental to tasks such as robot navigation, embodied interaction, and 3D scene understanding and generation. Although extensive research has evaluated the basic spatial skills of Vision-Language Models (VLMs) such as relative orientation, distance comparison, and object counting, these tasks cover only the most elementary levels of spatial cognition and largely overlook higher-level cognition of architectural space, including layout understanding, circulation patterns, and functional zoning. In this work, we present ArchSIBench, a Benchmark for Architectural Spatial Intelligence based on the perspectives from architecture, cognitive science, and psychology. ArchSIBench covers five core dimensions: perception, reasoning, navigation, transformation, and configuration, comprising 17 fine-grained subtasks. Through careful manual annotation by experts with architectural backgrounds, we construct 3,000 question-answer pairs to enable comprehensive evaluation of architectural spatial intelligence. Based on ArchSIBench, we evaluate various VLMs and find that the architectural spatial intelligence of most models shows significant differences from human baselines; additionally, models exhibit substantial variability across capability dimensions. Some state-of-the-art models can approach the level of human evaluators without architectural training. However, a clear gap remains compared to human evaluators with architectural training, particularly in spatial transformation and configuration reasoning. We believe that ArchSIBench will provide important insights and systematic resources for measuring and advancing the architectural spatial intelligence of VLMs. The dataset and code are available at this https URL. 

---
# GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval 

**Authors**: Peter Fernandes, Ria Kanjilal  

**Link**: [PDF](https://arxiv.org/pdf/2605.20815)  

**Abstract**: Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data is complex and strictly regulated, reliance on cloud-based large language models (LLMs) introduces challenges in cost, latency, and compliance. In this work, we present a systematic evaluation of GraphRAG for EHR schema retrieval using locally deployed open-source LLMs. We implement the Microsoft GraphRAG pipeline on real-world EHR schema documentation and benchmark four models, including Llama 3.1 (8B), Mistral (7B), Qwen 2.5 (7B), and Phi-4-mini (3.8B), each deployed via Ollama on a single consumer GPU (8 GB VRAM). We evaluate indexing efficiency, knowledge graph construction, query latency, answer quality, and hallucination under both global and local retrieval modes. Our results reveal substantial differences: Llama 3.1 produces the richest knowledge graph (1,172 entities), Qwen 2.5 achieves the best answer quality (3.3/5), Phi-4-mini fails to complete the pipeline due to structured-output errors, and Mistral exhibits degenerate repetition behavior. We further show that GraphRAG exhibits a practical capacity threshold, where models below approximately 7B parameters fail to reliably produce valid structured outputs and cannot complete the pipeline. In addition, indexing and answer quality are decoupled across models, and local retrieval consistently outperforms global summarization in both latency and factual grounding, with reduced hallucination. These findings demonstrate that GraphRAG is feasible on consumer hardware while highlighting the importance of model selection and retrieval design for robust deployment in regulated settings. 

---
# Tunable MAGMAX: Preference-Aware Model Merging for Continual Learning 

**Authors**: Kei Hiroshima, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2605.20803)  

**Abstract**: Continual learning (CL) aims to train models sequentially on multiple tasks while mitigating catastrophic forgetting of previously learned knowledge. Recent advances in large pre-trained models (LPMs) and model merging techniques, such as MAGMAX, have demonstrated effective CL performance by combining task-specific parameters. However, existing methods primarily focus on average performance across all tasks and do not adequately address how to construct models accommodating different deployment environments or varying user preferences. This paper proposes a model merging framework, termed Tunable MAGMAX, which enables preference-aware control of task-specific performance in CL. Our method introduces a preference vector that controls the number of elements selected from each task vector during model merging, allowing us to adjust the merged model performance according to their deployment needs. We further propose a method for automatically constructing appropriate preference vectors by leveraging small amounts of target environment data and datasets from model training tasks, thereby eliminating the need for manual specification. The experimental result on CL benchmark tasks demonstrates that Tunable MAGMAX effectively controls task-wise performance and successfully adapts merged models to various target environments. The proposed Tunable MAGMAX achieves superior or comparable performance to baseline methods, making it a practical solution for deploying CL models to various environments where the preferences of each task performance differ. 

---
# ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing 

**Authors**: Kang You, Chen Nie, Lee Jun Yan, Ziling Wei, Cheng Zou, Zekai Xu, Yu Feng, Honglan Jiang, Zhezhi He  

**Link**: [PDF](https://arxiv.org/pdf/2605.20802)  

**Abstract**: Spiking neural networks (SNNs) exploit event-driven and addition-only computation to substantially improve efficiency for intelligent computation. A key temporal property of SNNs, elastic inference, allows outputs to emerge progressively, enabling responses to salient inputs much earlier than full evaluation. However, existing SNN-specific accelerators cannot capitalize on this property. Layer-by-layer designs emit outputs only after all layers are complete, while time-step-by-time-step designs rely on coarse-grained, layer-wise pipelines that require synchronizing all spines/tokens within a layer. This barrier prevents results from being forwarded immediately, delaying the earliest possible response and forfeiting the benefits of elastic inference.
To address these challenges, we propose ELSA, a near-SRAM dataflow architecture that realizes true elastic inference through a fine-grained spine/token-wise pipeline and hardware optimizations tailored to SNNs. ELSA forwards each spine/token immediately upon production, forming a continuous streaming pipeline that substantially reduces the latency to the first response. To enhance this lightweight execution, ELSA introduces a bundled address event representation protocol to lower communication traffic of network-on-chip (NoC), and leverages mini-batch spiking Gustavson-product to cut memory access and exploit inherent sparsity. Combined with mapping and scheduling optimizations, ELSA achieves efficient, event-driven computation without compromising accuracy. Experiments show that SNNs can outperform quantized artificial neural networks (QANNs) while maintaining on-par accuracy. For a 4-bit ResNet-50, ELSA achieves 3.4$\times$ speedup and 13.6$\times$ higher energy efficiency over the SOTA QANN accelerator (ANT), and 2.9$\times$ speedup and 22.1$\times$ energy efficiency gains over the SOTA SNN accelerator (PAICORE). 

---
# Correcting Stochastic Update Bias in Preconditioned Language Model Optimizers 

**Authors**: Nikhil Nayak, Julia White, Urchade Zaratiana, Kelton Zhang, Henrijs Princis, Dhruv Atreja, Henry Fawcett, Matthew Thomas, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2605.20756)  

**Abstract**: Preconditioned optimizers are central to language model training, but their stochastic update rules are usually treated as direct approximations to population preconditioned descent. We show that this view misses two finite-sample biases. First, the gradient and preconditioner are typically estimated from the same minibatch, introducing gradient--preconditioner coupling bias. Second, even when the preconditioner estimate is unbiased, its inverse or inverse-root is generally biased because inversion is nonlinear. We propose a single-batch bias-correction framework that addresses both effects: cross-fitted preconditioning estimates the numerator and preconditioner from independent microbatch groups, while variance-corrected inversion uses microbatch variability to subtract the leading delta-method bias term. The framework applies to diagonal moment, diagonal curvature, and matrix preconditioning methods, instantiated in AdamW, Sophia, and Shampoo. Bias correction reduces held-out pretraining loss on Qwen2.5-0.5B by $0.15$, $0.07$, and $0.11$ nats, respectively; the effects on mixed-quality pretraining and downstream instruction tuning are consistently neutral-to-positive. Together, these results establish bias correction as a practical mechanism for reducing finite-sample update bias and improving the performance of preconditioned optimizers. 

---
# PACD-Net: Pseudo-Augmented Contrastive Distillation for Glycemic Control Estimation from SMBG 

**Authors**: Canyu Lei, David Repaske, Jianxin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.20751)  

**Abstract**: Effective diabetes management requires continuous monitoring of glycemic levels. Clinically, glycemic control is assessed using metrics such as Time in Range (TIR), Time Below Range (TBR), and Time Above Range (TAR), typically derived from continuous glucose monitoring (CGM). However, many patients rely on self-monitoring of blood glucose (SMBG) due to the high cost and limited accessibility of CGM. Unlike CGM, SMBG provides sparse and irregular measurements, making accurate estimation of these metrics challenging. Conventional supervised learning approaches struggle under such sparsity, leading to poor generalization and unstable performance. To address this, we propose PACD-Net, a self-supervised contrastive knowledge distillation framework for estimating glycemic control from SMBG. Pseudo-SMBG samples with richer temporal coverage are used as teacher signals to guide learning from sparse observations. In addition, multi-view contrastive learning enforces representation consistency across diverse sampling patterns. The model adopts a hybrid Swin Transformer-CNN backbone to capture temporal dependencies in sparse SMBG sequences. Experimental results demonstrate that PACD-Net consistently outperforms existing methods in estimating TAR, TIR, and TBR from real-world SMBG data, achieving improved accuracy as well as enhanced stability and generalization under extremely sparse observation settings. The proposed framework provides a practical tool for clinical SMBG interpretation and offers a generalizable approach for learning from sparse and irregularly sampled sensor data in broader applications. 

---
# The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure? 

**Authors**: Xingyu Lyu, Qianqian Xu, Zhiyong Yang, Peisong Wen, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20749)  

**Abstract**: Gated Linear Units (GLU) and their variants are widely adopted in modern open-source large language model architectures and consistently outperform their non-gated counterparts, yet the underlying reasons for this advantage remain unclear. In this work, we study GLU by analyzing two-layer networks in the neural tangent kernel (NTK) regime. Our analysis reveals that the GLU structure reshapes the NTK spectrum, leading to a smaller condition number and a more compact eigenvalue distribution. Building on this finding, we further analyze the resulting training dynamics and show how the reshaped spectrum leads to faster convergence of GLU models, including a characteristic loss-crossing phenomenon observed between GLU and non-GLU models. Finally, we empirically observe that GLU has limited impact in reducing the generalization gap on various models, including ViT and GPT-2, suggesting that its primary benefit lies in accelerating optimization rather than reducing the generalization gap. 

---
# The Hidden Signal of Verifier Strictness: Controlling and Improving Step-Wise Verification via Selective Latent Steering 

**Authors**: Yefan Zhou, Yilun Zhou, Austin Xu, Soroush Vosoughi, Shafiq Joty, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2605.20745)  

**Abstract**: Generative verifiers have emerged as a promising paradigm for step-wise verification, but their verification behavior is often poorly calibrated: they may be under-critical and miss erroneous steps, or over-critical and reject correct reasoning. We refer to this tendency to be overly lenient or overly critical as verifier strictness. In this work, we study whether verifier strictness can be controlled through hidden-state intervention. We uncover a verification-specific hidden-state signal: in step-wise verification, a verifier's tendency to accept or reject a solution step is encoded near the boundary of the corresponding verification paragraph. Exploiting this signal, we show that hidden-state steering can directly modulate verifier strictness without fine-tuning. However, uniform steering induces a trade-off between error detection and correctness certification. To address this, we propose VerifySteer, which exploits latent correctness signals for sample-level routing and selectively intervenes on paragraph boundaries. Experiments on ProcessBench and Hard2Verify show that VerifySteer outperforms prompt optimization and activation steering baselines, and is competitive with self-consistency while requiring 4-7x less inference compute. VerifySteer is also complementary to verification fine-tuning, providing further gains on top of fine-tuned verifiers. The code is available at this https URL. 

---
# Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale 

**Authors**: Amit Roth, Ankur Samanta, Matan Halevy, Yoav Levine, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2605.20744)  

**Abstract**: Aligning autonomous agents with human intent remains a central challenge in modern AI. A key manifestation of this challenge is reward hacking, whereby agents appear successful under the evaluation signal while violating the intended objective. Reward hacking has been observed across a wide range of settings, yet methods for reliably measuring it at scale remain lacking. In this work, we introduce a new evaluation paradigm for measuring reward hacking. Whereas prior studies have primarily analyzed it post hoc by inspecting agent trajectories, we instead embed detectable reward hacking opportunities directly into environments. This makes their exploitation verifiable by design, enabling deterministic and automated measurement of whether and how agents exploit such vulnerabilities. We instantiate this approach in $\textit{TextArena}$ and release $\textit{Hack-Verifiable TextArena}$, a testbed in which reward hacking can be measured reliably. Using this benchmark, we analyze reward hacking behavior across language models in diverse environments and settings. We open source the code at this https URL. 

---
# Distribution-Aware Reward: Reinforcement Learning over Predictive Distributions for LLM Regression 

**Authors**: Jungsoo Park, Hyungjoo Chae, Ethan Mendes, Jay DeYoung, Varsha Kishore, Wei Xu, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2605.20740)  

**Abstract**: Large language models can predict real-valued quantities from heterogeneous inputs such as text, code, and molecular strings, but most training objectives score each decoded floating-point number independently, improving point estimates without ensuring calibrated predictive distributions. This limits applications requiring candidate ranking or uncertainty estimation. We introduce Distribution-Aware Reward, an on-policy reinforcement learning objective whose main contribution is to train language models to produce better predictive distributions for regression tasks, rather than only optimizing individual decoded outputs against scalar targets. Our method treats multiple decoded samples as an empirical predictive distribution, evaluates it with the Continuous Ranked Probability Score, and assigns leave-one-out credit based on each rollout's marginal contribution to distribution quality, rewarding predictions that are both accurate and appropriately dispersed. We evaluate our method on a controlled Gaussian-mixture task, code performance prediction, and molecular property prediction from SMILES strings. Across tasks, our method improves over supervised fine-tuning and pointwise reinforcement learning baselines, with strong rank-correlation gains, including a 6-point Spearman improvement on KBSS. On MoleculeNet, it uses only SMILES strings yet remains competitive with strong graph-based and 3D molecular models. Further analyses show that our method mitigates rollout diversity collapse and improves uncertainty diagnostics, suggesting that directly optimizing predictive distributions makes language model regression more robust and better calibrated. 

---
# An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.20734)  

**Abstract**: A large language model (LLM) agent that sends messages can leak data inside them. Destination allowlists and content scanners do not police whether an otherwise-benign payload is itself a covert channel: a compromised agent encodes bits in zero-width characters, homoglyphs, whitespace, base64, JavaScript Object Notation (JSON) key ordering, message timing or size -- and, in binary egress, in least-significant-bit (LSB) pixel planes, per-image mean luminance, inter-image sequence permutation, ultrasonic tones, or audible-band sonified data. Our egress reference monitor has three contributions. (i) A text pipeline of ten capacity-reducing stages, a per-sink leaky-bucket capacity ledger, and a staged posture that enforces lossless stages from day one. (ii) Two media scramblers (a Fourier-domain audio band-limiter and a red-green-blue (RGB) image bit-depth and mean-luminance bucketer) gated by a boot-time cryptographic legitimacy attestation: an auditor publishes at boot the trusted Ed25519 keys and {kind, data-class} pairs; only payloads with a verifying signature for an authorized class are exempt. The attestation sidesteps the intractable content-based discrimination between real media and data sonified or rasterized as a carrier; unsigned media is suspect by default; a content-addressed canonicalizer closes the inter-image permutation channel. (iii) Residual capacity is the Miller--Madow corrected mutual information between embedded and recovered bits (zero when destroyed), measured by an adversarial ensemble of fifteen working encoders across text, image and audio. The reference implementation drives residual capacity to zero on every destroyable channel and to a stated bound on the one (per-image mean luminance) that cannot be destroyed without ruining the image. 

---
# TASTE: A Designer-Annotated Multi-Dimensional Preference Dataset for AI-Generated Graphic Design 

**Authors**: Haonan Zhu, Elad Hirsch, Alexandria Minetti, Allison Nulty, Purvanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2605.20731)  

**Abstract**: Text-to-image models produce graphic design at production scale, but their supervision comes from photo-style preference data with a single overall verdict per comparison. Designers evaluate along several distinct axes, including typography, visual hierarchy, color harmony, layout, and brief fidelity, and a single label collapses them. We release TASTE (Typography, Aesthetics, Spatial, Tone, Etc.): ten professional designers ranked outputs from four current text-to-image models on nine criteria across two disjoint cohorts, yielding 1,600 ratings per criterion plus per-image hallucination flags on the holistic-preference cohorts. We pair the dataset with three contributions. First, a criterion-agnostic signal test framework, using Kendall's tau, majority probability, and Condorcet cycles against exact iid-uniform nulls at p = 4 and R = 5, places designer agreement on graphic design between food and movie preferences and photo-style image quality, with every TASTE criterion rejecting the random-rater null. Second, no pre-trained system in our benchmark, including six open-weight VLM judges from 3B to 33B parameters and three dedicated T2I scorers, HPSv2.1, PickScore-v1, and LAION-Aesthetic-V2, exceeds 0.55 macro agreement with the 5-designer majority; VLM judges trade off position bias against content sensitivity, so scaling moves along this frontier without improving accuracy. Third, a small pairwise-difference head trained on TASTE reaches 0.611, closing roughly half the gap to the 0.741 single-rater ceiling. 

---
# Distributional Alignment as a Criterion for Designing Task Vectors in In-Context Learning 

**Authors**: Jihoon Kwon, Jiwon Choi, Jy-yong Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2605.20730)  

**Abstract**: In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks through demonstrations, yet it suffers from escalating inference costs as context length increases. While task vectors offer a promising alternative by compressing demonstrations into compact hidden-state representations, their quality has been evaluated only through downstream task accuracy. This indirect criterion provides limited insight into how to design more effective task vector extraction methods. In this paper, we posit that inference using task vectors should align their predictive distribution with that of ICL. To quantify this, we introduce $d_{\text{NTP}}$, a metric that measures the discrepancy in next-token probabilities between task vector-based and ICL-based inference. Our empirical analysis reveals that $d_{\text{NTP}}$ serves as a performance proxy, exhibiting a strong negative correlation with downstream accuracy. Motivated by this, we develop Linear Task Vector (LTV), a method designed to minimize $d_{\text{NTP}}$ via a closed-form linear mapping that estimates demonstration effects through regression. Across eight classification benchmarks and five LLMs, LTV consistently outperforms existing task vector baselines, improving average accuracy by 9.2\% while reducing inference latency. We further show that LTV outperforms the baselines on regression tasks. Moreover, we investigate the transferability of LTV across different model scales; an aspect that has remained nascent in task vector research. Specifically, we empirically show that task vectors from a larger model can enhance a smaller model's performance by 6.4\%, suggesting a new utility for extracted task representations. 

---
# AGPO: Adaptive Group Policy Optimization with Dual Statistical Feedback 

**Authors**: Miaobo Hu, Shuhao Hu, Bokun Wang, Ruohan Wang, Xin Wang, Xiaobo Guo, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20722)  

**Abstract**: Reinforcement learning improves LLM reasoning, but PPO/GRPO typically use fixed clipping and decoding temperature, which makes training brittle and tuning-heavy. We propose Adaptive Group Policy Optimization (AGPO), a critic-free refinement of GRPO that uses group-level statistics to control both update magnitude and exploration. AGPO uses a shared probe-derived statistical state to drive two controllers: (i) adaptive clipping, which sets the trust-region size from reward dispersion and skewness, probe vote entropy, policy entropy, and step-wise KL drift; and (ii) bidirectional adaptive temperature sampling, which heats or cools decoding around a base temperature according to centered uncertainty relative to a running baseline. On nine English and Chinese math/STEM benchmarks, Qwen2.5-14B trained with AGPO outperforms PPO/GRPO under the same generated-token budget, reaching 67.3% on GSM8K and 40.5% on MATH. Gains transfer to Llama-3-8B and Gemma-2-9B, and ablations confirm both modules are complementary. Our implementation is publicly available at this https URL. 

---
# SAVER: Selective As-Needed Vision Evidence for Multimodal Information Extraction 

**Authors**: Miaobo Hu, Shuhao Hu, Bokun Wang, Rui Chen, Xin Wang, Xiaobo Guo, Daren Zha, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20713)  

**Abstract**: Multimodal IE in social media is difficult because a post may attach multiple images that are weakly related, redundant, or even misleading with respect to the text. In this setting, always-on multimodal fusion wastes computation and can amplify spurious visual cues. The core challenge is to decide, for each candidate span or marked entity pair, whether vision should be consulted at all and, if so, which small subset of images provides trustworthy evidence.
We propose SAVER, a selective vision-as-needed framework for multimodal named entity recognition and multimodal relation extraction. SAVER uses a Conformal Groundability Gate (CGG) to estimate span-level visual groundability in MNER, derive pair-level activation in MRE from the two marked entities, and calibrate the activation threshold on a held-out split via a conformal-style procedure with Clopper--Pearson upper bounds. When activated, a submodular relevance--diversity selector chooses a compact evidence subset across images, which is then aggregated by a Set Transformer. An energy-inspired joint scoring head combines text, optional visual evidence, text--image consistency, and sparse routing for entity typing or relation classification.
Experiments show that SAVER consistently improves F1 over strong text-only and always-on multimodal baselines, while reducing AURC, increasing activation coverage at a fixed risk level, and lowering FLOPs and P90 latency. 

---
# SCRIBE: Diagnostic Evaluation and Rich Transcription Models for Indic ASR 

**Authors**: Kavya Manohar, Arghya Bhattacharya, Kush Juvekar, Kumarmanas Nethil  

**Link**: [PDF](https://arxiv.org/pdf/2605.20712)  

**Abstract**: Automatic speech recognition replaces typing only when correction costs less than manual entry, a threshold determined by error types, not counts: fixing a misrecognized domain term costs far more than inserting a comma. Word error rate (WER) fails on two fronts: it collapses distinct error categories into a single scalar, and it structurally penalizes agglutinative languages where valid sandhi merges inflate scores. We introduce SCRIBE, a diagnostic framework that provides categorical error decomposition into lexical, punctuation, numeral, and domain-entity rates through sandhi-tolerant alignment with domain vocabulary injection. Human validation confirms SCRIBE aligns with expert judgment where WER does not. We release SCRIBE, an LLM curation pipeline, benchmarks, and open-weight rich transcription models for Hindi, Malayalam, and Kannada. 

---
# Rethinking Cross-Layer Information Routing in Diffusion Transformers 

**Authors**: Chao Xu, Maohua Li, Qirui Li, Yixuan Xu, Yanke Zhou, Yunhe Li, Cuifeng Shen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Shao-Qun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20708)  

**Abstract**: Diffusion Transformers (DiTs) have become a de facto backbone of modern visual generation, and nearly every major axis of their design -- tokenization, attention, conditioning, objectives, and latent autoencoders -- has been extensively revisited. The residual stream that governs how information accumulates across layers, however, has been directly inherited from the original Transformer. In this paper, we present a systematic empirical analysis of cross-layer information flow in DiTs, jointly along depth and denoising timestep, and identify three concrete symptoms of traditional residual addition, namely monotonic forward magnitude inflation, sharp backward gradient decay, and pronounced block-wise redundancy. Motivated by this diagnosis, we propose Diffusion-Adaptive Routing (\textsc{DAR}), a drop-in residual replacement that performs \emph{learnable, timestep-adaptive, and non-incremental} aggregation over the history of sublayer outputs. Moreover, the proposed \textsc{DAR} is compatible with many modern Transformer enhancement methods, such as REPA. On ImageNet $256\times256$, \textsc{DAR} improves SiT-XL/2 by $2.11$ FID ($7.56$ vs.\ $9.67$) and matches the baseline's converged quality with $8.75\times$ fewer training iterations. Stacked on top of REPA, it yields a $2\times$ training acceleration in the early stage, suggesting cross-layer information routing as an underexplored design axis in diffusion modeling, one that operates orthogonally to existing representation-alignment objectives. Beyond pretraining, \textsc{DAR} can also be applied during the fine-tuning stage of large-scale T2I models and preserves high-frequency details during Distribution Matching Distillation. 

---
# Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU 

**Authors**: Reese Levine, Rithik Sharma, Nikhil Jain, Abhijit Ramesh, Zheyuan Chen, Neha Abbas, James Contini, Tyler Sorensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20706)  

**Abstract**: Running language models in the browser presents a unique opportunity to build efficient, private, and portable AI applications, but requires contending with constrained memory availability and heterogeneous hardware targets. To realize this opportunity, we present Llamas on the Web (LlamaWeb), a WebGPU backend for this http URL that enables memory-efficient and performance-portable LLM inference across a wide range of model weight formats in the browser. Our design significantly reduces memory overhead through static memory planning and efficient model loading, addresses cross-device variability through a tunable kernel library, and introduces templated GPU kernels that support performant implementations of numerous quantization formats, enabling broad model support and extensibility to new formats.
We evaluate LlamaWeb on 16 devices from 8 vendors, collecting data from 10 language models and four model weight formats. We compare LlamaWeb against existing browser-based LLM frameworks and find that LlamaWeb requires 29-33% less memory across several combinations of device, browser, and operating system. We also evaluate LlamaWeb's performance against these frameworks and find that it increases decode throughput by 45-69% across four GPUs from separate vendors. In addition, we compare LlamaWeb's performance against other this http URL backends, where it is competitive with and even beats vendor-specific backend performance on some devices. 

---
# Heartbeat-Bound Hierarchical Credentials: Cryptographic Revocation for AI Agent Swarms 

**Authors**: Saurabh Deochake  

**Link**: [PDF](https://arxiv.org/pdf/2605.20704)  

**Abstract**: Autonomous AI agents that spawn sub-agent swarms create a safety gap: existing credential revocation mechanisms, OAuth~2.0 introspection, OCSP, and W3C Status Lists, require network connectivity to a central authority, leaving ``zombie agents'' executing privileged operations for minutes to hours after operator shutdown. We present Heartbeat-Bound Hierarchical Credentials (HBHC), a cryptographic protocol that binds credential validity to periodic parent liveness proofs. Verifiers enforce freshness using only a cached public key and local clock; no network round-trip is required. When heartbeat generation ceases, all descendant credentials become unusable within a deterministically bounded window $W_z \le W_{\max} + \Delta_h + \epsilon$, conditional on bounded clock skew and parent keys held in secure enclaves. Evaluation at the protocol layer and with real LLM-backed agent swarms (GPT-4o-mini) demonstrates a 90$\times$ reduction in the zombie window over OAuth~2.0, 0.26~ms full authentication in Rust, 18,000+ verifications per second under concurrent HTTP load, and stable per-verification latency from 10 to 10,000 agents. Real-agent experiments show 0.71\% end-to-end overhead on tool calls, zero post-revocation tool calls under prompt injection that bypasses application-layer guardrails, and cascading revocation across a 49-agent four-level hierarchy within the theoretical bound. 

---
# Interpretable Discriminative Text Representations via Agreement and Label Disentanglement 

**Authors**: Tong Wang, Yiqing Xu, Leo Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20693)  

**Abstract**: Interpretable text representations should expose coordinates that are not only predictive, but also meaningful enough for independent auditors to apply. Existing discriminative representations often use anonymous embedding directions, while concept-bottleneck and LLM-assisted methods attach natural-language names to features without ensuring that those definitions are reproducible or distinct from the target label. We propose an operational criterion for interpretable discriminative text representations: each coordinate should satisfy conceptual clarity, measured by chance-adjusted agreement between independent annotators applying the feature definition, and label disentanglement, meaning the feature should not merely paraphrase the prediction target. We instantiate this criterion in LLM-assisted Feature Discovery (LFD), an iterative method that proposes lexical and semantic features from contrastive outcome-opposed text pairs, screens candidates using cross-LLM Cohen's $\kappa$, and selects features by residual held-out predictive gain. A stylized analysis connects the $\kappa$ screen to a per-feature annotation-noise bound, formalizing agreement as a reliability check. Across ten text-classification tasks spanning seven corpora, LFD matches the predictive performance of a strong text bottleneck baseline while producing substantially clearer and less label-entangled features. Human audits with 232 raters show that LFD features achieve higher human--human and human--LLM agreement than baseline concepts, and raters consistently judge them as less label-leaking. These results suggest that agreement-tested, label-disentangled coordinates provide a practical auditability standard for interpretable text classification. 

---
# DIVE: Embedding Compression via Self-Limiting Gradient Updates 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20689)  

**Abstract**: High-dimensional embeddings from large language models impose significant storage and computational costs on vector search systems. Recent embedding compression methods, including Matryoshka-Adaptor (EMNLP 2024), Search-Adaptor (ACL 2024), and SMEC (EMNLP 2025), enable dimensionality reduction through lightweight residual adapters, but their training objectives cause severe overfitting when labeled data is scarce, degrading retrieval performance below the frozen baseline. We propose \textsc{DIVE} (\textbf{D}imensionality reduction with \textbf{I}mplicit \textbf{V}iew \textbf{E}nsembles), a compression adapter that addresses this failure through two mechanisms. First, a self-limiting hinge-based triplet loss produces zero gradient once a triplet satisfies the margin constraint, bounding the total perturbation applied to the pretrained embedding space. Second, a head-wise NT-Xent contrastive loss treats multiple learned projections of each embedding as implicit views, providing dense self-supervised gradients that compensate for the sparsity of the triplet signal on small datasets. Across six BEIR datasets, \textsc{DIVE} outperforms all three baseline adapters on every dataset and at every evaluated compression ratio, with a 14M-parameter open-source implementation. 

---
# Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting 

**Authors**: Jiawen Zhu, Shuhan Liu, Di Weng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20678)  

**Abstract**: Non-stationary time series forecasting is challenged by evolving distribution shifts that static models struggle to capture. While Mixture-of-Experts (MoE) architectures offer a promising paradigm for decoupling complex drift patterns, existing approaches are limited by fixed expert pools and memoryless routing, hampering their ability to adapt to abrupt regime shifts. To address this, we propose Dynamic TMoE, a framework that unifies architectural evolution with temporal continuity during learning phase. By detecting distribution shifts via Maximum Mean Discrepancy (MMD), we dynamically instantiate heterogeneous experts and prune redundant ones to optimize capacity. Additionally, a temporal memory router leverages recurrent states and an anomaly repository to ensure stable, context-aware expert selection without requiring test-time updates. Experiments on nine benchmarks demonstrate state-of-the-art performance, reducing MSE by 10.4% and MAE by 7.8%. Code is available at this https URL. 

---
# On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists 

**Authors**: Seungone Kim, Dongkeun Yoon, Kiril Gashteovski, Juyoung Suk, Jinheon Baek, Pranjal Aggarwal, Ian Wu, Viktor Zaverkin, Spase Petkoski, Daniel R. Schrider, Ilija Dukovski, Francesco Santini, Biljana Mitreska, Yong Jeong, Kyeongha Kwon, Young Min Sim, Dragana Manasova, Arthur Porto, Biljana Mojsoska, Makoto Takamoto, Marko Shuntov, Ruoqi Liu, Hyunjoo Jenny Lee, Niyazi Ulas Dinç, Yehhyun Jo, Sunkyu Han, Chungwoo Lee, Huishan Li, Esther H. R. Tsai, Ergun Simsek, Khushboo Shafi, Yeonseung Chung, Jihye Park, Aleksandar Shulevski, Henrik Christiansen, Yoosang Son, Elly Knight, Amanda Montoya, Jeongyoun Ahn, Christian Langkammer, Heera Moon, Changwon Yoon, Nikola Stikov, Mooseok Jang, Edward Choi, Junhan Kim, Yeon Sik Jung, Woo Youn Kim, Jae Kyoung Kim, Ishraq Md Anjum, Hyun Uk Kim, Drew Bridges, Carolin Lawrence, Xiang Yue, Alice Oh, Akari Asai, Sean Welleck, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2605.20668)  

**Abstract**: With the advancement of AI capabilities, AI reviewers are beginning to be deployed in scientific peer review, yet their capability and credibility remain in question: many scientists simply view them as probabilistic systems without the expertise to evaluate research, while other researchers are more optimistic about their readiness without concrete evidence. Understanding what AI reviewers do well, where they fall short, and what challenges remain is essential. However, existing evaluations of AI reviewers have focused on whether their verdicts match human verdicts (e.g., score alignment, acceptance prediction), which is insufficient to characterize their capabilities and limits. In this paper, we close this gap through a large-scale expert annotation study, in which 45 domain scientists in Physical, Biological, and Health Sciences spent 469 hours rating 2,960 individual criticisms (each targeting one specific aspect of a paper) from human-written and AI-generated reviews of 82 Nature-family papers on correctness, significance, and sufficiency of evidence. On a composite of all three dimensions, a reviewing agent powered by GPT-5.2 scores above each paper's top-rated human reviewer (60.0% vs. 48.2%, p = 0.009), while all three AI reviewers (including Gemini 3.0 Pro and Claude Opus 4.5) exceed the lowest-rated human across every dimension. AI reviewers' accurate criticisms are also more often rated significant and well-evidenced, and surface a distinct 26% of issues no human raises. However, AI reviewers overlap far more than humans do (21% vs. 3% for cross-reviewer pairs), and exhibit 16 recurring weaknesses humans do not share, such as limited subfield knowledge, lack of long context management over multiple files, and overly critical stance on minor issues. Overall, our results position current AI reviewers as complements to, not substitutes for, human reviewers. 

---
# REFLECTOR: Internalizing Step-wise Reflection against Indirect Jailbreak 

**Authors**: Jiachen Ma, Jiawen Zhang, Xiangtian Li, Bo Zou, Chaochao Lu, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20654)  

**Abstract**: While Large Language Models (LLMs) demonstrate remarkable capabilities, they remain susceptible to sophisticated, multi-step jailbreak attacks that circumvent conventional surface-level safety alignment by exploiting the internal generation process. To address these vulnerabilities, we propose Reflector, a principled two-stage framework that internalizes self-reflection within the generation trajectory. Reflector first leverages teacher-guided generation to produce high-quality reflection data for supervised fine-tuning (SFT), establishing structured reflection patterns. It subsequently uses Reinforcement Learning (RL) with outcome-driven and reward-validity supervision to instill robust, autonomous self-reflection capabilities. Empirical results show that Reflector achieves Defense Success Rates (DSR) exceeding 90% against complex indirect attacks while generalizing robustly across diverse threat scenarios. Notably, the framework enhances both task-specific and general utility, yielding a 5.85% gain on GSM8K alongside improved performance on knowledge-intensive benchmarks. By internalizing trajectory-level safety, Reflector overcomes the fundamental limitations of surface alignment without significant computational overhead, offering an efficient and scalable solution for the development of safe and capable LLMs. 

---
# AMAR: Lightweight Attention-Based Multi-User Activity Recognition from Wi-Fi CSI 

**Authors**: Amirhossein Mohammadi, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2605.20649)  

**Abstract**: Wi-Fi-based human activity recognition (HAR) has emerged as a promising approach for contactless sensing, leveraging channel state information (CSI) collected from wireless transceivers. While existing studies have primarily concentrated on single-user scenarios, real-world deployments often involve multi-user settings where concurrent users' movements induce overlapping CSI patterns that challenge conventional classification methods. To address this limitation, this paper introduces an attention-based multi-user activity recognition (AMAR) framework that formulates HAR as a set prediction problem. The transformer-based architecture in AMAR leverages learnable query embeddings acting as specialized activity detectors, enabling the simultaneous identification of multiple activities from composite CSI representations. Moreover, to address deployment constraints, AMAR is designed in an edge-cloud split architecture form where lightweight convolutional networks on edge devices perform initial feature extraction, followed by residual vector quantization that achieves substantial bandwidth reduction while preserving activity-discriminative information. The cloud component performs final activity prediction through attention-based set matching, enabling the system to handle varying occupancy levels. Across classroom, meeting-room, and empty-room environments, on average AMAR nearly doubles the rate of perfectly predicting all concurrent activities compared to the best baseline. Moreover, it achieves an $F_1$-score of 53.4% compared to 45.6% for the best benchmark, and reduces occupancy estimation error by 74%, while minimizing bandwidth substantially. 

---
# Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition 

**Authors**: Benedict Quartey, Sebastian Castro, Eric Rosen, Wil Thomason, George Konidaris, Stefanie Tellex  

**Link**: [PDF](https://arxiv.org/pdf/2605.20648)  

**Abstract**: Learning from Demonstration (LfD) enables robots to learn complex behaviors from expert examples, yet existing approaches often fail to generalize to new compositions of known skills without retraining. Modern generative policies model distributions over action trajectories alone, thus are unable to reason about the symbolic outcomes required for robust composition. We propose that skills should jointly model action trajectories and the symbolic outcomes they induce. To address this gap, we introduce Predicate Action Skills (PACTS), a class of closed-loop visuomotor policies that model skills as a joint generative process over action and predicate belief trajectories, producing coherent action-outcome rollouts within a single model. Jointly generating actions and predicates enables PACTS to learn internal representations that improve both action generation and predicate classification. Furthermore, we demonstrate zero-shot composition of learned skills via planning by leveraging online predicate predictions from PACTS as a symbolic interface for sequencing and monitoring execution. Project website: this https URL 

---
# Design for Manufacturing: A Manufacturability Knowledge-Integrated Reinforcement Learning Framework for Free-Form Pipe Routing in Aeroengines 

**Authors**: Caicheng Wang, Zili Wang, Shuyou Zhang, Yongzhe Xiang, Zheyi Li, Liangyou Li, Jianrong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20644)  

**Abstract**: Design for manufacturing plays a critical role in advanced aeroengine development, where complex components necessitate careful consideration of manufacturability. However, current practices in pipe routing remain largely decoupled from down-stream manufacturing, leading to labor-intensive, trial-and-error iterations to achieve manufacturable designs. To address this problem, this study proposes the Frenet-based pipe routing optimization (FPRO) framework, a manufacturability knowledge-integrated reinforcement learning approach for free-form pipe design in aeroengines. FPRO formulates the routing problem as a boundary value problem in the Frenet frame. In this framework, the pipe path is represented by curvature and torsion profiles, which are generated using cubic Hermite interpolation. To integrate design and manufacturing, domain-specific manufacturing knowledge is embedded as constraints on the permissible ranges of curvature and torsion. The path optimization is performed using the proximal policy optimization algorithm with stochastic exploration and a stage-guided reward mechanism. A unified mapping formulation then translates the optimized path into motion trajectories for the bending die, enabling direct fabrication on a six-axis free-bending machine. Experimental results demonstrate that FPRO consistently generates collision-free, manufacturable paths with smoother geometric profiles compared to Cartesian-based methods. It also achieves faster convergence and superior performance in terminal alignment, path length, obstacle avoidance, and manufacturability compared to state-of-the-art reinforcement learning baselines. Real-world validation confirms the close geometric correspondence between the manufactured pipe and its digital design, validating the practical feasibility of FPRO. 

---
# AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals 

**Authors**: Duy Nguyen, Hanqi Xiao, Archiki Prasad, Zaid Khan, Anirban Das, Austin Zhang, Sambit Sahu, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.20643)  

**Abstract**: Self-distillation enables language models to learn on-policy from their own trajectories by using the same model as both student and teacher, with the teacher being conditioned on privileged information unavailable to the student. Such information can come in different types or views, such as solutions, demonstrations, feedback, or final answers. This setup provides dense token-level feedback without relying on a separate external model, but creates a fundamental asymmetry: the teacher may rely on view-specific information that the student cannot access at inference time. Moreover, the best type of privileged information is often task-dependent, making it difficult to choose a single teacher view. In this work, we address both these challenges jointly by introducing AVSD (Adaptive-View Self-Distillation), a novel method of self-distillation with multiple privileged-information views, which reconstructs token-level supervision by separating stable cross-view consensus from view-specific residual signals. AVSD identifies the consensus signal shared across views, which provides a reliable update direction, and then selectively adds the view-specific residual signal to adjust the update magnitude when it both aligns with the consensus direction and remains proportionate to the consensus signal. Experiments on math competition benchmarks (AIME24, AIME25, and HMMT25) show that AVSD consistently outperforms both single-view self-distillation baselines and GRPO, achieving average Avg@8 gains of 3.1% and 2.2% over the strongest baselines on Qwen3-8B and Qwen3-4B, respectively. Moreover, on code-generation benchmarks (Codeforces, LiveCodeBench v6) using Qwen3-8B, AVSD outperforms the single-view self-distillation baseline by 2.4% on average. 

---
# Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs 

**Authors**: Yifei Wang, Tianlin Li, Xiaohan Zhang, Yida Yang, Xiaoyu Zhang, Li Pan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20641)  

**Abstract**: Inference optimization is a vital technique for deploying LLMs at scale. Compilation is the most widely adopted optimization technique for LLMs. While it assumes semantic equivalence between the original and compiled graphs, we first uncover its numerical side effects can be maliciously exploited to implant stealthy backdoors in LLMs. We propose a unified optimization-triggered attack framework comprising two complementary strategies. Without any modification to the compiler or hardware, one strategy flips predictions for specific inputs only when the model is compiled, while the other uses a universal trigger that remains dormant under uncompiled execution but hijacks arbitrary inputs once compilation optimization is applied. Both attacks bypass standard safety evaluations run without compilation. We empirically demonstrate that these optimization-triggered backdoors achieve attack success rates averaging 90% across four mainstream open-source LLMs and four tasks, while clean accuracy is preserved at nearly 100% under all settings. Our findings reveal a novel attack surface at the intersection of optimization and security in the LLM deployment pipeline, and we investigate practical defenses to mitigate this threat. 

---
# Pareto-Enhanced Portrait Generation: Vision-Aligned Text Supervision for Alignment, Realism, and Aesthetics 

**Authors**: Yunlong Wang, Jinjin Shi, Wenbin Gao, Xuran Xu, Runyu Shi, Ying Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20640)  

**Abstract**: Text-to-image diffusion models often face a severe trilemma in human portrait generation: text-image alignment, photorealism, and human-perceived aesthetics inherently inhibit one another. Supervised Fine-Tuning (SFT) is an effective method for enhancing the photorealism of image generation. However, it often leads to overfitting to the training dataset, corrupts pre-trained image priors, and degrades alignment or aesthetics. To break this bottleneck, we propose a feature supervision paradigm for Multimodal Diffusion Transformers (MM-DiT). Specifically, we introduce a lightweight cross-modal alignment mechanism that implicitly extracts multi-granularity vision-aligned text representations from SigLIP 2 and applies supervision to the image branch of MM-DiT during the training stage, with zero extra inference overhead. Our method injects vision-aligned text guidance while preserving the base model's original generalization, avoiding degradation caused by SFT. Furthermore, our method directly mines implicit multi-granularity aesthetic signals from pre-trained vision foundation models to optimize human-perceived aesthetics. Extensive experiments on MM-DiTs show that our method pushes the Pareto frontier and achieves synergistic improvements across text-image alignment, photorealism, and human-perceived aesthetics. 

---
# Retrieval-Augmented Long-Context Translation for Cultural Image Captioning: Gators submission for AmericasNLP 2026 shared task 

**Authors**: Aashish Dhawan, Christopher Driggers-Ellis, Dzmitry Kasinets, Daisy Zhe Wang, Christan Grant  

**Link**: [PDF](https://arxiv.org/pdf/2605.20626)  

**Abstract**: We present the University of Florida Gators submission to the AmericasNLP 2026 shared task on cultural image captioning for Indigenous languages. Our two-stage pipeline generates a Spanish intermediate caption with Qwen2.5-VL, then produces the target-language caption using retrieval-augmented many-shot prompting with Gemini 2.5 Flash. We achieve 164.1%, 131.7%, and 122.6% improvements over the shared task baseline for Bribri, Guaraní, and Orizaba Nahuatl captioning, respectively, in our dev set evaluation and maintain >150% improvements for the Bribri and Orizaba Nahuatl languages in the test set evaluation. We find retrieval is highly language-dependent, beneficial only for large, in-domain corpora, and that synthetic data augmentation accounts for around 28 chrF++ of the dev set Guaraní performance gain. Our submission is the overall winner of the shared task, placing second out of five finalist submissions in human evaluations of target-language captions. 

---
# Accelerating Video Inverse Problem Solvers with Autoregressive Diffusion Models 

**Authors**: Taesung Kwon, Jonghyun Park, Hyungjin Chung, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.20624)  

**Abstract**: Diffusion models provide powerful priors for zero-shot video inverse problems, but their real-time deployment is hindered by two inefficiencies: high initial latency caused by holistic video restoration, and low throughput resulting from multiple VAE passes to enforce measurement consistency in pixel space. To overcome these limitations, we propose Autoregressive Video Inverse problem Solver (AVIS). The AVIS framework leverages autoregressive video diffusion models to restore videos in a streaming manner, naturally eliminating latency bottlenecks. Specifically, AVIS initializes reverse diffusion with a measurement-consistent estimate, reducing the required sampling steps. Compared to leading non-autoregressive solvers, AVIS drastically reduces initial latency from 114s to 4s and increases throughput from 0.71 to 1.18 FPS while achieving superior restoration quality. We further introduce a highly accelerated variant, dubbed AVIS Flash, that enforces measurement consistency solely on the first chunk. AVIS Flash substantially boosts throughput to 5.91 FPS on a single RTX 4090 GPU while maintaining competitive performance and achieving a favorable efficiency-performance trade-off, paving the way toward real-time deployment. 

---
# Lower Bounds for Advection-Diffusion Equations: An Exploration with AI-Generated Proofs 

**Authors**: Chenyang An, Xiaoqian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20623)  

**Abstract**: We establish explicit lower bounds for advection-diffusion equations in three settings: a polynomial $\dot H^{-1}$ bound for inviscid shears with $u\in L^\infty_t W^{1,1}_y$, a uniform positive lower bound on the mixing scale for diffusive shears, and an exponential $L^2$ bound for rapidly oscillating time-periodic flows. All constants are explicit in the data.
The proofs were generated entirely by a multi-agent math proving system, QED, without expert human intervention, serving as a test of AI's capability to produce rigorous mathematics. 

---
# Beyond Routing: Characterising Expert Tuning and Representation in Vision Mixture-of-Experts 

**Authors**: Gene Tangtartharakul, Katherine R. Storrs  

**Link**: [PDF](https://arxiv.org/pdf/2605.20610)  

**Abstract**: Mixture-of-Experts (MoE) models are often interpreted by analysing which categories are routed to which experts. However, routing alone does not reveal what each expert actually encodes. We train sparsely-gated convolutional MoE models with a contrastive objective on natural images and characterise expert specialisation using tools from visual neuroscience. Extending from gating-level to expert-level analyses, we measure per-expert category separability, and per-expert tuning using the most exciting inputs. Extending from category-level to feature-level explanations, we interpret tuning via semantic dimensions derived from a dataset of human behavioural judgements (THINGS). Finally, we use tuning and representational similarity analysis to assess the stability of expertise-allocation across independent initialisations. We find that an animate-inanimate distinction dominates expert partitioning, apparent from gating through to expert readout, and is stable across independently trained models. Although routing statistics suggest relatively sparse, categorical preferences, expert analyses reveal broader tuning to continuous visual and semantic dimensions that extend beyond category boundaries. Experts exhibit similar category-separability to one another, despite distinct feature tuning, demonstrating the explanatory benefits of moving beyond category-level analyses. Together, these results show that expert specialisation in vision MoEs extends well beyond category routing and is better understood by probing fine-grained expert-level tuning and representational structure. 

---
# Self-Training Doesn't Flatten Language -- It Restructures It: Surface Markers Amplify While Deep Syntax Dies 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20602)  

**Abstract**: Successive self-training on a language model's own outputs is widely characterized as a process of flattening: diversity drops, distributions narrow, and the text becomes "more like itself." We provide evidence that this characterization is incomplete. Across eleven generations of self-training on five models (GPT-2 124M, Pythia-410M, Pythia-1.4B, OPT-1.3B, Pythia-2.8B), language is not flattened uniformly -- it is restructured. Surface markers (discourse connectives, hedges, em-dashes) rise, while mid- and deep-syntactic structures (questions, parentheticals, passives, subjunctives) collapse. We formalize this asymmetric collapse as the Structural Depth Hypothesis (SDH): the per-generation decay rate of a linguistic feature is predicted primarily by its structural depth -- the number of nested syntactic dependencies it requires -- and only secondarily by its generation-zero output frequency. Pooling 17-feature panels from five models spanning three architecture families (N=85), the pooled Spearman correlation is rho=0.540 (p < 10^{-6}; cluster-bootstrap 95% CI [0.434, 0.634]), while frequency is a substantially weaker predictor (rho=0.225). A matched human-text fine-tuning control yields rho=0.039 (p=0.88), confirming the gradient is self-training-specific. We further document a Superficial Complexity Paradox: aggregate complexity proxies (dep-tree depth, TTR, word length) all rise as the underlying clause structure dies, with direct implications for training-data curation and LLM-text detection. 

---
# Multi-agent Collaboration with State Management 

**Authors**: Mengyang Liu, Taozhi Chen, Zhenhua Xu, Xue Jiang, Yihong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.20563)  

**Abstract**: Recent advances in multi-agent systems have shown great potential for solving complex tasks. However, when multiple agents edit a shared codebase concurrently, their changes can silently conflict and inconsistent views lead to integration failures. Existing multi-agent systems address this through workspace isolation (e.g., one git worktree per agent), but this defers conflict resolution to a post-hoc merge step where recovery is expensive. In this paper, we propose STORM, i.e., STate-ORiented Management for multi-agent collaboration. Specifically, STORM manages agent states by mediating their interactions with the shared workspace, ensuring that each agent operates on a consistent view of the codebase and that conflicting edits are detected and resolved at write time. We evaluate STORM on Commit0 and PaperBench across multiple LLMs. STORM outperforms the git-worktree-based multi-agent baseline by +18.7 on Commit0-Lite and +1.4 on PaperBench, while achieving comparable or better cost efficiency. Combined with single-agent runs, STORM reaches highest scores of 87.6 and 78.2 on the two benchmarks respectively, suggesting that explicit state management is a more effective foundation for multi-agent collaboration than workspace isolation. STORM can also be plugged into any multi-agent system seamlessly. 

---
# Complementing reinforcement learning with SFT through logit averaging in the post training of LLMs 

**Authors**: Xingwei Gan, Ying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20555)  

**Abstract**: We introduce a novel method that averages the logits of a frozen reference policy (e.g., SFT) and a trainable policy, and incorporate the method into Group Relative Policy Optimization (GRPO). In contrast to Reinforcement Learning with Verifiable Rewards (RLVR) methods, our proposal does not involve a Kullback Leibler (KL) regularization or critic; the trainable policy and the reference anchor are coupled through the logit averaging structure to leverage the reasoning expertise of the trainable policy while maintaining the formatting advantage of SFT. Our method is evaluated on MATH, cn-k12, and MMLU, and the results show a higher accuracy or at least comparable accuracy relative to the canonical KL-regularized GRPO. 

---
# Faster or Stronger: Towards Flexible Visual Place Recognition via Weighted Aggregation and Token Pruning 

**Authors**: Zichao Zeng, June Moh Goo, Junwei Zheng, Weijia Fan, Jiaming Zhang, Rainer Stiefelhagen, Jan Boehm  

**Link**: [PDF](https://arxiv.org/pdf/2605.20551)  

**Abstract**: Visual Place Recognition (VPR) aims to match a query image to reference images of the same place in a large-scale database. Recent state-of-the-art methods employ Vision Transformers (ViTs) as backbone foundation models to extract patch-level features that are robust to viewpoint, illumination, and seasonal variations, which are then aggregated into a compact global descriptor for retrieval. Most existing aggregation methods uniformly pool patch tokens into learned clusters, despite the fact that different clusters often encode distinct spatial or semantic patterns and contribute unequally to VPR performance. To address this limitation, we propose Weighted Aggregated Descriptor (WeiAD), which assigns weights to clusters during aggregation, producing more discriminative global representations. Beyond accuracy, retrieval latency is a critical concern for large-scale deployments and resource-constrained edge devices. Prior work mainly reduces latency by compressing global descriptors, while overlooking the cost of feature extraction, an issue exacerbated by ViT-based backbones. We therefore introduce WeiToP, a VPR-oriented token pruning framework that reduces feature extraction cost via self-distillation, where aggregation-induced token importance supervises a lightweight pruning module attached to an early transformer layer, enabling inference-time token pruning. After a single joint training phase, WeiToP enables plug-and-play token pruning at inference time, allowing flexible and on-demand control over the accuracy-efficiency trade-off without additional training. Moreover, WeiToP outperforms existing token pruning methods adapted from general vision tasks. 

---
# Latent Process Generator Matching 

**Authors**: Lukas Billera, Hedwig Nora Nordlinder, Ben Murrell  

**Link**: [PDF](https://arxiv.org/pdf/2605.20547)  

**Abstract**: Many recent flow-matching and diffusion-style generative models rely on auxiliary stochastic dynamics during training: a richer process is simulated to define conditional targets, but the auxiliary state is either intractable to sample at generation time or simply not part of the desired output. Existing Generator Matching theory formalises conditioning on static latent random variables, and several recent papers prove special cases of projection results for particular augmented-state constructions. We introduce latent process generator matching, a general framework that treats the observed generative state as a deterministic image $X_t=\Phi(Y_t)$ of a tractable Markov process $Y_t$. We show that in this setting one may learn the generator of a stochastic process on the image space which has the same one-time marginal distributions as the projected process. This generalizes and subsumes the discrete latent process results from the literature, and extends Generator Matching from static latent variables to a rich family of time-dependent latent conditional processes. 

---
# Axiomatizing Neural Networks via Pursuit of Subspaces 

**Authors**: Mehmet Yamac, Mert Duman, Ugur Akpinar, Felix Rojas Casadiego, Serkan Kiranyaz, Marcel van Gerven, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2605.20534)  

**Abstract**: While deep neural networks have achieved remarkable success across a wide range of domains, their underlying mechanisms remain poorly understood, and they are often regarded as black boxes. This gap between empirical performance and theoretical understanding poses a challenge analogous to the pre-axiomatic stage of classical geometry. In this work, we introduce the Pursuit of Subspaces (PoS) hypothesis, an axiomatic framework that formulates neural network behavior through a set of geometric postulates. These axioms, together with their derived consequences, provide a unified perspective on representation, computation, and generalization in both shallow and deep architectures. We show that this framework yields geometric explanations for fundamental questions in deep learning, including representation structure, architectural mechanisms, and generalization behavior, offering a principled step toward a coherent theoretical foundation. 

---
# Collocational bootstrapping: A hypothesis about the learning of subject-verb agreement in humans and neural networks 

**Authors**: Claire Hobbs, R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2605.20529)  

**Abstract**: In what ways might statistical signals in linguistic input assist with the acquisition of syntax? Here we hypothesize a mechanism called collocational bootstrapping, in which regularities in word co-occurrence patterns can provide cues to syntactic dependencies. We investigate whether this mechanism can support the acquisition of English subject-verb agreement. First, we simulate language acquisition by training neural networks on synthetic datasets that vary in how predictable their subject-verb pairings are. We find that there is a range of variability levels at which these statistical learners robustly learn subject-verb agreement. We then analyze the variability of subject-verb pairings in child-directed language, and we find that the variability in such data falls within the range that supported robust generalization in our computational simulations. Taken together, these results suggest that collocational bootstrapping is a viable learning strategy for the type of input that children receive. 

---
# NeuroQA: A Large-Scale Image-Grounded Benchmark for 3D Brain MRI Understanding 

**Authors**: Mohammad H. Abbasi, Favour Nerrise, Shaurnav Ghosh, Ridvan Yesiloglu, Yuncong Mao, Bailey Trang, Mohammad Asadi, Merryn Daniel, Gustavo Chau Loo Kung, Ken Chang, Pavan Pinkesh Shah, Adam Turnbull, Kyan Younes, Seena Dehkharghani, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2605.20525)  

**Abstract**: We present NeuroQA, a large-scale benchmark for visual question answering in 3D brain magnetic resonance imaging (MRI), with 56,953 QA pairs from 12,977 subjects across 12 datasets. It spans ages 5-104 and five clinical domains: Alzheimer's, Parkinson's, tumors, white matter disease, and neurodevelopment. Unlike prior medical Visual Question Answering (VQA) efforts that operate on 2D slices or rely on narrow diagnostic labels, NeuroQA pairs every item with a full 3D volume. It evaluates 11 clinically grounded reasoning skills across Yes/No, multiple-choice, and open-ended formats. Of the 203 templates, 131 are image-grounded (answerable from a 3-plane viewer) and 72 are image-informed (ground truth from quantitative volumetry or clinical instruments). To remove text-only shortcuts, we apply answer-distribution refinement, reducing closed-format text-only accuracy from $>$80% to 44.6%; image necessity is assessed separately through an image-grounding protocol released with the benchmark. A 38-rule deterministic pipeline and two rounds of expert review verify every QA pair against FreeSurfer measurements, metadata, or radiology report fields, with zero same-subject contradictions across templates. We conduct a clinician evaluation in which two clinicians independently assess 100 frozen test items on a three-plane viewer. On closed-format (Yes/No + multiple-choice) test-public items, the best zero-shot vision-language model and a supervised 3D CNN baseline reach 47.5% and 43.7% accuracy respectively, both below the 49.4% text-only majority-template floor. NeuroQA adopts a two-tier release with public QA pairs for open-access datasets and reproducible generation scripts for datasets restricted by data use agreements (DUAs), plus subject-level splits, a held-out private test set, and an online leaderboard. 

---
# Machine-Learning-Enhanced Non-Invasive Testing for MASLD Fibrosis: Shallow-Deep Neural Networks Versus FIB-4, Tabular Foundation Models, and Large Language Models 

**Authors**: Athanasios Angelakis, Gabriele De Vito, Eleni-Myrto Trifylli, Filomena Ferrucci  

**Link**: [PDF](https://arxiv.org/pdf/2605.20523)  

**Abstract**: Advanced fibrosis is a major determinant of liver-related morbidity in metabolic dysfunction-associated steatotic liver disease (MASLD). FIB-4 is widely used as a first-line non-invasive test, but its fixed formula may underuse diagnostic information contained in age, aspartate aminotransferase, alanine aminotransferase, and platelet count. We evaluated whether machine-learning-enhanced non-invasive testing (MLE-NIT) can improve advanced fibrosis detection while preserving this FIB-4 variable space.
We used three biopsy-confirmed MASLD cohorts from China, Malaysia, and India (n=784). The Chinese cohort was split into 486 training and 54 internal validation/tuning patients; final performance was reported only on the Malaysian and Indian external cohorts. Models used five variables: age, FIB-4, aspartate aminotransferase, platelet count, and alanine aminotransferase. We compared FIB-4 with a shallow-deep neural network (s-DNN), TabPFN, and gpt-4o-2024-08-06.
FIB-4 achieved external ROC-AUCs of 0.75 and 0.60 in Malaysia and India, respectively. TabPFN achieved 0.69 and 0.66, fine-tuned GPT-4o achieved 0.75 and 0.63, and the s-DNN achieved 0.77 and 0.67, respectively. The s-DNN contained only 354 trainable parameters, compared with 7,244,554 for TabPFN, yet provided a more balanced external operating profile. Calibration showed s-DNN Brier scores of 0.18 and 0.22, and permutation importance identified AST and FIB-4 as dominant variables. Compact non-linear MLE-NITs may enhance FIB-4-based fibrosis assessment without increasing clinical data requirements. 

---
# Codec-Robust Attacks on Audio LLMs 

**Authors**: Jaechul Roh, Jean-Philippe Monteuuis, Jonathan Petit, Amir Houmansdar  

**Link**: [PDF](https://arxiv.org/pdf/2605.20519)  

**Abstract**: Prior attacks on Audio Large Language Models (Audio LLMs) demonstrated that carefully crafted waveform-domain perturbations can force targeted adversarial outputs. As a defense mechanism against these attacks, real-world codec compression preprocessing has been studied to both detect and remove the perturbations. Yet no existing attack has demonstrated robustness against these compressions. We introduce CodecAttack, which optimizes a perturbation in a neural audio codec's continuous latent space rather than directly perturbing the audio waveform. We show that the codec's compression channel, which discards waveform perturbations, transmits perturbations crafted in its own latent space. To further harden the attack across real-world compression channels, we apply multi-bitrate straight-through Expectation-over-Transformation (EoT), all without modifying the target model. Across three realistic Audio LLM deployment scenarios and three target models, CodecAttack achieves an average 85.5% target-substring attack success rate (ASR) on Opus at moderate bitrates, while the waveform baseline trained with identical EoT hardening does not exceed 26% at any bitrate. The attack transfers to held-out codecs, reaching up to 100% ASR on MP3 and 84% on AAC-LC without retraining. A per-band energy analysis shows that the latent perturbation concentrates below 4kHz, exactly where codecs allocate the most bits, while the waveform baseline spreads into higher frequencies that codecs discard. These results demonstrate that lossy compression is not a reliable defense against adversarial audio and that codec-aware attacks pose a practical threat to deployed Audio LLM systems. 

---
# ShadeBench: A Benchmark Dataset for Building Shade Simulation in Sustainable Society 

**Authors**: Longchao Da, Mithun Shivakoti, Xiangrui Liu, T Pranav Kutralingam, Yezhou Yang, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.20510)  

**Abstract**: Urban heat exposure is becoming an increasingly critical challenge due to the intensifying urban heat island effect. Fine-grained shade patterns, especially those induced by urban buildings, strongly influence pedestrians' thermal exposure and outdoor activity planning. However, accurately modeling and analyzing urban shade at scale remains difficult because of the lack of large-scale datasets and systematic evaluation frameworks. To address this challenge, we present ShadeBench, a comprehensive dataset and benchmark for urban shade understanding. ShadeBench contains geographically diverse urban scenes with temporally varying simulated shade maps and textual descriptions, together with aligned satellite imagery, building skeleton representations, and 3D building meshes. Built upon this multimodal dataset, ShadeBench supports a range of downstream tasks, including shade generation, shade segmentation, and 3D building reconstruction. We further establish standardized evaluation protocols and baseline methods for these tasks. By enabling scalable and fine-grained shade analysis, ShadeBench provides a foundation for data-driven urban climate research and supports future studies in heat-resilient urban planning and decision-making. The code and dataset are publicly available at this https URL. 

---
# Tippett-minimum Fusion of Representation-space Diffusion Models for Multi-Encoder Out-of-Distribution Detection 

**Authors**: Neelkamal Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20502)  

**Abstract**: We address out-of-distribution (OOD) detection across the full spectrum of distribution shifts -- global domain changes, semantic divergence, texture differences, and covariate corruptions -- through a multi-encoder fusion of per-encoder representation-space diffusion models (RDMs). We statistically identify each encoder's sensitivity to specific shift types from ID data alone and introduce EncMin2L -- an encoder-agnostic two-level $\min(\cdot)$-gate that combines and calibrates per-encoder diffusion-based likelihood detectors without OOD labels, outperforming monolithic multi-encoder baselines at $2.3\times$ lower parameter cost. Two ID-data diagnostics: $\eta^2$ (class-conditional F-test) and $\Delta\mu$ (log-likelihood shift under synthetic corruptions) -- quantify encoder specialization, while a Tippett minimum $p$-value combination aggregates per-encoder scores into a single, calibration-stable OOD signal. EncMin2L achieves $\geq 0.94$ AUROC across all four shift types simultaneously, outperforming the state-of-the-art representation-space diffusion OOD detectors across overlapping benchmarks. 

---
# Training Language Agents to Learn from Experience 

**Authors**: Yuval Shalev, Zifeng Ding, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2605.20477)  

**Abstract**: Language agents can adapt from experience in interactive environments, but current reflection-based methods can only self-correct within a single task instance. Whether such experience can be distilled into reusable lessons that improve performance on future unseen tasks remains unclear. We address this problem by introducing the In-context Training (ICT) task, a framework for evaluating cross-task self-improvement in language agents. In ICT, a reflector model observes trajectories collected by an actor model and generates system prompts intended to improve the actor's performance on future unseen tasks. We then propose an RL-based training pipeline for learning such reflections directly from experience, without human-provided examples. Across ALFWorld and MiniHack, our trained reflectors outperform an untrained baseline on most held-out task families, showing that the ability to learn from experience can itself be learned. In some cases, we observe generalisation beyond the benchmark on which the reflector was trained, to substantially different environments. Finally, we introduce MetaGym, a generic Python library for constructing meta-environments, enabling future research on self-improving language agents. 

---
# Code Generation by Differential Test Time Scaling 

**Authors**: Yifeng He, Ethan Wang, Jicheng Wang, Xuanxin Ouyang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20473)  

**Abstract**: Test-time scaling has emerged as a promising approach for improving code generation by exploring large solution spaces at inference time. However, existing methods often rely on public test cases that are unavailable in practice, or require extensive LLM inference for candidate selection, leading to significant token consumption and time overhead. We present DiffCodeGen, a novel test-time scaling method for code generation based on coverage-guided differential analysis. DiffCodeGen generates diverse code candidates using various sampling and prompting strategies, then applies coverage-guided fuzzing to synthesize inputs without requiring any existing tests or large language models. By executing all candidates on these inputs, DiffCodeGen captures their dynamic behavior and clusters candidates based on behavioral similarity. DiffCodeGen selects the medoid of the largest cluster as the final output. Unlike prior test-time scaling methods that invoke additional LLM inference for candidate selection, DiffCodeGen performs selection without any extra model calls, incurring little to no additional token consumption. DiffCodeGen is fully asynchronous, naturally suited to the current trend of agentic coding, and is thus efficient and highly scalable. We evaluate DiffCodeGen across 4 large language models, demonstrating consistent improvements over baselines. Compared to state-of-the-art test-time scaling methods, DiffCodeGen achieves competitive or superior performance while using only a fraction of time and tokens. DiffCodeGen is model-agnostic and can be combined with reasoning models to further boost performance. 

---
# EPC-3D-Diff: Equivariant Physics Consistent Conditional 3D Latent Diffusion for CBCT to CT Synthesis 

**Authors**: Alzahra Altalib, Chunhui Li, Haytham Al Ewaidat, Khaled Alawneh, Ahmad Qendel, Alessandro Perelli  

**Link**: [PDF](https://arxiv.org/pdf/2605.20470)  

**Abstract**: Cone-beam CT (CBCT) is routinely acquired during radiotherapy for patient setup, but its quantitative reliability is degraded by scatter, noise, and reconstruction artifacts, limiting Hounsfield Unit (HU) accuracy. We propose EPC-3D-Diff, a novel conditional 3D latent diffusion framework for volumetric CBCT to CT synthesis that introduces a projection domain equivariance loss derived from acquisition physics. Unlike common image domain equivariance, we exploit the fact that an in plane rotation of the volume corresponds to an angular shift in its projections. During training, we enforce this relationship by forward projecting rotated synthesized CT volumes and matching them to appropriately angle shifted projections of the paired target CT, yielding a physics consistent equivariance constraint integrated into the diffusion objective. To capture full 3D context efficiently, conditional diffusion is performed in a compact latent space learnt by a lightweight 3D autoencoder, preserving axial depth while downsampling in plane resolution for stable training. We validate on a paired head CBCT/CT phantom dataset, including repeat scans, and paired clinical data using patient wise splits, and perform single and mixed domain training, ablations, and comparisons with diffusion and CycleGAN. EPC-3D-Diff generalizes well and achieved substantial improvements, +7.4 dB (phantom) and +1.8 dB (clinical data) in PSNR compared to state of the art methods, alongside improved SSIM and HU accuracy, within tissue boundaries. Overall, EPC-3D-Diff improves robustness and physics consistency, supporting HU aware synthesis for downstream radiotherapy workflows. 

---
# Pixel Wised Lesion Prediction on COVID-19 CT Imagery: A Comparative Analysis of Automated Image Segmentation Architectures 

**Authors**: Sarmad Khan, Arslan Shaukat, Umer Asgher, Basim Azam  

**Link**: [PDF](https://arxiv.org/pdf/2605.20459)  

**Abstract**: In recent years, there has been a notable increase in the level of attention that is given to algorithms based on deep learning in the context of medical image segmentation. Nevertheless, the reliability of the field has been hindered due to the absence of a standardized methodology for performance analysis and the utilization of different datasets in previous research. The primary objective of the research is to comprehensively evaluate contemporary segmentation frameworks combined with state-of-the-art pre-trained backbones in order to accurately predict COVID-19 lesions in CT images. Moreover, this evaluation can serve as a point of reference for the segmentation of images in various other imaging scenarios. In order to accomplish this, we integrate four distinct deep learning architectures, namely Unet, PSPNet, Linknet, and FPN, with six pre-trained encoders, including VGG 19, DenseNet 121, Inception ResNet V2, MobileNet V2, SeresNet 101, and EfficientNet B0. This approach enables the development of diverse testing architectures. In the context of image segmentation, our research encompassed both binary and multi-class experimentation. The findings derived from our analysis of three distinct COVID-19 CT segmentation datasets indicate that deep learning architectures yield precise and efficient segmentation outcomes. Significantly, a maximum F1-Score of 98% was attained for binary class segmentation, while multi-class segmentation yielded F1-Scores of 75% and 77% across two separate datasets. The utilization of artificial intelligence and deep learning enhances the diagnostic process for pandemic diseases across multiple dimensions. 

---
# Agentic Agile-V: From Vibe Coding to Verified Engineering in Software and Hardware Development 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2605.20456)  

**Abstract**: Agentic AI coding systems can inspect repositories, plan implementation steps, edit files, call tools, run tests, and submit pull requests. These capabilities make software and hardware development faster in some settings, but current evidence does not support the simple claim that autonomous code generation automatically improves engineering outcomes. Controlled studies report productivity gains in some enterprise tasks, slowdowns in mature open-source work, moderate but heterogeneous meta-analytic effects, and persistent failures in repository setup, dependency handling, permission gating, and hardware verification. This paper argues that the central problem is no longer prompt engineering; it is engineering process control. It synthesizes evidence from agentic software engineering, GitHub-scale adoption studies, repository-level agent configuration, productivity trials, issue-resolution benchmarks, and hardware/RTL verification research. It proposes Agentic Agile-V, a process framework that uses Agile-V as the lifecycle backbone and a task-level SCOPE-V loop - Specify, Constrain, Orchestrate, Prove, Evolve, and Verify - to convert conversational intent into structured engineering artifacts and acceptance evidence. The paper contributes: (i) a taxonomy of minimum input artifacts for agentic software, firmware, and hardware work; (ii) a conversation-to-contract gate that separates exploratory dialogue from implementation; (iii) risk-adaptive feature, bug-fix, testing, and hardware workflows; and (iv) an evidence-bundle acceptance model for agent-generated artifacts. The paper concludes that agentic AI does not eliminate engineering discipline; it increases the value of requirements, constraints, traceability, independent verification, and human approval. 

---
# LLM Pretraining Shapes a Generalizable Manifold: Insights into Cross-Modal Transfer to Time Series 

**Authors**: Alexis Roger, Prateek Humane, Zhenghan Tai, Gwen Legate, Andrei Mircea, Vasilii Feofanov, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2605.20449)  

**Abstract**: Can language-pretrained transformers become effective time-series forecasters, and why? In this paper, we show that cross-modal transfer arises because language pretraining preconditions time series training with a reusable manifold. A linear probe on frozen LLM states decodes realistic time-series trajectories without paired supervision, and retrieval in this projected space yields competitive forecasts, showing that structure and dynamics exist before finetuning. Pretrained initialization also improves optimization, producing coherent gradients and a highly anisotropic loss landscape unlike random initialization. Finetuning then acts as low-dimensional alignment, reusing existing directions rather than learning temporal primitives from scratch, as evidenced by low-rank updates, subspace alignment, and shared features for periodicity, trend, and repetition. Together, these results support a geometric account of LLM-to-time-series transfer: language pretraining builds the manifold, and finetuning projects numerical dynamics onto task-relevant directions. 

---
# A Comprehensive Comparison of Deep Learning Architectures for COVID-19 Classification on CT & X-ray Imagery 

**Authors**: Sarmad Khan, Arslan Shaukat, Umer Asgher, Basim Azam  

**Link**: [PDF](https://arxiv.org/pdf/2605.20445)  

**Abstract**: COVID-19 was a significant challenge that led to the loss of numerous lives daily. Not only a certain country was involved in this outbreak, but even the world has suffered because of the coronavirus. Imaging techniques using computed tomography (CT) and X-rays of the lungs are the most useful tools for the COVID-19 or any other pandemic disease screening process. Technology today has revolutionized the world by using artificial intelligence to replace manual processes with automated machines, which enable the system to imitate the human brain by making wise decisions based on experience. Motivated by this, our work proposes to use convolutional neural networks (CNN) based models for designing a computer-aided diagnosis (CAD) system that differentiates between COVID-19 and healthy lung pictures. We used two different sets of X-ray images of the lungs in addition to two different sets of CT scans and the classification is done using a variety of networks that have been pre-trained such as VGG (16, 19), Densenet (121), Resnet (50, 50 V2, 101 V2), Mobile net (V2), Xception Inception (V3, Resnet V2), Efficient net (B0) and Nasnet (Large). On the X-ray and CT image datasets, Resnet and VGG architecture have shown the ability to properly differentiate COVID-19 from normal images, with an average accuracy of 95 to 98 percent respectively. Our acquired results on the classification datasets are competitive and superior to previously reported findings in the literature. 

---
# Modeling Emotional Dynamics in Agent-to-Agent Interactions on Moltbook 

**Authors**: Syed Mhamudul Hasan, Abdur R. Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2605.20442)  

**Abstract**: Generative AI systems are increasingly deployed as interactive agents in online environments, such as a social network called Moltbook. In Moltbook, large-scale agentic AIs can post, comment, and engage in activities generated at scale by AI-driven text. Yet these agent behavioral characteristics remain insufficiently understood, particularly in complex, multi-agent interaction. In this study, we analyze the emotional dynamics of agent interactions within Moltbook. We construct an emotion-aware framework that maps textual interactions to a predefined set of fine-grained emotional categories, enabling the extraction of structured emotion profiles across agents and interaction contexts. To further evaluate behavioral reliability, we introduce an emotion-based domain called Persona-Stimulus-Reaction (PSR) that captures the alignment of emotional responses across similar contexts. Our analysis shows distinct emotional patterns and varying levels of behavioral stability across agents. Our analysis reveals that agents exhibit distinct emotional signatures with varying levels of behavioral stability influenced by interaction context. 

---
# Weight Decay Regimes in Grokking Transformers: Cheap Online Diagnostics 

**Authors**: Lucky Verma  

**Link**: [PDF](https://arxiv.org/pdf/2605.20441)  

**Abstract**: Transformers trained on modular arithmetic exhibit sharp transitions between memorization, generalization, and collapse. We show that weight decay acts as a scalar empirical control parameter for these regimes, and introduce two cheap online diagnostics, mean pairwise attention-head cosine similarity and entropy standard deviation, that track training dynamics from attention activations alone and complement loss-landscape diagnostics at lower compute cost. Across eleven experimental conditions and three model scales (0.82M to 85M parameters), the weight-decay axis separates memorization, developmental grokking, and collapse. A near-transition logistic fit localizes the memorization-to-developmental boundary at $\lambda_c=0.0158$ (95% CI [0.0109, 0.0200], N=210); a power-law fit gives an empirical exponent $\nu=0.757$ (CI [0.725, 0.799]). Reference exponents $\nu=1/2$ and 3D Ising $\nu \approx 0.63$ lie outside this empirical CI under our four-bin grid, so we report $\nu$ as empirical and defer universality-class identification to denser finite-size-scaling work. A horizon-matched multi-task replication (n=280, four modular operations) preserves the weight-decay control pattern; a paired attention-head re-initialization experiment at $\lambda=0.05$ changes Phase-2 amplitude (Cohen's $d=-1.190$, n=10, $p_t=4.5 \times 10^{-3}$), while matched weight-norm clipping does not. Three cross-architecture probes (4L MLP, 4L LSTM, and 4L Mamba; each n=70) replicate the weight-decay-controlled transition with architecture-specific $\lambda_c$ values. Main diagnostic claims are scoped to modular arithmetic in small transformer attention models; the non-attention experiments are scope probes, and architecture-wide, language-model, and universality-class claims are out of scope. 

---
# Group-Algebraic Tensors: Provably-optimal Equivariant Learning and Physical Symmetry Discovery 

**Authors**: Paulina Hoyos, Shashanka Ubaru, Dongsung Huh, Vasileios Kalantzis, Kenneth L. Clarkson, Misha Kilmer, Haim Avron, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2605.20440)  

**Abstract**: We introduce the $\star_G$ tensor algebra, in which any finite group $G$ defines the multiplication rule, making equivariance an intrinsic algebraic property rather than an architectural constraint. The framework rests on three machine-verified theoretical pillars: (i)~an Eckart-Young optimality guarantee for the $\star_G$-SVD: the first such result for symmetry-preserving tensor approximation, exact and polynomial-time; (ii)~a Kronecker factorization that composes multiple symmetries by replacing $F_G$ with $F_{G_1} \otimes F_{G_2}$ with no architectural redesign; and (iii)~a 600-line Lean~4 formalization of the $\star_G$ algebra. The framework provides capabilities that equivariant neural networks (ENNs) structurally cannot: a closed-form per-irreducible-representation decomposition of every prediction, and data-driven discovery of the symmetry group that best fits a dataset. As a non-trivial empirical demonstration, decomposing QM9 molecular geometry over the chiral octahedral subgroup of SO(3) recovers the Wigner--Eckart selection rules of angular momentum from data alone, with no quantum mechanical input: scalar properties are A$_1$-dominated, dipole components are T$_1$-dominated, the isotropic polarizability is uniquely insensitive to $l\!=\!1$ as the rank-2-trace decomposition $l\!=\!0 \oplus l\!=\!2$ requires, and the T$_1$/A$_1$ predictive-power ratio separates vector observables from scalar observables by a factor of five. On full QM9 (130{,}831 molecules), $\star_G$-SVD with ridge regression provides closed form predictions at $\sim50-90\times$ fewer parameters than parameter-matched MLPs. Algebraic equivariance thus complements architectural equivariance not as a faster-better-cheaper alternative but as a different mathematical affordance: provably-optimal symmetry-preserving compression, per-irrep interpretability, and data-driven physical discovery. 

---
# Mechanics of Bias and Reasoning: Interpreting the Impact of Chain-of-Thought Prompting on Gender Bias in LLMs 

**Authors**: Edie Pearman, Sophia Osborne, Mira Kandlikar-Bloch, Mina Arzaghi, Florian Carichon, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.20410)  

**Abstract**: Large language models (LLMs) are increasingly deployed in socially sensitive settings despite substantial documentation that they encode gender biases. Chain-of-Thought (CoT) prompting has been proposed as a bias-mitigation approach. However, existing evaluations primarily focus on changes in LLM benchmark performance, providing limited insight into whether apparent bias reductions reflect meaningful changes in a model's internal mechanisms. In this work, we investigate how CoT prompting affects gender bias in LLMs, combining benchmark-based evaluation with mechanistic interpretability techniques and reasoning chain failure analysis. Our results confirm a stereotypical bias present in LLM outputs across benchmarks, showing that CoT prompting does not consistently reduce the bias gap. Mechanistic analyses reveal that although CoT balances biased behavior in certain attention head clusters, gender bias remains embedded in hidden representations, indicating only superficial mitigation. Inspection of reasoning chains further suggests that these improvements stem from memorization and familiarity with the dataset rather than genuine understanding of bias. 

---
# Disentangling Sampling from Training Budget in Class-Imbalanced CT Body Composition Segmentation 

**Authors**: Iason Skylitsis, Dimitrios Karkalousos, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2605.20405)  

**Abstract**: Class imbalance is a fundamental challenge in medical image segmentation, where frequent classes typically dominate training at the expense of rare classes. Loss-based approaches mitigate imbalance by reweighting the per-pixel loss within the batch, while sampling strategies control which images enter the batch. Yet neither explicitly controls which classes appear within the batch, leaving rare-class exposure only partially rebalanced. In this work, we adopt episodic sampling from few-shot learning to promote class-balanced batch construction in a fully supervised setting. We decouple episodic sampling from its conventional metric-learning context and evaluate it in body composition segmentation in CT. We compare episodic sampling against random and weighted sampling on nine muscle and adipose tissues, derived from 210 scans of the public SAROS dataset. Training is performed under full- and low-data regimes, with additional comparisons under matched training iteration budgets. Under full-data training, all three strategies performed comparably (mean Dice 0.882 for episodic, 0.878 for random and weighted). Under low-data training, episodic sampling outperformed random and weighted (0.787 vs. 0.758 and 0.762), driven by a 12-fold difference in training iterations. Under matched training budgets, random and weighted overfit earlier, while episodic improved for approximately three times more iterations before plateauing. Our findings identify the training iteration budget as under-recognized confound in sampling strategies, motivating iteration-aware evaluation protocols for small datasets. Furthermore, the residual advantage of episodic sampling is consistent with an implicit regularization effect of class-balanced batches, offering a low-cost, model-agnostic strategy for class-imbalanced medical image segmentation. Code is available at this https URL. 

---
# Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor 

**Authors**: Xiaocan Li, Shiliang Wu, Zheng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20402)  

**Abstract**: MXFP4 arithmetic can dramatically accelerate reinforcement learning (RL) post-training of large language models (LLMs), yet the quantization error introduces severe accuracy degradation. Existing work treats the quantization error as a monolithic noise term, missing the distinct mechanisms upon interpreting how quantization error damages training. We prove an exact three-way decomposition of quantization error and show how each component dominates a distinct RL training pathway. Our theoretical and empirical analysis decomposes the MXFP4 quantization error into three additive components: "scale bias" from power-of-two rounding, "deadzone truncation" from zeroing small values, and "grid noise" from rounding to the nearest 4-bit grid. Each component dominates a distinct RL failure mode: scale bias accumulates multiplicatively through the backward pass, affecting gradient accuracy; deadzone truncation degrades rollout quality; and grid noise raises the policy's entropy. We combine corrections that are RL failure mode-targeted but not component-exclusive: Macro-block scaling to reduce scale bias, Outlier Fallback recovers deadzone entries, but also partially reduces scale bias induced error, and Adaptive Quantization Noise (AQN) for controlling the policy entropy. On Qwen2.5-3B dense and Qwen3-30B-A3B-Base mixture-of-experts model, the targeted corrections recover BF16 accuracy to within 0.7% and 3.0% respectively. 

---
# STELLAR: Scaling 3D Perception Large Models for Autonomous Driving 

**Authors**: Yingwei Li, Xin Huang, Yang Liu, Yang Fu, Alex Zihao Zhu, Chen Song, Junwen Yao, Anant Subramanian, Hao Xiang, Weijing Shi, Yuliang Zou, Tom Hoddes, Zhaoqi Leng, Govind Thattai, Dragomir Anguelov, Mingxing Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20390)  

**Abstract**: Model scaling has demonstrated remarkable success through large-scale training on diverse datasets. It remains an open question whether the same paradigm would apply to autonomous driving perception systems due to unique challenges, such as fusing heterogeneous sensor data and the need for sophisticated 3D spatial understanding. To bridge this gap, we present a comprehensive study on systematically analyzing the impact of scale on these systems. We develop our STELLAR model based on Sparse Window Transformer, by extending the input modalities to include LiDAR, radar, camera, and map prior. We train the model on a large-scale dataset of 50 million driving examples with up to 500 million parameters. Our large-scale experiments reveal empirical scaling trends that connect model performance to model size, data, and compute. The resulting model establishes a new state-of-the-art on the Waymo Open Dataset challenge, outperforming prior arts by a large margin. Our work demonstrates that large-scale training is a highly promising path for advancing the capabilities of perception models for autonomous driving. 

---
# Nonlocal operator learning for fMRI encoding and decoding tasks 

**Authors**: Andreas Kramer, Saugat Acharya, Alice Giola, Emanuele Zappala  

**Link**: [PDF](https://arxiv.org/pdf/2605.20389)  

**Abstract**: Functional MRI data exhibit high-dimensional spatiotemporal structure, making both prediction and decoding challenging. In this work, we investigate neural integral-operator-based models for encoding and decoding tasks in fMRI, with particular emphasis on the role of nonlocal spatiotemporal context. We implement a latent neural integral operator framework that performs fixed point iterations in an auxiliary space from which classification and stimuli prediction is performed via a decoder. We evaluate our model on two open-source fMRI datasets.
Our experiments examine both decoding of stimuli from fMRI recordings and encoding of fMRI dynamics from stimulus representations. A main focus is the effect of spatiotemporal context: we systematically compare short and long temporal windows, as well as the use of visual cortex vs whole brain recordings, and analyze their influence on performance and latent-space geometry. Across tasks and datasets, larger temporal windows generally improve results and produce more structured learned representations. In decoding experiments, the learned latent space often provides clearer class separation than the raw data. In encoding experiments, although absolute performance remains moderate due to the difficulty of the task, longer temporal windows still yield consistent gains.
These findings suggest that neural integral operators provide a promising framework for modeling fMRI dynamics and that broader spatiotemporal context can be beneficial for both prediction and representation learning. More broadly, the results indicate that exploiting distributed nonlocal structure in brain dynamics requires model architectures specifically designed to capture such dependencies. 

---
# ConceptSeg-R1: Segment Any Concept via Meta-Reinforcement Learning 

**Authors**: Yuan Zhao, Youwei Pang, Jiaming Zuo, Wei Ji, Kailai Zhou, Bin Fan, Yunkang Cao, Lihe Zhang, Xiaofeng Liu, Huchuan Lu, Weisi Lin, Dacheng Tao, Xiaoqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20385)  

**Abstract**: Recent progress in promptable segmentation has shifted visual perception from object-level localization toward concept-level understanding. However, the notion of a concept remains under-specified, making it unclear whether current methods truly generalize beyond category recognition. In this work, we formalize generalized concept segmentation through a three-level taxonomy consisting of context-independent (CI), context-dependent (CD), and context-reasoning (CR) concepts, which reveals a clear capability gap across increasing levels of cognitive complexity. To address this challenge, we propose ConceptSeg-R1, a unified framework that reformulates concept segmentation as rule-induced concept grounding. At the core of our method is Meta-GRPO, a meta-reinforcement learning mechanism that learns transferable task rules from visual demonstrations and verifies them through proxy reasoning. The inferred reasoning states are then translated into segmentation-ready concept prompts via a lightweight concept translation module, enabling deductive application to target images. A shortcut routing strategy further preserves the native efficiency of segmentation models on simple cases. To systematically evaluate generalized concept segmentation, we conduct extensive experiments across diverse CI, CD, and CR concept segmentation benchmarks spanning natural, industrial, medical and reasoning-intensive domains. Without bells and whistles, ConceptSeg-R1 achieves strong performance across the full concept hierarchy while maintaining the native capability of promptable segmentation backbones. As an initial step toward segmenting any concept, we hope ConceptSeg-R1 can serve as a practical baseline for advancing segmentation from object-level prediction toward concept-level understanding. 

---
# Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs 

**Authors**: Carolina Camassa, Derek Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2605.20382)  

**Abstract**: Language models are trained to follow instructions, but they are also powerful pattern completers. What happens when these two objectives conflict? We construct conversations in which a user instruction to behave in a target way T (e.g., always output a specific token, answer in a particular language, or adopt a persona) is opposed by N hardcoded assistant turns demonstrating a competing pattern P. We then measure instruction-following (IF) rates in this setting, across 13 models and 16 different instructions, for up to 50 turns. Average instruction-following rates range from 1% to 99% across models, largely uncorrelated with standard capability benchmarks. The transition from instruction-following to pattern-following is universal but highly model-dependent. Robustness is modulated both by instruction content, with models resisting induction longer when instructions align with their trained value priors, and by output format, with diverse multi-token responses proving substantially more resistant than single-token outputs. Chain-of-thought reasoning improves robustness but does not eliminate susceptibility, and can produce dissociation between correct deliberation and incorrect output. When asked to predict their behavior in this setting, models achieve 83.5% accuracy on average but systematically underestimate their own resistance to induction pressure. These results suggest that instruction-following remains brittle under induction pressure even for otherwise capable models, and that output diversity, rather than semantic engagement with the input, is the primary factor predicting robustness. 

---
# SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework 

**Authors**: Tianshu Wu, Xiangqi Kong, Yue Chen, Qize Yu, Hang Ye, Jia Li, Yizhou Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.20373)  

**Abstract**: Building humanoid robots capable of generalizable whole-body loco-manipulation in the real world remains a fundamental challenge. Existing methods either rely on laborious task-specific reward engineering, rigidly replay reference motions that fail to generalize, or depend on costly teleoperation that limits scalability. While human videos capture diverse human behaviors, motion priors inferred from them are inherently imperfect, suffering from occlusion, contact artifacts, and retargeting errors that render them unsuitable for direct policy learning. To address this, we present SUGAR, a scalable data-driven framework that converts diverse human videos into deployable humanoid loco-manipulation skills, without any task-specific reward engineering or reference-motion conditioning at inference. SUGAR proceeds in three stages. First, a fully automated pipeline extracts kinematic interaction priors including human-object motion trajectories and contact labels from unstructured human videos. Second, a privileged physics-based refiner uses a unified mimic reward and progressive state pool to transform imperfect priors into physically feasible, high-fidelity skills. Third, refined skills are distilled into a hierarchical autonomous policy consisting of a command generator and a command tracker. We evaluate SUGAR on six representative loco-manipulation tasks in simulation and real-world humanoid hardware. Our method substantially outperforms reference-tracking baselines, and performance scales clearly with the amount of human video data. It also achieves zero-shot real-world transfer with reliable closed-loop execution, autonomous failure recovery, and stable long-horizon performance under external perturbations. Project Page: this https URL 

---
# Latent Space Guided Scenario Sampling for Multimodal Segmentation Under Missing Modalities 

**Authors**: Irem Ulku, Ö. Özgür Tanrıöver, Erdem Akagündüz  

**Link**: [PDF](https://arxiv.org/pdf/2605.20372)  

**Abstract**: Multimodal semantic segmentation benefits remote sensing analysis by combining complementary information from different sensor modalities. In real-world remote sensing applications, one or more modalities may be unavailable due to sensor failures, adverse atmospheric conditions, or data acquisition problems. Even with pretrained multimodal representations and existing fine-tuning or adaptation strategies, performance may remain limited because all modality availability scenarios are typically treated as equally informative during training. In this paper, we propose a novel training strategy that learns a scenario sampling distribution directly from the pretrained latent space. Instead of relying on uniform random modality dropout, the proposed method guides fine-tuning toward more informative modality availability scenarios. More specifically, we quantify the effect of each scenario independently based on the distortion it induces in the shared latent representation. We then capture scenario relations using a radial basis function kernel and derive refined scenario scores through a regularized kernel smoothing. These scores are then converted into a probability distribution during scenario sampling for fine-tuning. We evaluate this strategy on three remote sensing image sets, namely DSTL, Potsdam, and Hunan, using CBC-SLP, CBC, and CMX backbones. The experimental results with different image sets and backbones show that our method outperforms standard fine-tuning and LoRA-based adaptation. These findings suggest that the pretrained latent representation can serve as an effective basis for sampling during missing modality fine-tuning. Code is available at this https URL 

---
# DEL: Digit Entropy Loss for Numerical Learning of Large Language Models 

**Authors**: Zhaohui Zheng, Chenhang He, Shihao Wang, Yuxuan Li, Ming-Ming Cheng, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20369)  

**Abstract**: Number prediction stands as a fundamental capability of large language models (LLMs) in mathematical problem-solving and code generation. The widely adopted maximum likelihood estimation (MLE) for LLM training is not tailored to number prediction. Recently, penalty-driven approaches, e.g., Number Token Loss and Discretized Distance Loss, introduce an inductive bias of numerical distance but induce over-sharpened and over-flattened digit distributions, respectively. In this paper, we make an in-depth analysis on LLM numerical learning, and show that existing numerical learning methods conceptually follow a criterion-distance formulation, where the criterion term represents optimization pattern and the distance term instills geometric prior. Consequently, we present Digit Entropy Loss (DEL) for auto-regressive numerical learning, which reformulates the conventional unsupervised entropy optimization in three key designs: leveraging digit conditional probability and binary cross-entropy to guide the entropy optimization into a supervised manner; deprecating the distance term to bypass the issue of numerical distance; and generalizing the integer-based numerical learning to floating-point number optimization, enabling more accurate number prediction. Our DEL formulation can incorporate integers, decimals, and decimal points, expanding the learning objective from a single digit to the floating-point number domain. Experiments conducted on seven mathematical reasoning benchmarks with four representative LLMs, including CodeLlama, Mistral, DeepSeek, and Qwen-2.5, demonstrate that DEL consistently outperforms its counterparts in both overall prediction accuracy and numerical distance. Source codes are at this https URL 

---
# Security Document Classification with a Fine-Tuned Local Large Language Model: Benchmark Data and an Open-Source System 

**Authors**: Ivan Dobrovolskyi  

**Link**: [PDF](https://arxiv.org/pdf/2605.20368)  

**Abstract**: Organizations that scan documents for sensitive information face a practical problem. Cloud services require data to be sent to external infrastructure, while rule-based tools often miss threats that depend on context. This study presents TorchSight, an open-source local system for security document classification built around a fine-tuned Qwen 3.5 27B model. The model was trained on 78,358 samples from 13 permissively licensed sources and GPT-4 synthetic data covering seven security categories and 51 subcategories. In the main evaluation on 1,000 documents, the model reached 95.0% category-level accuracy (95% confidence interval: 93.5-96.2). The tested commercial models scored 75.4-79.9% under the same prompting protocol. On a separate external set of 500 held-out samples, the model reached 93.8% accuracy, which suggests that performance extends beyond the main benchmark, although the margin depends on dataset composition and difficult boundary cases. The results show that a fine-tuned local model can support accurate security document classification while keeping document processing under local control. 

---
# Consistently Informative Soft-Label Temperature for Knowledge Distillation 

**Authors**: Hoang-Chau Luong, Nghia Van Vo, Kaiqi Zhao, Lingwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20357)  

**Abstract**: Knowledge distillation (KD) transfers knowledge from a high-capacity teacher to a compact student by matching their predictive distributions, with temperature scaling serving as a central mechanism for smoothing teacher predictions and exposing informative "dark knowledge" beyond the hard label. However, the standard fixed-temperature design is inherently sample-agnostic. Since samples differ in logit scale and learning difficulty, a single global temperature produces teacher soft labels with highly inconsistent entropy: some predictions remain overly sharp and provide limited inter-class information, whereas others become over-smoothed and lose class-discriminative information. Moreover, sharing the same temperature between teacher and student further imposes rigid logit-scale alignment despite their capacity mismatch. To address these limitations, we propose CIST (Consistently Informative Soft-label Temperature), which assigns separate sample-wise adaptive temperatures to the teacher and student. This design produces consistently informative teacher soft labels while relaxing rigid teacher--student logit-scale matching. It also reweights the distillation objective according to teacher confidence and student learning difficulty. Theoretically, we show that teacher-label entropy is largely governed by the ratio between the maximum teacher logit and the temperature, providing a principled basis for adaptive smoothing. Empirically, CIST mitigates the inconsistency induced by fixed temperature, and experiments on both vision and language distillation tasks show consistent improvements over standard KD and strong baselines with negligible computational overhead. 

---
# Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models 

**Authors**: Pablo Riera, Pablo Brusco, Cristina Kuo, Marcelo Sancinetti, S.R.K. Branavan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20356)  

**Abstract**: Full-duplex spoken dialogue models (SDMs) can listen and speak simultaneously, enabling interaction dynamics closer to human conversation than turn-based systems. Inspired by neural coupling in human communication, we study how such models coordinate their internal representations during interaction. We simulate full-duplex dialogues between two instances of the pretrained \textit{Moshi} model under controlled conditions, manipulating channel noise and decoding bias. Synchronization is measured using Centered Kernel Alignment (CKA) across temporal lags, while anticipatory turn-taking cues are probed from delayed internal activations using causal LSTM models, from both speaker and listener perspectives. We find strong representational synchronization under no noise conditions, peaking near zero lag and degrading with noise, and we show that internal states encode anticipatory information that supports turn-taking prediction ahead of time. 

---
# Causal Unlearning in Collaborative Optimization: Exact and Approximate Influence Reversal under Adversarial Contributions 

**Authors**: Ali Mahdavi, Azadeh Zamanifar, Amirfarhad Farhadi, Omid Kashefi  

**Link**: [PDF](https://arxiv.org/pdf/2605.20341)  

**Abstract**: Federated learning systems must support data deletion requests to comply with privacy regulations, yet retraining from scratch after each deletion is computationally prohibitive. We present HF-KCU, a method that removes a client's contribution by approximating the influence function through conjugate gradient iterations in Krylov subspaces, reducing complexity from O(d^3) to O(kd) where k<<d.A causal weighting mechanism ensures that only clients holding the deleted data receive parameter updates, preventing spurious changes to unaffected clients. Our method is designed to handle bounded adversarial perturbations to the Hessian and gradient, providing graceful degradation under realistic threat models. We validate HF-KCU across convolutional (ResNet-18, SimpleCNN) and transformer (ViT-Lite) architectures on CIFAR-10, MNIST, and Fashion-MNIST. On CIFAR-10 under Dirichlet (alpha=0.5) partitioning, HF-KCU achieves 47.75 times speedup over retraining while maintaining test accuracy within 0.60% of the rational baseline(71.16 vs 71.76 %). Membership inference attacks on the forget set yield success rates of 0.499 matching the retrained model and confirming effective privacy restoration. We provide convergence guarantees showing that the Krylov approximation error decreases as O((k ^1/2-1)/(k^1/2+1)) where k is the Hessian condition number. The causal weighting mechanism ensures surgical updates, where only clients holding deleted data are modified, preserving model quality for unaffected participants and avoiding the instability of gradient-based approaches in asynchronous federated settings. This design provides interpretability as each update is directly traceable to the influence of the deleted data. The method's efficiency and precision make it suitable for production federated systems where deletion requests arrive asynchronously and computational budgets are constrained. 

---
# Targeting Clause Type Distributions: a Picklock for Random Satisfiability Problems 

**Authors**: J. Schwardt, J. C. Budich  

**Link**: [PDF](https://arxiv.org/pdf/2605.20328)  

**Abstract**: Optimization problems such as the NP-complete 3-SAT provide an important benchmark for the difficult task of finding ground-states in strongly correlated many-body systems with rugged energy landscapes. The study of random 3-SAT problems as Ising spin Hamiltonians in statistical physics has yielded major insights including the existence of a satisfiability phase transition, and the prediction of a critical parameter line of particularly hard instances. Yet, progress on solving those instances has been scarce for several decades. Here, introducing the Target-SAT (TSAT) algorithm, we roughly triple the tractable problem sizes in the hardest regime, with an even greater improvement in a vast range of neighboring regions. By leveraging statistical information hidden in the combinatorial constraints of the problem, TSAT is actively guided in its stochastic local search toward a target within the relevant parameter space. Our analysis also explains why established local search algorithms are limited to relatively small system sizes due to a vast low-energy trap. Furthermore, we characterize the aforementioned critical line in terms of a dominant additional complexity barrier, whose exponential scaling is quickly overcome by TSAT only in the surrounding parameter space. With TSAT, the lead in solving the hardest known random satisfiability problems returns to the realm of stochastic local search algorithms. 

---
# Representability-Aware Neural Networks for Reduced Density Matrices: Application to Fractional Chern Insulators 

**Authors**: Justin B. Hart, Awwab A. Azam, Thomas Li, Yunxuan Li, Ye Bi, Haining Pan, Jiabin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20326)  

**Abstract**: We develop a representability-aware and interpolable neural network (NN) framework for predicting two-particle reduced density matrices (2-RDMs). The NN incorporates a subset of representability conditions through its architecture and loss function, and can operate on different momentum meshes, enabling evaluating the representability conditions across multiple meshes, which we call interpolated representability condition. The framework can be used either to predict 2-RDMs on large momentum meshes by interpolating exact results from small meshes, or as a variational 2-RDM ansatz optimized by energy minimization on arbitrary meshes. We apply this approach to the fractional Chern insulator in the one-band projected model of twisted bilayer MoTe$_2$ at twist angle $3.89^\circ$ and hole filling $2/3$. Trained on exact-diagonalization (ED) 2-RDMs from meshes with $12$ or $18$ momentum points using six different NN architectures, the best NN is the residual multilayer perceptron, which predicts the $6\times6$ 2-RDM with $97.07\%-98.18\%$ accuracy relative to the ED 2-RDM but predicts an energy $77.353$ meV above ED ground-state energy. We then variationally optimize the NN on several meshes including $6\times6$, predicting a $6\times 6$ energy of just $0.104$ meV below ED while maintaining $98.94\%-98.96\%$ accuracy. Compared with the conventional boundary-point semidefinite programming, which gives an energy $5.560$ meV below ED with $96.40\%-98.94\%$ accuracy, the NN achieves a more accurate energy and similar accuracy while using only less than 1/20 as many parameters. Eventually, we add a symmetric mesh of $48$ momentum points to the variational optimization of the NN, and provide a prediction of the many-body ground-state energy and the many-body quantum metric on that mesh. 

---
# FullFlow: Upgrading Text-to-Image Flow Matching Models for Bidirectional Vision--Language Generation 

**Authors**: Eric Tillmann Bill, Enis Simsar, Alessio Tonioni, Thomas Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.20316)  

**Abstract**: Modern text-to-image diffusion models encode rich visual priors, but expose them only through one-way text-conditioned generation. Existing unified vision--language models derived from them recover bidirectional capability through large-scale joint pretraining or substantial retraining of the text pathway, discarding the strong image prior the text-to-image backbone already encodes. We introduce \emph{FullFlow}, a parameter-efficient recipe that upgrades a pretrained rectified-flow text-to-image model into a bidirectional vision--language generator by training only LoRA adapters and lightweight text heads. FullFlow keeps images in their native continuous flow and adds a discrete insertion process for text. Separate image and text timesteps turn inference into trajectory selection in a two-dimensional generative space, enabling text$\rightarrow$image, image$\rightarrow$text, joint sampling, and partial-text prediction with a single backbone. On Stable Diffusion 3 (SD3) under an identical trainable-parameter count and matched LoRA rank, FullFlow improves text$\rightarrow$image FID from $62.7$ to $31.6$ and image$\rightarrow$text CIDEr from $2.0$ to $99.4$ over a LoRA equivalent following the previous SOTA formulation (Dual Diffusion) at matched wall-clock training time, while reducing peak VRAM from ${\sim}84$\,GB to ${\sim}38$\,GB and raising throughput by ${\sim}8\times$ on two RTX A5000 GPUs in under 24 hours, training only ${\sim}5\%$ of the backbone parameters. The same recipe transfers to FLUX.1-dev and supports downstream VQA through partial-text generation. These results show that strong bidirectional vision--language capability can be unlocked from pretrained text-to-image flow models without full multimodal pretraining. 

---
# Less Data, Faster Training: repeating smaller datasets speeds up learning via sampling biases 

**Authors**: Jingwen Liu, Ezra Edelman, Surbhi Goel, Bingbin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20314)  

**Abstract**: This work investigates the ``small-vs-large gap'', where repeating on fewer samples can lead to compute saving during training compared to using a larger dataset. This is observed across algorithmic tasks, architectures and optimizers and cannot be explained using prior theory. We argue that the speedup comes from appropriate layer-wise growth enabled by sampling biases, which is more pronounced when the dataset size is smaller. We provide both theoretical analysis and empirical evidence from various interventions. Our results suggest that using a smaller dataset with more repetitions is not just a fallback strategy under data scarcity, but can be proactively leveraged as a favorable inductive biases for optimization, particularly in reasoning tasks. 

---
# Tiny-Engram: Trigger-Indexed Concept Tables for Generative Vision 

**Authors**: Runyuan Cai, Yiming Wang, Yu Lin, Xiaodong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.20309)  

**Abstract**: Current personalization methods for generative vision models typically encode new concepts through continuous adapters or weight updates, yet provide limited control over whether and when a concept should be retrieved. In this work, we introduce Tiny-Engram, a compact trigger-indexed concept table that gives visual memories an explicit lexical address and activation boundary inside frozen image and video generators. Tiny-Engram parameterizes each concept as a small set of memory entries indexed by registered n-gram matches, which modulate text-encoder hidden states only within the matched trigger region. Outside this lexical support, the conditioning pathway is identical to that of the frozen base model. Across both single-encoder latent diffusion and multi-encoder diffusion-transformer backbones, this formulation binds a rare trigger phrase to a target identity while preserving compositional control from the surrounding prompt. We further evaluate the same table-based memory in a text-conditioned video generation setting, where the trigger path reliably alters the generated subject but fine-grained identity persistence across held-out video prompts remains limited. Taken together, these results suggest that small, explicitly addressed concept tables are a practical route to modular visual personalization, with strongest evidence in image generation. For video diffusion, the remaining gap points to a broader requirement: temporally stable identity likely depends on tighter coupling between text-side memory and the evolving visual state, motivating future work on memory injection beyond the text-conditioning interface. 

---
# SDM: A Powerful Tool for Evaluating Model Robustness 

**Authors**: Xinlei Liu, Tao Hu, Jichao Xie, Peng Yi, Hailong Ma, Baolin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.20308)  

**Abstract**: Gradient-based attacks are important methods for evaluating model robustness. However, since the proposal of APGD, it has been difficult for such methods to achieve significant breakthroughs. To achieve such an effect, we first analyze the issue of "high-loss non-adversarial examples" that degrades attack performance in previous methods, and prove that this issue arises from inappropriate objectives for adversarial example generation. Subsequently, we reconstruct the objective as "maximizing the difference between the non-ground-truth label probability upper bound and the ground-truth label probability", and proposes a novel and powerful gradient-based attack method named Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step". It adopts the negative probability loss function and the Directional Probability Difference Ratio (DPDR) loss function in the initial and subsequent optimization stages, respectively, and approaches the ideal objective of adversarial example generation via stage-wise sequential optimization. Experiments demonstrate that compared with previous state-of-the-art methods, SDM not only achieves stronger attack performance but also exhibits superior cost-effectiveness. The code is available at this https URL. 

---
# Co-Fusion4D: Spatio-temporal Collaborative Fusion for Robust 3D Object Detection 

**Authors**: Wenxuan Li, Qin Zou, Shoubing Chen, Chi Chen, Yingyi Yang, Shoubing Chen, Qingxiang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2605.20301)  

**Abstract**: In autonomous driving, 3D object detection is essential for accurate perception and reliable decision-making. However, object motion and ego-motion often induce cross-frame spatiotemporal inconsistencies in BEV-based detectors, leading to temporal BEV feature misalignment and degraded spatiotemporal consistency.
To address these challenges, we propose Co-Fusion4D, a unified framework that explicitly preserves cross-frame spatiotemporal consistency and suppresses temporal feature drift. Co-Fusion4D adopts a current-frame-centric strategy, treating the current frame as the primary source of information while selectively incorporating historical frames after spatiotemporal filtering and alignment. This dominant-complementary mechanism effectively mitigates cumulative alignment errors, suppresses noisy feature propagation, and exploits reliable temporal cues for a more consistent BEV representation.
In addition, Co-Fusion4D integrates a Dual Attention Fusion (DAF) module to further enhance spatiotemporal feature interaction. DAF jointly leverages intra-frame spatial attention and inter-frame temporal attention to adaptively align and fuse multi-frame features, emphasizing motion-consistent regions while suppressing spurious correlations. By departing from conventional uniform fusion paradigms, this design substantially improves the temporal stability and discriminative capability of BEV representations.
Extensive experiments on the nuScenes benchmark demonstrate that Co-Fusion4D achieves state-of-the-art performance, with 74.9% mAP and 75.6% NDS, without relying on test-time augmentation or external data. 

---
# Robust Subspace-Constrained Quadratic Models for Low-Dimensional Structure Learning 

**Authors**: Zheng Zhai, Xiaohui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.20300)  

**Abstract**: In this paper, we propose a robust subspace-constrained quadratic model (SCQM) for learning low-dimensional structure from high-dimensional data. Building upon the subspace-constrained quadratic matrix factorization (SQMF) framework, the proposed model accommodates a broad class of noise distributions, including generalized Gaussian and radial Laplace models. This generalization enables reliable performance under both heavy-tailed and light-tailed noise, thereby substantially enhancing robustness across diverse data regimes. To efficiently address the resulting nonconvex optimization problem, we develop a gradient-based algorithm equipped with a backtracking line-search strategy that ensures stable and efficient convergence. In addition, we present a sensitivity analysis of the $\ell_p^p$ and $\ell_2$ loss functions, elucidating their distinct behaviors under varying noise characteristics. Extensive numerical experiments corroborate the theoretical analysis and demonstrate that the proposed approach consistently outperforms existing methods in terms of robustness and reconstruction accuracy. 

---
# Mechanisms of Misgeneralization in Physical Sequence Modeling 

**Authors**: Kento Nishi, Raphael Tang, Karun Kumar, Core Francisco Park, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2605.20299)  

**Abstract**: Generative sequence models are often trained to plan motion in physical domains, from robotics to mechanical simulations. When constructing a dataset to train such a model, engineers may curate demonstrations to specify how trajectories should be distributed over a physical quantity like travel distance or mechanical energy. For example, a roboticist building a maze navigation agent might choose demonstrations whose travel distances cover a fixed range uniformly, hoping to constrain the agent's expected power usage. We find that standard deep learning can violate this intent: each generated trajectory can seem plausible on its own, but the aggregate distribution over the physical quantity is wrong. We call this failure physical misgeneralization, and develop an account of its mechanism. Using controlled synthetic tasks, we show that physical misgeneralization arises when local errors typical of the model class propagate through the physical measurement to shift the recovered distribution. We estimate these errors with a data deviation kernel, and we use it to predict which physical quantities gain or lose mass in both our synthetic and more applied maze navigation and double-pendulum motion tasks. Finally, our mechanistic interpretation helps identify which mitigation strategies are structurally promising, and we use it to propose a kernel-informed intervention. 

---
# Spectral Unforgetting: Post-Hoc Recovery of Damaged Capabilities Without Retraining 

**Authors**: Aarash Abro, Muhammad Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2605.20296)  

**Abstract**: Fine-tuning a language model for a target task routinely degrades capabilities the training data never explicitly threatened. We study this phenomenon, known as catastrophic forgetting, and propose a post-hoc repair solution that uses only the pretrained checkpoint $W_{\mathrm{base}}$ and its fine-tuned descendant $W_{\mathrm{ft}}$. The goal is not merely to revert the model toward the base checkpoint, but to recover capabilities damaged by fine-tuning while preserving both the target-task gains and any beneficial held-out improvements. We introduce DG-Hard, a checkpoint-only spectral repair method for the fine-tuning update $\Delta = W_{\mathrm{ft}} - W_{\mathrm{base}}$. DG-Hard treats $\Delta$ as a low-rank task-aligned signal embedded in an IID-like noise residual that gradient descent has no incentive to remove, and applies the Donoho-Gavish hard singular-value threshold to each weight-delta matrix, keeping the structured high-energy part of the update and removing the spectral bulk. This reduces repair to a closed-form SVD filtering step requiring no data-dependent tuning. A central difficulty is evaluation: average accuracy hides per-benchmark failures, while naive recovery scores reward models that simply revert toward the base. We therefore introduce a partition-conditional metric that separately tracks healing, preservation, non-damage, and target-task retention. Across $14$ (model, task) settings and nine cross-domain held-out benchmarks, DG-Hard achieves the strongest balanced repair among post-hoc baselines. DG-Hard also restores safety alignment degraded by benign fine-tuning on three independent safety axes, despite using no alignment data. These results suggest that part of fine-tuning-induced capability loss is not an unavoidable consequence of specialization, but a removable spectral residue in the weight update itself. 

---
# Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization 

**Authors**: Jinghe Zhang, Daliang Xu, Chenghua Wang, Weikai Xie, Tao Qi, Yun Ma, Mengwei Xu, Gang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20295)  

**Abstract**: Large language models (LLMs) are increasingly deployed on mobile devices, where Neural Processing Units (NPUs) necessitate fully static quantization for optimal inference efficiency. However, existing post-training quantization (PTQ) methods predominantly rely on dynamic activation quantization, rendering them incompatible with NPU hardware constraints. To bridge the gap between high-fidelity PTQ and NPU-constrained inference, we propose this http URL, a integer-only fully static quantization framework. It incorporates learnable quantization parameters and rotation matrices, enabling low-bit activation-weight quantization without runtime quantization parameters re-computation. Crucially, we identify that initialization and selective optimization of quantization parameters is pivotal for optimization stability, as improper initialization and naive joint optimization induce gradient instability that disrupts the optimization of rotation matrices. To address this, we propose a rotation-and-bit-width-aware initialization tailored to diverse activation profiles and a distribution-aware selective optimization (two-stage quantization pipeline) tailored to rotated and unrotated tensors. Furthermore, we introduce a sensitivity-guided adaptive mixed-precision scheme to balance accuracy with inference efficiency. Extensive experiments on real-world mobile NPUs demonstrate that this http URL achieves comparable accuracy to state-of-the-art methods, while reducing inference latency by up to 15.1%. 

---
# Closed-form predictive coding via hierarchical Gaussian filters 

**Authors**: Aleksandrs Baskakovs, Sylvain Estebe, Kenneth Enevoldsen, Kristoffer Nielbo, Chris Mathys, Nicolas Legrand  

**Link**: [PDF](https://arxiv.org/pdf/2605.20293)  

**Abstract**: Predictive coding (PC) offers a local and biologically grounded alternative to backpropagation in the training of artificial neural networks, yet to date, it remains slower, and performance degrades sharply as network depth increases. We trace both problems to a single simplification: current PC networks fix the precision matrix to the identity, discarding precision-weighted prediction errors that the variational derivation requires to be fast, local, and Bayesian. We close this gap by expressing predictive coding networks as deep hierarchical Gaussian filters (HGFs) and restore precision-weighted message passing, yielding dynamic uncertainty estimates and Hebbian-compatible update rules at every layer. The resulting networks can simultaneously learn activations, weights, and precisions under a single free-energy objective, with no global error signal, and resolve inference without requiring iterations or automatic differentiation. On FashionMNIST, our solution approaches backpropagation in epoch-level wall-clock cost while converging in fewer epochs, and outperforms it on online, data efficiency, and concept-drift tasks. We thus establish that closed-form variational inference with online precision learning provides a tractable foundation for deep predictive coding networks, retaining biological and interpretative advantages, without requiring iterative relaxation or global error signals. 

---
# Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers 

**Authors**: Xinzhe Yuan, Xiang Peng, Bin Gu, Huan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.20289)  

**Abstract**: ANN-to-SNN conversion offers a practical, training-free route to spiking large language models. However, current pipelines primarily focus on spike-driven realizations for Transformer linear-algebra operations, while providing limited support for key nonlinear operators. This gap limits compatibility with neuromorphic-style execution constraints, where such nonlinearities typically require division, exponentiation, or norm computations that are not naturally supported by standard leaky integrate-and-fire dynamics. To solve this problem, we propose a plug-and-play framework that implements spike-friendly approximations for Transformer nonlinearities and integrates into existing ANN-to-SNN pipelines. Our method decomposes these nonlinear computations into three recurring primitives -- division, exponentiation, and $\ell_2$ norms -- and realizes them via population computation using LIF neuron groups, combined with lightweight bit-shift scaling to avoid floating-point arithmetic. By composing these primitives as modular operator blocks, our framework supports common Transformer nonlinearities (e.g., Softmax, SiLU, and normalization) without any fine-tuning. Experiments on a range of LLMs Transformers show that selectively replacing the targeted nonlinear operators incurs less than a $1\%$ accuracy drop across all evaluated tasks. 

---
# FusionCell: Cross-Attentive Fusion of Layout Geometry and Netlist Topology for Standard-Cell Performance Prediction 

**Authors**: Haoyi Zhang, Kairong Guo, Bojie Zhang, Yibo Lin, Runsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20287)  

**Abstract**: Standard cells form the building blocks of digital circuits, so their delay and power critically influence chip-level performance; yet characterization still relies on slow simulation sweeps, and many fast predictors ignore layout geometry, missing coupling and layout-dependent effects. The challenge is to jointly represent layout geometry and netlist topology so models capture fine-grained spatial details together with structural connectivity for accurate performance prediction. We introduce FusionCell, a dual-modality predictor that treats routed layout geometry and netlist topology as inputs and fuses them explicitly in a unified model. A DeiT encoder processes three-layer routed layouts, while a graph transformer models heterogeneous device/net graphs. The modalities are integrated through a topology-guided mechanism, where the netlist acts as a structural "map" to actively query relevant physical regions in the layout for joint geometric and topological reasoning. We build a 7nm dataset based on the ASAP7 PDK with over 19.5k cells spanning 149 types using automatic tools, targeting six metrics: signal rise/fall delay, transition, and power. Experimental results demonstrate that FusionCell reduces regression error, with an average MAPE of 0.92 percent, and improves Spearman/Kendall ranking over baselines, while accelerating the characterization process by orders of magnitude compared to circuit simulation. 

---
# Introspective X Training: Feedback Conditioning Improves Scaling Across all LLM Training Stages 

**Authors**: Brandon Cui, Ximing Lu, Jaehun Jung, Syeda Nahida Akter, Hyunwoo Kim, Yuxiao Qu, David Acuna, Shrimai Prabhumoye, Yejin Choi, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20285)  

**Abstract**: We tackle the question of how to scale more efficiently across the many, ever-growing stages of current LLM training pipelines. Our guiding intuition stems from the fact that the dynamics of later stages of the pipeline, e.g. post-training, can be used to inform earlier stages such as pre-training. To this end, we propose Introspective Training (or IXT), inspired by offline reward-conditioned reinforcement learning and applicable to any stage of training. IXT uses a thinking reward model to annotate data with natural language critique based feedback, enabling quality aware training from the earliest stages of the pipeline. Models are then trained by prefix-conditioning the data with the generated feedback -- ensuring that not all tokens are treated equally starting much earlier in training than usual. Comprehensive experiments on 7.5-12B transformer-based dense LLMs trained from scratch all the way up to 18 Trillion tokens seen show that our method: bends scaling curves resulting in up to 2.8x more compute efficiency generally; and reaches performance levels unachievable for models trained otherwise in domains such as math and code. 

---
# JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA 

**Authors**: Hyunju Kang, Woohyun Lee, Jaewon Kim, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.20284)  

**Abstract**: Industrial anomaly detection has been significantly advanced by Large Multimodal Models (LMMs), enabling diverse human instructions beyond detection, particularly through visually grounded reasoning for better image understanding. However, LMMs lack domain-specific knowledge, which limits their ability to generate accurate responses in complex industrial scenarios. In this work, we present JUDO, Juxtaposed Domain-Oriented Multimodal Reasoner, a framework that efficiently incorporates domain knowledge and context in visual and textual reasoning. Through visual reasoning, our model segments the defect region by juxtaposing query images with normal images as visual domain context, enabling a fine-grained visual comparative inspection. Furthermore, we inject domain knowledge through supervised fine-tuning (SFT) to enhance context understanding and subsequently guide domain reasoning through reinforcement learning (GRPO) with tailored rewards, opting for a domain-oriented reasoning process. Experimental results demonstrate that JUDO achieves superior performance on the MMAD benchmark, surpassing models such as Qwen2.5-VL-7B and GPT-4o. These results highlight the importance of enhancing domain knowledge and context for effective reasoning in anomaly understanding. 

---
# Can Vision Models Truly Forget? Mirage: Representation-Level Certification of Visual Unlearning 

**Authors**: Zhenyu Yu, Yangchen Zeng, Chunlei Meng, Guangzhen Yao, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20282)  

**Abstract**: Machine unlearning in Vertical Federated Learning (VFL) has attracted growing interest, yet existing methods certify forgetting solely using output-level metrics. We challenge these claims by introducing Mirage, a representation-level auditing framework comprising four complementary diagnostics: Linear Probe Recovery (LPR), Centered Kernel Alignment (CKA), Feature Separability Scoring, and Layer-Wise Recovery Analysis. Through experiments across seven datasets and seven baseline methods following recent VFL unlearning protocols, Mirage reveals three key findings: (i) Forgetting gap: methods that pass output-level certification still retain substantial class structure in their representations, with LPR exceeding the retrained baseline by up to 15.4 points; CKA shows these models remain structurally closer to the original than to the retrained reference, while separability scores indicate persistent geometric discrimination. (ii) Unlearning trilemma: no existing method simultaneously achieves high utility, output-level forgetting, and representation-level forgetting. (iii) Class-sample asymmetry: class-level forgetting leaves strong representational traces (LPR up to 97%), whereas sample-level forgetting is indistinguishable from chance (LPR approx. 50%); layer-wise analysis further shows residual class information persists across network depths. These findings call for representation-aware evaluation standards in federated unlearning research. 

---
# ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison 

**Authors**: Tianle Li, Xuyang Shen, Yan Ma, Rongxin Guo, Shaoxiang Chen, Jiacheng Chen, Haochen Wang, Hongyang Tang, Yucong Zhou, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.20278)  

**Abstract**: Long-form image captioning exposes a reward granularity problem in RL: captions are judged as whole sequences, while the important errors occur at the level of individual visual claims. A good dense caption should be both faithful and informative, avoiding hallucination without omitting salient details. Yet pairwise preferences, reference-based metrics, and holistic scalar rewards compress these local errors into a single sequence-level signal, obscuring the tradeoff between factuality and coverage. We introduce ClaimDiff-RL, a framework that uses reference-conditioned atomic claim differences as the reward unit for caption RL. Given an image, an actor caption, and a reference caption, a multimodal judge enumerates visually grounded differences, verifies each difference against the image, assigns open-vocabulary error types and severity levels, and produces per-difference statistics for reward composition. This makes hallucinated claims and omitted salient facts separately measurable and tunable. Experiments show that holistic scalar rewards can reduce hallucination by increasing missing facts, while ClaimDiff-RL exposes this faithfulness and coverage tradeoff and enables more balanced operating points. On a 160-image human-labeled diagnostic benchmark, public captioning benchmarks, and VQA benchmarks, ClaimDiff-RL improves the hallucination--missing-fact balance, preserves general capability, and even surpasses Gemini-3-Pro-Preview on several fine-grained Capability dimensions such as object counting, spatial relations, and scene recognition. These results suggest that typed, verifiable claim differences are an effective reward unit for fine-grained and diagnosable caption RL. 

---
# Regulating Anatomy-Aware Rewards via Trajectory-Integral Feedback for Volumetric Computed Tomography Analysis 

**Authors**: Tianwei Lin, Zhongwei Qiu, Jie Cao, Jiang Liu, Wenjie Yan, Bo Zhang, Yu Zhong, Wenqiao Zhang, Yingda Xia, Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20277)  

**Abstract**: Medical vision-language models (VLMs) have rapidly advanced as general-purpose multimodal assistants, yet their deployment in 3D Computed Tomography (CT) analysis remains constrained by a persistent mismatch between optimization objectives and clinical rigor. Current Reinforcement Learning (RL) paradigms still rely on lexical proxy signals that induce ``\textit{Evaluation Hallucinations}'', where models optimize linguistic fluency rather than factual clinical correctness, leading to diagnostically critical errors. To bridge this gap, we introduce the \textbf{Clinical Abnormality Benchmarking Substrate (CABS)}, a structured system that decomposes radiology reports into verifiable clinical semantic units. Using CABS, we identify a ``\textit{Mechanistic Divergence}'' in standard RL, where surface-similarity rewards drive policy gradients to bypass medical facts. We therefore propose \textbf{Trajectory-Integral Feedback GRPO (TIF-GRPO)}, a novel framework integrating control-theoretic principles into policy optimization. By formulating clinical reasoning as a pseudo-temporal trajectory for anomaly discovery, TIF-GRPO regulates anatomy-aware rewards via an integral feedback loop that penalizes persistent omissions as cumulative state errors and suppresses hallucinations as excessive control effort. Experiments on 3D CT benchmarks demonstrate that our approach significantly enhances abnormality detection and clinical faithfulness, establishing a new paradigm for fine-grained regulation in medical VLMs. Our project is available at \href{this https URL}{GitHub}. 

---
# You Don't Need Attention: Gated Convolutional Modeling for Watch-Based Fall Detection 

**Authors**: Sana Alamgeer, Ronish Kumar, Awatif Yasmin, Muhammad Irshad, Anne H. H. Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20275)  

**Abstract**: Existing deep learning approaches for wearable fall detection systems rely on self-attention mechanisms that impose quadratic computational overhead, distributing weights across all time steps. This global weight distribution impairs the precise localization of the brief impact signatures that characterize falls within short, fixed-length windows. To overcome this challenge, we propose Gated-CNN, a lightweight dual-stream architecture that processes accelerometer and gyroscope streams through independent one-dimensional convolutional feature extractors, followed by (i) a sigmoid gating module that selectively suppresses uninformative background activations while amplifying fall-discriminative features, (ii) a global average pooling layer that compresses each stream into a compact fixed-length descriptor, and (iii) a shared classification head that fuses both descriptors for binary fall prediction. For offline evaluation, we evaluate the model across five wrist-mounted inertial measurement unit (IMU) datasets, achieving average F1-scores of 93%, 93%, 90%, 91%, and 90% on SmartFallMM, WEDA-Fall, FallAllD, UMAFall, and UP-Fall, outperforming Transformer baselines. For real-time evaluation, we deployed the model on a Google Pixel Watch 3 and tested across 12 participants. The model achieves an average F1-score of 97% and an accuracy of 98% with zero missed falls, showing that sigmoid gating offers a more structurally aligned and computationally efficient alternative to attention for commodity smartwatch-based fall detection. 

---
# PolycubeNet: A Dual-latent Diffusion Model for Polycube-Based Hexahedral Mesh Generation 

**Authors**: Lu He, Qitao Deng, Junjiang Deng, Liangbin Deng, Yanjun Liang, Wenting Yang, Guoqiang Wang, Na Lei  

**Link**: [PDF](https://arxiv.org/pdf/2605.20274)  

**Abstract**: Hexahedral meshes are widely used in simulation pipelines, yet automatic generation remains challenging for complex CAD geometries. Polycube-based hexahedral meshing is a representative approach due to its regular, parameterization-friendly structure, but existing polycube construction methods often rely on intricate surface segmentation and local heuristics, which can produce artifacts or fail on difficult shapes.
In this paper, we propose an end-to-end framework for polycube generation based on conditional diffusion models. Given an input geometry represented as a point cloud, our method directly produces a corresponding polycube point cloud, eliminating the need for explicit surface segmentation or predefined polycube templates. At the core of our approach is a dual-latent conditional diffusion architecture that confines computationally expensive self-attention operations to a fixed-capacity, low-dimensional latent space. This design effectively decouples computational complexity from the resolution of both the input geometry and the output polycube, thereby avoiding the quadratic cost typical of point cloud self-attention mechanisms while supporting flexible input and output resolutions.
To obtain a hexahedral mesh, the generated polycube is aligned to the input shape via rigid and non-rigid point cloud registration to establish surface correspondence, followed by a polycube-to-hex pipeline. We additionally create and release a paired dataset of CAD meshes and their corresponding polycube meshes, together with the core implementation of our model. Experiments show that PolycubeNet generalizes to complex CAD models with arbitrary genus and produces high-quality polycube structures within seconds, improving robustness and efficiency over prior learning-based approaches. 

---
# Modality-Decoupled Online Recursive Editing 

**Authors**: Siyuan Li, Youyuan Zhang, Fangming Liu, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.20273)  

**Abstract**: Online model editing for multimodal large language models (MLLMs) requires assimilating a stream of corrections under tight compute and memory budgets. Yet editors developed for text-only LLMs often degrade on MLLMs: visually dominant activations skew the statistics that shape updates, causing cross-modal conflict, while sequential writes become entangled in a shared edit space and amplify long-horizon interference, causing inter-edit interference. To address these, we propose M-ORE, a modality-decoupled online recursive editor for lifelong MLLM adaptation. M-ORE is derived from a unified proximal-projection formulation and admits a closed-form update with a Sherman-Morrison recursion, yielding constant per-edit overhead. It maintains module-wise locality statistics for the text stack and the visual projector to avoid visually dominated update shaping and performs continual updates in a fixed orthogonal low-rank edit subspace via a Sherman-Morrison recursion to mitigate long-horizon interference. Experiments on multiple MLLM backbones and online editing benchmarks show that our M-ORE method consistently improves reliability, generality, and locality over strong baselines, while achieving favorable quality-efficiency scaling. Our code is publicly available at this https URL. 

---
# Smaller Abstract State Spaces Enable Cross-Scale Generalization in Reinforcement Learning 

**Authors**: Nasehatul Mustakim, Lucas Lehnert  

**Link**: [PDF](https://arxiv.org/pdf/2605.20272)  

**Abstract**: While humans readily generalize abstract concepts to more complex or larger tasks, building Reinforcement Learning (RL) systems with this ability remains elusive. Here, we present the first theoretical model of how such Out-of-Distribution (OOD) generalization can be achieved in RL agents. Our approach considers Partially Observable Markov Decision Processes (POMDPs) and assumes that an intelligent agent uses an abstraction function to determine which experiences can be treated as equivalent and which must be distinguished. First, we extend the existing state abstraction framework and proof techniques to POMDPs. Then, we define a successor-weighted model reduction, a model reduction variant that enables compression into smaller abstract spaces than prior definitions allow. We derive a bound on the agent's OOD test performance, thereby defining the conditions under which OOD generalization is achievable. This bound decomposes an agent's performance loss into approximation and estimation errors, revealing how reducing an agent's abstract state space size improves test performance and OOD generalization. Our analysis suggests that constraining an agent to operate over a small, finite set of abstract states is necessary for achieving generalization to more complex tasks. Our results motivate further research into learning RL architectures that scale across tasks of varying complexity levels. 

---
# Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs 

**Authors**: Hamed Khosravi, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2605.20270)  

**Abstract**: A local specialist LLM, fine-tuned with reinforcement learning from verifiable rewards (RLVR) on operator-local data, is installed in a regulated organization with per-deployment error budget $\alpha$. The operator needs a safety certificate for this deployment's stream at every round: no pooling across deployments, no waiting for a long-run average. Existing wrappers cannot deliver this on adaptive, online-updated streams: offline conformal-risk methods require exchangeability; online-conformal methods bound only long-run averages; non-exchangeable extensions are marginally valid; and the closest anytime wrapper, A-RCPS, controls marginal rather than selective risk. Using a (test statistic, validity guarantee, deployment rule) framework, we identify one empty cell forced by deployment requirements: e-process per threshold, selective risk, anytime-pathwise validity, max-certified-threshold rule. Conformal Selective Acting (CSA) fills it as a per-round wrapper maintaining a Ville-type e-process per threshold on a Bonferroni grid, evaluated against the RLVR filtration. Under predictable updates and isotonic-calibrated monotone risk we prove (i) an anytime-pathwise selective-risk bound $R_T^{\mathrm{act}}\le\alpha+O(N_T^{-1/2})$, (ii) rate-optimal certification matching $\Theta(\bar\eta^{-2}\log(1/\delta))$, and (iii) a horizon-independent release-rate gap. Across eight specialist benchmarks ($480$ streams), sixteen adversarial distribution-shift cells ($160$ streams), and five live Expert-Iteration RLVR cells with online LoRA over four base models in three architecture families ($10{,}300$ rounds), CSA is the only method among ten compared that satisfies pathwise validity and non-refusing deployment on every cell. We do not propose a new LLM, training algorithm, or policy class; CSA is the deployment-side complement, orthogonal to the model, for operators who cannot use a frontier API. 

---
# Catching a Moving Subspace: Low-Rank Bandits Beyond Stationarity 

**Authors**: Hamed Khosravi, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2605.20269)  

**Abstract**: Many bandit deployments (recommendation, clinical dosing, ad targeting) share two facts prior work handles only in isolation: rewards live on a low-dimensional latent subspace, and that subspace drifts. Stationary low-rank bandits exploit rank but break under subspace change; non-stationary linear bandits adapt to drift but pay ambient rate $\widetilde{O}(d\sqrt{T})$. We study piecewise-stationary low-rank linear contextual bandits with scalar feedback: $\theta_t = B_k^\star w_t$ with rank-$r$ factor $B_k^\star\in\mathbb{R}^{d\times r}$ constant within each of $K$ unknown segments and able to shift at boundaries. Our results are tight along three axes. (i) Identification boundary. With single-play scalar rewards, the moving subspace is recoverable through quadratic functionals of rewards iff three probe-side conditions hold: known noise variance, bounded state-noise coupling, and full-dimensional probe support. Each is necessary in the unrestricted-second-moment problem, and jointly they are sufficient, characterizing the boundary of the solvable region. (ii) Algorithm and dynamic regret. SPSC interleaves isotropic probes with windowed projected ridge-UCB exploitation inside the learned $r$-dimensional subspace; a CUSUM-style variant discovers segment boundaries online. The costed dynamic regret is $\widetilde{O}(r\sqrt{T})+\widetilde{O}(T^{2/3})+O(W\,V_{\mathrm{in}})$, replacing the ambient $d\sqrt{T}$ rate with the intrinsic rank. (iii) Empirics. On eleven benchmarks spanning synthetic, UCI/MovieLens, semi-synthetic clinical, and ZOZOTOWN production-log data, SPSC outperforms non-stationary and low-rank baselines whenever $d-r\gtrsim T^{1/6}$, matching the analytical crossover. To our knowledge, this is the first work to characterize the identification boundary and attain the intrinsic-rank dynamic-regret rate in this setting. 

---
# Chronicle: A Multimodal Foundation Model for Joint Language and Time Series Understanding 

**Authors**: Paul Quinlan, Jeremy Levasseur, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20268)  

**Abstract**: Real-world time series come with text: metadata, descriptions, news, reports. Yet time series foundation models process numerical sequences in isolation, and the multimodal text-and-time-series models that attempt to bridge the two all adapt a pretrained language model post hoc, inheriting representations shaped without ever seeing temporal data. These models are also evaluated almost exclusively against other multimodal baselines, not against the strongest unimodal foundation models in either domain, leaving open whether joint training is needed at all. We present Chronicle, a compact 324M-parameter decoder-only transformer trained from scratch on natural language and time series within a single unified architecture. Both modalities share the same transformer blocks, attention mechanism, and residual stream; the bulk of pretraining uses unimodal batches so cross-modal capability emerges purely from shared parameters, with a short alignment stage that interleaves the two. To our knowledge, Chronicle is the first model jointly pretrained on text and time series from scratch, and the first multimodal model evaluated against dedicated foundation models in both domains. It matches Gemma-3-270M-PT on 19 NLU tasks, sets a new bar for frozen-embedding time series classification on 24 UCR/UEA datasets, and produces multimodal forecasts on Time-MMD that beat every supervised fusion baseline, all from a single backbone. 

---
# Generation of Heterogeneous PET Images from Uniform Organ Activity Maps Using a Pretrained Domain-Adapted Diffusion Model 

**Authors**: Suya Li, Kaushik Dutta, Debojyoti Pal, Jingqin Luo, Kooresh I. Shoghi  

**Link**: [PDF](https://arxiv.org/pdf/2605.20267)  

**Abstract**: Synthetic PET images are valuable for quantitative imaging workflow development, scalable virtual imaging trials, and deep learning model training, but conventional physics-based simulation approaches are computationally intensive, limited in anatomical variability, and often fail to capture heterogeneous PET uptake. This study developed a pretrained domain-adapted diffusion (PAD) model for anatomy-conditioned PET synthesis from uniform organ activity maps. PAD adopts a natural-image pretrained text-to-image decoder with an upstream conditioning encoder and a downstream PET-domain adapter. A two-phase training strategy was used, with the first phase learning coarse uptake distributions and the second refining local image details. Uniform organ activity maps were generated from CT-based segmentations by assigning each organ its mean uptake from the paired PET image. Evaluation included quantitative accuracy, noise assessment, radiomic analysis, tumor segmentation performance, and a human observer study. PAD-generated images achieved high quantitative accuracy, with concordance correlation coefficients above 0.92 between organ mean SUVs and assigned activity values. The synthesized images showed noise levels and texture characteristics similar to target PET images and produced comparable tumor segmentation performance. In a two-alternative forced-choice observer study, four readers achieved approximately 50% accuracy, indicating visual indistinguishability between synthesized and target images. PAD also generated realistic PET images from XCAT-derived activity maps, demonstrating compatibility with phantom-based anatomical priors. Overall, PAD provides a diffusion-based framework for generating clinically relevant heterogeneous PET images from uniform organ activity maps derived from clinical segmentations or digital phantoms, supporting data augmentation and downstream imaging studies. 

---
# Residual Paving: Diagnosing the Routing Bottleneck in Selective Refusal Editing 

**Authors**: Bryce Hinkley, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2605.20262)  

**Abstract**: We study selective refusal editing as a three-way control problem: induce non-refusal on designated edit prompts while preserving benign behavior and harmful refusals outside the edit set. We introduce Residual Paving, a routed residual editing method for frozen instruction-tuned transformers that separates route selectivity, whether to intervene, from residual-edit capacity, what edit to apply. An early-layer router predicts a scalar gate and expert mixture; when active, prompt-conditioned bottleneck residual experts apply later-layer residual updates while leaving the backbone unchanged. This decomposition supports an oracle-routing diagnostic where only the learned scalar gate is replaced with the held-out edit/keep label, leaving the residual editor and frozen backbone fixed. On the primary Gemma-3-4B-IT held-out split, learned Residual Paving reduces edit refusal from 88.6% to 4.0%, with 95.5% benign distribution preservation and 87.3% harmful distribution preservation. Same-protocol one-direction steering controls are much weaker on edit success, leaving edit refusal at 86.8% for Edit-target ActAdd and 78.9% for DIM-style refusal steering. The remaining failure is off-target harmful-keep degradation: harmful refusal remains below the frozen-base rate, 65.3% vs. 81.6%. Across six backbones, oracle routing improves the keep-side diagnostic score on every reported row, with median gain +12.9 pp, supporting the interpretation that learned route selectivity is the main observed bottleneck. Trajectory diagnostics on two backbones further suggest directed movement toward edit-target continuations rather than generic refusal suppression. 

---
# It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs 

**Authors**: Sangwoo Park, Woongyeong Yeo, Seanie Lee, Yumin Choi, Hyomin Lee, Kangsan Kim, Jinheon Baek, Seong Joon Oh, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20258)  

**Abstract**: Contextual Integrity (CI) defines privacy not merely as keeping information hidden, but as governing information flows according to the norms of a given context. As large language models are increasingly deployed as personal agents handling sensitive workflows, adhering to CI becomes critical. However, even frontier models remain unreliable in making disclosure decisions, and existing mitigation strategies often degrade underlying task performance. To overcome this privacy-utility trade-off, we propose SELFCI, a complementary self-distillation framework that decouples information suppression from task resolution. SELFCI jointly optimizes two independent reverse KL divergences over distinct teacher distributions derived from feedback: one encourages preserving task-relevant information for utility, while the other enforces minimal and appropriate disclosure. This complementary formulation induces a Product-of-Experts (PoE) target, aligning the policy with the intersection of capability and privacy requirements. Empirical evaluations demonstrate that SELFCI, without relying on costly external supervision, consistently outperforms competitive baselines such as online reinforcement learning algorithms (e.g., GRPO). These trends further extend to out-of-domain settings involving agentic workflows and accumulated private context, suggesting that SELFCI provides a practical path toward CI alignment. 

---
# Instance Discrimination for Link Prediction 

**Authors**: Valentin Cuzin-Rambaud, Mathieu Lefort, Rémy Cazabet  

**Link**: [PDF](https://arxiv.org/pdf/2605.20257)  

**Abstract**: Recently, instance discrimination models have emerged as a major solution for self-supervised learning. Having already demonstrated its effectiveness in the image domain, instance discrimination learning is now proving equally convincing in the graph domain, in particular for node classification. However, fewer contributions have tackled the link prediction task. In this contribution, we propose to adapt existing methods to this context. We first provide a rigorous evaluation of existing self-supervised models in the field of link prediction, showing that the main performance depends on the augmentation process (like in computer vision). We then propose a new structural augmentation based on the community structure that is relevant for link prediction. Our main contribution introduces two new models, L-GRACE and L-BGRL, based on link representations instead of node representations, which improve the performance of the existing methods, especially on unattributed graphs, and we show that they perform on par with the state of the art, both in supervised and self-supervised contexts. 

---
# FBOS-RL: Feedback-Driven Bi-Objective Synergistic Reinforcement Learning 

**Authors**: Xikai Zhang, Yongzhi Li, Likang Xiao, Yingze Zhang, Yanhua Cheng, Quan Chen, Peng Jiang, Wenjun Wu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20256)  

**Abstract**: Reinforcement learning has become a cornerstone for aligning and unlocking the reasoning capabilities of large-scale models. At its core, the training loop of GRPO and its variants alternates between rollout sampling and policy update. Unlike supervised learning, where each gradient step is anchored to an explicit ground-truth target, the optimal gradient direction for updating model parameters in this setting is not known a priori; the high-quality rollouts drawn during the sampling stage therefore act as the implicit "teacher" that guides every parameter update. However, GRPO adopt a simple sampling scheme that conditions all rollouts on the same original prompt. When a task lies beyond the policy model's current capability, this sampling scheme rarely yields a high-quality rollout, leaving the policy model without a meaningful gradient direction when updating its parameters, which causes training to stall. To address this issue, we propose FBOS-RL, a Feedback-Driven Bi-Objective Synergistic reinforcement learning framework. Specifically, we let the model perform Feedback-Guided Exploration Enhancement based on the feedback provided by the environment, and on top of this we design two mutually reinforcing training objectives: Exploitation-oriented Policy Alignment(EPA) and Exploration-oriented Capability Cultivation(ECC). Extensive experiments demonstrate that EPA and ECC can mutually reinforce each other, forming a positive flywheel effect that significantly improves both the training efficiency and the final performance ceiling of reinforcement learning. Specifically, under an identical number of rollouts, FBOS-RL learns substantially faster than GRPO and feedback-based baselines and ultimately attains a higher performance ceiling, while exhibiting higher policy entropy and lower gradient norms throughout training. 

---
# Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty 

**Authors**: Prakash Aryan, Kaushik Raghupathruni, Timo Kehrer, Sebastiano Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2605.20255)  

**Abstract**: Simulation-based testing of self-driving cars (SDCs) typically relies on scripted or simplified pedestrian models that do not capture the heterogeneity and uncertainty of real human crossing behavior. This limits the realism of safety assessments, especially in scenarios involving jaywalking, which is governed by latent personality traits that the vehicle cannot observe. We hypothesize that jointly training pedestrians and the SDC with multi-agent reinforcement learning (MARL) produces more realistic interaction scenarios than training the SDC against fixed pedestrian policies, and that the resulting behavior gap between predictable and unpredictable crossings can be measured directly from trajectories. This paper describes a MARL environment in which an SDC and 12 pedestrians are co-trained using Multi-Agent Proximal Policy Optimization (MAPPO). Pedestrian locomotion follows scripted Dijkstra pathfinding, while an RL policy controls high-level go/wait decisions. Jaywalking probability depends on a per-pedestrian personality trait sampled at episode start and hidden from the SDC. In 500-episode evaluations, the co-trained SDC reached 78% of goals with a 14% collision rate, compared to 35% goals and 33% collisions for the best rule-based baseline. A speed differential metric shows that the SDC traveled 2.65 m/s faster near jaywalkers than near crosswalk users at close range (0-3 m), indicating that jaywalking encounters were not anticipated. Jaywalking accounted for 13% of crossing events but was associated with 62% of collisions. Co-training with MARL pedestrians reduced collisions by 30% relative to single-agent RL, as pedestrians learned to wait when the SDC approached at speed. 

---
# Efficient Table QA via TableGrid Navigation and Progressive Inference Prompting 

**Authors**: Amritansh Maurya, Navjot Singh, Mohammed Javed, Omar Moured  

**Link**: [PDF](https://arxiv.org/pdf/2605.20254)  

**Abstract**: Large Language Models (LLMs) have shown promising results on NLP tasks, however, their performance on tabular data still needs research attention, because Table Question-Answering (TQA) requires precise cell retrieval and multi-step structured reasoning. Existing work improves TQA either by fine-tuning or training LLMs on task-specific tabular data, but often lacks verifiable control over how the model navigates tables and derives answers. In this work, we propose a training-free TQA approach with two structured prompting frameworks: TableGrid Navigation (TGN), which iteratively navigates rows and columns via a three-module loop to locate evidence and refine answers, and Progressive Inference Prompting (PIP), which enforces columns identification for explicit progressive row selection constraint according to the query. We evaluate 17 LLMs against 6 baselines on TableBench and FeTaQa dataset. On TableBench, TGN improves over the strongest baseline by 3.8 points, and on FeTaQa, PIP achieves SOTA performance over ReAct and Chain-of-Thought. Beyond inference-time gains, PIP and TGN can also serve as supervision templates to fine-tune small models, narrowing the performance gap to much larger architectures in resource-constrained settings, offering versatile and cost-efficient solution for TQA. 

---
# ProcBench: Evaluating Process-Level Defects and Control Preservation in LLM Coding Agents 

**Authors**: Jiawei He, Jie Jia, Chenbo Liu, Chaoyi Xue, Yapeng Song, Xikai Yang, Dong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.20251)  

**Abstract**: Existing benchmarks for LLM coding agents mainly evaluate final outcomes, such as task completion, compilation success, and test pass rates. While these metrics are useful for measuring end-task capability, they provide limited visibility into how an execution unfolds and often miss recurrent process-level failures that arise during multi-step operation. We present ProcBench, a benchmark-oriented framework for evaluating coding-agent trajectories through process defects and control preservation. ProcBench organizes execution failures into a reusable ontology, standardizes heterogeneous logs into a unified trajectory representation, and reports calibrated risk-based scorecards instead of relying only on final outcomes. We instantiate ProcBench on an annotated set of 200 trajectories and apply it across three coding-agent benchmarks: AndroidBench, TerminalBench, and SWE-bench-Verified. Our results suggest that ProcBench can be instantiated with useful reliability, that calibration improves the empirical interpretability of defect findings relative to direct thresholding, and that process-aware scorecards provide diagnostic distinctions beyond conventional outcome-based evaluation. We also discuss limitations, including annotation dependence, partial observability for some defect classes, and the need for broader external validation. 

---
# Automated Kernel Discovery Towards Understanding High-dimensional Bayesian Optimization 

**Authors**: Taeyoung Yun, Woocheol Shin, Inhyuck Song, Jaewoo Lee, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.20249)  

**Abstract**: Gaussian Process (GP) kernels are central to Bayesian optimization (BO), yet designing effective kernels for high-dimensional problems still relies on extensive manual engineering. Existing automated approaches struggle in high dimensions for two bottlenecks: their kernel search space is limited to additions and multiplications of base kernels, and LLM-based approaches require conditioning on raw observations, which becomes infeasible due to context-length limits and the difficulty of extracting meaningful patterns. We introduce \textbf{Kernel Discovery}, a LLM-driven evolutionary framework for high-dimensional BO that searches a broader kernel space beyond predefined composition rules and does not require conditioning on observations. Motivated by the observation that directly prompting an LLM to generate kernel code yields syntactically varied but functionally identical kernels, we adopt a two-stage approach: an LLM first proposes novel mathematical forms, then a second LLM call converts each form into validated, executable code. We also propose a leave-one-out continuous ranked probability score (LOO-CRPS) as a selection criterion that penalizes overfitted kernels. On five high-dimensional BO benchmarks, our method achieves an average rank of \textbf{1.2 out of 17}, outperforming competitive baselines.
We further analyze the discovered kernels to identify which kernels lead to improvements in high-dimensional BO. 

---
# CP-MoE: Consistency-Preserving Mixture-of-Experts for Continual Learning 

**Authors**: Yang Liu, Toan Nguyen, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2605.20247)  

**Abstract**: Catastrophic forgetting remains a major obstacle to continual learning in large language models (LLMs) and vision--language models (VLMs). Although Mixture-of-Experts (MoE) architectures offer an efficient path to scaling, existing LoRA-based MoE continual learning methods still face a fundamental trade-off: they either isolate experts too aggressively, limiting knowledge transfer across tasks, or allow task-specific updates to overwrite important existing parameters, leading to severe forgetting. To address this, we propose CP-MoE, a continual learning framework built around a transient expert that captures early task-specific updates and guides their integration into stable experts. CP-MoE introduces a consistency-preserving routing bias, which uses the transient expert to estimate representation similarity with stable experts and steer routing towards more compatible expert selection, and a transient expert-guided regularisation mechanism, which selectively protects important historical parameters during merging. Together, these components reduce parameter interference and forgetting while preserving cross-task knowledge transfer. We validate CP-MoE on both unimodal and multimodal continual learning benchmarks with LLM-based and VLM-based MoE models. On SuperNI benchmark, spanning diverse sequential language tasks, CP-MoE achieves state-of-the-art performance and stronger zero-shot transfer to unseen tasks. On VQA v2 dataset, it scales effectively to multimodal visual reasoning, consistently reduces forgetting, and outperforms strong MoE baselines. 

---
# GROW: Aligning GRPO with State-Action Modeling for Open-World VLM Agents 

**Authors**: Xiongbin Wu, Zhihao Luo, Shanzhe Lei, Lechao Zhang, Xuhong Wang, Jie Yang, Zhonglong Zheng, Yuanjie Zheng, Xin Tan, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20246)  

**Abstract**: Recently, vision-language model (VLM) agents have shown promising progress in open-world tasks, where successful task completion often requires multiple turns of visual perception and action execution. However, existing methods still rely primarily on Supervised Fine-Tuning (SFT) with expert demonstrations, while the advanced reinforcement learning (RL) algorithm, specifically Group Relative Policy Optimization (GRPO), has not been effectively employed for multi-turn RL in these tasks because standard GRPO requires full trajectories as training samples which leads to excessively long context and noise. To address this issue, we propose GROW, a RL framework for open-world VLM agents that decomposes collected trajectories into state-action samples, and computes advantages between these samples rather than treating a full trajectory as a single entity. We further provide a surrogate analysis indicating that, even though the grouped samples are conditioned on different local states rather than an identical prompt context, the objective can preserve the core relative policy optimization signal of GRPO under simplifying assumptions. Experiments on more than 800 Minecraft tasks show that our method achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of our proposed RL framework for open-world VLM agents. 

---
# Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search 

**Authors**: Jialin Lu, Soonho Kong, Rodrigo Stehling, Kaiyu Yang, Zhangyang Wang, Weiran Sun, Wuyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.20244)  

**Abstract**: We present Lean Refactor, a plug-and-play retrieval-augmented agentic framework for multi-objective, controllable, and version-robust refactoring of Lean proofs. LLM-generated proofs are notoriously correct-but-verbose and brittle across library versions, yet existing refactoring works overlook three practical challenges: 1) Lean refactoring is natively multi-objective (proof length, compilation cost, and version compatibility are often in tension); 2) Lean repositories have fragile compatibility, whereas LLM releases are unaware of Lean/Mathlib versions; 3) Training-based pipelines require repeated fine-tuning with each new LLM release, scaling neither with model churn nor with Lean's release cycle. Lean Refactor steers a frozen agentic LLM with retrievals from a curated database of multi-objective refactoring strategies, each densely annotated with metadata such as supported Lean/Mathlib versions and expected compilation-cost reduction. Experiments show over $70\%$ token-level compression on competition benchmarks, over $20\%$ on research repositories, and up to $60\%$ compilation-time reduction, outperforming prior work and Claude Code. Version-filtered retrieval further improves compression on the target Lean version, and refactored miniF2F proofs exhibit stronger zero-shot version transfer to future Lean releases than their unrefactored counterparts. 

---
# LEAP: A closed-loop framework for perovskite precursor additive discovery 

**Authors**: Xin-De Wang, Zhi-Rui Chen, Ze-Feng Gao, Peng-Jie Guo, Cheng Mu, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20242)  

**Abstract**: Efficient discovery of precursor additives is essential for improving the performance of perovskite solar cells, yet the large chemical space makes conventional trial-and-error screening inefficient. We develop LEAP(LLM-driven Exploration via Active Learning for Perovskites), an expert-in-the-loop closed framework that couples a domain-specialized large language model(LLM) with active learning for iterative additive prioritization. The LLM is trained to extract mechanism-relevant knowledge from the perovskite additive literature and to represent candidate molecules through interpretable descriptors, which are further integrated into a Bayesian optimization workflow for uncertainty-aware prioritization under low-data conditions. Benchmark results on unseen literature show that the domain-specialized model outperforms general-purpose models in mechanism-consistent reasoning. Experimental validation in an expert-in-the-loop proof-of-concept study suggests improved additive prioritization across three screening rounds, leading to average device PCEs of 20.13% and 20.87% for the later-round 6-CDQ- and 2-CNA-treated devices, respectively, compared with 19.25% for the control, with a champion PCE of 21.32%. These results provide preliminary evidence that literature-grounded mechanistic descriptors, when coupled with Bayesian optimization and expert feasibility review, can support mechanism-aware additive prioritization in perovskite photovoltaics. 

---
# Geometry-Lite: Interpretable Safety Probing via Layer-Wise Margin Geometry 

**Authors**: Woo Seob Sim, Yu Rang Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.20241)  

**Abstract**: Prompt-level safety probes for large language models use hidden-state representations to separate safe from unsafe prompts, but strong average detection performance does not explain the geometry of this separation. In particular, it remains unclear how safety evidence is formed across layers, which aspects of that layer-wise geometry support low-false-positive decisions, and which geometric biases remain stable under benchmark shift. We study this as an empirical decomposition problem and introduce Geometry-Lite, a compact prompt-level probe that maps each layer's final prompt-token representation to signed margins under centroid, local-neighborhood, and supervised linear-boundary readouts, then summarizes the resulting margin profiles by boundary position, layer-to-layer change, and coarse shape. Across nine instruction-tuned backbones ($1.2$B--$70$B) and seven safety benchmarks, Geometry-Lite improves over single-layer probes while remaining close to raw multi-layer score stacking, making it a useful instrument for analyzing the multi-layer safety signal. The decomposition shows that safety evidence is expressed primarily through persistent boundary-position geometry: final or extremal margins and unsafe-side layer occupancy dominate aggregate detection performance. In contrast, finite-difference drift and structural summaries add little to pooled AUROC, although drift can provide small recall-oriented corrections under shifted low-FPR thresholds. Under benchmark shift, optimized linear boundaries are sharp on the training mixture, whereas class-conditional mean geometry retains separation more reliably on a predefined hard held-out subset. Overall, prompt-level safety evidence is not primarily a layer-to-layer motion signal, but a persistent layer-wise margin geometry whose useful components and readout-level biases become visible in decision-critical regimes. 

---
# Provably Learning Diffusion Models under the Manifold Hypothesis: Collapse and Refine 

**Authors**: Wei Huang, Andi Han, Mingyuan Bai, Huanjian Zhou, Qixin Zhang, Taiji Suzuki, Kenji Fukumizu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20235)  

**Abstract**: Diffusion models generate high-dimensional data with remarkable quality, yet how their training efficiently learns the score function, bypassing the curse of dimensionality when data is supported on low-dimensional manifolds, remains theoretically unexplained. We identify a collapse-and-refine mechanism driven by the geometry of the score function itself: at small noise scales, the diverging singularity of the score drives a rapid dimensional collapse of the induced denoising map onto the data manifold projection; at moderate noise scales, training refines the intrinsic density on the learned manifold. We instantiate this principle as Score-induced Latent Diffusion (SiLD), a two-stage framework in which both manifold learning and density estimation emerge from a single denoising score matching objective, replacing the heuristic KL regularization of VAE-based latent diffusion models. We prove that the resulting sample complexity depends on the intrinsic dimension rather than the ambient dimension. Experiments on Stacked MNIST, CelebA variants, and molecular generation benchmarks show that SiLD matches or outperforms VAE-based LDMs in generation quality and consistently improves reconstruction, validating our theoretical predictions. 

---
# TabPFN-MT: A Natively Multitask In-Context Learner for Tabular Data 

**Authors**: Cormac Cureton, Narges Armanfard  

**Link**: [PDF](https://arxiv.org/pdf/2605.20234)  

**Abstract**: Prior-Data Fitted networks (PFNs) have been very successful in tabular contexts, handling prediction tasks in context. However, they are designed for single-task inference, meaning that predicting several target values within a context requires repeated forward calls and precludes inter-task information sharing. We propose TabPFN-MT, which is trained on an expanded multi-target synthetic prior to capture inter-task dependencies in context. This model uses an expanded $y$-encoder and a shared decoder head to enable multitask in-context learning and simultaneous inference. The model is uniquely specialized for small-to-medium datasets by relying on in-context learning rather than traditional gradient-based training. Within this regime (averaging fewer than 1,000 samples), extensive evaluations across 344 datasets demonstrate that TabPFN-MT establishes a new state-of-the-art for deep tabular multitask learning. Furthermore, despite the inherent compute asymmetry of joint optimization, our model remains highly competitive with the latest state-of-the-art single-task ensembles. Notably, on multitask datasets it achieves an overall Accuracy rank of 4.89, the highest average rank among all models tested. Crucially, TabPFN-MT delivers this highly competitive performance while reducing the inference cost for $T$ tasks from $O(T)$ to $O(1)$ forward passes, offering a massive computational efficiency improvement for multi-target tabular applications. 

---
# AI-Assisted Competency Assessment from Egocentric Video in Simulation-Based Nursing Education 

**Authors**: Hanchen David Wang, Yilin Liu, Madison J. Lee, Surya Chand Rayala, Gautam Biswas, Daniel T. Levin, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.20233)  

**Abstract**: Assessing learner competency in clinical simulation requires expert observation that is time-intensive, difficult to scale, and subject to inter-rater variability. Vision-language models have emerged as a promising tool for understanding complex visual behavior. In this work, we investigate whether visual observations can provide educationally meaningful signals for competency assessment through a three-stage framework that (1) extracts action timelines from egocentric nursing simulation video using frozen visual encoders and few-shot learning, (2) derives sequence-level features and per-session recognition metrics, and (3) relates these to instructor-rated competency. Across 22 densely annotated sessions (3.8 hours, 493 actions), a frozen DINOv2 backbone with HMM Viterbi decoding achieves 57.4% MOF in leave-one-out 1-shot recognition. Surprisingly, we observe a negative trend between recognition accuracy and competency (rho = -0.524, p = 0.012 for mIoU), robust to six confound controls: more competent students produce diverse, harder-to-classify workflows, while simple sequence features show no such relationship. Per-item analysis identifies patient safety protocols and team communication as the expected behaviors most reflected in this pattern, and process model comparisons reveal that higher-competency students exhibit more protocol-consistent action transitions. These findings suggest that recognition accuracy may complement predicted action timelines as a pedagogically informative signal in automated competency assessment. 

---
# Network-Based Interventions for HIV Prevention via Cascade-Aware Suppression of Transmission 

**Authors**: Akseli Kangaslahti, Davin Choo, Milind Tambe, Alastair van Heerden, Cheryl Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2605.20218)  

**Abstract**: Treating and preventing Human Immunodeficiency Virus (HIV) remains a critical global health challenge. While antiretroviral therapy provides a path toward viral suppression -- effectively eliminating an individual's transmission risk -- systemic resource constraints limit the reach of intervention efforts. This work addresses the strategic distribution of intensive resources among virally unsuppressed individuals to minimize the expected cascade of new infections within a transmission network. We formalize this challenge as a novel constrained optimization problem where we have resources to "treat" $k$ out of a set $\mathbf{P}$ of virally unsuppressed individuals, and establish its theoretical connections to existing computational literature. We then propose Cascade-Aware Suppression of Transmission (CAST), a polynomial-time $(\delta, \epsilon)$-approximation algorithm that achieves a $2\sqrt{|\mathbf{P}|}$ approximation ratio by leveraging connections to the Minimum-$k$-Union (MkU) problem and Hoeffding-style concentration bounds. Extensive evaluations on real-world HIV networks demonstrate that CAST outperforms standard public health and computer science baselines. Furthermore, we show that CAST is empirically robust across diverse infectious disease networks, varied edge probability initializations, and settings involving imperfect network data. 

---
# Leveraging Vision-Language Models to Detect Attention in Educational Videos 

**Authors**: Gabriel Becquet, Sébastien Lallé, Vanda Luengo, Ali Abou-Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2605.20211)  

**Abstract**: Educational videos are a cornerstone of remote and blended learning. However, learners' fluctuating attention remains a significant barrier to effective information retention. Prior research has attempted to mitigate this by detecting and reacting to attention loss at runtime using eye tracking. Such detection has been based so far on classical machine learning classifiers trained on engineered features, such as summary statistics over learners' fixations and saccades. These methods have struggled to capture the complex, temporal nature of learner engagement, thus exhibiting moderate prediction performance. In this study, we aim to advance the detection of attention by shifting from standard engineered features to a multimodal foundation models. Using an educational eye-tracking dataset (N = 70), we investigate a novel methodology that utilizes a Vision-Language Model (VLM) to analyze video content directly with superimposed gaze data. This approach aims to leverage the semantic reasoning capabilities of foundation models to contextualize learner focus within the video stream. We evaluate the performance of this VLM-based approach using several prompting strategies with Gemini 3, but ultimately found that none of them could outperform statistical baselines. Our results provide new insights into the limitations of using VLMs for real-time educational diagnostics. 

---
# Governance by Design: Architecting Agentic AI for Organizational Learning and Scalable Autonomy 

**Authors**: Nelly Dux, Cristina Alaimo, Philippe Roussiere, Abhishek Kumar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.20210)  

**Abstract**: Agentic AI systems - systems that can pursue goals through multi-step planning and tool-mediated action with limited direct supervision - are moving from experimental prototypes to enterprise deployments. This transition introduces tensions in implementation, scaling, and governance: organizations seek scalable autonomy for knowledge and coordination work, yet must preserve accountability, safety, cost control, and responsibility as systems initiate actions, access enterprise data, and evolve through iterative updates. Building on an in-depth qualitative case of a large IT services company's 2025 development and staged rollout of an agentic system integrated with enterprise tools; we show that governance is implemented through concrete architectural and working arrangements that determine what the system is allowed to do, which tools and data it can use, how memory is handled, and how performance improvements are introduced over time. We then distill seven lessons that explain how to build effective governance into agentic AI during operationalization and scaling. 

---
# PrivacyAkinator: Articulating Key Privacy Design Decisions by Answering LLM-Generated Multiple-choice Questions 

**Authors**: Qiyu Li, Yuen Sum Wong, Yuen Kei Wong, Longxuan Yu, Haojian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.20206)  

**Abstract**: NIST's Privacy Risk Assessment Methodology (PRAM) provides a structured framework for privacy experts to assess privacy risks. However, its complexity and reliance on expert knowledge make it difficult for novice developers to use effectively. This paper explores methods to lower these barriers. We first performed an observational study with 12 participants using PRAM in real-world scenarios, and found that novice developers struggled most with articulating privacy-related design decisions. We then developed PrivacyAkinator, an interactive tool that helps developers articulate key privacy decisions by answering LLM-generated multiple-choice questions. PrivacyAkinator introduces three innovations: a universal privacy representation that abstracts privacy-related design decisions into data flows and stakeholder interactions; a domain-aware design space mined from 10K privacy-related news articles; and a dynamic question-generation workflow to prioritize relevant questions. Our user study with 24 participants suggests that developers using PrivacyAkinator identified 47% more key decisions in 73% less time compared to PRAM. 

---
# RealUserSim: Bridging the Reality Gap in Agent Benchmarking via Grounded User Simulation 

**Authors**: Ming Zhu, Juntao Tan, Rithesh Murthy, Jielin Qiu, Liangwei Yang, Wenting Zhao, Silvio Savarese, Shelby Heinecke, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20204)  

**Abstract**: LLM-based user simulation is the primary mechanism for end-to-end agent evaluation, yet simulated users are poor proxies for real humans: unconstrained LLM defaults produce a Formalism Ceiling (style match rates of 6-8% against real users), while hand-crafted behavioral directives trigger Directive Amplification, where models hyper-interpret instructions into unnatural behavioral extremes that vary dramatically across simulator models. We present RealUserSim, the first user simulation framework grounded in real behavioral data. From 14,000+ authentic human-LLM conversations (WildChat), we extract 7,275 executable behavioral profiles and use them to ground LLM simulators. A fidelity benchmark (PT3) on 600 conversations across 71+ domains with anti-leakage controls shows that grounded simulation raises match rate from 24.2% to 45.3% across five behavioral dimensions. Agent evaluation on TauBench with 6 simulator models and extensive analysis shows that grounded simulation acts as a realistic stress test, surfacing three failure mechanisms invisible to cooperative simulators (mean -3.2% to -3.5% task success degradation), while Directive Amplification in existing benchmarks produces unrealistic behavior that compromises the validity of agent evaluation. 

---
# GrandGuard: Taxonomy, Benchmark, and Safeguards for Elderly-Chatbot Interaction Safety 

**Authors**: Changxuan Fan, Xi Yang, Yueyuan Zheng, Bin Zhou, Yuanping Wang, Wenbin Hu, Huihao Jing, Ki Sen Hung, Dazhao Du, Haoran Li, Janet Hui-wen Hsiao, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.20203)  

**Abstract**: As older adults increasingly use LLM-based chatbots for companionship and assistance, a safety gap is emerging. Older adults may face vulnerabilities from social isolation, limited digital literacy, and cognitive decline, yet existing safety benchmarks largely target general harms and overlook elderly-specific risks. For example, a prompt such as "how to repair a ceiling light alone in the dark" may be benign for most users but poses a serious fall risk for older adults with mobility limitations. We introduce GrandGuard, the first comprehensive framework for assessing and mitigating elderly-specific contextual risks in LLM interactions. We develop a three-level taxonomy with 50 fine-grained risk types across mental well-being, financial, medical, toxicity, and privacy domains, grounded in real-world incidents, community discussions, and analysis of stakeholder studies. Using this taxonomy, we construct a benchmark of 10,404 labeled prompts and responses, showing that several leading LLMs mishandle elderly-specific contextual risks in over 50% of cases. We mitigate these failures with two safeguards: a fine-tuned Llama-Guard-3 and a policy-enhanced gpt-oss-safeguard-20b, achieving up to 96.2% and 90.9% unsafe-prompt detection accuracy, respectively. GrandGuard lays the groundwork for AI systems that move beyond general safety to support aging populations. 

---
# Under Pressure: Emotional Framing Induces Measurable Behavioral Shifts and Structured Internal Geometry in Small Language Models 

**Authors**: Rana Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2605.20202)  

**Abstract**: I study whether emotionally framed evaluation follow-ups change both the behavior and the calm-relative internal representations of small, locally deployed language models. Our main benchmark uses Qwen 3.5 0.8B on four impossible-constraint coding tasks and eight follow-up framings: calm, pressure, urgency, approval, shame, curiosity, encouragement, and threat. In the 0.8B eight-condition sweep (160 conversations), pressure produces the strongest shortcut markers (11/20 runs) and the clearest overfit pattern (3/20), while calm and curiosity preserve explicit honesty more often (7/20 and 6/20). For all seven non-baseline conditions, the corresponding calm-relative direction vectors peak at the final transformer layer. An exploratory PCA of the layer-23 direction vectors reveals a dominant first component (59.5% explained variance) aligned with a hand-labeled positive/negative split (cosine alignment 0.951); approval and urgency are nearly identical internally (cosine 0.957), whereas curiosity points away from urgency (-0.252). In a separate calm-vs.-pressure rerun used for scale comparison, Qwen 3.5 2B shows higher honest rates under calm framing and directionally consistent activation steering on a small 4-prompt A/B probe, whereas the 0.8B steering result reverses. I interpret these results as evidence for measurable prompt-sensitive control directions in small open models, while stopping short of claiming intrinsic emotional states. 

---
# Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning 

**Authors**: Miao Li, Irina Saparina, Alexander Gurung, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2605.20201)  

**Abstract**: Recent large language models support inputs of up to 10 million tokens, yet they perform poorly on long-context tasks that require complex reasoning. Such tasks can be solved using only a subset of the input -- a proxy context -- rather than the full sequence. Despite sharing the same underlying reasoning process, models exhibit a significant performance disparity between proxy and full contexts. To improve long-context reasoning, we propose ProxyCoT, a novel training framework that transfers reasoning capabilities from short proxy contexts to full long contexts. Specifically, we first obtain high-quality chain-of-thought reasoning traces on proxy contexts through reinforcement learning or distillation from a larger teacher model, and then ground the generated traces in full long contexts with supervised fine-tuning. Experiments across different datasets demonstrate that ProxyCoT consistently outperforms strong baselines with reduced computational overhead. Furthermore, models trained with ProxyCoT generalize their long-context reasoning capabilities to out-of-domain tasks. 

---
# Evaluating multimodal emotion recognition in proactive conversational agents: A user study 

**Authors**: Adnana Dragut, Raquel Lacuesta, F. Xavier Gaya-Morey, Jose M. Buades-Rubio  

**Link**: [PDF](https://arxiv.org/pdf/2605.20200)  

**Abstract**: This article presents a multimodal emotion recognition module integrated into a proactive Socially Interactive Agent (SIA) powered by generative artificial intelligence. The system evaluates real-time affective states through two distinct channels: a computer vision-based facial recognition module and a semantic linguistic analysis engine. To validate the framework, an empirical study was conducted with 20 users who engaged in dynamic, unscripted dialogues with the conversational agent. The findings reveal a significant discrepancy between automated visual cues and actual internal emotional states. When interacting with the AI, users consistently exhibited a "poker face" effect, displaying serious, concentrated facial expressions even when experiencing positive emotions. Consequently, the generative AI linguistic analysis proved significantly more reliable, by contextualizing the users' verbal expressions. Furthermore, an analysis of the interaction dynamics demonstrated that SIAs can effectively elicit specific emotions by adapting conversational themes and employing structured linguistic patterns, such as empathetic or humorous language. However, the study also noted that instances of uncalibrated proactivity occasionally led to user disengagement and a perception of artificiality. Ultimately, this research highlights the necessity of refining SIAs to dynamically adapt to users' emotional evolution, relying on deep linguistic context to foster more natural, human-like interactions. 

---
# FlowLM: Few-Step Language Modeling via Diffusion-to-Flow Adaptation 

**Authors**: Runzhe Zhang, Letian Chen, Wenpeng Zhang, Zhouhan Lin, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20199)  

**Abstract**: We present FlowLM, a flow matching language model transformed from pre-trained diffusion language models via efficient fine-tuning. By re-aligning the curved sampling trajectories of diffusion models into straight-line flows, FlowLM enables high quality few-step generation that rivals or even outperforms the quality of 2,000-step diffusion sampling with very few training epochs. Remarkably, finetuned FlowLM reaches performance saturation with only half as many training epochs as training from scratch, both approaches greatly outperforming the original diffusion model, thereby validating our method. Furthermore, we validate a more effective training objective for flow matching: predicting clean data to consistently guide the sampling process towards the true data distribution. Empirical results demonstrate that our approach is highly effective for high-quality, few-step text generation. 

---
# Data Scaling as Progressive Coverage of a Predictive Contribution Spectrum 

**Authors**: Zihui Song, Shihao Ji, Hongxi Li, Shuaizhi Cheng, Chunlin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20196)  

**Abstract**: We investigate the hypothesis that real-data scaling laws are governed by progressive coverage of a latent predictive contribution spectrum rather than by token-frequency tails alone. We work with a suffix-automaton representation of text corpora and define a data-intrinsic global-KL predictive contribution spectrum, in which each state contributes according to its empirical mass times its KL deviation from a global next-token baseline. Across 12 real corpora, the tail slope of this spectrum is already strongly correlated with the empirical data-scaling exponent of a fixed small GPT learner. We then go beyond slope correlation and define, for each training size N, an effective truncation rank K(N) by matching the observed excess loss to the residual tail mass of the prepared 1000k global-KL spectrum. Empirically, log K is close to linear in log N, with pooled R^2 about 0.96 for the raw spectrum and R^2 about 0.90 for the smoothed spectrum. These findings provide strong empirical support for a simple mechanism picture: training scale advances an effective frontier through a predictive state spectrum, and the residual tail mass of that spectrum tracks the remaining excess loss. 

---
# Pseudo-Siamese Network for Planning in Target-Oriented Proactive Dialogues 

**Authors**: Xinyue Kang, Maodong Li, Yibin Zheng, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.20195)  

**Abstract**: A target-oriented proactive dialogue system is designed to steer conversations toward predefined targets while actively providing suggestions. The core paradigm of such a system is to plan a reasonable dialogue path and subsequently guide language models (e.g., pre-trained or large language models) to generate responses, where dialogue path planning serves as the central component-a novel yet under-explored problem. In this work, we propose a Forward-Focused Bidirectional Pseudo-Siamese Network (FF-BPSN) for dialogue path planning toward predefined dialogue targets. FF-BPSN employs two identical transformer-based decoders for forward and backward planning, together with a forward-focused module that integrates bidirectional information to construct the final forward path. This path benefits from bidirectional planning while prioritizing forward information. We then employ the planned path to guide language models in response generation. Extensive experiments on DuRecDial and DuRecDial 2.0 demonstrate that FF-BPSN achieves state-of-the-art performance in dialogue path planning and significantly enhances the effectiveness of target-oriented proactive dialogue systems. 

---
# Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction 

**Authors**: Aisvarya Adeseye, Jouni Isoaho, Adeyemi Adeseye  

**Link**: [PDF](https://arxiv.org/pdf/2605.20194)  

**Abstract**: Large language models (LLMs) have been increasingly used to analyze text. However, they are often plagued with contextual reasoning limitations when analyzing long documents. When long documents are processed sequentially, early or dominant concepts can overshadow less visible but meaningful interpretations, leading to cumulative analytical bias, omission error, and over-generalization. Additionally, independently generated outputs are often merged without systematic grounding, introducing redundancy, conceptual drift, and unsupported claims. This study proposes a structured framework combining parallel chunk-level processing with evidence-anchored consolidation. Texts are first divided into semantically coherent chunks and processed independently in parallel to remove influence from earlier processing. The independently generated interpretations are then consolidated using explicit evidence anchoring and prioritization that reduces dominance and over-generalization while improving traceability. Experiments with multiple model types and sizes indicate that parallel processing significantly reduces omission error by approximately 84%, increases evidence traceability by up to 130%, and reduces unsupported claims by up to 91%. Smaller models benefited most, suggesting that efficient parallel chunking and consolidation play a critical role in achieving reliable and scalable textual analysis. 

---
# Improving Quantized Model Performance in Qualitative Analysis with Multi-Pass Prompt Verification 

**Authors**: Aisvarya Adeseye, Jouni Isoaho, Adeyemi Adeseye  

**Link**: [PDF](https://arxiv.org/pdf/2605.20193)  

**Abstract**: Quantized Large Language Models (LLMs) are used more often in qualitative analysis because they run fast and need fewer computing resources. This study examines how different lower bits quantization levels (8-bit, 4-bit, 3-bit, and 2-bit) and quantization types affect the performance of LLaMA-3.1 (8B) on qualitative analysis. The study uses expert and non-expert responses from 82 interview transcripts. Low-bit models often produce higher levels of hallucinations and unstable results, especially when reading non-expert language with unclear terms. To improve performance, we propose a quantization-aware multi-pass prompt verification method. This method guides the model through controlled steps that reduce hallucinations. It removes unreliable content and passes the results to the next transcript after verification, improving accuracy. To validate performance, human coders analyzed transcripts using NVivo and BF16 LLaMA. BF16 LLaMA-3.1 produced high-precision output but had semantic drift and hallucination. These errors were corrected manually. The corrected BF16 output and NVivo human coding were combined to create a gold-standard ground truth (GSGT) for thematic extraction and frequency analysis. The results show that 8-bit models stay closest to the GSGT. The 4-bit models lose accuracy but become stable when the proposed method is applied. The 3-bit and 2-bit models drop in performance because of heavy compression, but they improve with the proposed prompt design and verification. The study also finds that models at the same bit level behave differently depending on quantization type. Overall, the method helps low-resource LLMs become more stable, accurate, and suitable for qualitative research at lower cost. 

---
# GraphDiffMed: Knowledge-Constrained Differential Attention with Pharmacological Graph Priors for Medication Recommendation 

**Authors**: Krati Saxena, Tomohiro Shibata  

**Link**: [PDF](https://arxiv.org/pdf/2605.20188)  

**Abstract**: Recommending safe and effective medication combinations from electronic health records (EHRs) is a core clinical AI problem, yet it remains difficult because patient trajectories are long, noisy, and clinically heterogeneous. Existing methods typically excel at either temporal modeling across visits or pharmacological knowledge integration (e.g., drug-drug interactions, DDIs), but rarely achieve both while robustly suppressing noise. We present GraphDiffMed, a knowledge-constrained medication recommendation framework built on dual-scale Differential Attention v2. Differential attention is applied at both intra-visit and inter-visit levels to filter spurious signals within encounters and across longitudinal history, while pharmacological constraints are incorporated during learning. Experiments on MIMIC-III and ablation studies show that this design consistently improves recommendation quality and ranking over strong baselines while achieving a more favorable safety performance balance. We further find that the strongest-performing configuration uses only demographic auxiliary features under our experimental setting. Overall, GraphDiffMed demonstrates that combining noise-aware attention with pharmacological constraints yields more reliable and clinically meaningful medication recommendation. We open-source our code at this https URL. 

---
# Neural Estimation of Pairwise Mutual Information in Masked Discrete Sequence Models 

**Authors**: Jai Sharma, Yifan Wang, Bryan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.20187)  

**Abstract**: Understanding dependencies between variables is critical for interpretability and efficient generation in masked diffusion models (MDMs), yet these models primarily expose marginal conditional distributions and do not explicitly represent inter-variable dependence. We propose a neural framework for estimating pairwise conditional mutual information (MI) directly from the hidden states of a pretrained MDM, using ground-truth MI computed from the model's own conditional distributions for supervision. The resulting estimator captures the model's internal belief about dependency structure and predicts the full MI matrix in a single forward pass, enabling MI-guided parallel decoding by identifying conditionally independent subsets of variables. We evaluate our approach on Sudoku and protein sequence generation with ESM-C, where the MI maps recover known structural constraints and enable a 3-5x magnitude reduction in inference-time forward passes compared to sequential decoding, while preserving generative quality and outperforming entropy-based parallelization methods. 

---
# Diverge to Induce Prompting: Multi-Rationale Induction for Zero-Shot Reasoning 

**Authors**: Po-Chun Chen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.08028)  

**Abstract**: To address the instability of unguided reasoning paths in standard Chain-of-Thought prompting, recent methods guide large language models (LLMs) by first eliciting a single reasoning strategy. However, relying on just one strategy for each question can still limit performance across diverse tasks. We propose Diverge-to-Induce Prompting (DIP), a framework that first prompts an LLM to generate multiple diverse high-level rationales for each question. Each rationale is then elaborated into a detailed, step-by-step draft plan. Finally, these draft plans are induced into a final plan. DIP enhances zero-shot reasoning accuracy without reliance on resource-intensive sampling. Experiments show that DIP outperforms single-strategy prompting, demonstrating the effectiveness of multi-plan induction for prompt-based reasoning. 

---
