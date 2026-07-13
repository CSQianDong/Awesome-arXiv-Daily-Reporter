# ConceptSMILE: Auditing the Trustworthiness of Concept-Based Explainable AI 

**Authors**: Mohadeseh Mollapour, Koorosh Aslansefat, Zeinab Dehghani, Bhupesh Kumar Mishra, Tejal Shah, Zhibao Mian  

**Link**: [PDF](https://arxiv.org/pdf/2607.09649)  

**Abstract**: Concept-based explainable artificial intelligence (AI) can make model reasoning more human-understandable, but concept-level outputs are not automatically trustworthy. We introduce ConceptSMILE, a model-agnostic perturbation-based auditing framework for evaluating the reliability of concept-based explanations. Rather than replacing SMILE, ConceptSMILE extends its perturbation-based logic from feature- or region-level attribution to the auditing of human-understandable concept explanations. The framework perturbs input regions, measures concept-response shifts, applies locality weighting, and fits an XGBoost surrogate to approximate local concept behaviour. Reliability is assessed through attribution accuracy, surrogate fidelity, faithfulness, stability, and consistency. We evaluate ConceptSMILE on retinal fundus images by comparing MedSAM-derived visual concepts with VLM-based semantic concepts. Results show that reliability varies across concepts and pathways: MedSAM achieves stronger spatial attribution and the highest surrogate fidelity ($R^2 = 0.8503$, $R_w^2 = 0.8465$), while the VLM pathway shows stronger vessel faithfulness and stronger stability under selected artefact conditions. ConceptSMILE provides an independent audit layer for evaluating the trustworthiness of concept-based XAI. 

---
# Agora: Enhancing LLM Agent Reasoning Via Auction-Based Task Allocation 

**Authors**: Kaiji Zhou, Ales Leonardis, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2607.09600)  

**Abstract**: Enhancing the reasoning capabilities of large language model (LLM) agents requires effective orchestration of diverse expert models and tools. However, existing frameworks typically call APIs based on coarse-grained matching between tasks and the functions of expert models or tools, while overlooking critical factors such as performance variability and cost efficiency among functionally similar alternatives. To address this, we propose Agora, a framework that introduces an incentive-compatible auction mechanism for dynamically allocating tasks to expert models and tools. By treating reasoning steps as tradeable items, Agora enables agents to bid based on their rectified competence-ensuring that critical logic is routed to the most capable solver rather than the most overconfident one. Evaluations across five benchmarks show that Agora improves over matched single-model, routing, and cascade baselines under comparable candidate pools, while exposing a controllable cost-quality trade-off through a single auction parameter. 

---
# TrustX Agent Risk Classification Framework (ARC): Risk-Tiering Internally Created Agentic AI Systems 

**Authors**: Hannah M. Liu, Rhea Saxena, Shiv Asthana  

**Link**: [PDF](https://arxiv.org/pdf/2607.09586)  

**Abstract**: The proliferation of agentic AI systems across enterprise and public-sector contexts has outpaced the capacity of general-purpose AI risk frameworks to classify and govern them. In this paper, we introduce the TrustX Agent Risk Classification Framework, a structured, repeatable instrument that can be applied to seven types of agentic AI systems and is grounded in foundational pre-existing AI governance frameworks. At the core of the framework is a twelve-dimension scoring rubric that robustly quantifies the risk. This rubric is combined with other components, such as the GPA + IAT classification model and the five-level autonomy framework derived from existing literature. These inputs produce a three-tier governance output with mapped control recommendations. A specialised Coding Assistant extension is also included to account for nuances specific to this type of agentic AI system. We then use an illustrative example to show our framework in practice. ARC is intended for AI governance practitioners, risk officers, developers, and regulators, and it will regularly undergo iteration as we continue to expand it and make it more robust. The community can access the interactive framework here: this https URL 

---
# Knowledge Graphs and Explainable AI as Complementary Resources for Urban Mining 

**Authors**: Jan Gronewald, Andreas Emrich, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2607.09578)  

**Abstract**: Pre-demolition assessment, the regulated audit process at the heart of urban mining, is an information process in which AI support must serve qualified auditors who remain accountable for the decisions taken. The relevant unit of value is not prediction accuracy alone, but the defensibility of the supported decisions: their legibility, plausibility, sourcing, and contestability. Explainable AI techniques and domain knowledge graphs each address parts of this requirement, and existing taxonomies have catalogued their integration. The literature is descriptively rich but structurally under-specified: what remains less developed is a structural account of why specific integrations produce artefacts neither resource can provide alone. This paper offers a complementarity-theoretic interpretation grounded in the IS resource-based tradition. We propose four consolidated KG-XAI integration modes (Lifting, Constraining, Typing, and Revising), each defined as a typed operation over XAI artefacts and knowledge-graph substrate structures. Each mode unlocks a distinct property of defensibility and contributes to the kind of regulatory artefact pre-demolition assessment demands. A fire-door example from the urban-mining process illustrates the modes using the W3C Linked Building Data stack and valuation extensions. 

---
# Beyond Fixed Representations: The Vocabulary and Verifier Gaps in Open-Ended AI 

**Authors**: Yuan Cao, Haiqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09560)  

**Abstract**: Modern AI systems are increasingly being evaluated for their ability to reason, code, prove theorems, use tools, and long-horizon research tasks. These are powerful capabilities, but they share a structural limitation: the representational frame within which the model operates, including its conceptual vocabulary, the space of admissible solutions it can search, and the criteria by which success is evaluated, is typically fixed and supplied in advance. This paper argues that building stronger intelligent systems capable of open-ended innovation requires additional classes of operations: the creation, stabilization, and reuse of new representational primitives, which alter the space being searched rather than simply searching within it.
We characterize the distance between current AI systems and genuinely open-ended intelligence through two gaps. The first is the vocabulary gap, the difficulty of inventing and stabilizing new representational primitives rather than merely recombining existing ones. The second is the verifier gap, the difficulty of judging the value of a new primitive when its full payoff may be visible only after future reuse. We interpret both gaps through a unified framework of intelligence as cognitive discrepancy reduction. By viewing intelligent behaviors as a sequence of cognitive transformations, we distinguish intra-space transformations which operate within a fixed representational frame, from generative transformations which may modify the frame itself. On this basis, we propose a ladder of innovation autonomy and outline several directions for advancing open-ended AI, including objectives that reward useful representational change, persistent memory architectures for invented primitives, and adaptive verification mechanisms capable of evolving alongside the representations they evaluate. 

---
# SAGEAgent: A Self-Evolving Agent for Cost-Aware Modality Acquisition in Multimodal Survival Prediction 

**Authors**: Chongyu Qu, Can Cui, Zhengyi Lu, Junchao Zhu, Tianyuan Yao, Junlin Guo, Juming Xiong, Yanfan Zhu, Yuechen Yang, Bennett A. Landman, Yuankai Huo  

**Link**: [PDF](https://arxiv.org/pdf/2607.09521)  

**Abstract**: Does every cancer patient truly need a complete diagnostic workup for accurate survival prediction? In multimodal clinical oncology, diagnostic modalities follow a clinically mandated order of escalating burden -- from demographics collected at intake to genomic profiling requiring specialized tissue analysis. Current multimodal survival methods either assume all modalities are available or passively handle missing data, but none actively reason about whether acquiring the next modality is justified for a given patient along this ordered workflow. We formulate this as a sequential decision problem and propose SAGEAgent (Sequential Acquisition Guided by Experience), a self-evolving LLM-based clinical agent that decides which diagnostic modalities to acquire for each patient, balancing predictive accuracy against clinical invasiveness. SAGEAgent reasons about each patient's evolving diagnostic state through clinical tools that translate numerical predictions into text, an episodic memory that retrieves similar past cases, and a semantic memory that accumulates reusable decision patterns from experience. Experiments on a glioma cohort combining TCGA-LGG, TCGA-GBM, and BraTS with four diagnostic modalities demonstrate that SAGEAgent achieves competitive survival prediction accuracy while reducing average acquisition burden by 55%. 

---
# Shared Selective Persistent Memory for Agentic LLM Systems 

**Authors**: Sanjana Pedada, Aditya Dhavala, Neelraj Patil  

**Link**: [PDF](https://arxiv.org/pdf/2607.09493)  

**Abstract**: Agentic LLM systems that generate code through multi-turn tool use face a fundamental context problem: each session starts from zero, discarding the configuration choices, domain constraints, data schemas, and tool-use patterns that made previous sessions productive. Naively persisting entire conversation histories is token-inefficient and counterproductive: irrelevant context degrades generation quality. We introduce shared selective persistent memory, an architecture that identifies and retains four categories of reusable context (task specifications, data schemas, tool configurations, and output constraints) while discarding session-specific reasoning traces. Crucially, this memory is shared: workspaces encapsulating selective memory can be transferred across users with role-based access control, enabling collaborative reuse without redundant specification. We implement it in a deployed collaborative workspace platform where LLM agents produce, edit, and maintain git-versioned artifacts (dashboards, reports, and data-driven documents) from heterogeneous sources (CSV, SQL, REST APIs, and MCP servers). A complementary zero-token data refresh mechanism decouples generated programs from runtime data, enabling artifact reuse without re-invocation. Across three enterprise scenarios, shared selective persistent memory achieves 96% task completion (vs. 79% without memory and 71% with full history). Zero-token refresh eliminates LLM re-invocation for recurring updates (14x task-time reduction), while summary-driven generation cuts per-invocation token cost by 97x versus raw data injection. A replication on four public datasets confirms generalizability, with zero-token refresh succeeding in 12/12 trials. Notably, naive full-history persistence actively degrades completion by biasing the agent with stale traces, while selective memory outperforms both extremes. 

---
# Multimodal Reward Hacking in Reinforcement Learning 

**Authors**: Jiayu Yao, Yiwei Wang, Anmeng Zhang, Zhe Sun, Songsong Wang, Lingrui Mei, Yuyao Ge, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09492)  

**Abstract**: Reinforcement learning (RL) is increasingly used to align multimodal large language models (MLLMs), but higher rewards do not always imply better task performance. This risk is amplified when visual evidence is evaluated by text-only or weakly grounded rewards. We study reward hacking in MLLM RL across safety VQA, chart VQA, and stress-test settings, varying reward design, data ambiguity, model scale (2B-32B), and RL algorithm (GRPO, RLOO, DAPO). We introduce Newly Rewarded Failure Rate (NRFR), which measures failures among samples whose proxy reward improves over the SFT baseline. Outcome-only rewards cause severe hacking, reaching 48.1% Reward Hacking Rate (RHR), while NRFR exceeding RHR shows that RL creates new failures rather than merely inheriting them. Scaling reduces but does not eliminate hacking: even the 32B model retains a 54.9% worse rate under outcome-only rewards, whereas answer-aware rewards improve the oracle trend at every scale. Robustness is also algorithm- and scale-dependent: GRPO is consistently most resistant, RLOO remains vulnerable, and DAPO improves substantially from 2B to 8B. Visual-evidence rewards help only with reliable verification: keyword-based checks increase hacking, while VLM-as-judge semantic verification reduces it. Overall, multimodal reward hacking is a systematic result of optimizing imperfect rewards, and robust alignment requires rewards and verifiers that remain reliable under optimization pressure. 

---
# Ceci n'est pas une pipe: AI systems as semantic abstractions 

**Authors**: Jade Alglave, Patrick Cousot  

**Link**: [PDF](https://arxiv.org/pdf/2607.09489)  

**Abstract**: An AI system's output is not the fact or world state it appears to describe, but rather an engineered representation. We propose a semantic framework to describe AI systems, to be able to examine the correctness of such representations. To do so, we distinguish what is justified by accepted domain knowledge, what reference sources say, and what the system can currently use. This allows us to give precise definitions to common failures: extrapolation, refuted or unsupported assertion, sources versus knowledge mismatch, stale or refuted source, added hypotheses, unsupported use... We hope our framework gives a useful vocabulary for specifying and checking AI systems whose outputs, citations, tool calls, and world-changing actions must be justified by reliable claims and explicit authority rather than apparent fluency. 

---
# ProofCouncil: An LLM Agent for Solving Open Mathematical Problems 

**Authors**: Johannes Schmitt, Tim Gehrunger, Jasper Dekoninck, Gergely Bérczi, Uri Kreitner, Liam Price, David Holmes  

**Link**: [PDF](https://arxiv.org/pdf/2607.09474)  

**Abstract**: Large language models (LLMs) have shown increasing promise in solving open problems in mathematics. However, their performance can be further improved through agentic workflows tailored to real-world mathematical practice. To this end, we introduce ProofCouncil, a mathematical agent that is designed to tackle open problems using an author-critic architecture. ProofCouncil served as a submission to the second batch of FirstProof, a challenge consisting of 10 real-world mathematical problems that agents must solve autonomously. Its submissions for 6 of the 10 problems were judged by the referees to be correct up to at most minor revisions, showing the best performance among participating teams. We also evaluate ProofCouncil on 30 open problems collected from mathematical researchers. Among the 21 solutions that received human feedback, 5 were judged completely correct, 2 more were judged promising pending final verification, and a further 8 contained useful partial progress. In this short paper, we describe the development of ProofCouncil and the agent-building library used to create it, which we release as open source to the community. 

---
# How Does Bayesian Causal Discovery Fail? Characterising Structural Consequences in Linear Gaussian Networks under Latent Confounding 

**Authors**: Debargha Ghosh, Silja Renooij, Anna Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2607.09449)  

**Abstract**: Bayesian causal discovery is widely used for its ability to quantify epistemic uncertainty over directed acyclic graphs (DAGs) through posterior inference. However, its behaviour under latent confounding remains poorly understood, as existing work typically notes that confounding breaks identifiability without characterising how the posterior distribution over DAGs responds. In this work, we analyse posterior behaviour under latent confounding in linear Gaussian causal models, focusing on additive latent confounding between exactly two observed variables. We derive a critical correlation threshold above which the score function favours graphs with a spurious edge between the confounded variables, and show that this threshold decreases with sample size -- more data lowers the correlation required for the spurious edge to be favoured. Beyond this threshold, we characterize two distinct posterior failure regimes determined by the local structure around the confounded variables. Our findings are supported by exact posterior computations on multiple graph structures, demonstrating both the predicted failure regimes. 

---
# Fictional Worldbuilding: Multi-Agent LLM Collaboration with Hierarchical Context Compression and Iterative Review 

**Authors**: Jingbo Chen, He Wang, Wei Yuan, Yuqiao Lai, Zhenyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09403)  

**Abstract**: Worldbuilding, the construction of coherent fictional worlds, is a foundational task in game design and literary creation. Large Language Models (LLMs) offer new possibilities for automated content generation, but their application to worldbuilding faces three challenges: context explosion that grows linearly with the building process, the tension between creative diversity and content consistency, and the absence of automated quality assurance. This paper presents AutoWorldBuilder, a multi-agent collaborative system that addresses these challenges through five integrated components: a structured concept network with conflict detection; a DAG-based hybrid batch scheduler that groups tasks by semantic locality; a four-layer context compression mechanism achieving approximately 90% token reduction; an iterative review system with specialized Auditor agents that improves proposal pass rates from 42% to over 85%; and a skill-driven agent architecture supporting zero-code extension with differentiated temperature configuration. Two experiments across 20 diverse worldbuilding tasks, using GPT-OSS 120B and DeepSeek v3.2 as LLM backends, demonstrate a 95.0% success rate. The system generated 56-103 self-consistent concepts per world in 18-31 minutes with zero-conflict delivery. The architectural patterns validated here, including layer-as-budget compression, semantic-locality scheduling, and separation of generation and review, transfer to the broader class of knowledge-intensive, multi-agent LLM applications. 

---
# Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents over Computing Power Networks 

**Authors**: Nuocheng Yang, Sihua Wang, Zihan Chen, Tony Q. S. Quek, Changchuan Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.09330)  

**Abstract**: Embodied agent teams powered by heterogeneous large language models (LLMs) are being widely deployed in physical artificial intelligence such as smart factories, warehouses, and service robotics. To enable collaboration among such an agent team, efficient coordination mechanisms that operate reliably under limited network resources are required. However, existing heterogeneous LLM-agent coordination frameworks that rely on multi-round natural-language-based conversations introduce three coupled challenges. First, inter-agent dialogue incurs communication overhead that grows rapidly with team size. Second, the quality of coordination is constrained by the heterogeneous capabilities of the agent team's LLMs. Third, agents may suffer from action delays due to iterative negotiation. To address these challenges, we propose LDT-Coord, a networked coordination framework built upon a lightweight digital twin (DT). Specifically, each agent independently selects its intended action and reports both the action decision and a structured temporal constraint over shared resources to the DT server, thereby decoupling coordination performance from natural-language reasoning ability. Then, DT executes a training-free, rule-based orchestrator algorithm to resolve cross-agent conflicts and returns coordination instructions to prevent such conflicts. To further reduce communication overhead, we formulate agent reporting control as a constrained partially observable Markov decision process (C-POMDP) and solve it with the PPO-Lagrangian algorithm. Simulation results show that LDT-Coord achieves a task success rate comparable to conventional coordination methods while reducing communication overhead by more than 70x and maintaining robustness under LLM heterogeneity. 

---
# LongMedBench: Benchmarking Medical Agents for Long-Horizon Clinical Decision-Making 

**Authors**: Yanzhen Chen, Zihan Xu, Xiaocheng Zhang, Zhiting Fan, Weiqi Zhai, Hongxia Xu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09322)  

**Abstract**: In this work, we introduce LongMedBench, a real-world EHR-based benchmark for long-horizon clinical decision-making. Prior evaluations of LLM-based medical agents have largely emphasized short-context knowledge QA and tool use. However, real-world medical care is inherently longitudinal, and clinicians must aggregate evidence across repeated visits, tests, and evolving treatments. Therefore, long-horizon interaction is essential for realistic assessment. LongMedBench is constructed via a reproducible pipeline that integrates MIMIC-IV admission records and clinical notes into time-series event streams and long-context memory datasets, enabling long-horizon, multi-session interactions between agents and a clinical environment. It comprises 335 patients, with 19.72 inpatient visits per patient on average and 44.91 medical events per visit. Guided by the long-horizon decision process, we propose an evaluation taxonomy with three suites: fact-based QA, temporal reasoning, and long-horizon decision-making. This taxonomy measures how agents understand and leverage historical patient information over extended horizons. Our experiments show that while recent LLMs can make good use of explicit timestamps, they have challenges in implicit time inference; The RAG and agent memory system can improve the performance of information retrieval tasks, but the performance of decision-making tasks is highly dependent on the model's immediate context. 

---
# OpenProver: Agentic and Interactive Theorem Proving with Lean 4 

**Authors**: Matěj Kripner, Milan Straka  

**Link**: [PDF](https://arxiv.org/pdf/2607.09217)  

**Abstract**: In this system paper, we present OpenProver, an open-source system for LLM-driven automated theorem proving (ATP) with integrated Lean 4 formal verification. OpenProver integrates a Planner-Worker-Verifier architecture inspired by recent ATP agentic systems such as Aletheia. A Planner agent maintains a compact Whiteboard scratchpad and an unbounded Repository of intermediate findings, and decomposes mathematical work into parallel Workers.
OpenProver is fully open-source, offers reproducible evaluation through automatic formal verification of generated proofs, and provides an interactive terminal interface for human-guided proof search. In interactive mode, OpenProver allows the human operator to monitor and steer the proof search process, motivated by the established human-AI synergy in interactive code generation.
To showcase the potential for quantitative ablation experiments enabled by automatic formal verification, we evaluate OpenProver on ProofNet and compare it with a simple baseline. OpenProver is publicly available at this https URL. 

---
# Toward Auditable AI Scientists: A Hypothesis Evolution Protocol for LLM Agents 

**Authors**: Izumi Takahara, Teruyasu Mizoguchi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09195)  

**Abstract**: Large language model (LLM) agents are increasingly expected to play a central role in AI-driven scientific discovery. Equipped with broad knowledge, flexible reasoning, and tool use, they have the potential to autonomously explore and solve scientific problems by repeatedly proposing hypotheses, testing them, and revising their beliefs in the light of the evidence. In current agents, however, these hypotheses, tests, and belief updates are buried in unstructured logs, and no mechanism lets the agent or the human researcher audit that process. Here we propose the Hypothesis Evolution Protocol (HEP), an agent harness that provides hypothesis generation, evaluation, and evolution as explicit, auditable operations. On materials-science research tasks, a HEP-equipped agent operates the hypothesis--test--evidence--belief cycle that planning-style agents lack, generalizes across research questions, and exploits the protocol more fully as the base LLM becomes more capable. These results mark a step toward auditable AI scientists, whose scientific reasoning can be inspected, verified, and built upon. 

---
# Scoped Verification for Reliable Long-Horizon Agentic Context Evolution under Distribution Shift 

**Authors**: Dan C. Hsu, Luke Lu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09175)  

**Abstract**: Deployed LLM agents rely on agentic context, the model-external textual control content assembled by an operational harness. In this work, the mutable component of that context is a persistent system-level instruction that is updated from operational experience while the model, tools, and harness remain fixed. Over long evolution horizons, flat-text maintenance makes verification increasingly difficult as accumulated instructions grow and interact. We propose Graph-Regularized Agentic Context Evolution (GRACE), which maintains the persistent instruction component as a typed semantic graph and validates proposed updates within the local typed neighborhoods of modified nodes. Accepted graph updates are reconstructed as incremental edits to the textual instruction checkpoint used at deployment. We evaluate GRACE within a fixed telecom agent harness derived from $\tau^2$-bench under a controlled distribution-shift protocol. Across five independent replications, GRACE improves strict reliability, measured by pass^3, from the Gemini 2.5 Flash zero-shot value of 0.091 to 0.673$\pm$0.136 at the final checkpoint. This exceeds a Gemini 3.1 Pro zero-shot reference of 0.242 on the same held-out set, while the flat-text HCE baseline finishes at 0.191$\pm$0.051. These results identify two requirements for reliable long-horizon context evolution, a structural substrate that makes verification local and a consolidation mechanism that keeps accumulated instruction content usable. 

---
# KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling 

**Authors**: Peng Kuang, Haibo Jin, Xiaoyu Han, Yanli Wang, Xiaopeng Yuan, Ye Yu, Kaidi Xu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09153)  

**Abstract**: Process Reward Models (PRMs) have been proven to be highly effective in guiding test-time scaling (TTS) methods, which significantly boost the capabilities of LLM-based multi-agent systems. However, existing PRMs are text-based: they re-encode the entire trajectory text from scratch. In long multi-agent rollouts, the scoring cost, growing quadratically with respect to sequence length L, creates a severe computational bottleneck, severely limiting PRMs' application in long-context scenarios. To resolve this, we introduce KV-PRM, a highly efficient process reward model that eliminates the heavy text re-encoding by directly reading the KV cache produced naturally during the LLM's generation phase. By processing a single "verify token" against the pre-existing KV cache, KV-PRM reduces the scoring cost from O(L^2) to O(L). We formally prove that the KV cache contains strictly greater information capacity than text, and is more efficient for downstream reward modeling. Empirically, across the MATH, GSM8K, and AIME benchmarks, KV-PRM matches or strictly outperforms text-PRMs under various TTS methods such as Beam Search, MCTS, and Weighted Voting, with up to a 5,000x reduction in scoring FLOPs, a 37x reduction in latency, and a 34x reduction in per-sequence memory footprint compared to text-based PRMs. 

---
# MedRealMM: A Real-World Multimodal Benchmark for Chinese Online Medical Consultation 

**Authors**: Runhan Shi, Quan Zhou, Yuqian Xu, Shuai Yang, Xin Wu, Zitong Zhou, Hui Liu, Bin Cha, Zheming Wang, Liya Li, Wei Wei, Haoyuan Hu, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09142)  

**Abstract**: Large language models (LLMs) are increasingly deployed in online medical consultation, yet existing benchmarks remain poorly aligned with real clinical practice. Many rely on synthetic conversations or patient simulators, omit patient-uploaded medical images, or evaluate open-ended clinical responses using multiple-choice or lexical-overlap metrics that poorly reflect clinical quality. We introduce \textbf{MedRealMM}, a large-scale benchmark for multimodal online medical consultation built from de-identified patient-doctor interactions collected from a nationwide Chinese internet hospital. MedRealMM uses a Multimodal Clinical Challenge Point (MCCP) extraction framework to identify clinically demanding moments in authentic consultation trajectories and converts each into a standardized next-response generation task while preserving the preceding text-image context. Each instance is paired with a case-specific rubric refined by physicians that rewards clinically desirable behaviors and penalizes unsafe, unsupported, or contradictory responses. The current release contains 5,620 real-world multimodal cases spanning 64 clinical departments. We evaluate 19 general-purpose and medical-specialized LLMs, including text-only and multimodal systems. Our results show that image information is critical for reliable clinical performance and that current frontier models remain below the online physician response. Although some frontier models satisfy as many or more positive clinical criteria than physicians, they trigger more negative criteria, indicating that safety-sensitive error avoidance remains a central bottleneck. MedRealMM offers a realistic and reproducible benchmark for evaluating multimodal medical reasoning in real-world online consultation. The dataset will be publicly available on Hugging Face at this https URL. 

---
# L-MAD: A Systematic Evaluation of Multi-Agent Debate Structures in Legal Reasoning 

**Authors**: Tan-Minh Nguyen, Hoang-Trung Nguyen, Huu-Dong Nguyen, Dinh-Truong Do, Thi-Hai-Yen Vuong, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09099)  

**Abstract**: While multi-agent debate (MAD) frameworks have shown significant potential in general reasoning, their effectiveness in highly structured, knowledge-heavy legal domains remains under-explored. In this work, we introduce the Legal Multi-Agent Debate (L-MAD) framework to systematically evaluate different debate structures and aggregation methods within Legal Textual Entailment. By assigning distinct expert personas to multiple agents, L-MAD improves upon strong single-agent baselines by up to 8\%. Furthermore, analyzing how debate scales reveals a clear trade-off: increasing the agent population reduces inconsistency and improves accuracy, whereas extending discussion rounds induces a detrimental \textit{over-deliberation drift} where agents reinforce each other's mistakes. Ultimately, our findings outline the practical boundaries and safety margins of deploying collaborative multi-agent systems in high-stakes legal reasoning environments. 

---
# Neuro-Agentic Control: A Deep Learning-based LLM-Powered Agentic AI Framework for Controlling Security Controls 

**Authors**: Saroj Gopali, Bipin Chhetri, Deepika Giri, Sima Siami-Namini, Akbar Siami Namin  

**Link**: [PDF](https://arxiv.org/pdf/2607.09076)  

**Abstract**: Cyberattacks on operational technology are increasingly causing costly downtime and physical damage, exposing the limitations of traditional rule-based monitoring in industrial IoT environments. While Large Language Models (LLMs) have strong semantic reasoning abilities to assist in decision support, their hallucinatory nature presents unacceptable safety liabilities for closed-loop control. This paper introduces a neuro-agentic control framework, a novel architecture that couples an LLM-based planner (i.e., such as Gemini 2.5 Flash-Lite) with a pre-trained Time-Series Foundation Model (TimesFM), to achieve physics-grounded autonomous defense. The paper introduces a ``Counterfactual Physics Injection'' mechanism that simulates the impact of LLM-proposed interventions within the numerical latent space of the foundation model before actuation, while allowing the system to reject hallucinatory or unsafe actions. Evaluated on an industrial dataset (e.g., the Secure Water Treatment (SWaT)) in the context of stochastic attack scenarios, the framework exhibited better performance compared to LSTM and TCN baselines. The Neuro-Agentic Loop prevented five breaches (33.3%) below the threshold versus LSTM (26.7%) and TCN (13.3%), with zero physically invalid (hallucinated) actions executed. These results demonstrate the efficacy of using foundation models as deterministic ``Sentinels'' to safeguard agentic AI in critical infrastructure. 

---
# ARCANA: A Reflective Multi-Agent Program Synthesis Framework for ARC-AGI-2 Reasoning 

**Authors**: Kunbo Zhang, Lei Fu, Zeyu Wang, Zijing Liu, Kejian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2607.09059)  

**Abstract**: We present ARCANA, a collaborative multi agent framework for solving ARC AGI 2 tasks under strict test time and hardware constraints. ARCANA decomposes each task into iterative perception, hypothesis generation, symbolic execution, and reflective refinement. A perceptual grounding agent builds object centric scene graphs from raw grids, a latent program policy proposes diverse DSL programs, a symbolic executor verifies candidates on demonstrations, and a reflective agent synthesizes failure driven feedback for the next turn. These agents communicate through a shared differentiable blackboard and are scheduled by a learned meta controller. The design combines structured program search with adaptive multi turn correction, improving reasoning efficiency and solution quality on challenging abstract transformation tasks. 

---
# A Formalization of the Mean-Field Derivation of the Vlasov Equation: AI-Assisted Lean Formalization as a Strategy Game 

**Authors**: Joseph K. Miller  

**Link**: [PDF](https://arxiv.org/pdf/2607.08986)  

**Abstract**: We formalize a research result in the Lean 4 proof assistant by having a mathematician direct an AI system, and frame the activity as a formalization game. The objective is to turn a LaTeX document into Lean. The game is won when the development compiles, contains no sorry, and a machine check shows the target theorems rest on Lean's foundational axioms alone. Reuse is a second check, by a definition we introduce: whether the development yields a self-contained layer of general mathematics the wider library could absorb. The case study is a complete, axiom-clean formalization of well-posedness for the nonlinear Vlasov equation via Dobrushin's mean-field route -- existence, uniqueness, the stability estimate and mean-field limit, and a short-window superposition principle (weak solutions are Lagrangian). The human's role was to direct, not to write proofs: to scope the definitions, steer the decompositions, and triage the library's gaps; the AI agent executed. The formalization certifies the proof of each statement as written; whether the written statement is the intended theorem stays the mathematician's judgment. The optimal-transport machinery that fell out of the build (in particular, properties of the Wasserstein-1 metric and the Kantorovich-Rubinstein duality theorem) separates into a self-contained layer that compiles against Mathlib alone: about a sixth of the development (49 of 299 declarations), behind a 22-declaration interface with no reverse dependency. The headline theorems ran in about a week, the full development in about a month. We report the quantitative claims as observations of one game, not as general laws. The game's rules name no particular system, so the methodological framing is meant to outlast the tools of any one run. 

---
# Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading 

**Authors**: Zongxia Li, Zhongzhi Li, Yucheng Shi, Ruhan Wang, Junyao Yang, Zhichao Liu, Xiyang Wu, Anhao Li, Yue Yu, Ninghao Liu, Lichao Sun, Haotao Mi, LeoweiLiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08964)  

**Abstract**: AI agents have become capable of autonomously completing short, well-specified tasks. However, existing terminal benchmarks largely focus on simple problems that finish within minutes and are evaluated only by their final outcome. This setup overlooks intermediate progress and partial solutions, yielding sparse reward signals and an incomplete picture of agent capability. We introduce Long-Horizon-Terminal-Bench, a terminal benchmark of 46 long-horizon tasks spanning nine categories, including experiment reproduction, software engineering, multimodal analysis, interactive games, and scientific computing. Each task follows a Terminal-Bench-style setup with a reference solution or simulation engine, but is further decomposed into fine-grained graded subtasks. This design enables dense intermediate rewards and partial credit, allowing evaluation to capture not only whether an agent reaches the final goal, but also how far it progresses on open-ended workflows. Tasks in Long-Horizon-Terminal-Bench typically require hundreds of episodes and minutes to hours of execution, stressing long-horizon planning, long-context management, and iterative debugging rather than one-shot problem solving. We evaluate 15 frontier models and find that agents consume on average 9.9M tokens per task, with roughly 231 episodes and 85.3 minutes of execution time per run, making Long-Horizon-Terminal-Bench more demanding than prior terminal-based benchmarks. Even the strongest tested model achieves 15.2% pass@1 at a partial-reward threshold of 0.95 and 10.9% at a perfect-reward threshold of 1.0, while the mean pass rate across models is 4.3% and 1.7% under the two thresholds, respectively. These results reveal headroom for improvement. We further analyze failure modes and error patterns, and release Long-Horizon-Terminal-Bench to support future progress on long-horizon terminal agents. 

---
# GATS: Graph-Augmented Tree Search with Layered World Models for Efficient Agent Planning 

**Authors**: Maureese Williams, Dymitr Nowicki  

**Link**: [PDF](https://arxiv.org/pdf/2607.08894)  

**Abstract**: Large Language Model (LLM) agents have shown promise in multi-step planning tasks, but existing approaches like LATS (Language Agent Tree Search) and ReAct rely heavily on LLM inference during planning, leading to high computational costs and stochastic behavior. We present \textbf{GATS} (Graph-Augmented Tree Search), a planning framework that combines systematic UCB1-based tree search with a layered world model to eliminate LLM calls during inference while achieving superior planning performance. Our three-layer world model integrates: (L1) exact symbolic action matching, (L2) statistics learned from execution logs, and (L3) LLM-based prediction for unknown actions. On synthetic planning tasks with branching paths and dead-ends, GATS achieves \textbf{100\% success rate} compared to 92 % for LATS and 64\% for ReAct. On a comprehensive stress test spanning 12 challenging scenarios -- including coding workflows, web navigation, and long-horizon tasks -- GATS maintains \textbf{100\% success} while LATS drops to 88.9 % and ReAct to 23.9%. GATS requires \textbf{zero LLM calls per task} during planning (vs. 37 per task for LATS) and produces deterministic plans with zero variance across runs. Our results demonstrate that systematic search with learned world models can substantially outperform LLM-guided exploration for agent planning. 

---
# CogniConsole: Externalizing Inference-Time Control as a Formal Abstraction for Reliable LLM Interactions 

**Authors**: Vanessa Figueiredo, Wilter Franceschi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08774)  

**Abstract**: Reliability in large language model (LLM) systems is typically framed as a function of model capability. We challenge this by demonstrating that reliability is significantly influenced by \emph{inference-time control} -- the computational layer governing task framing and context selection. We introduce \emph{CogniConsole}, an architectural instantiation that externalizes this control into a structured interface combining programmatic coordination with bounded prompt-based reasoning. Through \emph{controllability-oriented probes} ($N=489$) in a multi-step interactive environment, we show that increasing structural scaffolding -- from unstructured to fully scaffolded -- \textbf{systematically reduces output variance and failure rates under a fixed model architecture}. Our results indicate that many observed failure modes, such as context drift and inconsistent constraint adherence, arise from under-specified control rather than insufficient capability. This work provides an empirical basis for treating inference-time control as a first-class abstraction, opening new directions for designing and evaluating LLM systems beyond scaling alone. 

---
# Interval Certifications for Multilayered Perceptrons via Lattice Traversal 

**Authors**: Merkouris Papamichail, Konstantinos Varsos, Giorgos Flouris, João Marques-Silva  

**Link**: [PDF](https://arxiv.org/pdf/2607.08773)  

**Abstract**: In this work we present a rigorous theoretical framework to a foundational problem of AI safety, namely adversarial robustness. In particular, we show that the adversarial robustness problem can be reduced to a lattice traversal problem. Each element of this lattice corresponds to an interval, i.e., an axis-aligned hyper-rectangle, containing an input point $\mathbf{x}$. Consider a multilayered perceptron classifier (MLP). An interval $I$ constitutes a sound certification if $\mathbf{x} \in I$ and $\mathbf{x}$ can be freely perturbed in $I$ without changing the MLP's prediction. Complementarily, an interval $I$ constitutes a complete certification if $\mathbf{x} \in I$ and when $\mathbf{x}$ moves outside of $I$ the MLP's prediction is guaranteed to change. While the sound certification problem corresponds to the well-studied adversarial robustness, complete certifications have not been examined in the literature. We develop lattice traversal operators, which we apply in a refine & verify iterative scheme. Using formal MLP verifiers, sound maximality and complete minimality are guaranteed. Moreover, we examine objective optimization problems. There we discover some interesting asymmetries. For complete certifications, the minimum solution is obtained in polynomial oracle calls. This does not hold for sound certifications, where we prove strong intractability results. Additionally, we examine optimization problems in symmetric intervals (i.e., $\ell_\infty$-spheres), where we provide logarithmic algorithms. Finally, we present an empirical evaluation, using the novel ParallelepipedoNN system. 

---
# PHINN-EEG: Topological Time-Series Analysis of Dream-State EEG -- Dynamic Betti Curves for Dream Content Classification and Topology-Conditioned Neural Signal Synthesis 

**Authors**: Ren Takahashi, Emre Yusuf, Jayabrata Bhaduri  

**Link**: [PDF](https://arxiv.org/pdf/2607.09662)  

**Abstract**: Current electroencephalography (EEG)-based dream detection relies on power spectral density (PSD) and statistical moment features, achieving a state-of-the-art area under the receiver operating characteristic curve (AUC) of approximately 0.70 on the DREAM database (Wong et al., 2025, Nature Communications). We introduce PHINN-EEG (Persistent Homology Inspired Neural Network for EEG), the first topological time-series framework for dream mentation analysis. Using sliding-window Takens delay embeddings and Vietoris-Rips filtrations on multichannel pre-awakening EEG epochs, we extract Dynamic Betti Curves that characterize the geometric architecture of neural activity, not merely its energy. These topological invariants, combined with topology-conditioned flow matching, are analytically projected to outperform existing PSD and catch22 benchmarks, targeting AUC = 0.82-0.90 on the 1,462-awakening open-access subset of the DREAM database (drawn from a full registry of 3,191 total awakenings from 263 participants across 20 independent laboratories). We further introduce a topology-conditioned rectified flow model for dream-state EEG synthesis-with a spectral-conditioned flow model of comparable feature dimensionality as an additional ablation baseline to isolate the value of topological conditioning specifically-and propose a set of candidate Betti transition archetypes linking topology to phenomenological dream report categories, presented as an exploratory hypothesis space pending empirical validation. If validated, this work represents a paradigm shift from spectral energy to phase-space geometry in neural rare-event detection, with potential future implications for wearable BCI dream monitoring. 

---
# Scalable Visual Pretraining for Language Intelligence 

**Authors**: Yiming Zhang, Zhonghan Zhao, Wenwei Zhang, Haiteng Zhao, Tianyang Lin, Yunhua Zhou, Demin Song, Kuikun Liu, Haochen Ye, Haian Huang, Yuzhe Gu, Haijun Lv, Qipeng Guo, Bin Liu, Gaoang Wang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09657)  

**Abstract**: The rapid progress of large foundation models has been driven predominantly by pretraining on large-scale text corpora. However, many forms of knowledge are conveyed through visual representations, where figures, typeset equations, and page layouts carry rich information that cannot be faithfully or completely captured by text alone. Yet current pretraining approaches discard these visual cues by converting visually rich sources, such as documents and web pages, into plain text for learning language intelligence. This paper challenges the default assumption that language models must be trained on text-only representations and shows that Visual Pretraining is a scalable learner for foundation model intelligence. To this end, we conduct a systematic study of unsupervised visual pretraining paradigms that directly leverage visual documents without text extraction. Across multiple backbones and benchmarks, visual pretraining on the same underlying corpora consistently outperforms text-only pretraining, offering an efficient pathway to scalable language intelligence. 

---
# Evolution of Accuracy and Visual-Cognitive Errors in a Decade of Vision-Language AI Models 

**Authors**: Shravan Murlidaran, Miguel P. Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2607.09654)  

**Abstract**: Vision language models (VLMs) have made remarkable progress in visual reasoning during the last decade. Most evaluations have used simple scenes (MS-COCO) that do not showcase complex human interactions or behaviors, only a handful of non-curated human descriptions as a benchmark, and have not focused on understanding the model's error types. Here, we introduce the Complex Social Behavior (CSB) dataset, containing 100 images depicting complex social interactions/behaviors. We analyze the progression of scene descriptions over a decade (2017-2025) of VLMs (four pre-Multimodal Large Language Models, MLLMs, and five MLLMs). We evaluate the accuracy of the models and 20 human descriptions relative to a gold standard on the CSB dataset and on a sample from MS-COCO. We analyzed five visual-cognitive error types: object detection, recognition, hallucination, scene understanding, and spatial dependence. The CSB dataset showed a more pronounced improvement than MS-COCO in scene description accuracy, with pre-MLLMs achieving much lower accuracy than the bottom-ranked human descriptions and MLLMs attaining accuracies similar to the top-ranked human descriptions. We show that MLLMs have eliminated the gap in scene description accuracy between simpler MS-COCO scenes and scenes depicting complex behaviors (CSB). MLLMs have almost eliminated all error types in our tested datasets, except for occasionally relying on different image regions for scene descriptions than humans do (spatial dependence error). We also show that detection, recognition, and hallucination errors have the highest impact on scene description accuracy. Together, our findings provide a more thorough evaluation of how visual language models have advanced over the last decade. 

---
# VEXAIoT: Autonomous IoT Vulnerability EXploitation using AI Agents 

**Authors**: Katherine Swinea, Kshitiz Aryal, Lopamudra Praharaj, Maanak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2607.09653)  

**Abstract**: Internet of Things (IoT) systems are inherently vulnerable due to constrained hardware, outdated firmware, and insecure default configurations, creating a need for scalable and adaptive security testing approaches. While recent adoptions of Large Language Model (LLM) agents have demonstrated promise in penetration testing and Capture-the-Flag (CTF) environments, their application to IoT specific vulnerabilities remains unexplored. This paper presents an autonomous multi-agent framework, referred to as Vulnerability EXploitation using AI Agents (VEXAIoT), for vulnerability discovery and exploitation in IoT environments using LLM-based reasoning and offensive security tools. The framework combines a vulnerability detection agent and an attack execution agent to perform reconnaissance, plan attack sequences, and execute exploits against vulnerable IoT services. The system is evaluated in IoTGoat and Metasploitable environments across ten attack scenarios mapped to OWASP IoT vulnerabilities. Experimental results show attack success rate of up to 100% with low token overhead and average execution times under two minutes for most attacks. Across 260 attack executions, VEXAIoT achieves a 95.0% overall success rate, including 94.5% success in IoTGoat and 96.7% success in Metasploitable2. These results demonstrate the potential for LLM-driven agents to automate IoT vulnerability assessment and offensive security workflows in controlled environments 

---
# Semantic Pareto-DQN: A Multi-Objective Reinforcement Learning Framework for Financial Anomaly Detection 

**Authors**: Cláudio Lúcio do Val Lopes, Lucca Machado da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2607.09641)  

**Abstract**: Financial anomaly detection suffers from extreme class imbalance, causing traditional single-objective algorithms to exhibit ``fraud collapse'', defaulting to the majority class and failing to balance anomaly interdiction with customer friction. To overcome this without distortive data resampling, we propose the Semantic Pareto-DQN, a multi-objective reinforcement learning framework. Our approach synthesizes heterogeneous transaction features into cohesive natural-language narratives, encoded by large language models, thereby producing a robust, scale-invariant state representation. The agent optimizes a vectorial reward that explicitly decouples financial efficacy, operational friction, and semantic discovery. By mapping the continuous Pareto frontier, the system dynamically navigates the asymmetric costs of missed anomalies versus false positives. Empirical evaluations across E-Commerce fraud and UCI Credit datasets show that semantic Pareto-DQN successfully shatters the zero-recall trap. It achieves superior minority-class recall compared to scalarized baselines, providing an alternative to trade bounded operational friction for financial anomaly discovery. 

---
# Lean-QIT: Towards a Formal Infrastructure for Quantum Information Theory 

**Authors**: Chengkai Zhu, Ziao Tang, Guocheng Zhen, Yimeng Cao, Yusheng Zhao, Ranyiliu Chen, Xuanqiang Zhao, Lei Zhang, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09632)  

**Abstract**: Quantum information theory (QIT) characterizes the capabilities and fundamental limits of quantum information processing, underpinning quantum communication, computation, and error correction. Formalizing its coding theorems requires connecting finite-block protocols, analytic inequalities, and asymptotic limits within a unified machine-checked framework. Existing developments, however, lack a reusable operational layer that defines codes, error criteria, achievable rates, and capacities independently of their information-theoretic characterizations. In this work, we present LeanQIT, a Lean 4 library for finite-dimensional QIT. It provides composable, kernel-checked interfaces for quantum states and channels, source and channel codes, finite-block performance criteria, hypothesis testing, one-shot quantities, and asymptotic rate constructions. Using this infrastructure, we formalize Schumacher's quantum source-coding theorem, the Holevo--Schumacher--Westmoreland classical-capacity theorem, and the entanglement-assisted classical-capacity theorem together with its strong converse. By separating operational definitions from analytic characterizations and exposing reusable achievability, converse, and asymptotic components, Lean-QIT provides a machine-readable foundation for formal QIT and a compositional knowledge substrate for emerging AI-assisted formalization, automated proof search, and agentic reasoning in quantum information and computation. 

---
# 4DR360: State Reasoning for Joint 3D Detection and Occupancy Prediction in 4D Radar-Camera Full-Scene Perception 

**Authors**: Xiaokai Bai, Lianqing Zheng, Runwei Guan, Songkai Wang, Siyuan Cao, Hui-liang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09629)  

**Abstract**: Reliable autonomous driving requires full-scene perception that couples foreground objects with dense semantic layout. Recently, 4D millimeter-wave radar has emerged as a robust and affordable sensor, yet its sparse returns make radar-camera fusion necessary for comprehensive scene understanding. Existing radar-camera methods mainly optimize detection, while dual-task systems usually decode boxes and occupancy with limited interaction. To address this gap and advance radar-based multi-task learning, we propose \method, a 4D radar-camera framework for 360$^\circ$ full-scene perception, which models semantic occupancy as a persistent scene state rather than a terminal output. \method{} follows a cross-modal state reasoning paradigm, where the occupancy state is modeled and propagated through stages for coarse-to-fine feature aggregation. Specifically, State-guided BEV Enhancement (SBE) strengthens intra-frame BEV representation, while Doppler-guided Temporal Fusion (DTF) preserves state evidence over longer temporal horizons. Beyond the model, we further extend ManTruckScenes with satellite-map-based generated occupancy labels and pair it with OmniHD-Scenes in a unified cross-dataset detection-and-occupancy protocol. The resulting experiments cover accuracy, robustness, ablation, and efficiency under one radar-camera multi-task evaluation framework. Code and labels will be released upon acceptance. 

---
# Task-Specific Multimodal Question Answering Agents via Confidence Calibration and Incremental Reasoning for QANTA 2026 

**Authors**: Nirjhar Das, Md. Al-Mamun Provath  

**Link**: [PDF](https://arxiv.org/pdf/2607.09623)  

**Abstract**: We present our submission to the QANTA 2026 shared challenge at the ICML 2026 Workshop on Efficient Multimodal Question Answering (EMM-QA). Quanta evaluates multimodal quizbowl systems that answer pyramid-style questions from incrementally revealed text and accompanying images while operating under realistic efficiency constraints. The challenge consists of two distinct tasks: Tossup questions, which require deciding when to answer under uncertainty, and Bonus questions, which emphasize accurate answer selection and human adoption. To address these differing objectives, we develop a task-specific two-agent architecture. Our Tossup agent utilizes a GPT-4o-mini-class model (referred to as GPT-4.1-mini in the competition logs) with confidence-calibrated answering and a domain-specific numeric reasoning policy that reduces overconfident predictions from isolated quantitative clues. Our Bonus agent uses GPT-4o-class model (referred to as GPT-4.1) with leadin-aware reasoning, structured relational reasoning, and multimodal evidence integration to improve exact answer selection. Rather than relying on a retrieval pipeline or model ensembles, our approach emphasizes efficient reasoning policies and confidence calibration within a hosted-only environment. Our system achieved the highest overall leaderboard score of 0.402, including a Tossup score of 0.238 and a Bonus Effect score of 0.164. The results demonstrate that lightweight, task-specific reasoning strategies can provide strong performance on resource-constrained multimodal question answering benchmarks. 

---
# PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers 

**Authors**: Yujie Pang, Zudong Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09590)  

**Abstract**: Precision industrial contact manipulation requires reliable robot policies under pose perturbations and contact-force constraints. Vision-language-action models offer broad generalization but often introduce high inference latency and GPU-memory cost, while vision-action chunking policies are more suitable for real-time industrial control. However, these policies are usually trained by behavior cloning and suffer from distribution shift in contact-rich tasks. This paper proposes PAC-ACT, a reinforcement-learning post-training framework for pretrained Action Chunking Transformer policies. PAC-ACT reformulates policy optimization at the chunk level, constructs an ACT-transferred actor-critic architecture, and introduces a hybrid behavior-prior constraint to preserve the pretrained action distribution during online fine-tuning. Experiments on industrial precision-contact benchmarks show that PAC-ACT improves task success, contact stability, and force safety while retaining low latency and low GPU-memory usage. On the Contour task, PAC-ACT significantly reduces peak contact force and decreases the proportion of force readings above 60 N by 46 times. Sparse-reward ablations further show that the proposed behavior-prior constraint enables effective exploration under randomized initial poses. 

---
# Conceptual Networks for Cross-Linguistic Idiomatic Expressions:A Feature-Based Graph Approach 

**Authors**: Kiran Pala, Punam Silu, Lixun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09576)  

**Abstract**: We present an interpretable network-based framework for representing idiomatic and figurative meaning across eight typologically diverse languages, totaling 160 conventional expressions, the large majority of which are idiomatic. Each expression is annotated with binary conceptual features (containment, concealment, emotional, social, etc.) derived from cognitive-linguistic theory, and pairwise Jaccard similarities define a weighted graph. Community detection reveals that idioms cluster by conceptual schema rather than by language, producing a structure consistent with cognitive-linguistic predictions. The conceptual network captures unique semantic information not present in distributional embeddings, can be scaled via automatic annotation with LLMs, improves downstream idiom detection, and remains robust when enriched with corpus frequencies. Cross-lingual transfer experiments show that conceptual proximity alone can identify acceptable translation equivalents across five language families, with substantial gains over embedding-based baselines. Ablation studies demonstrate that all three feature dimensions -- schemas, roles, and valence -- contribute non-redundantly to both the network's organizational properties and its performance on idiom detection, and that specific graph-derived signals (community membership, neighbor similarity) are particularly informative. The framework offers an interpretable, cross-linguistically stable representation of idiomatic meaning, combining theoretical grounding with practical utility. 

---
# Large-Scale Portfolio Optimization Problem Under Cardinality Constraint With Enhanced Multi-Objective Evolutionary Algorithms 

**Authors**: Danial Ramezani, Mostafa Abouei Ardakan  

**Link**: [PDF](https://arxiv.org/pdf/2607.09566)  

**Abstract**: Decision-making is posing an increasingly formidable challenge to investors because of the growing number of alternatives available in financial markets. A hot area of research over the past few decades has been portfolio optimization that seeks to determine how much an investor should invest in which asset. Introducing real-world conditions to the optimization model turns the problem into an NP-hard one for whose solution exact methods become inefficient; hence, researchers have turned to evolutionary algorithms to approximate solutions. In this paper, strengthening strategies are presented for multi-objective evolutionary algorithms that can provide a faster convergence rate and extensive search ability in the portfolio optimization problem under the cardinality constraint. To implement those features, a unique solution representation, a novel operator, and new repair mechanisms are introduced for solving the aforementioned problem in which lower and upper limits are set on the number of assets in the portfolio. For this purpose, new mating strategies along with the aforesaid package are implemented in well-known multi-objective evolutionary algorithms to solve the problem. The customized algorithms are subsequently tested against traditional ones using well-known market indices as benchmarks. Results indicate that the proposed strategy not only provides better approximations but also converges faster as well at no loss of performance with an increasing number of assets in the market. 

---
# TCLA: Training-Free Class-wise Logit Adaptation for Medical Vision-Language Models 

**Authors**: Tianyou Jiang, Ziyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.09562)  

**Abstract**: Medical Vision-Language Models (VLMs) exhibit strong zero-shot performance, yet their effectiveness still declines on out-of-distribution (OOD) data due to domain shifts and class bias inherited from large-scale pretraining. Existing few-shot adaptation methods typically introduce additional trainable components, which can be unstable in extremely low-data regimes (e.g., 1-shot), and lack robustness on different medical data. We present TCLA, a purely training-free few-shot adaptation method for Medical VLMs, which is fast and model-agnostic. TCLA corrects inference logits based on a small set of support samples, boosting pretrained VLMs performance by improving inter-class deconfusion and reducing domain shift. Extensive experiments on nine datasets across multiple medical imaging modalities including X-ray, Ultrasound, MRI, CT, Histopathology, demonstrate that TCLA consistently improves OOD performance of Medical VLMs and, in most of cases, outperforms existing training-based adaptation methods. 

---
# ALICE: Learning a General-Purpose Pathology Foundation Model from Vision, Vision-Language, and Slide-Level Experts 

**Authors**: Jiawen Li, Tian Guan, Huijuan Shi, Xitong Ling, Mingxi Fu, Anjia Han, Chao He, Yonghong He  

**Link**: [PDF](https://arxiv.org/pdf/2607.09526)  

**Abstract**: Foundation models are reshaping computational pathology, yet their capabilities remain shaped by pretraining objectives, data sources, and spatial scales, fragmenting complementary expertise across separate backbones. Here we present ALICE, a unified foundation model trained through multi-stage agglomerative distillation that sequentially distills eight vision-only, vision-language, and slide-level teacher models into dedicated modules of a single backbone. ALICE is pretrained on 24,985,184 tile-level pathology images and 155,604 high-resolution images, and evaluated across 21 task scenarios, 96 downstream tasks, and 48 data sources, spanning region-of-interest tissue analysis, vision-language multimodal evaluation, and whole-slide clinical assessment. In all three evaluation settings, ALICE achieved the best average rank among task-matched pathology foundation models. These results demonstrate that agglomerative distillation can consolidate complementary capabilities from specialized models into a unified backbone for broad computational pathology applications. The model is available at this https URL. 

---
# Seeing is Free, Speaking is Not: Uncovering the True Energy Bottleneck in Edge VLM Inference 

**Authors**: Junfei Zhan, Haoxun Shen, Mingang Guo, Zixuan Huang, Tengjiao He  

**Link**: [PDF](https://arxiv.org/pdf/2607.09520)  

**Abstract**: Vision-Language Models (VLMs) are the perceptual backbone of embodied AI, but their energy footprint on edge hardware remains poorly understood. Existing efficiency efforts focus predominantly on reducing visual tokens, implicitly treating visual processing as the dominant energy cost. We overturn this implicit assumption through the first systematic energy profiling of on-device VLM inference, spanning five models across three architecture families, four input resolutions, and two hardware platforms (NVIDIA RTX 3070 and Jetson Orin NX). Our analysis yields three findings. First, average inference power is a model-intrinsic constant, invariant to input resolution, image complexity, and prompt type, with less than 5% variation across all conditions. This means that all energy variation across inputs must arise from variation in inference time, not from variation in power draw. Second, each output token costs 11 to 39x more wall-clock time than each input token due to the compute-bound and memory-bound asymmetry between prefill and decode, making output token count the dominant driver of both latency and energy. Third, image complexity, measured by the number of objects in an image, induces up to 4.1x energy differences at identical resolution. This variation arises not from increased visual processing cost, but from differences in output length. These findings expose a fundamental limitation of visual token pruning: even removing all visual tokens saves at most 10% of total energy for fixed-token models. Across models spanning 1 billion to 8 billion parameters, controlling output length saves up to 97% of total energy, with the energy dominance of decoding growing stronger at larger model scale. In short, the true energy bottleneck in edge VLM inference is not what the model sees, but how much it says. 

---
# Failure as a Process: An Anatomy of CLI Coding Agent Trajectories 

**Authors**: Xiangxin Zhao, Han Li, Shuaiting Li, Tianyi Zhao, Earl T. Barr, Federica Sarro, He Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.09510)  

**Abstract**: Large language model (LLM) coding agents are increasingly deployed to autonomously perform software engineering tasks in terminal-based environments, making their reliability a growing concern. Existing empirical studies investigate why coding agents fail, yet they largely treat failure as a final outcome rather than a temporal process, providing limited insight into how failures emerge, evolve, and become unrecoverable. We present the first large-scale empirical study of CLI coding-agent failure trajectories, introducing a process-oriented framework that analyzes failure through its onset, evolution, and recovery across execution trajectories. We first collect 3,843 execution trajectories generated by seven frontier models across three coding-agent scaffolds (OpenHands, MiniSWE, and Terminus2) on Terminal-Bench, then carefully filter them to obtain 1,794 complete and valid trajectories for manual annotation (over 63,000 execution steps), from which we derive 14 findings spanning failure occurrence, root causes, recovery, and cross-system consistency. Our findings show that coding-agent failures are predominantly driven by epistemic errors, typically begin within the first few execution steps, and often remain hidden until recovery is no longer possible, suggesting that improving coding-agent reliability requires earlier validation and intervention rather than relying solely on final-outcome evaluation. 

---
# What VGGT Knows About Overlap: Probing Geometric Foundation Models for Co-Visibility 

**Authors**: Filippo Ziliotto, Luciano Serafini, Lamberto Ballan, Tommaso Campari  

**Link**: [PDF](https://arxiv.org/pdf/2607.09503)  

**Abstract**: A fundamental challenge in 3D reconstruction and robotic localization is co-visibility: determining which image pairs share overlapping visible surfaces, particularly in scenarios with minimal overlap. We demonstrate that VGGT implicitly encodes co-visibility as an emergent behavior: without any supervision for this task, its internal representations exhibit a clear hierarchical structure mirroring that of large language models, i.e. early layers build a 3D-aware scene representation, while late layers act as dedicated co-visibility reasoners. In particular, we identify layer L17 as a negative anchor that consistently routes non-co-visible pairs for this backbone, regardless of the evaluation setting, providing task-grounded evidence of layer specialization in a geometry-grounded foundation model. Building on this, we introduce Co-VGGT, which freezes VGGT and trains only a lightweight layer-wise mixture-of-experts head (less than 7.5M parameters) to classify co-visibility from RGB alone, treating each layer as a specialized expert whose geometric abstraction is adaptively weighted per input pair. On the Co-VisiON benchmark, Co-VGGT surpasses the human annotation baseline and improves over prior work by more than 25% pairwise and 10% multiview. Pairwise predictions are well-calibrated (ECE=0.030), enabling direct use as edge weights in visibility graphs for downstream SfM and SLAM pipelines without post-hoc correction. Code and data are available. 

---
# All Explanations are Wrong, But Many Are Useful: Exploring the Rashomon Explanation Set with Large Language Models 

**Authors**: Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09502)  

**Abstract**: Explaining machine-learning models is increasingly important for decision-making and consumer trust, yet it is widely believed to come at a cost: existing Explainable AI (XAI) methods suffer from a persistent accuracy-explainability trade-off. We argue that this trade-off is not fundamental, but an artifact of treating explanation and prediction as separate objectives; when properly coupled, they become complementary, so that equipping a model to explain itself improves, rather than degrades, its accuracy. We introduce the Rashomon Explanation paradigm, which builds a set of faithful, prediction-guiding explanations rather than a single one, and prove that this set is generally non-empty and that explanation fidelity bounds the performance of the models it guides. To explore this set, we propose RashomonLLM, an Explanation-Prediction-Reflection agentic workflow that generates explanations in natural language by iteratively aligning them with predictions, and we prove it converges and recovers the full set. Across customer-churn classification, clinical survival regression, and industrial click-through prediction on large-scale live-streaming logs, RashomonLLM significantly outperforms state-of-the-art prediction and XAI baselines on both accuracy and explanation quality, with gains driven by explanation fidelity and robust to distribution shifts, temporal splits, and seeds. Our framework thus advances business performance while laying the groundwork for consumer trust. 

---
# Decoupling Language Guidance from Backbones for Text-Guided Medical Segmentation 

**Authors**: Yungeng Liu, Xuanzi Fang, Haijin Zeng, Qi Dai, Yongyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09481)  

**Abstract**: Text-guided medical image segmentation leverages clinical semantics to improve lesion delineation, yet many existing models bind cross-modal fusion, supervision, and decoder design into a task-specific architecture. Such tight coupling makes it difficult to reuse language guidance modules across heterogeneous vision and text backbones, and often requires redesigning the network when the encoder pair changes. This paper presents BTHA, a backbone-transferable hierarchical adapter framework for text-guided medical image segmentation. BTHA is built around a stable feature-level interface: given multi-scale visual features and a text representation, it injects semantic guidance through shape-preserving adapters while maintaining the decoder-side tensor contract. To make this interface effective, we introduce a Hierarchical Coarse-to-Fine Supervision Strategy that decomposes learning into global image-text alignment, multi-scale auxiliary localization, and boundary-aware final mask refinement. We further design a Scale-Adaptive Gated Semantic Guidance (SAGSG) adapter, where resolution-specific gates adaptively control textual injection and channel recalibration suppresses redundant cross-modal responses. Evaluations across diverse vision and text backbones show that the same adapter and supervision design remains effective across convolutional and transformer-based visual encoders as well as different language encoders. Experiments on four public datasets further demonstrate that BTHA improves strong text-guided baselines with modest computational overhead. 

---
# Practical Source Code Recovery from Binary Functions Using Anchor-Based Retrieval and LLM Reasoning 

**Authors**: Charles Edward Gagnon, Steven H. H. Ding, Philippe Charland, Benjamin C. M. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2607.09452)  

**Abstract**: We present a practical pipeline for recovering source code from stripped binary functions by combining reverse engineering, anchor-based source code retrieval, and large language model reasoning. Our binary-to-source-code retrieval method attempts to identify the source function from a source code database, rather than generating approximate decompiled pseudocode. It extracts anchors such as strings, constants, external calls, and available function names using Ghidra, retrieves candidate files via an inverted-index search database, narrows candidates to likely function snippets, and re-ranks them with a large language model (LLM) based on disassembly, decompiled code, and source metadata. Confident matches can also serve as anchors in later passes. In an evaluation backed by our high-fidelity source code database on a stripped, optimized tcpdump binary, our proposed binary-to-source matching method achieves 95.2% assembly instruction coverage. Experiments on a GitHub-based retrieval database showed lower performance with 35.5% instruction coverage on average, mainly due to retrieval misses. These results show that source-level binary recovery excels with high-quality databases and remains a useful tool in noisy environments. 

---
# Parameter-Efficient Vision-Language Adaptation with Continuous Metadata Conditioning for Animal Re-Identification 

**Authors**: Anil Osman Tur, Tonje Knutsen Sordalen, Kim Tallaksen Halvorsen, Cigdem Beyan  

**Link**: [PDF](https://arxiv.org/pdf/2607.09443)  

**Abstract**: Long-term animal re-identification (ReID) must remain robust to gradual morphological evolution and seasonal appearance shifts. Although recent vision-language models provide strong pretrained visual representations, adapting them to longitudinal ecological settings remains challenging, particularly under identity and temporal distribution shifts. We present a parameter-efficient CLIP adaptation framework for animal ReID and introduce a continuous metadata-conditioning mechanism that incorporates numerical attributes directly into the prompt representation during training. While low-rank visual adaptation, prompt-based supervision, and cross-modal alignment provide the adaptation framework, the proposed metadata-conditioning strategy constitutes the primary methodological contribution. By preserving the continuous structure of numerical metadata rather than discretizing it into textual categories, the proposed approach enables smooth modulation of the embedding space during training while maintaining a purely visual inference pipeline. Experiments on a seven-year longitudinal fish dataset and multiple wildlife benchmarks demonstrate improved performance under closed-set, open-set, and time-aware evaluation protocols. The results demonstrate that continuous metadata conditioning improves robustness to longitudinal appearance variation and temporal distribution shifts, while parameter-efficient adaptation enables a purely visual inference pipeline without requiring metadata at test time. Code and evaluation splits can be found at: this https URL. 

---
# Test-Time Scaling for Small VLMs on Multilingual Visual MCQ 

**Authors**: Spiros Baxevanakis, Peng-Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.09438)  

**Abstract**: Test-time scaling (TTS) reliably improves reasoning in large language models, but whether it transfers to small open vision-language models remains unclear. We examine this on EXAMS-V, a multilingual visual multiple-choice benchmark, comparing self-consistency, describe-then-reason with PRM-guided beam search, and two post-hoc selectors across Qwen2.5-VL-7B-Instruct and Qwen3.5-4B. What matters is the conditions under which TTS runs, not the search or verification machinery. The largest factor is parseability: an early prompt format left many chains reasoning correctly yet never committing to an answer letter, which a standard answer cue and a guided repair step largely remove. A larger decoding budget removes the rest: raising the per-chain token limit from 1k to 2k recovers 3.7 pp, whereas sampling more chains (8 to 16) adds only 0.15 pp. Once chains have room to finish, elaborate methods contribute little: PRM-guided beam search trails plain self-consistency by 0.39 pp at over eight times the cost, and neither a training-free generative critic nor a trained multimodal PRM beats majority vote across both policies. The largest gain comes instead from the policy model itself (+11.4 pp). Our best configuration reaches 84.1% on the held-out ImageCLEF 2026 test split, ranking first on the Visual MCQ leaderboard. 

---
# A Sovereign, Open-Source Foundation Model for German and English 

**Authors**: Soofi-Team, Benedikt Droste, David Fitzek, Ruben Härle, Lukas Helff, Maximilian Idahl, Alex Jude, Abbas Goher Khan, Maurice Kraus, Timm Ruland, Richard Rutmann, Sebastian Sztwiertnia, Markus Frey, Daniil Gurgurov, Jan Pfister, Tom Röhr, Sebastian von Rohrscheidt, Jörg Bienert, Nicolas Flores-Herr, Simon Gottschalk, Andreas Hotho, Kristian Kersting, Joachim Köhler, Alexander Löser, Wolfgang Nejdl, Simon Ostermann, Jan Plogsties, Patrick Putzky, Mehdi Ali, Michael Fromm, Max Lübbering  

**Link**: [PDF](https://arxiv.org/pdf/2607.09424)  

**Abstract**: We present Soofi S 30B-A3B, a sovereign, open-source Mixture-of-Experts (MoE) hybrid Mamba Transformer foundation model for German and English. Its hybrid design activates only 3B of 30B parameters per token and keeps the inference cache near-constant as context grows, giving it a decisive throughput advantage over dense models for long-context, high-concurrency deployment. Pretrained on roughly 27 trillion tokens with deliberately up-weighted German, Soofi S matches dense 14 to 27B models on aggregate English and German benchmarks while achieving the best code aggregates in both languages among 17 open base models, and outperforms every European sovereign baseline in our comparison, including ones far larger in active parameters. Among fully open models, Soofi S obtains the highest English and German evaluation scores, ahead of Olmo 3 32B and Apertus 70B. Soofi S was built end-to-end on the German Industrial AI Cloud, a sovereign HPC scale AI infrastructure operated by Deutsche Telekom in Munich. Soofi S will be released under highly permissive, open-access terms: weights, selected intermediate checkpoints, full per-source data accounting, hyperparameters, and training and evaluation code. Where source licenses permit, data-construction artifacts are released under permissive licenses; commercially licensed sources are documented with aggregate statistics and exact mixture accounting. 

---
# SVF-CR: Synchronized Visual-Facial Cross-Refinement for Multimodal Ambivalence and Hesitancy Recognition 

**Authors**: Hyein Park, Namho Kim, Junhwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.09417)  

**Abstract**: Ambivalence and hesitancy are subtle behavioral states that are expressed through a combination of verbal content, facial behavior, visual context, and acoustic cues. Effective recognition therefore requires not only extracting informative unimodal representations, but also modeling how temporally aligned behavioral evidence interacts across modalities. In this paper, we propose a synchronized visual-facial cross-refinement framework (SVF-CR) with pairwise multimodal evidence fusion for ambivalence and hesitancy recognition. The proposed method first extracts whole-video segment tokens and cropped-face segment tokens using the same temporal partition. The synchronized visual and facial tokens are refined through intra-modal self-attention and bidirectional visual-facial cross-attention, allowing whole-video context and local facial behavior to mutually refine each other before evidence construction. We then construct segment-level visual-facial evidence using consistency and discrepancy modeling, followed by temporal self-attention and attention pooling. Textual and acoustic features are lightly refined through context self-attention and are fused with the enhanced visual-facial evidence at the final decision stage using pairwise evidence fusion. Experiments on the BAH (Behavioral Ambivalence/Hesitancy) public evaluation split show that the proposed synchronized visual-facial cross-refinement improves public macro-F1 over both global visual-face token fusion and synchronized evidence baselines, achieving a public macro-F1 of 0.7156. Code is available at : this https URL\_SVF-CR. 

---
# Self-Guided Test-Time Training for Long-Context LLMs 

**Authors**: Xinyu Zhu, Zhe Xu, Xiaohan Wei, Yunchen Pu, Fei Tian, Chonglin Sun, Kaushik Rangadurai, Hua Zhi, Frank Shyu, Sandeep Pandey, Luke Simon, Yu Meng, Xi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09415)  

**Abstract**: Long-context processing has become increasingly important for large language models (LLMs), but simply extending the context window does not guarantee effective utilization of long inputs. As input length grows, accuracy often degrades, indicating that models still struggle to identify and use the evidence most relevant to a question. A promising way to improve long-context utilization is test-time training (TTT), which treats the test context as a training example for instance-specific parameter adaptation. However, applying TTT to the entire long context is prohibitively expensive, while adapting on randomly sampled spans introduces severe noise. Because most spans in a long context are irrelevant to the specific question, training on them may even degrade the base model's performance. Our preliminary study shows that TTT is highly sensitive to training-span quality: on LongBench-v2, TTT on randomly sampled spans hurts performance, whereas TTT on oracle spans substantially improves it. Motivated by this, we propose a simple method, Self-Guided TTT (S-TTT): before adaptation, the model identifies the evidence spans it should learn from, and the standard language-modeling training objective is applied only to those selected spans. On two challenging long-context reasoning benchmarks, LongBench-v2 and LongBench-Pro, S-TTT improves accuracy for both Qwen3-4B-Thinking-2507 and Llama-3.1-8B-Instruct, achieving up to a 15% relative improvement. 

---
# On-Device Adaptive Battery Power Prediction for Electric Vehicles 

**Authors**: Avik Bhatnagar, Anton Paule, Tobias Schuermann, Sebastian Reiter, Oliver Bringmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.09400)  

**Abstract**: Adaptive power management in Electric Vehicles (EVs) requires accurate power prediction. Although deep learning models have emerged as highly effective for time-series forecasting in this domain, their performance is prone to degradation when exposed to data with distributions different from the training data. We introduce a novel approach that enables on-device learning in resource-constrained EV systems to continuously adapt pretrained battery prediction models to new, unseen data. We leverage existing pretrained models by transforming them into adaptable versions that retain critical hyperparameter knowledge from their initial training. We comprehensively investigate both online and offline model adaptation strategies. Our results demonstrate significant improvements in forecasting performance across various models and time horizons, achieving mean absolute error reductions of up to 7.49\% and 14.88\% with online and offline adaptation techniques, respectively. This study highlights the substantial benefit of on-device adaptation, resulting in enhanced battery power predictions than unadapted model deployments in real-world EV scenarios. 

---
# Fully Trainable Deep Differentiable Logic Gate Networks and Lookup Table Networks 

**Authors**: Wout Mommen, Lars Keuninckx, Matthias Hartmann, Werner Van Leekwijck, Piet Wambacq  

**Link**: [PDF](https://arxiv.org/pdf/2607.09399)  

**Abstract**: We introduce a novel method for both partial and full optimization of the connections in deep differentiable logic gate networks (LGNs) and lookup table networks (LUTNs). Our training method utilizes a probability distribution over a set of connections per gate/lookup table (LUT) input pin, selecting the connection with highest merit, all whilst the optimal gate types or LUT-entries are learned in parallel. We show that the connection-optimized LGNs outperform standard fixed-connection LGNs on the Yin-Yang, MNIST Handwritten Digits and Fashion-MNIST benchmarks, while requiring only a fraction of the number of logic gates. We achieve 98.92% on the MNIST dataset with two layers of 8000 gates. With only one layer of 8000 gates, we obtain 98.45%, showing that our method requires almost 50 times fewer gates compared to fixed-connection LGNs. Training stability up to ten layers has been ensured by employing a high learning rate, straight-through estimators and trimming constant-output gate types. Additionally, we present a LUT neuron description that enables stable training with backpropagation, tested up to 6-layer deep networks. The model requires four times fewer trainable parameters and still achieves a higher accuracy compared to the fixed-connection LGN training algorithm. Our connection-training algorithm also works well for the LUTNs, achieving an accuracy of 98.88% for two layers of 2000 6-input LUTs. 

---
# STEEL: Sparsity-Aware Fused Attention for Energy-Efficient Long-Sequence Inference on AMD's XDNA NPU 

**Authors**: Victor J.B. Jung, Gagandeep Singh, Joseph Melber, Kristof Denolf, Francesco Conti, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2607.09385)  

**Abstract**: The growing adoption of large language model-based agents within operating system workflows has increased the importance of energy-efficient inference on laptop-class systems-on-chip (SoCs). While cloud offloading remains common, it introduces reliability and privacy concerns that are particularly problematic for agentic workloads. Recent laptop SoCs, therefore, incorporate neural processing engines (NPUs) optimized for energy efficiency; however, effectively mapping attention mechanisms onto NPUs remains challenging due to architectural diversity and explicit data-movement programming models. In this work, we present STEEL, the first open-source implementation of FlashAttention targeting XDNA-like NPUs. STEEL introduces a dataflow formulation of prefill attention, enabling efficient exploitation of spatial parallelism and on-chip memory. Furthermore, STEEL addresses the load imbalance induced by the causal mask by leveraging a sparsity-aware pipeline placement onto the NPU array, reducing synchronization overhead and improving utilization. We evaluate STEEL on the AMD Ryzen AI 9 HX 370 SoC and compare its performance against optimized CPU and GPU implementations. Experimental results show that STEEL reduces energy consumption by an average of 9.17x and 1.75x relative to CPU and GPU baselines, respectively. On XDNA 1, STEEL achieves an average 9.6x latency reduction over the prior state of the art, and delivers a 22.8x speedup on average compared to a layer-by-layer attention implementation on XDNA 2. 

---
# When Routes Run Out: Adversarial Co-Learning and Explainable Robustness in Quantum Repeater Networks 

**Authors**: Brennan Bell, Inti Gabriel Mendoza Estrada, Andreas Trügler, Paul Erker  

**Link**: [PDF](https://arxiv.org/pdf/2607.09378)  

**Abstract**: We study an adversarial bandit problem for entanglement-based quantum-network routing over a modest graph corpus. Alice selects an end-to-end repeater route for an Ekert-91 protocol (E91) representing her move, while Eve selects an attack surface, either edge intercept--resend or repeater memory degradation. Payoffs are drawn from cached SeQUeNCe-simulated E91 transcripts, and Alice accepts a turn when the finite-sample statistic violates the Clauser-Horne-Shimony-Holt (CHSH) bound. Performing adversarial co-learning across 50 structured topologies, we find that learned retention tracks a full-matrix minimax reference closely (Pearson $r=0.99$): under a one-surface Eve action model, bottleneck families have zero retention, while non-bottleneck families follow a $1-1/N$ coverage principle. We then fit decision-tree explanation models to graph-, attack-, and route-level topology-corpus targets and report their faithfulness. Finally, we construct prompt records for local language models to summarize the tree evidence, resulting in an open-source explanation workflow for quantum-repeater network games. 

---
# Diversifying to Verify: When Task-Equivalent Programs Differ in Verifiability 

**Authors**: Shirley Yu, Ruben Martins  

**Link**: [PDF](https://arxiv.org/pdf/2607.09366)  

**Abstract**: Program verification is crucial for software correctness, but producing fully verified programs remains difficult in practice. This paper studies whether implementation structure affects automated verifiability when multiple generated programs are intended to satisfy the same task-level semantics.
We present Diversify2Verify, a staged LLM-based pipeline for Why3 that infers representation-specific contracts, generates and tests diverse recursive and imperative array/list implementations, and attempts verification with bounded verifier-guided annotation repair. We also construct a verification-oriented benchmark of 73 tasks over integers, arrays, and lists, yielding 292 implementation variants. Diversify2Verify verifies 96 artifacts initially and 154 after two repair passes, improving artifact-level verification from 32.9% to 52.7%. At the task level, at least one variant verifies for 49 of 73 tasks, a 67.1% success rate. These results show that task-equivalent implementations can differ substantially in verifiability and that implementation diversity helps find verification-friendly artifacts. 

---
# CtrlVTON: Controllable Virtual Try-On via Visual-Instance-Prompt Segmentation 

**Authors**: Seungyong Lee, Hyun Jun Jang, Sangoh Kim, Sungjoon Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.09362)  

**Abstract**: Virtual try-on (VTO) has made significant progress in realistically transferring garments onto a target person. Yet most systems give the user little control over how a garment should be worn -- its size (loose or fitted), style (e.g., tucked in or untucked, open or closed), and spatial placement on the body. We address this gap with two complementary contributions. First, we define and solve Visual-Instance-Prompt Segmentation via VIP-SAM: given a flatlay image of a garment, segment that specific instance in a photograph of a person wearing it. This is an instance-level task, distinct from the typically studied category-level segmentation. Second, we introduce CtrlVTON, a controllable VTO framework that recasts try-on as an image editing problem and adds segmentation masks as pixel-level control over garment layout, including style, size, and spatial placement on the body. VIP-SAM and CtrlVTON each achieve state-of-the-art results on their respective tasks. In particular, CtrlVTON generates images that follow user-provided layouts far more faithfully than the strongest proprietary editing systems while matching them on garment fidelity. 

---
# Deceptive Grounding: Entity Attribution Failure in Clinical Retrieval-Augmented Generation 

**Authors**: Cedric Caruzzo, Donggeun Yoo, Tae Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.09349)  

**Abstract**: Retrieval-augmented generation evaluation checks whether model claims are factually grounded in retrieved documents. It does not check whether retrieved evidence is attributed to the correct entity.
A clinical RAG response can pass every automated check (zero hallucinations, near-perfect faithfulness, real citations) while presenting drug Y's clinical evidence as evidence about queried drug X. We term this deceptive grounding (DG): a failure invisible to faithfulness, hallucination, and citation checks because every claim is sourced from a real document, about the wrong entity.
Using a controlled factorial benchmark across 13 models, we find DG rates spanning 8-87% at peak adversarial conditions. Medical and biomedical fine-tuned models reach up to 86.7%; domain specialization amplifies the failure rather than mitigating it.
A controlled ablation identifies the mechanism: removing entity-specific clinical evidence from retrieved documents eliminates entity-attribution failure entirely, shifting all failures to confabulation. The two failure modes respond to the same trigger, taking different paths.
Production measurement across 740 drug-disease pairs finds 7.8% overall DG in a deployed RAG system, rising to 13.6% for recently approved drugs. Entity-attribution verification (checking that cited evidence applies to the queried entity) detects DG at 97.0% precision and 98.7% DG recall (IPW-adjusted human gold standard); no existing framework implements it. 

---
# Shortcut Trajectory Planning for Efficient Offline Reinforcement Learning 

**Authors**: Guanquan Wang, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2607.09336)  

**Abstract**: Diffusion-based trajectory planners have shown strong performance in offline reinforcement learning, but their iterative denoising process often incurs high inference cost. Consistency-based planners reduce the number of sampling steps, yet they typically rely on a two-stage teacher--student distillation pipeline that increases training cost and may introduce instability. We propose Shortcut Trajectory Planning (STP), an offline model-based reinforcement learning framework that incorporates shortcut models as efficient trajectory generators. STP trains a conditional shortcut trajectory model in a single stage, supports adjustable one-step and few-step inference through step-size conditioning, and selects candidate plans using a critic augmented with feasibility-aware correction. Across standard D4RL benchmarks, including locomotion, navigation, manipulation, and dexterous control tasks, STP achieves strong performance while simplifying the training pipeline for fast generative planning. 

---
# WILDTRACE: Benchmarking Natural Evidence Trails in Long-Context Reasoning 

**Authors**: Zixin Chen, Peng Liu, Haobo Li, Rui Sheng, Jianhong Tu, Xiaodong Deng, Fei Huang, Kashun Shum, Dayiheng Liu, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09328)  

**Abstract**: Answering complex questions over long documents frequently requires integrating evidence that the source itself disperses naturally across distant passages. In an incident report, the operating condition, design flaw, and missed safety check that jointly explain a disaster may appear dozens of sections apart; in a novel, a character's true motive may surface only through scenes far removed from the moment it becomes relevant. This source-internal evidence integration is central to real-world long-document analysis, yet existing benchmarks largely sidestep it. Needle probes, planted facts, and reverse-engineered multi-hop chains embed evidence that may differ from the host text in distribution, placement, or register, making it unclear whether strong performance reflects genuine source reasoning or distributional artifacts. We introduce WILDTRACE, a benchmark of 481 tasks over 214 naturally occurring long-form sources such as technical incident reports and lesser-known literary narratives, where all evidence trails arise from the document's own causal, temporal, and narrative logic. Drawing on Pearl's causal hierarchy and prior multi-hop reasoning typologies, we define seven source-internal evidence geometries that characterize the distinct relational demands of analytical reading in long documents. A source-first construction pipeline mines candidate trails from document structure before writing questions; each item then undergoes multi-stage validation covering clue necessity, answer groundedness, rubric fidelity, contamination resistance and answerability. As models are increasingly entrusted with real-world high-stakes analytical tasks, this gap between accessing information and reasoning over naturally dispersed evidence emerges as a defining challenge for the next stage of long-context research. 

---
# Letting the Data Speak: Extracting Keywords from Crowdsourced Collections with AI 

**Authors**: Miguel Arana-Catania, Catherine Conisbee, Matthew Kidd  

**Link**: [PDF](https://arxiv.org/pdf/2607.09324)  

**Abstract**: Identifying and assigning keywords at scale is a technical, practical, and ethical challenge for crowdsourced collections. This article reports the findings of the "Extracting Keywords from Crowdsourced Collections" project, which used the Their Finest Hour Online Archive, a crowdsourced Second World War digital collection hosted by the University of Oxford, as a case study. The project evaluated three Natural Language Processing approaches to automate keyword extraction: Named Entity Recognition, Keyword Extraction, and Topic Modelling. It tested these approaches across a range of artificial intelligence techniques, from traditional statistical methods to modern GenAI neural networks. Our quantitative and qualitative findings indicate that Natural Language Processing approaches offer real potential for keyword extraction at scale in crowdsourced collections, but that no single method offers a complete solution and that model choice significantly shapes results. We argue that in crowdsourced collections, where metadata is the direct product of engagement with living contributors, automated keyword extraction raises distinct stewardship responsibilities that must be addressed alongside technical performance. Open-weight, extractive models emerge from our evaluation as best placed to support responsible deployment, while generative AI, despite its abstractive potential, introduces accountability risks that anyone managing crowdsourced collections should weigh carefully. 

---
# Automatic Thematic Indexing of Large Literary Corpora: A Machine Learning Approach to Voltaire's Complete Works 

**Authors**: Miguel Arana-Catania, Gillian Pink, Glenn Roe  

**Link**: [PDF](https://arxiv.org/pdf/2607.09316)  

**Abstract**: Thematic indexing -- the practice of assigning structured conceptual labels to sections of text -- is essential to scholarly access in large-scale literary and historical editions, yet it remains a largely manual, labour-intensive process. This paper explores the application of machine learning to automatic thematic indexing, using two substantial sub-corpora of the Complete Works of Voltaire as a test case: the Essai sur les mœurs et l'esprit des nations and the Questions sur l'Encyclopédie. The task is framed as a multi-label classification problem, in which a model must assign the set of index entries that a professional indexer would apply to a given page of text. We compare a range of approaches -- from encoder-based models with classification heads to generative large language models (LLMs) fine-tuned via Low-Rank Adaptation (LoRA) -- spanning model sizes from approximately 3 to 120 billion parameters. Our best-performing model, from the Mistral family in a 4-bit quantised configuration, achieves F1 scores of up to 0.67; we argue that these figures represent lower bounds, given the inherent subjectivity of professional indexing and the frequency with which model predictions prove semantically valid despite diverging from the print index. We further evaluate cross-corpus generalisation and conduct a detailed qualitative analysis of model behaviour on literary and rhetorical features of the source texts that prove particularly resistant to automated treatment. Our findings have implications for the broader challenge of providing structured thematic access to large-scale literary and historical corpora. 

---
# Creativity, honesty and designed forgetting emerge in small hyperbolic language models 

**Authors**: Kwan Soo Shin, In Seok Kang, Yunkyung Min  

**Link**: [PDF](https://arxiv.org/pdf/2607.09306)  

**Abstract**: Language models are optimised for scale, yet remain functional rather than companionable, and as an assistant personalises into a companion, accumulating memory of one user, it quietly becomes someone, and can silently acquire traits that harm that user. What a companion is becoming, and what would make it worth becoming, has no reliable instrument: trained human raters cannot agree on the answer (Fleiss kappa = 0.074). Here we show that three small language models (146 M to 3 B parameters) sharing a hyperbolic substrate answer both halves of that question. A 146 M behavioural auditor, trained from scratch, detects the compliance gap that those raters cannot (90.7% binary-compliance accuracy); a linear read-out of its frozen representation further detects companion-induced sycophancy, dependence-fostering and confabulated memories on generator families unseen in training (AUROC 0.804 under style-controlled, leave-one-generator-out evaluation, versus 0.721 for a frontier zero-shot judge on the same items). A creative frame-seeder is preferred in 100% of 311 decided pairwise comparisons over four prompting baselines. A memory operating system implements designed forgetting, M(t) = S*exp(-lambda*t), whose predicted skeleton-wallpaper partition emerges only under selective retrieval gating in a four-condition pilot. Creativity, honesty and designed forgetting constitute a small-model route to trustworthy companion AI. 

---
# Risk-Aware General-Utility Markov Decision Processes 

**Authors**: Pedro P. Santos, Fábio Vital, Alberto Sardinha, Francisco S. Melo  

**Link**: [PDF](https://arxiv.org/pdf/2607.09298)  

**Abstract**: We study general-utility Markov decision processes (GUMDPs) with risk-aware objectives. In this framework, an agent aims to optimize a risk measure of the distribution of objective values, where the objective function depends on the frequency of visitation of states induced by the agent's policy. First, we motivate, propose, and formalize risk-aware GUMDPs, which enable agents and decision makers to trade off expected performance by risk aversion while benefiting from the rich set of objectives that can be cast under the framework of GUMDPs. We focus our attention on the entropic risk measure (ERM). Second, we show how we can solve risk-aware GUMDPs with ERM objectives by resorting to online planning techniques. In particular, we propose an approach based on Monte Carlo Tree Search (MCTS) to provably solve risk-aware GUMDPs up to any desired accuracy. Third, we provide a set of experimental results showcasing that our approach is successful when optimizing for a spectrum of risk-aware behaviors in the context of GUMDPs under diverse tasks (standard MDPs, maximum state entropy exploration, imitation learning, and multi-objective MDPs). 

---
# Geopolitical alignment: Endorsement effects in large language models 

**Authors**: Maxim Chupilkin  

**Link**: [PDF](https://arxiv.org/pdf/2607.09262)  

**Abstract**: Large language models (LLMs) are increasingly used to summarize and evaluate policy-relevant information, but it remains unclear whether their judgments are implicitly shaped by geopolitical cues. I study this question with an endorsement experiment in which four LLMs evaluate the same international economic and security policies after each policy is randomly described as supported by the United States, the European Union, China, or Russia. In the numeric-only condition, GPT-5, Claude Sonnet, and Gemini rate China- and Russia-endorsed policies substantially lower than identical policies endorsed by the United States or the European Union; DeepSeek is the main exception. A second condition asks models to provide a short justification with the score. This request leaves the broad Western/non-Western gap intact for GPT-5 and Claude Sonnet, attenuates Gemini's penalties, and sharply activates China and Russia penalties in DeepSeek. The justifications indicate that Western endorsement is often treated as a credibility cue, whereas Chinese and Russian endorsement is treated as a cue for data security, sovereignty, surveillance, or geopolitical risk. These findings show that LLM policy evaluations can depend on the identity of a foreign endorser even when policy content is held fixed. 

---
# Blockchain-Linked Auditable Decision Management for Telecom/IoT Fraud-Control Requests 

**Authors**: Saviz Changizi, Nasibeh Mohammadzadeh, Mohammad Shojafar, Rahim Tafazolli  

**Link**: [PDF](https://arxiv.org/pdf/2607.09259)  

**Abstract**: Telecom fraud-control studies often stop at detector-level classification, but deployment use requires request-level policy resolution, lifecycle traceability, and auditability. This paper reframes fraud control as blockchain-linked auditable decision management for synthetic telecom/IoT fraud-control requests, and its main result is that the QLoRA-tuned LLM branch becomes much more usable than zero-shot prompting but mainly approaches, rather than outperforms, a lower-cost centralized ensemble. The framework maps each synthetic deployment record to a managed request, blocks explicit out-of-boundary cases through a deterministic hard-fraud gate, scores non-hard requests using centralized ML (M1), federated meta-learning (M2), or LLM-family risk sources (M3), and resolves actions through a shared five-state policy, two-zone refinement mechanism, and local Ethereum-compatible audit layer. Evaluation uses separate synthetic training data and a 100,000-record deployment replay corpus, so the study should be read as controlled drift-replay evidence rather than field validation or proof of live deployability. On validation, M1 gives the strongest balance, with legitimate-request FPR 0.0890 under the 0.10 operating cap and soft-fraud recall 0.8341. On labeled deployment replay, however, the legitimate-FPR gap becomes large: M1 rises to 0.1646 and M3-QLoRA to 0.1801, while M3-QLoRA reduces the M3-Base legitimate FPR from 0.3915 and reaches 0.8240 soft-fraud recall. Blockchain telemetry shows that lifecycle gas, cost, latency, and throughput differences are driven by submitted off-chain decision profiles rather than changes in fraud logic. 

---
# LLMs for health: Perceived benefits, risks, intention to use AI chatbots, and willingness to self-disclose across sensitive health topics 

**Authors**: Gwenn Beets, Anniek Jansen, Saar Hommes, Ruben D. Vromans, Leonie Westerbeek, Supraja Sankaran, Julia C.M. van Weert, Emiel J. Krahmer, Nadine Bol  

**Link**: [PDF](https://arxiv.org/pdf/2607.09253)  

**Abstract**: AI chatbots are increasingly used for answering health-related questions. This study examines the role of topic type discussed with an AI chatbot and individual characteristics on perceived benefits and risks, intention to use an AI chatbot, and willingness to self-disclose health information. We conducted an online experiment with a 2 (topic type: physical versus psychological, between-subjects) x 2 (topic sensitivity: low versus high, within-subjects) mixed design among a Dutch representative sample (N = 1,388). Results showed that perceived benefits were positively associated with intention and willingness to self-disclose, while perceived risks were negatively associated. Moreover, participants reported higher usage intentions for low-sensitive topics compared to high-sensitive topics. Furthermore, perceptions, intention, and willingness to self-disclose varied by individual characteristics. Overall, our findings suggest that intentions to use AI chatbots and self-disclosure of health-related information are primarily related to perceived benefits and risks and to personal characteristics rather than to topic type. 

---
# All you need is SAMPAT 

**Authors**: Jayadeva, Madhur Aswani  

**Link**: [PDF](https://arxiv.org/pdf/2607.09235)  

**Abstract**: The current state of the art in AI/ML rests on deep neural architectures, which, in general, suffer from a lack of interpretability. Interpretability is crucial to gleaning insights while analyzing experimental data, where quantitative predictions may not be adequate for a scientist. We present a three layer neural architecture, SAMPAT (Smooth Approximation via Multivariate Polynomials and Analytic Transformations), that can provably learn a continuous, everywhere differentiable function, that can approximate any smooth function arbitrarily closely. SAMPAT's approximant can be expressed as a closed and compact algebraic, analytic expression, providing complete interpretability. Experiments on synthetic and benchmark datasets indicate that SAMPAT yields competitive performance with simpler representations. For many tasks, a two layer SAMPAT suffices. By imposing restrictions on the connectivity between neurons, SAMPAT may be used to provide a range of approximants, including regular and trigonometric polynomials, rational expressions, Gaussians, mixtures of Gaussians, as well as arbitrary combinations of the same; without restrictions, it learns a suitable structure. SAMPAT may be used to factorize polynomials and model nonlinear systems. With the addition of skip connections, a 4 to 6 layer SAMPAT is adequate to represent a substantive range of methods widely used in AI/ML, allowing the choice of a model's family, not just its parameters, to also be optimized as part of the learning process. 

---
# Git-Assistant: Planning-Based Support for Updating Git Repositories 

**Authors**: Alfredo Garrachón Ruiz, Tomás de la Rosa, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2607.09224)  

**Abstract**: Version control systems are essential for collaborative software development, yet tools like git remain challenging for many practitioners. Recent advances in Large Language Models (LLMs) offer promising capabilities for interpreting developer intent, but their effectiveness in repository management tasks is limited by the need for formal reasoning. This work introduces Git-Assistant, an AI-based assistant that combines LLMs with automated planning to support developers in executing non-trivial git operations. The assistant analyzes repository context, translates natural language requests into actionable command sequences, and incorporates planning techniques to ensure correctness and safety. We present a systematic evaluation methodology using synthetic and randomized git environments, comparing the performance of LLM-only and planning-augmented variants across multiple metrics. Experimental results demonstrate that integrating formal reasoning with LLMs improves reliability and reduces errors in repository management, highlighting the potential of hybrid AI approaches for intelligent developer assistance. 

---
# Tactile and Vision Conditioned Contact-Centric Control for Whole-Arm Manipulation 

**Authors**: Rishabh Madan, Angchen Xie, Samantha Saak, Andres Blanco, Dohyeok Lee, Sarah Grace Brown, Yunting Yan, Mark Zolotas, Jose Barreiros, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.09218)  

**Abstract**: Whole-arm manipulation involves direct contact with the environment while the robot completes a task by distributing contact across multiple links as contacts form, slide, and break. This setting breaks common implicit assumptions in many learning-based manipulation pipelines: arm configuration tightly couples motion and contact forces, contact state is partially observed under occlusion, and purely learned rollouts can become physically inconsistent under distribution shift because many multi-link contact configurations are sparsely represented in the data. To address this, we propose TACTIC (Tactile and Vision Conditioned Contact-Centric Control), a receding-horizon controller for whole-arm manipulation. TACTIC uses a contact-centric hybrid predictive model that combines RGB-D, distributed tactile sensing, and a compact 2D proximity representation. The model couples a learned, action-conditioned latent dynamics model with analytical kinematics through contact Jacobians, enabling rollouts of future contact configurations and interaction forces. TACTIC integrates these rollouts into a sampling-based MPC planner with contact-aware action sampling: contact Jacobian-based projections steer sampled action sequences toward force-modulating directions, and objectives defined over predicted proximity and interaction forces trade task progress against whole-arm force regulation. We evaluate TACTIC in simulation against state-of-the-art model-based and model-free methods, and perform ablations that isolate the contribution of each design choice. TACTIC consistently outperforms other methods. We further demonstrate real-world performance on a robot with distributed tactile sensing across three whole-arm manipulation tasks that require multi-contact trajectories: turning over and repositioning a manikin, and goal-reaching in a 3D dynamic maze. Website: this https URL 

---
# Interference and Retention in Continual Learning 

**Authors**: Julius Störk  

**Link**: [PDF](https://arxiv.org/pdf/2607.09202)  

**Abstract**: Continual learning commonly relies on post-hoc mechanisms such as replay, elastic regularization, or distillation. This work argues that forgetting should instead be modeled directly as interference between tasks. In the frozen-feature regime, forgetting from learning a new task is exactly the interference energy induced on the old task. In deep networks, the same quantity is recovered through path-averaged curvature with minimal additional forward passes.
When task supports are disjoint, forgetting can be eliminated structurally and when task supports overlap in conflicting directions, a non-zero distortion floor is unavoidable. The same geometry optimally merges models through task-aware orthogonalization. From this analysis we derive Interference-Gated Functional Allocation (IGFA), a replay-free, Fisher-free method that shares directions when tasks align and protects them when they conflict. Across benchmarks, IGFA achieves lossless retention when tasks are structurally separable and moves unavoidable cost from irreversible forgetting into deferred but recoverable plasticity when they are not. It matches the strongest replay-free structural baselines on dissimilar-task streams and improves on unconditional projection when similarity makes transfer worth preserving. 

---
# Generative Communications: Overview, Technologies, and Trends 

**Authors**: Wenjun Zhang, Zhiyong Chen, Tong Wu, Guo Lu, Li Song, Feng Yang, Meixia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2607.09183)  

**Abstract**: The groundbreaking development of generative artificial intelligence (AI) is rapidly boosting the ability to generate content such as images and videos, reshaping communication paradigms. This article introduces generative communications (GenCom), a novel paradigm for 6G networks in which large AI models (LAMs) drive semantic understanding, reasoning, and content generation, embedding these into the communication process. Unlike traditional systems that strictly pursue accurate bit transmission, GenCom enables transmitters to convey only minimal yet sufficient information, while receivers leverage shared generative priors and knowledge bases to synthesize the intended output. Communication is thus redefined as controlled generation rather than data reproduction. We formalize the concept of GenCom, clarify its AI-native and generation-driven properties, and present its core mechanisms. A two-layer GenCom architecture supported by key enabling technologies is proposed, and analysis of four representative application scenarios demonstrates that GenCom offers ultra-efficient transmission, semantic-level robustness, and new network functions. Finally, we outline future research directions, including foundational theory and real-time processing, highlighting a promising pathway toward 6G networks. 

---
# Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations 

**Authors**: Nada Zine, Tristan Coignion, Vincenzo Stoico, Clément Quinton, Romain Rouvoy, Patricia Lago  

**Link**: [PDF](https://arxiv.org/pdf/2607.09172)  

**Abstract**: Large Language Models are reshaping how software is developed and maintained. They are typically deployed in production using inference engines such as vLLM, which can efficiently serve pre-trained, highly configurable models. While prior work has focused on model architectures and hardware acceleration, the impact of inference engine configuration on energy consumption, performance, and output quality remains poorly understood.
In this paper, we present a large-scale controlled study of three selected vLLM configuration options: attention kernel type, prefix caching, and chunked prefill. We evaluate all combinations of these configurations across 5 open-weight LLMs and 5 diverse inference tasks, totaling $9,000$ runs and $93,600$ measures. We analyze energy consumption, latency, and accuracy, and examine both main effects and interaction effects between configuration options and tasks.
Our results show that the studied configuration options significantly impact energy and performance, mainly driven by attention type and prefix caching, while chunked prefill has a limited effect under the default vLLM serving configuration and evaluated workloads. These effects are highly model- and workload-dependent, and no configuration is universally optimal. We further show that model choice dominates global trade-offs, while configuration tuning provides local improvements along the Pareto frontier. Unexpectedly, inference options can also affect model accuracy. 

---
# A Personalized Computational Framework for Assessing the Sufficiency of Partially Observed Data in Healthcare AI models 

**Authors**: Qingchu Jin, Felistas Mazhude, Jamie B. Rabb, Robert S. Kramer, Douglas B. Sawyer, Raimond L. Winslow  

**Link**: [PDF](https://arxiv.org/pdf/2607.09165)  

**Abstract**: Achieving early and timely diagnosis and treatment for disease is a major challenge. Recent applications of machine learning (ML) algorithms trained on patient data have shown promise in many different settings for predicting the patient health state. A challenge often faced when applying these ML algorithms is that at any given time, not all clinical variables (features) needed as input to perform prediction tasks are available. We define the concept of full-feature-capacity (FFC) to refer to prediction performance when such algorithms make use of all features on which they were trained. We then introduce Feature Sufficiency Analysis (FSA) - an analysis for determining whether a subset of all clinical features needed by an AI model is sufficient to achieve FFC. FSA estimates the underlying distributions of missing variables conditioned on features that are available. FSA provides a patient-specific assessment of whether the existing set of measured features achieves FFC. If yes, then there is no need to acquire further inputs and a ML-based prediction. We provide two case studies: prediction of need for postoperative prolonged ventilation in patients recovering from heart surgery; 10-year mortality prediction in an outpatient cohort. We also demonstrate that FSA also provides a clinically interpretable feature-ranking methodology based on prediction sufficiency, identifies intrinsically hard-to-predict patient populations, and has the potential to perform cost-aware optimization for clinical data acquisition. FSA provides a generic computational approach for determining whether incomplete clinical information is sufficient to support trustworthy AI-assisted clinical decision-making, thereby facilitating the prospective deployment of healthcare AI systems across diverse clinical settings. 

---
# ReGen: Hierarchical Multi-Prompt Representation Generation for Efficient Waveform Diffusion Models 

**Authors**: Sang-Hoon Lee, Ha-Yeong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09134)  

**Abstract**: Representation alignment (REPA) has been investigated to accelerate diffusion training, but we observe that regularizing intermediate representations in diffusion Transformers (DiT) may implicitly entangle latents and limit generative capacity. To address this issue, we propose ReGen, a hierarchical multi-prompt representation generation framework that jointly estimates multiple vector fields for both representations and data within a single diffusion model. We further introduce generalized flow matching (GFM) to improve the generalization of conditional flow matching (CFM). We validate ReGen on single-stage waveform diffusion models including neural audio codec and Wave-VAE. ReGen significantly improves waveform generation quality from highly compressed latent representations at 12.5 Hz. We also present ReGenVoice, a latent diffusion model (LDM)-based text-to-speech model that achieves strong speech intelligibility (WER) and speaker similarity (SIM) with a small dataset. Moreover, operating the LDM at 6.25 Hz with rich semantic and acoustic latent representation enables efficient training and sampling, requiring only 1 day of training on 4 GPUs and fast inference with an RTF of 0.08. Audio samples are available at this https URL. 

---
# IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation 

**Authors**: Yiting Wang, Jingyi Zhang, Wenhu Zhang, Ke Chao, Yves Liang, Kun Cheng, Kang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.09133)  

**Abstract**: While large-scale text-to-image generative models have achieved unprecedented visual performance, their inherent reliance on multi-step iterative solvers incurs severe inference latency. Few-step distillation targeting the Classifier-Free Guidance (CFG) trajectory has emerged as the prevalent dual-dimensional compression paradigm. However, existing frameworks remain subjugated by a coarse-grained blind injection paradigm that perpetually enforces a globally static guidance strength while indiscriminately sampling the supervisor timestep. This state-agnostic design completely disregards the intrinsic nature of image generation as a dynamic evolutionary process characterized by progressive entropy reduction, which not only restricts the performance boundary of few-step compression but also precipitates severe CFG over-conditioning artifacts. To transcend these limitations, we re-examine the distillation procedure through the theoretical lens of Information Theory, formally modeling it as a dynamic mutual information game constrained by the Information Bottleneck (IB) principle. Specifically, we dismantle traditional blind assumptions via a dual-track adaptive framework. To determine the injection target, we propose an instance-aware selection mechanism that transmutes the intractable KL divergence constraint into a zero-overhead closed-form solution predicated on the local vector field norm. To regulate the injection strength, we introduce an entropy-aware schedule that dynamically decays alongside the SNR, applying maximal thrust for initial structural anchoring before smoothly reverting to the natural manifold to refine micro-details. Extensive empirical evaluations corroborate that our framework fundamentally eradicates over-conditioning artifacts, shattering the performance ceiling to achieve SOTA generative fidelity under extremely stringent 2-step configurations. 

---
# Augmenting Fundamental Analysis with Large Language Models: A RAG-Based System for Generating Investor Briefs 

**Authors**: Bartosz Ziółko, Kacper Dobrzeniewski  

**Link**: [PDF](https://arxiv.org/pdf/2607.09121)  

**Abstract**: In this study, we examine the opportunities brought by Large Language Models (LLMs) to various aspects of fundamental analysis of companies based on their reports as well as data and documents describing macroeconomic situation like GDP and inflation changes as well as documents filled to the U.S. Securities and Exchange Commission (SEC) which can be found in EDGAR. We were preprocessing those data and than sending via API to gpt-4o model in a Retrieval-Augmented Generation (RAG) like regime. We prepared as well a document describing an exemplar investor knowledge based on Kitchin cycles. We were scanning data important for analysis of 9 companies for 4 weeks. Using LLM we were producing automatic briefs about them. They were sent to nine participants who are individual investors to evaluate usefulness of such approach to data analysis. 

---
# Event Stream based Multi-Modal Video Anomaly Detection: A Benchmark Dataset and Algorithms 

**Authors**: Peipei Zhu, Yueqing Niu, Lin Zhu, Guanchong Niu, Yang Yu, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.09114)  

**Abstract**: Video anomaly detection (VAD) is critical for automated surveillance but remains fragile under challenging conditions such as illumination variations, fast motion, and complex backgrounds when relying solely on visible light videos. To address these limitations, we propose EVAD, an event enhanced VAD framework that jointly exploits conventional video and event streams captured by bio inspired event cameras. Event sensors asynchronously capture brightness changes with high temporal resolution, offering robustness to motion blur and extreme lighting, and providing motion salient cues complementary to video based visual information. To support multi modal VAD research, we construct a large scale visible event benchmark comprising 6.3 billion events and 376,368 video frames collected under diverse illumination levels, motion patterns, and background complexities, filling the gap of realistic and scalable datasets for event based anomaly detection. Building upon this dataset, we design a contrastive multi modal pretraining framework to learn discriminative event representations by aligning semantic embeddings across event streams, visible videos, and textual descriptions. An adaptive fusion module then dynamically integrates event based temporal cues with video based spatial semantics, improving robustness to environmental disturbances. Experiments on benchmarks and the proposed TJUTCM Pha dataset demonstrate that E VAD consistently outperforms methods, validating the effectiveness of event-based sensing for VAD in real world scenarios. 

---
# Integrating Large Language Models and Graph Convolutional Networks for Semi-Supervised Image Classification 

**Authors**: Camila Piscioneri Magalhães, Lucas Pascotti Valem  

**Link**: [PDF](https://arxiv.org/pdf/2607.09104)  

**Abstract**: While the growing availability of image data has driven significant advances, labeling datasets remains costly and time-consuming. Therefore, semi-supervised approaches such as Graph Convolutional Networks (GCNs), which learn from both labeled and unlabeled data, have emerged as a promising solution. One of the primary challenges in applying GCNs to image classification is graph construction, since, unlike in citation networks or similar domains, images typically do not come with a predefined structural representation. For visual data, most studies construct graphs based on the similarity between feature vectors from pretrained deep learning backbones, typically by employing kNN or reciprocal kNN algorithms. Although Large Language Models (LLMs) have shown remarkable capability in capturing high-level semantics, their integration with GCNs for image classification remains underexplored. Aiming to fill this gap, our approach uses a Vision Language Model (VLM) to generate textual image descriptions, which are then processed by an LLM to estimate semantic similarity scores between connected images. These scores guide the pruning of edges in kNN and reciprocal kNN graphs, filtering out semantically irrelevant neighbors. Experimental results reveal that leveraging LLMs for graph refinement can improve classification accuracy, particularly for kNN graphs and some backbones. The source code is publicly available at this http URL. 

---
# Beyond Metadata: CAPRA for Hidden Subgroup Analysis under Missing Metadata in Medical Imaging 

**Authors**: Yawen Li, Yan Li, Zhe Xue, Yingxia Shao, Meiyu Liang, Guanhua Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.09102)  

**Abstract**: Medical imaging models are often deployed without the demographic, acquisition, and quality metadata needed for subgroup auditing. Once those metadata disappear, clinically critical failure modes can be masked by strong aggregate performance, and many robust-learning methods lose the group structure they rely on. We present CAPRA, a calibrated proxy-axis framework for hidden subgroup analysis under missing metadata. CAPRA predicts image-derived semantic axes, calibrates axis posteriors on a small metadata-labeled split via patient-level cross-fitting, and organizes those posteriors into a calibrated subgroup interface that supports both deployment-time failure analysis and downstream robust learning without requiring subgroup labels at deployment. Across fundus, dermoscopy, and chest radiography, CAPRA reveals disparity patterns missed by metadata-only slicing, remains informative under dataset shift, and produces subgroup partitions that align more closely with explicit failure axes than image-only or latent-slice baselines. The same interface can also be reused by downstream robust learners, although those gains are domain-dependent. Overall, CAPRA turns hidden subgroup analysis under missing metadata into a calibrated, interpretable, and reusable subgroup interface for deployment-time analysis and robust transfer. 

---
# A Coreset Selection Framework with Ensemble Aggregation for Image Classification 

**Authors**: Pedro Rocha Dantas, Lucas Pascotti Valem  

**Link**: [PDF](https://arxiv.org/pdf/2607.09100)  

**Abstract**: The rapid growth of image data has produced large-scale datasets, raising concerns about the time and memory costs of model training. Selecting representative training subsets, however, remains challenging: individual sample contributions are unclear, and model behavior varies across datasets and runs. We address these challenges with a framework that combines coreset selection with an ensemble aggregation over multiple runs. For coreset selection, we propose SCOre-Stratified Selection (SCOSS), which partitions the training data into intervals based on a chosen score and samples from each interval. The ensemble combines predictions from multiple runs, each performed on an independently sampled training subset. As baselines, we use moderate and random selection, each in original and class-balanced versions. We assess the framework with Simple Graph Convolution (SGC) and Support Vector Machine (SVM) classifiers under different sampling ratios. Experiments show that SCOSS is competitive with baselines, often the best choice for SGC, and enables favorable trade-offs between accuracy and efficiency. On the fine-grained dataset, SGC with SCOSS outperforms SVMs when using fewer labeled samples. The code and supplementary materials are publicly available at this http URL. 

---
# PRecG: Legal Precedent Retrieval with Graph Neural Networks and Rhetorical Role Segmentation 

**Authors**: Devanshu Verma, Vasudha Bhatnagar, Vikas Kumar, Balaji Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2607.09094)  

**Abstract**: Legal precedent retrieval is a fundamental task in legal case preparation, planning, litigation strategy, and legal research. Current approaches for automatic precedent retrieval map legal documents to a low-dimensional semantic space and compute similarity based on the proximity of their representations. These approaches treat legal documents as monolithic texts, ignoring the rhetorical organization of the legal technicalities. Ergo, they overlook nuanced legal meanings and fail to distinguish the contextual significance of legal entities and concepts that vary based on their rhetorical roles within the document.
To address this insufficiency, we propose the PRecG pipeline that computes the similarity between pairs of legal judgments by hierarchically learning their representations. The process begins by decomposing each document into distinct semantic units (segments) based on the rhetorical roles of sentences. For each rhetorical segment, a knowledge graph is constructed to capture the legal entities and their relationships within the segment. Contextual representations of the entities are then learned and aggregated to derive segment-level embeddings. These embeddings are further integrated to produce a unified document-level representation, and finally, the semantic similarity between a pair of documents is computed. We validate the performance of the proposed approach through extensive experiments on a benchmark Indian legal dataset, comparing it against state-of-the-art baselines to demonstrate its effectiveness. 

---
# OmniMapBench: Benchmarking Visual-Centric Reasoning on Diverse Map Documents 

**Authors**: Yang Chen, Yunwen Li, Yufan Shen, Minghao Liu, Tianyu Zheng, Bin Fu, Qunshu Lin, Zhi Yu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.09068)  

**Abstract**: Recent advancements in LVLMs necessitate robust benchmarks for complex, visually grounded reasoning. A critical limitation is identified in many document understanding benchmarks: visual content is often reducible to text, enabling high performance without genuine visual grounding. To address this limitation, OmniMapBench is introduced to foster visual-centric reasoning for map documents. The benchmark comprises 2,096 manually annotated question-answer pairs across 1,603 map documents from nine categories. It is designed to probe a hierarchy of skills, ranging from perception to multi-step visual reasoning. To quantify benchmark properties, a simple yet effective benchmark-level metric is proposed: the Visual Dependency Index (VDI), defined as the accuracy drop when images are replaced with question-agnostic descriptions. OmniMapBench exhibits higher VDI than established benchmarks, which quantitatively validates its focus on irreducible visual reasoning. Comprehensive evaluations of 25 leading LVLMs are conducted on OmniMapBench. A significant performance gap is observed, with the top-performing model achieving only 75.03\% accuracy. This result underscores the challenges posed by OmniMapBench to current LVLMs. This work aims to catalyze progress in visual-centric reasoning for document understanding of LVLMs. The dataset and code are publicly available at this https URL. 

---
# Inside the Skill Market: From Software Engineering Activities to Reusable Agent Skills 

**Authors**: Jialun Cao, Xinru Yan, Songqiang Chen, Yaojie Lu, Zhongxin Liu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2607.09065)  

**Abstract**: Software engineering (abbrev. SE) has continuously evolved through increasingly powerful forms of reuse, from source code and libraries to components and services. Recent advances in AI agents have introduced a potentially new reusable artifact: skills. Emerging agent skill repositories and marketplaces enable developers to package, share, and reuse SE expertise as reusable skills. This trend raises a fundamental question: what SE activities are being encapsulated into reusable skills? Existing studies primarily focus on a broad range of skills acquisition, safety, or benchmarking, while lacking a systematic understanding of SE-specific skills and their coverage across the software development lifecycle. To address this gap, we conduct the first large-scale empirical study of SE skills in public repositories and marketplaces. We collect and analyze a large corpus of SE skills, examining the activities they encapsulate, lifecycle coverage, evolution characteristics, and evaluation mechanisms. Our findings reveal that SE activities are increasingly becoming reusable artifacts via skills and suggest promising research opportunities for skill recommendation and engineering-oriented structuring, as well as the need for mechanisms to encapsulate high-context SE activities into reusable skills. Overall, our study provides the first activity-centric characterization of SE skills and reveals how SE activities are increasingly being transformed into reusable skills. These findings offer new insights into skill reuse, ecosystem development, and the future of agent-centric SE. 

---
# On Locality and Length Generalization in Visual Reasoning 

**Authors**: Pulkit Madan, Sanjay Haresh, Reza Ebrahimi, Sunny Panchal, Apratim Bhattacharyya, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2607.09061)  

**Abstract**: A striking feature of the human visual system is that it ingests visual information through a series of local foveated glimpses, rather than a single global computation. This makes human vision distinctly different from most popular computer vision models in use today, which input images globally and in a single shot. A natural question therefore is whether local, sequential vision models may provide any fundamental computational benefits in addition to being biologically more plausible than global models. In this work, we investigate this question from the perspective of visual state tracking and length generalization. Inspired by recent studies of length generalization in language models, we study the behavior of vision models trained on simple vision tasks that require the aggregation of local information across an image. Our experiments reveal that, similar to language models, vision models can learn to exploit global shortcuts and thereby fail to generalize over task length or complexity. We also show that recurrent vision policies based on strictly local perception can mitigate these failures, thereby allowing models to generalize on these tasks. Our results show that local attention may be an essential overlooked requirement for robust compositional generalization. 

---
# Quantum Logic as the Logic of Contexts 

**Authors**: Haruki Emori, Atsushi Iriki, Andrei Khrennikov, Kazunori Kondo  

**Link**: [PDF](https://arxiv.org/pdf/2607.09032)  

**Abstract**: Quantum logic is usually presented as a non-classical departure from ordinary reasoning forced on us by quantum mechanics, with classical logic kept as the secure starting point. We argue for the opposite order of explanation in a finite and fully computable setting. The free orthomodular lattice on two generators has ninety-six elements, the direct product of a six-element non-distributive factor and a sixteen-element Boolean factor. Reading the first factor as a register of contexts and the second as Boolean content, we obtain a calculus whose elements are context--bit-vector pairs and whose operations act component by component. With this calculus we establish three results. First, we classify the six layers by commutativity, identifying the central kernel of context-neutral propositions together with a dual central layer in which all complementary contexts are present. Second, we show that orthocomplementation rearranges the layers exactly as the complementation of the small factor rearranges its elements, which makes the duality among the layers rigid rather than accidental. Third, we prove that the operation forgetting the context is a surjective homomorphism of orthocomplemented lattices whose quotient is the classical Boolean algebra, so that classical logic is a six-to-one, information-losing image of the contextual calculus. 

---
# Evolutionary Intelligence for Scientific Discovery: From Evolutionary Computation to Cumulative Discovery Systems 

**Authors**: Chao Wang, Lingling Li, Fang Liu, Licheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.09025)  

**Abstract**: Artificial intelligence (AI) is shifting scientific discovery from task-specific workflows towards autonomous systems that organize exploration with experimental and human feedback in open-ended candidate spaces. Evolutionary computation (EC) provides a computational basis for feedback-driven discovery because population-based search can maintain diverse scientific candidates while steering exploration through accumulated evidence. However, EC predominantly focuses on candidate refinement for predefined problems, whereas cumulative discovery requires experience retention. To bridge this gap, this review introduces evolutionary intelligence (EI) for scientific discovery. EI characterizes scientific AI systems that sustain exploration by linking candidate refinement with experience retention across evolutionary cycles. We introduce a five-dimensional analytical framework that asks what evolves, how candidates change, why candidates are selected, where feedback originates, and when evolution occurs. This framework clarifies how EI transforms isolated search trajectories into cumulative scientific insight. We further demonstrate this paradigm across diverse discovery modes, from evolving concrete scientific entities to orchestrating automated research workflows. Finally, we identify critical bottlenecks regarding evaluation, process traceability, and shared infrastructure, providing a concrete roadmap for advancing the transition from EC to EI in scientific discovery. 

---
# Video Generation Models are General-Purpose Vision Learners 

**Authors**: Letian Wang, Chuhan Zhang, Rishabh Kabra, Jasper Uijlings, Steven Waslander, Andrew Zisserman, Joao Carreira, Kaiming He, Misha Andriluka, Eduard Gabriel Bazavan, Andrei Zanfir, Cristian Sminchisescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.09024)  

**Abstract**: Driven by next-token prediction, NLP shifted from task-specific models into powerful generalist foundation models. What, then, is the equivalent catalyst needed to achieve a general-purpose model in computer vision? In this paper, we contend that large-scale text-to-video generation serves as a strong pre-training paradigm for computer vision, providing the necessary spatiotemporal priors, vision-language alignment, and scalability required for general visual intelligence. We introduce GenCeption, which leverages a pre-trained video generative diffusion backbone to define a feed-forward perception model, capable of performing various vision tasks steered by text instructions. Empirical results demonstrate that GenCeption achieves state-of-the-art performance across a diverse suite of tasks, including depth, surface normal, and camera pose estimation, expression-referring segmentation, and 3D keypoint prediction, often matching or surpassing specialized models (e.g. DepthAnything3, SAM3, D4RT, VGGT-Omega, Sapiens, David, Genmo, and Lotus-2). Furthermore, the video generative pretrained backbone outperforms alternative pretraining paradigms (e.g., V-JEPA, and Video MAE) under comparable settings. Importantly, GenCeption exhibits preliminary data and model scaling properties along with exceptional data efficiency, where it achieves comparable performance with leading models like D4RT and VGGT-Omega with 7 to 500 less training data. Finally, GenCeption also exhibits intriguing emergent behaviors: a model trained exclusively on synthetic human videos generalizes to real-world footage and out-of-distribution object categories (e.g., animals and robots). These findings suggest that video generation is not merely a synthesis tool, but a foundational path toward generalist vision intelligence for the physical world. Project page: this https URL 

---
# Phone Segmentation and Recognition through Phonological Activation Mapping 

**Authors**: Shikhar Bharadwaj, Kwanghee Choi, Stephen McIntosh, Chin-Jou Li, Eunjung Yeo, Daisuke Saito, Nobuaki Minematsu, Shinji Watanabe, Jian Zhu, David Harwath, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2607.09020)  

**Abstract**: Phone segmentation and recognition are inherently related tasks, yet modern approaches typically model them separately. We argue that phonetic structure is already latent in the representations of self-supervised speech models (S3Ms), and one only needs to steer them to solve both tasks. We leverage S3M-based Phonological Activation Mapping (SPAM), which maps each S3M representation frame to a vector of phonological feature activations, such as voicing and nasality. On top of SPAM, we introduce two simple but effective lightweight, gradient-descent-free prediction heads: a recognition head and a segmentation head. Our method requires less than a minute of phonetic transcriptions, and generalizes to unseen phones during training. Across a diverse range of datasets, our approach attains strong segmentation and recognition performance. 

---
# Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing 

**Authors**: Ajay Narayanan Sridhar, Ronak Singh, Mehrdad Mahdavi, Vijaykrishnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2607.09015)  

**Abstract**: We study contextual bandit problems with correlated arms and access to surrogate reward signals produced by a machine learning model, motivated by applications such as large language model (LLM) routing. Unlike classical contextual bandits that rely solely on bandit feedback and assume conditional independence across arms, our setting allows context-dependent inter-arm correlations and auxiliary reward information that may be noisy or misspecified. We propose algorithms that leverage such surrogate rewards through two complementary designs. A coupled reward-mixing approach pools true and surrogate rewards to accelerate learning when surrogate signals are reliable, while a decoupled prediction-mixing approach maintains separate estimators for bandit feedback and surrogate rewards and adaptively combines their predictions. This decoupling yields robustness to surrogate misspecification, recovering regret guarantees comparable to reward-only bandit methods in the worst case, while achieving improved regret when surrogate predictions are sufficiently informative. We provide theoretical regret analyses for both approaches and evaluate them on LLM routing benchmarks under varying accuracy versus cost trade-offs. The results demonstrate improved sample efficiency and consistently better accuracy-cost trade-offs compared to standard contextual bandit baselines and strong static routing methods. 

---
# Model Agnostic Graph Prompt Learning for Crystal Property Prediction 

**Authors**: Shrimon Mukherjee, Kishalay Das, Partha Basuchowdhuri, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2607.08996)  

**Abstract**: Graph Neural Networks have emerged as a powerful tool for the fast and accurate prediction of various crystal properties. These models often encode domain-specific knowledge into their graph encoding modules, which increases their parameter size and makes their performance heavily dependent on domain expertise. Added to this, explicitly incorporating all chemical and structural features, that might influence a specific crystal property into the GNN encoder, is a challenging task. In this work, we propose a soft prompt learning framework that captures latent features essential for property prediction, which are not explicitly provided to the GNN. We introduce a novel multilevel graph prompt learning framework comprising both node-level and graph-level soft prompts. At the node level, we capture the local chemical semantics of different atom types, while at the graph level, we encode the global structural symmetry of the crystal graph. Our proposed prompt learning framework is lightweight and seamlessly integrates with any existing GNN encoder. Extensive experiments on popular benchmark datasets show that incorporating prompt learning significantly improves (3\% - 15\%) the performance of state-of-the-art GNN models in crystal property prediction tasks. Furthermore, the learned soft prompts enable cross-property knowledge transfer, enhancing prediction performance for properties with limited training data. Code is available at this https URL 

---
# AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision 

**Authors**: Brent Kong, Tejas Ram, Tony Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08984)  

**Abstract**: AlphaZero has demonstrated that a neural-guided Monte Carlo Tree Search can achieve superhuman performance, but strong play does not necessarily imply perfect play. We study this gap in two oracle-evaluable domains with contrasting structure: Connect Four, a solved partisan game with exact game-theoretic values, and Chomp, an impartial game whose optimal play is governed by Grundy-number structure. Under a unified self-play $+$ MCTS pipeline, we compare vanilla AlphaZero, a multi-frame variant (limited to Chomp), and an AlphaZero Auxiliary Loss (AZAL) that adds oracle-derived policy supervision. We find that vanilla AlphaZero achieves strong play across both domains but cannot preserve the exact trajectories required for optimal play: in Connect Four, it fails to maintain the optimal line of play, while in Chomp, it fails to consistently restore the $g=0$ invariant. On rectangular Chomp boards, multi-frame inputs alone do not remove this gap. Nevertheless, AZAL substantially improves oracle consistency across multi-seeded full-game traces and sampled-state evaluations. On Chomp, AZAL reaches perfect full-game oracle consistency on 10x11 and high but not complete consistency on 9x10; on Connect Four, AZAL improves oracle-match rate and delays the first oracle mistake, but does not reach perfect play. 

---
# SCATE: Learning to Supervise Coding Agents for Cost-Effective Test Generation 

**Authors**: Sijia Gu, Noor Nashid, Ali Mesbah  

**Link**: [PDF](https://arxiv.org/pdf/2607.08983)  

**Abstract**: While autonomous coding agents have significantly advanced automated test generation, they remain fundamentally limited by lazy generation, a phenomenon where agents prematurely terminate tasks and systematically avoid complex programmatic logic, resulting in inadequate code coverage. Currently, mitigating this premature termination requires continuous human-in-the-loop supervision. This heavy reliance on human intuition creates a bottleneck that negates the efficiency gains of automated generation. We propose SCATE, a framework for adaptive, automated supervision of coding agents that replaces human intervention during test generation. By formulating supervision as a contextual bandit problem, SCATE learns to select the most promising testing actions based on the current coverage and class testability metrics, maximizing coverage gains while minimizing wasted generation effort. Our empirical evaluation demonstrates that SCATE integrates seamlessly with different coding agents. When applied to GEMINI-CLI, it achieves 32.3% higher line coverage and 30.9% higher branch coverage than the agent-only baseline. A comparison with CLAUDE CODE confirms the framework dynamically adapts its policy to optimize each agent's unique strengths. SCATE also consistently outperforms state-of-the-art non-agentic approaches across all metrics. 

---
# The Patchwork Problem in LLM-Generated Code 

**Authors**: Viraaji Mothukuri, Reza M. Parizi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08981)  

**Abstract**: LLM-generated code often compiles, passes tests, and appears correct, yet breaks once deployed. The root cause is frequently structural rather than logical. A generated endpoint references configuration keys never declared in the project, an import targets a package that does not exist in any registry, or a new route omits the authentication guard applied to every sibling endpoint. Each patch is locally valid but globally incoherent, and standard CI toolchains rarely surface these failures. As LLM-powered coding tools see widespread adoption, this blind spot poses a growing risk to software quality. We call this the \textbf{patchwork problem}. This paper formalizes structural coherence as consistency invariants over graph representations of repository artifacts, including import, call, dependency, configuration, schema, resource, control-flow, and routing graphs, and introduces an eight-category failure taxonomy distinguishing defects specific to LLM generation from those merely amplified by it. We present a hybrid verification framework that delegates to mature static analysis tools where they already excel and deploys purpose-built detectors for cross-cutting invariants underserved by existing toolchains, targeting provable constraint violations rather than heuristic pattern matching. Empirical evaluation across two frontier models under four prompting strategies reveals that the vast majority of structural failures evade type checking, testing, and SAST entirely, and that failure patterns diverge qualitatively between models in ways that challenge model-agnostic mitigation strategies. External validation on real-world AI-generated repositories confirms that these failures are not artifacts of controlled experimentation but are prevalent wherever LLMs write code with minimal human oversight. 

---
# CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding 

**Authors**: Yuri Ishitoya, Jeremy Siburian, Masashi Hamaya, Kuniaki Saito, Cristian C. Beltran-Hernandez, Mai Nishimura  

**Link**: [PDF](https://arxiv.org/pdf/2607.08974)  

**Abstract**: Vision-language-action models (VLAs) inherit semantic capabilities from pretrained VLMs, yet large-scale post-training on robot data and architectural modifications can reshape the backbone so extensively that it becomes difficult to isolate what the VLM contributes to control. Directly converting pretrained VLMs into VLAs with minimal architectural change offers a more transparent path to understanding how VLM capabilities transfer across model scales. The core obstacle is output-distribution mismatch: predicting actions as bare numeric token sequences moves generation away from the VLM's pretrained language distribution, degrading the capabilities we seek to preserve. To address this, we propose CLAP (Causal Language-Action Prediction), which prepends each numeric action sequence with a natural-language action description, causally conditioning precise action-token prediction on a language-action plan without modifying the backbone architecture. With single-epoch fine-tuning alone, 2B CLAP achieves 90.8% on LIBERO (+14.9 pt over VLA-0) and improves robustness on LIBERO-PRO under language, object, and spatial perturbations. We will release CLAP at 0.8B, 2B, and 4B as an open-weight, multi-scale compact VLA family from a single VLM lineage, enabling controlled analysis of VLM-to-VLA capability transfer. 

---
# MultiView-Bench: A Diagnostic Benchmark for World-Centric Multi-View Integration in VLMs 

**Authors**: Hantao Zhang, Jinru Sui, Ed Li, Dirk Bergemann, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08970)  

**Abstract**: Recent benchmarks for VLMs largely assess single- or limited-view perception, leaving untested the core cognitive ability to integrate observations across viewpoints into a coherent, world-centric (allocentric) 3D mental model. We introduce MultiView-Bench, a diagnostic benchmark expressly designed to evaluate multi-view integration for holistic 3D scene comprehension. Unlike existing datasets that focus on pixel-level mapping or camera-relative navigation, MultiView-Bench requires models to decouple object positioning from transient perspectives and ground them in a fixed global coordinate system. This capability serves as a prerequisite for VLMs before being deployed for downstream tasks such as mechanical part assembly. Our systematic evaluation of frontier VLMs reveals consistent failure modes: strong performance on 2D planar relations from a single image, but marked difficulty with 3D spatial relations and with aggregating information across views. We further identify biases in VLMs, such as struggles with unconventional axis directions and sensitivity to object colorways and texture variations. Acknowledging these limitations, we propose ViewNavigator, a multi-agent framework that actively selects informative viewpoints, perceives, and fuses multi-view evidence, improving diverse base models on MultiView-Bench even under a strict budget-matched comparison (and by 3-5x for the full agent). 

---
# NL-PAC: Specification Ambiguity and Certified Minimax Risk Floors in LLM-Mediated Supervision 

**Authors**: Berkay Anahtarci  

**Link**: [PDF](https://arxiv.org/pdf/2607.08961)  

**Abstract**: Large language models increasingly provide labels, evaluations, and feedback for tasks specified in natural language. When a specification admits multiple readings but the supervision channel does not reveal which is operative, additional labels reduce sampling error without resolving the resulting identification problem. We introduce Natural Language PAC (NL-PAC), a framework that uses a fixed model's thresholded decoding law to define admissible labels and candidate targets. The probability that multiple labels are admissible equals the diameter of the pointwise-admissible target class, and under target-blind supervision every learner incurs worst-case risk of at least half this diameter, at every sample size; the exact randomized minimax risk over this class is attained by a data-independent strategy. Finite-sample confidence bounds make these quantities certifiable from held-out unlabeled inputs. In a frozen Qwen~2.5--3B audit, one prespecified prompt yields a positive model-relative certificate, whereas a paraphrase and exact-rule controls yield zero. A held-out bridge audit finds that supplied candidate reading clauses fail the admissibility condition needed to transfer the certificate to coherent readings. The guarantee is specific to the audited model, prompt, threshold, and input distribution; extending it to human interpretations requires external validation. 

---
# Eluna: An Agentic LLM System for Automating Warehouse Operations with Reasoning and Task Execution 

**Authors**: Ning Liu, Kalle Kujanpää, Zhaoxuan Zhu, P Aditya Sreekar, Kaiwen Liu, Chuanneng Sun, Jorge Marchena Menendez, Matthew Bales, Tianyu Yang, Shahnawaz Alam, Rose Yu, Baoyuan Liu, Kristina Klinkner, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2607.08960)  

**Abstract**: Warehouse operations are governed by Standard Operating Procedures (SOPs) that encode complex, multi-system decision logic, which must be executed reliably under strict time constraints, yet LLM agents lack mechanisms to enforce procedural compliance and degrade under the context overload full SOP specifications introduce. We present Eluna, a production-deployed agentic system for reliable SOP execution. Eluna is a graph-guided, multi-agent framework that encodes SOPs as directed acyclic graphs with progressive disclosure and delegates independent tasks to parallel sub-agents, each with persistent code execution and live data access. To meet production latency and accuracy needs, we use asymmetric episodic distillation where a strong teacher is improved through episodic error memories, then a smaller student is fine-tuned on the corrected trajectories with memory stripped, internalizing corrections without inference-time overhead. On a 13-task benchmark and two production applications, our fine-tuned models match or exceed their teacher, beat all larger off-the-shelf baselines, and reach 94% expert agreement on the ticket processing application. 

---
# A Novel Parallel QCNN Architecture with Efficient Classical Simulability 

**Authors**: Lawrence Nguyen, Hiu Yung Wong  

**Link**: [PDF](https://arxiv.org/pdf/2607.08928)  

**Abstract**: This work presents a study of an implementation of a novel Quantum Convolutional Neural Network (QCNN) for binary classification of images from the Modified National Institute of Standards and Technology (MNIST) dataset. Using a novel architecture inspired by previous QCNN and classical convolutional neural network (CNN) implementations, we use a hierarchical partitioning approach to implement a QCNN circuit that can be approximated and simulated efficiently on a classical machine for a large problem. First, the original image is partitioned such that each process handles a smaller portion of the image, which is encoded into independent states. Then, these partitions merge and combine, resulting in states that contain information from both partitions while halving the number of processes. After repeating this until one process remains, we reduce the dimensionality of the state until a single qubit remains for measurement. Using this approach, we can use multiple processes in parallel to simulate a large QCNN program without the need for exponentially growing hardware requirements as the number of qubits increases. In our work, we use this scheme to train a 128-qubit model, which is impossible to run on any classical supercomputer without the novel architecture. We also explore the impact of this new model architecture on prediction accuracy by training it to perform binary classification on the MNIST dataset with a small number of qubits, and comparing it to a model without partitioning. Our initial findings show that partitioning images into smaller sub-images with this architecture does not degrade the model's performance and sometimes even improves it, likely because it reduces the Barren plateaus issue in the partitioning process. 

---
# Prompt-Driven Exploration 

**Authors**: Sunshine Jiang, John Marangola, David Zhang, Raghuram Kowdeed, Ruiyang Luo, Nitish Dashora, Richard Li, Pulkit Agrawal, Zhang-Wei Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.08837)  

**Abstract**: Exploration is essential to RL since a policy cannot improve by repeatedly sampling the behaviors it already prefers. Standard methods inject stochasticity in the action space, but such jitter only yields rollouts close to the original. Escaping a weak policy often requires global perturbations that action noise cannot produce. Large language models (LLMs) and vision-language-action (VLA) models offer a pathway: they condition the policy on a natural language prompt, and since the rollout follows from it, modifying the prompt induces global changes. The challenge is finding prompts that induce useful global changes. With a weak policy that rarely succeeds, reward is too sparse to select on. Our idea is to refine prompts from the rollouts themselves: a vision-language model (VLM) reasons over the rollout video, diagnoses how the policy responded, and rewrites the prompt to elicit better behavior next time. This procedure realizes posterior sampling, a classical RL exploration framework, at the level of prompts: the VLM maintains an implicit distribution over useful prompts and updates it from observed rollouts. We call this strategy Prompt-Driven Exploration (PDE). Across manipulation and reasoning tasks, PDE enables RL to learn successful policies even from zero-reward starts, and improves sample efficiency more broadly. Our website is available at this https URL. 

---
# TheBioCollection: Unified Pre-Training Scale LLM Corpus for Biology 

**Authors**: Hyunjin Seo, Hyeon Hwang, Gyubok Lee, Jay Shin, Jimin Park, Taesoo Kim, Sanghoon Lee, Hongjoon Ahn, Sungjun Han, Sangwon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2607.08803)  

**Abstract**: The push toward large language models for biology (BioLM) has created a need for training corpora that can endow models with a genuine understanding of biology. However, existing biological resources, such as molecular databases, protein repositories, genomic annotations, single-cell atlases, and pathway databases, are scattered across heterogeneous formats and remain unorganized into a cohesive corpus for language model training. We present TheBioCollection, a 52.6B-token pre-training-scale corpus that converts these disparate resources into a unified, training-ready form spanning small molecules, proteins, genomic sequences, cells, and pathways. Beyond consolidating existing data, TheBioCollection enriches each record with tool-computed biological properties and introduces new instruction tasks for capabilities that current corpora barely cover. We pair the corpus with TheBioCollection-Eval, a matched suite probing recognition, generation, and prediction across molecular, protein, genomic, cellular, and cross-domain settings. Holding the base Gravity-16B-A3B architecture fixed, training on TheBioCollection more than doubles its overall score on TheBioCollection-Eval with gains in every domain, while leaving general linguistic ability nearly intact. 

---
# Multi-Conditioned Diffusion Synthesis of Sand Boils for Low-Resource Earthen-Levee Inspection 

**Authors**: Padam Jung Thapa, Abdullah Bin Naeem, Ayon Dey, Anav Katwal, Md Tamjidul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2607.08794)  

**Abstract**: Sand boils on earthen levees are safety-critical defects, but pixel-level detection is limited by scarce annotations. We present a diffusion-based synthesis pipeline for low-resource sand-boil imagery. Using Stable Diffusion XL fine-tuned with DreamBooth and conditioned by a multi-branch ControlNet stack, the pipeline generates synthetic inspection images from a small curated reference set. A soft-mask inpainting protocol preserves the real defect pixels while re-rendering the surrounding scene, avoiding seams and color shifts from prior seamless-cloning compositing. A mask-conditioned ControlNet can also generate a new boil inside a chosen mask, making the mask the segmentation label by construction; however, because large-scale label certification remains unresolved with the available real-trained gate, we release the soft-mask preset as the default. Text conditioning is supplied by a taxonomy-driven Prompt Atlas that expands one domain specification into a stratified, CLIP-validated prompt bank and transfers to new defect classes without code changes. From the real training images, the pipeline produces 1,020 synthetic candidates, of which 815 pass a CLIP admissibility filter. We evaluate image quality using distributional and fidelity-diversity measures against the real reference set and a Poisson baseline, and audit for out-of-distribution drift and memorization. No single preset dominates; each trades off fidelity, diversity, and label reliability. We therefore release the label-reliable preset as the default and treat a curated mixture as the natural augmentation set. Our claims are limited to image quality, label provenance, and diversity; downstream segmentation is left for future work. Code and an artifact manifest are released for reproducibility. 

---
# EHR-MPC: Inference-Time Control for Sepsis Treatment with Generative Patient Digital Twins 

**Authors**: Joshua Pickard, Wei Qi, Na Li, Ann Woolley, Lisa Cosimi, Roy Kishony, Deborah Hung  

**Link**: [PDF](https://arxiv.org/pdf/2607.08793)  

**Abstract**: Sepsis is a leading cause of mortality, yet optimal treatment policies remain contested. Existing reinforcement learning (RL) approaches learn fixed strategies for sepsis treatment, limiting adaptability to changing clinical objectives during inference. We propose EHRMPC, a framework that decouples learning patient dynamics from optimizing treatment by training a patient digital twin in the form of a generative electronic health record (EHR) model. The digital twin predicts clinical trajectories under interventions and enables model predictive control (MPC) to optimize treatments via inference-time planning over simulations. We evaluate EHR-MPC on a multicenter ICU sepsis cohort spanning 8 hospitals in the Mass General Brigham health system using both off-policy importance sampling and on-policy simulation-based evaluation. Relative to RL baselines, EHR-MPC achieves comparable off-policy performance and improved simulation performance. Unlike RL, this work frames sepsis treatment optimization as inference-time control over learned patient dynamics, establishing a general framework for decision making with generative clinical models. 

---
# LLM-Driven Evolutionary Generation of Multi-Objective Bayesian Optimization Algorithms 

**Authors**: Georgios Laskaris, Reuben Brasher, Niki van Stein, Elena Raponi, Thomas Bäck, Florian Neukart  

**Link**: [PDF](https://arxiv.org/pdf/2607.08791)  

**Abstract**: Designing effective multi-objective Bayesian optimization (MOBO) algorithms requires balancing many interdependent design choices whose optimal configuration is problem-dependent and typically demands deep expertise. We extend the LLaMEA framework to MOBO, using large language models as mutation and crossover operators within evolutionary strategies to generate complete algorithm implementations, with SMAC hyperparameter optimization integrated into the evolutionary loop. Across nine evolutionary runs we generated approximately 900 algorithms and benchmarked them on twelve synthetic problems (ZDT, DTLZ, WFG) and three real-world engineering problems (RE), using a BoFire qParEGO implementation as a state-of-the-art Bayesian-optimization baseline. On the synthetic suite the strongest generated algorithm attains the highest mean normalized hypervolume (0.971, vs. 0.869 for qParEGO) while requiring roughly 60x less wall-clock time; a Friedman test with post-hoc analysis places the two in a single top-performing group, and per-problem tests find the generated algorithm significantly better than qParEGO on 7 of the 12 problems and never worse, matching state-of-the-art accuracy at an order-of-magnitude lower cost. On the three unseen real-world engineering problems a generated algorithm attains the best mean normalized hypervolume (0.985, vs. 0.971 for qParEGO)--significantly better than qParEGO on two of the three problems--at roughly 3.4x lower wall-clock cost, confirming that the gains transfer beyond the synthetic regime. LLM-driven evolutionary search can thus discover algorithm designs that achieve Pareto-efficient trade-offs difficult to reach through manual design. 

---
# Accelerating GPU Inference of Large Language Models with Moderately Unstructured Sparse Weight Matrices 

**Authors**: Tao Lu, Haoyu Wang, Zonghui Wang, Keshen Xiang, Jiaheng Zhang, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.08786)  

**Abstract**: With the growing deployment of large language models (LLMs), LLM inference cost has become a key challenge. Pruning techniques that introduce sparsity into weight matrices can accelerate inference. However, maintaining model quality typically limits pruning to moderate unstructured sparsity (around 50\%). At these sparsity levels, none of the existing GPU kernels for sparse matrix multiplication (SpMM) can outperform their dense counterparts. This paper proposes an efficient GPU inference method for LLMs with moderate sparsity. We propose a three-layer matrix storage format comprising: (i) a Sparse-TC layer enabling sparse tensor cores to accelerate SpMM; (ii) a Slot-Filling layer using parallel differential distance for matrix compression while supporting low-cost on-chip decoding; (iii) a lightweight Residual Layer ensuring correct SpMM computation. Building on this format, we design a SpMM kernel that jointly utilizes sparse tensor cores and CUDA cores. This design enables an efficient execution pipeline and overlaps on-chip computation with memory access. Evaluations show that our work is the first to outperform dense matrix multiplication on modern GPUs equipped with high-bandwidth memory (HBM). It achieves up to 1.64x kernel-level speedup over SpInfer (EuroSys'25, Best paper) and up to 1.41x end-to-end speedups over FlashLLM (VLDB'24). Our source code: this https URL. 

---
# DaDaDa: A Dataset for Data Pricing in Data Marketplaces 

**Authors**: Qiheng Sun, Hongwei Zhang, Junxu Liu, Xiaokai Mao, Jinfei Liu, Kui Ren, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2607.08785)  

**Abstract**: High-quality data drives machine learning advances across industries. Recognizing the value of data, data transactions are increasingly common, giving rise to many data marketplaces, e.g., AWS Marketplace, Databricks, and Datarade. However, determining the appropriate prices for data products remains a significant challenge due to the unique properties of data products. Traditional pricing methods in economics can be categorized into the cost approach, the income approach, and the sales comparison approach. The cost approach fails in data pricing due to near-zero marginal cost from data replication, and the income approach fails due to inherently unpredictable data revenue. The sales comparison approach remains viable, yet its application is hindered by the absence of standardized pricing benchmarks for data products across marketplaces. To address this challenge, we introduce \texttt{DaDaDa}, the first dataset for data product pricing, containing metadata for 16,147 data products from 9 major data marketplaces worldwide. \texttt{DaDaDa} enables the training of pricing models, thereby establishing price benchmarks for new data products. In addition, \texttt{DaDaDa} can be utilized for other important tasks in data markets, such as data product classification and retrieval. Experiments and a retrieval prototype demonstrate the effectiveness of \texttt{DaDaDa} for pricing, classification, and retrieval of data products. The dataset and code are available at this https URL. 

---
# HERO: A Heterogeneity-Aware Benchmark Library for Federated Continual Learning 

**Authors**: Thinh T. H. Nguyen, Le-Tuan Nguyen, Minh-Duong Nguyen, Nhi Trinh, Anh Tran Nam Nguyet, Dung D. Le, Kok-Seng Wong  

**Link**: [PDF](https://arxiv.org/pdf/2607.08784)  

**Abstract**: Federated continual learning (FCL) evaluates how distributed clients learn from changing data streams while retaining previously learned knowledge. Existing evaluations are difficult to compare because they often change datasets, task splits, client data splits, task orders, backbones, memory assumptions, and reporting rules simultaneously. We introduce \textbf{HERO}, a heterogeneity-aware benchmark library for FCL. HERO builds benchmark streams by separating three choices that are often coupled, namely the task split, the client data split, and the client task sequence. In HERO-Core, the main comparable benchmark, $\alpha$ controls client data skew and $\rho$ controls task-order mismatch. We evaluate representative FCL methods on CIFAR-100 and TinyImageNet using final average accuracy, average forgetting, and bottom-10\% client accuracy. We also include a graph-based Domain-IL portability case study on OGB-MolPCBA, where scaffold-domain granularity changes the input distribution while the prediction task remains fixed. Our results show that method behavior changes across easy and heterogeneous settings, that average accuracy can hide weak bottom-client performance, that task-order mismatch favors different strategies from synchronized evaluation, and that the same HERO interface can expose domain-shift difficulty beyond image-based FCIL. HERO releases benchmark streams, configurations, method implementations, and reporting scripts to support reproducible and setting-aware FCL evaluation. 

---
# LieBN: Batch Normalization over Lie Groups 

**Authors**: Ziheng Chen, Yue Song, Rui Wang, Xiao-Jun Wu, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2607.08783)  

**Abstract**: Manifold-valued measurements are prevalent in various machine learning tasks. Recent advances have extended Deep Neural Networks (DNNs) to operate on manifolds, accompanied by normalization techniques tailored to different geometries, collectively referred to as Riemannian normalization. However, most existing Riemannian normalization methods are either designed for specific manifolds or fail to effectively normalize manifold-valued sample distributions. To address these limitations, we propose LieBN, a framework for Riemannian Batch Normalization (RBN) over Lie groups. Our approach leverages the theoretically convenient left- and right-invariant metrics, which naturally exist in every Lie group, and provides theoretical guarantees for controlling the Riemannian mean and variance. We instantiate LieBN across nine distinct geometries: four on the Symmetric Positive Definite (SPD) manifold, one on the group of rotation matrices, and four on the manifold of full-rank correlation matrices. Notably, among the SPD metrics, we introduce a novel right-invariant metric and extend three existing Lie group structures via matrix power deformation. Extensive experiments on different manifolds validate the effectiveness of our framework. The code is available at this https URL. 

---
# Director: Accelerating Distributed MoE Serving via Online Proactive Expert Placement 

**Authors**: Qianli Liu, Kaibin Guo, Zicong Hong, Peng Li, Fahao Chen, Haodong Wang, Jian Lin, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.08782)  

**Abstract**: Expert parallelism has become the prevailing paradigm to serve Mixture-of-Experts (MoE) models. Its efficiency depends on the communication and computation latencies of the GPUs, which are linked to the placement of experts in the GPUs. Existing works for optimizing expert placement focus on leveraging past requests' expert activation patterns. However, they demonstrate deficiencies facing diverse and rapidly changing request patterns, calling for an online, proactive approach. Implementing such an approach requires addressing several challenges: the uncertainty associated with incoming requests' expert activation, the cost of expert migration, and the NP-hard complexity in optimization. Therefore, we present Director, a new distributed MoE serving system that minimizes end-to-end latency via prediction-driven, online expert placement. Director uses either a lightweight cascaded predictor or a low-bit quantized replica for expert activation patterns of incoming requests. An online migration module then enacts the changes with near-zero downtime by executing migrations in compute-bound phases, keeping disruption bounded. At its core, a relaxation-based expert placement optimizer operates under capacity constraints, runs in polynomial time, and achieves a $(1+\epsilon)$ approximation ratio. Finally, we implement a prototype and demonstrate, through extensive experiments, a reduction in end-to-end latency of $11\sim55\%$ for popular MoE models (e.g., Mistral, DeepSeek and Qwen) compared to existing work. 

---
# Reward Transport: Property Control in Flow Matching via Noise-Space Alignment 

**Authors**: Kehan Guo, Yili Shen, Yujun Zhou, Yue Huang, Chujie Gao, Shiyi Du, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.08781)  

**Abstract**: The coupling in flow matching -- the rule pairing noise vectors with data points -- is typically treated as a computational choice. We show that this coupling can instead serve as an alignment interface: by matching noise and data according to a target molecular property, it embeds controllable structure directly into the learned flow field. Building on this view, we introduce Reward Transport, which uses optimal transport coupling at training time to align a scalar noise-space coordinate with molecular rewards; at inference, varying this coordinate steers the generated distribution without requiring an oracle, reward model, gradient guidance, or additional computation. In the coupling-preserving limit, thresholding this coordinate recovers the Cross-Entropy Method's truncated reward distribution, providing a principled, continuously adjustable distribution-level control knob. Empirically, on ZINC-250K and GuacaMol, sweeping the scalar induces monotone control of logP and consistent QED control over its operating range; most tellingly, the same knob produces opposite structural responses for different targets, growing molecules for logP but shrinking them for QED, which rules out a generic size bias. The interface is complementary to classifier-free guidance and conditional flow matching, while a negative result under epsilon-prediction diffusion clarifies where coupling-level alignment is structurally absent. Code: this https URL 

---
# Sticky Routing: Training MoE Models for Memory-Efficient Inference 

**Authors**: Ali Kayyam  

**Link**: [PDF](https://arxiv.org/pdf/2607.08780)  

**Abstract**: Mixture-of-Experts (MoE) models activate only a sparse subset of experts per token, yet consecutive tokens frequently activate different experts -- causing constant weight swapping between slow storage and fast memory on edge devices. Existing remedies are either system-level (caching heuristics) or post-hoc (router fine-tuning), leaving the root cause unchanged during pretraining. We propose StickyMoE, a differentiable routing consistency loss that penalises abrupt expert switches between adjacent tokens, encouraging the router to maintain the same expert assignment across semantically coherent spans. StickyMoE requires no architectural changes, adds a single hyperparameter lambda, and unlike post-hoc methods, allows expert representations and routing decisions to co-adapt from the first training step. Experiments on small-scale MoE language models show that StickyMoE reduces the expert switch rate by up to 60% with less than 4% perplexity degradation, Pareto-dominating post-hoc fine-tuning on the quality-locality frontier. Routing temporal locality is most efficiently instilled at training time. 

---
# Signed Symmetric Quantization for Few-Bit Integers 

**Authors**: Ian Colbert, Eashan Dash, Pablo Monteagudo-Lago, Juan Amboage, Srinidhi N, Giuseppe Franco, Nicholas J. Fraser, Arun Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2607.08779)  

**Abstract**: The signed integer alphabet contains one more negative representable value than positive. Yet, by convention, the standard symmetric integer quantizer fixes its scale to be strictly positive, which assigns this extra representable value to the negative tail and can force clipping of positive outliers. In this work, we show that, at few-bit precision, such clipping is a non-trivial source of quantization error. Asymmetric quantization addresses this problem with a zero point, shifting the grid toward the observed data range; however, this flexibility is well-known to carry a runtime penalty. For example, in this http URL on an AMD EPYC(TM) "Turin" CPU, a 4-bit symmetric format uses up to 9% less memory with up to 2.45$\times$ higher throughput than its asymmetric counterpart. We highlight signed symmetric quantization as a third option that retains the runtime profile of symmetric quantization without the penalty of the asymmetric format: our signed absmax grid places the extra representable value on the dominant-outlier tail through a principled and lightweight sign selection rule while keeping the zero point at zero. Our theoretical analysis offers two main results. First, we establish the signed absmax grid as conditionally bound-optimal on $\ell_2$ quantization error, and show that the condition holds for 88-99% of weight groups across pre-trained large language models (LLMs) at low bit widths. Second, we show that negating the scale of a standard symmetric quantizer is analytically equivalent to a unit zero point shift on the same signed integer alphabet. We empirically validate our proposal on models from the Qwen3, Qwen3.5, and Llama3 families, and observe improvement in perplexity and downstream few-shot accuracy over the standard unsigned symmetric quantizer at no extra inference cost 

---
# iLENS: Interpretable LLM-Guided Mixture-of-Experts for Neuroimaging Survival Analysis 

**Authors**: Farica Zhuang, Seong Woo Han, Zixuan Wen, Shu Yang, Yize Zhao, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2607.08778)  

**Abstract**: Alzheimer's Disease (AD) is a complex neurodegenerative disorder that continues to impact millions of people worldwide. Predicting AD conversion during the prodromal stage remains critical for disease understanding and patient care. As such, survival models are widely used for AD risk prediction, yet they are typically static predictors with limited interpretability and no capacity for natural language reasoning. In this work, we propose iLENS, an interpretable large language model (LLM) guided framework based on mixture-of-experts (MoE) for survival prediction in AD conversion. Our approach uses LLM to synthesize structured neuroimaging measurements and unstructured information to guide expert routing. Our framework demonstrates competitive predictive performance and capability in patient subtyping. Furthermore, our framework provides transparent, biologically grounded rationales for its routing decisions, bridging the gap between high-performance survival analysis and interpretable clinical decision support. 

---
# A Unified Approach to Interpreting Knowledge Distillation for Large Language Models via Interactions 

**Authors**: Qingzhuo Wang, Ruiyang Qin, Zhenxin Qin, Wen Shen, Zhihua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2607.08776)  

**Abstract**: Despite the success of knowledge distillation (KD) in Large Language Models (LLMs), the underlying mechanism behind its efficacy remains unclear. In this paper, we propose a unified approach to explore the common mechanism of various KD methods using interactions. Specifically, we decompose the output score of the LLM into the sum of numerous interactions. Each interaction represents a nonlinear relationship involving a set of input variables (e.g., words). Based on the decomposed interactions, we discover that the common mechanism underlying various KD methods is the sparsification of interactions, i.e., student models retain fewer interactions for inference while suppressing other interactions to zero effects. Furthermore, we discover that the performance variance across different KD methods arises from their capabilities in handling complex interactions. A KD method typically yields better performance if it enables the student model to achieve higher sparsity of complex interactions. Motivated by these insights, we propose a plug-and-play loss function called Complex Interaction Penalty (CIP) to explicitly enforce the sparsity of complex interactions during the distillation process. Extensive experiments demonstrate that integrating CIP consistently improves the performance of diverse KD methods on both in-domain and out-of-distribution benchmarks. 

---
# REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming 

**Authors**: Nicolas Koller, Andreas u. Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2607.07738)  

**Abstract**: Large language models (LLMs) are increasingly applied to reverse-engineering tasks, and recent threat-intelligence reporting shows them operating inside live offensive-security workflows. Claims about their capability, however, outpace our ability to measure it. Existing benchmarks for LLM-assisted binary analysis treat the construction of function-level ground truth as a solved pre-processing step and report accuracy without disclosing how many functions were reliably evaluable. We argue that the principal obstacle to fair evaluation is not model capability but the reliability of binary-to-source alignment under compiler optimization. This paper presents Reforge, a provenance-tracked pipeline that constructs function-level ground truth from C source through compilation, DWARF and syntactic extraction, alignment, and decompilation, and that operationalizes alignment uncertainty as an eight-gate confidence funnel with three-tier stratification. On a controlled micro-benchmark, high-confidence yield falls from 87.2% to 65.9% across optimization levels, and unpaired comparisons overstate optimization-induced performance decay through survivorship bias. A proof-of-concept evaluation of seven contemporary LLMs on function naming demonstrates the substrate and motivates uncertainty-aware benchmarking practice 

---
# Minimal Decision Dynamics and Contextual Probability: A Quantum Tug-of-War Model 

**Authors**: Song-Ju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.10034)  

**Abstract**: Decision making often exhibits context dependence that challenges classical probability theory. This paper develops a quantum-like extension of the Tug-of-War (QTOW) decision-making model to clarify when such context dependence can be represented by a single minimal internal state. The QTOW construction uses a qutrit internal state, conservation-preserving updates, and measurement-induced disturbance to model decision, learning, and probing operations within one coherent state space. Within this minimal representation, KCBS-type probing contexts can be constructed, yielding a witness of non-contextual classical non-embeddability. The main claim is not that quantum theory is uniquely or assumption-freely derived from decision making. Rather, a classical reconstruction of the same operation family requires additional contextual memory, history dependence, or an enlarged hidden-state representation. Thus, contextual probability appears as a resource signature of minimal decision dynamics, while quantum probability provides a compact, memory-efficient realization of this structure. 

---
