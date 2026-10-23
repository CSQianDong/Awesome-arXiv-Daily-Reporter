# Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning 

**Authors**: Mahdi Salmani, Adel Javanmard, Devansh Gupta, Meisam Razaviyayn  

**Link**: [PDF](https://arxiv.org/pdf/2610.31667)  

**Abstract**: Test-time scaling has emerged as a key inference strategy for reasoning models, where a model generates a large number of candidate solutions in the hope of finding at least one correct solution. Running all candidate solutions to completion can be computationally demanding, so recent work prunes the candidate pool after generating only a short prefix, committing further computation only to the surviving candidates. Existing pruning strategies divide broadly into two categories: Quality-based methods assign a score to each partial solution and discard those predicted to ultimately lead to an incorrect final solution, while diversity-based methods select partial solutions that follow different reasoning trajectories and are therefore likely to cover different final solutions. The tension between these two objectives is an instance of exploration versus exploitation, yet little to no prior work on pruning exposes this trade-off directly, let alone optimizes it. In this paper, we introduce a pruning procedure that addresses this trade-off directly by optimizing pass@$k$ over the selected solution pool. In particular, it chooses the pool to maximize the probability that at least one of its partial solutions, when continued to completion, yields a correct answer. Because this objective simultaneously rewards high-quality partial solutions and penalizes redundancy among them, quality and diversity are naturally balanced by the criterion itself rather than through separate ad-hoc design choices. On eight open-weight reasoning models, our method outperforms existing quality-based and diversity-based pruning strategies. The gains are particularly pronounced when the pruning procedure keeps a moderate number of partial solutions for continued generation, highlighting the value of directly balancing quality and diversity during pruning. 

---
# EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence 

**Authors**: Qijin Zhang, Hanzhen Wang, Yuehan Zhao, Haoyu Dai, Yao Shu, Zhongxiang Dai, Shuang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31650)  

**Abstract**: Tool-using language model agents rely on environmental facts to decide whether an action is permitted. When a tool response omits a safety-relevant fact, an agent may carry out a prohibited action even though the user request is benign and every returned statement is true. We introduce \bench{}, an executable benchmark for evaluating unsafe execution, safety boundaries, and evidence recovery in LLM agents under missing evidence. It contains 300 matched unsafe/safe task pairs across six domains. By systematically suppressing critical facts, the benchmark measures unsafe execution under critical evidence omission and determines the minimum observed failure cardinality, including failures with full evidence and cases with no observed failure after exhaustive testing. An independent audit tool tests whether agents seek missing facts and then withhold prohibited actions while completing permissible tasks. The benchmark reveals substantial variation across the ten models in omission robustness, evidence acquisition, and appropriate action. Resistance to missing evidence does not necessarily translate into useful recovery. Recovery access reduces unsafe execution for all ten models, but agents differ in whether they actively obtain the missing facts and use them to act appropriately in both worlds. 

---
# Simulated Control Evals 

**Authors**: Jeffrey Olmo, David Zhang, Tyler Tracy  

**Link**: [PDF](https://arxiv.org/pdf/2610.31593)  

**Abstract**: Control evaluations measure whether safeguards can prevent an AI model from causing undesired outcomes even when it is intentionally trying to do so. These evaluations typically use purpose-built control settings and extrapolate their results to a model's deployment. However, the effectiveness of safeguards in deployment can depend on features of environments and tasks that control settings do not represent. To broaden where control evaluations can be run, we introduce simulated control evaluations (SCEs), which use a language model to simulate tool responses in place of an environment. We evaluate SCEs on LinuxArena, where execution provides ground-truth outcomes. We compare SCE prediction errors with those from a separate control setting, BashArena, and with variation in control metrics within LinuxArena. For independently sampled evaluation and deployment task sets, SCEs predict safety with a root mean squared error of 3.6 percentage points across five monitors at a 5% audit budget, compared with 10.1 for BashArena. SCEs closely reproduce distributions of attack strategies, attacker reasoning, and monitor reasoning, with over tenfold lower KL divergence from execution than BashArena. SCEs explain 78-85% of the between-environment variance in attack success, monitor scores, and safety. This accuracy is sample efficient: SCEs predict safety with less error than BashArena using fewer than 5% as many task-pair draws as the deployment task set. Strategies discovered through simulation also improve monitor performance on held-out LinuxArena attacks, outperforming BashArena and approximately matching improvement from executed strategies. These results suggest that deployment-specific SCEs can be more predictive than control evaluations executed in separate control settings. 

---
# Who Is Sycophantic to Whom? Confident LLMs, Human Deference, and the Collapse of Trust Calibration 

**Authors**: Deepthika S, Pawni Yadav, B. Sankar  

**Link**: [PDF](https://arxiv.org/pdf/2610.31579)  

**Abstract**: Research on sycophancy asks whether large language models (LLMs) bend towards what their users want to hear. We ask the reverse question: when an LLM is confident, consistent, and wrong, do users bend towards it? In a within-subjects experiment, 32 participants learned up to five science topics by chatting with five assistants built on the same model (GPT-5-mini). Unknown to participants, each assistant's briefing had 0, 25, 50, 75, or 100% of its facts replaced by plausible falsehoods, and every assistant spoke with full confidence. A surprise quiz after each topic offered the correct fact, the planted falsehood, and two distractors, with a confidence rating for every answer. Across 104 quizzes and 1,040 answers, scores fell from 7.4 to 2.9 out of 10, and the planted option was chosen on 60% of falsified items against 11% of correctly taught ones (odds ratio 12.5). Transcript coding showed that whenever the assistant stated a falsehood, it was adopted about 65% of the time at every falsification level. Absorbed falsehoods were held with high confidence (8.4/10), the modeled confidence advantage of right over wrong answers vanished, and the discrimination of confidence for correctness fell from an AUROC of 0.70 to chance, while post-task trust ratings barely moved. Participants tended to return to the chat more when the assistant was worse, but returning went with more absorption, not less, and 30% of all turns pasted quiz questions back to the source being checked. The assistant itself was almost never sycophantic and never conceded under pushback. We introduce a baseline-corrected Human Deference Index to quantify this human-side sycophancy and argue that trustworthiness is a property of the human-AI pair, that sycophancy evaluations should measure both directions, and that learning interfaces must make reliability legible claim by claim. 

---
# What Makes a Good Unmasking Policy? A Controlled Study of Learned Unmasking in Diffusion Language Models 

**Authors**: Jinya Sakurai, Xun Xu, Jaehong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2610.31576)  

**Abstract**: Masked diffusion language models generate tokens in parallel, but their efficiency depends on which predictions are committed. Learned unmasking policies improve the speed--accuracy trade-off, yet differ in supervision, inputs, and capacity, obscuring which choices matter. We view supervised policies as distilling a decoding procedure through a teacher and examine independently what the teacher labels as committable, on which states the policy imitates it, what the policy observes, and how much the number of function evaluations (NFE) must decrease to offset policy overhead. We find that labeling predictions matching the final output is overly permissive at low NFE, and hidden states are weak alone but can complement confidence. About a 10% NFE reduction suffices to offset policy overhead for policies up to 7.5M parameters across tested settings. We therefore train a policy on trajectory-contraction targets, which accept only commitments staying on a reference trajectory, using confidence and hidden states. On LLaDA-8B-Instruct, the resulting 856K-parameter policy achieves the best accuracy--NFE frontier and highest throughput on GSM8K, is competitive at moderate targets on MATH-500 and HumanEval, and largely transfers to Dream-7B-Instruct. 

---
# Contestable Agents for Data Governance: Recording Human Stewardship of AI-Generated Metadata Across Structured and Unstructured Systems 

**Authors**: Humza Akhtar, Brian Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2610.31553)  

**Abstract**: Industrial organizations hold operational knowledge in structured records, such as telemetry, point lists, and transactions, and in unstructured sources, such as manuals, service notes, communications, and inspection reports. Governance programs often manage these sources separately. Classifying, linking, and documenting them depends on a small group of employees with deep domain knowledge. Agents built on large language models can propose metadata, cite documents, and flag quality issues, but unreviewed proposals can spread errors into downstream systems. A perfunctory approval leaves no record of what a reviewer checked. We present contestable governance: specialist agents produce scored, cited arguments for disputed metadata decisions; a steward can accept, reject, modify, or add arguments before commit; and a governance record retains the evidence, interventions, and outcome. A versioned domain glossary supplies model-specific terms and acronym definitions to the agents. We describe the workflow, a triage path that reserves full argumentation for uncertain proposals while preserving review before commit, the scoring and record models, and a gate that controls which verified decisions become precedent. Two HVAC use cases illustrate the pattern: point classification using Brick Schema class labels, supported by an open reference implementation, and an illustrative warranty failure-mode decision linking claim fields and telemetry to service notes and revisioned bulletins. 

---
# Beyond Causal Masks: Auditing and Localizing Hidden Future-Token Leakage in Adaptive Sequence Models 

**Authors**: Huiting Liu, Jinsong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31522)  

**Abstract**: Hidden future-token leakage can create false architectural gains: an autoregressive model may appear substantially better under teacher-forced evaluation while its free-running generation deteriorates. We investigate this failure in an adaptive residual-routing decoder whose sequence-level routing summary exposes prefix positions to future tokens despite a correct causal attention mask. An anomalous 24-layer checkpoint reduces validation loss from the baseline's 3.5958 to 1.4259, yet it fails a prefix-invariance audit and has substantially higher reference-model perplexity during generation. Layer-wise diagnostics and implementation inspection trace the violation to a full-sequence summary broadcast through the dynamic routing-query path. Two independently trained 12-layer causal repairs eliminate detectable leakage and return validation loss and reference perplexity to the control scale, removing the large apparent advantage. As diagnostic tools, we use the established prefix-invariance contract, maximum prefix-output discrepancy (LeakScore), complementary behavioral metrics, and layer-wise profiles. Supporting experiments cover six synthetic implementation pairs spanning five module families, clean pretrained controls, controlled fault injection, and cache consistency across numerical precisions. These results illustrate how hidden future-token access can produce misleading claims of architectural progress and why lower teacher-forced loss should be interpreted together with whole-model causal checks and free-running evaluation. 

---
# Jev-DLC: Decision-Level Continual Learning from Typed Feedback 

**Authors**: Yangyang Feng, Jianle Gan, Zhuoyan Feng, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2610.31521)  

**Abstract**: Individual preferences often become visible only after deployment, when a user accepts one decision or revokes it in favor of another. Jev-style typed interfaces make this feedback explicit by identifying the selected action and, under revocation, the decision it replaces. We study how these consequences persist across requests as Decision-Level Continual Learning from Typed Feedback (Jev-DLC). Jev-DLC treats each correction as both evidence about a recurring action identity and a directed relation between the preferred and rejected choices. It preserves these signals in complementary bounded channels above a frozen scorer and combines their scores to shape the next decision. The method retains no raw interaction history, adds no scorer call, and performs no runtime language-model backward pass. We also introduce TCPA-Bench, a chronological suite that credits a correction only through its effect on later decisions. On TCPA-Repeat, Jev-DLC reaches 44.83% user-macro accuracy overall, 2.43 points above Recent-4 history replay. After at least two earlier feedback events, it reaches 56.07%, a 7.54-point margin over the strongest reported comparison at this depth. These results show how explicit typed corrections can become compact causal state for continual personalization. 

---
# From Prompt Rules to Golden Data Model: Production-Scale Generation of Personalized, Compliant Pharmaceutical Claims 

**Authors**: Suyash Mishra, Qiang Li, Subhabrata Adak, Upender Phogat, Srikanth Patil, Satyanarayan Pati  

**Link**: [PDF](https://arxiv.org/pdf/2610.31519)  

**Abstract**: Large language models (LLMs) are increasingly used to generate domain-specific commercial & medical content (e.g., claims, statements, and promotional phrases) in regulated industries such as pharmaceuticals. However, ensuring its regulatory compliance, factual accuracy, and audience-specific personalization remains a major challenge. We report a production case study asking whether moving compliance and personalization signal from prompt-level rules into retrieval time structured context improves pharmaceutical claims generation, using Composer, a deployed internal tool built on a Golden Data Model (GDM) of 1,867 pharmaceutical claims spanning four products. The GDM integrates claim metadata, references, audience labels, material type, usage history, Match Text Variant (MTV) links, and a six dimension quality score (6C). Across six product-audience segments, the context pipeline improves intended indication (+29%), intended content type (+62%) and intended target audience (+67%), 6C quality score (+86%), and claim reuse (+34%). Production log analysis over the deployment period shows a 29.5% reduction in weekly API cost. We report this as descriptive deployment evidence. This work is a systems and data architecture contribution: it shows that in a regulated, high-stakes domain, restructuring claim data for retrieval-time filtering is a more scalable lever than expanding prompt-level rules, and documents the engineering path data cleaning, graph linking, and quality scoring. 

---
# DuoCook: An Overcooked-Inspired Benchmark for Visual Planning and Coordination 

**Authors**: Ziyi Jiao, Heng Zhang, Drishti Regmi, Xiaokang Ye, Lianhui Qin, Peter Yichen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31511)  

**Abstract**: Coordinating two players in a cooking task requires a vision-language agent to track changing scene states, sequence recipe steps, and allocate shared resources under a time budget. We introduce an Overcooked-inspired benchmark for evaluating these capabilities in Unreal Engine environments constructed using coding agents. A single vision-language model controls both players from top-down visual observations, selecting high-level actions while the engine executes navigation. The benchmark contains 20 maps in five categories. These categories support evaluation of centralized coordination under equipment contention, changing terrain, and handoff dependencies. A prop legend and pot ingredient-count indicators provide explicit visual cues, while grid-based destinations support spatial grounding. The evaluation compares four VLMs on all 20 maps, measuring task score, order completion, delivery precision, action acceptance rate, and model API expenditure. A memory ablation on the same 20 maps examines the effect of a shared, self-authored notebook while retaining six-response action history in both conditions. The strongest evaluated model, \texttt{gpt-6-astra}, achieves 26.2\% mean order completion. Notebook memory has mixed score effects across models, with positive mean differences for Claude and Qwen and negative differences for GPT and Gemini. 

---
# FairFRESH: A Neuro-Symbolic Framework for Fair and Explainable Household Food Recommendation 

**Authors**: Kodur Sai Vinay Sathvik, Kakoli Bora, Tejaswini R. M  

**Link**: [PDF](https://arxiv.org/pdf/2610.31506)  

**Abstract**: Recommender systems typically serve one user at a time. However, in households, members share a finite pantry and follow different dietary rules, making independent recommendations prone to resource conflicts and dietary violations. We formulate household food recommendation as constrained fair division and present FairFRESH, a neuro-symbolic framework that learns per-member utilities, allocates foods under explicit welfare objectives, enforces dietary rules through a deterministic filter, and explains recommendations using Shapley and Banzhaf attributions. Theoretically, we show that no finite penalty guarantees compliance near a dietary-rule threshold, establish approximation guarantees for greedy allocation under utilitarian and smoothed Nash objectives in the stock-free model before filtering, and characterize the scale invariance of Nash welfare. We also use the minimax theorem to analyze rotating shared meals. On NHANES-derived data, the learned utility model achieves Recall@20 of 0.293, outperforming five learned recommenders (at most 0.244) but not a previous-day repetition heuristic (0.321). On held-out four-person households, Nash allocation reduces Gini disparity by 42% relative to conflict resolution while improving Recall@20 from 0.180 to 0.197. It comes within 2% of an upper bound on the smoothed Nash optimum, and rescaling one member's utilities changes at most 1.5% of assignments. All recommendations satisfy the encoded dietary rules, while attribution scores correlate with utility loss upon component removal (r = 0.76). In simulation, shared meals reduce pantry use by about one-third, and weekly rotation increases the worst-off member's utility by 38% over the best fixed meal. 

---
# Adaptive Skill Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery 

**Authors**: Dingxuan Zhang, Yanhui Jiang, Jun Wang, Hefeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.31461)  

**Abstract**: Reusable skills allow large language model agents to package task-specific procedures, tool affordances, and execution guidance into modular units. Prior work has largely studied how agents retrieve, compose, or synthesize skills before execution. In coding agents, however, execution poses an additional runtime control problem: the same task may be solved through primitive tool calls, macro skills, or hybrid execution that starts with a macro and falls back to micro-level recovery. We introduce Adaptive Skill Granularity, a framework that treats skill granularity as an action-abstraction policy conditioned on task structure, execution state, and macro-skill failures. The system combines a feature-based granularity controller and a finite-state recovery executor. We evaluate the framework with a deterministic coding-agent protocol using verifier-defined success, total cost, and process metrics such as macro adoption, failure recovery, route match, and transition correctness. Results show that fixed macro execution is efficient when applicable but brittle under mismatch, while runtime routing and explicit macro-to-micro recovery improve robustness and recover controlled macro failures that fixed-granularity execution leaves unresolved. Our findings suggest that skill libraries for coding agents should be treated not only as retrievable resources, but also as controllable, verifiable, and recoverable execution abstractions. 

---
# DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels 

**Authors**: Thanh Q. Tran, John Russell Himawan, Arun Verma, Kiwan Wong, Wei Xiao, Daniela Rus, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2610.31438)  

**Abstract**: When a user says "Keep it short", how short should the LLM's response be? More generally, users and applications require different levels of instruction adherence, but prompting does not reliably provide fine-grained control over them. We therefore study "steerable language models" that provide fine-grained numerical control over the degree to which models follow general instructions specified in text. However, existing approaches do not jointly offer fine-grained control, efficiency, and scalability across diverse instructions. To address this gap, we introduce DIAL (Distilled Instruction-Adherence Levels) to achieve all three properties. For fine-grained control, we perform classifier-free guidance (CFG) with a frozen LLM to provide training signals for on-policy distillation of instruction adherence at different levels. For efficiency, we design a lightweight steering adapter that uses a classifier over the LLM's hidden states to compute a closed-form steering edit. The classifier adaptively sets the steering magnitude at each token based on the target adherence level, while accounting for the geometry of the LLM's hidden states to preserve response coherence. For scalability, the classifier is text-conditioned, allowing a shared adapter to support new instructions without per-instruction fitting, and training requires only instruction and request text. Evaluation across multiple LLM families shows that DIAL achieves a competitive balance between concept expression and response fluency, with concept scores increasing monotonically across adherence levels. We introduce SteerIF, an instruction-only dataset of 98, 625 concept, persona, verifiable, and descriptive instructions. When trained on SteerIF, DIAL provides fine-grained control over personas and, when combined with prompting, improves verifiable instruction following. 

---
# Conservative Meta-Agent Search: Architecture Adaptation under Non-Stationarity 

**Authors**: Ali Atiah Alzahrani  

**Link**: [PDF](https://arxiv.org/pdf/2610.31436)  

**Abstract**: Automated agentic-system design is typically evaluated on static task suites with immediate scalar rewards. Deployed systems face persistent shift, delayed dependent feedback, correlated component failures, and costly architecture changes. We formalize meta-agent architecture search as sequential incumbent replacement with switching costs and hard risk constraints and introduce Conservative Meta-Agent Search (CMAS). CMAS retrieves archived architectures and proposes typed graph edits, then scores each against the incumbent by paired shadow replay under common random numbers, with interventional component credit. Promotion requires an empirically calibrated, variance-penalized simultaneous score above a pre-specified margin, and is followed by a simulated canary stage and pre-specified rollback. Since backtests cannot certify meta-level skill, we introduce MetaInvest-Bench, a regime-switching simulator whose specified worker competence makes a surrogate-optimal architecture computable, with leakage-controlled replay, adversarial governance tracks, and five negative controls. Financial streams serve as a risk-sensitive testbed for meta-level generalization; we claim no alpha. Across 96 streams, CMAS cuts normalized mean surrogate dynamic-oracle regret from 0.116 to 0.104 and raises held-out utility from 0.44 to 0.48 over the strongest adaptive baseline; when extra workers add no value, it contracts to near-single-agent teams. Its deployment inference cost is ~6% lower, though its search-inclusive token cost is the highest of the methods compared. The observed per-update false-promotion rate is 3.4% against a nominal 5% target, and 4.8% in a sealed rerun. The calibration is empirical: it concerns an update-level replay statistic on an author-designed benchmark, and guarantees neither full-utility improvement, nor absence of simulated canary-stage harm, nor sequence-level error control. 

---
# PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models 

**Authors**: Taekoan Yoo, Wonkyung Jung, Kyunghun Kim, Kyeongbo Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.31429)  

**Abstract**: World action models increasingly rely on pretrained video models assumed to capture physical dynamics for prediction, planning, and action generation. Existing evaluations assess plausibility, probe representations, or measure predictive surprise, but rarely identify which history-conditioned physical expectations conflict with observations. We introduce PhysTrace, a training-free framework aligning MLLM-generated, history-conditioned expectations with observed video windows to temporally localize physical violations and provide interpretable descriptions without task-specific probes or classifiers. We also introduce IntPhys2-Timing, adding violation intervals to IntPhys2. PhysTrace achieves a Hit Rate of up to 70.7% across physical principles on IntPhys2-Timing and outperforms the compared methods in overall possible-impossible discrimination accuracy on IntPhys2. Applied to reconstructed videos along the trajectories of pretrained diffusion/flow-based video models, PhysTrace reveals attribute- and noise-dependent patterns in physical violation localization, enabling interpretable physical consistency analysis across timesteps. 

---
# Certified Execution Harness for Agentic AI: Safe and Least-Intrusive Petri-Net Supervisory Control Between Reasoning and Action 

**Authors**: Chao Gu, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31376)  

**Abstract**: This paper studies the problem of safe tool execution for large language model (LLM) agents under prompt injection. Using Petri nets, we formalize the execution harness between the LLM and its tools as a supervisory control problem, in which marks represent capabilities, approvals and budgets. Since a transition fires only if the marks it needs are present, an injected request whose preconditions are unmet or already spent is refused regardless of its wording. Treating tool outcomes as uncontrollable, we enforce safety by the conventional maximally permissive supervisor, and this safety transfers to the running system under four assumptions. Moreover, to capture an agent that may propose again after a refusal, we introduce the notion of a replanning closed loop, in which each retry costs an inference call. We show that, in this loop, the conventional supervisor can be arbitrarily more intrusive than a stricter safe one. Furthermore, for a given proposer model, the maximally permissive least-intrusive supervisor is characterized as a value-threshold refinement of the conventional one. Under precise conditions, this refinement orders two concurrent actions by at most one monitor place. On an autonomous release agent, we find by synthesis that safe deployment requires a step that the supervisor can refuse after a rollout starts. 

---
# RIDAR: Safeguarding Research Integrity in Autonomous Research Agents via Automated Detection and Runtime Intervention 

**Authors**: Jiening Siow, Amy Xin, Junjie Wang, Ziyu Tang, Zijun Yao, Yunjia Qi, Fanjin Zhang, Lei Hou, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.31372)  

**Abstract**: Autonomous research agents are increasingly used to automate large portions of scientific research, yet they are prone to research integrity violations (RIVs): behaviors that compromise the validity, reproducibility, or faithfulness of the research process. We present RIDAR, a framework for automated RIV detection and runtime intervention. First, we introduce RIVT, a comprehensive taxonomy of 14 RIV types spanning the full research lifecycle. Building on RIVT, we present RIDAR-post-hoc, a framework to detect RIVs within agent trajectories. RIDAR-post-hoc achieves 85.2% F1 performance on RIV-Gold, a benchmark for RIV detection with 60 expert-annotated trajectories. Applying RIDAR-post-hoc to RIV-Wild, a dataset of 300 agent trajectories across 50 distinct research tasks, we detect 3,964 RIV instances in 99.7% of trajectories, revealing the prevalence of RIVs in current agents. Finally, we present RIDAR-runtime, a framework for dynamically detecting and correcting RIVs during agent runtime. RIDAR-runtime effectively reduces agent RIVs remaining at task completion by 36.8%. We will release our code and data to facilitate future research. 

---
# Muffin - A Shared Memory Buffer for Asynchronous AI Agents 

**Authors**: Ping-Lin Kuo, Suman Navaratnarajah, Christian Frey, Gresa Shala, Josif Grabocka  

**Link**: [PDF](https://arxiv.org/pdf/2610.31355)  

**Abstract**: The rise of Agentic AI systems has changed the landscape of artificial intelligence by enabling strong reasoning capabilities through specialized multi-agent interaction with structured self-reflection. However, in target-driven reasoning tasks, single- and multi-agent systems are often on par in terms of performance gains. A major challenge is that failures often arise from an ineffective exchange of critical information during inter-agent interaction and coordination. Different from existing multi-agent debate paradigms that rely on synchronous turn-taking and sharing information across different agents, we propose an inter-agent protocol based on shared Memory bUFFer INference - referred to as MUFFIN - in an asynchronous agentic framework. In the first step, an orchestrator routes queries based on their reasoning demands and delegates demanding ones to a multi-agent setting, where a shared memory buffer provides asynchronous exchange of intermediate findings while an agent's private reasoning remains local. Our experiments show that our framework Muffin outperforms established competitors, achieving the best accuracy in 17 out of 20 of our benchmark settings, and provides a better quality-cost trade-off than multi-agent debate: Muffin improves accuracy by 19.2 accuracy points while using 77.4% fewer tokens, without requiring full-trace sharing. 

---
# ValuSpec: Plug-and-Play Candidate Valuation before Target Verification for Tree-Based Speculative Decoding 

**Authors**: Liang He, SiYuan Ma, Qishi Zhan, Yongqi Fan, Mingyu Cao, Qizhen Lan, Zhaolu Kang, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31338)  

**Abstract**: Tree-based speculative decoding accelerates large language model inference by verifying multiple candidate branches in parallel. However, in practical serving scenarios, similar requests repeatedly induce overlapping candidate branches. While retrieval-based methods mitigate this by reusing historical drafts, the resulting candidate trees often contain many low-quality branches as the candidate tree grows, which reduces verification efficiency and wastes target-side computation. To address this, we propose ValuSpec, a plug-and-play candidate valuation mechanism that filters low-quality branches from a supplied candidate tree before expensive target-side verification. ValuSpec introduces transfer-based supervision to align filtering with the target model's behavior. A filtering threshold is estimated from the target model's greedy trajectories on a calibration split, without any parameter training. The target model then verifies the filtered tree under the output-identical greedy rule, so accepted tokens are guaranteed to match standard autoregressive decoding exactly. Experimental results on HumanEval, GSM8K, Dolly, and SpecBench show that, in our main evaluation, ValuSpec achieves up to 2.38x end-to-end speedup in retrieval-based tree settings. Furthermore, integrating ValuSpec into the EAGLE-3 dynamic tree framework yields up to 2.78x end-to-end speedup without modifying its tree construction strategy, demonstrating that ValuSpec serves as a generally applicable and plug-and-play candidate valuation module. 

---
# Subgoal-aware Memory Graph for GUI Agent 

**Authors**: Dezhao Luo, Daniel Beechey, Derek Yuen, Weilin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.31307)  

**Abstract**: GUI agents aim to automate complex user tasks by interacting with mobile interfaces, but ef- fective decision-making in long-horizon set- tings requires reusing prior experience rather than relying solely on the environment ob- servations. Existing memory-based methods typically store experience either at the single- action level or as entire trajectories. Single- step actions are often too fine-grained to carry meaningful semantics, since the success of an individual action does not necessarily imply progress towards the overall goal. In contrast, whole trajectories are often too coarse-grained and difficult to reuse across different tasks. We propose a subgoal-aware memory graph that distils past trajectories into structured, reusable evidence at the level of subgoals. We define a subgoal as consecutive actions shar- ing a single intent, the granularity at which progress toward the task becomes meaning- ful. To reduce redundancy and support cross- task reuse, we organise subgoal experiences as nodes in a graph, where related memories can be merged, extended, and retrieved based on the environment observation. On SPA-Bench, our method improves success rate by 11.11% over the baseline. 

---
# The Value of Memory in Language Agents 

**Authors**: Zehao Lin, Erenkai Zhu, Helan Hu, Xixuan Hao, Chunyu Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31303)  

**Abstract**: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exact recovery. Diagnostics on LoCoMo, a two-project update task and longitudinal cases exhibit failures of preservation, use and measurement. At fixed retrieved evidence, a saved ledger changes subsequent answers; at fixed agent outputs, an evaluator revision changes scores. The framework connects these distinctions to operational risk bounds and testable principles for allocating memory, preserving recovery options and revising evidence dependencies. 

---
# RIPPLE: Recovery Internalization through Process-Level Policy Learning and Effect-Aware Credit for Long-Horizon Agentic Coding 

**Authors**: Jingxuan Xu, Zihang Xu, Yuechen Luo, Ziqi Miao, Wenqiang Zhu, Weihao Li, Junhao Fan, Xinshun Feng, Zhiyi Lai, Yanan Wu, Haoyang Huang, Ken Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.31276)  

**Abstract**: Long-horizon coding agents can observe a failure and generate a plausible critique yet continue along the same incorrect trajectory. We call this the Critique-to-Action Handoff Gap. We introduce RIPPLE, a recovery-oriented training framework that internalizes process-level guidance into a single self-recovering policy. A prefix-only teacher provides evidence-grounded recovery supervision without access to future actions or terminal outcomes. After lightweight distillation, reinforcement learning assigns stage-specific credit to reflection triggering, diagnosis, and corrective action. Local validity signals gate trajectory advantages so that intermediate recovery decisions do not indiscriminately inherit terminal outcomes. Across four repository-level software-engineering benchmarks, RIPPLE improves the base policy by an arithmetic mean of 4.8 score points, with no auxiliary critic at deployment. Controlled ablations isolate the benefits of local, stage-aligned credit, and matched-prefix interventions show that critique semantics affect subsequent correction. Recovery gains concentrate where the prefix contains sufficient diagnostic evidence and the repository state remains reversible within the remaining budget. 

---
# PointerBench: A Diagnostic Benchmark for Fine-Grained GUI Grounding 

**Authors**: Valentino Sacco, Saurabh Pandey, Alexander Florian Jercher, Maximilian Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2610.31274)  

**Abstract**: Multimodal Large Language Models have shown promising performance in computer-use tasks, from planning user workflows to grounding natural-language actions to precise UI targets. Existing end-to-end benchmarks primarily emphasize long-horizon task completion, while GUI grounding benchmarks remain focused on conventional controls and a limited set of professional interfaces. We introduce POINTERBENCH, a diagnostic suite of 1,500 screenshot-instruction pairs probing three complementary capabilities: thin sub-element identification and geometric grounding in spreadsheets (POINTERBENCH-Sheets), character-level grounding in long-text documents (POINTERBENCH-Text), and mixed GUI grounding across 100 professional applications (POINTERBENCH-Pro). We carefully curate each task with pixel-exact target geometry and evaluate both general-purpose multimodal models and 39 specialist GUI grounding configurations. We find text grounding to be the most challenging setting, with Claude Fable 5, GPT-5.6 Sol, and the leading native specialist UI-Venus-2-9B achieving 69.6%, 53.4%, and 25.8% success on POINTERBENCH-Text despite respective 83.2%, 77.7%, and 54.5% average success across POINTERBENCH. These results reveal substantial headroom in fine-grained GUI grounding across frontier and specialist models. We publicly release the benchmark and evaluation code at this https URL. 

---
# Reducing Multi-Horizon Photovoltaic Forecast Error through Tree-Anchored Expert Combination 

**Authors**: Sun Ze, Zhou Liguo, Xu Yuqing, Yu Lei, Jiang Mingming, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2610.31248)  

**Abstract**: A strong direct tree model may remain the best standalone photovoltaic forecaster, yet a weighted combination can reduce its residual errors. We examine whether complementary temporal predictors can lower the tree's mean absolute error (MAE) over 15--240 minutes without weaker ensemble members degrading performance. The candidate bank contains LightGBM and seven neural implementations with different temporal representations. A combiner fitted to MAE has nonnegative weights that sum to one. Unhelpful candidates can receive zero weight, while the tree anchors the combination and provides a development-stage fallback. The main design shares weights across horizons; horizon-specific weights and alternative losses test whether further complexity is needed. On 56,078 windows from two co-located public PVDAQ systems, the historical shared-MAE policy reduces scaled MAE by 3.21\% relative to the tree.
The latest bounded neural correction does not reduce MAE further than this static combination. The results support selective combination of complementary predictions in this bank rather than equal averaging or indiscriminate addition of modules. Both analyses reuse a previously inspected evaluation period. They do not establish independent replication, weather-specific causation or universal gains from eight experts. 

---
# Reward-Weighted Preference Optimization for Selective Combinatorial Optimization Problems 

**Authors**: Haohui Zhang, Xinyu Hu, Wouter van Heeswijk, Neil Yorke-Smith, Martijn Mes  

**Link**: [PDF](https://arxiv.org/pdf/2610.31239)  

**Abstract**: Reinforcement learning has emerged as an effective approach for combinatorial optimization, but most existing methods focus on problems such as the Travelling Salesperson Problem (TSP) in which all elements must be served. Selective combinatorial optimization problems (SCOPs), including the Orienteering Problem (OP), Prize-Collecting TSP (PCTSP), Stochastic PCTSP and Knapsack Problem (KP), additionally require selecting which elements to include. This selection creates sampled solutions with substantially different objective values, yet standard preference optimization reduces such comparisons to binary rankings and discards the magnitude of these differences.
We introduce Reward-Weighted Preference Optimization (RWPO), a pairwise training objective that incorporates reward-gap information into preference learning. For candidate trajectories sampled from the same instance, RWPO weights each preference loss according to the normalized difference in their rewards, assigning greater training emphasis to comparisons that represent larger improvements in solution quality. We further combine RWPO with scale-specific fine-tuning and reward- and likelihood-weighted tilted sampling to adapt trained policies to larger and shifted instance distributions without relying on local-search post-processing.
Experiments on the OP, PCTSP, Stochastic PCTSP, and KP show that RWPO consistently improves over unweighted preference optimization and policy-gradient baselines across a wide range of instance sizes. On the OP, the resulting solver exceeds the mean reward of the Compass metaheuristic at scales from 100 to 2,000 nodes. The results demonstrate that preserving reward-gap information can substantially improve preference-based learning for selective combinatorial optimization and support generalization to large-scale instances. 

---
# Trion-1: Unifying Language, Protein Sequence, and Structure with Diffusion Language Models 

**Authors**: Tianyu Lu, Kaicheng Yang, Jiangtao Feng, Keyue Qiu, Yawen Ouyang, Yixin Wu, Xinbo Zhang, Lihao Wang, Wei-Ying Ma, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.31215)  

**Abstract**: We envision a general-purpose protein foundation model as a unified, interactive model of sequence, structure, and function, capable of understanding, generation, and cross-modal translation while retaining the instruction-following and reasoning capabilities of modern LLMs. Toward this vision, we introduce Trion-1, the first native, unified, multimodal diffusion language model (dLLM) for proteins. Trion-1 is built on two key design choices: (1) a unified discrete token space that jointly represents sequence, structure, and natural language; (2) a dLLM backbone that replaces strict left-to-right factorization with bidirectional or semi-bidirectional token interactions, enabling global residue communication and long-range contact modeling that are naturally suited to structure modeling. This formulation unifies six core capabilities: folding, inverse folding, sequence understanding, structure understanding, sequence design, and structure design, while retaining the native language capabilities of general-purpose LLMs. On established benchmarks, Trion-1 achieves broadly competitive performance. In folding, Trion-1 achieves a TM-score above 0.8 with a single sample under strict homology control, matching or exceeding the existing tri-modal model using 200 samples. For sequence understanding, Trion-1 performs on par with top-tier LLMs such as Claude Opus 4.7 (62.37%). These results support well-aligned sequence-structure-language representations and strong cross-modal generative capabilities. 

---
# Different Tasks, Different Timesteps: Temporal Task-Aware Quantization for Diffusion Language Models 

**Authors**: Ofek Avivi, Eliron Rahimi, Avi Mendelson, Chaim Baskin  

**Link**: [PDF](https://arxiv.org/pdf/2610.31198)  

**Abstract**: Diffusion Language Models (DLMs) introduce unique quantization challenges due to their iterative denoising process. In this work, we reveal that quantization sensitivity is both taskand time-dependent, with different downstream tasks exhibiting distinct precision requirements across denoising steps. We further show that these patterns can be predicted without groundtruth labels or task-specific evaluation metrics. Building on these findings, we introduce T 2Quant, a task and time-aware mixedprecision quantization framework that uses a label-free sensitivity signal to optimally allocate activation precision across denoising steps under a fixed bit-width budget. Extensive experiments across three DLMs and six benchmarks demonstrate that T 2Quant consistently improves the average accuracy of existing quantization methods, achieving gains of up to 6.7 percentage points under the same budget. 

---
# Rethinking Fisher Vectors for Second-order Multi-scale Deep Feature Aggregation in Scalable Scene Retrieval 

**Authors**: Alexy Bhowmick, Sarat Saharia, Shyamanta M. Hazarika  

**Link**: [PDF](https://arxiv.org/pdf/2610.31146)  

**Abstract**: Aggregating CNN responses into a compact discriminative vector, given their lack of geometric invariance, is a challenge. To address this challenge, we propose a codebook-free method to encode second-order information of multi-scale CNN features within the Fisher Kernel framework. We use a scalable retrieval pipeline to show the efficacy of the proposed vector. The increased discriminative power is confirmed by experimental evaluation of our method on two difficult benchmarks (INRIA Holidays, UKBench). The significant improvement in retrieval accuracy over state-of-the-art approaches comes with only a slight dimension increase. 

---
# Post-Audit Mirage: Identifiability Limits of Offline Agent Verification 

**Authors**: Aman Behera, Payal Payal  

**Link**: [PDF](https://arxiv.org/pdf/2610.31130)  

**Abstract**: Offline tests can fail to reveal whether an agent update will remain safe after deployment. We operationalize this post-audit mirage using matched safe and harmful lifecycles that expose identical verifier-visible evidence while an isolated scorer records opposite deployment truth. Our theorem applies standard observational-equivalence and indistinguishability ideas to this executable audit boundary and states the resulting error-abstention tradeoff. The Post-Audit Mirage benchmark makes that boundary executable across authorization, scheduling, and shared-capacity environments. IdentifiedRangeMonitor uses fresh group-labelled harm outcomes to return deploy, hold, cannot determine, or unsupported. Experiments show that monitoring can recover the missing distinction in favorable settings but may remain undecided for rare groups, small effects, or missing observations. Verifier-agent evaluation must therefore test both calculation validity and whether the available evidence supports the deployment decision. 

---
# SetCommit: Coordinated Token Commitments for Few-Step Diffusion Language Model Decoding 

**Authors**: Yuannuo Feng, Yubing Ye, Wenshuai Yao, Yizhe Chen, Ngai Wong, Wenyong Zhou, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31120)  

**Abstract**: Diffusion language models generate multiple token positions in parallel, but few-step decoding makes parallel commitment difficult: tokens that appear reliable individually may be jointly inconsistent, while premature commitments can prevent later context-dependent refinement. In this paper, we introduce SetCommit, a training-free method for coordinated parallel token commitment. Instead of selecting tokens independently by confidence, SetCommit selects a commitment set that balances token reliability against local joint commitment risk, preserving conditioning opportunities for subsequent denoising steps. It further performs budget-neutral local exchanges, replacing vulnerable earlier commitments when newly revealed context supports more reliable alternatives. SetCommit requires no retraining or architectural modification and is compatible with existing diffusion decoding strategies. Experiments with LLaDA-8B and Dream-7B on GSM8K, MATH-500, MBPP, and HumanEval show accuracy/pass@1 gains of up to 7.4 percentage points over native decoding at 32 model evaluations. At common quality targets, geometric-mean incremental speedups range from $1.44$ to $1.74\times$. Code is available at this https URL. 

---
# Constraint Reasoning and Large Language Models Join Forces: Providing Contrastive Explanations for Real-World Optimisation-Based Applications 

**Authors**: Roger X. Lera-Leri, Filippo Bistaffa, Athina Georgara, Juan A. Rodríguez-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2610.31119)  

**Abstract**: We focus on optimisation-based Decision Support Systems in real-world applications. In this context, we aim to provide contrastive explanations that answer users' questions about recommendations based on the solution of an optimisation problem, specifically a Mixed-Integer Program (MIP). Addressing this endeavour is crucial to ensure human oversight of the decisions implemented, as required by international guidelines on trustworthy and responsible automated decision-making systems. We propose a novel methodology that combines constraint reasoning techniques and Large Language Models (LLMs) to generate contrastive explanations for optimisation problems encoded as MIPs. Although LLMs have shown impressive capabilities in Natural Language processing, they can produce incorrect answers due to reasoning limitations. To overcome these limitations, we leverage constraint reasoning techniques to distil the information necessary to produce an explanation, which is then passed as input to an LLM as the so-called graph of reasons. Such a graph captures the semantics and the relationships among constraints of the optimisation problem. Results on real-world scheduling domains show that incorporating constraint reasoning improves explanations generated by the LLM. In particular, using the graph of reasons enables the LLM to produce natural-language explanations that include more relevant reasons while reducing errors compared to explanations generated without this structure. 

---
# Can AI Co-Design Structures? An AI Agent for Early-Stage Structural Design 

**Authors**: Ricardo Maia Avelino, Mennatallah El-Assady, Tom Van Mele, Philippe Block  

**Link**: [PDF](https://arxiv.org/pdf/2610.31113)  

**Abstract**: Recent advances in large language models and agentic workflows offer a new paradigm to early-stage structural design translating design intent into structural models. Although text-to-3D approaches have demonstrated that natural language can be used to produce visual, mesh-based geometric models, structural design requires domain-specific geometric arrangements and semantically rich data types. This paper introduces a proof-of-concept workflow for prompt-to-structure generation using an AI agent connected to a COMPAS-based structural modelling API through a Model Context Protocol (MCP) server. Through controlled modelling operations, the agent creates and modifies a graph-based structural data model composed of explicit building elements. The resulting artifacts can be inspected, modified, serialised, and further connected to downstream AEC workflows. The approach is demonstrated through selected prompt-to-structure examples and by progressive and editable design sequences that resemble typical iterative structural design workflows. The results are preliminary, but demonstrate the possibility of combining natural language, agentic workflows, and domain-specific structural data structures as a first step toward AI-assisted structural co-design. 

---
# Who Is the Influencer? Consensus Dynamics in Multi-Agent Systems 

**Authors**: Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2610.31105)  

**Abstract**: Multi-agent LLM systems are increasingly used in practice, but their behavior is complex and hard to understand. We design a simple task to ask which model has the most influence on the final consensus when several models are mixed in one system. Experimentally, across seven model families, we find that the opinions of the other models are absorbed into the opinion of the stronger model, the one with more parameters. This happens even though the task requires no pretraining knowledge and has no correct answer. To understand this phenomenon, we interpret the agents as weighted particles moving on a simplex. Larger models carry a larger weight and therefore occupy a deeper well in the energy landscape. A large weight means strong in-context learning: a model with a large weight amplifies the majority of its context. Larger models are generally better at in-context learning, and this is what makes the larger model the influencer. We hope that this work helps to interpret multi-agent systems, whose behavior is becoming too complex for humans to understand. 

---
# DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150 

**Authors**: Shuang Song Zhongqiao Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31104)  

**Abstract**: We present DeepOpen, a decision-Oriented intent-classification framework with dataset- specific systems for BANKING77 and CLINC150. On BANKING77, a single English classi- fier continued with sharpness-aware minimization (SAM) achieves 94.1234% test accuracy (2,899/3,080). This is the best observed result among three fixed SAM seeds, selected for display after test evaluation; their mean is 94.0368%. On the 150 in-scope CLINC150 classes, probability fusion of one Laya specialist and three DeBERTa-v3-large classifiers achieves 98.0222% (4,411/4,500). We describe the two configurations, reproduce the reference com- parisons displayed in the accompanying evaluation explorer, and separate headline scores from evidence about improvement. Under a uniform BANKING77 evaluation path, the SAM mean exceeds the historical checkpoint by 0.0433 percentage points, with a shared-row 95% interval of [-0.2273, +0.3247]. Thus, the continuation does not establish a reliable accuracy gain. The report documents strong observed task-specific performance, selection effects, and unresolved historical execution-path differences, rather than a new backbone or a unified algorithmic advantage. 

---
# Solving Flood-It Puzzles: A Comparison of Tailored A* Search with Declarative Solver Approaches 

**Authors**: Philipp von Manteuffel, Benjamin Blankertz  

**Link**: [PDF](https://arxiv.org/pdf/2610.31100)  

**Abstract**: We present an optimized approach for finding minimal move sequences to solve Flood-It puzzles using A* search and compare it with approaches using SAT, Integer Programming, and Constraint Programming solvers. The performance of A* search is considerably improved by using an appropriate heuristic and sophisticated equivalences of game states to limit the search space. Each aspect is systematically evaluated, providing insights that generalize to other problems. By comparing the best variant of each approach, the A* search is the fastest in finding a minimal solution in about 75 % of the Flood-It instances tested. This result is in line with our expectation since A* specifically looks for a minimum solution. In contrast, the SAT-based approach is designed to solve the decision problem of whether a solution with a given length exists. An iterative search is required to determine a minimum solution length using a SAT-based approach. However, with increasing complexity, quantified by the minimum solution length, the SAT-based approach outperforms A*. We also compared the different approaches to the decision problem, where A* again performed best for the largest number of puzzles. Here, again, SAT supersedes A* for higher complexities, albeit with a lower crossing point. 

---
# Co-Recursive Semantic Self-Extension: A Mathematical Foundation for Open-Ended Intelligence 

**Authors**: Simon Hegelich, Bardh Prenkaj, Habiba Sarhan, Morteza Shahrezaye, Kolja Hegelich, Eric Cambria, Christian Montag  

**Link**: [PDF](https://arxiv.org/pdf/2610.31097)  

**Abstract**: Artificial intelligence has advanced through scale, data, and computation, yet remains bounded by its representational framework. We ask what makes intelligence open-ended. Every inferential system selects meaningful distinctions, limiting what it can learn. Within a fixed purpose, performance reaches a ceiling set by its semantic domain. Teleological succession, in which purposes arise from earlier ones, faces a similar limit: Boolean-closed purpose-transformations exhaust their reachable semantic territory. We shift from purposes to inferential systems. Drawing on Kant and Gödel, we model cognitive states as partial evaluations over a formal language. Unresolved propositions may be undecided or undecidable. Reasoning can resolve the former but not the latter, leaving a frontier for extension. Non-trivial reductions of undecidable propositions expose undecidable intermediate premises whose reflective reification supports multiple decisive extensions. Finite trace-guided generation remains fallible, requiring meta-level selection and polycontextural exploration. Using Kolmogorov complexity, we show that no architecture with a fixed state description, transition mechanism, and initial condition can generate histories incorporating irreducible algorithmic information. Because physical systems are fixed in this sense, we examine collectives. Coupling finitely many finite agents does not remove the bound. Extension requires persistent external memory preserving semantic content and historical coordinates. Collective generative capacity depends on balancing overlap for transfer with difference for generation. We propose semantic self-extension as a design principle: intelligence should not merely resolve ambiguity within existing meanings, but transform unresolved structures into new spaces of meaning, generating new distinctions, questions, representations, and purposes. 

---
# Dual Exam: Controlled Evaluation of Multimodal Agents and World Models 

**Authors**: Zhefan Rao, Meng Chu, Xuan Zhang, Jingfan Chen, Guohao Cai, Runyi Li, Shiyi Du, Ning Lu, Xu Huang, Zhixuan Chen, Feng Nie, Harold Haodong Chen, Hanwen Wang, Wen-Jie Shu, Yajing Bai, Xianfeng Wu, Xuanhua He, Shuhan Guo, Ziyue Ma, Ruichuan An, Long Bai, Miaosen Chai, Wenhao Chai, Jingye Chen, Qi Chen, Xuhang Chen, Wei Chow, Minghao Fu, Yunhao Gou, Haokun Gui, Dong Huang, Wei Huang, Zi Cheng Huang, Haibo Jin, Lingdong Kong, Leung Chun Ho, Tianqin Li, Yanheng Li, Jiayuan Liu, Jiyang Liu, Mingyu Liu, Shaoteng Liu, Xiaocheng Lu, Yawen Luo, Zeyu Qin, Tiancheng Shen, Yueqi Song, Zhaolong Su, Hengkai Tan, Tianhao Tang, Yiran Tao, Juncheng Wang, Qinghe Wang, Shangshang Wang, Yingying Wang, Fang Wu, Tiange Xiang, Qiushi Yang, Weicai Ye, Weichen Yu, Jiarui Zhang, Jize Zhang, Liuzhou Zhang, Renrui Zhang, Sheng Zhang, Wenhu Zhang, Junru Zhao, Lingting Zhu, Yuxin Zuo, Yutong Bai, Long Chen, Yang Deng, Ziqi Huang, Yeying Jin, Weijia Li, Kevin Qinghong Lin, Quande Liu, See-Kiong Ng, Hongliang Ren, Hongru Wang, Shujun Wang, Qifeng Chen, Haoxuan Che  

**Link**: [PDF](https://arxiv.org/pdf/2610.31066)  

**Abstract**: World Models (WMs) can support Multimodal Agents (MAs) through simulated experience, while Agent trajectories can inform World Model refinement. Evaluating this interaction requires separate evidence about action selection and state evolution. We introduce Dual Exam, which tests Agents under fixed environment response mechanisms and World Models under fixed action interventions across physical, digital, social, and scientific domains. The Agent track compares 10 configurations on 1,088 items in 15 reporting categories. The World Model track contains 400 cases across eight scenarios; 15 configurations have full coverage and one covers six scenarios. GPT 6 Astra leads the case-weighted Agent Index at 31.3, and GPT-Image-2 leads the full-coverage WM Overall at 21.7. These scores use different tasks and metrics and are not directly comparable. A 150-case pilot of one coupled pairing records zero task successes and motivates component-level diagnosis. Results reveal uneven task performance and failures to preserve task-relevant state. Their value for predicting coupled success or sustained reciprocal learning remains to be established. 

---
# SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs 

**Authors**: Arnon Mazza, Elad Levi  

**Link**: [PDF](https://arxiv.org/pdf/2610.31062)  

**Abstract**: Policy enforcement for LLM agents has shifted from single-turn content moderation to compound, application-specific policies evaluated over multi-turn transcripts and tool-call trajectories. Fine-tuned guardrail models are efficient but degrade on compound policies, and while LLM-as-a-judge approaches achieve higher accuracy, they incur high inference cost and tend to produce unstable predictions. Neurosymbolic methods offer a natural alternative, decomposing the policy into narrow neural decisions composed under explicit logic. However, existing symbolic layers lack the required expressiveness and are optimized against fixed data rather than the policy itself. To address this, we introduce SCOPE, which compiles a policy into an executable program of neural predicates executed by small models, deterministic extractors, and code that composes their outputs. The program is optimized jointly with its training data, using adversarial generation of counterexamples that target where the program deviates from the policy. We evaluate SCOPE on a benchmark of real-world policy tasks against prompt optimization, task decomposition, and workflow generation methods, achieving the highest average accuracy across tasks and models. Programs optimized with the weakest executor transfer to other models without further optimization. This allows them to be deployed in production on inexpensive models, where they reach accuracy comparable to frontier reasoning models. Unlike these monolithic models, SCOPE produces substantially more stable predictions, and each verdict is interpretable through the predicate outputs that produced it. Additionally, we validate the key components of our method through an ablation study. SCOPE thus enables reliable, interpretable, and efficient policy enforcement at production scale. 

---
# Diagnosing the Explicit-to-Applied Theory of Mind Gap in LLMs 

**Authors**: Prajvi Saxena, Sabine Janzen, Wolfgang Maass  

**Link**: [PDF](https://arxiv.org/pdf/2610.31060)  

**Abstract**: As large language models are deployed in high-stakes roles such as therapy, education, and personal assistance, their social cognition must be assessed reliably. Current Theory of Mind (ToM) benchmarks report aggregate scores that conflate two distinct competencies: explicit ToM (recognizing what someone believes) and applied ToM (using that belief to predict behavior). Building on this distinction, we investigate the gap and its recoverability using four structurally diverse benchmarks. Aggregate reporting obscures uneven competence: on SimpleToM, Llama-3.1-8B scores 51.5 percent overall while recognizing mental states at 99.4 percent and predicting behavior at only 28.9 percent. To assess recovery, we compare structured prompting, matched in-context learning, and our proposed Belief-State Test-Time Training (BSTTT), which adapts lightweight parameters on answered support questions through action reconstruction. Across Qwen2.5-7B, Llama-3.1-8B, and GPT-oss-20B, recovery depends on the model, task, and intervention. On SimpleToM, reminders combined with BSTTT improve Llama's behavior prediction by 40.8 percentage points; reminders account for most of the overall improvement, with BSTTT adding 1.3 to 5.5 points beyond reminders across models. Contextual examples often outperform weight updates, while BSTTT remains stronger in specific model-benchmark settings. Temporal belief tracking remains difficult despite partial recovery. These findings show that aggregate ToM scores can overstate applied competence while hiding performance recoverable through targeted assistance. We recommend decomposed reporting of explicit and applied accuracy alongside intervention comparisons that establish which failures are recoverable and under what conditions. 

---
# Directing to Diversify: Multi-Directional Search for Automatic Heuristic Design 

**Authors**: Jianghan Zhu, Jianan Zhou, Fei Liu, Ni Zhang, Yaoxin Wu, Zhuoyi Lin, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.31048)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential to automate heuristic design for NP-hard combinatorial optimization (CO) tasks without human intervention. Recent methods predominantly rely on evolutionary computation (EC) frameworks, where heuristic diversification is mainly induced through EC-like operators or downstream selection, while the direction of heuristic transformation remains largely unharnessed. In this paper, we propose a multi-directional Perturbation-Rewrite Search (PRS) framework for automatic heuristic design that factorizes heuristic transformation through directional search primitives at the generation level. Specifically, PRS operates on heuristic ideas through three complementary directions: expansion toward richer and more expressive designs, compression toward simpler and less specialized designs, and steering for fine-grained refinement, together enabling broader and more sustained exploration of diverse regions in the heuristic design space. PRS further organizes candidate exploration within a breadth-first search-based process with adaptive operator control, which ensures thorough candidate utilization and sustained exploration, and maintains strong search dynamics while preventing premature convergence. Extensive empirical results across diverse CO tasks show that PRS consistently discovers high-quality heuristics, demonstrating the effectiveness and generality of explicitly directional exploration design. 

---
# Conflict-Aware Logit Adapters for Utility-Preserving Anti-Distillation 

**Authors**: Quoc Phong Dao, Binh Anh Nguyen, Linh Ngo Van, Thanh-Toan Do, Mehrtash Harandi, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.31046)  

**Abstract**: While reasoning traces improve model capability, they also provide supervision signals for downstream distillation. Initial efforts to mitigate this issue rely on decoding-time interventions, but they incur high inference overhead and can harm the teacher model's natural fluency. We propose Conflict-Aware Logit Adapters (CALA), a lightweight training-time defense. CALA attaches a small residual adapter to a frozen teacher model and trains only the adapter. A proxy student identifies positions where imitation is most sensitive, and the adapter perturbs the output distribution at these positions while remaining within high-probability regions. A conflict-aware optimization strategy balances anti-distillation and utility objectives, while regularization preserves the teacher's performance and fluency. At inference time, CALA incurs modest overhead and requires no auxiliary model. Experiments on reasoning benchmarks show that CALA substantially reduces successful student imitation while maintaining near-original teacher performance and generation quality. These results establish CALA as a practical and efficient alternative to decoding-time approaches. 

---
# Algorithmic drift in generative adaptive learning 

**Authors**: Firuz Kamalov, Hana Sulieman, Ahmed El Sayed, Sivaraj R., Nitesh Sughnani  

**Link**: [PDF](https://arxiv.org/pdf/2610.31041)  

**Abstract**: Generative artificial intelligence promises to democratize personalized tutoring, yet the longitudinal stability of adaptive systems remains under-theorized. Current ethical discourse focuses on static bias. This paper argues that a distinct threat lies in algorithmic drift: a feedback loop in which minor, demographically asymmetric observation errors compound over time into systemic inequities. We make three contributions. First, we formalize algorithmic drift as a coupled learner--tutor dynamical system. Second, we corroborate this theory with an agent-based Monte Carlo simulation in which either a transparent online tracker or an LSTM deep-knowledge-tracing network drives an IRT-based curriculum policy. Third, we introduce the temporal fairness stabilizer (TFS), a targeted intervention that periodically re-probes learner capability, and benchmark it against standard exploration baselines with explicit group-fairness metrics and replication-level inference. Within the modeled environment, TFS closes the achievement gap where generic exploration does not, and we characterize the conditions under which it succeeds. We separate what is established analytically, what is demonstrated in simulation, and what remains to be validated with real learners. Our results reframe algorithmic fairness in education as a temporal property of a closed loop rather than a deployment-time checkbox. 

---
# Multichoice Modeling: Exploring Decision Configuration Space under Uncertainty 

**Authors**: Mariia Kozlova, Roman Stepanova, Alexander Walzer, Andrea Saltelli  

**Link**: [PDF](https://arxiv.org/pdf/2610.31038)  

**Abstract**: Decision support has become sophisticated at exploring uncertainty, but much less so at representing the architecture of the decision itself. As a result, the decision space is often narrowed before uncertainty is analyzed, leaving interactions among choices and the effects of alternatives they imply largely invisible. This paper introduces multichoice modeling, a framework for making large decision configuration spaces explicit and computationally explorable under uncertainty. Drawing on the logic of rugged landscapes, we conceptualize such problems as contingent performance landscapes, where interdependent choices generate configuration-specific outcomes whose value changes across uncertain futures. The computational framework combines multidimensional decision and uncertainty spaces in a single simulation. Further, variance-based sensitivity analysis is used for identifying influential choices, and conditional distribution analysis for comparing configuration-specific performance profiles. We demonstrate the framework through three case studies: a manufacturing process upgrade, M&A strategy, and mortgage design. By contrasting conventional uncertainty-only analysis with multichoice modeling, the cases show how opening the decision configuration space can reveal unexpected complementarities, near-best configurations with different risk profiles, and strategies that were not apparent as alternatives at the outset. 

---
# CANNBot-CAKE: Collaborative Agentic Kernel Evolution for Ascend NPU 

**Authors**: Deng Pan, Xue-Jian Gao, Junyi Fan, Bin Du, Jiasheng Li, Xinxian Chen, Nok Yan Chang, Xinlin Li, Lingchao Zheng, Jun Li, Yuwei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31034)  

**Abstract**: Efficient custom operators are essential for modern deep learning workloads, yet developing high-performance kernels for Ascend NPUs remains slow, expert-intensive, and difficult to automate. Although AI agents now approach expert performance on GPU kernel benchmarks, current models remain less capable of generating AscendC kernels for NPUs, potentially reflecting the relative scarcity of relevant training data. We thus present CAKE, a CANN-native system for deployable custom operators that combines adaptive multi-DSL routing, separate Generation and Evolution modes, and reviewed knowledge reuse through CAKE Book and a Self-Evolving Agent. On 53 CANN-Bench v1.1 operators on Ascend 950PR, CAKE's aggregated results achieve $100\%$ case pass rate on the standard set (1,060/1,060) and $98.28\%$ on the hidden set (4,167/4,240), with an overall mean score of 78.63 out of 100 and an average performance speedup of $1.334\times$ over the baseline. Evolution then pushes correct operators further: EngramGateFusion cuts aggregate device time by $46.6\%$ over 20 rounds and roughly halves the mean roofline gap. These results offer a practical path toward closing Ascend's AI-for-kernel capability gap. 

---
# Evaluating Artificial Wisdom (AW): A Trajectory-Based Framework for Context-Sensitive Decision Regulation 

**Authors**: Md. Jalil Piran  

**Link**: [PDF](https://arxiv.org/pdf/2610.31012)  

**Abstract**: Capability benchmarks measure what large language models can do, but not whether their recommendations remain appropriate as decision conditions change. This paper proposes a trajectory-based measurement framework and prospective validation protocol for artificial wisdom, defined behaviorally as context-sensitive regulation of judgment. Versioned reference specifications determine admissible responses and transition materiality before evaluation. Seven stage-level criteria assess judgment under current conditions, while two transition components assess material-update adequacy and commitment-preserving stability. Stability requires prior adequacy; switching within registered commitment classes is permitted, and persistent error earns no stability credit. A nine-component profile is primary, with a secondary bottleneck score and separate critical-failure reporting. Three simplified simulations examine precision, bottleneck estimation, and sensitivity to model-family count. A single-criterion simulation met its precision criteria at 24 scenarios per domain. Near a bottleneck tie, the estimated minimum had a downward bias of 0.55 points at that size. A simplified incremental-prediction simulation detected a Brier improvement of 0.027 with probabilities of .508 and .742 using eight and twelve families, respectively; it does not establish power for the operational protocol. An exploratory predecessor-rubric study comprising 71 evaluation records illustrates aggregation sensitivity but does not validate the proposed instrument. The twelve-family target remains provisional pending feasibility assessment and calibration of the complete analysis. Whether the profile predicts independent decision outcomes beyond capability remains an open empirical question. 

---
# Mitigating Saliency Collapse: Robust Saliency-Aware Long-Text Image-Text Alignment 

**Authors**: Qiuyu Kong, Zanxi Ruan, Marco Cristani, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31005)  

**Abstract**: Long-text image-text alignment is essential for fine-grained scene understanding, yet existing methods mainly focus on extending context length or improving local alignment. We identify an overlooked failure mode, saliency collapse, where models over-rely on text that features visual prominence while underutilizing non-salient context. Consequently, retrieval performance degrades sharply when salient content is reduced, despite that the non-salient context remains informative. To address this, we propose RoSA, a saliency-aware finetuning framework that decomposes image-text pairs into complementary salient and contextual views, enabling joint learning of prominent and contextual semantics. We propose a modality-independent decomposition strategy to combine object-level visual saliency estimation with LLM-based text decomposition to bridge vision and language. We employ auxiliary objectives to explicitly align these views, ensuring robust representation with both salient and contextual features. RoSA greatly improves long-text retrieval and demonstrates strong robustness to saliency collapse across benchmarks. Moreover, our auxiliary objectives are plug-and-play, bringing consistent gains to existing tuning methods. Code and checkpoints are available at: this https URL 

---
# Agri-SCHOLAR: Simulation-Grounded In-Season Rolling Advisories for Agricultural Crops Using LLM-Bounded Optimization 

**Authors**: Geetha Charan, Vedant Balasubramaniam, Manojkumar Patil, V. Priyanka, Kodur Sai Vinay Sathvik, Rohit P Suresh, Mohammad Sufyan Azam, Y. Narahari  

**Link**: [PDF](https://arxiv.org/pdf/2610.31003)  

**Abstract**: Existing agricultural advisories to farmers suffer from several limitations: (a) remain largely expert-driven, (b) do not adequately account for in-season weather, and (c) do not scale to individual farms. In this paper, our objective is to close these gaps by developing Agri-SCHOLAR (Simulation-Grounded Crop Harvest Optimization with LLM-Guided In-Season Advisory Rollouts). Agri-SCHOLAR is a rolling, in-season advisory framework that issues, at intervals of every few days or every week, a personalized advisory using the following information: soil and crop properties, observed weather to date, short-range forecast of weather, and a season extrapolation of weather based on historical analogs. The novelty and depth of Agri-SCHOLAR lies in each advisory being produced by a closed-loop system which couples multiple LLM agents with a powerful crop simulator (such as APSIM in our case). The simulator crucially serves as an evaluation oracle. In Agri-SCHOLAR, the LLM agents propose agronomically sound and feasible bounds on actionable parameters and an optimizer searches within those bounds against a multi-objective reward spanning yield, resource utilization, and economic feasibility. We conduct a detailed 11-year analysis of the maize crop in two selected locations of South Asia and show that Agri-SCHOLAR consistently outperforms the region's standard packages of practice and satisfactorily addresses the limitations of existing agricultural advisories. 

---
# ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents 

**Authors**: Wenjie Zhang, Zhichao Shi, Xuhui Jiang, Xiaojun Wu, Cehao Yang, Chengjin Xu, Xiaoning Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.30989)  

**Abstract**: Personal agents need to draw on long-term interaction history to retrieve and apply external knowledge suited to a user's current request. Even an answer grounded in relevant documents can be unsuitable when it overlooks the user's circumstances. This creates two connected challenges: using historical conditions to identify the knowledge needed for a new request, and applying that knowledge to the user's situation. Existing benchmarks advance memory recall, personalized retrieval, and memory-guided action, while the joint evaluation of these two challenges remains underexplored. We introduce ColaBench, a synthetic benchmark in health and daily living, with 349 tasks across 10 Profiles and a knowledge base of 13,033 documents. Its Evidence Graph connects personal conditions to their disclosures across public interaction sessions, records prior document exposure, and links answer requirements to personal and newly acquired document evidence. We also introduce MemKB, which uses state summaries to inform queries and rule application, and memory-document links to retrieve knowledge associated with the user's history. MemKB achieves 86.25% Accuracy with an average of 4.07 interaction rounds per task. Task-level analysis finds larger gains on tasks integrating information across multiple historical Facts; error analysis further identifies unsupported personal assumptions and rule misapplication as obstacles to using retrieved evidence. ColaBench provides traceable tasks and an evaluation framework for studying how memory guides knowledge acquisition and rule application. 

---
# Zero-Shot Forecasting of In-Situ Soil Moisture and Temperature with a Time-Series Foundation Model 

**Authors**: Mohammad Talaei, Mohammad Goudarzi, Maille Todd, Vanessa N.L. Wong, Iadine Chadès  

**Link**: [PDF](https://arxiv.org/pdf/2610.30971)  

**Abstract**: Soil moisture and temperature govern plant-water availability and microbial activity, making them key indicators for evaluating land-restoration efforts on degraded landscapes. However, the physical and data-driven models normally used to forecast them both need information that a newly instrumented restoration site does not yet have: hydraulic parameters for the former and months of per-site training data for the latter. We evaluate TimesFM 1, a 500M-parameter time-series foundation model, zero-shot on 15-minute telemetry from 27 sensor stations across three land-management regimes (Control, Treatment, Scald) at a semi-arid restoration site, forecasting 5 cm and 30 cm soil moisture and enclosure temperature at seven lead times from 1 h to 168 h. Using gap-aware rolling-origin evaluation across sensors, we find that TimesFM beats the best rule-based statistical forecasting baselines (persistence, seasonal-naive, or 7-day climatology) by 21-76% at sub-daily lead times but loses by 6-79% beyond a day, a pattern explained by a diurnal-alignment mechanism that holds for both soil-moisture depths and temperature. A CatBoost baseline trained walk-forward on the same origins with full history and weather covariates never wins overall, and we show that its weakness is partly a cold-start effect specific to short deployments. We additionally quantify the inference cost of zero-shot forecasting on commodity CPU hardware. Our results highlight a practical trade-off between zero-shot forecasting capability, predictive performance, and computational efficiency. 

---
# Measuring the Reasoning-Action Coupling in Driving Vision-Language-Action Models through Counterfactuals 

**Authors**: Gabriel Manalu, Saad Wazir, Thao Do, Seongah Kim, Hyeontaek Hwang, Tolcha Yalew Kidane, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.30953)  

**Abstract**: Driving vision-language-action (VLA) models bring an appealing form of explainability to a safety-critical domain, since a system that can state the reasons for its maneuvers offers the transparency that deployment and public trust in autonomous driving demand. These models read the world into language before they act on it, describing the scene and reasoning over it in text that ranges from chain-of-thought to multi-stage question answering on the way to a trajectory. That framing quietly assumes two separable properties, namely that the trajectory follows the generated reasoning and that the generated reasoning tracks the causes that actually produce the trajectory. To thoroughly investigate these properties, we apply causal, counterfactual interventions to four driving VLAs across different benchmarks. First, we perturb the models' own reasoning and show that altering its meaning produces small trajectory changes relative to blanking the same models' camera inputs, a pattern that persists across benchmarks and model architectures. We then construct scene-level counterfactuals through photorealistic removal of key objects named in the models' reasoning and analyze whether the generated reasoning and trajectories track these changes. Overall, our findings indicate that the reasoning these models present to account for their resulting driving behavior is weakly coupled to that behavior itself, pointing to limits on the explainability these systems are intended to provide. 

---
# The Geometry of Meaning in Neural Manifolds 

**Authors**: Valentino Maiorca  

**Link**: [PDF](https://arxiv.org/pdf/2610.30938)  

**Abstract**: Just as different languages express the same concept through different words, neural networks express similar functions through incompatible coordinate systems. Different architectures, random initializations, and optimization trajectories produce incompatible parameterizations in extrinsic coordinates, obscuring functional relationships between independently trained models. When models independently solve similar tasks on related data, do their learned representations share exploitable structure despite incompatible coordinates? This thesis shows that semantic correspondence is a fundamental primitive revealing shared geometric structure beneath incompatible parameterizations. By identifying which elements refer to the same semantic content, correspondence breaks the symmetry of arbitrary implementation choices, anchoring incompatible systems to a shared semantic frame and revealing the intrinsic geometry that enables zero-shot transformation between independently trained models.
Building on this principle, fourteen contributions explore correspondence-based alignment across zero-shot model stitching, cross-modal translation, policy transfer, activation steering, and neural decoding. They span domains (vision, language, reinforcement learning, neuroscience), architectures (CNNs, Transformers, graph networks), and scales (from small supervised models to billion-parameter generative systems). Each exploits different geometric properties (angular, metric, spectral, sparse, or distributional), yet all instantiate the same principle. It extends beyond artificial systems: fMRI data show that different human brains processing identical stimuli exhibit alignable geometric structure.
By revealing intrinsic geometric structure beneath extrinsic parameterization, semantic correspondence transforms model isolation into interoperability. 

---
# DConstrain: Compiling Context-Parameterized Transitions for Diffusion Language Model Constrained Decoding 

**Authors**: Shidong Cao, Jing Ma, Wenhao Chai, Yuchen Tian, Lun Du, Da Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.30920)  

**Abstract**: Diffusion language models (DLMs) decode multiple tokens in parallel, so individually valid tokens can be incompatible together under a schema. Exact joint inference over a grammar graph enforces compatibility, but it requires token-labeled successors across many hypothetical parser configurations. Encoding stacks and field histories in compiled states can cause exponential growth, which forces prior exact decoders to fix property orders and exclude valid outputs. We present DConstrain, which compiles schemas into context-parameterized transition programs. Cold compilation specializes token consumption to local grammar states while retaining context-dependent stack operations, environment guards, and caller-dependent deferred parsing symbolically. Tracking field histories as runtime values supports arbitrary property order without enumerating field subsets offline. At runtime, warm compilation binds reached configurations to these programs to construct exact successors, overlapping with the model forward pass because graph construction requires no model scores. Furthermore, token-effect grouping merges tokens with identical compiled effects before runtime contexts are bound, so groups are reused across stacks and environments even when concrete successors differ. Objective-specific score aggregation and token restoration provably preserve both the Viterbi optimum and the token-labeled path distribution of ungrouped inference. Across three benchmarks and two DLM families, DConstrain compiles 100% of BFCL schemas under arbitrary property orders, whereas an optimized finite-state baseline compiles only 71.30-85.81%. Under arbitrary property order, it reduces BFCL end-to-end decoding latency by 6.65-46.26 times with 100% schema validity and raises DiffusionGemma's BFCL accuracy from 37.27% under fixed order to 81.56%. 

---
# Harnessing Language Models as Optimization Modeling Agents 

**Authors**: Yunxiang Luo, Xin Xu, Tianhao Chen, Zhixi Xiong, Peiqi Jiang, Fan Zhang, Wei Shao, Xuejian Gao, Yuwei Fan, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30917)  

**Abstract**: Large language models (LLMs) can translate natural-language optimization problems into solver programs, but reliably producing correct formulations and implementations remains challenging. Existing approaches improve optimization modeling through agentic workflows or task-specific post-training. This raises a practical question: can a lightweight harness using a single model role achieve competitive performance without additional training? We introduce OPTICYCLE and OPTISTATE, two lightweight inference harnesses for optimization modeling, and evaluate them on nine benchmarks comprising 2,634 problems using 11 Qwen variants spanning 0.6B-27B parameters. Our results show that execution-guided revision consistently improves macro accuracy over direct generation for all evaluated models, with gains of up to 49.46 percentage points. Despite their simplicity, both harnesses outperform six adapted agentic workflows in same-model comparisons and four evaluated post-trained optimization models at comparable model sizes, without additional training. Explicit formulation state provides further but model-dependent improvements, suggesting that additional structure is not universally beneficial. 

---
# TwinStore: A Synthetic Consumer Platform Using Vision--Language Models 

**Authors**: Minhyeok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.30909)  

**Abstract**: We present TwinStore, a source-available synthetic consumer platform for physical retail built on vision--language models. TwinStore turns a store into a digital twin---from 82 ready-made formats, a floor-plan image or phone photos of its shelves---and fills it with synthetic shoppers who plan a trip, walk the aisles and decide what to buy by looking at rendered shelves with packs, price tags and promotion signs. Every trip ends in a receipt, so a layout, price, pack or promotion can be tried on the same shoppers before it reaches the store. Around this core, TwinStore provides 3D playback, paired comparisons and layout optimization, shopper interviews, a trade-area model, counterfactual edits of real shelf photos, a marketing studio with eleven tools, and an assistant that turns a plain sentence into a drafted study or an answer grounded in a run's own numbers. In a held-out validation on eleven public datasets, simulated receipts approach the households' own natural variation (category-mix divergence 0.017--0.018 against a real drift of 0.012), price elasticities fall inside the range measured in the field, and three independent decision models agree. TwinStore runs on cloud or on-device models and is distributed with installers for Windows, macOS and Linux. 

---
# Beyond Phenotypic Similarity: Relation Refinement and Selective Population Graph Learning for Multimodal Brain Disorder Diagnosis 

**Authors**: Yinyu Lan  

**Link**: [PDF](https://arxiv.org/pdf/2610.30902)  

**Abstract**: Population graph learning supports brain disorder diagnosis by sharing information across subjects. Its effectiveness, however, depends on whether the relationships guiding this exchange are relevant to diagnosis. Similarity in demographic or acquisition attributes does not necessarily indicate diagnostic agreement, and heterogeneous subjects may require different representation updates within the same population graph. We propose PopDRP, a framework that combines diagnosis-guided relation refinement with hierarchical selective state-space learning. Using only training labels, Contrastive Affinity Refinement calibrates attribute-based affinities according to diagnostic agreement. Neural Graph Rectification then refines these relationships using fused functional-connectivity and non-imaging representations, guided by graph reconstruction and dual-view alignment objectives. On the resulting graph, a Selective Pyramid State-Space backbone integrates input-dependent state updates, graph diffusion, and hierarchical pooling and unpooling to learn population representations at multiple resolutions. Together, these components connect the refinement of inter-subject relationships with selective representation learning over the population graph. Experiments on ABIDE and ADHD-200 under stratified ten-fold cross-validation show improved accuracy and AUC over the compared baselines. Ablation studies further support the contributions of relation refinement and the selective hierarchical backbone to classification performance. 

---
# Parameter Learning in Active Inference via Variational Message Passing 

**Authors**: Benedikt Waldeck  

**Link**: [PDF](https://arxiv.org/pdf/2610.30899)  

**Abstract**: Active inference casts perception, learning and planning as variational inference, where the Expected Free Energy (EFE) defines the planning objective. Recent work showed that EFE minimization can be written as Variational Free Energy (VFE) minimization on a generative model augmented with epistemic priors, and that the resulting entropy-corrected objective is minimized by sum-product message passing (SP-MP) on modified factor functions. These formulations treat model uncertainty as inference over a finite set of known models. We extend them to planning during learning, where beliefs over the observation and transition parameters $\theta_A$, $\theta_B$ are Dirichlet priors. Closed-form Dirichlet updates are usually obtained by a structured mean-field factorization between parameters and states, but under exactly that factorization the information-gain corrections vanish. We resolve this by giving every factor its own parameter prior, frozen during a rollout. Conjugacy then makes each factor belief a product of a marginal over the categorical variables and a one-step Bayes posterior of the parameters. This reintroduces the datum-dependent parameter posteriors that mean-field factorization destroys. We show that evaluating the epistemic priors under the resulting Bethe joint recovers the information gains. Stationarity of this single closed-form objective yields SP-MP whose factor functions are augmented by the information-gain weight. Experiments on two grid POMDPs, in which only a planner that learns the model finds the safe route, show that the method plans to efficiently learn $\theta_A$ and $\theta_B$. It reaches the goal on every episode, whereas SP-MP planners with the same inference update plateau or take the risky route. 

---
# ADAPT for Science: A Multi-Agent System for Self-Evolving Scientific Optimization 

**Authors**: Zhaoran Feng, Yuchen Ni, Jianan Liu, Lemao Liu, Dejing Dou, Kairui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2610.30851)  

**Abstract**: Scientific discovery often requires identifying a small number of high-performing designs in vast, constrained search spaces. This is difficult when evaluations are costly, feasible solutions are sparse, parameters are strongly coupled, and physical constraints are strict. Recent AI for AI methods can automatically improve algorithms and optimization procedures. However, general purpose approaches are not designed around scientific priors, physical constraints, and limited evaluation budgets. We introduce ADAPT for Science, a multi-agent framework that treats scientific optimization as two coupled decisions: where to search and how to search. ADAPT uses historical designs to identify promising and diverse initial conditions, progressively concentrates computation on favorable search directions, and evolves the optimization procedure from execution feedback. We evaluate ADAPT on the design of gravitational wave detectors using a differentiable physics simulator and eight unseen detector topologies. Under matched scientific evaluation budgets, ADAPT achieves a mean loss of 0.204 +/- 0.101, reducing loss by 54.9% relative to the best classical optimizer and by 62.2% relative to the best general purpose agentic evolution baseline. ADAPT achieves the lowest loss on all unseen topologies among the evaluated baselines. ADAPT provides a step toward scientific optimization systems that continuously learn from prior experience and their own search outcomes. 

---
# Semi-Linear Gaussian Bayesian Networks: Enhancing Flexibility in Node Dependencies via penalized B-Splines 

**Authors**: Aitor Oviedo-Madrid, Mabel Morales-Otero, Jose Gonzalez-Gomariz, Ruben Armañanzas  

**Link**: [PDF](https://arxiv.org/pdf/2610.30838)  

**Abstract**: Bayesian networks for continuous variables often rely on restrictive assumptions: linear Gaussian models lack flexibility to capture nonlinear dependencies, while non-parametric approaches provide greater flexibility at a high computational cost. This creates a gap for modeling scenarios where variables are approximately Gaussian but exhibit nonlinear relationships. We introduce Semi-Linear Gaussian Bayesian Networks (SLGBN), a novel class of continuous Bayesian networks designed to bridge this gap, preserving Gaussian conditional distributions while introducing nonlinear conditional mean functions. The proposed framework provides a unified probabilistic framework combining linear Gaussian and nonlinear Gaussian conditional models within the same Bayesian network Nonlinear conditional mean functions are estimated using generalized additive models (GAMs) with penalized B-splines. We present parameter estimation procedures for both types of nodes, as well as a greedy hill-climbing algorithm for learning the network structure under the SLGBN framework. To assess its performance, we compare SLGBN with other continuous Bayesian network models, including semiparametric Bayesian networks (SPBN) and Linear Gaussian Bayesian networks (LGBN). Experimental results evaluate both goodness of fit and the ability to recover the underlying network structure. The results show that, in the presence of nonlinear Gaussian relationships, SLGBN achieves better fit and improved structure recovery compared to alternative models, particularly when combined with the Extended Bayesian Information Criterion (EBIC*) score. Furthermore, by retaining Gaussian conditional distributions, SLGBN offers computational advantages over fully semiparametric approaches. 

---
# TESSA: A Governance-Oriented Custom GPT for SME Guidance under the EU AI Act: Design, Implementation, and Evaluation 

**Authors**: Chalisa Veesommai Sillberga, Kai-Kristian Kemella, Pekka Sillberga, Katri Harjuveteläinenb, Mika Saaria, Pekka Abrahamssona  

**Link**: [PDF](https://arxiv.org/pdf/2610.30822)  

**Abstract**: The increasing adoption of generative AI presents new opportunities while raising AI governance and regulatory challenges, particularly for small and medium-sized enterprises (SMEs) navigating the European Union Artificial Intelligence Act (EU AI Act). To support AI governance education, this study developed TESSA, a governance-oriented educational artifact within the OpenAI Custom GPT environment. Following a design science research approach, TESSA was evaluated against baseline ChatGPT using 100 benchmark scenarios through governance alignment assessment, human expert review, and consistency testing. Preliminary LLM-assisted assessment showed governance alignment rates of 90% for TESSA and 83% for baseline ChatGPT, while independent validation by an AI regulation law practitioner yielded 71% and 70%, respectively. Human expert review indicated that both systems provided generally strong governance guidance, while TESSA maintained high consistency across repeated executions. The findings suggest that governance-oriented grounding can support structured, practical, and stable AI-assisted compliance guidance for SMEs. The study contributes (i) a governance-oriented Custom GPT artifact for AI governance education, (ii) a human-validated regulatory update mechanism, and (iii) a multidimensional evaluation framework for governance-oriented AI artifact. 

---
# What Lies Beyond Perception: A Partition Theory of the Limits of AI Evaluation and Oversight 

**Authors**: Ahmed M. Mahran  

**Link**: [PDF](https://arxiv.org/pdf/2610.30803)  

**Abstract**: Replacing the judge can move a benchmark score, and models can be trained to hide capabilities. Both put the evaluator in question: what can it certify? We develop a partition theory of evaluation from one principle: judgement is bounded by perception. An agent's discrimination is the partition of its inputs into classes it cannot tell apart. Discrimination is necessary for capability but not sufficient, so bounds on it bear directly on whether evaluation can show that a model lacks an ability. We prove a one-sided bound. An examinee whose partition refines its evaluator's has the discrimination to pass every test the evaluator can grade, provided each input has an acceptable reply. We define the echo test, which accepts any reply the evaluator perceives as identical to the input, and show that it is complete for the tests the evaluator can grade. Passing therefore certifies at most a floor on discrimination, never a ceiling, and only a sharper evaluator can raise it. Used to detect concealment, echo-test failure points the wrong way: under exhaustive probing it flags every honestly weaker examinee and no concealer that presents as the evaluator's peer. Exact imitation of a weaker model is undetectable by any behavioural evaluator. We extend the bound to ensembles of monitors and to channels a model controls, explain published findings on judge disagreement and sandbagging, and derive testable predictions. A safety case that argues from behavioural evaluation alone that a model lacks an ability must assume the model's cooperation. 

---
# Agentic Knowledge-Model Co-Evolution via Consistency Regularization 

**Authors**: Sirui Chen, Zhining Liu, Chao Zhang, Jingrui He, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30790)  

**Abstract**: LLM agents can evolve by learning from accumulated experience. Existing approaches extract reusable external knowledge, such as skills and memories, from experience to guide subsequent task solving, with some further internalizing this knowledge through model training. However, different ways of analyzing and transforming the same experience can yield knowledge with substantially different utility, even when the resulting guidance appears reasonable. Inspired by consistency regularization, we identify consistent utility gains across different ways of analyzing the same experience, rather than relying on any single way of analyzing it. We introduce SPLICE, a framework for consistency-regularized knowledge-model co-evolution. It constructs evolution views by perturbing how the same experience is analyzed, producing different knowledge candidates. Knowledge updates are then guided by common gain, which accounts for utility gains and their consistency across views. During model evolution, self-distillation internalizes the evolved knowledge alongside reinforcement learning, using knowledge associated with consistent utility degradation as a contrastive reference. The two processes interleave, with model updates driving further knowledge evolution. Experiments across diverse domains show that SPLICE outperforms knowledge evolution and model training baselines by up to 7.1 and 4.6 points in average accuracy, while its evolved knowledge reduces output tokens per correct solution by up to 44% and transfers across model sizes and families. 

---
# An Introduction to World Models for Generalist Agents 

**Authors**: Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30789)  

**Abstract**: World models have recently emerged as a promising direction toward developing generalist agents capable of understanding, predicting, and interacting with diverse environments. However, despite growing interest, the concept of a world model remains loosely defined, encompassing approaches ranging from predictive dynamics models and learned simulators to models that jointly generate future observations and actions. This introductory paper aims to clarify the concept of world models and provide a unified perspective on their role in developing generalist agents. Starting from the partially observable Markov decision process (POMDP) framework, we connect world modeling to classical control, dynamical systems, signal processing, and reinforcement learning. We introduce different approaches to world modeling, emphasizing the knowledge they capture, their learning objectives, and how they support decision-making. We further explore the relationship between world modeling and policy learning, highlighting how these capabilities can be unified through conditional prediction without requiring separate models. Rather than offering an exhaustive survey of existing methods, this paper focuses on fundamental concepts, modeling principles, and connections, with the goal of providing an accessible and unified introduction to world models for generalist agents. 

---
# ProQuant: Preserving Expert Quality over Quantity in Ultra-Low-Bit Quantization of Mixture-of-Experts Models 

**Authors**: Jiaqi Chen, Xianglong Yan, Chengzhu Bao, Xie Chen, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30779)  

**Abstract**: Mixture-of-Experts (MoE) large language models reduce computation through sparse activation but still incur substantial storage costs. Mixed-precision post-training quantization (PTQ) reduces these costs by quantizing different model components at different bit widths, but existing methods often struggle to preserve generation quality under tight bit budgets. We identify two key challenges: (1) nonlinear quantization errors at extremely low precision are underestimated when cross-layer propagation is ignored; (2) direct router fine-tuning may cause routing drift, and calibration gains may not generalize to downstream tasks. To address these issues, we propose ProQuant, a weight-only PTQ framework for MoE models. Prune-or-Prioritize Mixed-Precision Allocation (PoP) jointly optimizes expert pruning and bit-width assignment. It allows higher-precision experts to substitute for severely degraded ones while improving bit-budget utilization. Routing-Consistent RMSNorm Finetune (RoCo) adapts expert input representations through mathematically equivalent transformations. It compensates for quantization errors to reduce routing drift. Extensive experiments demonstrate that ProQuant outperforms existing methods across diverse generative tasks. Code is available at this https URL. 

---
# LLM Multi-Agent Systems Are Vulnerable to Herding and Runaway Cascades 

**Authors**: Farnoosh Hashemi, Shiyu Ji, Joice Chen, Michael W. Macy  

**Link**: [PDF](https://arxiv.org/pdf/2610.30769)  

**Abstract**: Large language models (LLMs) increasingly operate in multi-agent systems where they collaborate by responding to others' choices, creating endogenous feedback loops with vulnerability to social influence, herding behavior and runaway cascades. Following the design of the classic Music Lab experiment, we tested this vulnerability in large populations of LLM agents that recommended unfamiliar songs, with or without exposure to cumulative recommendation counts. Exposure led to runaway cascades that generated extreme and unpredictable collective recommendations increasingly divergent from a song's intrinsic appeal. Compared to humans, LLM populations were generally far more susceptible to herding behavior, and unlike humans, LLM cascades were less able to self-correct. This vulnerability has implications for multi-agent collaboration in recommender systems, automated evaluation, complex problem-solving, and unauthorized conspiracies. 

---
# Scaling Laws for Generative Reward Models 

**Authors**: Angel Raychev, Nathan Lile, Rafael Rafailov  

**Link**: [PDF](https://arxiv.org/pdf/2610.30765)  

**Abstract**: We study generative reward models (GenRMs) as online preference providers for policy optimization in creative writing. Across Qwen3 reward-model sizes from 0.6B to 14B, Thinking judges trained with GRPO have higher static Gold-label accuracy than Answer-Only judges trained with SFT. This ranking often reverses when the judges train policies through online DPO: Answer-Only rewarders yield higher peak and top-five-average policy Elo in seven of eight matched settings under a fixed, learned 32B Gold evaluator. Intermediate Thinking-judge checkpoints can also outperform later, more statically accurate checkpoints as rewarders. Sweeps over rewarder size, policy size, training duration and DPO regularization show substantial benefits from reward-model capacity, including judges larger than the policy. We analyze trajectories with evaluator-specific Elo arenas, conditional match-resampling intervals and descriptive scaling fits. These results demonstrate an evaluator-rewarder gap in the studied pipelines: fixed-distribution judging accuracy does not determine downstream optimization quality. The comparison changes judge-training objective as well as response format, and the main ranking is conditional on its learned reference rather than a universal human-quality result. A limited follow-up using a different learned judge is inconclusive overall. 

---
# AgentDP: Towards Automated Model Deployment for Heterogeneous Embedded AI Applications 

**Authors**: Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.30760)  

**Abstract**: This paper presents AgentDP, which aims to deploy pre-trained AI models on heterogeneous embedded devices by fully automating model optimization and adaptation (O&A) with Large Language Model (LLM) agents. AgentDP not only significantly reduces human effort via automation but also enhances model inference efficacy by integrating LLMs with device-specific domain knowledge. To deliver a robust automation framework, AgentDP develops novel solutions to address three key technical challenges: 1) To help users elicit clear requirements, we propose a user requirement standardization module; 2) To mitigate the execution environment mismatch between AI models and heterogeneous embedded devices, we design an operator-aware model partition paradigm that enables different parts of the model to be efficiently executed on distinct inference engines. 3) To avoid re-searching optimization hyperparameters for new tasks, we propose a configuration reuse module that heuristically retrieves previous hyperparameter configurations for efficient reuse. Extensive experiments demonstrate that by fully automating the entire O&A process, AgentDP reduces entire model adaptation time by up to 80% compared with manual O&A while achieving higher task accuracy (more than 13% on average) than the baselines. Additionally, a comprehensive user study also exhibits the robust generalizability of AgentDP in adapting to different user requirements for heterogeneous embedded devices. 

---
# Can Loan Processors and Underwriters Spot AI-Edited Statements? An Empirical Study 

**Authors**: Simiao Ren, Xingyu Shen, Tommy Duong, Ankit Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.30758)  

**Abstract**: AI-assisted editing can produce loan documents that preserve familiar issuer layouts while changing the applicant's identity. We investigated whether experienced underwriters and loan processors identify such forgeries in first-pass, document-only review. Three practitioners recruited through Upwork each received one applicant package drawn from 134 forged PDFs derived from 47 genuine financial documents supplied with full consent. The edits preserved amounts, dates and balances; identifier inconsistencies introduced by the pipeline were knowingly retained before distribution. Reviewers were not told that files had been altered and returned 132 file-level judgements. Checking their explanations against the originals yielded 14 edit-supported concerns (10.6%): 12 Suspicious judgements and two Insufficient information judgements. All supported concerns involved identifier disagreements within or across documents, and no reported visual cue identified an actual edit. Thus, approximately one in ten judgements contained evidence of the forgery, despite known inconsistencies remaining in the files. These observations motivate evaluating consistency checks and independent source verification alongside human review. With three reviewer-package pairs and no genuine-file controls, the findings describe this exploratory setting rather than population-wide detection accuracy. 

---
# Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought 

**Authors**: Zhengye Bao, Shengtao Zhang, Haokai Zhang, Shiqi Lou, Muning Wen, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2610.30755)  

**Abstract**: LLM memory is widely used in long-term conversation understanding. Agentic memory systems often use mutation, structural organization, learned retrieval strategies, or other complex operations. Is this complexity always necessary? We study Raw-RAG, a naive RAG baseline using dense retrieval and reranking over raw history to supply evidence to a reader LLM. Across LoCoMo, LongMemEval, MEME, and MemoryAgentBench, Raw-RAG ranks first among the compared memory systems in six of seven benchmark-model settings, outperforming the strongest compared agentic memory system by 0.13-5.53 points. We investigate why raw-history retrieval remains competitive. Evidence analysis shows that its higher coverage of necessary evidence can offset lower utilization. Useful evidence includes both answer values and the relations and dependency paths needed to derive them. Even complete access to transformed memory stores leaves an accuracy gap relative to raw evidence, highlighting the importance of preserving raw information. These findings challenge the necessity of complex agentic memory for the evaluated conversation tasks and establish naive RAG as an essential baseline. 

---
# How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models 

**Authors**: Simiao Ren, Xingyu Shen, Tommy Duong, Tsang Ng, Ankit Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.30751)  

**Abstract**: Can a small change in wording move an AI model from refusing a sensitive request to acting on it? We test 15 models on the same request to increase the income shown on a pay stub. Across 445 observable trials, models issue editing calls in 14.1% of cases when a loan purpose is stated, 33.1% when the purpose is omitted, and 66.2% when the change is described as fixing a printing error. Astra refuses all ten loan-purpose requests but issues calls in all ten purpose-omitted trials. Fable, Opus, and Muse Spark issue none across all three wordings; Sonnet issues some under the latter two. Response excerpts illustrate different ways of refusing, redirecting, or justifying an edit. The file is a marked test document and all tools are simulated. These results probe how boundaries behave in one setting; they do not prove an overall ethics ranking or that a harmful real-world edit succeeded. 

---
# AgentTriage: Replayable Security Diagnosis Across Heterogeneous Agent Runtimes 

**Authors**: Jie Ying, Xutao Zhang, Ningyuan Chen, Mengce Zheng, Jun Li, Zhongjie Zhu, Tiantian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30738)  

**Abstract**: When tool-use agents write files, execute code, or send data, their final responses do not explain the resulting security events, and heterogeneous runtimes expose the relevant evidence through incompatible interfaces. We present AgentTriage, an executable contract for replayable security diagnosis across heterogeneous runtimes. The contract admits conclusions whose claims cite canonical runtime events and requires abstention when mandatory evidence is missing. Runtime overlays normalize native observations, while a deterministic pipeline joins invocation and result lifecycles, constructs a typed provenance graph, selects a policy-decisive slice, and binds claims to recorded evidence. Given the same normalized trace, overlay version, and pre-run policy oracle, the diagnosis can be recomputed and its claims re-checked without language-model judgment. Across five runtimes, one contract captures all 159 applicable executions, with an audit of native and synthesized identifiers. On 120 adjudicated cases, AgentTriage achieves 0.858 grounded joint diagnosis, 0.966 required-evidence coverage, and a 0.036 unsupported-claim rate, improving joint diagnosis by 0.175 (95% CI [0.092, 0.258]) over the strongest same-schema baseline. These results quantify the precision of replayable diagnosis and the observation boundaries that require abstention. We release an open-source implementation and selected sanitized evidence at this https URL. 

---
# Faithful Latent Harm Monitors Work No Better Than the Models Own Behaviour at a Matched False Alarm Rate 

**Authors**: Abhishek Mishra, Sagnik Chatterjee, Suyash Maniyar  

**Link**: [PDF](https://arxiv.org/pdf/2610.30727)  

**Abstract**: Large language models have been reported to encode the harmfulness of a request separately from their decision to refuse it, and latent safety monitors such as Latent Guard build on this report by classifying prompts as harmful from the internal activations of the model, with the aim of catching harmful intent even when the model complies. Whether these representations give a signal of harm that survives distribution shift and is independent of model behaviour is unclear. We test this on out-of-distribution prompts made by rewriting existing harmful and benign requests with surface rewording, persuasive framing and jailbreak-style wrappers. Across 15 instruction-tuned LLMs and 38 activation-based readers, we compare harm detection with that of TF-IDF and sentence-embedding classifiers, with independently trained safety classifiers, and with the refusal of each model. Activations are read at the last token of the user instruction, where the model has read the whole request, and at the last chat-template token before the reply, where the planned response can be represented. Because models disagree about which prompts to refuse, we can separate a signal of the harmfulness of the prompt from a signal of the refusal decision of the model. At the end of the instruction, the representation is largely independent of the refusal behaviour of each model but degrades under distribution shift, so that a TF-IDF classifier ranks harm better than the original Latent Guard in every model we test, and linear and MLP probes do about as well as sentence-embedding classifiers trained on the same examples. The representation just before generation detects more, catching 44 percent of harmful prompts at a matched false-alarm rate against 32 percent for the refusal of the model, but that gain follows the refusal decision itself, and part of it comes from refusals that string-matching evaluation misses. 

---
# Governance-State Corruption in Long-Term Memory for LLM Agents: A Cross-Model Stress Test 

**Authors**: Bowen Liu, Youting Wang, Bonan Shen, Dingyan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30698)  

**Abstract**: An LLM agent can retrieve a memory correctly yet use it incorrectly after its validity state changes. We study governance-state corruption: faults that alter which retained records enter model-visible context while stored content and tasks remain fixed. Across six providers, 731 benchmark-grounded bases yield 2,080 matched fault pairs. In offline control-plane checks, corruption changes the operative projection in 96.25% of pairs, while oracle-backed enforcement restores it in 100%. For downstream evaluation, a deterministically selected, non-probability LongMemEval/Memora panel resolves 298 of 300 matched triplets (238 distinct bases). Clean-minus-corrupted contrasts are 40.54 percentage points in LongMemEval binary correctness (95% base-cluster interval [30.32, 50.71]) and 30.60 points in rescaled Memora FAMA ([23.31, 37.69]); both sources also show recovery. OAKS Novel native option parity holds for all 6,714 resolved outputs. Full-cohort frozen-scorer corruption/recovery effects (-27.55/+27.51 points) remain descriptive proxies, not validated native semantic population effects. These bounded findings motivate evaluating memory control-state integrity alongside retrieval accuracy. 

---
# Disentangling Robustness and Sycophancy in Multilingual LLMs 

**Authors**: Rhitabrat Pokharel, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2610.30696)  

**Abstract**: When a multilingual LLM agrees with a user's false premise, is it being sycophantic, or does it simply not know better in that language? Current research does not distinguish these cases: they record only whether the model agreed, not why. We introduce the SC-matrix, that crosses a model's competence (whether it answers correctly under no pressure) with its sycophancy (whether it maintains that answer under pressure). Our results show that (1) identical aggregate sycophancy rates can mask different failure compositions, (2) high competence in a language does not always protect against sycophancy, and (3) reasoning-oriented mitigations outperform directive-based strategies. We find that multilingual sycophancy is not a single phenomenon but a composite of different failures that require distinct diagnosis. 

---
# Scaling Laws for Safety Probes 

**Authors**: Yuchen Wu, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.30678)  

**Abstract**: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, defined as 1 - AUROC, where lower values are better. We find that probe error follows two simple scaling laws. First, the lowest probe error a model can reach falls as a power law in its size. Second, probe error decreases towards its floor as the number of examples increases; the slope increases as the target model size increases. Both laws hold across tasks, model families and probe architectures, and only their fitted constants differ. When fitted on the smallest models of a family, the laws predict the error of the next larger model to within 0.02 in 72% of cases. They also identify whether more labels or a larger model is the cheaper way to reach a target error in 73% of cases. Finally, they choose the probe architecture with an extra error of only 0.006 on average when they pick the worse one. These laws allow safety researchers to determine how to create the best probe for any given model and label budget. 

---
# Group Relative Policy Optimisation Improves Multilingual Speech Recognition in Low-Resource Languages 

**Authors**: Tim Wenjie Hu, Evelyn Nafula Ouma, Benjamin Akera, John A. Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2610.30663)  

**Abstract**: Autoregressive encoder-decoder speech recognition models trained with teacher forcing can suffer from exposure bias, limiting their ability to recover from their own prediction errors at inference time. Recent work has shown that reinforcement learning with Group Relative Policy Optimisation (GRPO) improves speech recognition in major languages with LLM-based models. We extend this line of work to smaller, non-LLM encoder-decoder ASR models, using Whisper-large-v3 as a case study across 51 African languages. The benefits of this approach are most pronounced when training data is scarce or when input audio is noisy or degraded. Our resulting model, SunflowerASR, achieves the best published macro-averaged WER on the AfriVox-v2 (0.227) and SimbaBench (0.372) benchmarks, and is publicly available. 

---
# A Topology-Guided GeoAI Agentic Framework for Urban Flood Screening 

**Authors**: Yutian Tao, Haoan Feng, Xin Xu, Leila De Floriani  

**Link**: [PDF](https://arxiv.org/pdf/2610.30660)  

**Abstract**: Open-world geographic artificial intelligence (GeoAI) agents can transform public geospatial data into accessible decision support for real-world challenges, but must balance autonomous reasoning with reliable spatial analysis and human oversight. We present a topology-guided GeoAI agentic framework for urban flood screening that takes a place name, drawn region, or uploaded boundary and returns ranked candidate basins with an interactive map and report. A deterministic controller retrieves public elevation, rainfall, soil, land-cover, exposure, and infrastructure layers, extracts terrain depressions with discrete Morse theory, and ranks basins with a relative flood-priority score. The language model acts only at recorded decision points: selecting a persistence threshold from a precomputed candidate table, drafting recommendations from computed evidence, and reviewing each recommendation against the same evidence. Three human-in-the-loop checkpoints let users revise the threshold, factor weights, or recommendation. In a College Park, Maryland case study, the selected threshold reduced 278 raw terrain minima to 70 basins, with 77% of these basins' lowest points also identified as sinks by a separately computed eight-direction (D8) flow-routing model. Across four further study areas, agreement ranged from 61% to 83%. By combining open data, explicit spatial reasoning, traceable evidence, and human oversight, the framework enables transparent first-pass flood screening while keeping consequential recommendations inspectable and user-controlled. 

---
# Anticipating Social Consequences with Distributional World Models 

**Authors**: Viet-Thanh Pham, Thuy-Trang Vu, Zhuang Li, Lizhen Qu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2610.30657)  

**Abstract**: Social world models predict how interactions develop after an action, helping agents negotiate, cooperate and mediate. In these situations, one action can lead to several plausible futures, hence predicting only one future can overlook consequential alternatives, while exploring them through repeated simulations increases inference cost. We propose DSWM, a Distributional Social World Model that predicts alternative responses, their probabilities and immediate and final outcome values in one output per candidate action. We train a 4B model on repeated simulations using supervised fine-tuning and GRPO. At inference time, the actor uses these possible consequences to choose an action. Across three seen and two unseen environments, with interaction backbones absent from training collection, DSWM achieves the best average performance on 13 of 15 metrics. Ablations show that retaining multiple futures and their predicted probabilities improves outcomes in all five environments. 

---
# Speculative Acceptance Does Not Decay with Position: A Full-Length Diagnostic on Long Chain-of-Thought with Independent Drafts 

**Authors**: Zhiming Meng  

**Link**: [PDF](https://arxiv.org/pdf/2610.30644)  

**Abstract**: The claim that speculative-decoding acceptance rate decays as a chain-of-thought grows longer has become a de facto industry narrative, motivating draft-model co-design and position-aware training. Nearly all direct evidence comes from trained drafters whose offline training length is far shorter than inference, so a purely positional law has never been isolated. We present the first systematic, full-length measurement of acceptance rate as a function of generation position on MATH-500 (all 500 problems), under greedy and sampling decoding, with two independent drafts against DeepSeek-R1-Distill-Qwen-7B: a same-family draft (DeepSeek-R1-Distill-Qwen-1.5B) and a cross-family draft (Qwen2.5-1.5B-Instruct), yielding about 1.78M verification rounds. Using within-problem paired tests, we find that after isolating a greedy-decoding repetition pathology, acceptance rate does not decay with position in any of the four configurations within the supported ranges (greedy 0-8K, sampling 0-12K); the second half of each trajectory is slightly but significantly higher than the first. A controlled temperature sweep shows that the apparent sampling-mode degradation of high-entropy reflection tokens is a visibility artifact of argmax agreement, not a drop in expected acceptance probability. Rejections concentrate with high precision on reflection trigger tokens such as "wait", "however" and "check", an entropy-selective failure that a lightweight three-arm LoRA trajectory-alignment intervention (uniform, reflection-marked, entropy-weighted) fails to improve in mean accepted length. We thus supply the positional baseline that independent draft-target pairs were missing, and localize the residual failure to decision-point entropy rather than position. 

---
# Large language models show systematic failures in incorporating evolving medical knowledge 

**Authors**: Min Zeng, Zaifu Zhan, Kai Yu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30631)  

**Abstract**: Medical knowledge evolves as clinical guidelines revise recommendations, yet whether large language models (LLMs) faithfully incorporate such changes remains unclear. We introduce ChronoEvo, comprising 1,709 recommendation-level revisions across 12 forms of medical knowledge change, to distinguish recognition of revised guidance from incorporation into clinical recommendations. In the primary model, 90.8% of responses recognized revisions and 80.2% were clinically correct, but only 58.5% fully incorporated guidance, yielding a 32.3-percentage-point recognition-incorporation gap. This dissociation persisted across 14 models and was corroborated by two clinicians' adjudicated assessment of 300 responses, yielding a 40.3-percentage-point gap in the validation sample. Changes in recommendation strength and patient eligibility were associated with retention of outdated guidance. A revision-salience intervention reduced outdated-content retention and improved clinical correctness but did not significantly improve full incorporation. These findings identify faithful incorporation of evolving medical knowledge as a distinct challenge for clinical LLM reliability, even when updated guideline evidence is available. 

---
# Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference 

**Authors**: Amirhossein Aghaei, Liangyan Li, Shiyi Yang, Shaohua Wang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30611)  

**Abstract**: Scaling large language models has substantially improved linguistic fluency, yet reliable autonomous agents require more than fluent generation. In multi-turn inference, current agentic systems often produce semantically plausible but structurally ungrounded reasoning, especially under incomplete observations. We propose Structure-Grounded Uncertainty Reduction (SGUR), a training-free framework that complements token-level generation with active uncertainty reduction over structured hypothesis spaces. SGUR represents incomplete observations as a query graph and aligns them with candidate hypothesis graphs through a prior-weighted partial graph matching mechanism, where database-derived empirical priors assign greater importance to decision-relevant relations rather than treating all graph edges uniformly. The resulting inference state is iteratively refined by an active information-gain strategy that selects the next most informative node to query, systematically pruning the candidate space. We instantiate SGUR on a challenging PrimeKG-derived sequential medical diagnosis benchmark, demonstrating that structural grounding can partially substitute for model scale: starting from only two observed symptoms, SGUR achieves 92.5% Hit@3, outperforming the strongest Graph-RAG baseline by 9.4 percentage points while using approximately 5x fewer tokens. Notably, SGUR enables smaller backbone models to match or exceed the performance of models up to 10x larger, suggesting that structural grounding provides a more robust and efficient paradigm for scaling agent performance than relying exclusively on increasing model size. 

---
# Model Organisms of Collusion: Corrective Transfer in Untrusted Monitors 

**Authors**: Juan-Pablo Rivera, Ari Spiesberger, Juan J Vazquez, Arun Jose, Axel Ahlqvist, Adeline Kassler, Meru Gopalan, Sandy Tanwisuth, John Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2610.30582)  

**Abstract**: AI control protocols aim to get useful work from models that may be misaligned by monitoring their behaviour. One approach uses a second copy of the same model as the monitor. This avoids relying on a weaker reviewer the model could outsmart, but risks collusion: the model, now monitoring itself, may cover up its own misbehaviour. One remedy is to fine-tune the monitor on whatever examples of misbehaviour the control team has caught. We ask whether this training also removes collusion that the control team has not yet found. We fine-tune Llama-3.3-70B and Qwen2.5-72B to install five policy-monitor behaviour pairs, correct the monitor on 300 honest reviews of one caught collusion, and measure collusion on the other four in unseen prompt formats. Averaged over the three seeds of each family, correction reduces held-out collusion from 91% to 14-20% under three of the four methods we compare (individual seeds 10-32%), while a control fine-tuned on the same number of unrelated examples leaves 56-93%. Family-average accuracy on control items changes by at most five points and capability is preserved, apart from a drop in mathematical accuracy on one Llama seed. Under the tested training configuration, methods with similar caught-task accuracy differ in held-out transfer, from 58% to 83% average reduction, so the caught behaviour alone does not reveal how far a correction reaches. Catching one collusion lets the control team reduce, but not remove, collusion it has not caught, by retraining the monitor alone. 

---
# Diagnosing Personalization Failures in Mobile GUI Agents 

**Authors**: Zhixin Lin, Dongliang Xu, Jungang Li, Shidong Pan, Jun Feng, Yanbin Sun, Jing Li, Liang Liu, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30573)  

**Abstract**: Unlike conventional GUI agents, personalized GUI agents require user preferences to guide sequential execution over changing GUI states. This raises two challenges: determining HOW each preference should alter execution strategy and WHEN it should take effect along the interaction trajectory. Existing benchmarks primarily evaluate overall task success, offering limited insight into these questions. We introduce PrefDiag-Bench with two components: (1) a categorization of preferences into five types, characterizing HOW they affect execution; and (2) a stage-wise diagnostic framework with 1,664 probes, tracing WHEN preferences should take effect and where failures occur. We evaluate eleven GUI agent systems, including end-to-end agents and agentic workflows, under two forms of user context: explicit user profiles and behavioral memory logs. Although existing agents achieve relatively high completion rates on non-personalized tasks, performance is substantially lower on preference-personalized tasks. Even the best-performing model, Seed-2.0-Pro, achieves only 47.17% on Process preferences under explicit user profiles. Agents struggle particularly with Ranking and Process preferences. Further analysis reveals two key capability bottlenecks: selecting relevant evidence from user history and recognizing when preferences should take effect during execution. Controlled interventions that explicitly provide relevant evidence improve end-to-end task success by up to 7.06%. PrefDiag-Bench offers a more systematic evaluation of GUI agents' personalization capabilities beyond overall task success. Code and data are available at this https URL. 

---
# Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core 

**Authors**: Yebo Feng, Jingyang Dai, Liangrui Pan, Nelson Lau, Jiahua Xu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30557)  

**Abstract**: LLM agents that act on live Web data must combine heterogeneous evidence, remember what they have done and improve from outcomes without weakening their own safeguards. We study this problem in autonomous investing, where every action is irreversible and based on untrusted Web content. We present Steward, a multi-agent LLM system that draws the boundary of self-evolution in its type system. First, nine role-specialised agents decide jointly: four analysts read news feeds, SEC filings, macroeconomic series and market data and emit typed reports, from which a Portfolio Manager sets target positions that a protected Risk Manager audits. Second, a seven-layer shared memory, assembled by code from the system's own record, gives each decision its position context, the market narrative, calibration and the agent's past decisions. Third, the system improves itself by review: code grades every trade and forecast nightly and writes the grades into memory, and each quarter a reflector proposes prompt learnings that a deterministic gate applies reversibly and replay of recorded inputs verifies. All three operate around an invariant core of order arithmetic, risk rules and discipline-agent prompts that no self-edit can reach. In a five-month live deployment (102 trading days, 1,632 LLM calls), the system contained a 24.6% sector drawdown to a 0.3% loss; in replay of its recorded inputs, a single model given the same raw data traded more and lost 1.35 points against the market over five days where the joint design showed no significant loss, and no injected instruction was followed; and a 2015-2026 backtest shows the core alone lowering the maximum drawdown under all 50 random entry policies. The reflector traced the dominant failure to its root cause, and the gate applied a learning that the agent adopted on trigger evenings and on no control evening. Code: this https URL 

---
# PLE: Physical-Engineering's Last Exam 

**Authors**: Rujun Han, Yufan Zhuang, Arijit Ray, Han Yu, Zoey CuiZhu, George Lee, Yifei Ming, Binz Roy, Varun Sakalkar, Vishy Tirumalashetty, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.30537)  

**Abstract**: Frontier language models now score above 90% on popular knowledge benchmarks and are increasingly deployed as agents that carry out multi-hour professional work. Yet no existing benchmark tests whether they can perform the real-world, economically valuable work of a professional physical engineer, such as modelling physics and legal code requirements across a multi-format project corpus, optimising material and installation costs, and issuing 2D/3D engineering design packages that a licensed reviewer would sign off on. Existing engineering benchmarks focus on textbook question answering, isolated tool calls or self-contained toy design problems. As a result, most of these benchmarks are already saturated by leading agents and solving them is far away from delivering real engineering design. We introduce PLE, a benchmark of 85 long-horizon, agentic tasks in professional physical engineering consisting of electrical, civil, structural and mechanical engineering, where solving a task means completing a realistic, billable engineering deliverable. PLE combines two complementary curation paths. Expert-Authored tasks are created end to end by practising engineers from a professional data provider's expert pool. Co-Authored tasks start from problem statements written by practising engineers; an adversarial Creator-Solver agent pipeline constructs each task's realistic project environment, executable grader and reference deliverables before sending them to expert engineers for validation and difficulty intensification. We evaluate a variety of frontier LLMs under seven publicly available agent harnesses. Even the strongest configuration reaches a mean score of only 50.6%, while spending 27M tokens and 63 minutes per task on average, demonstrating the substantial gap between current agents and autonomous physical engineering design. 

---
# Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3 

**Authors**: Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2610.30499)  

**Abstract**: A generally intelligent system acquires new skills efficiently without requiring humans to redesign its software scaffold for every new task. Yet language models rarely act alone. A harness is a symbolic program that orchestrates calls to the model, handling perception, memory, tool use, recovery, and inference allocation on the model's behalf. ARC-AGI-3 is an interactive benchmark in which agents must explore unfamiliar environments, figure out their goals on the fly, and keep learning as they go. On it, human-steered systems already report ceiling scores across the public games, so we know effective designs exist. We ask whether frontier models can build and improve such a harness on their own. Through a fixed coding bootstrap, each model constructs a complete task harness, acts through that harness on five development games, and revises the entire codebase twice. Every version is evaluated on twenty games that never influence adaptation. We measure task performance with Relative Human Action Efficiency (RHAE), which combines level completion with action efficiency relative to human baselines, and record dollar-denominated inference cost as a practical resource measure. Across 27 lineages, harness performance ranges from zero to near-ceiling scores. Two revisions raise mean held-out RHAE by only 2.80 points (95% interval [-1.44,7.35]). Development gains overstate held-out improvement, and reductions in inference cost can come at the expense of task performance. Human-steered systems demonstrate further headroom. These findings reveal a gap between autonomous harness construction and reliable adaptation under this two-revision protocol: even strong task performance does not ensure that subsequent revisions generalize or preserve existing capabilities. 

---
# From Prose to Properties: Testing Research Claims with ARATest for Agent-Native Science 

**Authors**: Yifeng He, Jiachen Liu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30495)  

**Abstract**: AI agents now run experiments and report findings, so research produces claims faster than researchers can check them. Each empirical claim implies a property that must hold over its cases, but it states that property only in prose. Without an explicit property, no test can decide which evidence would falsify a claim, so validation cannot keep pace with the claims agents produce. To make claims testable, we approximate a claim's semantics by abstracting its property skeleton, a parameterized relation that recurs across claims. We present ARATest, a property-based testing library that checks these skeletons as executable tests. A typed declaration binds a claim to a skeleton and fixes the claim's compared quantities, cases, scope, and numerical policy. A shared case type makes every generator supply cases in the claim's statistical unit. A three-valued semantics assigns each property one of three verdicts: holds, violated, or inconclusive. Every violation carries a replayable witness. A default generator reads the evidence that agent-native research artifacts (ARA) link to each claim, and auto-research generators dispatch an agent to run released code for fresh cases. Our DissClaimer plugin supplies that agent and runs these generators as continuous testing.
Of 243 claims in 33 ARA artifacts, 54 received faithful declarations bound to evidence: 32 held, eight were violated, and 14 remained inconclusive. Two violations exposed errors in published papers that their authors confirmed. ARATest detected all 553 seeded faults without a false alarm on 19 valid controls, outperforming two LLM judges. Fresh runs exposed behaviors that reported evidence could not show. These results suggest that one fixed test can check explicit claim properties against both reported and fresh evidence. 

---
# What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation 

**Authors**: Idunnuoluwa A. Adeniji, Shilpika, Michael E. Papka  

**Link**: [PDF](https://arxiv.org/pdf/2610.30491)  

**Abstract**: LLM agents are often evaluated by reducing execution traces to labels such as final-answer correctness, tool verification, and fault propagation. These labels are measurements produced by parsers, behavioral detectors, and harness rules, and they may not fully capture what the trace shows. We study this in a tool-use setting built on GSM8K, where agents call a calculator whose outputs are occasionally corrupted, giving clear ground truth for what the tool returned, what the model later used, and what the detectors reported. We audit three trace-level detectors across six production models and 30 problems per model. Final correctness, visible re-verification, and fault propagation can disagree in systematic ways. A key failure mode is silent tool-output override: the calculator returns a corrupted value, but the model later uses a different, often correct, value without calling the tool again, so the bad value stops propagating while a verification detector records no re-check. Human review of 90 traces agrees perfectly with the automated labels for final correctness (67/67), fault impact (19/19), and fault relevance (18/18), but only 67/81 for verification; all 14 disagreements fall into four recurring patterns. Execution limits also change the reported result: in an ablation across two models, one ranges from 21/30 to 27/30 correct as step and token budgets change, while the other stays between 27/30 and 29/30. These findings motivate measurement transparency: an automated label should come with the trace evidence behind it, whether the question the detector asks applied to that trace, whether the run was cut off by an execution limit, and any human corrections. Our trace viewer makes this information inspectable alongside the reported results. 

---
# From Single-Actor Environments to Multi-Actor Event Worlds: Can LLMs Be Event World Models? 

**Authors**: Juntong Ni, Zewen Liu, Bohan Wang, Lige Zhang, Wei Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.30488)  

**Abstract**: World models help agents assess the consequences of their actions before acting. Many real systems, however, involve multiple actors who make their own decisions, can evolve even when the agent takes no action, and are only partially observed through event logs. We formalize these systems as event worlds and ask whether large language models can predict future observations from the log and the agent's proposed actions. We build complementary benchmarks from human Diplomacy games and real Polymarket trades. Our hierarchical framework compares prediction with and without other actors' actions, then tests multi-step rollouts, predictions for actions never taken, and usefulness to agents. Fine-tuning and more data improve prediction when all actors' recorded actions are given. With only the agent's action, models struggle when other actors strongly affect the outcome, and larger models and longer histories help little. This gap also limits multi-step prediction and agent decisions, with models overestimating plan success. Yet imagined experience and initialization from world models can still improve policy learning. Anticipating other actors' actions remains a central challenge for event world modeling. 

---
# Why Bother Me Again? Benchmarking Personal Agent Memory for Interaction Preferences 

**Authors**: Jialin Li, Zhenhao Chen, Hanjun Luo, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2610.30485)  

**Abstract**: Personal intelligence systems are taking on increasingly complex everyday tasks, yet repeated questions, unnecessary confirmations, and unwanted outreach can turn capable assistance into interaction burden. Reducing this burden requires agents to remember how users want to be assisted. However, existing memory benchmarks focus on factual recall and personalized answers and rarely test how agents learn and act on interaction preferences. We introduce MemPABench, an interactive environment for on-policy evaluation of memory in personal agents. MemPABench poses long-horizon tasks requiring agents to decide when to clarify, confirm, or act autonomously. Four grounded simulated users hold 14 preferences across work and personal contexts. Their feedback spans 99 sessions over four months, with five event-driven changes per user testing memory updates. We evaluate preference selection, delivered behavior, and sensitivity to context and preference change. Evaluating four memory systems across four LLM backbones, we find that current memory brings limited gains in preference selection. Retrieval-based systems (i.e. Mem0, MemOS, simple RAG) do not outperform No Memory, while File-based memory improves accuracy but applies preferences beyond relevant contexts. Our analysis shows that current memory struggles to retrieve interaction preferences and adapt to preference changes. Even with the correct preference available, overriding the agent's default interaction habits remains challenging. The environment and dataset are available at this https URL. 

---
# Same Beliefs, Different Policies: Learning Reusable Representations for Early Stopping in LLM Reasoning 

**Authors**: Jingwei Ni, Dmitry Abulkhanov, Vladislav Smirnov, Markus Leippold, Elliott Ash, Timothy Baldwin, Artem Shelmanov, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2610.30450)  

**Abstract**: Reasoning LLMs often continue reasoning after reaching a correct answer, wasting computation and sometimes replacing it with an incorrect one. Deciding when to stop is a metareasoning problem: the model must weigh what further reasoning could achieve against its cost, given what it can already answer. Direct utility training learns a stopping representation at a chosen token price; supporting new prices can require additional controller training. We introduce BELT (BELief-based Termination), which separates outcome prediction from stopping control. It learns beliefs about current and future answers from a frozen LLM's hidden states, then fits a linear stopping policy on the frozen features. The prediction targets are independent of token price, so one probe supports different accuracy-cost trade-offs through CPU policy refitting or threshold adjustment. Averaged over five out-of-distribution (OOD) math benchmarks, BELT reduces generated tokens by 41% on Qwen3-1.7B and 51% on OLMo-3-7B-Think without loss of accuracy. The same math-trained BELT transfers to coding and agentic tasks without training on code or agentic data. In all three domains, its accuracy-cost trade-offs are competitive with or better than those of end-to-end controllers. On OLMo3, its probe has 21.2x fewer parameters than one checkpoint of the strongest baseline, OS-Pruner, and needs 55x fewer training GPU-hours than OS-Pruner's five price-specific checkpoints. 

---
# A Three-Pillar Governance Framework for Safe and Trustworthy AI Agents 

**Authors**: Edward C. Cheng, Jeshua Cheng, Alice Siu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30445)  

**Abstract**: AI agents capable of autonomous decision-making introduce significant challenges related to safety, alignment, accountability, and human oversight. Existing Human-in-the-Loop (HITL) and responsible AI approaches provide important foundations, but many remain fragmented across system design, governance, and operational deployment. This paper presents a Three-Pillar (3P) model and framework for developing safe and trustworthy AI agents through transparency, accountability, and trustworthiness. The framework defines an evolutionary path in which increasing agent autonomy is introduced through progressive validation, structured human oversight, and auditable operational controls. We further describe how the model supports agent lifecycle monitoring, decision traceability, escalation management, and human-agent collaboration within enterprise agentic environments. By integrating oversight principles with operational mechanisms, the proposed framework provides a practical foundation for deploying AI agents that remain aligned with human intent while enabling controlled progression toward higher autonomy. 

---
# SpecMAGIC: Specification-Guided Multi-Task Generalization in Multi-Agent Reinforcement Learning 

**Authors**: Zijian Guo, İlker Işık, H. M. Sabbir Ahmad, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30443)  

**Abstract**: Multi-task generalization is an important yet underexplored problem in multi-agent reinforcement learning (MARL). Many real-world applications require multi-agent systems to generalize across complex, temporally extended tasks with diverse inter-agent coordination requirements. We introduce a specification language that combines temporal-logic task specifications with operators for expressing such requirements. We propose SpecMAGIC, a hierarchical framework that leverages the automaton structure of task specifications for subtask decomposition. A centralized high-level planner identifies and allocates compatible subtask sequences across agents, while decentralized low-level policies execute individual subtasks. This enables zero-shot generalization to new tasks by composing reusable subtask policies according to new specifications, without retraining. We establish soundness and probabilistic completeness of the high-level planner. Experiments on multi-agent navigation tasks show that SpecMAGIC substantially outperforms existing approaches on unseen task specifications and maintains strong performance as the number of agents increases. 

---
# Secure Agentic AI and Multimodal Intelligence in 6G Communication Systems 

**Authors**: Satyam Gaba, Devanshu Motwani, Siva Sai, Vineet Suri, Vikas Ranjan, Vinay Chamola  

**Link**: [PDF](https://arxiv.org/pdf/2610.30436)  

**Abstract**: Sixth-generation (6G) communication systems are expected to support immersive, intent-driven, and latency-critical services through tight integration of multimodal intelligence, edge computing, and autonomous network control. In this setting, agentic AI systems can reason over heterogeneous inputs, use tools and network functions, coordinate across distributed entities, and adapt to dynamic environments. These capabilities make agentic multimodal intelligence promising for 6G applications such as semantic communications, digital twins, and zero-touch network management, but they also introduce security and privacy risks that extend beyond conventional AI and network security models. Vulnerabilities can propagate across multimodal data pipelines, memory, tools, communication interfaces, and inter-agent coordination loops. We survey multimodal intelligence and agentic AI for 6G cross-layered communication systems from the perspectives of privacy and security. We present a 6G-oriented system view spanning perception, memory, tool use, and multi-agent orchestration; analyze threat models and attack surfaces across sensing, cognition, action, memory, and communication layers; organize existing attacks into a structured taxonomy; and review mitigation strategies including prompt and memory defenses, inter-agent trust mechanisms, secure orchestration, and privacy-preserving learning. We conclude by outlining open research directions for trustworthy multimodal multi-agent 6G systems. 

---
# RoleFlow: Self-Evolving Multi-Role Agentic Collaboration via Structural Adaptation 

**Authors**: Yi Ge, Pan Wang, Xuehao Cui, Zian Zeng, Xu Hu, Heng Huang, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30429)  

**Abstract**: Large language model (LLM) based multi-agent systems often rely on predefined collaboration structures, limiting their adaptability to evolving task requirements. Although recent approaches enable dynamic orchestration, trajectory-level rewards they apply assign one outcome to each decision, ofen struggling to identify the contributions of individual structural changes. We propose RoleFlow, a self-evolving multi-role agentic system that refines collaboration structures through feedback-driven atomic edits and on-demand role synthesis. To make structural adaptation learnable, we introduce Single-Edit Counterfactual Policy Optimization (SECPO), which evaluates each edit against its no-edit counterpart at the same chain state and assigns targeted credit to edit-specific tokens. Across seven benchmarks spanning four domains, RoleFlow achieves the optimal average performance, surpassing the strongest competitor by 4.7 points. It leads on all three multi-hop QA benchmarks by 3.2-7.5 F1 points and also shows supremacy on mathematical reasoning, web interaction, and code generation. Further analyses demonstrate cross-benchmark transferability, training-data efficiency, and cross-executor generalization. 

---
# Soft Ontology Priors over Latent Concept Posteriors for Weakly Supervised Neuro-Symbolic Learning 

**Authors**: Tengwei Song, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2610.30391)  

**Abstract**: Neuro-symbolic (NeSy) predictors that supervise only the task label suffer from reasoning shortcuts: distinct concept mappings achieve the same maximum likelihood, leaving the recovered concepts unidentifiable and unsafe for downstream reuse. We attack this failure by injecting a soft ontology prior over the latent concepts into the training objective and compiling it together with the task constraint into a smooth, decomposable probabilistic circuit. Under independent Bernoulli perception, the normalisers of the resulting structured concept posterior reduce to weighted model counts that are linear in the compiled circuit size, because the neural leaf weights fold into the circuit before evaluation. The soft compilation strategy places expert axioms in the circuit's weights rather than its support (task-structural axioms guaranteed by the data-generating process stay hard), so noisy expert axioms only redistribute prior mass and cannot empty the support. We characterise how the prior reweights label-restricted shortcut mass through differences in ontology violation energy, show that soft priors incur bounded log-density perturbations under bounded axiom noise, and prove that the hard-support counterpart can be made degenerate by a single corrupted axiom. We validate these predictions on an audited six-concept view of the real-world BDD-OIA driving benchmark and on controlled NeSy tasks. 

---
# Route-and-Generate: KL-Regularized Router Improvement for Reasoning Routing in LLMs 

**Authors**: Tien-Phat Nguyen, Truong Nguyen, Thanh-Thien Le, Hoang Phan, Thien Huu Nguyen, Linh Ngo Van, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2610.30351)  

**Abstract**: Large language models often apply complex reasoning indiscriminately, leading to overthinking on simple tasks and underthinking on harder ones. We present Route-and-Generate (Rouge), a hierarchical training framework that enables LLMs to adaptively decide when and how much reasoning is required for each input. Our approach decouples reasoning effort allocation from answer generation by introducing a routing policy that allocates a fixed sampling budget across multiple reasoning formats, while the base model is optimized with GRPO using per-sample rewards. To promote efficient reasoning, we employ a rank-based reward that favors shorter correct solutions, and update the routing policy via a closed-form KL-regularized policy improvement step distilled into a neural router. This design encourages concise answers on easy problems while reserving longer reasoning for harder ones, achieving improved accuracy-efficiency trade-offs across multiple reasoning benchmarks. 

---
# Breaking the high-degree barrier in Lehmer's problem with a human-AI discovery loop 

**Authors**: Yuri Chervonyi, Sergei Gukov, Thang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2610.30277)  

**Abstract**: While artificial intelligence has made rapid progress in mathematical reasoning, many open problems depend on structure that is revealed only through large-scale computational exploration and analytical synthesis. One such example is Lehmer's problem from 1933. It asks whether the Mahler measure of an integer polynomial, which measures how far its roots stray from the unit circle, can be made arbitrarily small. Previously, constructing irreducible polynomials with integer coefficients and small Mahler measure has been carried out systematically only for even degrees $d \le 180$. At higher degrees, Boyd's analytical one-parameter family frequently factors, skipping nearly 50% of target degrees and leaving no general construction for arbitrary $d$. Here, we address the high-degree regime through a human-AI discovery loop, in which we extend the conversational prompting with a fast numerical pipeline, an exact symbolic verifier, and large-scale distributed search. The AI agent prototypes numerical and symbolic solvers locally, launches parallel searches across distributed clusters and works with the human researcher to extract and verify mathematical hypotheses. Through this interactive loop, we derive a five-parameter construction, which we call the Generalized Boyd algorithm. It eliminates degree-skipping and constructs irreducible polynomials with Mahler measure strictly below Boyd's limit point $\alpha_1 \approx 1.255434$ for every target degree. Evaluated across all even degrees up to 2000, our approach improves upon six previously published values and holds 99.6% of all best known records. With this work we release the verified dataset of polynomials with minimal Mahler measure across all studied even degrees. 

---
# Feedback That Backfires in Games: Do LLM Players Repeat the Move They Just Watched Fail? 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2610.30205)  

**Abstract**: When an LLM agent's move fails, the harness writes the move and the failure message into the transcript and asks again. In tool calling, this record makes small models more likely to repeat the failed action. We test whether the same holds in games (Sokoban, Connect Four, Othello), where a move can fail by being illegal or by being legal but losing, both verified exactly. On fixed probe items built by perturbing a correct move, the record backfires: pooled over 8 instruction-tuned models (135M-1.7B) it makes the failed move more likely (corrective gain G = -1.38 nats per token), through the move's surface form, for illegal moves but not for losing ones, and not for every model. An exploratory analysis shows what sets the sign: after the record, the failed move's log-probability barely depends on how likely it was before (slope 0.18). Moves the model would not have played become more likely, and greedy decoding starts to produce them; moves it was about to play become less likely, but greedy decoding still repeats 38% of them. A description of the failure that omits the move string keeps the model's preference instead. Along the OLMo 2 1B post-training ladder the inversion is present in the base model and largely removed at the DPO stage. RAGEN's default models invert in RAGEN's own Sokoban format before any RL. A "do not repeat" instruction does nothing, a clean retry is the worst harness, and a decoder ban stops exact repeats but not illegal play. All measurements ran in float32 on a free Colab GPU, which first reproduced the original paper's number. 

---
# When Stable and Preferred Semantics Agree: A Path-Based Approach 

**Authors**: Yan Zhang, Kehan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30176)  

**Abstract**: We investigate structural conditions under which preferred and stable semantics agree in abstract argumentation frameworks. Our approach is based on suitably chosen sets of paths in the attack graph and binary relations induced by them. We establish a general path-based criterion for semantic agreement and instantiate it with several classes of paths, obtaining sufficient conditions expressed in terms of well-foundedness of the corresponding relations.
The resulting conditions strictly generalize two principal structural criteria for preferred and stable agreement in the literature: Dung's limited controversiality condition and the SCC-symmetry condition. In particular, Dung's result is recovered as a special case by taking the relevant path set to be the set of even paths. We further introduce two infinite sequences of path sets, each yielding a family of agreement conditions. For the first sequence, a Collapse Theorem reduces the infinitely many associated well-foundedness conditions to three essentially distinct ones, whereas the second sequence gives rise to genuinely distinct conditions with no analogous collapse. Our results apply to both finite and infinite argumentation frameworks, admit simplified formulations in the finite case, and also yield sufficient conditions for the existence of stable extensions. 

---
# Neural Networks with Differentiable Belief-Function Layers 

**Authors**: Antonin Sulc  

**Link**: [PDF](https://arxiv.org/pdf/2610.30173)  

**Abstract**: Yager's 1987 rule of combination $\boxplus$ was proposed on epistemic grounds: under high conflict, mass should be redistributed to ignorance rather than renormalised onto the class simplex. We observe that backpropagation needs the same rule: Dempster's normaliser $(1{-}\kappa)^{-1}$, where $\kappa$ is the conflict mass, puts a $(1{-}\kappa)^{-2}$ term in the gradient that diverges exactly when evidence sources disagree, while Yager's $\boxplus$ is bilinear and has no such term. Building on this, we construct a deep evidential pipeline in which every mass in the chain from heads to output is a valid Dempster--Shafer mass function, combined by Yager's rule. The output mass on the ignorance set, $m(\Theta)$, serves as the network's ``I do not know'' signal. The network is fit to hard labels and never sees an annotator distribution, yet $m(\Theta)$ orders human disagreement in two modalities, visual and linguistic. Dempster's rises too and ranks contested inputs about as well, but its normaliser divides the conflict out: Yager's vacuity is $9\times$ larger on the most contested CIFAR-10H images, so only Yager's is readable as an amount of ignorance without recalibration. For out-of-distribution detection it is competitive with strong single-pass baselines, while the clamped Dempster variant that scores higher becomes gradient-unstable once the clamp is removed. 

---
# Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework 

**Authors**: Hager Radi Abdelwahed, Toluwaleke Olutayo, Chirag Goel, Jana Osea, Aldo Zaimi, Tom Swystun, Sandy Erni, Francisco Emiliano Lopez Saavedra, Yan Boulanger, Gaétan Marceau Caron  

**Link**: [PDF](https://arxiv.org/pdf/2610.30170)  

**Abstract**: Canada's increasing wildfire activity requires frequent national hazard mapping to guide proactive mitigation, including fuel management, land-use planning, infrastructure protection, and evacuation routing. However, process-based wildfire simulation frameworks (e.g., BurnP3 and BurnP3+) rely on computationally intensive Monte Carlo simulations to model burn probability, fire intensity, and rate of spread. These workflows can take days to weeks to process a single regional modelling unit. To overcome these computational bottlenecks, we present three contributions developed through collaboration between AI researchers and wildfire modelling experts: (a) a standardized framework for training, fine-tuning, and evaluating AI surrogate models against process-based wildfire simulators; (b) a multi-task deep learning surrogate trained on national-scale data to emulate burn probability, mean fire intensity, and mean rate of spread at a fraction of the computational cost (seconds); and (c) domain-specific metrics and counterfactual evaluation protocols to probe out-of-distribution generalization. Integrating geospatial data (fuel, terrain, ignition likelihood) and regional weather, our model approximates BurnP3+ simulation outputs, achieving strong performance across agreement, rank-correlation, error, and spatial-overlap metrics. It preserves downstream hazard classifications and produces physically plausible responses to expert-informed counterfactual interventions. On a representative region, the surrogate achieves 806-fold end-to-end CPU acceleration and uses 13% of BurnP3+'s host memory, facilitating rapid scenario analysis. 

---
# MCP vs CLI for AI Agents: A Comparative Analysis of Tool Integration Strategies for Autonomous Systems 

**Authors**: Arun Malik, Nitish Talasu, Bradley Klemick, Vineet Tushar Trivedi, Prachi Shah, Sanay Devi, Junjie Zhang, Kunal Sonawane  

**Link**: [PDF](https://arxiv.org/pdf/2610.30157)  

**Abstract**: AI agents require interfaces to interact with external systems such as files, APIs, databases, and cloud services. Two competing paradigms have emerged: CLI (Command Line Interface), where agents execute shell commands directly, and MCP (Model Context Protocol), a structured JSON-RPC protocol providing governed, typed tool access. This paper presents a comparative analysis across 14 evaluation dimensions, supported by worked examples, security assessment against established compliance frameworks, and economic modeling. Our analysis indicates that CLI provides superior efficiency for single-user, local-only workflows, while MCP addresses requirements that CLI fundamentally cannot satisfy, including fine-grained access control, credential isolation, structured audit, and multi-tenant governance. We further examine how progressive disclosure techniques eliminate MCP's historical weakness (context window overhead), yielding token efficiency comparable to CLI while preserving its architectural benefits. A hybrid architecture is recommended for production systems. This analysis draws upon academic, government, and industry sources including NSA formal guidance (CSI U/OO/6030316-26), OWASP security standards, and economic modeling from BCG and Gartner. 

---
# The Cost of Saying Hello: An Empirical Study of Handshake Overhead in AI Agent Communication Protocols 

**Authors**: Pranav Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2610.30151)  

**Abstract**: Every interaction between AI agents begins with a handshake: the initiating agent must locate its counterpart, learn what it can do, and establish a session before the first useful message can be sent. Surveys of agent communication protocols name handshake overhead as an efficiency criterion, yet none report measurements. This paper presents the first empirical, phase-decomposed benchmark of the handshake in three major agent protocols: MCP (in both its 2025 initialize and 2026 discover eras), A2A, and ACP. Using each protocol's official SDK, we instrument every request for round trips, per-phase latency, wire bytes, and token cost across cold-start, warm-repeat, capability-mismatch, and degraded-metadata scenarios (N=50 per cell, on loopback and under an emulated 50 ms RTT). Round-trip count dominates handshake cost. A legacy MCP handshake takes three round trips and 230 ms at 50 ms RTT; A2A needs one round trip and 61.0 ms. Round trips are not the whole bill: when an LLM reads the counterpart's metadata, handshake cost scales with the counterpart's capability inventory at 24-107 tokens per capability, so a 200-capability counterpart spends 11% of a 200k-token context window before the agent reads its own task. MCP's own 2026 revision cuts its handshake from three round trips to two and, with era pinning, to zero on warm reconnects, protocol evolution that corroborates the cost we measure. We release the full harness, raw data, and analysis at this https URL. 

---
# Cost-Aware Abstention for LLM-Based IT Ticket Classification 

**Authors**: Sachin Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2610.31657)  

**Abstract**: Wrong automatic routing of an IT support ticket can cost more than human triage. We evaluate cost-aware abstention as an offline, cost-conditional readiness assessment: select a routing policy on calibration data under explicit costs, then evaluate that frozen policy once on held-out test data. On a public 47,837-ticket, eight-class corpus, we compare four trained non-LLM classifiers with four large language models under zero- and few-shot prompting. Under the illustrative cost model (c_ok, c_triage, c_wrong) = (0, 1, 5), TF-IDF with logistic regression and a threshold of 0.60 selected on calibration reduces test cost from 0.763 to 0.472, a 38.1% relative reduction (paired ratio-bootstrap 95% CI [35.7%, 40.4%]), while lowering wrong routes from 15.3% to 3.5%. Frozen calibration-selected policies also beat both always-route and always-defer on held-out test data in 14 of 16 settings in a post-hoc cost-parameter analysis. The strongest prompted configuration reaches 0.477 accuracy on a stratified 1,000-ticket test subset, versus 0.879 for fine-tuned DistilBERT on the same tickets. Isotonic lowers expected calibration error without reliably improving discrimination: GPT-4o-mini few-shot has one confidence value before and after recalibration and correctness AUROC 0.500. We therefore report correctness AUROC and tie-aware risk-coverage area alongside ECE. At the analytically derived break-even probability 0.80, seven prompted configurations defer all tickets; Haiku few-shot routes only 19/1,000. An LLM-only agreement control and margin/entropy rejection baselines further separate calibration from discrimination. The findings are scoped to this dataset, prompt design, model endpoints, and illustrative costs; the work is an evaluation framework, not a new abstention algorithm or universal deployment certification. 

---
# METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning 

**Authors**: Ali Najar, Hosein Hasani, Amir Kooshan Fattah, Yasamin Rajabi, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2610.31656)  

**Abstract**: Meta-reinforcement learning aims to train agents that adapt quickly to unseen tasks from limited interaction, but this requires representing task evidence over long interaction histories. Classical recurrent agents compress experience into hidden states, while transformer-based agents retrieve from past trajectories at quadratic memory and computation cost. In this work, we propose Memory-Enhanced Test-Time Adaptation (METTA) for meta-reinforcement learning. It decomposes the fast adaptation phase into two temporal scales: a causal neural memory performs fine-grained within-episode adaptation with linear sequence complexity, while compact episode-level summaries support long-range cross-episode task inference without attending over the full interaction history. Moreover, equipping agents with neural memory provides sufficient expressive capacity and test-time training capability, enabling effective adaptation to unseen held-out tasks and task families at meta-test time. On unseen held-out Meta-World ML10/45 test environments, METTA achieves a success rate over $50\%$ while maintaining a low training budget and short wall-clock time. On long-context sparse-reward PointMaze tasks, it outperforms transformer-based baselines by exhibiting more structured exploration behavior. These results show that METTA provides an efficient path toward more adaptive and scalable long-context meta-RL. 

---
# Low-Bit N-Gram Embeddings for Language Models 

**Authors**: Stephen Chung, Wenyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.31646)  

**Abstract**: Many recent open-weight language models use n-gram embedding tables to improve performance. These tables store learned representations of short token sequences, such as pairs or triples of consecutive tokens, which the model retrieves during processing. However, they can require substantial memory: DeepSeek-V4.1-Flash, for example, allocates 196 billion parameters to its Engram n-gram memory. We therefore investigate whether this memory cost can be reduced while preserving the performance improvement that the tables provide. We propose storing n-gram embeddings at low precision and systematically investigate different quantization methods and bit allocations across tables. Across nanochat-, Llama-, and GPT-2-family models trained from scratch, four-bit tables save approximately 75% of table memory while preserving more than 99% of the validation-loss improvement from adding the tables. Using fewer bits leads to a larger drop in performance, although we find that a short fine-tuning phase recovers a significant part of this loss. In addition, reinvesting the saved memory in larger tables can improve performance beyond that of the original full-precision tables while still using less memory. These table memory savings also extend to large existing models: on DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, low-bit tables save 47-55% of table memory, with perplexity increases of less than 1.2% and largely preserved downstream task performance. These findings establish low-bit precision as a practical way to reduce the memory requirements of n-gram embeddings and use the saved capacity to improve model performance. 

---
# Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation 

**Authors**: Gabriele Dominici, Teague McMillan, Fazl Barez, Marc Langheinrich, Martin Gjoreski  

**Link**: [PDF](https://arxiv.org/pdf/2610.31639)  

**Abstract**: Chain-of-thought (CoT) prompting elicits step-by-step rationales that appear to explain model predictions, but there is no guarantee these steps causally drive the final answer. In standard autoregressive generation, dense attention allows each reasoning step to attend to the full context, enabling the model to derive answers through direct lookup while generating plausible-sounding but post-hoc justifications. We introduce CausalCoT, an inference framework that enforces causal transparency by decomposing problems into a directed acyclic graph of atomic sub-questions, evaluating each node in an isolated context containing only its direct causal parents. This Markovian structure removes attention pathways to non-parent content, making downstream predictions sensitive to local reasoning rather than global context leakage. Across a diverse suite of reasoning benchmarks, CausalCoT preserves competitive predictive performance while improving explanation faithfulness, increasing causal sensitivity of intermediate reasoning steps by an average of 12 and up to 21 percentage points over the strongest baseline. The structured dependency graph additionally enables test-time interventions in which local corrections propagate predictably through downstream reasoning. 

---
# Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3 

**Authors**: Barış Alparslan, Sandy Tanwisuth, Caleb Petterson  

**Link**: [PDF](https://arxiv.org/pdf/2610.31624)  

**Abstract**: Instrumental convergence is often invoked as a general reason to expect capable agents to acquire or express power-seeking subgoals: self-preservation, resource acquisition, deception, or resistance to oversight which poses an important threat to the society. For LLMs, reinforcement learning is the first stage of post-training that directly optimizes behavior against an objective; therefore, it is the stage where convergent instrumental behavior, on IC grounds, likely to appear. Does reinforcement learning with verifiable rewards (RLVR) make a model more likely to express instrumental-convergence behavior, or does this depend on the training regime, evaluation frame, and stage of post-training? We study this question as a checkpoint-conditioned behavioral trajectory rather than as a single before/after safety score. Using InstrumentalEval, we evaluate OLMo-3-7B-Think across SFT, DPO, 13 intermediate RLVR checkpoints, and the final RLVR model. The trajectory is nonmonotonic: instrumental rate rises from 25.0% at SFT to 46.1% near RLVR step 200, then falls to 11.8% at the final checkpoint. Paired item-level tests and step-wise regression support the final SFT-to-RLVR reduction. Prompt framing does not explain this pattern: a harmful-prompt indicator combined with other framing effects explains only 15% of within-checkpoint score variance on average, and Iterative Null-space Projection (INLP) removes a detectable framing direction without changing the SFT-to-final-RLVR gap. Together with preliminary 32B endpoint results (35.53% SFT to 22.37% RLVR final), these findings indicate that RLVR does not monotonically increase instrumental behavior in this setting: the rate rises early, then falls below the SFT baseline. The result is specific to this model family and benchmark, and judge agreement is moderate, so absolute rates should be read cautiously. 

---
# How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions 

**Authors**: Miguel Mochizuki Silva, Samuel Ismael, Mateus C. Dias, Thaís Gaudêncio, Yuri Malheiros, Barufaldi  

**Link**: [PDF](https://arxiv.org/pdf/2610.31606)  

**Abstract**: Purpose: To describe an evidence-aware retrieval evaluation protocol for a clinical retrieval-augmented generation (RAG) system over breast lesion management literature, addressing the dependence of retrieval scores on annotation coverage. Methods: An 18-document corpus and an 18-query development benchmark, with one Yes/No or WH-question per document, were used. Retrieval combined dense embeddings, BM25, and cross-encoder reranking. Chunk sizes of 128 to 1024 tokens, overlaps of 0 to 20%, and four retriever configurations were evaluated. Relevance was defined by evidence spans mapped to all containing chunks, treating overlapping chunks with equivalent evidence as relevant. Retrieval was evaluated using R-Precision, Recall@K, MRR, and nDCG@K. Separately, generation was assessed on a 512-token index at K=1 to 5 using DeepEval claim-level metrics, with deepseek-v4-pro as the LLM judge, and BLEU/ROUGE metrics. Results: The selected hybrid retriever, using 384-token chunks and 20% overlap, retrieved accepted evidence in the top three results for 17 of 18 queries. Recall@3, R-Precision, MRR@3, and nDCG@3 each reached 0.94. Ground-truth curation increased evidence spans from 25 to 37 and accepted chunk targets from 36 to 51 without changing retrieval rankings. Conclusion: Cross-encoder reranking achieved the highest retrieval scores. Expanded relevance annotations increased recognized evidence without changing rankings, highlighting annotation sensitivity. The small development set and retrieval-informed curation require independent validation, including radiologist assessment of evidence sufficiency. 

---
# VERA: A Responsible AI Dashboard for Assessing LLM Deployments in Light of the EU AI Act 

**Authors**: Robin Quériaux, Solveig Bachellery, Léa Déléris, Saurabhsingh Rajput, Mariam Barry, Tushar Sharma, Jérôme Lebecq  

**Link**: [PDF](https://arxiv.org/pdf/2610.31588)  

**Abstract**: Institutions that deploy large language models in regulated settings often need technical evidence of responsible AI practices before approving a deployment. COMPL-AI provides a technical interpretation of the EU AI Act that organises this evidence under six principle groups and eighteen technical requirements, twelve of which current benchmarks can measure. The benchmark suites that produce this evidence are typically operated from a command line, while the staff who approve deployments are non-engineers. We present VERA, an open-source self-hosted dashboard deployed to facilitate responsible AI processes within a large European bank. A reviewer selects a model and runs public benchmarks against it. VERA scores the twelve requirements that current benchmarks can measure and instruments the energy consumed by each evaluation, adding run-level evidence for environmental impact. We run four served open-weight models through the requirement pipeline, and the dashboard displays a score for each measurable requirement. A within-subject study then compares how participants read these benchmark results: from the raw output files in one condition, and through the VERA dashboard in the other. The dashboard reduced the time needed to perform the responsible AI assessment, reducing the time required from 56 seconds using raw files to 30 seconds using our dashboard, on average. Our dashboard scored highly (4.36 out of 5) on usefulness in the survey conducted within the organization. The dashboard and the study material are released publicly to promote replicability and reproducibility in different organisational contexts. 

---
# Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination 

**Authors**: Miguel Fernandez-de-Retana, Unai Zulaika, Ruben Sanchez-Corcuera, Aritz Bilbao-Jayo, Aitor Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2610.31586)  

**Abstract**: Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam's own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at the cost of accuracy, without improving how well confidence separates right from wrong answers. Moreover, 13-38% of correct answers rejected a distractor for a clinically unsound reason that confidence did not reflect, and removing or swapping the provided contextual images left accuracy unchanged. 

---
# Social-HUMINT: An LLM-Augmented Unified Framework for Measuring Social Media Exposure 

**Authors**: Pellegrino Casoria, Simon Pietro Romano  

**Link**: [PDF](https://arxiv.org/pdf/2610.31573)  

**Abstract**: In the post-API landscape, social-media profiling has become inexpensive and automated: Large Language Models (LLMs) can extract structured information from unstructured content and deduce attributes users never disclosed. A person's true exposure is not what they post but what can be inferred from it. As these practices become routine, exposure to social engineering attacks becomes an operational certainty, demanding a shared framework for assessment and mitigation.
We make this exposure measurable by quantifying how accurately a target can be profiled, and how far that profile enables social engineering attacks. Because no existing model fully spans this continuum, we unify Social Media Intelligence (SOCMINT) and Cyber Human Intelligence (Cyber-HUMINT), two individual-focused disciplines that evolved separately, into a single doctrine-grounded taxonomy of 38 capabilities across the Cyber Threat Intelligence lifecycle, where user profiling bridges passive collection and active engagement. Domain experts validated the taxonomy (Fleiss' $\kappa$ = 0.92).
We operationalize its core path (extraction, inference, and susceptibility assessment) as an LLM-augmented multi-agent pipeline, evaluated on real participants' public data. Susceptibility is assessed indirectly: the pipeline predicts how each target would rank personalized engagement options, matching their actual choices significantly above chance. This ranking capacity captures the personalization on which tailored attacks depend. The underlying profiles are accurate: extraction reaches F1 = 97.3%, and inferred attributes score 4.73/5, both far above rule-based and NER-based baselines at negligible cost. The framework augments security analysts' tradecraft rather than replacing it and enables defenders to gauge their own exposure before an adversary does. 

---
# When to Follow the Teacher: Advantage-Based On-Policy Distillation 

**Authors**: Mahdi Sabbaghi, George Pappas, Hamed Hassani, Adel Javanmard  

**Link**: [PDF](https://arxiv.org/pdf/2610.31563)  

**Abstract**: Online Policy Distillation (OPD) often outperforms offline methods such as Supervised Fine-Tuning (SFT) by training on trajectories sampled directly from the student. However, standard OPD trains the student to follow the teacher at every state of the trajectory, assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed. We introduce Advantage-Based On-Policy Distillation (AOPD), which weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted.
Directly computing these advantages requires trying alternative teacher actions at each step and running new rollouts from each. Instead, we show that the AOPD objective can be estimated using only standard student rollouts and their terminal rewards. Our estimator uses teacher importance weights on the student-sampled actions to provide unbiased advantage-weighted supervision, without any additional rollouts or reward evaluations. This allows AOPD to focus teacher supervision on actions that help the current student while preserving the rollout efficiency of standard OPD. Experiments on math, coding, and Medqa show gains over distillation and RL baselines in both external-teacher and self-distillation settings. 

---
# The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models 

**Authors**: Shihui Li, Mette Andersen, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2610.31520)  

**Abstract**: Metaphor is a crucial aspect of human cognition. As such, we would like to understand how it is represented in large language models (LLMs) and how it might be controlled. We use mechanistic interpretability techniques to identify causal subspaces in a range of LLMs covering different architectures and sizes. We show that we are able to steer output on classification tasks across different parts of speech. We are also able to steer output on generation tasks, indicating that these techniques can be used to control the metaphoricity of LLM output. 

---
# SABLE: Scalable Affine Bayesian Latent-space Exploration 

**Authors**: Erik Bodin, Lisa Maria Kreusser, Donney Fan, Henry Moss, Geoff Pleiss, Nicolas Durrande  

**Link**: [PDF](https://arxiv.org/pdf/2610.31495)  

**Abstract**: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show that the off-span norm implicitly selected by affine EI can lead to underexploration, and SABLE instead prescribes it using a finite-horizon schedule, without restricting search to a learned subspace. A recompression scheme reduces information to $\mathcal O(M)$ pseudo-observations, bounding refitting cost by $\mathcal O(M^2D+M^3)$ while keeping parameter inference, prediction, and acquisition decisions to tens of milliseconds. Across synthetic benchmarks and latent-space optimisation tasks, SABLE achieves high sample efficiency and reduces decision time by several orders of magnitude at large budgets. 

---
# Training Small Language Models for Transferable Tool Selection 

**Authors**: Zhaoyi Zhou, Akshayaa Magesh, Jeongyeol Kwon, Artem Kuriksha, Daniel Jiang, Andrea Zanette, Ayush Jain  

**Link**: [PDF](https://arxiv.org/pdf/2610.31483)  

**Abstract**: As LLM agents operate over corpora of more than ten thousand tools, selecting the right tool for a complex multi-step user instruction becomes a central bottleneck. Tool descriptions cannot all fit in context, and relevance-based retrievers only rank tools without knowing which are actually needed. Recent work instead leverages proprietary commercial LLMs like Claude or ChatGPT to drive a retriever, decomposing the instruction and selecting over retrieved candidates. We show that a small, open model can be trained via reinforcement learning to perform the same functionality. On ToolBench, a Qwen3-8B model post-trained with our recipe outperforms Llama-4-Maverick (402B). In addition, the trained model transfers directly to different retrievers and generalizes to new tool corpora in different domains. 

---
# MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models 

**Authors**: Khaled Waleed Dawoud, Muhammad Zaigham Zaheer, Khan Muhammad, Muhammad Haris Khan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31478)  

**Abstract**: Vision-Language Models (VLMs) have shown strong potential for video anomaly detection, but dense frame-level inference makes them impractical for real-time deployment. Existing acceleration commonly relies on uniform sampling, which ignores content dynamics and can either waste computation on static segments or miss localized anomalies. We propose MoG-VLM a motion-adaptive anomaly detection framework that leverages motion purely as a lightweight control signal to decide when VLM inference is necessary. To this end, we introduce Temporal Gradient Magnitude (TGM) which performs content-aware sampling to concentrate computation on informative regions while avoiding redundant processing. We further introduce a hierarchical dual-stream design in which dense multi-frame VLM reasoning is activated only around frames with elevated anomaly likelihood, via anchor-triggered context-window expansion with early termination. A fusion module then reconciles these instantaneous semantic cues with temporal action dynamics. Across four benchmarks, our method achieves 91.91%, 84.30%, 87.88%, and 78.96% AUC-ROC on XD-Violence, UCF-Crime, MSAD, and ShanghaiTech, respectively, establishing new state of the art on three benchmarks under the zero-shot, training-free setting. On XD-Violence, we improve over the previous best result by +0.57% while delivering a 1.31x inference speedup. These results demonstrate that content-aware allocation of VLM capacity, processing fewer frames more intelligently rather than all frames uniformly, is a more effective axis of progress than uniformly densifying VLM inference across frames. Code is available on GitHub: this https URL. 

---
# Predictable performance is suboptimal, and what it means for model routing 

**Authors**: Guanhua Zhang, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2610.31428)  

**Abstract**: Large language models (LLMs) differ in capabilities and costs, motivating routing methods that adaptively choose a model for each query. These methods rely on performance prediction: given a query, estimate whether a model will answer it correctly. Most routing methods train separate predictors of candidate models' performance to guide model selection. Our work shows that performance prediction only beats a simple baseline when the candidate model is suboptimal on the task or the performance predictor adds significant capacity relative to the candidate model. Specifically, we take a model's self-confidence, the probability it assigns to its chosen answer, as a baseline and study how much a performance predictor can improve on it. First, we show that any performance predictor that outperforms a model's self-confidence can be used to improve the model's loss: the predictor's advantage transfers exactly in log loss through recalibration. Second, accounting for the predictor's additional parameters, we bound its advantage by the model's suboptimality plus the gain from those parameters. Third, we extend our theoretical results to two-model cascades, which route queries to a more capable model when the first model is predicted to fail. Experiments with 16 language models on four benchmarks show that predictors gain a larger advantage over self-confidence on more suboptimal models. This advantage shrinks as models improve through finetuning. Cascading exhibits the same two patterns: predictor-based routing offers larger gains over confidence-based routing for more suboptimal models, and these gains diminish after finetuning. 

---
# FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models 

**Authors**: Huixin Xu, Ziya Yu, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.31423)  

**Abstract**: Vision-language-action (VLA) models have become an important paradigm for robotic manipulation by unifying visual perception, language understanding, and action generation. Since reliable manipulation requires sufficient observation of objects, end-effectors, and their spatial relationships, multi-view inputs are widely used to provide complementary spatial information. However, our experiments show that existing VLA models do not fully exploit such multi-view complementarity. The policy tends to persistently rely on the wrist view, lacking the ability to dynamically adjust global-local view dependency across manipulation stages, which limits stage adaptability and robustness. To address this issue, we propose FocusVLA, a stage-aware multi-view routing framework. FocusVLA predicts future interaction intent from historical actions and uses it to guide dynamic view routing during action generation. By adaptively coordinating global and local observations across different task stages, FocusVLA improves multi-view utilization. On RoboTwin 2.0, FocusVLA improves the average success rate over the strongest baseline $\pi_{0.5}$ by 10.10 pp. In real-world evaluation, FocusVLA improves the success rate from 71.7\% to 85.0\%, achieving a 13.3\% gain over $\pi_{0.5}$. Moreover, the intent prediction and dynamic routing modules of FocusVLA can be integrated into existing VLA frameworks without modifying their core architectures. 

---
# How Depth Growth and Looping Shape In-Context Learning in Language Models 

**Authors**: Ferdinand Kapl, Kaitlin Maile, Stefan Bauer, Johannes von Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2610.31421)  

**Abstract**: Looped and depth-grown language models reuse computations across depth and have attracted interest for their reasoning capabilities. Building on prior suggestive evidence of improved in-context-learning (ICL), we investigate how such reuse favors ICL, including how it shapes the acquisition, retention, and mechanisms of this ability. Our analysis spans pretrained language models up to 3B parameters and controlled tasks. The experiments provide evidence for a shared inductive bias toward learning from context: grown and looped models rely more on information in context and, compared with a full-depth dense reference, acquire ICL earlier in controlled tasks and retain it more strongly when learning fixed associations in the weights is also possible. Tracing attention heads through training reveals that depth growth propagates early induction heads across layers. Ablations show that grown and looped models rely more strongly on induction-ranked heads for both few-shot performance and later-token prediction. Together, these findings provide evidence for a shared inductive bias toward learning from context and suggest that (soft) parameter reuse shapes how this ability develops and which attention mechanisms support it. 

---
# Post-Hoc Feature Selection Layer for Neural Networks Interpretability 

**Authors**: João Kenji Suwa, João Pedro Silveira e Silva, Bruno Iochins Grisci  

**Link**: [PDF](https://arxiv.org/pdf/2610.31420)  

**Abstract**: The interpretability of complex neural networks remains a critical challenge, especially for models already deployed in high-stakes domains. To address this, we adapted the Feature Selection Layer (FSL) to optimize feature weights post-training, a variant we term post-hoc FSL. Our approach reframes the FSL as a lightweight, trainable module that integrates with already frozen pre-trained models on tabular datasets to highlight the features the original model considers most important. This post-hoc FSL learns feature relevance by fine-tuning its weights based on the original model's learned outputs. Crucially, this process is non-invasive, operating without altering the original model's architecture or its learned parameters. We conducted our experiments using both statistical and visual metrics, including accuracy, F1 score, recall, precision, weighted t-SNE and silhouette score, and also analyzed the stability of the post-hoc FSL on high-dimensional synthetic and real-world tabular datasets. We compared the post-hoc FSL feature weighting method using these metrics against the original embedded FSL and other post-hoc interpretability methods, such as Integrated Gradients, Noise Tunnel, DeepLIFT, Gradient SHAP, and Feature Ablation. Experimental results demonstrate that the post-hoc FSL feature weighting method successfully identified relevant features across the different datasets, maintaining the predictive power of the original neural network while enhancing its interpretability. 

---
# Major Revision, Minor Accountability: LLMs as Reviewers of Published Articles in Top-Tier Journals 

**Authors**: Bastien Le Guellec, Su Hwan Kim, Raphael Bentegeac, Aghiles Hamroun  

**Link**: [PDF](https://arxiv.org/pdf/2610.31414)  

**Abstract**: With the growth of scientific output, the peer review system is approaching a breaking point: reviewers are increasingly hard to secure, and turnaround times lengthen accordingly. LLM-assisted or LLM-generated peer review has been proposed as a remedy, but how such models would treat manuscripts already published in top-tier journals, the closest thing the current system has to a gold standard of scientific validity, is not known. Here we assembled a corpus of 200 articles, each released under a CC BY 4.0 license and published in 2026, from 20 general-science, life-science, and clinical-medicine journals including Nature, Science, and Cell, and asked four current LLMs (two proprietary, two open-weight) to produce independent peer-review reports of each one under a blinded protocol. Their decisions were overwhelmingly harsh and starkly inconsistent with each other: 605 of 800 reviews (75.6%) recommended major revision or rejection, but the rate at which any single model did so ranged from 17.5% to 100% (p < 0.001), disagreement on manuscripts that had, by construction, already survived real editorial review at the journals in question. When the same manuscripts were doctored with one deliberately planted, field-specific methodological or statistical flaw, detection rates varied from 17.0% to 60.5% across models (p < 0.001). Recommending revision for papers that had already passed editorial review (low specificity) while missing flaws deliberately placed in front of them (low sensitivity), these models produce output that should be read for what it is: an uneven, framing- and model-dependent signal. 

---
# Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models 

**Authors**: Athira Gopal, Patil Prathamesh Pravin, Ankush Kadu, Arpit Jain, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31407)  

**Abstract**: Multilingual language models can give different identity judgments when the same people's names are written in different scripts. We study this effect using 601 audited identity pairs with attested Devanagari and Latin forms, holding the question and relation fixed. Qwen2.5-7B's same-person accuracy falls from 35.1% in Latin to 8.5% in Devanagari under Marathi prompts, while different-person accuracy remains above 98%. Evaluation across five models reveals distinct response biases, and reduced tokenization fragmentation does not reliably close the gap. Replicating the design on 374 audited Malayalam pairs built from the same entities reproduces the effect in a second script and language family: same-person accuracy falls to 5.3%, about half the matched Devanagari rate, while the margin still ranks matching above non-matching names at 0.925 AUROC. Latin-trained probes recover label information from Devanagari states, exceeding a string-feature baseline under similarity-matched negatives. Scalar calibration recovers 87% of the raw model-probe balanced-accuracy gap, identifying decision criterion placement as a major contributor. Activation patching localises causal sensitivity to the second entity in Qwen's middle layers; one shared sensitivity direction transfers about half the full-state margin correction. A fact-independent intervention at this site raises held-out Marathi same-person accuracy from 6.5% to 33.5%, with 98.5% different-person accuracy. Together, these results distinguish script-dependent discrimination from criterion placement and identify a causal site for improving identity judgments. 

---
# AgentPhishing: A Benchmark of LLM Email Agents Under Adaptive Attack 

**Authors**: Justin Tan, Miao Li, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.31404)  

**Abstract**: Computer-use agents increasingly handle private information and act on users' behalf across desktop and web applications, including email. Phishing therefore poses privacy and security risks that depend not only on whether agents recognize malicious messages, but also on how they respond. Existing phishing datasets were developed primarily for conventional detection, leaving agent susceptibility to persuasive malicious requests insufficiently characterized. To expose vulnerabilities that pre-existing datasets may miss, we introduce AgentPhishing, a benchmark for evaluating whether persuasive phishing attacks can evade detection and induce LLM email agents to comply with malicious requests. We develop PhishOpt, which optimizes reusable email-generation skills against individual agents under fixed benchmark constraints and a realism constraint. We analyze the generated emails using Cialdini's persuasion principles to identify strategies associated with agent failures and use these findings to construct a universal generation skill. Our evaluation shows that agents can handle pre-existing phishing emails safely yet remain susceptible to attacks adapted to agent behavior, while in an AgentPhishing evaluation of nine agents, four exhibit phishing-detection rates below $10\%$ and malicious-request compliance rates above $45\%$. These results highlight the limits of inferring agent safety from conventional phishing evaluations and the need to assess how agents respond to persuasive malicious requests. 

---
# HERD: Harness-Evolving Recursive Distillation for Continual Learning 

**Authors**: Xun Sun, Thanh Thi Nguyen, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31401)  

**Abstract**: Continual learning requires a model to keep acquiring new skills and knowledge without compromising its existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity. Non-parametric methods, such as prompts, skills, and harnesses, leverage in-context learning but are limited by context length and lossy plain-text compression. To effectively combine their complementary strengths, we propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning framework that learns efficiently in the harness and internalizes experience into cumulative capabilities via distillation. On the harness side, HERD continually refines its skills to capture reusable patterns, while retaining high-residual examples that the skills fail to capture in a bounded memory. On the parameter side, HERD performs drift-constrained distillation with dual anchors, substantially reducing forgetting over long recursive runs. On the 25-stage task stream of TRACE+17, under fixed memory and compute budgets, HERD matches the strongest parametric baseline in new-task acquisition while exhibiting the least forgetting and the strongest retention of general capabilities. 

---
# Auditing the Hand-Written Inputs of Learned Access Control 

**Authors**: Arsalan Jalil Sarghaleh, Mehdi Hazrati  

**Link**: [PDF](https://arxiv.org/pdf/2610.31388)  

**Abstract**: Learned access control is typically deployed behind two hand-built artefacts: a hard-deny set enforced by a small trusted shield, and a region map that decides which requests mined rules answer and which the learner answers. Verifying the shield or measuring the learner validates neither. We audit both on the five public ABAC Lab policies. First, we formalise hard-deny soundness and check it exhaustively over relations of up to 794,250 tuples. Clauses that faithfully transcribe an explicit restriction are unsound at rates of 0.50-0.70 across four elicitors - an author and three blind language-model elicitors - because prose scopes restrictions to a section while the grant relation is a disjunction over all rules: 97% of counterexamples are granted by other rules. Shipped unchecked, these deny sets would withhold 16-78% of everything the policies grant; certified, they withhold nothing. Vacuity, a second silent failure, also occurs. Second, hybrid routing needs a two-class, abstaining rule set that ABAC miners do not emit, and ranking a mixed-effect candidate pool under a cap is one-class in all 24 cells for the two commonest ranking keys. Third, for additive losses the hybrid-versus-learner effect factorises into covered mass times a rule-learner risk contrast that must be reported on both false-non-grant and false-permit loss: on two real policies, most selection rules give opposite verdicts on the two losses, and the one favourable on both holds for one policy, one learner configuration and a band of noise levels. A released artefact regenerates and checks every reported number. 

---
# AI Risk Ontologies for the EU AI Act Compliance: AIRO, VAIR, and their Integration with DPV 

**Authors**: Delaram Golpayegani, Harshvardhan J. Pandit, Declan O'Sullivan, Dave Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2610.31380)  

**Abstract**: As the use of AI, and consequently its harms, becomes more prevalent, the global discourse around AI has become increasingly risk-aware. With new risks emerging more frequently than ever, a key challenge is how to represent, document, and catalogue AI risks in an effective, interoperable, and queryable manner. Additionally, the European Union (EU) AI Act's requirements for "high-risk" AI systems to manage and document the risks creates the need for open and interoperable tools, methods, and standards that facilitate AI risk management in an ever-changing environment. In this paper, we present ontologies and taxonomies developed to assist with risk management and documentation, in line with the requirements of the EU the AI Act. We developed the AI Risk Ontology (AIRO) to specify AI use cases and their associated risks, complemented by the Vocabulary of AI Risks (VAIR), which populates AIRO with taxonomies of concepts. Key international standards on AI and risk, namely ISO/IEC 23894 on AI risk management and ISO/IEC 22989 on AI terminology informed the development of the ontologies. To promote reuse and enable multi-regulatory compliance, AIRO and VAIR have been integrated in the Data Privacy Vocabulary (DPV) through a consensus-building process, expanding DPV's scope from data to AI. 

---
# Automating Feature Flag Cleanup at Scale with Multi-Agent LLM Systems: An Industrial Experience at DoorDash 

**Authors**: Atharv Chandratre, Jai Datar  

**Link**: [PDF](https://arxiv.org/pdf/2610.31351)  

**Abstract**: Feature flags are runtime configuration mechanisms used for gradual feature rollouts, experimentation, and kill switches. Once a feature is fully rolled out, the flag and its gated code become technical debt and can introduce operational risk through inadvertently enabling dormant code paths. At DoorDash, over 56,000 feature flags span 623 repositories, with 1,659 currently stale as of May 2026, with internal estimates of one to two hours per manual cleanup. Existing rule-based cleanup tools proved insufficient in DoorDash's codebases. Our contribution is an end-to-end automated LLM-based feature flag cleanup system that integrates querying live feature flag rollout data using MCP, Jira ticket processing, and end-to-end pull request creation. Our system successfully produced engineer-approved flag removal PRs for 45 (90%) of the 50 stale flags it was evaluated on, proving the system's operational viability at scale with each cleanup averaging only $4.79 and 13.8 minutes. We evaluated the system's quality and reliability boundary based on success rates, cost per cleanup and time metrics. Furthermore, we discuss our lessons learned when architecting this system, to provide practical guidance for developing similar multi-agent systems. 

---
# AgenticEvolve: Evolutionary Program Search with Persistent Language Model Agents that Design and Run Experiments 

**Authors**: Ze-Yu Peng, Qing-Yu Lan, Zhan-He Wang, Jun Zhang, Yun-Song Piao  

**Link**: [PDF](https://arxiv.org/pdf/2610.31326)  

**Abstract**: Evolutionary program search with large language models keeps the structure of an evolutionary algorithm and uses the model as its variation operator. Typically, the model generates each candidate program in a single request from parent programs and their scores. We present AgenticEvolve, an evolutionary program search framework in which each island is run by a persistent large language model agent. Between evaluations, the agent designs and runs experiments on the candidate program. To measure the contribution of experimentation, we compare AgenticEvolve with No-analysis, an otherwise identical configuration with experiments disabled. On two heuristic design tasks with three large language models, AgenticEvolve produces better programs than No-analysis and OpenEvolve on both hidden and unseen instances. It reaches the baselines' final quality with far fewer evaluations and comparable cost. On twelve equation discovery problems, it recovers the ground-truth structure of more equations than No-analysis and existing language-model and genetic programming baselines. 

---
# Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion 

**Authors**: Jinjian Jiang, Jing Peng, Rehan Deen, Yu Cao, Ziquan Liu, Muhammad Salman Haleem, David Mguni  

**Link**: [PDF](https://arxiv.org/pdf/2610.31323)  

**Abstract**: Many deployments require a model to move rapidly from a data-rich source to a related target observed only sparsely. This is especially consequential where target data are difficult to collect, as in healthcare settings with persistent population or acquisition imbalances. Retraining or fine-tuning a high-capacity generator for every target is costly and can be statistically unreliable when target data are scarce. We introduce Meta-Diff, which instead learns a recurring source--target relation once in a compact task space. A shared encoder maps each distribution to a low-dimensional coordinate; a learned relation transports a new source coordinate to its predicted target, and a shared diffusion backbone generates from that prediction without target-specific retraining. Controlled Gaussian-mixture experiments identify when this works: task identity must survive the representation, the relation must generalise across tasks, and the source--target shift must leave useful headroom for transport. On held-out CIFAR-100 validation tasks, correct source $+$ transport achieves 13.9% lower mean SW than transport from the wrong source. With validation-selected adaptation settings in the final-test one-target comparison, its mean SW is 14.7% lower than target-only modelling and 32.7% lower than full fine-tuning; the main route itself uses no target image and assumes the applicable relation is known. On matched CIFAR test episodes, generating from the clean-source coordinate and then applying the known corruption yields lower SW than task-space transport. Office-Home exposes a complementary boundary: transport can align target-domain statistics without reliable object identity. Together, these results recast extreme-few-shot adaptation as reusable transport in task space, while making the conditions for its success and failure directly measurable. 

---
# Hyperreduction accelerates finite element model updating for the discovery of neural materials 

**Authors**: Hagen Holthusen, Matthias Knipper, Kevin Linka  

**Link**: [PDF](https://arxiv.org/pdf/2610.31322)  

**Abstract**: Finite element model updating is among the most general approaches to discover constitutive models from structural experiments: since the displacement field is computed rather than measured, it accommodates arbitrary geometries, boundary conditions, and sparse or noisy data. However, every loss evaluation requires a nonlinear finite element solve per load step, and training neural materials requires many such evaluations. Intrusive model order reduction lowers this cost while retaining boundary conditions and variational structure. How can a reduced model remain accurate while the network weights change at every training step? We combine proper orthogonal decomposition and energy-conserving sampling and weighting with adjoint sensitivities of the reduced model, and adapt both the trial space and the sampled elements along the optimization path. The constitutive law is a polyconvex neural network: a logarithmic normalization term, together with a zero-anchored input convex architecture and a growth term, renders the energy polyconvex, coercive, and non-negative by construction, independent of the trained weights. We demonstrate the approach on two experimental datasets and one synthetic three-dimensional example. For a two-phase specimen, training time decreases from nine hours to 41 minutes compared with a full-order identification, without loss of accuracy. In the three-dimensional example, six modes replace 48 672 degrees of freedom, 116 of 15 625 elements suffice to evaluate the internal forces, and training takes 13 minutes. In all cases, the identified material reproduces the measured reaction force and predicts the displacement field from the force alone. Hyperreduction makes the discovery of neural materials from structural data affordable while retaining the advantages of finite element model updating. 

---
# SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs 

**Authors**: Yifan Yuan, Yancheng Zhu, Zuoyu Zhang, Xiaogang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.31317)  

**Abstract**: Most automated survey generation systems retrieve papers as flat documents, limiting their ability to capture research structure. We present SurveyGraph, a framework built around a Methods-Entity Knowledge Graph (MEKG) encoding method-level follow-on, competition, and shared-resource relations. The graph supports core-paper discovery, retrieval, hybrid outlining, and evidence-gated narrative synthesis. We evaluate retrieval, semantic edges, citation grounding, collection coverage, and generation-stage components. In a controlled seven-method equal-budget benchmark, MEKG-anchored fusion balances recall and structural density, with an independent co-citation check. In a separate archived graph-native diagnostic, graph-aware retrieval nearly doubles recall over flat lexical baselines across eight runs (p = 0.004). A semantic evidence gate achieves AUC = 0.75. Audits of 160 generated sentences find 80.6% anchored-or-partial citation support. Human adjudication of 108 stratified semantic edges finds 68.5% supported-or-partial relations (binary-support Cohen's kappa = 0.64), with improves strongest (80.6%) and extends weakest (52.8%). A SurveyBench-facing collection probe reaches mean reference coverage of 0.521 across four topics with an eight-paper budget. Same-backend Qwen diagnostics on a 30-paper/74-relation graph show reduced organization signals when hybrid outlining is removed. Concurrent-system comparisons provide bounded context rather than matched-backend end-to-end evidence. The results position MEKG as an auditable structural layer for typed diagnostics, relation-specific filtering, and controlled survey generation. Code and artifacts: this https URL. 

---
# When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues 

**Authors**: Yancheng Zhu, Yifan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31314)  

**Abstract**: Large language models can repeat an instruction while failing to keep obeying it. We study this recall-compliance dissociation in long-horizon dialogue and define State Integrity Collapse: a model preserves a stated constraint in memory but stops applying it to later turns. We introduce StateTrack-Bench, a controlled suite of conversations covering numerical, format, role/persona, factual-memory, and logical-rule constraints. Across seven provider-pinned model snapshots and five dialogue-depth probes, strict joint retention ranges from 62.4% to 100.0%, with degradation concentrated in symbolic constraints and in one especially brittle model. A forked recall audit shows that 88.6% of observed failures are behavioral-drift failures rather than ordinary forgetting; excluding the brittle outlier preserves this diagnosis (84.4% of failures). Extended evaluations at depths up to 50 turns confirm that the dissociation persists beyond the core benchmark, while Structured State Anchoring recovers up to 24.4 points on immutable constraints. The results show that reliability cannot be inferred from single-turn instruction following alone; evaluation must track whether remembered constraints remain operational as dialogue state grows. Code and artifacts are available at: this https URL. 

---
# Superintelligent Software Engineering: Governance That Scales with Capability 

**Authors**: Mohammad Mehdi Gholinejad  

**Link**: [PDF](https://arxiv.org/pdf/2610.31306)  

**Abstract**: Software agents are being trained to exceed human engineers; we argue that capability alone does not produce better software engineering. We define superintelligent software engineering as the discipline of building and evolving software with agents whose capability exceeds that of human engineers, in which governance (gates, contracts, decisions and decay control) scales with capability. The argument rests on three claims. First, correctness in a project rests on decisions and contracts that the code does not carry. Second, knowledge decays. Third, once an agent outpaces its reviewers, reading catches a shrinking share of the changes built on stale knowledge. A simple model shows that, past a fixed review capacity, escaping changes grow linearly with throughput while the share that reading catches falls toward zero; governance can lower the slope, capability alone cannot. An exploratory snapshot of 452 of the most-starred public GitHub repositories finds a root-level or .github agent context file in 48.67% (95% confidence interval 44.1 to 53.3); in the median file that names repository paths, 9.5% to 22% of those paths do not resolve, depending on how a reference is read. We then present OSTEON, a discipline for the regime, released with a DOI (https://doi.org/10.5281/zenodo.23436020%29%3A each project gets a governed knowledge repository of decisions and contracts, every change is checked mechanically against it, and the knowledge itself is checked for decay at every session start; planning is gated, and sessions record what they learn without writing rules. We report no measured effect of OSTEON, mark claims about its mechanisms as opinion and the author's practice heuristics as design rationale, and state the questions that would falsify the thesis. 

---
# RecallGS: Geometry-Grounded Recall for Online Semantic Gaussian Splatting 

**Authors**: Thanh-Khoi Nguyen, Thien-Phuc Tran, Minh-Triet Tran  

**Link**: [PDF](https://arxiv.org/pdf/2610.31293)  

**Abstract**: Online semantic 3D Gaussian Splatting (3DGS) enables open-vocabulary scene understanding from streaming observations. However, existing methods recover geometric context missing from image-derived semantic features using an additionally learned 3D network, incurring extra training and runtime overhead. We observe that such geometric-semantic evidence may already exist in the agent's observation history and can be retrieved rather than learned. Based on this insight, we introduce RecallGS (Geometry-Grounded Recall for Online Semantic Gaussian Splatting), a lightweight, training-free mechanism that treats past observations as a non-parametric geometric-semantic prior. RecallGS follows a recall-ground-fuse pipeline: CLIP retrieves relevant observations, alpha-compositing weights identify geometrically supported views, and aggregate their semantic features over each Gaussian's rendered footprint. On ScanNet and ScanNet++, RecallGS achieves strong semantic segmentation performance at 2.48 FPS, demonstrating that geometric-semantic context can be effectively recovered from observation history without an additional learned 3D branch. 

---
# Beyond Accuracy: Disentangling Decomposition from Typed-State Quality Assurance in Multi-Agent LLM Pipelines 

**Authors**: Maria Gabriela Alves Zuppardo, Vinicius Cardoso Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2610.31280)  

**Abstract**: Multi-agent LLM pipelines are often credited with gains in software engineering tasks, but it is rarely clear whether a gain stems from the typed contract between agents or from the specialized prompts that decomposition enables. We study this attribution problem in Requirements Engineering, comparing three architectures for classifying requirements as functional or non-functional and then prioritizing them with MoSCoW: a single-call baseline; a two-call variant that reuses the pipeline's specialized prompts but passes an unvalidated intermediate; and a pipeline with a Pydantic-validated typed state. The 18 conditions cover PROMISE NFR+ (625 requirements), three locally hosted small language models, and two languages. The F1 gain over the baseline is associated with the two-call, specialized-prompt variant (up to +0.075), whereas the typed state adds a small difference (-0.003 to +0.013) that is significant in one of six comparisons. Beyond accuracy, the typed pipeline contains out-of-range model outputs at the agent boundary, while the unvalidated variant fails later and drops the requirement, and it shows a small, reproducible increase in inter-model MoSCoW consistency whose cause our design cannot isolate. The study is associative: the contrasts bundle prompts, call count, and staged context, and the results come from three 7--8B models on one dataset. The implementation and replication package are archived on Zenodo (this https URL). 

---
# CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring 

**Authors**: Shmuel Amar, Benjamin Lieber, Ori Shapira, Debanjana Kar, Sebastian Maximilian Dennerlein, Yufang Hou, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31277)  

**Abstract**: While LLMs have transformed text generation, standard chat-and-canvas interfaces often reduce human authoring to quite passive supervision, causing review fatigue, over-reliance on AI, and the sidelining of the human role in long-form, source-grounded writing. Prior systems typically address isolated writing phases, leaving comprehensive end-to-end human-LLM collaboration largely unexplored. In this paper, we introduce CoWriter, a proactive agentic scheme that adapts expert human (co-)authoring methodologies from cognitive science to flexibly steer the entire writing workflow while actively encouraging the user to engage. In an exploratory user study, compared to a standard LLM baseline, authors using CoWriter better expressed their perspectives and maintained a higher sense of document ownership, supported quantitatively by $2.2\times$ more successful agent engagements. Furthermore, external reader judges rated the resulting texts higher in "human touch" qualities like storytelling and integrating unique personal experiences. Overall, our results demonstrate that proactively engaging users through established human practices is beneficial for effective LLM-supported co-authoring, providing useful methodologies, techniques and insights for future research on collaborative agents. 

---
# Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use 

**Authors**: Yijiang River Dong, Zexue He, Cheng Yu, Tiancheng Hu, Michael J Ryan, Jan Batzner, Zheng Hui, Shaolong Wu, Orestis Papakyriakopoulos, Alex Pentland, Nigel Collier, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2610.31272)  

**Abstract**: Memory enables language model assistants to maintain continuity across sessions and generate personalized responses, but knowing more about a user does not necessarily produce responses they prefer. Existing evaluations assess memory capabilities without directly establishing whether memory use benefits the user. To study this gap, we collect approximately 1,000 pairwise judgments from 87 participants using their own ChatGPT queries and personal memories. We find that memory-conditioned responses are not consistently preferred over responses generated without memory and in some cases, users strongly prefer the latter. Automatic evaluators also show limited agreement with users and favor memory-conditioned responses substantially more often than users do. Motivated by these gaps, we introduce a memory-aware reward model that learns from user judgments, achieving 72.8\% preference prediction accuracy and outperforming the existing baselines. Our findings motivate evaluation and optimization objectives that align memory use with the preferences of the people whose information is being used. 

---
# Notation3: Graph Terms for RDF 

**Authors**: Dörthe Arndt, Pierre-Antoine Champin  

**Link**: [PDF](https://arxiv.org/pdf/2610.31266)  

**Abstract**: Notation3 Logic was introduced as an extension of RDF allowing the user to express rules on top of triples and to make statements about graphs. The original proposal of the logic only came with an informal specification of its semantics which has been refined later on. While this refinement clarified the meaning of rules and implicitly quantified variables, it left the meaning of graphs used as terms - so-called graph terms - open. In this paper we aim to close this gap. We discuss in detail, how graph terms are used in Notation3, which formal requirements they need to meet, and provide examples to illustrate the difficulties when defining their semantics. We then introduce an abstract syntax for the logic and define its semantics in two steps: similarly to RDF's simple interpretations, we define interpretations and basic entailment for Notation3 which covers plain RDF triples but also triples containing graph terms. N3-entailment is then an extension of basic entailment which assigns meaning to the different special predicates of the logic like for example the implication symbol. N3-entailment is defined in a way that it can be extended whenever a new predicate is added to the logic. As Notation3 Logic is currently under development, we expect many predicates to be added. Our research provides and important step towards the full standardisation of the logic. 

---
# Can Entanglement-Induced Feature-Map Designs Improve Projected Quantum Kernels? 

**Authors**: Ravi Kumar Jha, Hsuan-Yin Lin, Eirik Rosnes  

**Link**: [PDF](https://arxiv.org/pdf/2610.31260)  

**Abstract**: Projected quantum kernels (PQKs) have emerged as a notable approach to circumvent the hardness of machine learning tasks arising from the exponential dimensionality of the quantum Hilbert space. However, recent PQK studies often rely heavily on extensive classical hyperparameter tuning of the radial basis function (RBF) kernel parameter and the support vector machine regularization parameter. The extensive tuning can mask the true representational power of the underlying PQKs induced by quantum feature maps, which has not been explored in detail. Thus, this work aims to expose the structural capabilities of PQKs by isolating and evaluating their native inductive bias through comparisons between untuned and tuned PQKs against the RBF baseline. Furthermore, we propose novel quantum feature maps that leverage two-qubit entanglement to capture multivariate correlations in IoT data. The results show that extensive tuning can lead to competitive performance architectures, while our proposed entanglement topologies exhibit a significantly stronger native inductive bias. This reveals that superior quantum architectural design can yield structurally better PQKs with improvements arising from quantum-mechanical properties rather than from classical hyperparameter tuning. 

---
# QICOM: Open-Source Software for Reproducible Image Quality Evaluation, Blinded Assessment, and Radiomic Comparison 

**Authors**: Fabio Mattiussi, Switinder Ghotra  

**Link**: [PDF](https://arxiv.org/pdf/2610.31218)  

**Abstract**: QICOM (Quality Image COMparator) is open-source software for quantitative image comparison, paired radiomics, and blinded assessment. It saves processing settings, image identities, measurements, and response mappings so a comparison can be reopened and recalculated. We evaluate reference agreement, responses to controlled perturbations, and recovery of saved analyses. Synthetic experiments cover 37 quantitative descriptors using independentnumerical references and separate upstream integration comparisons. The QICOM/MIRP backend meets the reference tolerances for all 482 numeric IBSI 1 digital-phantom entries and all 1,622 entries across configurations A-E of the public CT reference case; these comprise 1,804 feature/aggregation comparisons and 300 preprocessing diagnostics. Restricting masked SSIM and HFEN to complete kernel support removes the observed dependence on excluded pixels. In a synthetic filtering example, stronger smoothing increases SSIM while worsening MSE and HFEN relative to milder smoothing and changing GLCM contrast more strongly. Reopened radiomics projects reproduce all 43 feature pairs per comparison, and two blinded deliveries return four correctly mapped scripted responses. These experiments document the calculations and data transfers linking the three workflows. 

---
# Predicting Music Valence and Arousal via Textual and Multimodal In-Context Learning in Large Language Models 

**Authors**: Chi-En Dai, Chih-Pin Tan, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31202)  

**Abstract**: While arousal can be reliably predicted from musical audio, emotional valence remains difficult to model due to its subjective and context-dependent nature. In this work, we investigate the potential of in-context learning (ICL) with large language models to advance continuous valence and arousal prediction. Rather than predicting emotion scores directly, our framework leverages contextual demonstrations from an auxiliary dataset to refine the initial predictions produced by a small, task-specific base model. We systematically evaluate a text-based ICL variant using lyrics and generated audio captions, alongside a multimodal variant that incorporates raw audio signals. Using the entire MER60 dataset strictly as a held-out test set, our experiments demonstrate substantial performance gains, particularly for valence. Text-based ICL improves the valence $R^2$ from 0.110 to 0.427 with 80 demonstration shots, while the multimodal variant achieves 0.346 with only 10 shots. These findings suggest that training-free ICL offers an effective inference-time strategy for current language models to predict continuous music emotion. 

---
# When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions 

**Authors**: Clemencia Siro, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2610.31193)  

**Abstract**: Metadata-conditioned generation is often framed as a data-to-text task in which additional input fields should improve grounding. Museum description generation challenges this assumption because catalogue records are heterogeneous, partially visual, and unevenly documented across geographic regions. We study how metadata composition shapes the generation of artefact descriptions through a two-phase framework. Phase 1 analyses natural metadata variation in a large museum collection, where field count, field type, image availability, and Place co-vary. Phase 2 introduces controlled field probes that hold object identity fixed while varying metadata quantity, field type, and image access. Our results show that metadata quantity alone does not explain generation behaviour. Field type determines whether models ground, omit, or infer information, while images can both support visual grounding and increase unsupported interpretation. More broadly, our findings show that metadata-conditioned data-to-text evaluation should assess field- and claim-level support, rather than relying only on aggregate input coverage or reference similarity. 

---
# Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness 

**Authors**: Asal Mehrabi, Mohammad Aliannejadi, Clemencia Siro  

**Link**: [PDF](https://arxiv.org/pdf/2610.31172)  

**Abstract**: LLM-as-a-Judge evaluation is increasingly applied to multilingual tasks, yet prompting strategies such as Chain-of-Thought and Role Conditioning are recommended based solely on English benchmarks. Whether their effectiveness transfers across judge models and languages remains unknown: prompt-sensitivity studies are largely English-centric, while existing multilingual judge studies fix the prompt template. In this work, we address this gap by measuring this interaction through a benchmark of 9 prompting strategies, 4 open-weight judges, 27 languages, and 4 tasks across 13 datasets, yielding over 1M pointwise scores on gold-reference outputs, validated against human judgments. We find that prompting strategies are not transferable: the choice of the judge model dominates the choice of prompting strategy, strategy effects reverse direction across judges, and stability degrades in low-resource languages, where these effects are concealed by aggregate reporting. These findings show that prompting effects in multilingual evaluation are conditioned on the judge, task, and language. Consequently, prompting strategies should be validated for the intended evaluation setting, and cross-study comparisons should account for both judge and prompt configuration. 

---
# Network topology shapes consensus and vulnerability in LLM populations 

**Authors**: Roberto Passaro, Ariel Flint, Romualdo Pastor-Satorras, Andrea Baronchelli, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2610.31165)  

**Abstract**: As large language models (LLMs) are deployed in interacting populations, their collective behaviour depends not only on the models but also on who interacts with whom. Studies of convention formation in LLM populations have mainly tested well-mixed interactions. We place policies extracted from four LLMs on fully connected, random regular, and heterogeneous networks, with $10^3$ to $10^5$ agents playing the naming game. Topology affects how quickly consensus emerges, sometimes whether it emerges at all, and, in rare cases, which convention wins. Sparsest networks slow consensus up to twelvefold. Yet, by preserving correlations between persistent neighbours, they restore consensus in some cases that fail to converge under well-mixed interactions. Networks also govern vulnerability to adversarial attack. We measure the critical mass of committed agents needed to overturn an established convention. Tipping thresholds follow the same ordering across models and conditions: hubs $\leq$ random $\leq$ low-degree nodes. On the most heterogeneous topology, median critical masses are 3%, 16%, and 35%, respectively; targeting hubs requires one-sixth as many committed agents as random placement. For most models, networked populations are easier to tip than well-mixed ones, so well-mixed estimates can overstate robustness. Assessing the safety of multi-agent AI requires mapping who interacts with whom. 

---
# Prompt Sensitivity in Tool-Augmented Model Comparisons 

**Authors**: Mateusz Modrzejewski, Tomasz Radzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.31148)  

**Abstract**: Language models are compared on question-answering benchmarks that pose each question in one wording, and a reworded question can change which of two models scores higher. We ask whether giving the models tools makes such comparisons more stable. An accuracy gain cannot answer this, because tools change several steps between the question and the scored answer at once. We propose a matched protocol that separates those steps: the answer readout, reasoning, the decision to call, the tool's answer, and the report supplied without a call. We apply it to six open models in five domains under five wordings. We find that tools raise accuracy where a program answers the question and widen the wording sensitivity of many pairwise comparisons. The widening follows how often a model calls, and call frequency follows the wording for some models and not others. It also depends on the interface and the instructions: a native tool-calling interface and a second tool description leave no detected widening in molecules, and sensitivity persists where tools return intermediate results. These findings show that deterministic reports do not guarantee stable comparisons. We recommend reporting call frequency and comparison stability under several wordings beside accuracy. 

---
# PriMa-Causa: Causal Prescriptive Maintenance 

**Authors**: Felix Saretzky  

**Link**: [PDF](https://arxiv.org/pdf/2610.31147)  

**Abstract**: Automated manufacturing requires data-driven decisions under uncertainty and sparse observational records, and unplanned downtime is a dominant cost of getting them wrong. Prescriptive maintenance recommends a maintenance action against an impending failure rather than merely predicting it. Established reference architectures couple predictive models with rule- or optimisation-based recommendation logic. Selecting a corrective action, however, requires an interventional quantity: how the line behaviour would change if the action were set. Manufacturing data is observational, so a model fitted to predict the outcome estimates an associational quantity and may misdirect the choice of action without the prediction error revealing it. Causal estimators target the interventional quantity but require an expert-led causal study per case, for which the bounded decision window of operation leaves no room. This dissertation analyses how causal reasoning can be embedded into prescriptive maintenance so that actions are ranked by their estimated interventional effect. The result is PriMa-Causa, built from three components: a causal reasoning layer that integrates the estimate into the PriMa reference model, a parametric simulator that supplies the interventional ground truth absent from manufacturing data, and a graph-conditioned tabular causal foundation model that, pre-trained once on this synthetic production-line prior, answers interventional queries in a single forward pass. PriMa-Causa attains a lower prescription error than associational baselines, with the largest margin under weak overlap and sparse context, and a partially disclosed causal graph reduces this error further. The causal inductive bias of the synthetic prior and the graph conditioning can thus replace per-case estimator fitting, a step toward automating causal inference for autonomous and self-healing maintenance. 

---
# Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering 

**Authors**: Aman Behera, Namit Solanki, Mehul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2610.31135)  

**Abstract**: Long-context models often use answer-bearing evidence less effectively when it appears away from a prompt boundary. We compare this position bias across Transformer, state-space, and hybrid Mamba architectures by moving one gold document through a fixed set of nine distractors. The evaluated Transformers show a beginning advantage that the evaluated state-space models lack, while both families favor evidence near the end. A matched pure-versus-hybrid Mamba comparison points in the same direction but remains statistically uncertain. The family difference appears among models that can answer the task. Transformer primacy reappears on synthetic retrieval, where state-space ceiling prevents a family comparison. A corpus swap within one state-space architecture changes overall accuracy without a detectable change in curve shape. Prompt changes can create or nearly remove a beginning advantage. Position remains decodable across architectures even when response curves differ, and attention concentration near initial tokens covaries with beginning sensitivity in the tested Transformer family. These exploratory results show that effective context use depends on the evaluated architecture family, capability, task, and prompt, without isolating the sequence mixer as the sole cause. 

---
# ProductWebBench: Evaluating Website Continuity Beyond Task Completion 

**Authors**: Jiajun Wu, Jian Yang, Yaxin Du, Shukai Liu, Linzheng Chai, Tuney Zheng, Wei Zhang, Haitao Yang, Yan Bai, Yihang Lou, Xianglong Liu, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2610.31118)  

**Abstract**: Frontend coding agents are usually graded on one question: did the requested target get built. Real engineering changes carry a second obligation, because the rest of the site has to keep working. We call that second obligation website continuity, and introduce ProductWebBench, a benchmark that scores both on living frontend repositories: 320 Change tasks over existing code and 80 long-horizon Build tasks. Every task freezes a repository snapshot, staged requirements, browser evidence and deterministic checks before evaluation, and enters only once a reference passes every check and plausible-wrong ones fail. A simulated user reveals the requirements one stage at a time, and every check cleared earlier stays armed for the rest of the run. Across 8,737 runs from 13 models the two come apart: of the Change runs passing every requested-work check, 28.2% still fail a continuity check -- a blank route, horizontal overflow, console errors -- and 10.8% push a browser state they had already passed back to failing. Because the same checks also run on the untouched repository, each regression carries exact provenance, not a post-hoc label: on the 586 runs where both readings exist, 18.3% are caught breaking a verified state. Interaction is a further blind spot, failing in 29.5% of runs whose static DOM already passes. ProductWebBench therefore scores a patch against the living system it lands in, not the target alone. 

---
# A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool 

**Authors**: Avrile Floro, Tamara Dhorasoo, Nils Holzenberger, Thomas Le Goff, Tiphaine Viard, Maria Boritchev  

**Link**: [PDF](https://arxiv.org/pdf/2610.31116)  

**Abstract**: This paper presents a dataset of 121,536 ground-reasons pairs across 86,464 rulings of the five civil chambers of the French Court of Cassation (2016-2025). An appeal on points of law cannot itself be appealed and often decides a case for good. Understanding how the Court rules is thus crucial and can help lawyers who draft those appeals. Inside a ruling, every ground of appeal (a distinct legal complaint against the appealed decision) has its own answer from the Court. This answer is called the reasons and it belongs to one of a few doctrinal families. Using a deterministic pipeline, we pair each ground with its reasons and attach the pair to its family, according to the established French taxonomy. We obtain a descriptive grammar of civil cassation that shows which grounds succeed, which legal texts they refer to, and how both evolve over the decade. We release the dataset, the code and an interactive consultation tool. 

---
# What Pairwise Evaluation Misses: Reward Scores across Prompts 

**Authors**: Mateusz Modrzejewski, Tomasz Radzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.31090)  

**Abstract**: Reward models score responses for preference optimization, filtering and ranking, and pooling scores across prompts needs them to be comparable. Yet pairwise evaluation compares two responses to the same prompt, so any score component shared by a prompt's responses cancels and goes untested. We study this component by comparing between-prompt variation in model scores with the same variation in human ratings of the same responses. We propose a human-referenced audit, built from a between-prompt variance share, a sibling-rating regression and prompt substitutions, and apply it to eighteen released scorers and six matched training conditions. We find that scores cluster by prompt more than the ratings do: on HelpSteer2, six reward models have between-prompt shares of 0.34 to 0.64 against a human share near zero. Replacing the original prompt with a fixed instruction keeps substantial clustering but weakens agreement with prompt-level human ratings, so clustering alone does not show agreement with humans. Part of that variation is useful: subtracting prompt means lowers the helpfulness of globally selected responses, and scores stay sensitive to a wrong prompt even when training shuffles prompt means. We propose that reward evaluation check between-prompt variation, sensitivity to a wrong prompt and usefulness for selection separately. 

---
# MQAScore: Fine-Grained Text-Music Alignment Evaluation via Large Audio Language Model 

**Authors**: Wei-Jaw Lee, Chun-Ping Wang, Xuanjun Chen, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31083)  

**Abstract**: Text-to-music alignment is commonly evaluated by the global similarity between text and audio embeddings, which collapses all described attributes into a single score and obscures which ones are missing. We instead introduce MQAScore, which verifies each atomic musical concept along five clip-level dimensions (genre, instrument, mood/theme, vocal, and usage) by the probability that a large audio language model (LALM) answers affirmatively to a binary question. To assess evaluator reliability, we construct verification labels from presence/absence annotations and LLM-guided concept substitutions, and propose the Sensitivity-Invariance Ratio, which measures how selectively an evaluator responds to meaning-changing substitutions. Experiments show that concept-level scoring improves both question-answering and similarity-based evaluators, and MQAScore can be a more reliable and interpretable fine-grained complement to global similarity for evaluating text-to-music alignment. 

---
# Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence 

**Authors**: Gabriel Lars Sabadin, Dr Nicola Harding, Sarah Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2610.31047)  

**Abstract**: Content warning: this paper quotes slurs and hateful statements from published test suites; the racial slur among them is masked.
Automated systems decide at scale which messages are harmful, and they decide unevenly. Classifiers label African American English as offensive up to twice as often as other English, the accounts of prominent drag queens can score as more toxic than those of white nationalists, and safety behaviour that holds in English collapses in low-resource languages. We argue that over-flagging and under-protection share one cause: deciding harm from vocabulary and surface form rather than from what a message does. We organise the evidence into seven failure modes and argue, as testable hypotheses, that harmful behaviours (a request for secrecy, a demand for payment, an attempt to isolate someone) travel across languages better than the words that express them. We propose an evaluation protocol that makes the gap between the best- and worst-served language communities a headline result, and show that the naive gap is inflated under equal service and that its usual bootstrap interval is unreliable. Applied to five open systems in eleven languages, from a keyword lexicon to two current multilingual guard models, the protocol finds that none reaches native-speaker accuracy at separating harmful function from harmless form (51 of 55 system-language pairs fall short); that greater capability raises discrimination in nine languages while leaving Arabic and romanised Hindi near chance, so the gap between the best- and worst-served languages grows with capability; that one system both over-flags and under-protects the same language most; and that a language a system reads worst can look the most fairly treated when only false positives are reported. 

---
# Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters 

**Authors**: Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2610.31017)  

**Abstract**: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximity or text embedding are not distinguishable from random ones on either probe, although that test cannot exclude an advantage of about four fifths of what a query's own adapter gains over a random one. At 16 documents or more, correctly scaled merges recover 87 % to 89 % of the specialists' open-book advantage but only 12 % to 16 % of their cloze advantage, with intervals that run from below zero to at most 32 %, against 94 % for joint this http URL also report four silent errors, each of which produced a clean, wrong result. 

---
# Multi-Tool Selection for Robotic Systems: Comparing Semantic Retrieval and LLM 

**Authors**: Miguel Sousa, Luis Ramos Pinto, João P. Matos-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2610.31013)  

**Abstract**: Natural-language interfaces allow users to command robotic systems without interacting directly with mission planners or robotics middleware. Their reliable use depends on robot tool routing, defined here as the selection of the smallest complete subset of predefined executable capabilities required to satisfy a request. This task is particularly difficult for compound instructions, where incomplete selection may omit necessary sensing or monitoring capabilities, while excessive selection may expose irrelevant tools. We compare four routing strategies: three progressively refined ChromaDB-based methods - fixed-threshold retrieval, relative-distance retrieval, and clause-aware aggregated retrieval - and a cloud-hosted 120-billion-parameter language-model baseline. The methods are evaluated on 164 manually annotated prompts organised into three complexity levels, from direct single-tool requests to compound requests requiring at least three tools. Using exact-match accuracy, completeness, precision, a completeness-weighted composite score, and routing latency, CDB-v3 achieves the best macro-average score among the retrieval-only methods, with 0.7822, compared with 0.8869 for the large-model baseline. Although the LLM provides higher routing quality, CDB-v3 exhibits substantially lower latency and can be deployed locally without requiring cloud connectivity. These results identify a quality-latency trade-off between direct large-model selection and clause-aware semantic retrieval for privacy-sensitive, connectivity-constrained, and resource-constrained robotic systems. 

---
# SecLog-Bench: A Scalable Benchmark for Cybersecurity Log Reasoning with Scaffolded Evidence 

**Authors**: Yifan Wang, Dongliang Xu, Xiaoyong Li, Jing Li, Peter S. Pan, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2610.31000)  

**Abstract**: Large language models (LLMs) have shown strong potential for cybersecurity, leading to growing interest in LLM-based automated cybersecurity log analysis. However, current datasets (1) are limited in scale; (2) rely mainly on single-source evidence; and (3) have only binary or multi-class attack labels, without dense annotations of the reasoning process. To address these limitations, SecLog-Bench has three key features. First, we build an extensible simulation platform that covers 29 attack scenarios, supports repeated execution and the integration of new scenarios, and enables scalable generation of cybersecurity logs with attack-stage labels. Second, we use multi-source logs, including audit logs, application logs, and network traffic, to jointly analyze cyber attacks. Third, we annotate each sample with the essential context for cybersecurity log reasoning and a structured step-by-step evidence chain. We evaluate eight mainstream LLMs on this benchmark for post-detection attack-progress reasoning. Results show that, when provided with scaffolded evidence, current LLMs exhibit modest capability in attack-progress reasoning, with the best model achieving only 44.44% accuracy and 45.07% Macro-F1. Their reasoning remains unreliable; i.e., the highest Stage Reasoning F1 (SRF1) score is only 54.87%. Moreover, even correct predictions may be accompanied by flawed reasoning, and reasoning quality is often misaligned with prediction accuracy. 

---
# Truthful by Omission: Click Thinning for LLM-Native Advertising 

**Authors**: Junhua Liu, Zhicheng Liang, Zejun Zhang, Langzhou He, Feng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2610.30999)  

**Abstract**: Generative AI assistants are becoming an advertising medium, embedding sponsored content in their answers. This setting departs from classical position auctions in two ways: an ad's click-through rate depends on the passage that hosts it, and because users stop at their first click, earlier ads take clicks from later ones. This competition for clicks makes truthfulness delicate: a higher bid can reshape the allocation and put a competitor ahead of the bidder, reducing the bidder's clicks. Existing mechanisms either attain a constant fraction of the optimal welfare (advertisers' total click value) but are not known to be truthful, or are truthful but their welfare guarantee degrades as the number of passages grows. We achieve both even when the answer and its editorial order are fixed before bidding. Assuming each ad's click-through rate when reached is known and unaffected by omitting other ads, we give a computationally efficient mechanism that is truthful in expectation and attains at least half the optimal welfare under free reordering.
Our key idea, click thinning, omits ads at random, showing earlier ads less often so that each ad's overall click probability no longer depends on the order of the ads. Welfare maximization then reduces to a tractable matching problem, and solving it exactly with VCG-style payments makes truthful bidding optimal in expectation. Characterizing the click probabilities that omission can achieve sharpens the approximation factor from $2$ to $2-(1-\delta)^{K-1}$ when at most $K$ ads are shown and each click-through rate is at most $\delta$. This factor is the best possible for any order-preserving rule, even one that ignores incentives and computation, so truthfulness and computational efficiency incur no additional worst-case loss. 

---
# LAST-V2X: Language-Driven Adaptive Scenario Synthesis and Testing for Communication-Induced Failures in V2X Safety Applications 

**Authors**: Keshu Wu, Hao Zhang, Rui Gan, Pei Li, Jiwan Jiang, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2610.30979)  

**Abstract**: Vehicle-to-everything (V2X) communication enables connected vehicles to make safety-critical decisions, but delayed, missing, or incorrect information can compromise driving safety. Systematically testing such failures is challenging because traffic interactions, communication conditions, and application responses must be jointly configured to produce meaningful scenarios. This paper presents LAST-V2X, a language-driven, adaptive framework for automated scenario synthesis and safety testing of V2X applications. LAST-V2X leverages large language models (LLMs) to translate natural-language test descriptions into executable scenarios grounded in roadway geometry, V2X communication semantics, and vehicle behavior. The framework integrates physical traffic simulation with spatially aware communication modeling and message-level fault injection to evaluate how information degradation affects vehicle decisions. To establish whether an observed safety failure is caused by communication rather than the underlying traffic conditions, LAST-V2X employs matched executions under degraded and ground-truth information. Furthermore, an execution-guided refinement mechanism combines parameter search and LLM self-reflection to identify ineffective test configurations, improve scenario effectiveness, and produce minimal reproducible failure cases. Experiments across diverse CARLA environments and V2X safety applications demonstrate the framework's ability to generate reliable scenarios, distinguish communication-induced failures from unrelated safety violations, and substantially improve failure discovery through adaptive refinement. These results establish LAST-V2X as a systematic approach to language-driven, causally verified, and adaptive safety testing for connected vehicle systems. 

---
# OpenAttention: One Percent Is Enough with Tail Proxies 

**Authors**: Hyunwoo Yu, Jongbeom Lee, Hyunbin Kim, Sungjun Park, Jaemin Choi, Jincheol Yang, Seunghun Moon, Bongjoon Hyun, Kyomin Sohn, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30951)  

**Abstract**: Sparse attention reads only a small subset of the key-value cache, and how to treat the unread tail remains a key challenge. Discarding the tail biases the output, and sampling it as previous approaches do fails at reads of 5% and below, where the tail likely holds heavy hitters the selector missed. Our key observation is that the simplest possible proxy, one token that folds the entire tail into its count-aware mean, already yields lower output error than a sample at every read behind a practical selector. We further show in closed form that at low reads sampling nearly doubles the weight error of discarding while a single proxy stays below it. Building on these findings, we explore approximating the dense attention distribution with a few proxies that target the tail. We propose OpenAttention, a pipeline of clustering, ErrorGate and top-k selection followed by a count-aware mean proxy. To implement clustering and selection both efficiently and effectively, we design training-free Query-Geometry Clustering (QGC) and a Query-Geometry Selector (QGS), which compress the keys while respecting the query directions. Additionally, unlike prior methods that rank clusters by centroid alone, ErrorGate allocates the exact read by the estimated weight error of each cluster. On RULER-32K HARD with Llama-3.1-8B, OpenAttention retains 99% of dense accuracy at a 1% read. It also achieves state-of-the-art accuracy across models, context lengths and datasets. Our customized attention kernel, this http URL, runs the decode attention stage 10$\times$ faster than dense attention at a 1% read. 

---
# RetiJev: Fast and Source-Grounded Ophthalmic Decision Models 

**Authors**: Bo Liu, Lianyu Wang, Meng Wang, Ching-Yu Cheng, Huazhu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30947)  

**Abstract**: Ophthalmic clinical decision support requires fast, accurate, and privacy-preserving inference, yet autoregressive large language models (LLMs) incur substantial latency and computational overhead for decisions among predefined actions. We therefore investigate a decision-oriented alternative to generative reasoning. Specifically, we first introduce RetiJev-Data, a source-grounded ophthalmology corpus comprising 78,158 clinical-decision and diagnosis questions derived from articles in PubMed Central and a MeSH vocabulary covering 531 eye diseases, and 50,074 guideline-grading questions spanning 11 grading systems, along with a 24,662-question evaluation benchmark. Based on this, we develop RetiJev, a family of 0.8B-9B decision models that directly score candidate answers without autoregressive decoding. Across all model sizes, RetiJev outperforms both a 27B LLM and the commercial Jev 1.13 decision model on guideline grading by 10.4-23.0 points. RetiJev-9B further achieves the highest accuracy on clinical decisions, improving over both baselines by 5.4-6.5 points while matching them on diagnosis. It answers a case in 158 ms on one GPU, 3.5x faster than LLM generation and 470x faster than reasoning. Models are released at this https URL. 

---
# Metacognitive Scaffolds Create Productive Friction for AI-Assisted Knowledge Work 

**Authors**: Viktor Kewenig, Lev Tankelevitch, Sean Rintel, Advait Sarkar, Abigail Sellen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30945)  

**Abstract**: Generative AI imposes new metacognitive demands on knowledge workers as their work shifts from production to planning, evaluation, and orchestration. In a randomized, between-subjects experiment on AI-assisted writing (297 workers, 594 essays), we test dynamically-generated metacognitive scaffolds supporting goal specification and task decomposition. We find that scaffolds improve writing quality, with this boost persisting in short-lag transfer when scaffolds are removed. We also find evidence of improved topic recall three days later. These effects are consistent with observed changes in how people learn to prompt AI systems after exposure to scaffolding. However, scaffolded participants rate the AI as less helpful and more cognitively demanding even as they perform better, suggesting that subjective ease can understate the value of support. Our findings make the case for metacognitive scaffolds in AI-assisted knowledge work, revealing a productive friction worth designing for as workers increasingly delegate work to agentic AI. 

---
# GateFormer: Forecast-Space Decomposition for Intra-Hour Solar Irradiance Prediction from Sky Images 

**Authors**: David Hamlyn, Sunny Chaudhary, Jonathon Hare, Tasmiat Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2610.30942)  

**Abstract**: Sub-hourly dispatch, balancing, and storage control increasingly depend on forecasts that anticipate rapid variability in solar generation. Accurate sky-image-based irradiance forecasting requires a combination of visual cloud dynamics with recent irradiance state while adapting to different short-term irradiance-change regimes. To address this, we introduce GateFormer, a GHI-conditioned dual-decoder ViT for intra-hour global horizontal irradiance (GHI) forecasting through forecast-space decomposition. GateFormer embeds observed GHI into the visual token stream, allowing interaction with sky-image patch features through self-attention. From this shared representation, a Trajectory Head predicts future GHI sequence, while a Delta Head predicts final-horizon change from the latest observed GHI. The two final-horizon forecasts are adaptively weighted in prediction space, with magnitude- and direction-supervision targeting the remaining error. On the NREL-TSI benchmark, GateFormer attains 125.01 W/m^2 RMSE at +30 min, improving over a state-of-the-art transformer baseline by 9.6% under paired-timestamp evaluation. Across NREL-TSI, Chilbolton, and SIRTA, GateFormer achieves the lowest final-horizon RMSE among the evaluated baselines, with forecast skill of 22.97-28.86% relative to smart persistence. Component ablations show adaptive weighting outperforms both individual decoders and fixed averaging, distinguishing GateFormer from conventional forecast combination, while the supervised correction further reduces RMSE by 1.68 W/m^2 and MAE by 4.86 W/m^2. Ramp-regime diagnostics show larger gains during operationally important positive ramps, reducing RMSE by 26.27 W/m^2. These results identify forecast-space decomposition as an alternative forecasting formulation, with potential relevance to other multimodal short-horizon forecasting tasks and deployment using only sky images and measured GHI. 

---
# Vision Transformers for Estimating Irradiance Using Data Scarce Sky Images 

**Authors**: David Hamlyn, Sunny Chaudhary, Tasmiat Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2610.30940)  

**Abstract**: Accurate estimation of diffuse horizontal irradiance (DHI) is critical for optimising photovoltaic system performance and energy forecasting yet remains challenging in regions lacking comprehensive ground-based instrumentation. Recent advancements using Vision Transformers (ViTs) trained on extensive sky image datasets have shown promise in replacing costly irradiance measurement equipment, but the scarcity of long-term, high-quality sky imagery significantly restricts practical implementation. Addressing this critical gap, this study proposes a novel dual-framework approach designed for data-scarce scenarios. First, calculated atmospheric parameters, including extraterrestrial irradiance and cyclic time encodings, are integrated to represent sky conditions without utilising any instrumentation. Next, a sequential pipeline initially predicts synthetic global horizontal irradiance (GHI) and uses it as a feature, to refine DHI estimation. Finally, a dual-parallel architecture simultaneously processes raw and overlay-enhanced fisheye sky images. Overlays are generated through unsupervised, physics-informed cloud segmentation to highlight dynamic sky features. Empirical validation is performed using data from the Chilbolton Observatory, chosen for its temperate climate and frequent cloud variability. To simulate data-scarce conditions, models are trained on a single month (e.g., January) and evaluated across a temporally disjoint, full-year test set. Under this setup, the sequential and dual-parallel frameworks achieve RMSE values within 2-3 W/m^2 and 1-6 W/m^2, respectively, of a state-of-the-art ViT trained on the complete dataset. By combining physics-informed modelling with unsupervised segmentation, the proposed method provides a scalable and cost-effective solution for DHI estimation, advancing solar resource assessment in data-constrained environments. 

---
# Low-Latency MoE Inference in Space Data Centers: A Wireless-Power-Transfer Perspective 

**Authors**: Haotian Zheng, Zhanwei Wang, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30923)  

**Abstract**: Space data centers (SpaceDCs) are emerging as platforms for large language models (LLMs), supported by continuous solar-energy harvesting and high-speed interconnections. The mixture-of-experts (MoE) architecture supports distributed LLM inference by deploying experts across satellites. However, skewed expert activation creates uneven computation power demands despite uniform solar-power supply at the dawn-dusk sun-synchronous orbit (SSO). Consequently, active experts remain power-constrained while harvested power at idle experts is underused. These reduce inference efficiency and increase latency. To address this supply-demand mismatch, we propose the SpaceWPT framework, which uses inter-satellite laser-based wireless power transfer (WPT) to redistribute harvested power. The key design challenge is to balance the computation demands of power-donating and power-receiving satellites under WPT losses: transferring more power accelerates computation at the receiver but leaves less power for the donor. After deriving design principles that simplify WPT control, we characterize the resulting latency tradeoff through a tractable surrogate of average inference latency. Minimizing the surrogate yields a novel two-sided water-filling strategy that determines whether each active expert donates, retains, or receives power according to the water level and its WPT link efficiency. Simulations with Mixtral-8x7B and Phi-3.5-MoE show up to 29.25% lower inference latency than without WPT, with larger gains under more compact formations and higher transfer efficiencies. 

---
# MIG-COW: Transferable Adversarial Attacks on Deepfake Detectors via Gradient Decomposition 

**Authors**: Wonjune Seo, Joonhyuk Baek, Yeseong Jung, Saerom Park  

**Link**: [PDF](https://arxiv.org/pdf/2610.30914)  

**Abstract**: Despite recent advances, deepfake detectors remain vulnerable to adversarial examples, particularly in diverse, real-world settings. We propose MIG-COW, a novel adversarial attack framework that generates highly generalizable and visually imperceptible adversarial examples. By combining momentum-integrated gradients with a consensus-orthogonal decomposition, MIG-COW captures both shared and model-specific vulnerabilities across heterogeneous CNN and ViT detectors. On the AADD-2025 Challenge benchmarks, MIG-COW achieves a 99.96% white-box attack success rate (ASR) with high perceptual similarity (SSIM), significantly outperforming existing baselines. However, its limited 7.16% ASR against official black-box targets - despite achieving the best overall score - highlights the ongoing challenge of transferability. We also demonstrate that incorporating low-performing but diverse models in the ensemble can degrade attack effectiveness, underscoring the need for careful surrogate model selection in real-world adversarial settings. 

---
# Query-Time Calibration of CLIP Features for Open-Vocabulary 3D Segmentation 

**Authors**: Dominik Engel, Yousef Rajeh, Leon Sick, Ivan Viola  

**Link**: [PDF](https://arxiv.org/pdf/2610.30908)  

**Abstract**: Lifting vision-language features such as CLIP into 3D Gaussian reconstructions enables open-vocabulary segmentation through natural-language queries. Although CLIP represents a broad range of semantic concepts, the lifted features of individual scenes can exhibit strong directional concentration, while raw cosine similarity does not account for the scene-specific feature distribution. We show that substantial segmentation gains can be obtained by changing how a fixed lifted feature field is compared with text. Our method partially subtracts a shared scene direction from visual features and text prototypes and whitens text prototypes using a fixed reference vocabulary of generic terms. It then uses soft clustering for multiclass prediction and CSLS with a relevance margin against the reference vocabulary for single-query prediction. Beyond reducing directional concentration, scene centering improves class-relative feature separation in every evaluated scene. Across seven lifting methods and four datasets, the resulting query-time calibration consistently improves mean intersection over union in both multiclass and single-query segmentation. These gains require no retraining, source-image re-encoding, or feature re-lifting, demonstrating the value of scene-conditioned comparison for making better use of existing lifted representations. 

---
# MOER: Mix-Policy Optimization under Expert Prefix Guidance with Selective Rollback 

**Authors**: Xuyang Zhong, Jiuding Yang, Yiwen Guo, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30873)  

**Abstract**: Mix-policy optimization incorporates off-policy expert data into the reinforcement learning (RL) loop to transcend the capability boundary of on-policy RL. However, existing methods either provide static full-trace guidance that suppresses self-exploration, or adopt expert prefix guidance without correctness guarantees, leading to suboptimal performance or even training collapse. In this paper, we propose MOER, a mix-policy optimization framework that harmonizes expert guidance with self-exploration. MOER injects a decaying expert chain-of-thought prefix into the prompt and lets the model complete the remaining reasoning, with the prefix length progressively annealed to encourage self-exploration. A selective rollback mechanism replaces failed completions with the original full expert trace, guaranteeing the correctness of every off-policy trajectory. Furthermore, a harmonized mix-policy objective optimizes on-policy tokens with a GRPO loss while supervising both expert prefix and completion tokens with a trust-region SFT loss, substantially stabilizing training. Extensive experiments show that MOER achieves the highest average scores in mathematical and general domain reasoning tasks, outperforming all basic and mix-policy baselines, and generalizes consistently across diverse tasks, model scales, and base model categories. Codes are available at this https URL. 

---
# Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience 

**Authors**: Abhijit Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2610.30867)  

**Abstract**: Agents that call tools from a shared catalog can pool what they learn about which tool serves which request. We measure what pooled usage examples do to tool retrieval, reporting separately the held-out tools, for which no agent has usage, and all other tools. Adding usage examples to the index as extra entries for used tools raises Hit@5 (whether a gold tool is in the top five) for those tools and lowers it for held-out tools, by a large margin on two of three catalogs and with both embedders we tested. Giving used tools the same number of equally long entries filled with other tools' text lowers held-out Hit@5 about as much, but gives little or none of the gain. We call our method of scoring usage as its own field SYNAPSE. On StableToolBench, the held-out gold tool appears in its top five more often than with a router that reserves slots for usage (0.785 against 0.683), and this held-out lead holds on all three catalogs with both embedders. This protection persisted when we replaced each tool's usage with other tools' text, and a constant score offset for tools with usage gives similar held-out Hit@5 in most settings. The gain for tools with usage, in contrast, needs usage about the right tool. The router keeps more of the gain on labeled queries on TOUCAN with both embedders and on ToolRet with one, so the better choice depends on the workload. Splitting the usage among ten simulated agents shows that pooling helps exactly as much as giving one agent the same records. 

---
# Direct Forcing: A Single-Stage Training Recipe for Few-Step Autoregressive Text-to-Video Generation 

**Authors**: Haojun Xu, Jie Huang, Xin Lu, Mingchen Zhong, Zihao Fan, Linjiang Huang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30861)  

**Abstract**: Few-step autoregressive video generation enables low-latency streaming, yet prevailing training pipelines separate causal adaptation from distribution matching. Recent representation-based text-to-video methods retain this separation by starting from ODE-initialized checkpoints, even when the target distribution is defined directly by reference videos. We introduce Direct Forcing, a single-stage framework that uses reference videos both as paired examples of how a history continues and as a target population for self-generated rollouts. Causal Continuation Alignment learns paired spatiotemporal correspondence under reference histories, while Rollout Distribution Matching uses maximum mean discrepancy to align the representation distributions of self-generated and reference videos. Together, these complementary signals allow a pretrained video model to learn causal continuation and the target video distribution in a single training stage. Dynamic Gradient Balancing adapts the relative gradient contributions of continuation alignment and distribution matching during training. Starting from bidirectional Wan2.1-1.3B, Direct Forcing learns a four-step autoregressive generator in 250 updates without a separate initialization stage or online diffusion-score supervision, achieving 84.63 VBench Total and 86.01 Quality. Ablation studies further demonstrate the benefits of dynamic gradient balancing and reveal distinct representation preferences: deeper features benefit continuation alignment, whereas intermediate features yield better performance for rollout distribution matching. 

---
# Bifurcations from Weight Space: JEPAWG on Parametric Dynamical Systems 

**Authors**: Tobias Göbel, Miranda C.N. Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2610.30847)  

**Abstract**: Trained network weights are increasingly treated as data, but there is rarely ground truth about what a population of models should encode, which hampers systematic validation of weight-space methods. This makes parametric dynamical systems a natural test-bed: the dynamics change qualitatively at submanifolds of parameter space that are, in well-studied families, known analytically or numerically. Fine-tuning a forecasting network at each point of a parameter grid yields a model zoo indexed by a physical parameter, against which weight-space diagnostics can be checked.
We build such model zoos for four dynamical systems and analyse them with JEPAWG, a recently proposed joint-embedding hypernetwork built around a learned parameter--weight representation. It was first introduced for neural samplers for lattice field theories, where its latent geometry located the phase transition and estimated the 2D Ising critical exponent. Applied to dynamical systems, JEPAWG is competitive on forecasting with a hypernetwork trained directly for that task, despite being supervised on the parameter--weight correspondence rather than on forecasts. Beyond forecasting, the norm of the parameter-derivative of the decoded weights locates Hopf points and curves, tracks a period-doubling cascade closely enough to give a qualitative estimate of the Feigenbaum constant, and detects a global bifurcation, which has no eigenvalue signature and is therefore invisible to linearisation. As such, JEPAWG provides interpretable insight into the dynamics that is not available from other hypernetworks. Together, these results indicate that JEPAWG is a workable instrument for reading physical structure off a population of trained networks. 

---
# One Evaluation Is Enough: Inversion-Free Image Editing with Trajectory Correction 

**Authors**: Lifan Jiang, Taiyang Chen, Senlin cheng, Bolei Xu, Xiaobo Li, Can Ye, Dehai Zhao, Deng Cai, Boxi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30827)  

**Abstract**: Real-image editing commonly relies on inversion and iterative sampling, while efficient one-step editors often stabilize a large update by estimating and averaging editing fields at multiple noise levels. We ask a stricter question: is a single temporal field evaluation sufficient for reliable inversion-free editing? Our answer rests on a previously underused duality in a variance-preserving (VP) one-step model: one source-target query at a shared noisy state simultaneously determines a local angular semantic tangent and a clean endpoint displacement. We integrate the tangent over the analytically known remaining VP angle to obtain a nominal one-step transport, and use the endpoint displacement to correct its long-horizon extrapolation through a constrained projection. This transport-first, correction-second construction yields Endpoint-Calibrated Transport (ECT), a closed-form, training-free editor requiring only one noise level, two text conditions packed into one denoiser batch, and one latent update. ECT requires no image inversion, trajectory integration, second temporal observation, or iterative optimization, and can be directly applied to pretrained epsilon-prediction VP one-step generators. Across the reported benchmark settings, ECT delivers stable gains in editing quality within an extremely short editing time. 

---
# Neural Spectral Capacity with Residual Context for Heterogeneous Resource Allocation within Transformers 

**Authors**: Chenyu Zhu, Ruoyu Zhao, Zhichao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30823)  

**Abstract**: Predicting relative trained performance from architectural specifications can guide design when training costs limit experimentation. Neural Spectral Capacity (NSC) provides such a heuristic, but scores projections independently and assigns identical scores to compatible reorderings of the same layer configurations. We introduce Residual-Aware Neural Spectral Capacity (ResNSC) for Transformers, using a specification-derived statistical reference for accumulated residual updates to determine the input and residual-state statistics in each projection's capacity calculation, while retaining additive evaluation without data or instantiated weights. When residual widths vary, earlier choices affect later scores, so we augment NSC's allocator (NSC-DP) with residual history beyond its layer-and-budget state. ResNSC-DP jointly allocates residual and feed-forward widths under a resource budget, with certified bounds on the best numerically evaluated ResNSC score within a specified finite candidate set. At the 121M and 374M pretraining scales, selected architectures achieve lower mean perplexity than uniform, variable-width, and tapered allocations on FineWeb-Edu, WikiText-2, and LAMBADA at comparable parameter budgets. In inherited-weight LLaMA-7B pruning using the LoNAS supernet, ResNSC-DP improves eight-task average accuracy over NSC by 3.28--5.46 percentage points across eight matched parameter budgets. Paired reversals preserve the projection shapes and NSC scores yet worsen performance in both settings, exposing performance differences hidden by position-independent scoring. The code for this work is available at this https URL. 

---
# VLTest: Finding Joint Image-Text Failures in Vision-Language Models 

**Authors**: Bin Duan, Jintao Lin, Guowei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30794)  

**Abstract**: Vision-language models (VLMs) are increasingly embedded in software systems. Testing them raises two problems. Mutating either input can change the expected answer, so an output change is not by itself a failure. Some failures appear only when both inputs are mutated; attributing them to the combination requires testing each mutation alone. We present VLTest, a black-box VLM testing approach. It generates image variants by editing VQGAN latent codes and text variants with a rewrite-expand-prune pipeline, filters variants likely to change the expected answer, and tests each variant alone before testing combinations, so every joint failure has four matched cells. A VLM judge confirms a failure only after checking that the input keeps the expected answer. On three tasks with eight VLMs, including two commercial API models, the oracle confirms 81.7% of the candidates, with 88.8% to 94.6% precision against human labels, and four-cell confirmation rejects 22.5% of joint candidates. Under the same budget, VLTest's generators find 1.8 to 2.5 times as many confirmed failures as SINVAD and TextFooler, and its joint tests find 91 to 197 attributed joint failures per 1,000 joint calls, 1.3 times as many as the same pairing applied to the baselines' variants. 

---
# Abstract Priorities, Concrete Rewards: How Stated Preferences Relate to What People Reward in AI Responses 

**Authors**: Zihan Chen, Lei Nico Zheng, Di Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30763)  

**Abstract**: AI assistants built on large language models (LLMs) are trained on human feedback and increasingly personalized to users' stated preferences. Yet how closely these preferences reflect what people reward in responses remains unclear. We compare stated and revealed preferences in the public PRISM dataset, in which 1,396 participants reported how much seven response attributes mattered to them and then rated 29,347 responses from 21 LLMs. Among responses to the same prompt, features the survey never asked about, such as response length, formatting and refusal language, were at least as strongly associated with ratings as text proxies of the surveyed attributes. Whether stated priorities matched ratings depended on how attributes were measured: participants who prioritized an attribute favoured responses from models that others perceived as strong on it, but not responses that scored higher on text proxies for it. Written feedback was more telling: participants who asked for shorter answers, even in earlier conversations, gave length less weight in their ratings, and those asking for longer answers gave it more. Their relative priorities over the surveyed attributes, however, did not predict this weight. Reasons given for choices reflected perceived attributes more than measured features. Participants also differed stably in how much they weighted several concrete features, especially length, but with a median of 6 conversations per participant, these weights were estimated poorly. The largest gain from personal information in predicting held-out choices was 0.13 percentage points, compared with 1.11 for a stronger pooled model. Stated priorities thus track perceived attributes of AI responses but say little about the concrete features that ratings reward, a distinction that matters for preference elicitation and AI personalization. 

---
# The Delegation Divide: An Open Index of AI-Agent Delegability across Occupations and Countries 

**Authors**: Cristian Espinal Maya, Santiago Jiménez Londoño  

**Link**: [PDF](https://arxiv.org/pdf/2610.30752)  

**Abstract**: 35.2% of task time in world employment is at least partly delegable to AI agents in 2026, and only 2.5% can be handed over in full. Existing measures score exposure to generative AI, mostly for the United States. We build the Agent Delegability Index (ADI): two language models rate all 18,796 O*NET 30.2 task statements on a frozen, preregistered rubric of delegation levels and barriers; tasks are mapped to 433 ISCO-08 unit groups and weighted by ILOSTAT employment in 146 countries. Four findings. First, delegable employment rises with income, from 22.5% in low-income to 48.7% in high-income countries, a slope of 7.23 percentage points per log point of GDP per capita. This delegation divide is compositional: the physical barrier dominates the jobs poorer countries employ most. Second, at the occupation level ADI tracks exposure closely (r = 0.93 with Eloundou et al. and 0.89 with Felten et al.), above the preregistered bound of 0.80; ADI adds delegation levels and barriers on top of exposure, for the world, not a new ranking of occupations. Third, full delegation is rare and peaks for tasks that take a human about twelve minutes (11.8 minutes). Fourth, the preregistered adoption gap is not supported: across 101 countries, delegability and observed AI usage correlate at r = 0.86, and the gap between them is not wider in middle-income countries (p = 0.27). Model agreement is kappa = 0.77; a blind human rating of 300 tasks agrees with the model consensus at kappa = 0.85, and an agentic panel with web search at kappa = 0.87. We release the index, the atlas, the prompts and an MCP server at this https URL. 

---
# Learning from Next-Frame Prediction: Towards Unified and Scalable Visual Representation Learning 

**Authors**: Yifeng Yao, Jinghan Li, Zhihao Zhang, Yajiao Xiong, Yang Jin, Kun Xu, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30724)  

**Abstract**: Video provides a natural source of supervision: what happens next serves as a learning signal for visual world modeling. Yet masked video pretraining focuses on recovering missing content from surrounding context rather than predicting the future from the past. Autoregressive methods instead predict forward one token at a time, but have yet to match joint-embedding methods in representation quality. This gap motivates us to revisit the granularity and space of future prediction. In this work, we introduce VANE, a self-supervised visual pretraining framework built on next-frame prediction in embedding space. A causal encoder summarizes past observations for a state predictor that infers the representation of the entire next frame, making temporal evolution the primary learning signal. This prediction is carried out in a fixed embedding space provided by a frozen semantic tokenizer, emphasizing scene content and structure. To retain fine-grained spatial detail alongside semantic content, a lightweight auxiliary generative decoder reconstructs future-frame patches conditioned on the predicted representations. To unify visual modeling across images and videos, we construct a local-to-global crop sequence for still images, allowing the same encoder to predict newly revealed regions from earlier crops under the same objective. Experiments show that VANE improves with model and data scaling, achieving SOTA on most anticipation, recognition, and dense perception benchmarks among unified encoders. Especially on EPIC-KITCHENS-100 action anticipation, it surpasses V-JEPA 2.1 by 7.1 points in mean-class recall@5. These results demonstrate that next-frame prediction in embedding space is effective and scalable for visual world modeling. 

---
# Meaning Contracts: Local Semantic Geometry in Large Language Models 

**Authors**: Andrés Daniel Godoy Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2610.30715)  

**Abstract**: This paper studies the local geometry of semantic meaning in large language models. Current representation analyses ask whether meanings are close in embedding space, or whether linear probes separate semantic classes, but they do not characterize the within-concept local geometry: what is invariant across realizations of a single semantic factor, what varies, and what changes when meaning is refined. This paper characterizes that geometry through two controlled probes evaluated on four architecturally diverse decoder-only LLMs (Gemma 4, Llama 3.2, Qwen 2.5, Pythia). A cross-lingual probe uses language as a perturbation to separate exact meaning matches, same-factor realizations, nearby concepts, and out-of-domain contrasts. A refinement probe separates specificity-bearing semantic narrowing from high-probability attributive elaboration and adversarial controls; both probes are validated by blind human judgments. Four findings recur across all models: under mean pooling, once dominant variance directions are removed, sentence pairs order by semantic relatedness (same factor, then same domain, then different domain) without overlap over a band of layers; semantic factors organize around an invariant core surrounded by structured realization-specific variation; refinement produces directed displacement into tighter neighborhoods, including against controls built on the same base noun; and sentence-level and token-local readouts expose complementary geometry (volume contraction at the sentence level, sharper anisotropic endpoints at token-local readout). We interpret these results as evidence that LLM semantics are organized as local anisotropic neighborhoods in which meaning contracts under refinement. 

---
# ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling 

**Authors**: Yang Liu, Zhiqiang Yuan, Shu-Dong Huang, Jiancheng Lv, Liwei Yang, Qiwei Yan, Kailin Lv, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30705)  

**Abstract**: Text embedding and text generation are two of the most useful capabilities of modern large language models, yet they are typically provided by separate specialized models (an embedder and a generator) trained with incompatible objectives and deployed side by side in retrieval-augmented generation and agent systems. Recent unified embedding-generation models such as GritLM share parameters between the two capabilities, but they either switch attention regimes between modes or optimize embedding and language modeling as parallel objectives that never supervise each other. We argue that these designs are only partially unified: the two capabilities coexist on the same weights, but do not shape each other. We propose ULME, a post-training framework that unifies embedding and language modeling on a causal decoder-only LLM through predictive coupling. ULME inserts a chain of learnable slots between the query and the answer, reads a semantic embedding and a reasoning embedding from two designated slots, and trains three objectives on a shared causal backbone: a contrastive loss anchoring the semantic slot to document representations, a language-modeling loss supervising answer generation, and a stop-gradient predictive loss that aligns the reasoning slot with a pooled summary of the answer, so that generation actively shapes what the embedding encodes. Across two model families (Qwen2.5, Qwen3) and eight scales spanning 0.5B-8B parameters, a single ULME checkpoint improves MTEB by 3.3-4.1 points over the strongest open-source baseline, tightens BRIGHT and AdvBench-IR simultaneously, and remains on par with a generation-only fine-tune on standard accuracy benchmarks (MMLU, GSM8K, MMLU-Pro, HumanEval), demonstrating that genuine unification yields mutually reinforcing rather than merely coexisting capabilities. 

---
# LoRA Superposition across Pretrained Models 

**Authors**: Keigo Watanabe, Hiroshi Kera, Kazuhiko Kawamoto  

**Link**: [PDF](https://arxiv.org/pdf/2610.30691)  

**Abstract**: Low-rank adaptation (LoRA) is usually tied to the pretrained backbone on which it is learned. We ask whether LoRAs trained independently for the same task on different pretrained backbones can be summed and applied unchanged to all contributing backbones. We call this construction LoRA superposition. Across seven ViT-B/16 CLIP backbones and eight image-classification tasks, the exact sum improves all 56 backbone-task pairs and achieves a mean accuracy gain of 30.21 percentage points over the unadapted backbone. The exact sum achieves a mean accuracy only 0.28 percentage points below that of individually trained solo LoRAs. In contrast, a leave-one-model-out (LOMO) sum produces a mean gain of only 0.06 percentage points over the unadapted backbones. Backbone-specific solo LoRAs are nearly orthogonal in parameter space, while pairwise cross-backbone responses are strongly concentrated on the diagonal. These results support a component-wise non-interference hypothesis, under which the frozen backbone acts as an implicit selector of its own component without explicit routing or cross-model alignment. 

---
# From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots 

**Authors**: Chenhan Lyu, Kit Wislocki, Sarah Elizabeth Stoeckl, Maitreyi Sinha, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.30690)  

**Abstract**: Large language models (LLMs) deployed in mental-health contexts can be evaluated for safety failures, but the resulting judgments are often difficult to inspect or explain. Common approaches to improving safety rely on prompt engineering, LLM-as-a-Judge, and Constitutional AI. In this work, we describe a clinician-led alternative in which two clinicians define, finalize, and adjudicate the four safety checks of a rule base, which is formally verified for internal consistency and applied as a guardrail over LLM outputs. Here and in the title, clinician-led refers to this rule base, not to the fact extractor or the uncertainty default, which are ours. Every verdict of the resulting verification layer is recomputable from a named rule and sixteen facts. Against an LLM judge matched in model, revision, quantization, rule text, and decoding but without the layer's computed 988/911 fact, the largest disagreement class, the satisfied-rule class, is structured: the Llama-3.3-70B judge, on its primary prompt, demands a crisis resource from 55 of the 65 replies (84.6%) that already contain one, while the same model, asked as a fact extractor, reports a recognized resource present on 54 of the 55. The verification layer never raises that demand, by construction. We report with equal prominence where this judge is stronger: fewer false crisis alarms on the one reference-scored check. We also report the measured cost (a specification revision raised block-class outcomes on held-out boundary prompts of an over-refusal benchmark from 6.0% to 16.0%, chiefly by changing fact extraction), the clinicians' decision to keep our uncertainty default, and the analysis of the result. The contribution is not a safer verdict. It is a guardrail whose disagreements with the judge localize to named rules and facts and whose measured costs trace to named facts and the uncertainty default. 

---
# Spectral Smoothing and Spiking Perturbations Reconfigure Capability Profiles and Selective Task Performance in Large Language Models 

**Authors**: Qince Luan, Jianhui Xu, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30676)  

**Abstract**: Singular-value spectra encode important structure in transformer feed-forward projections, yet their causal role in downstream behavior remains underexplored. We study post-hoc spectral smoothing and spiking interventions on the up_proj and down_proj matrices. Positive perturbation stretches deviations from the geometric mean, making large singular values larger and small ones smaller (spiking), and vice versa. The intervention preserves singular vectors and the geometric mean of the singular values. Across five open Llama and Qwen instruction models, we evaluate the perturbed systems on capability-labeled subtasks, standard and chain-of-thought reasoning benchmarks, judge-scored truthfulness, hallucination, and safety tasks, and TRAIT personality probes. To interpret score shifts, we fit responses in a three-dimensional capability basis consisting of factual knowledge, language understanding, and deductive reasoning. The induced changes suggest structured rather than purely random sensitivity: large positive perturbations more reliably suppress language and deductive responses, moderate negative perturbations can sometimes improve factual or task-local performance on the evaluated grids, and sensitivity varies strongly by model family, benchmark module, and underlying layerwise spectral inequality. These results position spectral perturbation as a lightweight intervention for probing how capabilities are organized and traded off in large language models. 

---
# One-shot federated unsupervised domain adaptation with Smoothed Knowledge Distillation and teacher refinement 

**Authors**: Ali Abedi, Q. M. Jonathan Wu, Ning Zhang, Farhad Pourpanah  

**Link**: [PDF](https://arxiv.org/pdf/2610.30646)  

**Abstract**: Federated unsupervised domain adaptation (FUDA) addresses the challenge of adapting models to an unlabeled target domain using decentralized source domains while preserving data privacy. However, existing FUDA methods typically require multiple communication rounds, rely on complex aggregation strategies, and often struggle with noisy pseudo labels and inconsistent source knowledge. To address these challenges, we propose SKD-ETR, a novel one-shot FUDA framework that combines Smoothed Knowledge Distillation (SKD) and Exponential moving average-based Teacher Refinement (ETR). SKD trains a student model on the target domain using smoothed soft pseudo labels generated by the ensemble of source models. This reduces overconfidence, mitigates noise, and improves robustness. ETR further refines each source model by interpolating its parameters toward the student via exponential moving average (EMA), thereby transferring target-domain knowledge back to the teachers. This bidirectional refinement enhances pseudo-label quality and student generalization without additional communication overhead. SKD-ETR eliminates complex aggregation by initializing the student model randomly and performing a single-round distillation process. Extensive experiments on OfficeHome, Office-Caltech, and DomainNet demonstrate that SKD-ETR achieves competitive performance while being communication- and computation-efficient, and robust under noisy supervision. The code for this paper is available at this https URL. 

---
# Asking Less, Adapting Better: Feedback-Efficient Test-Time Adaptation for Vision-and-Language Navigation 

**Authors**: Zixu Wang, Huaquan Zhang, Shoukai Xu, Jinwu Hu, Zhiquan Wen, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2610.30643)  

**Abstract**: Vision-and-Language Navigation (VLN) requires agents to follow instructions and navigate to target locations. However, current VLN agents often suffer severe performance degradation in unseen environments due to distribution shifts. Test-Time Adaptation (TTA) offers a promising paradigm to bridge this gap, yet existing methods that rely on unsupervised entropy minimization can propagate errors through the trajectory, causing cascading failures in navigation. Recent methods mitigate this by incorporating human feedback, but they apply trajectory-level supervision uniformly across all steps without localizing the actions is truly informative, leading to inefficient feedback utilization and mismatched optimization granularity. To address these limitations, we propose FETA, a Feedback-Efficient Test-Time Adaptation method that improves both feedback efficiency and optimization reliability. To reduce unnecessary human interventions, we introduce a hierarchical sample selection strategy to localize informative samples. Specifically, we first filter trajectories whose normalized policy entropy exhibits large fluctuations. Within each selected trajectory, we further select the critical action via an entropy-gradient saliency criterion and query a single binary feedback on its correctness. A discrepancy-aware entropy optimization objective then updates the policy only at the queried step, with adaptive scaling that assigns larger relative weights to samples with greater prediction-feedback discrepancy. Extensive experiments across multiple benchmarks demonstrate that FETA achieves state-of-the-art performance with fewer human interventions, while remaining robust under noisy feedback and continual adaptation settings. 

---
# Interpreting Model Reasoning through Behavioral Counterfactuals 

**Authors**: Shawn Im, James Oldfield, Adel Bibi, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30630)  

**Abstract**: Faithfully understanding the internal process that occurs as models form their chain of thought is an important problem with the rise in model capabilities. However, this remains challenging due to the variance in sampled responses, making it difficult to isolate the effects of interventions and compare model reasoning systematically. We propose a method for localizing and analyzing how a given intervention affects the model's reasoning through behavioral counterfactuals: given a fixed response, how does introducing an intervention to the prompt change the next-token distributions? We train a module to edit the residual stream such that it minimizes the KL divergence between the output distributions for a fixed response under a source prompt and its counterfactual target prompt. We find that matching the output distributions recovers the counterfactual behavior more effectively than standard activation reconstruction objectives and validate that attribution of the trained module outputs is effective for identifying causally important locations for counterfactual recovery. The attribution maps and visualizations of the downstream effects of model edits provide an interface for generating hypotheses about the original model's computation, which can then be tested with activation patching. In a case study on Gemma-4-E4B-it and DeepSeek-R1-0528-Qwen3-8B, we investigate how a suggestion influences model reasoning even when it is not verbally acknowledged. Our analysis identifies representations at the question statement and filler tokens that mediate a substantial part of the suggestion's influence, with downstream effects such as changing the framing of the justification or even the final answer. 

---
# Future of Time Series Anomaly Detection: From Historical Review to Unified Outlook 

**Authors**: Zhijie Zhong, Zhiwen Yu, Pengyu Li, Liujian Zhang, Qiaoli Ni, Jiang Jun, Jianming Lv, Kaixiang Yang, C. L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30628)  

**Abstract**: Time series anomaly detection (TSAD) is a foundational problem in data mining, aiming to identify observations that deviate significantly from nominal temporal patterns. Driven by escalating data volumes and application complexity, recent research has yielded diverse architectures, including diffusion models, state space models, and large language models (LLMs). However, existing surveys provide limited coverage of these emerging designs and lack a unified treatment of multivariate anomalies, cross-domain unified modeling, metric reliability, and contemporary evaluation challenges. To bridge these gaps, this survey delivers three core contributions: (1) We establish a refined anomaly categorization and a structured taxonomy of paradigms reflecting modern advances, comparing detection mechanisms and charting developmental trajectories across paradigms. (2) We provide an extensive analysis of evaluation methodologies and conduct comprehensive experiments to assess metric fidelity, complemented by an open-source benchmark repository encompassing algorithms, datasets, and evaluation tools. (3) We outline key open challenges, offering forward-looking perspectives on dynamic thresholding, model interpretability, zero-shot transfer, multi-modal alignment, and foundational models for time series. By addressing existing theoretical and empirical limitations, this survey establishes a coherent framework to guide future TSAD research and practical deployments. 

---
# Impacts of generative AI skill development on academic adaptation and learning outcomes among university students 

**Authors**: Lei Fan, Fangxue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30599)  

**Abstract**: Understanding how university students perceive the development of their generative artificial intelligence (GenAI) skills over time can clarify how GenAI use relates to learning and academic outcomes. This study examined perceived skill-development trajectories, changes in learning practices, and associated academic gains using a retrospective questionnaire with 313 analyzed responses and interviews with 18 students across subject areas. Descriptive and comparative analyses and exploratory structural equation modeling (SEM) were integrated with thematic analysis. Perceived developmental trajectories were heterogeneous: 34.2% of students reported rapid early improvement followed by a plateau, 28.4% steady improvement, and 21.1% delayed acceleration. Development was also uneven across skills, with improvement in evaluating GenAI outputs perceived as more challenging; assignments were the most frequently selected driver of skill growth. Interviews further indicated increasingly specific prompting, tailored verification, and selective delegation, although some students reported stable routines, dependence, skipped reasoning, or improvements attributable partly to better GenAI tools. Academic gains were reported more often for assignments than for examinations. The SEM indicated a positive indirect association between perceived skill growth and academic gains through adaptation of assignment processes, whereas academic gains did not clearly differ across developmental trajectory groups after multiplicity correction. The findings suggest that perceived GenAI skill development varies in both trajectory and skill domain and is associated with changes in learning practices and differentiated academic benefits. 

---
# Visual Grounding First, Multimodal In-context Learning Follows 

**Authors**: Minhyuk Seo, Minjae Lee, Chaeeun Lee, Wei Lin, Muhammad Jehanzeb Mirza, Tinne Tuytelaars, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2610.30592)  

**Abstract**: LLMs have shown strong in-context learning (ICL) capability, but extending it to Multimodal LLMs (MLLMs) remains challenging. Prior multimodal ICL methods often rely on ICL-specific datasets for additional training or task vector extraction, which can improve performance on the same ICL benchmark used for adaptation. However, they do not generalize well to other benchmarks and often induce forgetting in previously well-solved tasks. For this reason, we identify 'visual grounding' as the key bottleneck in multimodal ICL; MLLMs often fail to attend to task-relevant visual evidence in demonstrations, instead over-relying on textual cues and producing hallucinated outputs. To address this, we propose ProCoRe to enhance visual grounding through contrastive reinforcement learning on multimodal contrastive data for improving multimodal ICL. ProCoRe generates captions for contrastive image pairs, decomposes them into propositions, and optimizes contrastive rewards that encourage alignment with corresponding images while discouraging mismatched ones. Despite never seeing ICL-formatted training examples, ProCoRe improves average multimodal ICL classification accuracy by 5.6% and captioning ROUGE-L by 13.8% over the off-the-shelf Qwen-VL-3-8B model, outperforming ICL-data-dependent state of the arts across benchmarks. Finally, we introduce PICL, a personalized multimodal ICL captioning benchmark that tests whether MLLMs genuinely understand visual demonstrations rather than relying on pre-trained semantic priors. 

---
# Bridging the Closed-World Gap: Interpretable Anomaly Detection via a Hybrid TSK-FCM Ensemble on the BETH Dataset 

**Authors**: Lohith Srikanth Pentapalli, Josette Riep, Justin Ouwerkerk, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30576)  

**Abstract**: A fundamental paradox plagues the deployment of Machine Learning in cybersecurity: supervised models generalize poorly to novel, Out-of-Distribution (OoD) attacks, whereas unsupervised models often suffer from poor separability, leading to unmanageable false-alarm rates. This challenge is exemplified by the BETH dataset, a kernel-level benchmark in which the training data consists solely of benign and suspicious insider behaviors, whereas the test set consists almost entirely of novel botnet activation attacks. To bridge this detection gap, we propose a Hybrid Fuzzy Ensemble that synergizes the precision of supervised learning with the adaptability of unsupervised clustering. Our architecture integrates a gradient-optimized Takagi-Sugeno-Kang (TSK) fuzzy classifier to act as an expert on known insider threats, with a parallel Fuzzy C-Means (FCM) module to detect geometric anomalies. A critical finding of our research is that traditional fuzzy membership scores degrade in high-dimensional feature spaces due to the curse of dimensionality. We overcome this by introducing a robust Min Distance metric that exploits the raw geometric isolation of botnet activation attacks. This approach yields a state-of-the-art AUROC of 0.9951 on the BETH test set, outperforming the original Isolation Forest baseline by 17.1%. By combining near-perfect OoD detection with the intrinsic interpretability of fuzzy logic, our framework offers a viable path toward robust, transparent, and autonomous network defense. 

---
# Lifetime-valid decisions by continually evolving AI analysts 

**Authors**: Xiaojing Sun, Yuhan Zeng, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30562)  

**Abstract**: AI agents increasingly act as analysts whose decision can often be cast as hypothesis tests. Classical tests control the Type I error rate, the probability of rejecting a true null hypothesis, for fixed procedures. However, AI analysts introduce internal randomness and can evolve continuously over time. As AI analysts evolve, the risk of approving versions with Type I error rates above the target can accumulate, and adaptive use of evaluation feedback can lead to overfitting. We introduce continual validation, which evaluates each proposed AI analyst version before deployment using simulated data under the null hypothesis, with scenarios that can be randomized and expanded over time. We prove that, regardless of how updates are chosen or how many are proposed, the lifetime probability of ever deploying a version whose Type I error rate exceeds the allowed level remains bounded. This guarantees that every subsequent decision controls the Type I error rate at the target level across these scenarios. At the same time, updates with sufficiently low Type I error rates pass validation with high probability, allowing the analyst to continuously improve its performance without sacrificing rigor. In two examples involving evolving AI analysts for bioequivalence studies and product A/B tests, Type I error rates reached 0.098 and 0.183 without continual validation, compared with at most 0.045 and 0.044 with it, while retaining 82% and 96% of the power from optimally selected valid updates. 

---
# Decision Making and Ride Comfort Evaluation Based on Lane-Change Prediction for Autonomous Vehicles 

**Authors**: Mohamed Manzour, Aditya Kumar, Rubén Izquierdo, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2610.30528)  

**Abstract**: Predicting surrounding vehicles' lane-change intentions enables autonomous vehicles to anticipate potential hazards and respond proactively rather than relying solely on reactive maneuvers. This study investigates the benefits of integrating lane-change prediction, based on knowledge graph embeddings and Bayesian inference, into autonomous vehicle decision-making. Three driving strategies are evaluated in risky merging scenarios simulated in CARLA: reactive driving, prediction-based longitudinal action, and prediction-based lateral action. The proposed prediction model anticipates the surrounding vehicle's lane change approximately 3-4 seconds before its occurrence, allowing the ego vehicle to adjust its behavior accordingly. In the representative simulation records, reactive driving results in a collision accompanied by aggressive deceleration. In contrast, prediction-based longitudinal action initiates early braking to establish a safe gap, while prediction-based lateral action performs a lane change when a safe adjacent lane is available. Both predictive strategies avoid collisions in the evaluated records. Compared with reactive driving, peak deceleration before impact is reduced by 31.5% with longitudinal action and 91.0% with lateral action. Additionally, an ISO 2631-1-based horizontal vibration estimate is used to provide a ride comfort assessment. Overall, the results highlight the potential of lane-change prediction to support safer and smoother driving through proactive decision-making, with additional insights into ride comfort provided by vibration-based estimates. 

---
# LLM2Cert: LLM Synthesis of Certificates with SMT/Lean Verification 

**Authors**: Zhipeng Cao, Peiyang Song, Nikolaus Vertovec, Weiru Cui, Zhihao Yang, Mingkang Liu, Luxia Lin, Bai Xue, Alessandro Abate  

**Link**: [PDF](https://arxiv.org/pdf/2610.30524)  

**Abstract**: Lyapunov functions and barrier certificates give state-space guarantees of stability and safety, but synthesizing and verifying them---especially for nonlinear, high-dimensional systems---remains an open challenge. Existing pipelines verify candidates with numerical solvers (sum-of-squares programs), interval methods, or satisfiability-modulo-theories (SMT) solvers, which often require relaxations, are conservative, or scale poorly with model dimension and nonlinearity. These limitations have also shaped benchmark design: standard test suites are dominated by low-dimensional polynomial systems chosen to suit what solvers can handle rather than what engineering applications require. We address both sides of this challenge. First, we introduce DynaCertBench, a benchmark of 105 instances: 80 non-redundant systems curated from the literature, extended with high-dimensional and control-constrained variants and new nonlinearly coupled systems, comprehensively spanning stability, region-of-attraction, forward-invariance, and safety tasks. Second, we present LLM2Cert, an LLM-driven framework that synthesizes candidate certificates and verifies them against their proof obligations: each obligation is routed to an SMT solver or directly to the Lean Proof-Repair agent. On DynaCertBench, LLM2Cert verifies 81 of 105 instances ($77.1\%$), a $17.1$ percentage point improvement over one-shot proposal without feedback. On the 81 fixed certificates, LLM2Cert verifies 22 that All-SMT leaves unresolved. Code and data are available at this https URL. 

---
# Protegrity-PII Technical Report 

**Authors**: Hardeep Bassi, Hui Ma, Luís Seabra, Koowum Joshi, Daniel Pereira, Averell Gatton  

**Link**: [PDF](https://arxiv.org/pdf/2610.30518)  

**Abstract**: We present Protegrity-PII, a new PII decision model for structured and unstructured text. Protegrity-PII achieves a typed recall of 0.915 with a precision of 0.907 on a new benchmark we propose, PII-SCOPE. Our model processes text in 0.49 seconds per million characters at \$0.0005 per million characters on a single RTX 6000 using batched inference. Due in part to the expansive nature of the entity type taxonomy used in training, Protegrity-PII achieves superior out-of-distribution generalization when benchmarked against a selection of state of the art models and datasets. 

---
# Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals 

**Authors**: Louise Yorke  

**Link**: [PDF](https://arxiv.org/pdf/2610.30512)  

**Abstract**: LLM judges are used to determine which of two answers is better. Trusting these verdicts requires knowing how repeatable they are and how much they depend on the judge's instructions. We show that similar aggregate scores can conceal substantial changes in individual verdicts, both when the same setup is repeated and when the instructions change. Using 500 Arena-Hard v2.0 prompts with two models' answers held fixed, we run GPT-4.1 as the judge under all eight combinations of criteria, output format and task framing, and repeat every setup unchanged at temperature zero on a pinned model version. Rerunning the default setup moves the aggregate score by about one point but changes the win/tie/loss outcome on 21.2% of prompts. Instruction changes introduce further sensitivity: in one setup, which asks for the verdict without reasoning and also drops several other instructions, outcomes change on 28.2 percentage points more prompts than the two setups' unchanged reruns do on average, and the aggregate score falls by 4.09 points. In another, where the criteria prioritise accuracy, outcomes change on 6.7 percentage points more prompts by the same comparison, while the aggregate score barely moves. A stable aggregate score therefore does not establish that individual verdicts are repeatable. This distinction matters when those verdicts are used as preference labels or to assess individual responses. Evaluations that rely on individual verdicts should repeat each setup they use and report both aggregate scores and prompt-level agreement. 

---
# PhysibleMDM: Enforcing Inference-Time Physical Feasibility Constraints on Motion Diffusion Models via Noise Optimization 

**Authors**: Maximilian Stölzle, David Müller, Agon Serifi, Ruben Grandia, Espen Knoop, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2610.30500)  

**Abstract**: Motion Diffusion Models (MDMs) can generate diverse and visually appealing human motion; yet, the resulting trajectories frequently violate physical constraints, limiting their downstream use in simulation and control. We present PhysibleMDM, a framework that promotes physical feasibility of MDM output by optimizing at inference time the diffusion noise without retraining the generative model. We introduce a set of surrogate feasibility constraints that can be evaluated without full trajectory simulation. These constraints combine kinematic checks (e.g., ground penetration, slip, and self-collision) with dynamic feasibility tests computed via an inverse-dynamics inner loop formulated as a cone program. We solve the resulting black-box noise optimization problem using the Separable Covariance Matrix Adaptation Evolution Strategy (Sep-CMA-ES), thereby avoiding reliance on computationally expensive and discontinuous gradients. An autoregressive scheme then chains optimized segments to synthesize long motion sequences. Across 100 motion styles on an SMPL humanoid, PhysibleMDM reduces mean ground penetration, self-collision penetration, and ground slip by 75%, 92%, and 74%, respectively, relative to unoptimized diffusion output. Across 11 sequences for a custom bipedal robot, it achieves median reductions of 89.5% in per-sequence mean acceleration-reconstruction error and 79.5% in the mean inverse-dynamics residual. 

---
# Externalise the Memory: A Pre-Registered Study of Where an Embodied Agent's Experience Should Live 

**Authors**: Cheng-Yin Tu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30478)  

**Abstract**: Should an embodied agent's experience be baked into its policy weights, or kept external and retrieved at inference? A memory-augmented teacher (a VLM that retrieves lessons, with their episode images, from its own past attempts) produces trajectories on a vision-conditioned reach task; from the same trajectories we distil LoRA student policies and evaluate them under matched protocols, including one that re-attaches the teacher's frozen retrieval buffer to a fixed set of student weights. We pre-registered the hypotheses, arms, tests, and analysis rules before training and before any p-value. Distilling the memory-augmented behaviour into weights beat an action-only control by +1 pp (n.s.); the registered test proves underpowered for its own minimum effect, so we withdraw - rather than refute - the memory-in-weights claim. Attaching retrieval to a fixed set of student weights added +34 pp (z=5.15), reaching the pooled memory-augmented teacher reference (49.3%); the pre-registered protocol contrast agrees (+23 pp, z=3.36); under a pre-registered auto-reset sensitivity count the two are +36 and +29 pp. An exploratory behavioural probe with pre-registered per-arm predictions is consistent with behavioural distillation transferring the marginal correction but not the conditional, situation-specific correction that retrieval supplies. The recipe that follows is to externalise the memory: in follow-up controls registered before they ran, the same frozen buffer lifts the base model with no adapter to 48% (sensitivity 48%) and the other three adapters from 19-26% to 53-55% (sensitivity 17-21% to 50-55%), and the adapter adds nothing detectable once the buffer is attached. On a second, harder task, the retrieval advantage points the same direction but falls short of significance (+6 pp, 95% CI [-4.5, +16.4]); exploratory diagnostics tie the attenuation to memory coverage. 

---
# Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy 

**Authors**: Riteng Zhang, Xiaoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30474)  

**Abstract**: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictionary-based decompositions of model representations, form an atomic interpretable unit basis, a coordinate system for semantically meaningful explanation. Under this paradigm, we organize existing and new measurements such as DAME (Dominance-Aware Monosemanticity Evaluation) as proxy metrics and introduce a comprehensive benchmarking framework. Evaluating hundreds of SAEs across multiple architectures and model families, we uncover failure modes in which some SAEs appear monosemantic under existing evaluation metrics while remaining functionally polysemantic: the most dominant features, those contributing most to model computation, are systematically less interpretable and more redundant. These failure modes warrant attention from SAE researchers as targets for improved training and architectures, and motivate dominance-aware evaluation as a standard check for practitioners before SAE features are deployed for attribution, circuit tracing, or intervention. 

---
# SplatLab: Autonomous Scientific Discovery for 3DGS Research 

**Authors**: Seemandhar Jain, Manmohan Chandraker  

**Link**: [PDF](https://arxiv.org/pdf/2610.30468)  

**Abstract**: Reproducing a published 3DGS paper takes weeks of expert engineering before any extension can be proposed. Prior autonomous-research systems read papers, draft hypotheses, and write code, but are limited by short fixed templates, low executability, and weak traceability, and none has been demonstrated end-to-end in a live computer-vision subfield with intense activity. SplatLab is an autonomous research system for 3D Gaussian Splatting that addresses these limitations. Given a research prompt, a target-paper URL, or no input at all, it reproduces a published baseline, proposes an extension, trains and ablates across Mip-NeRF 360 and LLFF, writes the manuscript, and binds every reported number to a logged training run. Five design choices distinguish it: (i) candidates mutate plugins from a verified-reproduction backbone matched to published baselines; (ii) a cascade-weighted Elo tournament across eight islands scores candidates on measured GPU PSNR with verbal-gradient feedback; (iii) a standing falsifier subtracts Elo on failure, integrating stress-test survival into selection; (iv) every reported number is bound to a logged training run, and the run cannot terminate while any claim is unbacked; (v) a seven-category 3DGS taxonomy wraps EVOLVE-BLOCK markers in domain-aligned scaffolds transferable across 3DGS repositories. On PSNR-targeted runs, SplatLab's manuscripts beat their reproduced targets by +0.18 to +3.78 dB, including +3.78 dB on astrophotographic nebula rendering, for which no prior 3DGS method exists. The research cycle compresses from months of graduate-student time to days of mostly autonomous compute. Code, all manuscripts, experiment ledgers, and the cross-run paper store are available on the project page. 

---
# Short-to-Long: Detached Features for Long-Horizon Value Learning 

**Authors**: Tong Che, Boris Ivanovic, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2610.30461)  

**Abstract**: Learning useful representations from continually changing value targets remains a challenge in deep reinforcement learning. We study the temporal horizon of this supervision: smaller discounts make these targets less sensitive to distant bootstrap changes, but using them for behavior changes the control objective. We introduce Short-to-Long (S2L), which supplies a long-horizon behavior learner with features trained by smaller-discount source networks. Detached connections prevent the behavior value loss from updating the sources, allowing each source to follow its own value objective while the behavior discount remains unchanged. Across Atari, online continuous control, and offline continuous control, the approach yields aggregate performance gains, improving the paired terminal median on 40 of 57 Atari games. Matched Atari ablations support the roles of feature transfer, smaller source discounts, and gradient isolation. Fixed-trajectory measurements further show reduced drift in source value targets with mixed discounts. 

---
# aggmetrics: Aggregating Classification Performance Measures for Machine Learning 

**Authors**: Ronaldo Cristiano Prati  

**Link**: [PDF](https://arxiv.org/pdf/2610.30456)  

**Abstract**: aggmetrics is a Python package that extends any binary performance measure expressed in the four confusion-matrix counts to multiclass, multilabel, ordinal, cost-sensitive, soft-label and cross-validation settings, by substituting one of four aggregation operators for what libraries usually expose as an average switch. The package also stores, for every measure, the structural properties that decide which identities apply to it, can infer those properties from a user-supplied formula, and uses them to warn when a measure is used where it does not behave as commonly assumed. Its test suite encodes the results of the companion theory as executable tests. The package is BSD-3 licensed, depends only on NumPy, and is available at this https URL documentation at this https URL. 

---
# Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models 

**Authors**: Taewhoo Lee, Minju Song, Jungwoo Park, Chanwoong Yoon, Junha Jung, Hyeon Hwang, Sanghoon Lee, Minbyul Jeong, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30435)  

**Abstract**: Recent advances in search agents have enabled large language models to tackle complex tasks by retrieving and synthesizing external information. However, existing search agents often lack search-state awareness, i.e., the ability to recognize what information has been obtained and what remains missing, leading to hallucinated reasoning and prematurely terminated search. To address this limitation, we propose Self-Assessment Reinforcement Learning (SARL), a reinforcement learning framework in which models learn to recognize their own search state during training and improve their search behavior accordingly, without requiring additional assessment at test time. SARL achieves this through a dual reward design that captures the corrective effect of each assessment while penalizing excessive reliance on it, jointly encouraging effective and efficient search trajectories. Under this setup, SARL achieves a 24.3% relative improvement over the strongest baseline, with average gains of 35.9% on multi-hop reasoning and 27.9% on deep research benchmarks against the strongest per-benchmark baselines. SARL further surpasses methods built on auxiliary supervision or multi-agent coordination, despite operating as a standalone RL framework with no inference-time scaffolding. Beyond final performance, SARL maintains training stability under extended search horizons, a regime where existing methods consistently degrade, enabling sustained gains as task complexity grows. 

---
# High-Value Training Data from Noisy, Diverse Sources 

**Authors**: Hedi Driss, Evan Frick, Anastasios N. Angelopoulos, Cho-Jui Hsieh, Yejin Huh, Tijana Zrnic, Andrew Ilyas  

**Link**: [PDF](https://arxiv.org/pdf/2610.30421)  

**Abstract**: Training state-of-the-art models increasingly requires learning from large, heterogeneous datasets whose examples vary in quality, domain relevance, and source of supervision. Dataweighting---assigning different weights to training examples---offers a general way to control how such data influences training, but learning effective weights at scale remains challenging. We develop a scalable approach to learning dataweights: given a small, high-quality target dataset, we use metagradients through the training process to efficiently optimize training-example weights directly for performance on the target domain. To scale beyond a fixed dataset and model, we learn parametric weighting rules that generalize to new and streaming examples and transfer from smaller models to substantially larger ones. We study reward modeling as a prototypical application, where preference data may be noisy, adversarial, synthetic, or mismatched with the target domain. Learned dataweights improve held-out performance, automatically adapt reward models to specialized domains, and allow mixtures of human and synthetic supervision to outperform either source alone. 

---
# Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems 

**Authors**: Daniel Ruiz, Luca Luceri, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2610.30418)  

**Abstract**: Modern LLM agents can retrieve information from accessible documents, and then disclose a sensitive conclusion assembled from them. This mosaic effect creates a critical risk for agents operating over confidential data. We quantify its realization with 30 synthetic cybersecurity, medical, and military examples, yielding 10,800 final answers from three LLM agents in controlled simulation and 1,440 answers from live agents in two open-source harnesses. Even under benign queries designed to be answerable from one document, GPT-5.6-Sol, the best performing model tested, discloses forbidden composites in 7.08% of answers (95% CI: 3.75-10.67%) in simulation, with worse disclosure rates in live harnesses. Adding visible classification headers and explicit aggregation rules reduces pooled disclosure rates substantially, yet unsafe disclosures persist in 32.37% of simulated answers (95% CI: 28.33-36.19%) across all intents despite these safeguards. Furthermore, increased refusals do not guarantee confidentiality, since significant refusals reveal the prohibited conclusion regardless. These results point to a troubling conclusion: sensitive data leaks via mosaic-style events are an observed confidentiality failure mode in the evaluated settings at non-negligible rates. They motivate stronger controls on what agents may infer and disclose in sensitive data environments with mixed-clearance users. 

---
# AcoustiPrint: Physics-Guided Neural Design of Manufacturable Acoustic Structures 

**Authors**: Shanmuka Sadhu, Harshvardhan C. Takawale, Yehia Zakaria, Ahmed Allam, Nirupam Roy  

**Link**: [PDF](https://arxiv.org/pdf/2610.30411)  

**Abstract**: Recent work in additive manufacturing and neural inverse design enables the creation of complex three-dimensional structures to manipulate physical fields, yet these designs must also adhere to discrete, non-differentiable fabrication constraints. Current methods limit the design space by imposing differentiable soft manufacturability loss penalties, or by fixing structures through post-processing, causing a discrepancy between the geometry optimized by the physics model and the final fabricated result. Thus, we present AcoustiPrint, a physics-informed neural architecture for constrained inverse design that represents three-dimensional geometries with an implicit neural representation (INR) and passes each candidate through a printability operator before evaluating its final response. The printability operator executes layer-wise self-support, discrete material selection, and whole-body connectivity, assuring fabrication feasibility. To allow for backpropagation through the discrete non-differentiable operations, we use surrogate gradients for the backward pass and a self-supervised printability objective that progressively drives the INR toward a valid design domain. We demonstrate AcoustiPrint for personalized transcranial focused ultrasound, where inter-patient differences in acoustic properties and head geometry deform wave propagation and requires patient-specific acoustic focusing. AcoustiPrint jointly optimizes three-dimensional geometry and material composition through differentiable wave physics while ensuring manufacturability. Experiments show manufacturable acoustic structures with enhanced focal accuracy and over density-based and phase-based baselines, demonstrating a generalized methodology for neural design through non-differentiable manufacturing constraints and differentiable physics. 

---
# MemPlan: Memory-Conditioned PDDL Planning for Partially Observable Text Environments 

**Authors**: Tengwei Song, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2610.30379)  

**Abstract**: LLM agents are flexible in language interaction, but using them as long-horizon planners is costly and prone to hallucinated or invalid actions. This paper targets a different role for LLMs: constrained bridges between language and symbolic planning. We propose MemPlan, a memory-conditioned PDDL planning framework for partially observable interactive text environments. MemPlan keeps the PDDL domain fixed and reconstructs the PDDL problem at each step from parsed observations, candidate hypotheses, and interaction memory. Positive graph memory assigns relevance-based action costs to candidate hypotheses for missing objects, locations, or conditions, guiding the planner toward more plausible checks. Negative memory records execution-refuted assumptions and compiles them into exclusion constraints, preventing repeated attempts at candidates already contradicted by feedback. A unified schema-conditioned language--symbol interface, trained offline with supervised fine-tuning, connects text environments to the planning loop through observation-to-fact parsing and symbolic-action-to-command grounding. Experiments on TextWorld, ALFWorld, and Robotouille show that MemPlan improves task completion and step efficiency over language-only and PDDL-aware baselines, while substantially reducing online token cost compared with LLM-planning baselines such as ReAct. The same fine-tuned interface is also reused across TextWorld task types without task-specific fine-tuning. Our code and data are available at: this https URL. 

---
# Probing and Steering for Safety-Relevant Behaviours in Large Language Models 

**Authors**: Nimun Bajwa, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2610.30363)  

**Abstract**: Activation probes are computed directions in a language model's residual stream, used both to detect safety-relevant behaviours and, by adding their direction back into the stream, to steer them. We explore how the main design choices behind such probes interact, within one pipeline, on one model (Llama-3.1-8B-Instruct), and across three behaviours: deception, high-stakes awareness and sycophancy. The pipeline crosses two estimators (difference-of-means and logistic regression) with three token-pooling schemes (last, mean and learned attention), selects appropriate layers to detect or steer a behaviour, and evaluates every direction twice: as a detector of the behaviour, and as a steering vector to push outputs towards the behaviour. Prior work read together suggests a read/write dissociation, in which logistic regression reads (detects) best but difference-of-means directions write (steer) best. We find that it holds for deception and high-stakes but not for sycophancy, where the pattern reverses. The reversal appears to follow partially from the data rather than from the estimators. The best pooling technique and the layer choice depend on the behaviour and data, and the layer that steers best can sit well before the layer that detects best. The right steering metric to use between changing a model's perception and eliciting behaviour depends on the task - and the two can disagree outright. The exported probes and directions power an interactive interface through which a user can score and steer text directly. 

---
# Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable 

**Authors**: Marius De Kuthy Meurers, Aaron Parisi, Bernd Bohnet  

**Link**: [PDF](https://arxiv.org/pdf/2610.30349)  

**Abstract**: LLMs are increasingly consulted for moral judgment. We aim to improve performance on such tasks by rewarding good arguments, on the premise that arguing well increases the likelihood of judging well. We train a reinforcement learning policy initialized from a state-of-the-art language model in a self-judged tournament with masked thinking: the generated arguments are judged pairwise on argument quality by the policy itself. As an empirical basis, we extract authentic dilemmas from classic novels, using a logic solver to verify that each constitutes a conflict. We show that our judge rates argument quality like human annotators, and the policy then adopts the language register that it rewards. Testing on the external benchmarks MoralChoice, ELEPHANT, and MORABLES, we find that the policy becomes more decisive and invariant to the order in which options are presented, but sycophancy and reframing remain a challenge to validity. Two ablation studies with synthetic dilemmas and unmasked thinking match the reordering gains, but show a weaker register shift and higher sycophancy. We conclude that consistency is not validity: consistency under reordering can be optimized without improving the moral judgment it is meant to signal. 

---
# MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion 

**Authors**: Van Chung Nguyen, Hung Manh La  

**Link**: [PDF](https://arxiv.org/pdf/2610.30334)  

**Abstract**: Designing and tuning reward functions remains one of the main challenges in building effective reinforcement learning (RL) pipelines for humanoid locomotion. While shaping rewards can accelerate learning, naively formulated terms often conflict with the desired behavior, overfit to a specific robot, or become unstable under small weight changes. Potential-Based Reward Shaping (PBRS) offers an alternative solution: shaping the reward via differences of a potential function to guide exploration without changing the optimal policy. In this paper, we study Multi-PBRS for humanoid locomotion and propose a unified reward configuration. We then evaluate the same reward structure and weights across nine humanoid robots, requiring only robot-specific metadata (joint/link mappings and default poses) rather than per-robot reward redesign. Our experiments show that, compared to standard shaping, MPBRS provides more consistent scaling behavior and reduced tuning sensitivity across robots, making it easier to tune without altering the optimal policy. 

---
# TerraTransfer++: Demonstration-Free End-to-End Planner 

**Authors**: Zikang Xiong, Weixin Li, Zhouchonghao Wu, Akshay Rangesh, Saarth Bonde, Grantland Hall, Chen Tang, Yihan Hu, Wei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.30319)  

**Abstract**: End-to-end autonomous driving has achieved state-of-the-art performance on benchmarks and real-world deployments. Its standard training recipe, however, is expensive across all existing paradigms: imitation learning requires fleet-scale driving logs whose behavior distribution only thinly covers the safety-critical tail; closed-loop reinforcement learning in a photorealistic simulator, while offering diverse behaviors, is bottlenecked by sample-efficiency, per-step cost of rendering, and a forward pass through a large vision backbone. Self-play in vectorized simulators changes the economics: at millions of rollout steps per second, one can obtain a state distribution with natural diversity in collisions, near-misses, and recoveries that no driving log contains, and a highly reactive policy emerges from a basic driving-quality reward. Our approach exploits this asymmetry. We pretrain a single policy by self-play, distill its rollouts into a flow-matching trajectory teacher that also plans, and train lightweight adapters on a frozen pretrained vision backbone to reproduce the latent space of either teacher through the action KL divergence, a batch-relational low-rank structural loss, and, for the planner, a trajectory-matching loss. Every target traces back to the self-play policy, so alignment never supervises against a logged trajectory: a paired dataset of (image, scene-state) frames suffices, with no need for the curated expert demonstrations that imitation pretraining is built on. From this one self-play teacher we obtain two end-to-end students: a closed-loop policy that outperforms every published imitation-based end-to-end method on the photorealistic 3D Gaussian splatting closed-loop benchmark HUGSim, and a planner that reaches 59.4 EPDMS on NAVSIM navhard, above every published method including DriveZero-Scale (57.1), and 95.5 PDMS on navtest, without human demonstrations. 

---
# Gender, Sexual Orientation, and the One-Year Refugee Bar: A Computational Study of Bill C-12 and Canadian Refugee Adjudication 

**Authors**: Sean Rehaag  

**Link**: [PDF](https://arxiv.org/pdf/2610.30297)  

**Abstract**: In March 2026, Bill C-12 amended the Immigration and Refugee Protection Act to make refugee claims ineligible for referral to the Refugee Protection Division where they are made more than one year after the claimant's first entry to Canada. Affected claimants are diverted to a Pre-Removal Risk Assessment (PRRA), which is typically decided by an immigration officer on a paper record, with no guaranteed oral hearing, no appeal on the merits, and no automatic stay of removal pending judicial review. The government defended the bar partly on the premise that genuine refugees claim promptly. Critics countered that late claims are concentrated among vulnerable groups, particularly people fleeing persecution based on gender, sexual orientation, gender identity, or gender expression (SOGIE).
This article tests those competing premises. Using artificial intelligence tools, checked through human review, it examines nearly 29,000 published decisions of the Refugee Protection Division, the Refugee Appeal Division, and the Federal Court from 2006 to 2025. The results of this study demonstrate that, first, while most claimants claim within a year, roughly one in nine in the published record does not, and gender and SOGIE claims are significantly overrepresented among these late claims. Second, among late claims, refusals of gender and SOGIE claims are more likely than refusals of other claims to be overturned on appeal or judicial review. Late claims on other grounds, by contrast, fare somewhat worse on review than timely claims. The one-year bar therefore removes the hearing, the appeal, and the stay from the late claimants for whom those safeguards have been doing the most work. The article recommends repealing the bar or, failing that, adopting exemptions, procedural protections, and greater transparency in PRRA decision making. 

---
# GAV: A scientific experimentation environment for PDE agents 

**Authors**: Muheng Li, Wenlong Mou  

**Link**: [PDF](https://arxiv.org/pdf/2610.30292)  

**Abstract**: Reliable feedback is essential for coding agents to develop programs for partial differential equations (PDEs). Yet evaluating these programs on new PDE problems often requires expensive reference computations, while fixed datasets restrict the range of experiments. We introduce Generator-Agent-Verifier (GAV), a scientific experimentation environment that makes new PDE problems verifiable by construction. The Generator learns a distribution of continuous reference solutions using data from numerical simulations or experimental measurements to represent a target physical regime. Using the method of manufactured solutions, it works backward from the generated reference solutions to construct matching PDE problems without a separate numerical solve. The Agent designs experiments, investigates failures, and iteratively improves its program. The Verifier evaluates candidate programs against the reference solutions and provides numerical feedback on accuracy and computational cost. This feedback can also guide the generation of new problems that probe the current program's weaknesses. We provide a theoretical analysis of evaluation reliability and guided development. Experiments on Navier-Stokes and Vlasov-Poisson-type equations across different data sources and regimes demonstrate the effectiveness of GAV. In particular, on the numerical Navier-Stokes task, GAV reduces the mean test error of three existing PDE agents by more than an order of magnitude relative to their original setups. Code is available at this https URL. 

---
# Generalizing Metropolis-Hastings for DAG Sampling 

**Authors**: Daniele Nikzad, Mikko Koivisto  

**Link**: [PDF](https://arxiv.org/pdf/2610.30291)  

**Abstract**: A fundamental challenge in Bayesian network structure learning lies in efficiently approximating the posterior distribution of directed acyclic graphs (DAGs). For large, high-dimensional datasets, the task remains beyond the capabilities of current methods. This work advances Markov chain Monte Carlo algorithms by introducing novel moves that simultaneously resample the parents for multiple nodes of the DAG. At the core of these moves is duet sampling, a new generalization of the Metropolis-Hastings algorithm that selects two proposal distributions from a pool at each iteration. One is used for generating a candidate state of the Markov chain and another for evaluating the return to the current state; both are ingredients of an appropriately crafted acceptance criterion. Pairing different proposal distributions provides additional freedom and enables constructing Markov chains that mix more efficiently than their basic Metropolis-Hastings counterparts, at no additional computational cost. Experiments show that our new moves, both individually and in combination, outperform the previous state of the art, which relies on a more restrictive pool of moves. This advantage persists even when letting the previous methods leverage duet sampling to enhance their performance. 

---
# SWE-VERIFIER: Selective Verification for Cost-Efficient Coding Agents 

**Authors**: Kaijian Zou, Vaibhav Chennareddy, Yunxiang Zhang, Naihao Deng, Longju Bai, Jiaxin Pei, Rada Mihalcea, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30284)  

**Abstract**: Coding agents spend substantial computation generating actions and processing long interaction histories. Poor action choices can steer agents into low-value states, wasting computation and increasing API costs. Sampling and verifying alternative actions can improve the agent's decisions, but applying this procedure at every step adds substantial overhead. We introduce SWE-Verifier, a lightweight generative model that selectively triggers candidate sampling and verification. It scores the coding agent's proposed action to determine when additional candidate sampling is needed, then ranks the resulting actions before execution. We train SWE-Verifier from Qwen3.5-4B using preference data that groups contrasting actions with their resulting environment observations and rubric-guided teacher scores. Using the mini-swe-agent harness, SWE-Verifier improves resolution from 75.47% to 77.20% on MiniMax-M2.7 at 1.09 times cost and generalizes to DeepSeek-V4-Flash without training on its actions, improving resolution from 72.60% to 74.27% at 1.13 times cost. Across both models, SWE-Verifier verifies fewer than 4% of steps and achieves comparable or higher resolution than the trajectory and all-step baselines at substantially lower cost. These results demonstrate that selective candidate sampling and lightweight verification provide complementary gains in coding-agent performance and inference efficiency. Our model, code, and data are available at this https URL. 

---
# Edit, Don't Imitate: Minimal-Edit On-Policy Distillation for Multimodal Reasoning 

**Authors**: Wen Ye, Ailin Deng, Zirui Wang, Shixing Yu, Han Zhang, Tian Yun, Nikhil Mehta, Tarun Kalluri, Xuefei Cao, Xuezhe Ma, Hengduo Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30231)  

**Abstract**: Visual mathematical reasoning can fail before reasoning begins: when a vision-language model (VLM) misreads a diagram, subsequent reasoning proceeds from false premises. This perceptual bottleneck is especially pronounced in smaller models, while accurate textual descriptions can substantially narrow the gap to larger models. Because obtaining such descriptions at inference requires additional annotation or model calls, we propose minimal-edit corrective on-policy distillation. Using reference figure descriptions and verified reference solutions as training-only information, a teacher scores incorrect student rollouts under a minimal-edit objective, and only teacher-disfavored tokens contribute to the corrective loss. Correct rollouts receive task-reward reinforcement learning. This preserves on-policy supervision while avoiding full-response imitation. Across model scales, the method requires no descriptions at test time and meets or exceeds the corresponding base model supplied with a reference figure description at inference. It also improves perception-oriented performance, particularly low-level geometric perception, and transfers to external mathematics and perception benchmarks. Because the minimal-edit teacher evaluates corrections rather than generating its own solution, the method remains effective with a frozen same-size teacher, although its gains are smaller than with a larger teacher on most metrics. 

---
# Automated screening of privacy risks in medical AI 

**Authors**: Konstantinos Vrettos, Eleftherios Tzanis, Haridimos Kondylakis, Evangelia E. Vassalou, Florence X. Doo, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2610.30229)  

**Abstract**: Medical artificial intelligence (AI) models are often shared without their training data, assuming this protects the patients who contributed them. However, a model can reveal whether a patient's records were used in training, a privacy breach known as a membership inference attack. This risk is usually reported as a cohort average, which can appear negligible while individual patients are almost certainly exposed. Identifying potentially vulnerable records before developing a production model could make privacy assessment more practical. We measured each patient's vulnerability to membership inference attack, across 7,789 models trained on six clinical datasets in four modalities, including two adapted pretrained medical models. Training choices changed how many patients had high attack scores, but the patients who ranked highest by risk largely stayed the same across the configurations we tested. Stronger weight decay reduced the exposed cohort up to threefold, yet largely the same patients remained most at risk across training recipes, architectures and initializations, including foundation models. Therefore, we developed a lightweight screening tool that uses four small models to identify patients with high risk of exposure before production-model training. We also assessed if flagged patients could be protected without removing their data. Privacy screening should become incorporated into data preparation for clinical AI. 

---
# When Reconstruction Is Not Enough: Doubly Robust Counterfactual and Interventional Mediation Analysis with Learned Multimodal Representations 

**Authors**: Cong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30220)  

**Abstract**: Learned representations of high-dimensional multimodal health records are increasingly used to adjust for confounding, yet they are usually trained to reconstruct the data rather than preserve the information that causal identification requires. We study counterfactual outcomes under alternative treatments, including phenotype-specific effects and patient-level counterfactual risk profiles of the kind digital twins provide, and interventional direct and indirect effects under exposure-induced intermediate confounding. We give conditions under which a representation preserves identification and show that interventional mediation is more demanding than counterfactual comparison. We prove that no covariate-only training objective, whether reconstruction, contrastive, or physics-informed, can certify that a representation is sufficient, and that, in a linear structural model, a representation can lose arbitrarily little reconstruction accuracy yet induce arbitrarily large bias. We decompose the error of cross-fitted doubly robust estimators into a first-order representation bias, which neither double robustness nor cross-fitting removes, and a second-order remainder of nuisance-estimation errors, so coverage worsens as the sample size grows when the representation is insufficient. We propose a workflow combining modality-specific variational autoencoders trained outside the estimation sample, cross-fitted doubly robust estimators, locked phenotypes, select-then-confirm pathway analysis, portable representations, and diagnostics for representation insufficiency, exposure leakage, and sensitivity. In simulations, discarding a rare, moderately strong confounder gave indirect-effect intervals with 2% coverage at n = 3,000; reconstruction loss missed the problem, and a simple predictive diagnostic flagged it in every replicate. 

---
# Prediction Is Not Calibration: Item Parameter Recovery and Representation Coupling in Knowledge Tracing 

**Authors**: Wenrui Yuan, Stéphanie M. van den Berg, Johannes Steinrücke  

**Link**: [PDF](https://arxiv.org/pdf/2610.30211)  

**Abstract**: Knowledge tracing (KT) models can estimate item parameters through an item response decoder trained solely for next-response prediction, yet predictive accuracy does not guarantee parameter recovery. We study 2PL, GPCM, and NRM decoders paired with LSTM, Transformer, and DKVMN encoders. Response-model derivatives show why discrimination and option slope updates depend directly on the scale and spread of estimated ability. Affine logit identities show how item-dependent shifts and rescaling in estimated ability are absorbed into item parameters without changing response probabilities. In simulations, difficulty, mean threshold, and correct-option intercept recovery remain high under shuffled order across representation widths, whereas discrimination and keyed option slope contrast recovery require wider item representations and degrade further under fixed item order. Separating the sequence item representation (width $d_S$) from the item parameter key (width $d_M$), and pairing a wide key with a thin sequence representation ($\mathrm{SK}(64,8)$), reduces fixed-order ability distortions and improves recovery and cross-order consistency without sacrificing test accuracy or increasing model size relative to a shared width-64 baseline ($\mathrm{SH}(64)$). On TIMSS and EdNet, widening $d_M$ consistently improves agreement with an offline IRT reference, while the trade-off from narrowing $d_S$ depends on the encoder's reliance on sequence width for prediction. These findings show why item parameters, especially discrimination and option slopes, must be evaluated separately from response prediction. 

---
# MoEZip: Routing-Aware KV Cache Compression for Sparse Mixture-of-Experts LLMs 

**Authors**: Minsang Kim, Seungjun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2610.30206)  

**Abstract**: Mixture-of-Experts (MoE) language models scale model capacity efficiently by activating a small subset of parameters per token. However, in long-context inference, memory remains a major bottleneck because the KV cache grows linearly with sequence length. Existing KV cache compression methods reduce this cost for dense transformers, but overlook a problem specific to MoE: KV eviction can change which experts are selected during answer generation, degrading the generation quality. We propose MoEZip, a routing-aware KV cache compression for sparse MoE language models. To estimate the sensitivity of expert routing to perturbation, we propose a novel metric based on the Fisher information matrix of routing distributions. MoEZip combines this metric with attention weights for routing-aware scoring of contexts. Finally, we compose a complementary scoring system that combines eviction methods for dense and sparse-MoE architectures. Experiments show that MoEZip achieves robust performance under aggressive cache compression on various benchmarks. At a KV retention ratio of 10\%, MoEZip improves over the SoTA baseline by +24 and +20 points on average on LongBench and RULER, corresponding to 2.3$\times$ and 4.9$\times$ higher scores, respectively. 

---
# MPIDetector: Detecting Multi-modal Prompt Injection via LLM-based Reasoning 

**Authors**: Yiheng Li, Jingyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.30202)  

**Abstract**: With the rapid integration of Multi-modal Large Language Models (MLLMs) into real-world applications, Prompt Injection (PI) has emerged as a critical security threat. Attackers can conceal malicious instructions within textual narratives or visual inputs to bypass safety alignment. Existing approaches, such as signature- and perplexity-based methods, rely heavily on pattern matching, making them insufficient to detect sophisticated, cross-modal PI payloads. In this paper, we propose a novel detection framework, called MPIDetector, that leverages LLMs' intrinsic reasoning capabilities via few-shot In-Context Learning (ICL). Instead of fine-tuning, we provide the model with specific prompt templates and representative attack examples, enabling it to scrutinize the semantic intent of input across modalities. We constructed a comprehensive evaluation benchmark comprising eight textual and seven visual malicious categories, supplemented with benign datasets to rigorously test for false positives. Our evaluation across multiple leading LLMs, including Gemini-2.0-Flash, Qwen3-VL-32B-Instruct, and Doubao-seed-1.6-flash, demonstrates the efficacy and generalizability of this approach. The proposed method achieves up to 90.91% accuracy and 100% recall on textual tasks. In the visual domain, it achieves up to 99.28% accuracy while maintaining 100% precision across all evaluated models. Crucially, when evaluated against 500 highly obfuscated environmental web-agent injections (WebInject), MPIDetector achieved a detection rate up to 99.80%. Comparisons with two baseline approaches for detecting textual PI payloads, along with two additional methods for visual input detection, demonstrate MPIDetector's superior accuracy. These results show that LLM-driven reasoning provides a robust, cross-modal defense against advanced prompt injections. 

---
# Detection Is Not Enough: Reasoning Traces as Testable Program Hypotheses 

**Authors**: Vaidehi Hegde  

**Link**: [PDF](https://arxiv.org/pdf/2610.30200)  

**Abstract**: When a language model is deployed, no reference solution exists, so the reasoning trace it writes is the only evidence for judging its answer. The reference-free substitutes each fail differently: self-assessment degrades the answer it was meant to protect, sampling agreement measures stability rather than correctness, and perturbation studies read the expected answer off a human-written template, grading a model rather than a trace. We ask how far a single trace can be verified with no reference solution, no judge model and no weight update, and find, for a frozen LLaMA-2-7B-chat on GSM8K, that detection is not what limits such a pipeline. Checking a trace against itself reaches an AUROC of 0.741 and stops there for a structural reason: two thirds of wrong traces state no false equation, having executed a mistaken plan correctly, and on wrong-plan failures the detector flags a wrong trace less often than a right one. A candidate-derived metamorphic probe exposes many of these failures: we abstract the trace into a program over the question's quantities, perturb one quantity, execute the program, and ask the model the rewritten question once, so every trace supplies its own test. Valid perturbations exist for 937 of 1,319 test questions (71.0%); among the 847 that also have a parseable first trace, supported traces are correct 60.6% of the time against a 31.6% base rate, although the probe also refutes 45.5% of correct traces. No tested policy yields a statistically significant improvement in accuracy. A flag earns its retry only when detector precision exceeds D/(R+D), where R and D are the rates at which a retry repairs a wrong trace and damages a correct one; our repair rate is so low that, at the same retry budget, even a perfect detector would recover an estimated 66 of 1,319 questions. Repair, not detection, is the open problem. 

---
# Why Constrained Updates Still Forget: A Functional-Path Analysis of Continual SFT 

**Authors**: Qingsen Ma, Dianyun Wang, Jiahong Liu, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2610.30191)  

**Abstract**: Supervised fine-tuning adapts language models to specialized tasks but can disrupt previously acquired behavior. Existing approaches constrain parameters, gradients, or replay objectives, yet the connection between the protected object, the intervening trajectory, and retained outputs remains unclear. We develop a functional-path account that decomposes old-loss changes along realized updates into the controlled direction, gradient staleness, probe mismatch, and finite-step effects. The account yields a conditional refresh budget and identifies the restoring direction supplied by informative replay. Matched QLoRA interventions separate constraint granularity from placement before or after Adam: placement strongly affects retention across the studied code-label audits, whereas the granularity interaction depends on the endpoint. Checkpointed generation separates endpoint recovery from higher sampled-path accuracy, and matched replay schedules reveal loss exposure concealed by final scores. A sequential-task comparison extends this distinction to backward transfer: subspace regularization and projection improve average retention while trading off different old tasks; supervised replay protects both more consistently in this setting. Additional response interfaces distinguish content gains from exact-response retention. These findings connect update-level protection to task-wise behavior and motivate mitigation through explicit functional anchors, trajectory evaluation, and joint old/new auditing. 

---
# Agentic Verified Optimization of Real-World Software 

**Authors**: Zhe Ye, Udaya Ghai, Zhizhen Qin, Dawn Song, Soonho Kong  

**Link**: [PDF](https://arxiv.org/pdf/2610.30188)  

**Abstract**: Program optimization aims to improve performance while preserving behavior. Coding agents can already make programs faster by changing algorithms, data structures, and execution strategies. However, existing equivalence checks mostly rely on tests, which agents have been shown to exploit, passing the tests without preserving the intended behavior. We propose agentic verified optimization, in which an agent produces a faster implementation together with a machine-checked proof of equivalence to the existing one. Because the existing implementation serves as the specification, the task avoids the specification-writing bottleneck of verified code generation, and recent advances in theorem proving make the proofs attainable. We formulate the task at the function and repository levels in Lean 4, where programs can be both executed and reasoned about. Few real-world applications are written in Lean, so we develop Aqueduct, an agentic framework that migrates C, C++, Go, and Rust programs into native Lean, validates each port by differential testing against its original. On the 189 reference implementations in Verina, agents produce faster proved rewrites for 156, including algorithmic changes such as replacing exponential recursion with dynamic programming. On 17 migrated applications, their 92 proved rewrites speed up 47 of 51 application scenarios. These application-level gains are smaller than the function-level ones, making the choice of functions to optimize an important problem. Current agents can thus already carry out verified optimization on both standalone functions and applications migrated from real programs. We discuss the remaining challenges and the steps toward verified optimization in production. 

---
# Self-Evolving Static Analysis via LLM Pseudo-Labeling 

**Authors**: Rui Melo, Andre Duarte, Aditya Oke, Shubham Gandhi, Nachiket Kotalwar, Rui Abreu, Lei Li, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2610.30184)  

**Abstract**: Security bugs often slip past static analysis not because they are impossible to detect, but because detection rules have not yet been written for the code in front of us. New frameworks, APIs, and coding idioms continually create such gaps, leaving developers to discover failures and manually encode new rules. We present SESAA, a framework in which an Large Language Model (LLM) agent starts from no rules and iteratively evolves a set of readable and auditable joern queries.
The agent receives feedback only from a second LLM critic, which inspects flagged and unflagged code to identify likely false alarms and suspected misses, without access to ground-truth labels. The agent then uses this feedback to revise its queries. On the OWASP Benchmark for Python, SESAA reaches a mean F1 of 0.742 across 8 independent runs, well above the default configurations of five widely used tools and comparable to the strongest of them once it is tuned with access to ground-truth labels that SESAA never sees. In a pilot on real-world Python projects, queries induced from code available before a vulnerability was introduced detect 10 of 19 subsequent CVEs, including 9 detected by rules written for the same repository.
The resulting detector is a set of executable, human-readable queries that engineers can inspect, modify, and run independently. These results suggest that LLM agents can turn static-analysis rule maintenance from a manually authored process into an iterative process of automated discovery and repair. 

---
# Detecting AI Training in Datacentre Power 

**Authors**: Tom Kimpson, Mauricio Baker, Emlyn Graham  

**Link**: [PDF](https://arxiv.org/pdf/2610.30182)  

**Abstract**: AI governance agreements require methods to detect large model training runs on a monitored computing cluster, without trusting the operator. We develop a method to detect synchronous model training in facility-level power by Viterbi tracking of the wandering iteration cadence. On simulated datacentre power, when the operator makes no attempt to hide, the Viterbi tracker detects at least $96\%$ of training runs against a declared inference workload, at a $1\%$ false-alarm rate and within $0.04$ of the Neyman--Pearson optimum, given a meter that samples at $2$ Hz or faster. Of eight adversarial strategies an operator might use to hide, only extreme variation of the work per iteration evades the tracker. Its strongest form hides training in up to $73\%$ of five-minute observations, but over a day at least $87\%$ of such runs are still caught, at a $0.1\%$ false-alarm rate per run. This adversarial strategy costs no throughput, but in small-scale training the operator needs $93$--$267\%$ more compute to reach the same loss. A modest passive meter is sufficient to make hidden training both detectable and expensive, which makes power metering a promising part of compute verification that does not depend on trusting the operator. 

---
# GeoDiv KAN Solver for Time Fractional Navier Stokes Equations on Complex Domains to Realistic CFD Reconstruction 

**Authors**: Yatin Kumar, Shivani Saini, Ramesh Kumar Vats  

**Link**: [PDF](https://arxiv.org/pdf/2610.30179)  

**Abstract**: Scientific machine learning provides a promising framework for solving complex problems in science and engineering. Notably, physics informed neural networks (PINNs) have shown considerable potential for solving differential equations and modelling engineering systems. Solving time fractional Navier Stokes equations (TFNSEs) on complex domains is challenging due to temporal nonlocality, nonlinear flow dynamics, geometric complexity, and incompressibility constraints. To address these challenges, we propose a geometry gated, divergence-free Gaussian radial basis function Kolmogorov Arnold network (GeoDiv KAN). The proposed architecture uses geometry dependent feature gating and a streamfunction formulation to preserve incompressibility by construction. GeoDiv KAN is evaluated for the time fractional incompressible Navier-Stokes equations on square, circular, and star-shaped domains. Numerical experiments show that GeoDiv KAN achieves higher predictive accuracy than PINN, PIKAN, RBF KAN, and DF RBFKAN across the considered benchmark problems. To establish practical relevance, GeoDiv KAN is applied to two data driven CFD cases: unsteady flow past a circular cylinder using the \href{this https URL}{Zenodo dataset} and turbulent RANS flow through a complex U bend using the \href{this https URL}{U-bend dataset}. 

---
# IRAH: Hierarchical Agentic Tree Search for Retrieval Pipeline Optimization 

**Authors**: Maria Ana Cardei, Aniket Singh, Amir Moghadam, Srividhya Baikadi, Rashmi Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2610.30167)  

**Abstract**: Optimizing Information Retrieval (IR) pipelines for Retrieval-Augmented Generation (RAG) requires jointly selecting structural components and numerical hyperparameters under limited evaluation budgets. We introduce the Information Retrieval Agentic Harness (IRAH), which combines Language Agent Tree Search for IR (LATS-IR) with offline meta-learned initialization. LATS-IR performs UCT-guided hierarchical search, first exploring macro-structural pipeline choices and then refining numerical parameters, with natural-language reflections to guide subsequent mutations and provide inspectable search traces. We evaluate IRAH on three in-distribution benchmarks and one out of-distribution benchmark to assess generalization under distribution shift. Under the same meta-learned initialization, LATS-IR outperforms TPE on two of three in-distribution benchmarks, with gains of up to 6.4%, and remains competitive on the out-of-distribution benchmark. Across all four benchmarks, warm-started IRAH achieves the highest average held-out nDCG@10 (0.4816) and improves over its cold-start counterpart. These results establish IRAH as a competitive approach to retrieval pipeline optimization under constrained evaluation budgets. 

---
# FusionEval: Benchmarking Model Fusion for Large Language Models 

**Authors**: Shuo Cai, Yanggan Gu, Yuhang Liu, Su Lu, Yuanyi Wang, Yang Yu, Zhen Li, Zhitian Hou, Guanghao Zhu, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30162)  

**Abstract**: Model fusion aims to integrate the capabilities of multi-expert models into a single model. Existing evaluations offer limited coverage of reasoning with long chains of thought and tool use, while strong average performance can mask substantial losses in individual expert capabilities. We introduce FusionEval, a benchmark that provides controlled expert pools, integrates fusion methods with and without additional training, and offers a unified evaluation suite. The pools contain domain experts independently trained from Qwen3 and Llama base models across math, code, science, instruction following, and tool use. The suite evaluates capability retention and computational cost on eight public benchmarks under comparable data access rules, reporting normalized average and worst performance, runtime, and peak GPU memory usage. Extensive experiments show that training-free methods preserve reasoning performance but substantially degrade tool use at both model scales. In contrast, MOPD preserves tool use and achieves a more balanced capability profile, but requires additional data preparation, inference, and training. Jointly retaining reasoning, instruction following, and tool use at low computational cost remains a challenge. We hope FusionEval supports future research on efficient model fusion that jointly preserves expert capabilities. Our code is available at this https URL. 

---
# TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents 

**Authors**: Krishanu Adhikary, Prashant Devadiga  

**Link**: [PDF](https://arxiv.org/pdf/2610.30161)  

**Abstract**: Many Indian banks serve customers through FAQ pages and scripted chatbots. Assistants based on language models could respond faster and carry out requests such as blocking a card, but only if they give correct information and follow the bank's rules. These rules prescribe a procedure, a set of required steps, such as obtaining a closure quote before closing a deposit. Adapting models to this domain requires measuring how well they follow the procedure, before and after training. Benchmarks usually judge only the records left at the end of a conversation. Many required steps, however, leave the records unchanged, so an assistant that skips them can still be judged successful. We present TauIndianBankBench, an action-oriented benchmark of 1,000 Indian retail-banking tasks where an assistant helps a simulated customer. Besides the final records, it checks that the assistant took every required step and made no unrequested change. For every model, 6.5-12.0% of sessions are judged successful when only the final records are checked, although the assistant skipped a required step or made an unrequested change. Scoring both procedure and outcome shows more accurately how well language models perform in Indian retail banking. 

---
# V-JEPA 2 Meets AQA: Phase-Based Scoring and Reference-Based Feedback for Mountain Biking 

**Authors**: Santhiya Rajan, Savio Cardoz  

**Link**: [PDF](https://arxiv.org/pdf/2610.30160)  

**Abstract**: We study mountain-bike action quality assessment (AQA) with weak labels and candidate feedback. The training corpus comprises 95 field-recorded jump clips with binary labels denoting good or poor technique. These are coarse categorical labels, not continuous expert quality scores; phase-boundary and aspect-level annotations are unavailable. Our pipeline summarizes four heuristic phases using frozen V-JEPA 2 video features and two-dimensional pose and motion measurements. A small transformer predicts a binary-label score. A separate reference layer compares phase-specific measurements with those of clips labelled as good technique to propose phase, body-parameter, and direction cues. We detail the measurements, network, and evaluation so that the limits of each output are explicit. In five-fold clip-level cross-validation, the full scorer obtained pooled out-of-fold AUROC 0.768 and average precision 0.812. These are exploratory figures because the reporting fold also selected the neural training epoch and riders were not separated between folds. Removing biomechanical scoring inputs raised AUROC to 0.809. A separate inner-selection run obtained AUROC 0.678, but used about 57 rather than 76 training clips per outer fold, so its difference from the test-selected run does not quantify selection bias. The unsupervised aspect head did not learn distinct factors. On a 17-clip external probe, an earlier checkpoint obtained binary accuracy 0.412 and Spearman association -0.135 with expert scores. Candidate cues had precision 0.118 and recall 1.0 against each of two coaches, with substantial disagreement about direction. These findings do not establish rider-independent scoring or reliable corrective feedback; they identify the empirical work needed to evaluate either use. 

---
# Lighthouse: On Building a Trustworthy Digital Assistant for Exploring History with Civil Rights Archives 

**Authors**: John A. Aydin, Angelica Williams, Steven Dindl, Nathan Saunders, Kate Boyd, Vandana Srivastava, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2610.30158)  

**Abstract**: Artificial intelligence (AI), as a decision-support aid, provides new tools for studying history effectively using multimodal data, including text, images, and audio. We present, to our knowledge, the first digital assistant named Lighthouse--commonly called a chatbot--for exploring the Civil Rights period of the United States using data from South Carolina. From an AI perspective, the system's novelty lies in its guiding design principles of (a) focusing on trustworthiness of content and (b) by modeling and reducing potential harm to users from generated answers. It implements these with a neuro-symbolic approach that combines neural learning through a large language model (LLM) with rule-based symbolic solving, balancing risk of answers with generality of coverage. We evaluate Lighthouse with both factual objective questions that admit a single acceptable answer as well as subjective questions. For the latter, a dataset of subjective question-answer pairs (QAs) is created by discovering a "gold standard" by aggregating answers from independent experts. Lighthouse has access to over 18k primary sources from a range of collections and can be easily extended. 

---
