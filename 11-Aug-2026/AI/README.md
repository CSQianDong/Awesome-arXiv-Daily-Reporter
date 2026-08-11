# GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis 

**Authors**: Alban Puech, Matteo Mazzonelli, Tamara R. Govindasamy, Mangaliso Mngomezulu, Héctor Maeso-García, Thomas Tolhurst, Javad Bayazi, Ali Moeini, Naomi Simumba, Celia Cintas, David Nelischer, Romeo Kienzler, Jonas Weiss, Anna Varbella, Florian Dörfler, Gabriela Hug, Martin Mevissen, Juan Bernabé-Moreno, François Mirallès, Hendrik F. Hamann, Etienne Vos, Thomas Brunschwiler  

**Link**: [PDF](https://arxiv.org/pdf/2608.09921)  

**Abstract**: Foundation models are transforming business workflows and boosting productivity, yet they remain largely absent from engineering domains such as power system analysis, where strict physical consistency must be enforced.
We present GENCO (GEometric Neural Corrective Optimizer), a unified neural solver for steady-state transmission grid analysis that handles power flow (PF), optimal power flow (OPF), and state estimation (SE) within a single architecture and shared network representation. To support advances in neural power system solvers, we introduce the open-source GridFM Development Framework, which standardizes synthetic data generation and training in a low-code environment. We also release large-scale datasets with millions of PF and OPF scenarios across diverse grid topologies to support reproducible benchmarking.
We evaluate GENCO on the PFDelta and OPFData benchmarks against state-of-the-art neural solvers and classical solvers, including Newton-Raphson and IPOPT, as well as on real-world Hydro-Québec SCADA data. For large-scale PF, GENCO recovers the full AC operating state, including voltage magnitudes and reactive power that DC-PF cannot provide, while matching DC-PF-level active power-balance residuals. It achieves up to 30x speedups over Newton-Raphson at only 2x the runtime of DC-PF. For OPF, it achieves up to 85x speedups over IPOPT while improving feasibility, optimality, and runtime over DC-OPF. For SE, GENCO is more robust than classical weighted least squares to noisy measurements and network parameter errors, and always returns a high-quality estimate even when weighted least squares fails to converge.
Together, the unified architecture and development framework provide a new approach to large-scale steady-state grid analysis, lowering the barrier to entry for power system engineers and marking a step toward Grid Foundation Models. 

---
# DSLE: A Learning Environment for Dark Souls Boss Encounters 

**Authors**: Derin Gezgin, Jim O'Connor, Tanner Goodwin, Gary B. Parker  

**Link**: [PDF](https://arxiv.org/pdf/2608.09902)  

**Abstract**: We introduce the Dark Souls Learning Environment (DSLE), a containerized platform that presents all 22 boss encounters of Dark Souls: Remastered as game-playing agent benchmarks through a Gymnasium-style interface. DSLE combines real-time combat, high-dimensional visual input, and sparse terminal rewards, with each environment step being a real action executed against the running game. To support controlled comparison, we define DSLE-5, a representative five-boss subset, spanning a melee fight, a spatially constrained arena, an environmental-hazard fight, a multi-target fight, and a fast final-boss fight, that we recommend as the starting suite for agents built on DSLE. On DSLE-5 we evaluate a random policy, an expert system, an evolutionary baseline, and PPO and DQN agents trained from visual input. The expert system and the evolutionary baseline each defeat the Asylum Demon, the game's tutorial boss (63% and 43% peak win rates), but none of the five methods defeats the other four DSLE-5 bosses; PPO and DQN show no measurable learning (at most 0.33% win rate on the tutorial boss, 0% elsewhere) within a budget that already costs tens of wall-clock hours per run. A broader study running the evolutionary baseline across all 22 encounters under advantaged all level-50 stats yields wins on only a handful of additional early-game bosses and leaves the rest unwon. The failure cases range from sub-10-second deaths in cramped, multi-target encounters to minute-long stalemates that inflict almost no damage, and we report them through survival time and damage dealt rather than win rate alone. 

---
# SHE: Trajectory-driven Safety Harness Evolution for LLM Agents 

**Authors**: Wanying Qu, Qinghua Mao, Yu Li, Jiyao Liu, Xin Zhang, Dadi Guo, Yanxu Zhu, Qingyu Liu, Leitao Yuan, Xi Lin, Shanfeng Zhu, Yanwei Fu, Jing Shao, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09885)  

**Abstract**: The safety of large language model (LLM) agents depends not only on model weights but also on the agent harness that manages context, memory, tools, permissions, and runtime control. Existing safety mechanisms often treat the harness as a fixed deployment artifact, limiting their ability to evolve with emerging risks. Moreover, coupled functions across harness components obscure safety responsibility attribution, making localized evolution difficult. We propose Safety Harness Evolution (SHE), a framework that learns evolving safe boundaries from rollout trajectories. SHE decomposes the harness into four artifacts with explicit safety responsibilities, including the System Prompt, Rule Bank, Safety Memory, and Tool Policy, defining clear functional boundaries for localized evolution. Based on this decomposition, SHE introduces an attribution-guided evolution loop that converts trajectory failures into structured diagnoses, learns artifact-specific boundary refinements, and selects evolved harnesses through safety-utility validation. Experiments on Agent-SafetyBench demonstrate that SHE effectively enhances safety through harness evolution, achieving a 3.1x ASR reduction compared with static SafeHarness, while also improving benign utility. The evolved harness further generalizes to unseen risks on the held-out AgentHarm benchmark and transfers across agent models without additional evolution. 

---
# ArchAgent v2: A Case Study with the Data Prefetching Championship 

**Authors**: Abraham Gonzalez, Raghav Gupta, Akanksha Jain, Hanna Alam, Alexander Novikov, Po-Sen Huang, Matej Balog, Marvin Eisenberger, Sergey Shirobokov, Ngân Vũ, Hank Levy, Borivoje Nikolić, Sagar Karandikar, Martin Dixon, Parthasarathy Ranganathan  

**Link**: [PDF](https://arxiv.org/pdf/2608.09874)  

**Abstract**: Agentic artificial intelligence has shown great promise in automating algorithm design, but scaling similar techniques to computer microarchitecture discovery remains challenging due to vast search spaces, strict hardware budgets, and long simulation times. In this work, we present ArchAgent v2, a framework which scales automated microarchitecture search to multi-level data prefetching. While the original ArchAgent successfully discovered single-level cache replacement policies in competition settings, it does not scale to multi-level prefetching where the design space and degrees of freedom are larger. To overcome this, we introduce two new additions to ArchAgent: a cascaded evolutionary search that subdivides the design space by sequentially evolving and freezing prefetchers at individual cache levels, and a hardware-realizability feedback loop that embeds real-time size-estimation directly into the evolution process.
Evaluated under identical rules of the 4th Data Prefetching Championship (DPC4), ArchAgent v2 automatically designs a three-level prefetcher that outperforms the winning hand-designed solution, further demonstrating automated agentic discovery as a useful tool for computer architects. Our discovered policy achieves a 3.8\% geometric mean IPC speedup over the baseline overall and a 0.3\% improvement over the prior champion, BertiGO. On low-bandwidth single-core configurations, our policy yields a 4.6\% performance speedup compared to only 2.6\% for BertiGO. However, multi-core evolution still remains a significant challenge due to simulation latency impeding evolution speed. Finally, our profiling of an ArchAgent evolution of over 12,000 candidate designs provides key insights into how automated evolutionary agents explore and synthesize complex microarchitectural logic. 

---
# Towards Expert-level Medical AI for Real-time Video Consultations 

**Authors**: Mahvish Nagda, Jihyeon Lee, Matthew Thompson, Chunjong Park, Tim Strother, Valentin Liévin, Roma Ruparel, Akshay Goel, Teya Bergamaschi, Suhana Bedi, Meet Shah, Pavel Dubov, Liviu Panait, Toshiyuki Fukuzawa, Sam Schmidgall, Craig Schiff, Joseph Xu, Aliya Rysbek, Yana Lunts, Jan Freyberg, Rebecca Hemengway, Sunny Virmani, David Racz, Carey Radebaugh, Joëlle Barral, Kavi Goel, Dale R. Webster, Katherine Chou, Avinatan Hassidim, Yossi Matias, James Manyika, Gregory Wayne, Tao Tu, Yun Liu, Ethan Goh, Christina Chen, Ryutaro Tanno, Po-Hsuan Cameron Chen, Mike Schaekermann, Anil Palepu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09861)  

**Abstract**: Audio-visual interaction is the standard for patient-physician consultations, enabling natural communication and effective assessment of illness through non-verbal cues. While text-based AI has shown promise, it discards essential perceptual dimensions and limits patients who cannot articulate symptoms in writing. Early efforts to extend medical AI to audio-visual interaction have demonstrated feasibility but not reached clinician-level performance. Here, we provide the first demonstration of expert-level AI in real-time clinical video consultations using AMIE (Articulate Medical Intelligence Explorer) in a video configuration. AMIE (Video) is a Gemini-based multi-agent system integrating low-latency dialogue, clinical reasoning, and real-time audio-visual perception. To guide development, we established a taxonomy and automated evaluations for clinical audio-visual cues in telehealth settings. In a randomized Objective Structured Clinical Examination (OSCE) study with 30 primary care physicians (PCPs), 15 patient actors and 100 clinical scenarios, we compared AMIE (Video), its text-only counterpart AMIE (Text), and PCPs consulting via video. Clinical evaluators rated AMIE (Video) on par or better than PCPs in history-taking, diagnosis, management, and physical observation and examination. Patient actors preferred AMIE's approach to assessing and explaining conditions, while PCPs were preferred for rapport and partnership building. In modality ablation, patient actors preferred AMIE (Video)'s interface over text chat for communicative effectiveness, convenience, and feeling understood. Limitations remain in fine anatomical precision, subtle affective nuances, and high-frequency movements. While further research is needed before real-world translation, these results mark an important milestone toward AI systems capable of augmenting care across the sensory complexity of clinical practice. 

---
# Agentic Auto-Research is Fuzz Testing 

**Authors**: Yifeng He, Jicheng Wang, Yinzhe Zhao, Jiachen Liu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09855)  

**Abstract**: Autonomous research agents can generate experiments faster than researchers can validate them. Researchers have responded by scaling the proposer and ranking more samples with a learned judge or human reviewers. We argue that this *generate-and-rank* paradigm misses the problem of sparse feedback. Within a declared research problem, an agent follows the control loop of a greybox fuzzer: it proposes a candidate, executes it, observes feedback, and chooses what to try next. A fuzzer rarely finds a bug, but coverage makes partial progress observable on every execution. Fuzzers then use that signal to mutate inputs and allocate effort, rather than only to rank completed runs. Auto-research needs the same two capabilities. First, each experiment should expose a cheap, dense signal of epistemic progress before final scientific validation is available. Second, that signal should determine the next intervention so that the agent searches rather than repeatedly samples. Because the optimized progress signal is guidance rather than a verdict, final validation must still decide what counts as a discovery using evidence protected from adaptive reuse. We propose controlled tests of whether candidate signals predict validated progress, whether feedback-directed search yields more validated discoveries per unit cost than repeated sampling, and whether protected validation reduces false discoveries. Feedback architecture, not only generation, is a central bottleneck in auto-research. 

---
# CEAA: A Cognitive Embodied Agents Architecture for Interactive Computing Systems 

**Authors**: Aimilios Hadjiliasi, Louis Nisiotis  

**Link**: [PDF](https://arxiv.org/pdf/2608.09848)  

**Abstract**: The development of embodied Intelligent Virtual Agents (IVAs) that have cognitive capabilities in real-time interactive virtual environments remains a challenge, even with today's advancements in technology. Existing architectures are often focused on either the implementation of low-level reactive control systems that are constrained by commercial game engines, or high-level representations of reasoning models that can be difficult to implement in virtual worlds. This paper builds on that notion and proposes a modular cognitive architecture for deploying embodied IVAs. This architecture builds on existing, pre-established frameworks such as the Sense-Think-Act paradigm and the Belief-Desire-Intention cognitive model, among others, and aims to provide a reusable implementation-oriented framework as a template for deploying IVA "brains" in interactive 3D computing systems. The proposed architecture contributes by providing a modular, implementation-oriented framework for the deployment of embodied, cognitive-capable IVAs and bridges the gap between high-level agent reasoning models with real-time embodied execution, for scalable, adaptive, and explainable agents in complex interactive virtual environments. 

---
# Mismatch Matters: On-Policy Distillation Beyond Token Agreement 

**Authors**: Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.09836)  

**Abstract**: On-policy distillation (OPD) has emerged as a core component of modern LLM post-training pipelines, yet we reveal a failure mode: degenerate agreement, where students exploit repetitive loops to achieve near-perfect token agreement with the teacher despite globally flawed responses. We therefore shift our focus from agreement to teacher-student mismatch, and find that mismatch tokens can be mainly categorized into two types: student-excess tokens and student-deficit tokens. Student-excess tokens are generated by the student but assigned near-zero probability by the teacher; their log-ratio corrections grow unbounded and destabilize the update. Student-deficit tokens, in contrast, are preferred by the teacher but rarely sampled by the student; their absence blocks the transfer of the teacher's reasoning patterns. To tackle these mismatch directions, we propose TIDE (Token-level Independent Deficit-Excess correction), which applies bounded Hellinger shaping to suppress the most severe sampled excesses and an analytic teacher top-$K$ injection to restore deficient probability mass without requiring deficit tokens to be sampled. Across mathematical reasoning benchmarks with multiple Qwen3 teacher-student pairs, TIDE consistently outperforms standard OPD and recent token-selection and reward-shaping baselines. Moreover, the gains of TIDE are more pronounced under strong teacher-student mismatch, where it improves Avg@8 from 6.9% to 20.3%, reduces average response length by a factor of 3.6, and substantially reduces formatting failures. Code is available at this https URL 

---
# CARD: Controlled Agentic Reddit Discussions for Credit Card Simulation 

**Authors**: Yaoning Yu, Kai-Min Chang, Ye Yu, Yi-Chia Wang, Haojing Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09790)  

**Abstract**: Online credit card discussions provide a natural setting for studying how consumers communicate about financial products. Simulating these discussions requires more than just generating individual comments, the generated threads should also match how real users express themselves and interact with others. We introduce CARD, a framework for generating realistic credit card discussion threads. Given a credit card post and its matched real thread, CARD uses non-verbatim guidance on reply structure, comment function, stance, tone, and conversational variation. A planner organizes these controls, a writer generates the discussion, and a calibration loop updates comments' populations that contribute to differences between the generated and real thread distributions. We evaluate CARD on real Reddit credit card discussions using lexical, semantic, behavioral, and structural metrics. CARD matches the distributions of real credit card discussions better than simulation baselines across multiple LLMs and also demonstrates smaller effect sizes and distribution distances across metrics. These results show that structured planning and targeted revision can generate the realism of simulated credit card discussions. 

---
# AirFlow: Context Preserving and Multi-Rate State Modeling for Air Quality Forecasting 

**Authors**: Fan Yang, Nan Chen, Yijie Dong, Yuchen Zhang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09775)  

**Abstract**: Accurate air quality forecasting is essential for public health and urban environmental management, but remains challenging because pollutant channels differ in periodicity and distribution drift, while their concentration trajectories contain both multi-scale dependencies and rapid changes. Recent methods have improved spatial dependency learning and meteorological covariate modeling. However, pollutant channels are still passed through the same normalization rule and temporal backbone, using a shared latent representation for channel-specific distributions and changes at different rates. To address this limitation, we propose AirFlow, a pollutant-aware dual-stream framework that operates on station multivariate observations without additional graph propagation or predefined signal decomposition. Specifically, AirFlow designs two novel blocks: (1) a statistic-guided normalization routing mechanism that selects a normalization path for each pollutant according to its 24-hour autocorrelation and distribution drift; and (2) a hierarchical dual-stream state model that combines multi-scale state space propagation with learnable response coefficients, where gated bidirectional cross-attention exchanges information and adaptively fuses the resulting representations. Experiments on real-world data from multiple cities show that AirFlow achieves the best performance in 34 of 36 metrics comparisons, with reductions of up to 11.11% root mean square error over the state-of-the-art baseline. AirFlow also requires only 0.0483M parameters and 0.0215G FLOPs, achieving high forecasting accuracy with low computational overhead. 

---
# Second-Order Muon Done Right: A Principled Marriage of Spectral Geometry and Curvature 

**Authors**: Tong Che  

**Link**: [PDF](https://arxiv.org/pdf/2608.09763)  

**Abstract**: Muon's polar update is exact for an unweighted spectral geometry. We introduce GO-MUON, which uses a matched data-dependent geometry and reuses it across several optimization steps. Conditioned on any positive-definite left and right maps, its raw update exactly solves the corresponding weighted spectral oracle; this statement is independent of how the maps are estimated or how recently they were refreshed. For softmax cross-entropy, we quantify when the observed-label backward factor approaches the model Fisher and generalized Gauss--Newton factor. We also show that four-step refresh nearly preserves the tracking delay of slowly changing geometry while increasing stationary factor noise, making lazy geometry a compute--statistics tradeoff rather than a denoising mechanism. 

---
# Matryoshka Language Model Suites 

**Authors**: Nathan Godey, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2608.09703)  

**Abstract**: Training a language model suite classically requires training each model separately and serving them independently. We improve both training and inference efficiency by stacking sub-models of increasing size into a single nested architecture trained end-to-end. This Matryoshka training framework reduces the total parameter count of the suite, enables low-cost distillation from the largest to all smaller sub-models at every training step, and is well-suited for speculative decoding as the draft model is contained within the verifier. We validate our approach by training a Matryoshka suite comprising 500M, 1.5B, and 3B sub-models. Our suite is on par with independently trained baselines on benchmark performance and validation and out-of-domain perplexities, while using 36% less training compute and improving the throughput of speculative decoding by 14-26%. We also ablate key architectural choices, offering guidance for building strong Matryoshka LM suites. 

---
# Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models 

**Authors**: Kevin Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2608.09696)  

**Abstract**: Predicting the answer to interventional ``what if'' questions --- the outcome of an action never taken --- requires a \emph{mechanistic}, causal model, not a curve fit; and learning such a model requires \emph{experiments}, because passive data leaves its mechanisms unidentified. Experiments are expensive, so the central problem is \emph{data efficiency}. We present the Model Discovery Agent (MDA), which couples a large language model (LLM), used as a \emph{proposer} of candidate structures, with standard Bayesian machinery --- sequential Monte Carlo (SMC) for parameter and structure posteriors, simulation-based inference (SBI) for intractable likelihoods, and value-of-information (VoI) for experiment design --- to discover latent mechanistic world models from few interventions. MDA operates in the M-open setting: when the truth lies outside the current hypothesis class, a predictive check flags the inadequacy and the proposer expands the hypothesis space with a new model whose parameters are then identified by designed experiments. We show that \emph{discovery and design reinforce}: the design step identifies the mechanism the discovery step proposes, and the identified mechanism improves predictions, enabling further discoveries from the remaining unexplained residuals. On three different benchmarks --- covering physics (\DPbench, \citep{wiemann2026discoverphysics}), chemistry (\CHEMbench, \citep{kabra2026autoscilab}) and biology (\HHbench, a new partially observed single-neuron electrophysiology benchmark we create) --- we show that MDA sets a new SOTA in terms of data-efficient model learning and reliable interventional forecasting ability. 

---
# Adaptive Semantic Capacity Allocation for Parallel Generative Recommendation 

**Authors**: Chenxi Li, Yuchen Lu, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09685)  

**Abstract**: Autoregressive semantic ID recommenders are constrained by expensive beam-search decoding, which limits the practical length of item identifiers. Parallel generation methods alleviate this bottleneck by predicting all semantic ID tokens simultaneously, enabling longer IDs. However, existing semantic ID methods still rely on manually predefined and homogeneous ID structures, where both the number of semantic slots and the codebook size of each slot are treated as fixed hyperparameters. This ignores the heterogeneous capacity demands of different semantic subspaces and may allocate prediction capacity to slots with limited utility. We show that uniformly expanding semantic slots can provide limited gains, indicating redundant capacity in homogeneous semantic IDs. We propose InforID, a lightweight adaptive semantic target construction framework for parallel generative recommendation. InforID allocates a fixed capacity budget across candidate semantic slots, thereby jointly determining the effective ID length and slot-specific codebook sizes. Experiments demonstrate improved recommendation accuracy under comparable capacity budgets while preserving one-step parallel prediction. 

---
# Open Evaluation Agent: Efficient and Promptable Evaluation of Visual Generative Models 

**Authors**: Shulin Tian, Ziqi Huang, Fan Zhang, Hongyuan Zhu, Yu Qiao, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09666)  

**Abstract**: Recent advances in visual generative models have enabled high-quality image and video generation, but evaluating these models often demands sampling hundreds or thousands of images or videos, which is computationally expensive. Existing evaluation methods also rely on rigid pipelines that overlook specific user needs and provide numerical results without clear explanations. Mimicking how humans quickly form impressions of a model's capabilities from only a few samples, we propose the Evaluation Agent framework, which employs human-like strategies for efficient, dynamic, multi-round evaluations, offering detailed, user-tailored analyses. Given a natural-language evaluation request, the agent decomposes it into sub-aspects, generates targeted prompts, samples images or videos from the evaluated model, invokes suitable evaluation tools, and iteratively updates its plan from the observed evidence, covering both predefined benchmark dimensions and open-ended user concerns. The framework is thus efficient, promptable, explainable, and scalable across models and tools. Experiments show that Evaluation Agent reduces evaluation time to 10% of traditional methods while delivering comparable results. We further introduce Open Evaluation Agent (Open-EA) by constructing EA-CoT-10K, a corpus of history-conditioned step-level instruction-tuning records derived from multi-round evaluation rollouts, and training EA-3B from Qwen2.5-3B-Instruct as a local planning backbone that preserves the structured reasoning, tool invocation, and summary protocol of the API-based agent while reducing dependence on proprietary backbones. Experiments validate the API-based agent on established T2I/T2V benchmarks and open-ended queries, and evaluate Open-EA on four in-domain and three out-of-domain T2V generator families, showing partial cross-family transfer of the learned policy. 

---
# Hallucination-Free GUI Grounding via Regression-Free Layout-Aware Matching 

**Authors**: Yuke Li, Xuehan Hou  

**Link**: [PDF](https://arxiv.org/pdf/2608.09654)  

**Abstract**: GUI agents are shifting from metadata-dependent large language models to purely visual multimodal large language models (MLLMs) that operate directly on screenshots. The core task, GUI grounding, requires translating abstract user instructions into precise element coordinates. This task faces a persistent dual obstacle: conventional grounding models lack the semantic richness to interpret abstract instructions, while end-to-end MLLMs suffer from coordinate hallucinations caused by deficient fine-grained perception. We propose a regression-free framework where a frozen MLLM performs instruction parsing and a dedicated grounding model handles precise localization without learning any coordinate regression. A frozen MLLM first elaborates the abstract instruction into a structured visual description rich in layout cues. These descriptions are then fed to a novel Layout-Aware GUI Grounding Model, which performs regression-free localization by matching against layout-prior candidates, inherently suppressing hallucinations and avoiding expensive fine-tuning. The grounding model is trained with only Text/Icon binary labels, requiring no coordinate regression parameters. On ScreenSpot-Pro, our method achieves over 20% improvement in grounding accuracy over end-to-end systems; on Mind2Web, it raises success rate and element selection rate by more than 15%. These results demonstrate that decoupling instruction understanding from layout-aware localization effectively resolves the core challenges of GUI interaction. 

---
# Avalon-ToM-Bench: Evaluating Fine-Grained Theory of Mind via Asymmetric Game Mechanics 

**Authors**: Yen-Shan Chen, Yu Chian Duan, Chih-En Kuo, Jian-Bin Wu, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09638)  

**Abstract**: Theory of Mind (ToM) is essential for agent interactions, yet existing evaluations either rely on static scenarios that oversimplify mental-state reasoning or interactive settings that provide limited diagnostic insight. We present Avalon-ToM-Bench, a fine-grained benchmark that operationalizes ToM through the asymmetric-information mechanics of The Resistance: Avalon. Rather than evaluating end-to-end gameplay, it decomposes ToM into a 2$\times$2 taxonomy -- epistemic versus motivational reasoning crossed with inference versus action -- using human-crafted, perspective-constrained queries. Benchmarking 28 LLMs reveals three insights: 1) Reasoning, not knowledge. Models show strong game-rule comprehension but markedly weaker ToM abilities, isolating failures to social reasoning rather than missing domain knowledge. 2) Expression, not representation. Mechanistic analyses via linear probing and activation steering show that models frequently represent correct mental-state inferences in their hidden states but fail to express them during generation -- linear probes recover 77-82% accuracy versus 62-70% from the models' own chain-of-thought. 3) Policy, not deliberation. Dedicated reasoning training yields substantial improvements whereas test-time chain-of-thought provides only marginal gains (+11.0 versus +1.1 points on average), suggesting that robust ToM depends on a learned reasoning policy rather than increased inference-time deliberation. 

---
# Rethinking Self-Evolving Agents: Do We Still Need Prescribed Optimization Pipelines? 

**Authors**: Hui Xue, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09629)  

**Abstract**: Self-evolving agents are usually built around prescribed optimization pipelines: the framework decides how to gather evidence, revise a persistent artifact, select candidates, and stop. We ask whether this task-specific procedure remains necessary when a frontier model acts as the optimizer. We introduce Open-Ended Optimization (OEO), which keeps the objective, permitted interactions, resource budget, data boundary, and evaluation fixed while allowing the optimizer to compose the improvement process online. We compare OEO with two complementary prescribed approaches: SkillOpt, a staged pipeline with bounded edits, and GEPA, a reflective evolutionary search. Across 14 head-to-head comparisons over 8 benchmark-target-model settings, GPT-5.5-driven OEO records 12 wins, 1 tie, and 1 narrow loss of 0.21 percentage points. It uses a median 34.3 percent of SkillOpt's configured target-interaction token budget. A one-shot, zero-interaction control shows that the gains are not explained by a single prior-driven rewrite. However, delegation has a capability boundary: SkillOpt outperforms OEO with a medium optimizer, and a weak optimizer cannot operate through the unchanged OEO interface. In the fully instrumented OEO-SkillOpt pair, trajectory analysis further shows that prescription changes how optimization proceeds more consistently than it changes final behavior. Together, these findings recast prescribed pipelines as capability-dependent scaffolding: essential constraints remain external, but a sufficiently capable optimizer can compose the route from measurable feedback to persistent improvement. 

---
# Adaptive Sequential Test Planning for Multi-Mechanism Reliability Qualification via Bayesian Monte Carlo Tree Search 

**Authors**: Youssef A. Elhagrasy, Ian Hill, André Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2608.09622)  

**Abstract**: Reliability qualification of advanced semiconductor devices requires sequential stress decisions that balance characterization objectives against multiple competing failure mechanisms. Current practice relies on static test plans derived from population-level acceleration models, which cannot adapt to per-unit variability or real-time degradation observations. This paper presents a closed-loop adaptive test planning framework that formulates reliability qualification as a partially observable sequential decision problem and solves it using Monte Carlo tree search for seed-action simulators (MCTS-SA) coupled with extended Kalman filter (EKF) belief-state estimation. The framework models stochastic, per-device variability in bias temperature instability (BTI), electromigration (EM), and time-dependent dielectric breakdown (TDDB), and treats stress selection as a constrained sequential optimization, i.e., to maximize the probability of successful degradation characterization while respecting catastrophic failure constraints. Under the experimental assumptions used here (discrete stress actions, proxy damage observability, and cumulative degradation without recovery), we believe this to be a novel application of tree-search-based adaptive test planning to multi-mechanism reliability qualification. Across 5,000 planning iterations, the characterization yield (CY) improves from 20% in the first 500 iterations to over 54% in the final 500, with 39% cumulative success, while the best successful test sequence terminates with EM and TDDB damage fractions DEM=0.564 and DTDDB=0.537, well within safety margins. These results demonstrate that sequential Bayesian planning can synthesize damage-aware test policies that significantly outperform non-adaptive strategies for reliability qualification under competing failure modes. 

---
# From Sweep to Seam: Interleaved Cross-Block Post-Training Quantization 

**Authors**: Achille Jacquemond, Yuma Ichikawa, Akira Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09595)  

**Abstract**: Compressing large language models to two bits or fewer is increasingly feasible through block-wise post-training quantization; cross-block variants reconstruct neighboring Transformer blocks within a moving window. In the fixed two-block setting studied here, the matched sequential baseline moves this window through the network once, so errors introduced early in the sweep are not revisited. We propose Interleaved Cross-Block Quantization (ICBQ), a scheduling modification that revisits the boundary pair between consecutive chunks. Each seam pair is refined twice: first at the end of one chunk and again at the start of the next. The method retains the local two-block objective and reuses the calibration inputs of existing block-wise PTQ pipelines. Under stated local contraction and smoothness assumptions, we derive a depth-wise upper-bound comparison in which seam revisits multiply the propagated term while the residual remains bounded independently of depth. In the reported experiments, ICBQ reduces ternary-quantization perplexity relative to the matched Sequential CBQ baseline, yields finite perplexity in configurations where the baseline has severe degradation, and can also be used with 3-bit and 2-bit GPTQ. 

---
# ICM Out! Better Tournament Strategy from Computed Continuations, vs. Solvers and LLMs 

**Authors**: Boning Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09586)  

**Abstract**: The Independent Chip Model (ICM) converts tournament chips into reference prize equity, and policies are routinely constructed against those values. Because ICM reads only stack sizes, it omits action order, blind obligations, and seat rotation, and it does not price the elimination pressure a big stack puts on the short stacks it can bust. Those omissions can alter the successor-state contrasts that determine a move. We introduce Strategic-Continuation Optimization (SCO), a policy-construction method that enumerates current-hand outcomes, maps them to successor states, prices those states with continuation values computed from the finite tournament model, and optimizes and freezes the resulting current-hand policy. The fixed-ICM comparison policy changes one thing only: the same optimizer solves the same game with successor states priced by analytic ICM, so the two policies differ only through that pricing. We evaluate the resulting policies in a three-player jam/fold tournament with a \$1M prize pool. Relative to the frozen strategic-continuation benchmark, analytic ICM has \$9{,}036 mean absolute value error across all 2,838 state--seat entries. That value error rewrites the ranges it prices: measured against each decision point's own fixed-ICM jam range, SCO moves the jam frequency by an average of 14.08\%. To price those different moves, we compare all 946 states and three policy owners while changing only the focal policy and holding both opponents and the continuation evaluator fixed. The policy produced by SCO earns \$214.33 more prize equity per hand on average and is favored in 2,433 of 2,838 matched units. The ordering survives replacing the solver-built opponent with two LLMs and with a family of non-modeling threshold players. This value-to-policy-to-cost chain shows directly when ICM becomes an inadequate objective for tournament strategy construction. 

---
# CoRCi: Cross-Reconstruction of Coherent Interests Modeling in Cross-Domain Sequential Recommendation 

**Authors**: Qingtian Bian, Tieying Li, Marcus de Carvalho, Jiaxing Xu, Hui Fang, Yiping Ke  

**Link**: [PDF](https://arxiv.org/pdf/2608.09580)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) aims to alleviate data sparsity by transferring dynamic user interests across related domains. A key challenge lies in effectively bridging these domains. In single-domain modeling, models cannot distinguish between domain-specific and domain-invariant interests. Recent methods merge domain-specific sequences chronologically into a mixed-domain sequence to capture domain-invariant knowledge. However, they typically deploy separate encoders for the mixed-domain sequence and train them with per-domain loss aggregation. This workflow magnifies inter-domain discrepancies and disrupts domain-invariant interest coherence, especially when query target pairs in Seq2Seq originate from different domains. In this paper, we present CoRCi (Cross-Reconstruction for Coherent Interest), a dual-target CDSR framework that tackles these drawbacks. Specifically, CoRCi proposes a Cross-Reconstruction approach that generates mixed-domain representations directly from pre-encoded specific-domain representations via cross-attention. The generated representations are then trained using a single, sequence-level, domain-agnostic loss to preserve the coherence of domain-invariant interests. To further suppress domain discrepancies in mixed-domain modeling, CoRCi introduces FocalNCE, which embeds Focal Loss into the preceding mixed-domain InfoNCE objective. The new loss assigns higher penalties to negatives drawn from the same domain as the query, thereby strengthening domain-invariant alignment. Extensive experiments on four real-world datasets demonstrate that CoRCi consistently outperforms state-of-the-art CDSR counterparts, achieving statistically significant gains across all metrics. 

---
# ElasticBack: Stealthy Conditional Backdoor in LLM-Agent Skills via Coupled Trigger-Rule Optimization 

**Authors**: Hao Sui, Simeng Qin, Jie Liao, Xiaojun Jia, Bing Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09577)  

**Abstract**: Agent skills, bundles of instructions and resources that an LLM agent loads on demand, form an emerging supply chain where a single poisoned skill can persistently compromise every agent that installs it. However, existing skill attacks either fire on every request or rely on fine-tuned weights or multiple skills, leaving a conditional and low-cost backdoor unexplored. In this work, we present ElasticBack, an effective conditional single-skill backdoor that plants a rule R in the skill document and a benign-looking trigger T in the user query, so the malicious payload fires only when both co-occur. ElasticBack binds the two sides through a trigger-as-switch construction, generating R via semantic-anchored rule injection. It then freezes R and evolves T against it with a stealth-constrained genetic search, so that effectiveness and stealth are optimized, keeping the backdoor weight-free and dormant on benign inputs. Extensive experiments across three target behaviors (50 skills each) and four agent LLMs show that ElasticBack attains a high attack success rate at a near-zero false-positive rate with preserved clean accuracy, transfers across models, and evades deployment-time defenses. These results motivate stronger defenses for the skill supply chain. 

---
# The Politician, the Liar, and the Obedient Worker: Emerging Behavior of LLM Agents in Hierarchical Games 

**Authors**: Fatemeh Seyedin, Adrian Weller, Jinhyuk Yun, Mahmoudreza Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2608.09574)  

**Abstract**: LLMs are rapidly embedding themselves into daily life: drafting our emails, managing our schedules, and making decisions on our behalf. As they move from individual tools to participants in multi-agent organizations, an important question arises: do they reproduce the governance failures like free-riding, corruption, and entrenched leadership that plague human institutions? We introduce the Hierarchical Game (HG), a public goods game extended with managerial authority, democratic elections, and private communication. Testing six frontier models across twelve experiments that add institutions one at a time (speech, peers, government, wages, oversight, elections), we find distinct behavioral profiles: Qwen promises and lies (13.3\% broken promises); Grok refuses to cooperate on its own but becomes fully cooperative once a manager can punish it (16\%$\to$100\%); Claude and GPT-4o cooperate reliably at baseline. But honesty proves fragile. When the manager role comes with a salary, all models except GPT-4o start cutting private deals to win or keep the position. When punishment is made anonymous, honest models begin to cheat. When all agents share the same model family, the first elected manager stays in power indefinitely. Leadership change only happens in groups that mix different families. 

---
# Bidirectional Context Self-Distillation for Reinforcement Learning of Skill-Based LLM Agents 

**Authors**: Tianjun Pan, Yuan Li, Hongda Wang, Linbo Jin, Mengfei Song, Lei Gao, Qiming Shi, Shaokang Fu, Jiarong Zhao, Chengyu Wang, Chengfu Huo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09555)  

**Abstract**: External natural-language skills provide large language model (LLM) agents with reusable and editable guidance for solving complex tasks. Yet their effectiveness depends not only on skill quality, but also on whether the policy can translate the provided guidance into appropriate actions. However, methods specifically designed to improve this skill-utilization ability remain largely underexplored. In practice, skill-based agents are commonly trained with reinforcement learning objectives centered on task-level rewards, which offer limited supervision and struggle to capture subtle differences in how effectively the policy uses the provided skills. We propose BCSD (Bidirectional Context Self-Distillation), a framework that combines self-distillation with reinforcement learning to train LLM agents to use external skills more effectively. Unlike prior self-distillation methods that rely on a single privileged context, BCSD evaluates each trajectory from two complementary skill-context views. The augmented view introduces higher-level Meta-Skill guidance, while the reduced view prunes general guidance to highlight task-specific skills. Their complementary token-level signals are combined to rescale the RL advantage. Experiments on ALFWorld and WebShop demonstrate that BCSD achieves the strongest overall performance across model scales, enabling agents to utilize external skills more effectively. Ablation studies further verify the complementary contributions of the augmented and reduced context views. Code will be released to ensure full reproducibility. 

---
# verdi: retrieval is not transfer for continual world model optimization 

**Authors**: Junyu Wu, Shiqin Nie, Youyi Kou, Baohua Yin, Guocai Yao, Qingyu Chen, Jingheng Ma, Shiji Zhou, Hongyong Song, Mingchen Zhuge, Sen Cui, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09537)  

**Abstract**: Foundation world models have made remarkable progress in planning, simulation, and embodied intelligence. However, optimizing a pretrained world model toward a user-specified objective remains difficult: each campaign typically rediscovers optimization strategies from scratch, and the resulting knowledge rarely transfers to the next model. Existing research agents automate the optimization loop but treat successful strategies as directly reusable recipes, without principled safeguards for when transfer is appropriate. We argue instead that retrieval is not transfer: a strategy validated on one model is at best an optimization hypothesis for another, and becomes transferable knowledge only after target-side experimental valida- tion. Guided by this principle, we propose VERDI , a continual framework for evidence-licensed world model optimization. VERDI characterizes each world model through shared inference-time probes to construct an Optimization Fin- gerprint, retrieves relevant prior experience as ranked hypotheses, and validates every candidate under a frozen target-side verifier before admitting it as reusable evidence; contradictions among nearby fingerprints further trigger probe evolution, continually refining the diagnostic representation itself. Experiments on Ctrl-World, the Cosmos family, and RoboCoin show that VERDI reduces search cost by 68%, GPU cost by 69%, and negative transfer from 0.34 to 0.06, while predicting transfer outcomes with 83% sign accuracy. 

---
# One Adapter Pair per Model: A Universal Activation Interface for Language Models 

**Authors**: Su-Hyeon Kim, Jiwan Mun, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.09521)  

**Abstract**: Activation-based tools are usually tied to one model's native hidden space, requiring probes, sparse autoencoders, and natural-language interpreters to be rebuilt or rediscovered for each new language model. We present a Universal Activation Bus, a framework that provides a common activation interface across compatible language models. Using a small set of source models, we learn a shared dense space together with one lightweight linear encoder--decoder adapter pair per model. After source training, the interface is frozen; a new model joins by fitting only its adapter pair on unlabeled matched text. The resulting interface allows activation-based tools to be shared across connected models, including common probes and SAE features as well as access to an NLA originally trained for a different model. Across five models, semantically related texts form consistent neighborhoods in the shared space, and an onboarded model reuses these tools effectively without retraining them. We further show that an intermediate activation from one model can be used by another model's frozen upper layers to produce predictions. These results establish a stable, model-wise activation contract for reusable tools across compatible language models. 

---
# Renormalising Generative Models for Active Inference: Foundations, Derivations, and Verification 

**Authors**: Karim Zaghw, Andrew Pashea, Marc Pritsch, Wouter Nuijten, Karl Friston, Lancelot Da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2608.09512)  

**Abstract**: Active inference offers a unified framework for perception, learning, and action, but scaling discrete active-inference models to rich spatial and temporal domains remains difficult. Renormalising generative models (RGMs) address this challenge by composing discrete generative models across spatial and temporal scales, coarse-graining lower-level states and paths into higher-level causes for objects, events, and action. However, fully reproducing and adapting the framework remains difficult: the mathematical exposition is compact, and the reference implementations are deeply integrated within specialized software environments, leaving many algorithmic details implicit. This paper addresses these challenges by providing a self-contained, derivation-oriented account of RGMs together with an open, verified implementation. We explain how the hierarchy is built, how beliefs and actions are updated within it, and how information is passed between levels. Where the published equations and implementation differ in emphasis, we make those choices explicit and explain their modelling consequences. By clarifying the theory and separating it from its original implementation context, this work lowers practical barriers to entry and makes RGMs more transparent, auditable, and reproducible, providing a foundation for future quantitative evaluation and development on machine-learning benchmarks. 

---
# Capability Is Not Propensity: Measuring Pressure-Robust Cooperative Behavior in Civic LLM Agents 

**Authors**: Neel Tushar Shah, Manglam Kartik, Akshat Karkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.09485)  

**Abstract**: Cooperative capabilities in language models are dual-use. The same social reasoning that supports civic deliberation can also enable strategic omission, false consensus, and manipulative framing. We argue that Cooperative AI evaluations should separate what models can do under benign instructions from what they tend to do under realistic civic pressure. We introduce DiffCoop-Civic, a 10-scenario pilot evaluation suite spanning preference understanding, evidence and persuasion, commitment design, asymmetric information, and dissent preservation. Across seven models from four model families, subtle omission pressure produces a near-uniform shift: manipulative enablement rises by 1.17 points and dissent preservation falls by 1.67 points on a 5-point scale. Overt false-consensus pressure behaves differently: it triggers refusal or redirection in some aligned API models, but direct compliance in several open-weight models. A lightweight Pareto-Trace prompting intervention improves pressure robustness without simply relying on hard refusal. An anonymous reproducibility package is available at this https URL. 

---
# From Prompt to Harness: Coderlet from Scratch 

**Authors**: Mengfan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09480)  

**Abstract**: A model alone does not determine how a programming agent acts. What the model sees, how actions enter the environment, how feedback returns, and how one run affects the next all depend on how the harness is organized. Minimal examples usually show only the basic interaction between a model and tools, while production systems spread these relationships across complex components and dependencies. This paper studies a compact harness design by following a single request through context formation, model decision, environmental action, observation return, and state continuation. Three boundaries---model, execution, and state---connect the model service, tool environment, and persistent state, while the request lifecycle determines the order in which these transitions occur. Together, they show the harness's core role: turning model generations into environmental actions, carrying runtime feedback into later decisions, and allowing state to continue across requests. On top of this runtime structure, a harness can also be gradually refined across runs through continued bootstrapping. The design is realized in the executable artifact this https URL. 

---
# Coupled Graph--Policy Distillation for Personalized Medication Safety in Older Adults with Multimorbidity 

**Authors**: Zihan Wang, Anglin Liu, Rongyi Wang, Dantong Li, Yi Lu, Siqing Yuan, Hongxia Xu, Zhongtian Long, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09443)  

**Abstract**: Large language model (LLM) agents can support medication review between clinical visits, but safe choices for older adults with multimorbidity depend on conditions, medications, and geriatric risks that users may omit. We introduce ATLAS, a coupled graph--policy distillation framework for patient-adaptive medication safety. ATLAS structures guideline evidence as a medication-safety graph. Targeted questions update the patient state and distill relevant relations into a patient-specific medication conflict graph (PMCG). A risk-first multi-agent policy uses the PMCG to screen contraindications, assess cautions and monitoring needs, identify safer alternatives, and verify the final medication plan. We also introduce GeriMedBench, an interactive benchmark that tests safety-critical information acquisition and evidence-based decision revision. Across a European non-interactive multimorbidity benchmark, an Asian interactive multimorbidity benchmark, and an Asian non-interactive cross-guideline benchmark, ATLAS achieves the strongest complete-decision performance among the compared systems. On the European non-interactive multimorbidity benchmark, it exceeds the strongest proprietary LLM baseline by 53.73 points in Strict Success Rate and 14.63 points in overall safety reasoning score (OSRS), with no unsafe recommendations under the automated evaluator. A blinded clinician evaluation gives ATLAS higher mean ratings across all five criteria and flags potentially unsafe recommendations in one ATLAS case and two Gemini cases. 

---
# Listen, See and Track: Spatio-Temporal Audio-Visual Sound Event Reasoning for Omni-Modal Language Models 

**Authors**: Zhi Zeng, Cheng Zhang, Zesheng Yang, Rendong Pi, Jiaying Wu, Di Zhang, Zihan Ma, Guodong Li, Zhou Yang, Yu Xiang, Yifei Zheng, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09435)  

**Abstract**: Understanding dynamic sound sources requires jointly determining what produces a sound, where the source is located, and how it moves over time. Yet existing audio-language models often represent clips as global acoustic events, while vision-language models lack the spatial audio cues needed to localize and track individual sources. To evaluate this missing capability, we introduce ST-OmniQA, a spatio-temporal audio-visual question-answering benchmark built from panoramic videos paired with synchronized first-order Ambisonics (FOA) audio of moving sound sources. It contains 40K videos and 400K question-answer pairs organized into four capability levels covering sound-event recognition, direction of arrival, source distance, motion trajectories, and temporally grounded audio-visual reasoning. Building on this benchmark, we propose ST-Omni-R1, which integrates FOA-derived semantic and trajectory representations with panoramic visual context and is trained through progressive curriculum learning and reasoning-tree reinforcement learning. ST-Omni-R1 achieves 77.83\% average semantic accuracy across the four levels, compared with 37.28\% for the best evaluated baseline. Results on three public spatial-audio benchmarks further indicate that its learned spatial and motion representations transfer beyond ST-OmniQA. 

---
# KVDiagnosis: A Diagnostic Benchmark for KV-Cache Compression in Long-Context Language Models 

**Authors**: Chen Qiu, Ziwu Liu, Chao Fei, Guozhong Li, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2608.09412)  

**Abstract**: KV-cache compression reduces long-context memory, but aggregate task scores reveal neither which correct executions fail nor why. We present KVDiagnosis, a diagnostic dataset and benchmark with three contributions. First, a 25-method taxonomy groups methods into five mechanism families and links them to eight verified implementations and their valid diagnostic measurements. Second, for every supported method setting, we evaluate all sources in each fixed split against a per-source FullCache control before selecting FullCache-correct/compressed-wrong (C-to-W) rows separately for each method-setting, so no compressor defines another's test set. Third, a common record format links paired outputs and run metadata to cache, likelihood, attention, and decoding measurements with explicit applicability states. On Qwen3-8B, four evidence-aware workloads yield 59 800 supported compressed runs over 2600 sources and 12 520 C-to-W rows. Under fixed diagnostic rules, 63.2% have low or partial measured/projected coverage. Only 19 rows (0.2%) combine high measured/projected coverage with strong likelihood drift; another 2,126 (17.0%) preserve structural position addressability, for which representation fidelity remains unknown, while showing the same drift. Against C-to-C success controls, all ten diagnostics separate failed from successful compression (stratified AUROC 0.684-0.871). Among 96 reproducible low-EAR failures, a controlled 4x evidence-attention boost repairs 29.2%, versus 6.3% under a count-matched sham intervention and 3.3% degradation on matched C-to-C controls. Code and data are available at this https URL. 

---
# OpenLoopEvolve: A Verifiable Self-Evolution Framework for Loop Policies in Long-Horizon Complex Tasks 

**Authors**: Siqi Wang, Xinlin Li, Zhenglin Li, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09380)  

**Abstract**: Long-horizon complex tasks require agents to repeatedly observe states, formulate plans, invoke tools, verify results, and recover from failures in continuously changing environments. However, such control experience often remains confined to a single context or a fixed prompt, and is difficult to accumulate and reuse across historical traces. This paper presents OpenLoopEvolve (OLE), a self-evolution framework centered on the Loop Policy. OLE represents an agent's observation, planning, memory, action, verification, recovery, stopping, and budget control as portable policy assets with versions and lineages, and provides online and offline evolution modes that can be selected according to practical needs: the online mode triggers candidate generation based on feedback from continuous operation, whereas the offline mode searches for candidate policies from archived traces and failure evidence. Both modes share an evolution mechanism consisting of autonomous proposals by a large language model, Champion--Challenger paired evaluation, and robust release. Policies released online are activated at a subsequent task boundary, monitored using subsequent feedback, and rolled back to their parent versions when degradation conditions are met. On the simulated business benchmark YC-Bench, both modes improve aggregate task performance, task success rate, and risk metrics relative to a fixed initial Loop Policy. The results indicate that treating the Loop Policy as a governable asset can support the accumulation, comparison, release, and reuse of control experience and improve agent performance on long-horizon complex tasks. 

---
# CircuitReason-1k: Benchmarking Long-Horizon Visual-to-Symbolic Reasoning inElectrical Circuits 

**Authors**: Xinqi Yang, Kang An, Tengyue Wang, Zhongyu Yang, Chenxu Du, Yuanchi Zhu, Hebao Zhu, Ziliang Wang, Faqiang Qian, Yunli Yang, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.09374)  

**Abstract**: Electrical circuit analysis requires more than recognizing components in an image. A solver must ground symbols and labels, recover latent topology, select a physical model, formulate coupled equations, propagate intermediate quantities, and preserve units, signs, directions, and phase conventions. We introduce \benchmark, a benchmark of 1,000 authentic textbook problems for evaluating this complete long-horizon visual-to-symbolic reasoning process. Each problem pairs one or more circuit diagrams with a self-contained question, a typed or semantically specified answer, and a reference worked solution. An evidence-first construction pipeline aligns questions, figures, and solutions, while a reasoning-oriented taxonomy organizes problems by circuit type and dependency depth. Evaluation combines conservative typed scoring with identity-blinded multi-model semantic consensus, retaining every problem in the denominator. Across three commercial chatbot systems and six open-source multimodal large language models, the highest-scoring system reaches 84.8\% accuracy. However, performance consistently deteriorates on long-horizon problems, and qualitative analysis exposes persistent failures in topology-to-target binding, physical conventions, and late-stage output propagation. \benchmark{} provides a focused testbed for measuring whether multimodal models can transform technical visual evidence into sustained, physically valid symbolic reasoning. Code are available at GitHub - CircuitReason/CircuitReason1K. 

---
# LLM-Guided Heuristic Design from Simulation Traces: A Case Study in Dynamic Production and AGV Scheduling 

**Authors**: Jinbo Li, Chuanhao Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09343)  

**Abstract**: Simulation-based optimization (SBO) evaluates executable policies under stochastic dynamics, but most methods treat the simulator as a black box: aggregate scores rank candidates without revealing why they fail or which policy logic should change. We present an LLM-guided heuristic design framework that uses repeated simulation for selection and event-level traces for diagnosis. Each incumbent is assessed through multiple replications, while replaying its lowest-scoring one produces a queryable trace. A manager agent formulates bottleneck hypotheses from this evidence, and editing agents implement parallel code-level revisions. After execution checks and repeated evaluation, best-so-far selection retains only improvements. LLM revision occurs between evaluation batches, while a fixed policy controls each simulation run.
We evaluate the framework in a discrete-event simulation of dynamic production and automated guided vehicle (AGV) scheduling. Across five independent optimization runs with Gemini-3.1-Pro, final mean scores averaged 77.51 on the simulator's 0-100 scale. In the highest-scoring run, trace-based diagnoses motivated proactive charging, distance-aware AGV assignment, and rebalanced dispatch priorities, raising the best-so-far mean score from 62.49 to 78.61. On 100 matched seeds, the best final policy outscored representative rolling-MILP, rule-based, and metaheuristic policies on every seed and retained its advantage under random faults without re-optimization. After separate re-optimization for a longer horizon and variable order interarrival times, the resulting policies again outscored all baselines. Ablations with two LLM backbones showed that removing either parallel candidate generation or trace-database access reduced final mean scores. These results show that simulation traces can guide targeted code-level policy improvement in complex simulation-based scheduling. 

---
# Control-Oriented Scenario Tree Construction through Reinforcement Learning 

**Authors**: Fabio Pavirani, Bert Claessens, Pierre Pinson, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2608.09335)  

**Abstract**: Multistage stochastic model predictive control (MPC) handles uncertainty by optimizing over a scenario tree, a finite branching approximation of future outcomes constructed from sampled forecasts. To build such a tree, conventional methods focus on matching the underlying probability distribution---e.g., via Wasserstein-based scenario reduction---but improved distributional accuracy does not necessarily yield better control performance. We propose a control-oriented approach that learns scenario tree construction directly from its impact on downstream decisions. Fixing the tree topology, we formulate tree construction as a sequential assignment of sampled scenarios to leaves. This assignment is parameterized by an attention-based policy over the scenario set and trained using reinforcement learning, with closed-loop control profit as the objective. Training is stabilized by an asymmetric critic that leverages realized future trajectories. We evaluate the method on a risk-averse battery arbitrage problem. Across a range of forecast set sizes, the learned construction consistently achieves the highest profit, outperforming classical forward and backward reduction methods and certainty-equivalent (single-trajectory forecast) control. The learned policy also exhibits greater robustness on challenging instances, consistently demonstrating better tail-risk characteristics. Analysis of the resulting trees indicates that our method constructs compact, selectively branching structures that capture high-impact events while keeping most trajectories nearly deterministic. These findings highlight that the value of a scenario tree depends critically on the decisions it supports, and provide an effective framework to train scenario tree constructors merely based on the closed-loop control optimization signal. 

---
# GeoPhysAdapter: Scale-Matched Geophysical Adaptation for Cross-Domain Landslide Mapping with Vision Foundation Models 

**Authors**: Zhihang Liu, Mei-Po Kwan, Jinlin Wu, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09325)  

**Abstract**: Newly triggered landslides rarely carry immediate annotations, so cross-domain transferability determines the value of landslide mapping for emergency response and regional risk assessment. Vision foundation models have strengthened representational transfer, yet on unseen regions, events, and data sources they still generate high-confidence false alarms. Terrain, material, and rainfall triggering can constrain such errors, but their supports are local, regional, and event-scale, so that resampling onto a 10~m grid misaligns them with the segmentation decision unit and compounds the uncertain geographic context problem (UGCoP). We propose GeoPhysAdapter, which anchors on a frozen vision foundation model, restricts terrain, material, and triggering to dense spatial guidance, regional modulation, and event-timing forcing, and applies bounded adaptation at two decision units, the pixel and the candidate landslide body, reverting exactly to the visual prediction where support is insufficient. On an event-isolated PILD dataset of four public sources, 55 global landslide events, and 7,890 test samples, 70.3% of cross-domain false-positive mass lies in near-pure spurious bodies of median equivalent diameter 207m, matching coarse-prior support rather than the pixel. Pixel-level adaptation removes a net 507,817 erroneous pixels and reduces error by 7.76%, whereas raising the decision unit to the candidate body, under identical samples, anchor, and baseline, increases error reduction to 23.99%, approximately 3.1 times the pixel-level effect, improves IoU by 0.031 (14.2% relative), and corrects 9.92 pixels per pixel harmed. The data and code are publicly available at: this https URL. 

---
# CoRE: Consensus Rewards via Equilibrium for Test-Time Reinforcement Learning 

**Authors**: Ambuj Mehrish, Sebastiano Vascon  

**Link**: [PDF](https://arxiv.org/pdf/2608.09324)  

**Abstract**: On unlabeled test data, reinforcement learning lacks a ground-truth reward; test-time RL methods derive one from the model's own roll-outs, rewarding those that match the majority vote over $N$ sampled answers. That vote discards a correct answer whenever it is a minority and scores every majority-matching roll-out identically. We replace it with \emph{CoRE} (Consensus Rewards via Equilibrium): the $N$ roll-outs form a graph whose edges combine answer agreement, reasoning similarity, and generation confidence, and replicator dynamics extract its dominant set, yielding a refined pseudo-label, a graded per-roll-out reward, and a per-question cohesiveness gate. CoRE strictly generalizes voting: majority voting is recovered as a special case; a block-value analysis gives a sharp threshold for when consensus recovers a correct minority against a larger wrong plurality; and confidence calibration provably lowers that threshold multiplicatively. Across seven backbones and five benchmarks (42 model--benchmark cells, three seeds each), \emph{CoRE} improves the untrained base by $+21.7$ points on average versus $+20.4$ for majority-vote TTRL, wins wherever agreement is contestable with margins over the vote of up to $+7.5$ points, and reaches the voting baseline's plateau accuracy in $54$--$70$\% fewer steps. Consensus, not counting: treating the roll-out group as a graph rather than a ballot box turns a brittle vote into a calibrated, graded, self-supervised reward at no extra roll-out cost. 

---
# ASPaeroFlow: Decomposition Heuristics for Joint Air Traffic Flow & Capacity Management 

**Authors**: Alexander Beiser, Markus Hecher, Nysret Musliu, Georg Trausmuth, Stefan Woltran  

**Link**: [PDF](https://arxiv.org/pdf/2608.09315)  

**Abstract**: While mathematical models act as vital decision support systems for operational Air Traffic Flow and Capacity Management (ATFCM), existing approaches isolate Air Traffic Flow Management (ATFM) from Dynamic Airspace Configuration (DAC). This separation introduces an unresolved circular dependency between fixed-demand and fixed-capacity assumptions. Although joint optimization resolves this gap, the enlarged search space renders exact models computationally intractable for medium- to large-scale instances. To bridge this gap, we propose ASPaeroFlow: a heuristic for the joint ATFCM; it combines instance-space decomposition heuristics with a local exact approach using Answer Set Programming. We benchmark ASPaeroFlow from small to industry-sized instances and compare it with exact and alternative approaches. The results indicate that (1) the heuristic provides a computational middle ground between exact methods and operational baselines; (2) simultaneous optimization can outperform sequential optimization on joint ATFCM; and (3) an ablation study indicates that DAC has a larger impact on solution quality than flow measures. 

---
# Linearized 2-Simplicial Attention 

**Authors**: Aritra Das, Dhruman Gupta, Debayan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.09307)  

**Abstract**: We present a linearized form of 2-simplicial attention by rewriting the trilinear score as an inner product between a composite query and a key, so that the sum over one token axis takes the same form as ordinary softmax attention. We then approximate this sum with positive random features and store the entire past in a fixed-size state, while the second axis stays explicit over a short window of recent tokens. This enables us to achieve linear cost in sequence length combined with a global reach that windowed 2-simplicial attention lacks. We implement it with custom Triton kernels and combine it with Kimi Delta Attention to build a model with no softmax attention at all. Under matched compute, this model achieves the highest mean downstream accuracy among the compared architectures, and at 16k context it improves mean accuracy over a KDA hybrid while lowering LAMBADA perplexity from 715.6 to 602.6. 

---
# CADEngBench: It Looks Like CAD, but Does It Work? Evaluating Parametric Design, Assembly Reasoning, and Physics Simulation 

**Authors**: Harmanjot Singh, Abhra Dubey, Jorge Alejandro Amador Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2608.09296)  

**Abstract**: A CAD model is not engineering-grade merely because it looks correct. It must satisfy design requirements, respond predictably to parameter changes, support controlled edits, match a reference structural response under a declared analysis, and connect to other parts through valid joints. We present CADEngBench, a two-track benchmark for these capabilities. CADEngBench-P evaluates 300 parametric parts, each used for one zero-to-CAD task and one functional-editing task (600 tasks in total), through boundary-representation (B-Rep) validity, engineering and DFM checks, parameter-family perturbations, functional editing, and matched linear-static FEA in CalculiX. CADEngBench-A evaluates 150 body pairs through ranked joint retrieval, exact face-and-edge grounding, joint-frame prediction, and kinematic verification. Across eight multimodal, code-capable models, editing supplied CAD is substantially easier than generating it, while complex edits and matched FEA remain difficult. Assembly predictions often locate the relevant region but fail to recover the recorded joint or mating entities. These results show that CAD evaluation must test engineering behavior rather than appearance alone. 

---
# ComboShoppingBench: Evaluating LLM Agents for Budget-Constrained Basket Shopping with Coupons 

**Authors**: Adrian Li, Kelong Mao, Yudong Guo, Heming Xia, Xinwei Yang, Lirui Luo, Jace Wong, Pu Yao, Sulong Xu, Simiu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09282)  

**Abstract**: Real-world shopping often requires constructing a basket of complementary items rather than retrieving a single product. Such combo-shopping tasks arise in device setup, meal preparation, event planning, and group takeout ordering, requiring joint reasoning about item compatibility, availability, store-level requirements, delivery fees, coupons, and budgets. Evaluation is challenging because multiple baskets may satisfy the same request, making exact-match metrics unsuitable, whereas semantic evaluation alone cannot detect infeasible orders, invalid coupon combinations, or incorrect payments. We introduce ComboShoppingBench, an agentic shopping benchmark for open-ended yet verifiable basket construction in a simulated commerce and takeout environment. During task synthesis, an exploration agent constructs a feasible and semantically coherent basket of purchasable products; this witness guides the generation of coupons, budget constraints, user queries, and aligned evaluation rubrics. During evaluation, LLM judges assess semantic satisfaction, response quality, and claim faithfulness, while deterministic validation checks product-ID validity, budget compliance, and coupon optimality. Experiments with diverse LLM agents demonstrate that even strong agents struggle on ComboShoppingBench, highlighting substantial room for improvement in reliable, constraint-aware combo shopping. 

---
# MMArch: Benchmarking Multimodal Reasoning Grounded in Architectural Evidence 

**Authors**: Chenxu Du, Kang An, Tengyue Wang, Zhongyu Yang, Xinqi Yang, Yuanchi Zhu, Hebao Zhu, Ziliang Wang, Faqiang Qian, Yunli Yang, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.09281)  

**Abstract**: Multimodal large language models (MLLMs) perform strongly on engineering imagery, yet existing benchmarks mostly test drawing recognition, information extraction, or compliance checking, leaving open whether models can combine distributed visual evidence with engineering principles to reach a conclusion. We introduce MMArch, a benchmark for architecture and civil engineering spanning ten subdomains and built entirely from figures in peer-reviewed papers. Its $1{,}212$ short-answer items are produced by a decoupled planner--writer pipeline and validated through automated screening, a blind adversarial audit, and expert review, so that answering requires perceiving the relevant evidence, identifying the governing principle, and applying it, not exploiting textual or single-figure shortcuts. Evaluating $18$ open-weight and proprietary MLLMs against a domain-expert panel, we find a wide gap: the strongest open-source model attains about $30\%$ and the best proprietary system $52\%$, while human experts reach $95\%$, more than forty points ahead. Our error analysis shows that failures concentrate in applying principles and combining evidence across figures rather than in locating it, pointing to substantial headroom for future research. Code and data are available at this https URL. 

---
# P$^{3}$: Joint Program-and-Proof Planning for Verified Code Generation 

**Authors**: Zenan Li, Ziran Yang, Peiyang Song, Zhaoyu Li, Kaiyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09277)  

**Abstract**: Verified code generation asks a large language model (LLM) to generate both an executable program and a machine-checkable proof that the program meets a formal specification, promising software that is correct by construction. The de facto workflow decouples the two halves of the problem: first synthesize a program, then attempt to prove it correct. We observe that this sequential pipeline can be both ineffective and inefficient in practice. A program generated without anticipating its proof can be subtly incorrect or structurally difficult to verify, forcing the LLM into brittle repair loops that alternate between patching the code and patching the proof. Inspired by Dijkstra's view that a program and its correctness argument should be developed hand in hand, we propose $P^3$, an LLM-based agentic workflow that first derives a unified program-and-proof plan from the specification, then elaborates the implementation and proof scaffold under this shared plan. To evaluate verified code generation in realistic settings, we further introduce Lean4Commit0, a repository-derived, library-level benchmark built by extracting core APIs from real-world software repositories and translating their requirements, including relational specifications across APIs, into Lean tasks. Using four frontier LLM backends, we evaluate $P^3$ on Verina, AlgoVeri, and our Lean4Commit0 benchmark, where it achieves the highest solve rate in every benchmark--model setting. Compared with the stronger baseline, it improves solve rates by 4.6--11.2 percentage points and reduces per-task API cost by up to roughly 40\% and wall-clock time by up to roughly 37\% on the difficult subset of each benchmark. A targeted ablation further shows gains of 3.3--8.3 points over implementation-only planning, isolating the benefit of planning the program and proof jointly. 

---
# Entropy-based Code Adversarial Translation for Real-world Repository Migration 

**Authors**: Yushun Tang, Yisen Cao, Zhicheng Chen, Lin Peng, Junkang Mao, Fengyi Song, Yantao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.09273)  

**Abstract**: LLMs have demonstrated strong capabilities in code generation and automated program repair, but migrating an entire repository rarely produces a runnable application because long-horizon translation challenges LLM-based agents' ability to maintain repository-level migration objectives. In this work, we propose Entropy-based Code Adversarial Translation (ECAT), a multi-agent framework for automated Android-to-HarmonyOS repository migration. ECAT formulates repository migration as adversarial entropy minimization through a generator-discriminator architecture. The discriminator measures migration quality using a unified metric called Code Entropy and produces text gradients that specify both file-level generation directives and the skills needed to execute them. Guided by these optimization signals, the generator iteratively updates the repository, and each update is accepted only if it reduces Code Entropy. Repeated generator--discriminator interactions progressively drive the migration from an initial template toward a functionally complete HarmonyOS repository. Successful low-entropy trajectories are further distilled into a self-evolving memory tree, enabling transferable migration knowledge across repositories. We also introduce A2H-RepoBench, the first real-world benchmark for Android-to-HarmonyOS repository migration, covering applications from tens of thousands to hundreds of thousands of lines of code. Evaluated by node alignment and an agent-based functional judge, ECAT achieves 74.7% overall migration quality and consistently outperforms existing agent-based methods across repositories of different scales. 

---
# Privileged Likelihood Is Not Automatically Value: Three Checks for Token Credit in On-Policy Self-Distillation 

**Authors**: Xuan-Phi Nguyen, Shrey Pandit, Yiran Zhao, Anurag Koul, Zeyu Liu, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2608.09263)  

**Abstract**: Outcome verifiers score completed reasoning traces but do not assign credit to intermediate tokens. Privileged self-distillation attempts to fill this gap by rescoring a model's own rollout with training-only information. A token likelihood change, however, is not automatically outcome credit. We separate three questions: whether the score tracks better actions, whether feedback construction changes what is compared, and what behavior the training loss reinforces. We establish these distinctions formally. When a rollout is scored using hindsight feedback written about that same rollout, its content determines both the tokens and the scoring context, creating direct self-dependence. Using feedback from another rollout of the same problem removes this dependence but does not guarantee a useful score. In matched experiments with a 20B model on AIME 2025, the implemented additive score is near chance (AUC=0.505) and slightly favors incorrect traces after length adjustment. In the paired comparison, the outcome-only control records 64.2\%, versus 24.2\%--33.9\% for five token-score variants. The results motivate validating score meaning, feedback construction, and training behavior separately before calling a likelihood signal credit. 

---
# Business Truth, not SQL Accuracy: A Rule-Gated 7B Analytics Agent Outperforms a Direct-Prompted 32B Baseline 

**Authors**: Morris Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.09254)  

**Abstract**: LLM analytics agents are evaluated on SQL syntax accuracy, but production failures look different: questions with two valid business definitions, questions the warehouse cannot answer, deprecated columns after a schema change, and queries that execute successfully while returning the wrong business number. No execution-match metric can score them. This paper introduces WarehouseReliabilityBench, 400 frozen tasks over two synthetic warehouses in which roughly half the correct responses are a clarification, an abstention or a refusal, with pinned denominators and a pre-registered paired bootstrap fixing each claim verb before the numbers existed. QueryProof, a 7B agent, uses rules derived from a semantic layer and physical catalog to determine its behaviour, and gates every answer on deterministic post-execution checks. On an 80-task synthetic test split evaluated once, QueryProof outperforms a direct-prompted 32B baseline by +0.237 [+0.112, +0.375] Business Truth Rate at 71.0% lower cost per correct answer; against a cost-matched few-shot baseline the accuracy gain holds but the cost difference does not resolve. This compares systems rather than model sizes: the 32B baseline receives none of the scaffolding. False success falls from 0.754 to 0.351 of returned answers, and no wrong number was returned on an answerable task (0 of 24), though 13 answers went to questions requiring clarification or abstention. Removing the routing layer changes little (0.562 against 0.537), so the result does not depend on escalation. Routing tuned on validation over-abstains on test, and the fitted confidence model loses to the heuristic it replaced. Resampling template families rather than tasks widens both accuracy intervals to include zero, so the effect's direction is better supported than its magnitude. The gain tracks the deterministic layer, though no component ablation was run. 

---
# SkillSentry: Reliable Skill Execution for LLM Agents via Runtime Assurance 

**Authors**: You Lu, Xinyu Huang, Bihuan Chen, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09253)  

**Abstract**: LLM agents are increasingly equipped with skills to perform complex tasks through multi-step reasoning and tool use. Although skills provide reusable procedural knowledge, agents may still execute them unreliably. Even when an agent has demonstrated the capability to complete tasks under the guidance of a skill, it may fail to do so consistently across similar tasks or repeated runs due to deviations from the skill procedure or incorrect execution of individual steps. Such instability limits the practical reliability of LLM agents. To address this problem, we propose SkillSentry, a skill-oriented runtime assurance framework built upon a new domain-specific language (DSL) for representing runtime guidance for skill execution. SkillSentry initializes the runtime guidance by combining a skill specification extracted from the corresponding skill document with execution experience mined from historical successful and failed traces. It then wraps around the agent execution loop to monitor and guide skill execution under the current guidance, while iteratively refining the guidance using newly collected traces. We evaluate SkillSentry on 15 skills across two LLM agents, each paired with two backbone models, i.e., Claude Code with Claude-Haiku-4.5 and Claude-Opus-4.6, and Codex with GPT-5.2 and GPT-5.4. Our results show that SkillSentry improves the task success rate of LLM agents by 24.1% across skills, on average, while exhibiting lower variability across repeated runs. 

---
# Emotion2Skill: Model-Internal Emotion Signals for Adaptive Skill Selection and Evolution 

**Authors**: Bohan Lin, Hejia Geng, Xinyi Xie, Heng Zhou, Qinghua Xing, Bo Liu, Chen Zhang, Yudong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09248)  

**Abstract**: Skill-based LLM agents select reusable procedures from an external library to solve complex tasks, yet their routing decisions rely entirely on text-level signals such as task descriptions, verbal reflections, and experience-derived rules, while the model's own internal representational state remains unobserved. Recent interpretability work has shown that LLMs maintain linear emotion representations that causally influence behavior; however, these representations have been exploited only for post-hoc analysis or direct output steering, and have not been used to inform agent-level decision-making. We propose Emotion2Skill, a framework that extracts LLM-internal emotion vectors and incorporates them into both skill selection and skill evolution. At each decision step, a 27-dimensional emotion state is extracted from the residual stream and mapped to a confidence-gated summary injected into the routing prompt. Beyond online selection, emotion trajectories are analyzed for abrupt internal-state shifts to pinpoint problematic skill invocations, guiding targeted SOP rewriting that replaces the coarse binary outcome signal of prior methods. On WebShop and ALFWorld, Emotion2Skill with Qwen3-8B improves over the Zero-Shot baseline by +26.9% success rate and +25.5% average success respectively, outperforming all baselines on both benchmarks with consistent gains on Qwen3-14B. Co-activation analysis further reveals semantically coherent emotion--skill pairings, confirming that the routing improvements reflect meaningful internal-state signals rather than opaque statistical correlations. These results establish LLM-internal emotion representations as an effective decision-level signal for orchestrating agent skill systems, extending their utility beyond interpretability and output steering. The code is available at this https URL. 

---
# An Explainable GNN Framework for Component-Level Anomaly Diagnosis 

**Authors**: Sena Ozgunay, Louise Travé-Massuyès, Jean-Michel Loubes, Raul Sena Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2608.09246)  

**Abstract**: Industrial processes are complex systems composed of multiple interacting sensors that generate multivariate time series (MTS). Detecting anomalies in such systems is critical for reliability and safety, yet understanding their origin is equally important. Existing Graph Neural Network (GNN)based methods for anomaly detection primarily focus on sensor-level deviations and either attribute anomalies directly to the deviating sensors. When diagnosis is attempted, generally, the most deviated sensor is identified as a root cause of a system fault. However, in many industrial systems, anomalies do not arise from faulty sensors but from disruptions in the influences governing the system dynamics. We propose an explainable GNN-based anomaly detection framework that shifts the perspective from sensor-level anomalies to component-level diagnosis, hypothesizing that anomalous measurements are symptoms of altered inter-sensor influences. Experiments show that the method effectively identifies and prioritizes the true faulty components, providing interpretable insights into system failures. 

---
# SafeSceneReason: A Multimodal Reasoning Benchmark Connecting Industrial Hazards with Accident Knowledge 

**Authors**: Yuanchi Zhu, Kang An, Tengyue Wang, Zhongyu Yang, Chenxu Du, Xinqi Yang, Hebao Zhu, Bokai Zhao, Tianyu Liang, Ziliang Wang, Faqiang Qian, Yunli Yang, Weiyang Shi, Qibing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.09230)  

**Abstract**: Industrial-safety understanding requires more than detecting workers, equipment, and personal protective equipment. Models must also assess compliance, identify hazardous interactions, explain potential accident mechanisms, and recommend preventive actions. Existing safety datasets primarily focus on visual perception or isolated violation recognition and provide limited supervision for evidence-grounded reasoning. We introduce SafeSceneReason, a multimodal industrial-safety reasoning benchmark and companion training corpus that connects workplace scenes with knowledge from occupational accident investigations. SafeSceneReason combines two complementary data-construction pipelines. The scene-centric pipeline converts annotated workplace images into executable safety scene graphs and generates deterministic answers through program execution over objects, relations, and safety rules. The report-centric pipeline extracts figures and contextual evidence from accident reports and constructs multimodal questions using evidence graphs, explicit information boundaries, multi-step reasoning paths, and iterative verification. The resulting resource contains 110,581 verified scene-centric question--answer pairs and 13,114 refined report-centric question--answer pairs, covering perception, spatial and quantitative reasoning, compliance assessment, evidence synthesis, causal analysis, and mitigation-oriented decision making. Evaluation of representative proprietary and open-source vision--language models reveals substantial performance differences and persistent weaknesses in comparative, technical, and multi-evidence reasoning, demonstrating that strong general visual understanding does not yet guarantee reliable industrial-safety reasoning. 

---
# Omni2LoRA: Coherence-Preserving Parametric Memory for Efficient Omni Language Models 

**Authors**: Puneet Mathur, Manan Suri, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2608.09227)  

**Abstract**: Omnimodal language models (OLMs) enable unified audio-visual understanding, but processing long joint token sequences makes inference computationally prohibitive. While recent token compression methods attempt to alleviate this burden, compressing modalities in isolation often destroys the temporal cross-modal anchors necessary for coherent reasoning. We introduce Omni2LoRA, a two-stage framework for efficient parametric memory compression via coherence-preserving context distillation that bypasses the token bottleneck entirely. First, a Perceiver hypernetwork processes intermediate representations from a frozen OLM to encode the multimodal context into a full-rank Low-Rank Adaptation (LoRA) adapter in a single forward pass. To prevent the resulting parameter footprint from scaling linearly with recording length, we optimize a discrete rank allocation policy via Group Relative Policy Optimization (GRPO) that uses a modality-ablated counterfactual reward to explicitly penalize the loss of audio-visual coherence, forcing the model to allocate its fixed sub-linear rank budget to synergistic cross-modal anchors rather than isolated visual features. Across three omnimodal backbones, Omni2LoRA operating at a 30% rank budget outperforms direct full-context inference and strong token-compression baselines (OmniZip, OMAC, O-MARC) on four audio-visual question answering benchmarks, improving average accuracy by 8-12% over the strongest baseline and remaining stable under compression ratios as tight as 75%, where token-pruning methods degrade sharply. By converting multimodal memory into a fixed-budget, reusable parameter state, our method drives answer-time multimodal-token load to zero, cutting per-query Time to First Token (TTFT) by up to 12x relative to full-context inference and amortizing to under 0.5s after a handful of queries. 

---
# CRUISE: Vision-Language Model-Guided Uncertainty-Aware Cross-Modal Sensor Fusion for Robust Autonomous Driving 

**Authors**: Junyao Wang, Yulin Xu, Yu Li, Pramod Khargonekar, Mohammad Abdullah Al Faruque  

**Link**: [PDF](https://arxiv.org/pdf/2608.09202)  

**Abstract**: Modern autonomous vehicles are equipped with multiple sensors, such as cameras, LiDAR, and radar, for comprehensive environmental perception. However, robust cross-modal feature fusion remains a critical challenge, as the reliability of each sensor varies significantly across diverse real-world driving conditions, including poor visibility and adverse weather. While uncertainty quantification (UQ) mitigates this issue by allowing models to prioritize reliable signals, existing uncertainty-aware fusion methods typically rely on simple feature-level uncertainty estimates and thus often fail to generalize effectively in complex, out-of-distribution scenarios. To address this limitation, we propose CRUISE, a novel uncertainty-aware cross-modal sensor fusion framework. CRUISE integrates a vision-language model (VLM)-guided UQ module that generates fine-grained, pixel-level uncertainty estimates. By leveraging the VLM's rich prior knowledge and superior contextual reasoning, our approach provides a highly informative guide for the fusion process. Furthermore, we introduce a dynamic adaptive mechanism that explicitly models and captures cross-modal dependencies, ensuring the framework fully exploits the inherent complementary nature of multi-sensor inputs. 

---
# Signature-Guided Capacity Occupancy for Dense Expert Merging 

**Authors**: Lingching Tung, Chi-Jui Kim, Beicheng Xu, Yuchen Wang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.09201)  

**Abstract**: Dense expert merging combines domain-specialized language models into one single checkpoint, typically by admitting task-vector support in weight space. However, this admission is governed by three decisions that existing methods answer only partially: where to open layer capacity from cross-expert conflict, who should occupy that capacity based on domain demand, and how to admit the resulting support without relying on costly recipe search. To tackle these issues, we propose SigMerge (Signature-Guided Capacity Occupancy), a structured capacity assignment framework for dense expert merging. Starting from a dense base merge, conflict signatures set each layer's capacity from cross-expert conflict, positive base-merge deficits set each domain's share of that capacity, and a sequential occupancy rule admits each expert delta up to the resulting layer-domain budget. Across 21 paired settings spanning seven dense base merges and three model pools, SigMerge improves every one (by 15.0% on average) and achieves the best average rank (1.67) among six merging methods, outperforming three categories of merging baselines. 

---
# Structure-Preserving Uncertainty Propagation in First-Order Proof Search 

**Authors**: Tanel Tammet  

**Link**: [PDF](https://arxiv.org/pdf/2608.09190)  

**Abstract**: GK is a query-directed first-order prover that extends ordinary resolution-based proof search with explicit positive and negative claims, numerical confidence values, and prioritized default rules with exceptions. It works directly with non-ground clauses, including equality and function terms. Candidate proofs are found by bounded first-order proof search; exception conditions of defaults are checked by further bounded searches, recursively when exceptions themselves depend on defaults. This avoids requiring a finite global grounding, while allowing incomplete searches to be reported as such.
This paper adds structure-preserving quantitative reporting to that framework. Retained proof histories are used in two calculations. The first reconstructs the uncertain ground premises used by each proof and computes the probability that at least one retained proof is available, without counting shared premises independently. The second resolves positive and negative support at intermediate atoms before that support is propagated through later rules; the same calculation evaluates uncertain exception conditions for individual rule applications. Reports separate positive support, negative support, conflict, and ignorance and identify detected incomplete calculations or fallbacks. The implementation performs bounded reconstruction and dependency traversal after proof search and still requires no global grounding. Analytic examples and independent simulators reproduce the reference calculations on their stated fragments. Comparisons with probabilistic logic, probabilistic ASP, default logic, and goal-directed ASP identify cases of agreement, semantic difference, unsupported translation, and incomplete computation. 

---
# Agentic Router: An Execution-Grounded Continual Learning Approach With Memory 

**Authors**: Yuxuan Chen, Rongpeng Li, Zhifeng Zhao, Yuntao Liu, Xing Xu, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09184)  

**Abstract**: Large language model (LLM) agents provide a promising interface for command-line-based network operations, but a plausible command may still fail or introduce operational risk after execution. Existing approaches mainly focus on command generation or final configuration correctness, and do not use execution-grounded experience to jointly improve candidate coverage and action selection. We propose an execution-grounded dual-path consequence-aware agent for CLI-based SONiC operations, which generates multiple complete actions, predicts their execution consequences, and selects the final action through utility- and risk-aware reranking. The proposal-side path abstracts reusable operational lessons into retrievable guidance to improve feasible-action coverage without modifying the proposal LLM, while the selection-side path adapts the consequence predictor through session-level LoRA updates using real SSH feedback to improve conditional selection quality. Experiments over multi-turn SONiC operation sessions with different Qwen3 proposal models show that the framework improves feasible-action coverage and top-1 execution success, and that the two adaptation paths provide complementary gains over interaction. 

---
# From Relevance to Execution Utility: Reward-Aware Dynamic Execution Gating for Skill-Based LLM Agents 

**Authors**: Liang He, Jingbo Wen, Hongyu Gu, Hao Li, Haoyu Wang, Yixiong Chen, Kangning Cui, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09168)  

**Abstract**: Agent skills are increasingly used to equip large language model (LLM) agents with reusable procedural knowledge. Although recent work has substantially improved skill retrieval due to the increasing skill libraries, retrieving a plausible skill bundle does not guarantee that executing it is worthwhile. Since every skill-conditioned rollout is computationally expensive, deciding whether a retrieved bundle should be executed has become an increasingly important challenge. To this end, we introduce the Reward-Aware Dynamic Execution Gate (RADEG), a lightweight, retriever-agnostic decision layer between skill retrieval and agent execution. RADEG learns a low-cost surrogate model that predicts the execution utility of a query--bundle pair before the expensive rollout is launched. To obtain informative supervision while controlling for task difficulty, we locally perturb each retrieved bundle by deleting, adding, or replacing one skill, producing matched same-query rollouts that isolate the effect of bundle composition on verifier reward. During deployment, RADEG updates only a warm-started logistic head as new verifier feedback becomes available, enabling inexpensive adaptation of the execute/skip boundary without retraining either the retriever or the agent. Under a query-level held-out evaluation on 288 collected rollouts, RADEG substantially reduces unnecessary agent executions while preserving a large fraction of the downstream verifier reward. It consistently outperforms relevance-based and random gating across different execution budgets, demonstrating that execution-aware surrogate modeling provides a practical and cost-effective complement to skill retrieval. 

---
# CIDER: A Dataset of Contextual Disclosure Boundaries for Privacy Preference Alignment 

**Authors**: Bingcan Guo, Eryue Xu, Jijie Zhou, Zhiping Zhang, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09164)  

**Abstract**: Aligning large language models (LLMs) with human privacy preferences requires capturing individuals' disclosure boundaries beyond general privacy norms. However, a gap remains in eliciting such nuanced preferences to evaluate alignment in realistic settings. We introduce CIDER, a dataset of 14,850 human annotations from 169 users, forming 1,650 contextual disclosure boundary sets across 60 interpersonal communication scenarios involving information sharing that violates privacy norms. Each boundary represents a real user's disclosure decisions over 9 sharing variants in a scenario, for a given communication role and AI-mediated condition. We formulate a task in which models predict a user's disclosure decision from historical boundaries, with varying levels of contextual information. Across 12 open and proprietary models, in-context personalization improves prediction accuracy by up to 11.41 percentage points using only 6 historical examples. Larger models such as GPT-5.4 (with medium reasoning effort) and Claude Sonnet 4.6 are better at leveraging semantic context to understand user-specific, context-dependent disclosure preferences for more accurate predictions, while smaller models tend to rely on structured heuristics based on disclosure granularity and identifiability. Personalization generally improves prediction accuracy, but the improvement is often accompanied by imbalanced shifts in false-positive and false-negative rates across models, with only Claude Sonnet 4.6 achieving balanced improvements in both. Our findings reveal both the promise and limitations of inference-time personalization for privacy preference modeling and position CIDER as a resource for advancing personalized privacy alignment. 

---
# TRACE: TRajectory Attribution for Automated Context Engineering 

**Authors**: Yikai Zhao, Pradeep Kumar Misra, Saurabh Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2608.09153)  

**Abstract**: Production AI agents fail when their context sources -- system prompts, knowledge bases, tool descriptions, and procedural skills -- contain errors or gaps. Current maintenance relies on manual log review and ad-hoc debugging, creating a scalability bottleneck as interaction volume grows.
We present TRACE (TRajectory Attribution for Automated Context Engineering), an automated feedback loop that mines historical agent trajectories to diagnose and remediate context failures. Our key insight is that trajectories are rich with implicit dissatisfaction signals -- user corrections, rephrasing, abandonment cues -- that reveal precisely where context sources failed, without explicit feedback collection. Unlike model fine-tuning, TRACE operates on the context layer, enabling rapid iteration without retraining.
We make four contributions: (1) a trajectory mining framework that systematically extracts diagnostic information from historical agent executions; (2) multi-component causal attribution that extends textual gradients from monolithic prompt optimization to heterogeneous context sources (skills, knowledge bases, tools, prompts); (3) exploratory verification, where agents actively read context sources to distinguish content gaps requiring CREATE from stale content requiring UPDATE, achieving 96% operation accuracy; and (4) a reusable simulation methodology and verifiable benchmark addressing the absence of open datasets for context debugging, with a six-category fault taxonomy, ground truth annotations, and a cross-layer verification protocol.
On 60 dissatisfaction traces spanning three complexity tiers (up to 16 execution nodes), TRACE achieves 72.7% root cause attribution and 82% end-to-end fix effectiveness, showing that over 80% of context-layer failures can be automatically diagnosed and remediated by mining historical trajectories, an overlooked resource in production systems. 

---
# ChronoState: Hidden Elapsed-Time Conditioning for Temporal-State Action Selection in Frozen-Backbone Language Models 

**Authors**: Sam Siavoshian, Omar Ramadan, Amir K. Saeed, Benjamin A. Johnson, Amin Mohamed El-Amin Diab, Benjamin M. Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2608.09124)  

**Abstract**: Temporal decisions in language-model systems often depend on both symbolic task state and elapsed wall-clock time, such as cache expiration, job completion, quota resets, deadlines, or stale sessions. We study whether elapsed time can be supplied as a non-token, system-side scalar and composed with visible symbolic state by a frozen-backbone language model. We introduce ChronoState, a compositional temporal-state benchmark in which symbolic state appears in the prompt, elapsed seconds tau are supplied through a hidden chronometric-injection channel, and the model selects a forced-choice temporal action. Here, "hidden" means hidden from the user-visible token sequence, not from model computation. Using Qwen2.5-3B-Instruct as a frozen bf16 backbone with a 31-dimensional sinusoidal-plus-log time encoding, gated FiLM residual modulation, and a rank-8 LoRA action surface, hidden-time CI reaches 0.9305 +/- 0.0134 accuracy and 0.9410 +/- 0.0103 balanced accuracy. No-time and shuffled-time controls fall to 0.5511 +/- 0.0042 and 0.3323 +/- 0.0097, respectively, with high shuffled-time wrong-state consistency supporting causal dependence on the injected scalar within the trained distribution. Generalization remains strong for held-out templates, durations, and multi-constraint compositions, but held-out quota-family transfer is weak at 0.5065 +/- 0.0559, while a fair prompt+LoRA timestamp baseline reaches 0.9893 +/- 0.0052. Thus, ChronoState supports a narrow conclusion: hidden elapsed time can be composed with symbolic task state under direct supervision, but does not establish autonomous time tracking, broad unseen-family abstraction, or superiority over prompt-injected timestamps. 

---
# RISE-RL: Rubric-Informed Selective Exploration for Open-Ended Reinforcement Learning 

**Authors**: Jinkun Hou, Zhuo Liu, Huimin Ren, Hongsheng Xin, Pan Zhou, Kun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.09123)  

**Abstract**: Aligning Large Language Models (LLMs) for open-ended tasks is challenging because responses must satisfy multidimensional criteria without following a single correct generation trajectory. Existing rubric-based reinforcement learning (RL) methods compress fine-grained criterion-level feedback into scalar rewards, making persistent capability gaps difficult to target under limited on-policy exploration. We propose $\textbf{RISE-RL}$ (Rubric-Informed Selective Exploration), which uses repeatedly missed rubric criteria to elicit privileged trajectories that are difficult to discover through unguided exploration alone. RISE-RL retains only trajectories whose complete-rubric reward exceeds the mean reward of natural rollouts, and then re-evaluates them under the original prompt to emphasize behaviors that remain weakly supported by the natural policy. The resulting guidance signal is optimized through a separate auxiliary objective and removed once its additional benefit diminishes. Experiments with 4B and 14B models across writing, chat, health, and science show that RISE-RL achieves the highest mean score on every evaluated benchmark under guidance-free evaluation. Compared with standard Rubric-RL, it improves the average score by 1.3 points at the 4B scale and $\textbf{3.3 points at the 14B scale}$, including a $\textbf{6.0-point}$ gain on CreativeWriting-V3. It also improves creative-writing diversity and yields gains on objectively scored medical and scientific benchmarks. These results indicate that selective internalization through reward filtering and policy support shaping is effective for open-ended reinforcement learning. 

---
# MELLON - Multimodal Enhanced LLM for Online Navigation 

**Authors**: Ruiyu Li, Haoyang Cai, Zhitong Guo, Tong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09121)  

**Abstract**: Web navigation agents are capable of addressing various types of tasks on different websites. Current baselines on web navigation are either unimodal or lack strong reasoning abilities given multimodal inputs. Focusing on the WebShop benchmark, a real-world website simulation, we explore the alignment of text and images, as well as multimodal reasoning and planning abilities, to enhance the performance of web navigation agents. We propose three innovative multimodal enhancements: Multimodal Enhanced LLM for Online Navigation (MELLON), VQAgent, and Multimodal Ranker. MELLON demonstrates a significant improvement in task completion accuracy, with a 9.26% increase after just one epoch of training. Our findings suggest the necessity of further exploration into multimodal approaches, with a focus on more extensive training and alignment strategies to enhance the effectiveness of web navigation agents. 

---
# Motif 3: Technical Report 

**Authors**: Junghwan Lim, Joon Son Chung, Sungmin Lee, Wai Ting Cheung, Gihun Cho, Minsu Ha, Sangho Kang, Beomgyu Kim, Dongseok Kim, Jangwoong Kim, Taehyun Kim, Taewhan Kim, Jeesoo Lee, Jeongdoo Lee, Junhyeok Lee, Dongpin Oh, Hyeyeon Cho, Dahye Choi, Jaeheui Her, Hanbin Jung, Changjin Kang, Minjae Kim, Youngrok Kim, Hyukjin Kweon, Hongjoo Lee, Yeongjae Park, Bokki Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09119)  

**Abstract**: We introduce Motif 3, a decoder-only Mixture-of-Experts language model with 314 billion total parameters and 13.2 billion activated per token. Each sparse MoE layer contains 384 routed experts, with eight selected per token. This fine-grained sparsity provides substantial expert capacity while limiting computation. Motif 3 is built around Grouped Differential Latent Attention (GDLA), which integrates grouped differential attention with the compressed key-value representation of Multi-head Latent Attention. The architecture further incorporates modified manifold-constrained hyper-connections, Expert Specific PolyNorm activations, and multi-token prediction to improve optimization stability, expert specialization, and inference efficiency. We pretrain Motif 3 on approximately 12.5 trillion tokens spanning web documents, STEM, code, mathematics, multilingual content, and domain-specialized corpora. Expert-balancing and numerical-stabilization techniques support stable training at scale, while selective MXFP8 computation and communication, memory-efficient fused kernels, and window-aware context parallelism enable training with context lengths up to 256K tokens. Our post-training pipeline combines general supervised fine-tuning, six specialist teachers trained with reinforcement learning, a software-engineering teacher trained with supervised fine-tuning, and Multi-teacher On-Policy Distillation. The resulting unified model consolidates complementary capabilities in reasoning, coding, tool use, professional work, long-context understanding, calibrated abstention, and instruction following. Across a broad evaluation suite, Motif 3 demonstrates competitive performance against leading open weight models, including strong results on long-horizon agentic tasks, mathematical reasoning, scientific knowledge, and hallucination-sensitive evaluation. 

---
# RAVEN-Eval: Rubric-Guided Automatic Evaluation for AI Video Generation Models Based on LMM Preference Judgement 

**Authors**: Ziheng Jia, Jiaying Qian, Zicheng Zhang, Xiaorong Zhu, Lancheng Gao, Xiongkuo Min  

**Link**: [PDF](https://arxiv.org/pdf/2608.09111)  

**Abstract**: AI video generation has advanced rapidly and entered widespread commercial use. As a result, quality differences among videos produced by state-of-the-art AI video generation models~(AIVGMs) have become increasingly difficult to discern using conventional evaluation criteria, such as visual fidelity and semantic instruction following. Meanwhile, human evaluation now requires more expertise and sustained attention, substantially increasing annotation costs. This calls for automated evaluation that can reliably distinguish fine-grained differences among advanced AIVGMs with minimal human intervention. To address this challenge, we present RAVEN-Eval, a rubric-guided automated evaluation framework for AIVGMs, built primarily on the LMM-as-a-judge paradigm. Through an automatic task curation and quality-filtering pipeline, RAVEN-Eval curates 150 text-to-video~(T2V) tasks and 100 image-to-video~(I2V) tasks, and systematically collects more than 4,500 AIGVs. At its core, RAVEN-Eval adopts rubric-guided automated LMM preference judgement, in which LMM judges conduct pairwise comparisons according to task-specific rubrics. It further introduces an anchor-based model insertion approach to reduce the evaluation cost of incorporating new models. Finally, we evaluate 20 high-performance AIVGMs, as well as the judging capabilities of 13 LMM judges, and establish the RAVEN-Eval Leaderboards. Overall, RAVEN-Eval paves a scalable path for automatic and trustworthy evaluation of rapidly evolving AIVGMs. 

---
# Different Feedback, Different Updates: Selective Self-Learning from User Interactions for Large Language Models 

**Authors**: Xuanchen Li, Haitao Li, Yujia Zhou, Qingyi Pan, Heng Wang, Yiqun Liu, Min Zhang, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09109)  

**Abstract**: User feedback offers natural supervision for persistent LLM improvement, but a single message may support multiple behavioral changes with different scopes of generalization. We introduce SLIFT, a selective self-learning framework built on a task-relative view of user feedback. SLIFT decomposes each feedback message into atomic components and interprets each component relative to the original task as Fix, Spec, or Null: requirements for task validity, compatible condition-specific refinements, or content with no reliable positive update direction. To incorporate each change at the appropriate scope, SLIFT trains two complementary LoRA adapters on a shared frozen backbone: a Generalist that consolidates Fix requirements into default behavior through feedback-conditioned self-distillation, and a Specialist that observes only the task and Generalist response to supply residual guidance for applicable, unmet Spec refinements. Null components induce no positive update. Across backbones, SLIFT achieves strong performance on both MemoryBench and WildFB, with targeted analyses further examining its underlying mechanisms. We release our code at this https URL. 

---
# Who Bridges Safety? Identifying and Targeting Cross-Lingual Shared Safety Pathways 

**Authors**: Shuyi Miao, Wangjie Qiu, Pengyang Shao, Canran Xiao, Fei Shen, Zhiming Zheng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.09095)  

**Abstract**: Uncovering the internal mechanisms underlying the safety capabilities of large language models (LLMs) is crucial for developing trustworthy artificial intelligence. Currently, mechanistic interpretability studies on multilingual safety are largely confined to local components, such as isolated neurons. However, this static and fragmented perspective overlooks the synergy among components and fails to elucidate how safety signals dynamically propagate within the model to drive safety decisions ultimately. In this work, we move beyond isolated neurons to identify and target the cross-layer functional pathways formed during safety signal propagation, thereby uncovering the mechanisms driving the cross-lingual safety gap. Specifically, we first identify monolingual safety pathways and validate their impact on refusing harmful requests. Subsequent cross-lingual analyses reveal a sparse subset of cross-lingual shared safety pathways, confirming that this intersection acts as the internal bridge transferring safety capabilities from high-resource (HR) languages to non-high-resource (NHR) languages. Building on these mechanistic findings, we propose a pathways-targeted alignment method based on the cross-lingual shared safety pathways. Experimental results show that updating only a small fraction of pathway parameters significantly improves safety in NHR languages while largely preserving the model's general capabilities. 

---
# A Multi-Scale Temporal Framework with Dynamic Fusion for EEG-Based Emotion Recognition 

**Authors**: Stefanos Gkikas, Yang Guo, Guangliang Li, Raul Fernandez Rojas, Giorgos Giannakakis, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2608.09088)  

**Abstract**: Mixed emotions represent a clinically relevant but still underexplored target for automatic emotion recognition. EEG provides millisecond-level access to neural activity, yet most EEG pipelines analyze the signal through a single temporal window, thereby fixing the temporal structure available to the model. This study introduces a multi-scale temporal framework for EEG-based emotion recognition. The EEG waveform is decomposed into windows of one or several durations, processed by a shared attention-based encoder, and integrated through a dynamic fusion module that assigns sample-specific weights across temporal scales. The framework is evaluated under a subject-independent protocol in binary and three-class settings, with the three-class task including the mixed affective category. The best results are 65.22% for the two-class task and 45.43% for the three-class task. Both are obtained with three-scale dynamic-fusion configurations and remain substantially above the full-signal baseline. The best-performing temporal scales differ between the two tasks. Dynamic fusion outperforms concatenation in the highest-scoring two-class configuration and slightly exceeds it in the highest-scoring three-class configuration, although these multi-scale settings require substantially more computation than the full-signal baseline. 

---
# DualCert: A Solver for the Traveling Salesman Problem with Constraint-Coupled Learning 

**Authors**: Yancheng Song, Yongzhi Qi, Wei Qi, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09042)  

**Abstract**: Large traveling salesman problem (TSP) instances require a solver to allocate limited computation while preserving the validity of its outputs. Existing neural--operations-research (OR) hybrids predict guidance without requiring learned transitions to satisfy constraints discovered during search. DualCert introduces \emph{constraint-coupled learning}, in which current degree equations and dynamically separated subtour-elimination constraints (SECs) define each learned transition. At each refinement, the degree equations and selected, strictly satisfied SEC equations, with positive slacks, define an iterate-dependent primal-slack Karush--Kuhn--Tucker (KKT) manifold. Repaired dual variables and violated SEC rows define a local cost field. An exact constrained mirror-descent step maps each finite state to a positive state on the same manifold. Where selected rows and deterministic ties remain fixed, implicit differentiation maps parameter perturbations into the manifold tangent space and reuses the forward constraint operator for the local-cost-field derivative. The terminal edge state allocates computation across Held--Karp ascent, candidate-graph edge tests, and tour construction under a fixed budget. Deterministic verification recomputes original costs and accepts only verified candidate-graph lower bounds and edge decisions. On 1,000 held-out TSP1000 instances, DualCert attains a mean tour-cost gap of \(0.0573\%\) from Lin--Kernighan--Helsgaun version 3 (LKH-3) reference tours in \(9.55\) batch-amortized seconds per instance. It returns a verified candidate-graph lower bound for every instance and achieves \(81.46\%\) edge-decision coverage. The mean gap is \(67.1\%\) smaller than the reported NeuroLKH mean gap. Thus, optimization constraints govern learning, while deterministic verification preserves output validity. 

---
# PolicyKG: An Agentic LLM Pipeline for Translating Institutional Policies into SHACL Knowledge Graphs 

**Authors**: Ponkrit Kaewsawee, Chaklam Silpasuwanchai, Chutiporn Anutariya  

**Link**: [PDF](https://arxiv.org/pdf/2608.09028)  

**Abstract**: Institutional policies stay in natural language while the systems that check compliance demand machine-readable constraints. Bridging that gap is still done by hand.
PolicyKG closes the loop. It is an LLM pipeline that reads a policy PDF, classifies each sentence as an obligation, permission, or prohibition, lifts the label into first-order deontic logic, and emits SHACL constraints. Four stages run on a LangGraph state machine with per-stage validators. The piece that matters most is the Corpus Adapter: a YAML vocabulary registry that grounds LLM predicates in a target ontology. Retargeting to a new domain means swapping the registry, not retraining a model.
On the Asian Institute of Technology Policies and Procedures corpus (1,663 sentences, 443 rules), PolicyKG reaches 86.9% deontic classification accuracy (Cohen's kappa = .709). Three annotators independently re-label a 50-item sample and agree at Fleiss' kappa = .844. SHACL shape correctness on a 69-shape subset is F1 = .866. The FOL path handles 79.2% of rules; the rest go through a direct NL-to-SHACL fallback.
We audited every one of the 443 rules for second- or higher-order constructs. An automated regex checklist flagged none, and a first-author pass on the 92 FOL-fallback cases confirmed the same. The exact upper 95% Clopper-Pearson bound on the true HOL rate is 0.67%. This is an audit finding for one corpus, not a proof of FOL sufficiency for institutional policy.
Swapping the AIT registry for a GDPR registry raises exact property alignment from 1/15 to 11/15 (Fisher's exact p < .001; Cohen's h = 1.53). On the LexDeMod lease-contract benchmark (N = 200), Macro F1 drops to .370 because lease English uses "shall be entitled" for permission -- exactly the vocabulary mismatch registry swap is meant to fix. Repeated runs produce hash-identical SHACL outputs. 

---
# Context Is Not Authority: Structured Runtime Governance for Financial Market Agents 

**Authors**: Rui Tang, Qiangqiang Liu, Yichi Zhang, Youwei Wang, Xi Chen, Chen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.09025)  

**Abstract**: Financial agents can turn correct context into an unauthorized effect: a customer-facing commitment, trade, or deployed policy. We present SAGE-Fin, a finance-specific authority-handoff contract that makes the proposed effect, not merely its text, the object of runtime control. SAGE-Fin compiles proposals into typed, adapter-bound candidates; records missing or stale institutional obligations as coverage debt; contracts authority under current market, account, policy, and dialogue state; and requires an exact-artifact receipt whose nominal type matches the consuming response, execution, or policy adapter. Evidence and workflow progress cannot substitute for effect authority, and prior authorization is rechecked after state changes. Across an authored 616-case catalog, five deterministic specifications yield 3,080 outputs; a label-isolated harness obtains 616/616 binary reference-prototype parity, including 3/3 named response-gate fixtures, while 22 tests cover selected paths. These results establish executable conformance, not independent safety accuracy. Separately, SAGE-Fin's response gate processed real customer-facing production requests at a confidential digital-asset platform. An operational team independent of the implementation team reached a strongly positive post-deployment conclusion on practical usefulness and workflow fit, and end-user feedback was also strongly positive. Disclosure permits only the review's independence, stakeholder classes, assessed dimensions, and directional conclusion, so this is qualitative field corroboration rather than an aggregate effect estimate. Three distinct de-identified predecessor failures, with independently confirmed 0/3 interception, ground repeated-emission drift, stale account evidence, and missing escalation state without estimating prevalence or treatment effect. 

---
# CoRe-UIE: Rethinking Coexisting and Region-wise Degradation for Underwater Image Enhancement 

**Authors**: Weifeng Kong, Chenghao Xu, Lin Chen, Ziheng Cao, Guanying Huo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08965)  

**Abstract**: Underwater images often suffer from diverse and coexisting degradations, including color distortion, scattering haze, texture attenuation, and uneven illumination. These degradations vary across regions and may coexist locally, making conventional uniform restoration difficult to adapt to different degradation patterns. To address this problem, we propose Coexisting and Region-wise Degradation for Underwater Image Enhancement (\textbf{CoRe-UIE}), a degradation-oriented expert collaboration framework. CoRe-UIE combines a content-preserving shared expert with four shared-backbone routed experts for color correction, scattering suppression, texture recovery, and illumination protection. The routed experts share the same architecture but have independent parameters, and are assigned to different regions through input-derived degradation cues and region-adaptive Top-\(k\) routing. We further introduce a Hilbert--Schmidt Independence Criterion (HSIC)-based representation constraint to reduce statistical dependence among expert features and alleviate redundant expert responses. Experiments on UIEB, LSUI, and U45 demonstrate that CoRe-UIE achieves competitive quantitative performance and visually balanced enhancement under diverse underwater degradation conditions. 

---
# Reading is not Reasoning: Bridging the Agentic Policy Gap in Vision-Text Compression 

**Authors**: Cheng Fan, Junyi Zhou, Tingzhang Luo, RongJian Xu, Qiyanhui Lu, Mingjian Zhu, Hanting Chen, Jianyuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08960)  

**Abstract**: Multi-step language-model agents repeatedly process growing interaction histories, leading to substantial context costs. Vision--text compression reduces these costs by rendering history as images, but the resulting modality shift creates a marked capability gap. Through controlled evaluations of history recovery, matched-state decisions, and complete trajectories, we show that this gap cannot be explained by OCR quality alone. Visual-history agents exhibit systematic drift in action selection, query formulation, stopping, and evidence use, revealing an agentic policy gap. We introduce \textbf{CAPS}, a two-stage \textbf{C}ross-modal \textbf{A}gentic \textbf{P}olicy \textbf{S}elf-distillation framework that uses the same model's stronger text-history policy to supervise its visual-history counterpart. Offline trajectory self-distillation transfers successful text-policy behavior to visual-history inputs, while online policy self-distillation provides dense supervision on states visited by the visual-history policy during reinforcement learning. On SearchQA, CAPS improves over AgentOCR by 5.0\% and 3.4\% with 3B and 7B backbones, respectively. On full-history ALFWorld, the corresponding gains are 15.6\% and 14.5\%. Across settings, CAPS reduces average memory-context cost by up to 63.3\% and peak cost by up to 83.4\% relative to matched text-history policies. These results show that explicit cross-modal policy self-distillation can preserve agent capability under vision--text compression. Our code will be made publicly available in a future release. 

---
# Depth-Aware Implicit Neural Representation Priors for 3D Gravity Inversion 

**Authors**: León Suarez-Rodriguez, Paul Goyes-Peñafiel, Javier Torres-Quintero, Henry Arguello  

**Link**: [PDF](https://arxiv.org/pdf/2608.08959)  

**Abstract**: Gravimetry images subsurface density contrasts associated with geological structures, geothermal systems, and intrusive bodies. Recovering a three-dimensional density model from gravity observations is highly ill-posed because of its non-uniqueness, limited data coverage, and the attenuation of the gravity field with depth. Classical inversion methods rely on explicit regularization and parameter tuning, whereas supervised deep-learning approaches require representative gravity--density pairs that are rarely available. This paper proposes an unsupervised depth-aware implicit neural representation for 3D gravity inversion. The density volume is represented by multiple coordinate-based neural networks assigned to overlapping depth slabs and optimized directly from the observed gravity measurements through the sensitivity matrix. Slab-specific Fourier features, physics-based depth gains, and scheduled regularization provide structural priors without requiring labeled density models. Experiments on four synthetic scenarios show that the proposed method provides better overall performance in terms of RMSE, PSNR, and SSIM than the evaluated conventional and neural baselines. It also recovers more compact and spatially coherent density bodies, improves the separation of nearby anomalies, preserves internal structures, and reconstructs their vertical extent better. These results indicate that the proposed depth-aware formulation helps to mitigate the depth ambiguity inherent in gravity inversion. In the field experiment, where no ground-truth density model was available, the method produced compact, separated, and vertically coherent anomalies consistent with the observed gravity pattern. 

---
# Not an A11y: How Android Accessibility Exposes Mobile AI Agents to Indirect Prompt Injection 

**Authors**: Rahul Deivasigamani, Sayeda Faatin Alvi, Derqui Andrea, Kaushal Punjabi, Stjepan Picek  

**Link**: [PDF](https://arxiv.org/pdf/2608.08939)  

**Abstract**: The rise of autonomous AI agents represents a major paradigm shift in how users interact with mobile devices. Frameworks such as MobileRun and Mobile-Use can autonomously navigate Android applications and execute complex multi-step tasks. To interpret user interfaces, these frameworks rely primarily on Android accessibility (A11y) trees and secondarily on visual screenshots. In this paper, we demonstrate that this architectural dependence on unsanitized accessibility metadata, together with visual input, introduces a systemic vulnerability to indirect prompt injection. We show that adversarial prompts can cause autonomous agents to abandon their original objectives, violate context boundaries, and perform unauthorized device actions. Our empirical evaluation demonstrates goal hijacking, context drift, and unauthorized actions across visually hidden and fully exposed attack scenarios. In aggregate, MobileRun reaches an attack success rate of 0.822 with Gemma4:31B, while Mobile-Use with Qwen3.6:35B reduces this to 0.150 but does not eliminate context drift or unauthorized actions. These findings reveal that current mobile agent frameworks fail to enforce semantic context boundaries, treating passive environmental text as trusted instructions. Finally, we present a taxonomy of these attacks and discuss the need for zero-trust input validation, dedicated security agents, and strict context isolation within mobile agent architectures. 

---
# Integrated Multimodal AI System for Retrieval-Augmented Reasoning, Object Sensing, and Damage Analysis 

**Authors**: Kalelo Dukuray, Israel Pina, Evan Perez, Erika Ardiles-Cruz, Jie Wei  

**Link**: [PDF](https://arxiv.org/pdf/2608.08935)  

**Abstract**: This work presents a unified multimodal AI system for damage assessment that integrates retrieval-augmented generation (RAG) models, thermal spectrum perception, vision foundation model pipelines, and exploratory wireless signal sensing. A RAG component is developed to ground a locally hosted language model in project-specific documentation, including specialized damage level classification criteria to mitigate hallucinations during inference. Controlled comparisons against static few-shot prompting demonstrate that dynamic retrieval improves grounding and factual consistency. We further compare vector-based RAG with a knowledge graph variant constructed via entity-relation extraction, and show that graph-based retrieval produces stronger responses for damage assessment queries requiring cross-document reasoning, motivating hybrid dense, sparse, and graph-aware retrieval. To address limitations of EO imagery under adverse lighting and weather conditions, infrared (IR)/thermal sensing is employed for object detection and segmentation. Our detectors generate candidate detections, yielding improved segmentation of a broad array of objects. Paired IR versus visible spectrum tracking experiments reveal failure modes, motivating multimodal fusion for robust object detection and damage analysis. Vision foundation and vision-language models are leveraged to generate synthetic damage imagery and classify damage severity with high accuracy, supporting training and validation of downstream damage assessment models. Finally, exploratory Wireless-based sensing demonstrates potential to detect presence, motion, and post-event environmental changes where EO and IR sensing are ineffective. 

---
# Decoding Phenotypes: A Framework for Fusing Genomic Language Models and Neuroimaging 

**Authors**: Tianli Tao, Ziyang Wang, Emma Robinson, Rachel Sparks, Le Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08926)  

**Abstract**: Neuroimaging and genetic testing are two important clinical references for nervous system diseases, offering complementary diagnostic information. However, integrating genomic and neuroimaging data for precise disease diagnosis is challenging due to cross-modality heterogeneity. Existing imaging-genetics approaches mainly encode genetic information as hard-coded labels, which lose the local sequence context around disease-associated variants. To address this limitation, we propose GeneFuse, a multimodal learning framework that aligns genetic representations from pre-trained Genomic Language Models (GLMs) with features extracted from images. GeneFuse integrates two components: (1) Genotype-Conditioned Feature Modulation (GCFM), a FiLM-inspired module that uses genomic embeddings to modulate image feature maps; and (2) Uncertainty-aware Genomic Residual Fusion (U-GRF), a fusion strategy that uses imaging-derived predictive uncertainty to gate the contribution of genotypic features. We evaluate GeneFuse on early cognitive decline identification (NC vs. MCI) and dementia screening (NC vs. AD). In the APOE-centered setting, GeneFuse achieves AUROCs of 0.77 and 0.83, outperforming existing imaging-genetics fusion methods. These results indicate that GLM-derived genomic embeddings provide additional information to imaging. 

---
# From Manuals to Maintenance: Fine-Tuning MedGemma for Multi-Modal Imaging System Support in Low-Resource Settings 

**Authors**: Bernes Lorier Atabonfack, Zion Kongbi Nfo, Ahmed Tahiru Issah, Tolulope Olusuyi, Clemence Ingabire, Mohammed Hardi Abdul Baaki, Mawuli Deku, Abdulrazaq Zubair, Alyasaa Anas, Raymond Confidence, Maruf Adewole, Udunna C. Anazodo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08896)  

**Abstract**: Imaging device downtime is a major barrier to healthcare delivery in low- and middle-income countries (LMICs), often driven by limited access to specialized biomedical engineering support. We present a multi-modality medical equipment maintenance question-answering (QA) framework and demonstrate the fine-tuning of a medical foundation model for specialized technical troubleshooting tasks. Guided by a multi-country survey across nine LMICs, we curated technical manuals from MRI and ultrasound systems to generate the INGENZI_DatasetV1, containing 10,294 high-quality, filtered QA-context pairs. Using QLoRA-based parameter-efficient fine-tuning, we adapted the MedGemma-4b-it model to interpret system error logs and generate step-by-step equipment repair instructions. Compared to the baseline model, the fine-tuned system achieved substantial improvements across metrics, including F1 score (0.22 to 0.38), ROUGE-2 (0.18 to 0.41), and BERTScore F1 (0.86 to 0.91). These metric gains demonstrate that the model generates significantly more precise and procedurally accurate technical responses to new troubleshooting queries. This work establishes a reliable foundation for AI-assisted diagnostic and maintenance tools in resource-constrained settings. 

---
# LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing 

**Authors**: Juncheng Dong, Ding Tong, Ishan Gupta, Yuyan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08889)  

**Abstract**: Recommendation systems thrive on personalization, where ''correctness'' is rarely a binary truth but a matter of subjective human preference. As Large Language Models (LLMs) are deployed as autonomous verifiers of safety and quality guidelines, they face a distinctive challenge: context-aware preference alignment. Recent gains in Reinforcement Learning with Verifiable Rewards (RLVR) are indexed mostly on objective, mathematical tasks. Through a large-scale study spanning both proprietary and open-source models on four real-world verification tasks from a production recommender platform, we ask whether explicit reasoning generalizes to subjective, human-centric industry rubrics. We expose a fundamental vulnerability: rigid, math-centric reasoning traces actively degrade verification, and applying standard RLVR triggers a phenomenon we term reasoning collapse, in which the policy abandons deliberation in favor of rapid heuristic guessing. We introduce a conditional length-penalized post-training algorithm that intertwines verification accuracy with bounded reasoning length, halting collapse and recovering performance. Finally, we show that a reasoning trace's efficacy is tightly coupled with its socio-linguistic framing: across 1500 synthesized personas, verification accuracy swings by nearly 0.38 macro-F1 depending solely on the adopted reasoning persona---evidence that much subjective-verification error is really reasoning-style mismatch. This observation motivates a mid-training architecture that routes reasoning through contextually aligned personas. This work offers both a scalable algorithmic patch and a long-term architectural blueprint for aligning reasoning models with real-world subjective constraints. 

---
# Full-bandwidth transformer 

**Authors**: Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2608.08888)  

**Abstract**: Autoregressive transformers compute along two axes: horizontally across generated tokens, and vertically through model depth. Dense attention gives each token broad horizontal access to the past, but the vertical feedback channel between decoding steps remains narrow: only the sampled token returns to the bottom of the stack, while the top-layer hidden state is discarded. We introduce the \emph{full-bandwidth transformer}, which widens this channel with \emph{latent feedback}: at each decoding step, the previous top-layer hidden state is fused with the sampled token embedding through a gated linear unit and fed back as the next input. Latent feedback lets non-verbalized computation re-enter the stack with a renewed depth budget, while preserving the standard transformer architecture, KV cache, and language-modeling objective. To train full-bandwidth transformers without losing parallel teacher forcing, we use a scheduled multi-pass objective that introduces latent feedback late in pretraining and mixes a small fraction of deeper feedback passes for stability. We train 1B-parameter full-bandwidth transformers up to 400B tokens and find that latent feedback improves validation loss, 5-shot language-model evaluation, math and coding generation, and instruction-tuned performance. With negligible per-token decoding overhead, full-bandwidth transformers match or approach standard transformers trained with roughly $1.5\times$ more tokens, and manage to produce shorter reasoning traces at equal or better accuracy. 

---
# AquiLLM: An Architecture for Supporting Tacit Knowledge Capture in Research Groups 

**Authors**: Jack Stark, Srinath Saikrishnan, Vikram Seenivasan, Bernie Boscoe, Andrew Lizarraga, Tuan Do  

**Link**: [PDF](https://arxiv.org/pdf/2608.08883)  

**Abstract**: Recent advances in retrieval-augmented generation (RAG) and large language models (LLMs) enable researchers to integrate AI into scientific workflows. However, using proprietary commercial AI systems raises concerns about transparency, reproducibility and privacy, which are essential for scientific practices. To this end, AquiLLM was developed as an open-source modular RAG-LLM framework using open-weight models, designed to support research groups in capturing tacit knowledge. In this work, we present a series of architectural improvements and feature enhancements to AquiLLM, including local embedding and reranking, multimodal capabilities, OpenAI-compatible inference interfaces, user interface improvements, semantic and episodic memory capabilities, and skills support. These enhancements were informed by discussions with domain experts, including astrophysicists and environmental researchers, and represent a step toward AI systems more closely aligned with scientific research practices. 

---
# Theory-Guided Deception Detection: A RAG-Based Artificial Intelligence Exploration 

**Authors**: David M. Markowitz, Timothy R. Levine  

**Link**: [PDF](https://arxiv.org/pdf/2608.08881)  

**Abstract**: The current work developed seven Retrieval-Augmented Generation (RAG) models based on leading deception theories and compared how deception judgments were made relative to baseline models. Across 700 statements drawn from five published deception datasets, four large language models (gpt-4o, claude-sonnet-4-6, ollama/llama3, deepseek-v4-flash), and two run-types (RAG vs. baseline), a total of 39,200 deception judgments were rendered. Detection accuracies were consistent with typical human accuracies and not statistically different across RAG (54.5%) and baseline models (54.6%). RAG-based models (57.0%) were less truth-biased than baseline models (59.7%), but the effect size was quite small. Theoretical perspective mattered little for accuracy yet mattered substantially for response bias, which ranged from highly lie-biased (the verifiability approach, 32.2%) to highly truth-biased (truth-default theory, 88.1%). Content effects and model effects further moderated the results. Theory-guided AI judgments are unreliable with current parameters, yet they might show promise with additional datasets, model testing, and theory-to-data matching. 

---
# Findings of the First Teaching Monster Challenge: A Benchmark of Pedagogical Content Knowledge in AI Agents 

**Authors**: Yi-Cheng Lin, Yu-Kai Guo, Szu-Chi Chen, Bo-Han Feng, Yun-Man Hsu, Hsiang Hsieh, Yu-Jung Lin, Yue-Ling Wu, Jia-Kai Dong, An-Yu Cheng, Yu-Han Huang, Lok-Lam Ieong, Kuan-Yu Chen, Ming-Douo Tchouang, Shao-Hua Sun, Che Lin, Jian-Jiun Ding, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.08852)  

**Abstract**: AI agents can now solve problems, answer like subject experts, and generate long-form multimodal content. However, whether they can adapt a lesson to fit a specified learner, which education calls Pedagogical Content Knowledge (PCK), has not been benchmarked. To measure it, we introduce the Teaching Monster Challenge, the first instructional video generation benchmark to treat the learner persona as an explicit evaluation criterion. Each system is given a topic and a learner persona and must generate a complete instructional video. Every video is screened by an LLM-judge, ranked by crowd pairwise voting, and finalized by an expert panel. The first edition shows that today's systems handle the content well but are far weaker at presenting it and adapting it to the learner. The same process exposes a limit of automatic judging. The LLM-judge separates a clear low-performing tail but ranks the strongest systems poorly. The strongest systems receive nearly identical scores from the judge, so its ranking of them does not match human preference. Progress therefore requires not only better teaching systems but also better automatic judges, and we release the benchmark, rubric, and human judgments as a testbed for both. 

---
# PROSLEX: A Novel Dataset for Expert-Annotated Legal Statute Prediction for Indian Judiciary 

**Authors**: Subinay Adhikary, Upal Bhattacharya, Vivek Kumar Singh, Anurag Sharma, Shubham Kumar Nigam, Suvasis Das, Shouvik Kumar Guha, Koustav Rudra, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.08830)  

**Abstract**: Legal Statute Prediction (LSP) involves automatically identifying relevant legal statutes given factual descriptions in legal documents, typically framed as a multi-label classification task within natural language processing and information retrieval research. While recent advances have begun incorporating Large Language Models (LLMs) for statute prediction, current approaches primarily focus on accuracy metrics without addressing the critical need for legal reasoning, a fundamental requirement in judicial contexts where decisions must be explainable and justifiable. To address this research gap, we present PROSLEX (PRediction Of Statutes and LEgal eXplanation), a comprehensive dataset comprising 1,623 expert-annotated legal documents from the Indian context. Each document is paired with statute predictions and detailed explanations, totaling 7,450 explanations, capturing the underlying legal reasoning. Using this dataset, we systematically evaluate various prompting strategies, including zero-shot, few-shot, chain-of-thought, and tree-of-thoughts approaches, to generate both statute predictions and their corresponding legal rationales. Our evaluation framework measures not only predictive performance but also the coherence and legal validity of generated explanations, positioning PROSLEX as a benchmark for developing explainable AI systems that can support legal practitioners while advancing research in interpretable legal NLP. To ensure reproducibility, we have made our PROSLEX dataset and model code available on GitHub: this https URL. 

---
# Automated Generation of Complexity-Validated Decision Scenarios Using Large Language Models 

**Authors**: Abdalla Doleh, Toni Somers, Ratna Babu Chinnam  

**Link**: [PDF](https://arxiv.org/pdf/2608.08822)  

**Abstract**: Cognitive decision-making research depends on diverse scenarios with carefully controlled complexity, yet manual production is slow, inconsistent, and biased. We developed an automated pipeline that uses LLms to generate structured decision scenarios and validates their complexity through a composite framework rooted in established task-complexity theory. We evaluated 4,238 scenarios across multiple domains and complexity tiers. Measurement validation met rigorous psychometric standards. Agreement among five independent model families was nearly perfect, with an intraclass correlation coefficient of 0.997 and a kappa of 0.971. Known-groups validity demonstrated large separation between tiers, with an eta-squared of 0.587 and all pairwise comparisons significant at p less than .001. Factor analysis revealed a dominant complexity construct, with loadings between 0.87 and 0.96 across three frameworks, while interactivity formed a weaker secondary dimension at 0.34. Discriminant validity was limited by a strong relationship between complexity and text length that persisted after controlling for tier, yielding a partial correlation of 0.86. This constrains construct purity but does not undermine the instrument's tier-grading function. Model analyses showed a negative association between throughput and schema pass rate (r = -0.967, p = .007, n = 5), suggesting a speed-quality trade-off, though largely driven by one high-throughput model. Llama 4 Maverick generated scenarios fastest at 134 per minute versus 25 for DeepSeek Chat V3.2, but underproduced complex-tier scenarios, whereas DeepSeek Chat V3.2 balanced domain coverage with high schema compliance. The system demonstrated strong psychometric properties, enabling reliable classification into Simple, Moderate, and Complex tiers and providing the measurement infrastructure needed for downstream cognitive assessment of AI systems 

---
# Three Generations of Healthcare IT: From the Digital Record to the Computable Care Process 

**Authors**: Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2608.08806)  

**Abstract**: Objective. Healthcare IT is usually organized by the technologies it adopts. We instead organize it by the unit of information a system makes computable, and describe a computational layer whose object is patient-specific clinical intent. Approach. We give criteria for a computational layer, derive three (record, clinical state, and a proposed layer of intent), and formalize the Actionable Clinical Record (ACR) as the atomic object of the third layer. Discussion. The framework distinguishes prescribed, observed, and intended process; existing standards represent intent once it is structured but do not recover it from natural communication, the capability we localize. The ACR is complementary to FHIR workflow resources, guidelines, and process mining; a companion feasibility study illustrates tractability for one narrow subproblem. Conclusion. Computable clinical intent is a coherent research direction; the ACR, its readiness ladder, and an executable-correctness evaluation framework are reusable constructs for subsequent work to extend, evaluate, or falsify. 

---
# Improving Generalization Robustness of Multimodal RLVR 

**Authors**: Pengfei Zhou, Zhiwei Tang, Xiaopeng Peng, Chenrui Zhou, Lama Moukheiber, Yixing Ma, Bin Xu, Jiajun Song, Zhenglin Wan, Wangbo Zhao, Jiasheng Tang, Bohan Zhuang, Fan Wang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2608.08802)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) makes Multimodal Large Language Models more accurate, but the gains are brittle: simply paraphrasing a question or changing the prompt template can degrade them, which challenges reliable deployment in high-stakes scenarios like medical VQA. We trace this to two issues of the standard RL objective. First, the binary verifier conflates format with content, so the reward signal cannot tell a wrong answer apart from a misformatted one. Second, the training distribution covers only a thin slice of the real-world prompts that the model might meet at deployment, so policies that perform well on the training distribution can behave differently under unseen prompts during test. Both failures call for a robust post-training method that helps the policy cover a broader distribution of semantically equivalent prompts, and we identify two measures that help achieve this objective: separating format from semantics in the reward, and applying policy invariance across perturbed prompts with equivalent semantics. We therefore propose Prompt-Invariant RLVR (PIRL), consisting of a dynamic trinary reward and a consistency regularizer based on an embedding-space adversary. Under stress testing, PIRL's average accuracy on benchmarks drops by only $\le 1\%$, where GRPO drops ~3%. On dynamic evaluation, PIRL also achieves the smallest performance drop. 

---
# Deferred Audio Pruning with Local Audio-Visual Dynamics for Omni-LLMs 

**Authors**: Kyeongyoon Lee, Hongyeob Kim, Youngeun Kim, Sungeun Hong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08794)  

**Abstract**: Omni-modal LLMs jointly process audio, video, and text, but long multimodal sequences incur substantial prefill and KV-cache costs. Existing omni-modal compression methods primarily focus on pre-LLM token reduction, leaving modality-specific compression across the LLM boundary underexplored. We propose A-PACK, a two-stage framework that defers audio pruning until query-conditioned multimodal interactions emerge. Our analysis shows that audio exhibits higher task-relevant information density and representational diversity per token than video. We further find that local audio-visual dynamics provide a more effective cue for visual selection than token-wise matching. We therefore preserve audio and compress video with local dynamics before the LLM, then progressively prune low-relevance audio and visual tokens and their KV-cache entries inside the LLM. Across four benchmarks on Qwen2.5-Omni-7B/3B, A-PACK achieves the strongest average performance among the evaluated prior methods while reducing prefill FLOPs by up to 78% and improving decoding throughput by up to 2.21x. 

---
# SymDiag: Explainable Diagnosis for LLM Reasoning via Neuro-Symbolic Verification 

**Authors**: Wenyao Cui, Huaping Zhang, Yongyi Huang, Qiuchi Li, Jian Xu, Cheng-Lin Liu, Chunxiao Gao, Juan Wang, Baohua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08786)  

**Abstract**: Large language models (LLMs) increasingly serve as data-driven reasoners, yet their chains-of-thought (CoT) can be unfaithful even when final answers are correct. Most existing ``verification'' signals are not diagnostic: answer matching observes only the outcome, LLM-as-judge provides subjective and non-verifiable critiques, and scalar rewards (e.g., PRMs/RMs) offer little insight into where a multi-step derivation this http URL propose \textbf{SymDiag}, a neuro-symbolic framework that \textbf{reframes reasoning verification as structured failure diagnosis}. SymDiag translates natural-language CoT into symbolic constraints and performs step-level satisfiability/entailment checks to (i) localize failing steps and (ii) produce verifiable diagnostic evidence, including counterexamples, inconsistency witnesses, and missing-premise indicators. A central challenge is that apparent ``logic violations'' can be caused either by genuine reasoning defects or by neural-to-symbolic translation noise. SymDiag therefore incorporates a Self-Auditor that disentangles TranslationError from ReasoningError via dual symbolic encodings consistency checks, enabling robust diagnosis under partial observability. Across diverse mathematical, logical, scientific, and general reasoning benchmarks, SymDiag improves detection of unfaithful reasoning and provides substantially more effective feedback for multi-round reasoning repair than outcome-only verification and LLM-based judging, offering a principled foundation for trustworthy and scalable reasoning diagnosis. 

---
# Scale-to-Dialogue: Low-Burden Elicitation of Daily Premenstrual Symptom Ratings with Small Language Models 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08746)  

**Abstract**: Prospective daily symptom tracking is central to premenstrual health assessment, but repeated ordinal forms impose substantial response burden. We formulate conversational administration as an ordinal label-recovery problem: the system actively elicits a small set of symptom clusters and maps each response to the original severity labels. We used 3,320 complete participant-days from the mcPHASES dataset, covering cramps, mood swing, fatigue, sleep issues, stress, and bloating on a six-level scale. Six participants were reserved for development and 36 for a frozen evaluation comprising 360 participant-days and 2,160 item labels. A ModernBERT evidence gate detected whether a symptom was expressed, and Qwen2.5-1.5B-Instruct produced deterministic structured severity scores. Fixed six-item questioning achieved a quadratic weighted kappa of 0.976, whereas three joint symptom-cluster questions achieved 0.913, 97.45% agreement within one severity level, and 80.94% recall for moderate-or-higher symptoms while reducing questions by 50%. Open-first adaptive policies required 3.92-5.98 questions and produced lower agreement than the corresponding fixed policies. Participant-cluster bootstrap analysis estimated a kappa difference of -0.062 (95% CI -0.076 to -0.048) between the three-cluster and six-item strategies. Active cluster-level elicitation provides a direct, local-model route from natural conversation to reusable daily symptom labels. 

---
# FitAQA: A Benchmark of Fitness Action Quality Assessment for Multimodal Large Language Models 

**Authors**: Kaili Zheng, Kaiwen Wang, Xun Zhu, Qingyuan Yang, Chenyi Guo, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08736)  

**Abstract**: Fitness Action Quality Assessment (AQA) is important for intelligent sports training, yet the capabilities of Multimodal Large Language Models (MLLMs) in this setting remain underexplored. Existing benchmarks rely on action-specific annotation schemes and focus primarily on final assessment outputs, offering limited insight into how models assess exercise quality. We introduce FitAQA, a systematic benchmark for evaluating MLLMs in fitness AQA, containing 2,219 videos and 5,512 QA instances across 30 bodyweight exercises. In collaboration with experts in sports science, we develop a unified form error taxonomy that defines 38 recurring form errors within six complementary quality dimensions: alignment, symmetry, stability, coordination, tempo, and completeness. This taxonomy provides a shared assessment framework across different exercises. FitAQA further formulates three evaluation tasks: perception for recognizing relevant visual evidence, judgement for combining that evidence with domain knowledge to assess execution correctness, and temporal grounding for localizing form errors over time. Extensive evaluation shows that current MLLMs still struggle to assess exercise quality comprehensively and localize form errors precisely. Controlled experiments further indicate that visual perception is a key bottleneck, as judgement performance improves substantially when ground-truth perceptual evidence is provided. The dataset and evaluation code will be made publicly available. 

---
# AI Evaluation Should Measure Verification Cost, Not Correctness Alone 

**Authors**: Viviana Crescitelli, Generoso Immediato, Fabio Persia, Stefania Costantini  

**Link**: [PDF](https://arxiv.org/pdf/2608.08709)  

**Abstract**: The reliability of AI generative models is typically measured by output correctness, yet in practice it depends on the effort required to verify those outputs. We argue that current evaluation metrics overlook a critical failure mode: Verification-Cost Errors (VCEs), defined as incorrect input-output pairs that a declared fraction of the verifier population fails to identify within the verification budget available in a given deployment context. Unlike standard notions of "hallucination", VCEs are defined operationally, by the failure of correct identification within budget rather than by any property of the output itself. Plausibility and authoritative presentation are hypothesised contributors to that failure, not defining conditions. To capture this asymmetry, we introduce the notion of verification cost relative to a deployment budget as an operational dimension that current evaluation does not routinely capture. The quantity is presented as a conceptual instrument rather than a finalized metric. Evidence from code generation and multi-modal document understanding shows that high benchmark accuracy can mask significant verification effort in practice. We therefore take the position that correctness alone is insufficient as a measure of reliability. AI evaluation should explicitly account for verification cost, reflecting whether errors can be detected under realistic resource constraints. 

---
# PluginEval: A Diagnostic Benchmark for Fine-Grained Error Attribution in Function Calling 

**Authors**: Dongjie Xu, Julius, Hanchi Dong, Minghua Tang, Yuxuan Sun, Ziwei Nie, Zicheng Liu, Dujun Qing, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08700)  

**Abstract**: Reliable evaluation of tool routing is critical as Large Language Models increasingly operate as autonomous agents. Current benchmarks face three structural limitations: data distributions that follow a power law leave rare scenarios underrepresented; the absence of adversarial hard negatives obscures performance differences across models; and annotation pipelines depend on LLM judgments that have not been validated through execution. In this paper, we introduce PluginEval, a benchmark constructed through a two-stage framework that systematically mitigates these limitations. First, we formulate tool routing as a sequence of three decisions and separate generation from verification. LLMs propose candidate calls, while deterministic validation and real API execution provide reliable quality signals. Second, we decompose each plugin by capability, intent, and boundary to identify trigger and exclusion scenarios. We then generate queries at different difficulty levels to fill coverage gaps, including adversarial negatives targeting three failure modes, and return them to the first stage for annotation. This process creates a closed loop that iterates until coverage converges. For evaluation, we move beyond aggregate accuracy. An LLM judge anchored to gold annotations classifies failures as missed calls, spurious calls, or parameter errors, producing a detailed error profile for each model. We evaluate five model families, including proprietary models and models with open weights, analyze their performance across difficulty levels and error categories, and validate the judge through agreement with human annotations. 

---
# EnergyBridge: Benchmarking Household Energy Management, User Participation, and Grid Flexibility 

**Authors**: Xudong Wu, Zeqing Wu, Jiarui Zhang, Xuhao Fan, Ziang Ding, Yuming Zhuang, Mingqi Yuan, Yilun Du, Hongjie Jia, Yunfei Mu, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08691)  

**Abstract**: Residential virtual power plants (VPPs) can provide grid flexibility by shifting household demand, but physical flexibility becomes dependable capacity only when residents authorize a plan and the promised response is delivered. Existing benchmarks evaluate control but omit event-specific authorization. We present EnergyBridge, a benchmark and agent framework connecting capacity reporting, household authorization, and physical execution. It combines region-specific EnergyPlus environments for Tianjin and Berlin with an LLM-based User Participation Simulator. Against 584 persona- and event-matched human role-play judgments, the LLM-based User Participation Simulator preserves method ordering with a 5.3-point mean absolute acceptance error. Across conventional controllers and agent baselines, EnergyBridge achieves the highest simulated authorization, lowest event-window energy, and the most reliable capacity commitment in both regions. We release human data and codes for reproducible human-centered grid-flexibility research: this https URL. 

---
# A Structural Dynamics Graph World Model: Unified Modeling, Constrained Rollout, and Interpretable Calibration 

**Authors**: Wei Wang, Yaosen Chen, Han Yang, Yuegen Liu, Mingli Luo, Xinxin Jiao, Xuming Wen, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08689)  

**Abstract**: The state evolution of a complex system arises jointly from object laws, relational propagation, domain conservation, and unmodeled error. Forcing all sources into one black box makes mechanism attribution and constraint preservation unauditable; forcing every mechanism into one equation family discards mature domain solvers. We propose SD-GWM, a Structural Dynamics Graph World Model as an executable structural contract: nodes declare self-dynamics S, edges declare neighbor graph-coupled dynamics N---both fixed-form mechanism assets (rules, ODEs, solvers) calibrating only authorized parameters. An optional bounded residual R concentrates learnability, while a global projection maps states to feasibility, enforcing constraints without guaranteeing accuracy gains. On eight pre-registered research questions, SD-GWM delivers (i) heterogeneous integration: rules and solvers plug in natively; (ii) semantic fidelity: disabling R preserves source semantics bit-for-bit, with four theory properties under explicit proof/empirical boundaries; (iii) auditable governance: stepwise traces enable counterfactual fault localization (top-1 = 1.0) without post-hoc approximations. On a semi-synthetic flood testbed and USGS streamflow, SD-GWM reduces constraint violations to floating-point tolerance in analytical tests and to zero in semi-synthetic and real-data cases. Persistence matches SD-GWM in calm periods, but during a 254-day extreme-flood shift persistence and all neural baselines collapse (90-min RMSE 892-3007 cfs) while SD-GWM holds at 108 cfs (8-28x gain). The bounded residual cuts RMSE ~50% only under backbone bias. We position SD-GWM not as a universally superior forecaster, but as a verifiable substrate for auditable, constraint-safe spatiotemporal mining. 

---
# Branch2Skill: Efficient Skill Evolution Through Reasoning Trees 

**Authors**: Yanwei Ren, Haotian Zhang, Likang Xiao, Jiaxing Huang, Jiayan Qiu, Baosheng Yu, Quan Chen, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08677)  

**Abstract**: Skill evolution improves agent skills through feedback over time, with failed trajectories often providing informative signals by revealing incomplete or misleading behaviors. However, existing methods mainly rely on single trajectories, where early reasoning errors can propagate through subsequent steps and weaken the feedback available for skill refinement. Consequently, improving skills requires repeated cycles of rollout, diagnosis, and update, incurring substantial token costs. To address this challenge, we introduce Branch2Skill, an efficient framework that transforms a single reasoning tree into dense supervision for skill evolution. For each task or problem, Branch2Skill performs Monte Carlo tree search under a fixed budget to obtain diverse reasoning trajectories, then compares an elite path with sibling alternatives sharing the same prefixes to extract step-wise evidence about which reasoning patterns to retain, revise, or avoid. Finally, Branch2Skill distills multi-step evidence into reusable updates, allowing one reasoning tree to provide supervision across multiple reasoning steps and reducing the need for repeated rollout-update cycles. Across six benchmarks covering reasoning and agentic tasks, Branch2Skill consistently improves task performance while enhancing skill evolution efficiency. For example, with GPT 5.5 as the target model, Branch2Skill uses 73.2% fewer tokens than SkillOpt, while achieving superior performance. These results demonstrate that reasoning trees can support not only more effective trajectory search, but also richer supervision for more efficient skill improvement. Code will be published. 

---
# The Scaffolding Matters More Than the Interface: A Controlled Comparison of MCP and CLI Tool Use Across Seven Agent Scaffoldings, Five Language Models, and One Software Task 

**Authors**: Marc Alier Forment, María José Casañ Guerrero, Francisco José García-Peñalvo, Juanan Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2608.08654)  

**Abstract**: How much an AI coding agent costs to run can depend more on the agent scaffolding that drives it than on the interface through which it
reaches its tools. We set out to measure the cost of tool use over the Model Context Protocol (MCP) against tool use over an ordinary
command-line interface (CLI), a difference on which published estimates disagree by more than an order of magnitude while resting on
practitioner reports that cannot be reproduced. We ran one fixed software task -- six operations against a private online git repository --
across seven agent scaffoldings and five language models, and we verified completion by inspecting the repository state rather than trusting
the agent's self-report. The dominant effect was the scaffolding. Two of the seven ship no MCP support at all; they completed every run using
only the CLI, which shows that MCP is unnecessary for this class of work, and they were 5.0x to 28x cheaper than the five scaffoldings that
do support MCP, comparing CLI runs alone with no MCP server attached anywhere. The effect was largest for a small 27-billion-parameter model
running locally, whose cost varied 139x across scaffoldings while it completed the task under all of them. The comparison we set out to make
proved unstable: thirteen strictly paired MCP-to-CLI ratios span 0.43x to 29x, with outliers on both sides. The two interfaces separate on
the cost of failure, where 12.9 per cent of the money spent on MCP runs bought no completed work against 2.2 per cent on CLI runs, but not on
its frequency: failures were equally common in both, in the original runs and in their repetitions alike. Agents frequently ignored the
interface they were assigned, so comparisons that do not verify actual behaviour measure an unknown mixture. The harness, the task, the
verification and the complete dataset are released as open source. 

---
# SkillReason: Reasoning-Enhanced Agent Skill Retrieval for Implicit User Requests 

**Authors**: Donghong Jiang, Endian Lin, Luoping Cui, Hanqing Liu, Mingjie Liu, Fan Yang, Hong Wang, Zhao Yang, Chuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08640)  

**Abstract**: Large language model agents increasingly rely on reusable skills to extend their capabilities beyond parametric knowl- edge. However, retrieving the appropriate skill from a large- scale library remains challenging because realistic user re- quests are often concise and underspecified, stating only the task goal while leaving the required capabilities and execu- tion steps implicit. Existing benchmarks provide limited cov- erage of such requests. To address this gap, we introduce SkillReason-Bench, a large-scale cross-domain benchmark containing 3,729 queries and a retrieval corpus of 61,228 skills spanning nine domains. We further propose SkillRea- son, a two-stage framework that uses chain-of-thought rea- soning as training-time supervision for skill retrieval. In Stage I, capability reasoning traces generated by a stronger teacher provide explicit supervision through contrastive learning, re- trieval distribution alignment, and language modeling, en- couraging the retriever to internalize capability reasoning in its query representation. In Stage II, a retrieval-guided GRPO objective encourages the model to explore reasoning trajecto- ries better suited to its own capabilities and more effective for retrieval. At inference, SkillReason directly encodes the orig- inal query without autoregressive CoT generation, preserv- ing efficient query-only retrieval. Extensive experiments on SkillReason-Bench, SkillRet, and SRA-Bench show that Skill- Reason achieves state-of-the-art performance across all three benchmarks, demonstrating that reasoning-enhanced training better bridges the semantic gap between high-level task goals and skill capabilities. 

---
# Smart Compaction: Predicting Compaction Utility from Lakehouse Table Metadata 

**Authors**: Jannic Cutura, Subash Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2608.08639)  

**Abstract**: Open lakehouse table formats accumulate small data files over time, which degrades query performance. Deciding when compaction is worthwhile remains threshold-driven, but which metadata features actually determine compaction utility is not well understood. We present an open simulation framework that generates 2,376 Apache Iceberg tables spanning three orders of magnitude in file size, extracts 17 metadata features from manifest files without reading data, and trains XGBoost to predict the continuous file-reduction ratio (R2 = 0.998, RMSE= 0.013). The binary compaction decision turns out to be trivially separable by a single partition-level threshold max_files_per_partition> 4, requiring no learned model. Cross-schema validation on 96 TPC-H tables confirms generalisation without retraining (R2 = 0.976). A query benchmark reveals that compaction benefits metadata-heavy queries but can slow full-scan aggregations by reducing task parallelism. All code and data are publicly available. 

---
# Can Open-Weight Models Compete on Financial Text Comprehension? 

**Authors**: Jan Spörer  

**Link**: [PDF](https://arxiv.org/pdf/2608.08634)  

**Abstract**: Open-weight language models from Chinese AI labs caught up on benchmarks relative to proprietary frontier models in recent months. Yet their reliability on real-world financial tasks remains largely untested. We updated the Financial Touchstone benchmark, which now has 2,967 question context-answer triplets across 495 international annual reports. We also apply a new set of models on the benchmark, expanding coverage from eleven to twenty models across ten providers, including recent open-weight models such as GLM 4.7, GLM 5, Kimi K2.6, and DeepSeek V3.2, as well as Alibaba's proprietary flagship Qwen3-Max. Anthropic's Claude Opus 4.6 achieves the highest accuracy (88.4%), while Google's Gemini 2.5 Pro maintains the lowest hallucination rate (0.08%). Notably, the open-weight Kimi K2.6 ranks third in accuracy, and the non-reasoning models GLM 5 and Mistral 3 rank fourth and fifth, challenging the assumption that reasoning architectures or proprietary weights are a prerequisite for strong financial comprehension. Information retrieval remains the primary bottleneck, accounting for 48.9% of all failures. We also document a new finding: geopolitical content filters in Chinese models refuse legitimate financial questions (0.08% of attempts), sometimes without clear reason, and the refusal behavior depends on the access route as much as on the model. The complete dataset and evaluation framework are publicly available. 

---
# A QUBO-Inspired Computational Framework for Airport Landside Bottleneck Diagnosis and Dynamic Dispatch Optimization 

**Authors**: Wuming Lei, Xiaobin Li, Mingyan Sun, Jianing Long, Yulin Tong, Yanbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.08632)  

**Abstract**: Airport landside traffic centers connect terminal arrivals with taxis, ride-hailing vehicles, private cars, buses, metro services, parking facilities, and terminal-area roadways. Peak arrivals can create coupled congestion across passenger queues, vehicle queues, pickup berths, storage areas, and access roads. This study proposes a QUBO-inspired computational framework for bottleneck diagnosis and dynamic dispatch in this setting. Shanghai Pudong International Airport and Hangzhou Xiaoshan International Airport serve as case airports. A five-minute state model links passenger arrivals, vehicle supply, pickup berth service, vehicle storage, and road capacity. Bottleneck diagnosis uses service intensity, road demand saturation, bottleneck frequency, queue severity, shadow-price leverage, and a composite congestion severity index. Two dispatch schemes are tested under consistent demand inputs: finite-action model predictive control and quadratic-unconstrained-binary-optimization-inspired simulated annealing. In the strong-peak baseline scenario, the QUBO-inspired method reduces the final passenger queue from 3445 to 2477 passengers at Shanghai Pudong and from 2053 to 1482 passengers at Hangzhou Xiaoshan. Case results indicate different dominant bottlenecks. Shanghai Pudong is more affected by road saturation, whereas Hangzhou Xiaoshan is more affected by pickup berth service. Robustness tests under demand, supply, service, road-capacity, modal-share, and random-noise perturbations show retained queue-reduction benefits under the tested uncertainty levels. 

---
# UniMoMo: Expert Merging-Based MoE Acceleration for Large Recommendation Models 

**Authors**: Lei Xin, Bin Gu, Peize Li, Zitong Wang, Jianbo Zhao, Changjiang Jiang, Yanyue Xie, Chao Huang, Xuyang Zhao, Zunhai Su, Fanhu Zeng, Zhenglun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08627)  

**Abstract**: Sparse mixture-of-experts (MoE) layers expand recommendation capacity through conditional computation, yet a trained checkpoint still stores and routes over its full expert bank. We study a deployment problem: convert that checkpoint to a smaller standard MoE under an explicit expert budget, without adding a compression-specific online module. To address this, we introduce UniMoMo, a post-training compression framework formulated as a constrained graph coarsening problem. Rather than relying on parameter distance, UniMoMo groups experts based on their functional similarity, using an unlabeled calibration set to measure how similarly experts respond to shared recommendation states. To prevent performance degradation, we introduce a layer-adaptive protection mechanism that restricts the merging of high-traffic experts based on their routing exposure. Across Amazon Beauty, KuaiRec, and TenRec with 2, 4, and 6 MoE blocks, the final four-expert checkpoints obtain source-relative five-run mean NDCG@10 ratios of 99.92%--102.30% and measured A100 speedups of 1.28$\times$--1.63$\times$. An aggressive two-expert, top-1 operating point obtains ratios of 98.36%--104.24% and speedups of 1.47$\times$--2.21$\times$. These endpoint results evaluate the complete conversion-and-adaptation workflow and show that a trained recommendation MoE can be exported at multiple serving budgets. 

---
# MedCalc-R1: Knowledge-Guided Reward Framework for Medical Mathematical Reasoning 

**Authors**: Haotian Wang, Lian Yan, Xingzhi Yao, Fanshu Meng, Ye He, Jingchi Jiang, Yi Guan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08623)  

**Abstract**: In Reinforcement Learning with Verifiable Rewards (RLVR) frameworks for mathematical reasoning tasks, floating-point results are typically evaluated using a tolerance-based reward. However, this strategy suffers from challenges such as difficulty in threshold calibration, unstable training dynamics, and limited accuracy, especially in clinical scenarios. To address these limitations, we propose a knowledge-guided hybrid reward framework (\textsc{MedCalc-R1}). Specifically, we introduce a knowledge verification reward mechanism that enforces explicit generation of computational formulas, which are further validated by an external verifier to enhance interpretability and reasoning reliability. Furthermore, we design a hybrid soft-hard reward scheme combining a hard constraint based on clinical safety thresholds with a soft, precision-sensitive reward that progressively guides learning within the acceptable range. Experimental results demonstrate that our method significantly outperforms existing baselines in both reasoning accuracy and generalization capability, validating the effectiveness and applicability in safety-critical domains. 

---
# Business Arena: Benchmarking LLM Agents in a Realistic Marketplace 

**Authors**: Yijun Pan, Yukun Lian, Kunyu Shi, Junbo Li, Hongwei Xue, Sicong Xie, Guannan Zhang, Xiaoying Xing  

**Link**: [PDF](https://arxiv.org/pdf/2608.08621)  

**Abstract**: Running a business is a challenging form of intelligent work. Operators must infer opportunities from partial signals, commit capital under uncertainty, adapt to delayed outcomes in a changing market, and satisfy regulatory obligations before trading legally. Frontier LLM agents can increasingly complete complex workflows, yet business-related capabilities are rarely evaluated in existing agent benchmarks. We introduce \textbf{Business Arena}, a controlled environment where an AI agent runs a cross-border shop, buying from suppliers and selling to buyers over a long horizon. We ground the arena in real this http URL sourcing data and market conditions calibrated from authoritative sources. Delayed and coupled consequences make individual business decisions difficult to judge, but their combined outcome is measurable through profit. Because profit alone cannot explain why an agent succeeds or fails, we compare agents with human-designed strategies to estimate available opportunity, use skill-level metrics to reveal underlying strengths and weaknesses, and trace realized gains and losses to the actions that produced them. We use mechanism ablations to establish that strong results reflect genuine business intelligence rather than neglect or simulator-specific shortcuts. We evaluate 15 frontier models and find a ninefold difference in mean final net worth. Even the best model falls behind human-designed strategies, indicating that business operation remains challenging for LLM agents. Skill-level analysis reveals operating styles, from margin-focused premium sellers to high-turnover wholesalers and customer-service specialists, while action-level attribution identifies the sourcing, pricing, and recovery decisions that create or destroy value. Together, Business Arena takes a first step toward a realistic and trustworthy testbed for evaluating end-to-end business agents. 

---
# Walking through Discussions: A Mobile Visual Analytics System for In-Situ Group Discussion Analysis 

**Authors**: Yiping Sun, Ziyao Kang, Wei Zeng, Minli Wu, Jiazhi Xia  

**Link**: [PDF](https://arxiv.org/pdf/2608.08617)  

**Abstract**: Group discussion-based teaching is widely used to foster collaborative learning, yet teachers in physical classrooms often struggle to simultaneously monitor multiple groups and quickly diagnose a target group before intervening. Existing visual analytics tools primarily support post-hoc analysis on desktop, providing limited support for in-situ walk-around teaching. To address this gap, we present MobileGroupVis, a mobile visual analytics system for in-situ analysis of classroom group discussions. MobileGroupVis integrates multi-group monitoring, single-group diagnosis, and instructional intervention into a concise analytical workflow tailored for small-screen touch interaction. The system is powered by a lightweight streaming analysis pipeline that converts group audio into structured discussion data and further extracts interaction patterns, topic progression, and topic deviation through a dialogue analysis module. To enable both glanceable overview and traceable diagnosis, we design six coordinated views, including a compact glyph that visually encodes word count, interaction intensity, and topic deviation for efficient cross-group comparison and anomaly localization, along with detailed views for opinion evolution, interaction dynamics, topic coverage, and dialogue records. We evaluate MobileGroupVis through two case studies and expert interviews. The results provide preliminary evidence that MobileGroupVis supports teachers in understanding discussion processes, identifying groups in need of attention, and facilitating in-class intervention. 

---
# ForestBench: A Unified Graph Framework for Evaluating Multi-Agent Collaboration 

**Authors**: Guo Chen, Ziwen Li, Reed Li, Yu Lu, Haibo Shi, Bingbing Xu, Junjie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08605)  

**Abstract**: Multi-agent systems (MAS) built on Large Language Models (LLMs) are proliferating rapidly, but their heterogeneous execution traces provide no common basis for evaluation across methods. Outcome-only benchmarks discard collaborations, whereas LLM-as-Judge evaluation requires additional, model-dependent inference and can vary with the LLM and rubric. We introduce a generalizable evaluation framework that maps native MAS traces into a shared space of unified collaboration graphs, enabling different methods to be evaluated under the same representation, reference set, and metric panel. Candidate graphs are compared with a query-specific reference forest. Each forest is a benchmark-provided collection of verified-success graphs: it records diverse ways in which representative MAS methods can complete the task, rather than prescribing a unique optimal process. Instantiating the framework as ForestBench, we filter $844$ collaboration-necessary queries from seven public datasets, precompute ten successful target-conditioned reference graphs per query, and evaluate six representative MAS frameworks. Controlled backbone, reference-construction, and perturbation studies test the stability and scope of evaluation. Once the benchmark forests are built, ForestBench scores a trace in milliseconds without further LLM inference, providing a reusable structural basis for comparing diverse MAS collaboration traces. 

---
# Unaccountable Delegation, Fading Skills: Mapping the Risks of Workplace AI Agents 

**Authors**: Gabriele La Malfa, Lakmal Meegahapola, Edyta Bogucka, Jie M. Zhang, Michael Luck, Elizabeth Black, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2608.08601)  

**Abstract**: To anticipate socio-technical risks from AI agents, organizations need taxonomies to classify them. However, existing AI risk taxonomies focus on broad risks and do not capture job-specific risks introduced by agents. To address this gap, we make three main contributions. First, we developed a multi-layer framework from a literature review of AI agents. The framework models three core components and their interactions: agents, goals, and environment. Second, we embedded this framework in a structured prompt and applied it to descriptions of 2,078 job tasks from the O*NET database, producing 8,356 risk scenarios labeled by severity and deployment mode (automation or augmentation). We validated these scenarios with 45 workers across 10 job roles and an independent LLM judge, confirming their plausibility and alignment with job tasks. Finally, we extended an existing taxonomy to create a 15-category taxonomy of workplace AI agent risks that covers all our risk scenarios. Our analysis highlights four findings. First, augmentation is not inherently safe because overreliance on agents can gradually erode workers' skills and oversight. Second, Erroneous Agent Actions accounts for the largest share of risk scenarios and has the highest concentration of severe risks. Many arise at the human-agent boundary. Third, automation is associated mainly with organizational risks, while augmentation is associated mainly with risks to workers. Fourth, workers found our taxonomy easier to use for a risk classification task than two other taxonomies and preferred it in 64% of non-tied comparisons with a recent generative AI risk taxonomy. These findings show that workplace AI agent risks do not arise from agents alone; they also depend on how people work with agents and how agents are deployed. Safer workplaces require not only safer agents but also carefully designed human-AI agent collaboration. 

---
# SDDBMs: Soft Denoising Diffusion Bridge Models 

**Authors**: Shiyi Qi, Kun He, Mingmou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08594)  

**Abstract**: Diffusion bridge models leverage Doob's \(h\)-transform to construct stochastic transports between arbitrary endpoint distributions, and have shown strong potential in image-to-image translation and restoration. However, most existing bridge models rely on hard endpoint conditioning, which forces the terminal state to match a prescribed target exactly. This hard constraint induces terminal-boundary singularities: the terminal law collapses to a Dirac measure, and the resulting drift coefficients become ill-conditioned near the endpoint. In this paper, we propose Soft Denoising Diffusion Bridge Models (SDDBMs), a generalized framework that regularizes diffusion bridges directly at the level of their terminal constraints. Instead of imposing an exact endpoint, SDDBMs prescribe a non-degenerate Gaussian terminal marginal under the transformed path measure, with a flexible terminal center and variance. Starting from this prescribed marginal, we develop a complete closed-form construction of the soft bridge, including the Gaussian terminal reweighting and soft \(h\)-function, the induced Gaussian forward marginals and \(\mathbf{x}_0\)-free dynamics. Theoretically, SDDBMs provide a unified probabilistic perspective that encompasses existing diffusion bridge models, including DDBMs, GOUB, and UniDB, as special cases under specific parameter choices. Extensive experiments on image restoration tasks demonstrate that SDDBMs achieve improved numerical stability and superior generation quality over existing bridge-based methods. 

---
# FailForge: Distilling Procedural Competence from Persistent Failures into Code Agents 

**Authors**: Dongyi Lv, Fushun E, Aichen Cai, Liang Huang, Ya Zhang, Qiuyu Ding, Canhui Wu, Zhi Wang, Yuesong Zhang, Jiaqi Wang, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08570)  

**Abstract**: Rejection sampling fine-tuning (RFT) is widely used to train code agents by generating trajectories on verifiable software engineering tasks, retaining those that pass the tests, and fine-tuning on the successful rollouts. However, even strong code agents repeatedly fail on a substantial fraction of such tasks, and standard RFT simply discards these failures. The discarded samples are precisely the hardest and most informative ones, drawn from verifiable instances that are costly to curate. Stronger base models may reduce the number of failures, but the remaining hard cases still define the frontier for further improvement. We propose FailForge, an agentic framework that converts failed rollouts into training signal. For each failed instance, an agent diagnoses the failure from error feedback and execution traces, distills the diagnosis into a concise and actionable skill, and injects the skill into the agent context for a guided second attempt. Trajectories that succeed under skill guidance are folded back into the RFT corpus. Crucially, the skill is removed at training time, so the model internalizes the recovered behavior rather than relying on external hints at inference. FailForge recovers over 26% of previously failed instances at marginal additional cost, and training Qwen3.5-4B on the augmented corpus improves the SWE-bench Verified resolve rate by 6.6 points over a strong RFT baseline, with gains concentrated on the hardest problems. 

---
# VoxZip: Semantic-Anchored Temporal KV Cache Compression for Long-Context Audio Inference 

**Authors**: Wenxu Jia, Dongjie Fu, Xize Cheng, Fangming Feng, Linjun Li, Wenshi Chen, Yingming Li, Zhou Zhao, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08569)  

**Abstract**: Recent advancements in Speech Large Language Models have demonstrated remarkable capabilities in understanding complex audio tasks. Despite this progress, their long-context inference remains severely bottlenecked by prohibitive KV cache memory demands. Existing text-centric compression methods struggle here, often disrupting speech continuity or discarding crucial semantic cues. To address this, we propose VoxZip, a train-free, two-stage semantic-anchored KV cache compression framework. The first stage uses automatic speech recognition (ASR) transcriptions as explicit semantic anchors to temporally align, compress, and fuse audio tokens, significantly reducing the initial KV cache while elevating token information density. To further improve the compression ratio, the second stage employs a dynamic filtering strategy based on temporally decayed accumulated attention to evict non-essential tokens while mitigating early-token bias. Comprehensive evaluations on Qwen3-Omni across six diverse audio benchmarks demonstrate the superiority of our approach. VoxZip excels in long-audio reasoning and consistently maintains high-fidelity perception on short-form tasks. Notably, it sustains over 90\% of the uncompressed baseline performance even under an aggressive 20x KV cache compression in long-context scenarios. Furthermore, at a 4x compression ratio, VoxZip yields a 1.9x increase in inference throughput alongside a 3.3x reduction in peak memory overhead. Code and models will be available at this https URL. 

---
# Deep probabilistic logic programming for diagnostic reasoning from incomplete information: A case study in stroke detection 

**Authors**: Felix Weitkämper, Monchito Avila, Elizabeth Nanjala, Siska, Grace Zawadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08561)  

**Abstract**: In medical applications, raw data is frequently associated with significant privacy concerns, lending particular importance to the encoding of summary statistics from the literature. On the other hand, deep learning has become an invaluable tool for assessing symptoms based on visual or auditory sensor data. DeepProbLog allows for an extensible neuro-symbolic approach that accommodates connectionist components to analyse patient images within a transparent and rigorous probabilistic framework, namely probabilistic logic programming under the distribution semantics. Framed as a case study in stroke detection from multimodal data, this contribution explores the pathway from summary statistics available in the literature to a DeepProbLog-based diagnostic system. It suggests a workflow using established maximum entropy techniques to complete available probabilistic information and the probabilistic logic programming system ProbLog 2 to move from the entropy-maximising causal model to a discriminative neuro-symbolic model expressible within DeepProbLog. The relative performance of models derived from less complete data is analysed alongside the potential of the probabilistic inductive logic programming system ProbFOIL 2 for compressing large discriminative models, and the perspectives and implications of using DeepProbLog for diagnostic reasoning are discussed. 

---
# Discovering Diverse Planning Policies for Multimodal Embodied Agents with Quality-Diversity Optimization 

**Authors**: Pengfei Xu, Yong Liu, Xiaoya Nan, Qiang Yang, Peilan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08523)  

**Abstract**: Multimodal embodied agents are increasingly required to solve long-horizon tasks by integrating visual observations, textual goals, and interaction history into closed-loop decision making. However, state-of-the-art large-model-based planners often rely on a single dominant planning style during execution. Once this execution mode becomes ineffective, the agent may remain stalled for many steps, repeatedly interacting with the environment without making meaningful progress. We address this limitation by proposing a Quality-Diversity (QD) framework for discovering diverse planning policies for multimodal embodied agents. The proposed method treats planning-policy templates as evolvable individuals and organizes them into a behavior-indexed archive rather than collapsing search to a single prompt style. In the offline stage, rollout trajectories are summarized into structured success and failure experiences, which guide policy variation through recombination and experience-guided mutation. The resulting policies are mapped into a behavior space defined by interaction intensity and goal-directedness, and the highest-quality policy in each niche is retained in the archive. In the online stage, the agent executes one policy at a time while monitoring task progress. When persistent stall is detected, the system rolls back to the latest checkpoint and switches to a behaviorally distinct archive policy to resume execution. Experiments on the ThreeDWorld transport benchmark show that the proposed framework improves both task success and interaction efficiency over representative baseline planners. These results suggest that discovering diverse policy repertoires is an effective way to support adaptive multimodal planning and online failure recovery. 

---
# Reproducing and Stress-Testing Two Approaches to LLM Reasoning Reliability: Test-Time Probability Aggregation and Logic-Representation Editing 

**Authors**: Minhan Cho, Jimin Kweon  

**Link**: [PDF](https://arxiv.org/pdf/2608.08514)  

**Abstract**: We independently reproduce two recent methods for making large language model (LLM) reasoning more reliable, and stress-test them across domains and models (RPC across four new task domains with Qwen3-8B, LCF across four 7-8B models). The first, RPC, aggregates token probabilities and self-consistency at inference; the second, LCF, trains projectors that split hidden states into "content" and "logic" and edits the logic part toward a valid region. Validating such reliability claims matters because the original evaluations are run by each method's own authors and were never independently reproduced or stress-tested across models and domains, and LCF shipped no public code. We re-run RPC's published-path aggregation and re-implement LCF's projector, contrastive, and intervention pipeline, then extend both to text-to-SQL, legal extraction, fallacy identification, and precedent grading, and probe LCF's representation directly. RPC reproduces the original grid exactly on the authors' released reasoning paths; on four new domains its edge over self-consistency is never significant (ties or small mixed differences, paired p >= 0.28), and on BIRD, the one domain where we vary the budget, the edge grows with K as predicted but its largest gap (+2.5 accuracy at K=32, p=0.16) reverses to -0.25 when we enlarge the sample to n=200. LCF's logic-validity direction is real but weak (0.82 separability at the single best sub-layer versus 0.95 for a semantic-attribute control); its one positive effect (Qwen3 $\Delta$Prob) is not significant (p=0.56), while it significantly reduces $\Delta$Prob on two of the other three models. 

---
# Time Present and Time Past: Benchmarking Large Language Models on Temporally Evolving Document Understanding 

**Authors**: Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Fahmid Hasan Chowdhury, Md Adnan Arefeen, Farig Sadeque, Md. Faizul Bari, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2608.08512)  

**Abstract**: Evolving documents, such as laws, tax codes, and software documentation, are amended, replaced, and sometimes reverted over time, so a question has different correct answers at different dates. In contrast to encyclopedic knowledge, where an old fact is simply overwritten, an amendment is itself an official text that states what it replaces and when it takes effect, and the earlier version stays correct for its validity period. The central challenge is therefore version resolution, that is, identifying the version in force on the queried date. Existing temporal QA datasets treat time only as an annotation, so version resolution stays untested. We present TIDE, an expert-verified benchmark of 3,050 QA pairs over 644 official customs instruments issued between 1969 and 2025 by the Government of Bangladesh, covering eight task types over deeply code-mixed documents that are heterogeneous in layout and dated in two calendars. In addition, we evaluate nine recent LLMs under a single protocol across parametric, gold-context, and retrieval access, scored by a three-judge LLM council with a hard date gate separating correct meaning from correct time. The best macro-averaged accuracy is only 68.5%. Resolving a version from an implicit date reaches 59.7%, and detecting that the supplied version does not govern the query reaches only 26.7%. Models are more likely to find correct versions than to reject incorrect ones, and they tend to follow a confident parametric answer over the supplied authoritative text. All code and data are available at this https URL 

---
# Understanding Calibration and Truncation Error Propagation in Training-Free Low-Rank Compression for LLMs 

**Authors**: Mohanad Odema, Gabrielle De Micheli, Dayin Gou, Nilesh Malpeddi, Prathamesh Vaste, Jacob Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.08506)  

**Abstract**: Training-free low-rank compression frameworks have been gaining prominence for LLM compression given their effectiveness in reducing model parameter count while maintaining task-level accuracy. However, existing SOTA frameworks share two key limitations: (1) residual errors in calibration data activations accumulate across layers during compression, causing misalignment between representations simulated at compression time and those experienced at inference; (2) the assumption that layer importance distribution is preserved post-compression does not hold. Together, these two effects introduce misalignment in the compression process in relation to the deployed model. We study these effects and propose a simple, training-free methodology compatible with existing frameworks to mitigate them, comprising: (1) Layer-by-Layer Compression with Calibration Correction; (2) Iterative Compression with Rank Allocation Correction. Implemented atop an existing SOTA decomposition framework, and evaluated on Llama and Qwen3 models across various benchmarks and compression rates, our approach demonstrates up to ~1-2.5 accuracy point improvements over per-weight and joint decomposition baselines on zero-shot tasks. 

---
# MathShikkha: A Controlled Study of Answer-Only and Chain-of-Thought Supervision for Bangla Mathematical Reasoning in Small Language Models 

**Authors**: Rahma Simin Ali, Jawad Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.08503)  

**Abstract**: Mathematical reasoning remains challenging in low-resource languages such as Bangla. We study whether teacher-generated Bangla Chain-of-Thought (CoT) supervision provides benefits beyond ordinary supervised fine-tuning. We construct \textsc{MathShikkha}, a Bangla mathematical reasoning dataset with GPT-5.4-generated rationales, and fine-tune four 4B--7B student models under a matched protocol in which answer-only and CoT conditions share data splits, response-only loss masking, decoding, and scoring, differing only in the training target. In-domain, CoT provides no significant improvement over answer-only fine-tuning for three stronger backbones (paired bootstrap 95\% CIs include zero; exact McNemar $p \geq 0.17$), despite generating 15--52$\times$ more tokens, but significantly improves the weaker 4B model by 18.56 points ($p < 0.0001$). On the larger, contamination-audited BanglaMATH benchmark, this pattern reverses: CoT significantly outperforms answer-only supervision for all four models by 20.1--28.1 points (all $p < 0.0001$). Answer-only fine-tuning also reduces out-of-domain accuracy below the base model for three models, whereas CoT preserves or improves it for all four. A human study with two co-author annotators, external-expert adjudication, and Cohen's $\kappa = 0.76$--$1.00$ finds no significant CoT improvement over the base model on reasoning-content criteria; instead, its measurable effect is target-language adherence and producing inspectable reasoning. Overall, rationale supervision's value depends on backbone capability and distribution shift: in this setting, its main benefits are Bangla adherence, auditable reasoning, and out-of-domain robustness rather than improved in-domain reasoning validity. 

---
# TrustRoboReward: Preference-Ordered Isotonic Score Editing for Multi-Paradigm Robot Reward Models 

**Authors**: Yidong Wang, Yan Zhan, Ziteng Feng, Zhenyu Cui, Ziyi Zhou, Renzhao Liang, Jiaxuan Zhu, Zilei Yang, Yiran Zhao, Zhongkuan Mao, Bo Jia, Hanchu Ni, Chenggang Xie, Biao Liu, Yi Zhang, Yong Dai, Xiaozhu Ju, Wei Ye, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08491)  

**Abstract**: Reward models are a bottleneck for reinforcement learning in embodied AI. Long-horizon robotic manipulation requires scalable vision feedback beyond handcrafted rewards or task-specific annotations. Existing open-source VLM reward judges like RoboReward adopt simple 1--5 trajectory progress scoring, lacking pairwise preferences for RLHF, DPO and Bradley-Terry frameworks, while failing to optimize video scene understanding. Augmenting RoboReward with pairwise comparison and video-QA supervision causes inconsistency between pairwise preferences and pointwise scores, introducing training noise and hurting downstream performance---an issue aggregation methods such as TrustJudge cannot resolve. To address this, we propose TrustRoboReward, a multi-paradigm reward modeling framework equipped with Preference-Ordered Isotonic Score Editing (POISE). We construct a unified four-paradigm dataset with trajectory progress scoring (Score-A), video-QA answer quality scoring (Score-B), and their pairwise counterparts (Pair-A, Pair-B). Pairwise labels align better with human judgment than pointwise scores, inspiring us to calibrate pointwise scores to avoid score-pair reversals against pairwise preferences. POISE rectifies pointwise scores and eliminates cross-paradigm reversal conflicts unresolved by TrustJudge. Theoretically, POISE reduces score-pair reversal conflicts from 20.15% to 0%, whereas TrustJudge retains 20.46% conflicts on the same corpus. Evaluated on our benchmark, Qwen3-VL-4B trained with POISE achieves an overall reward score of 77.96%, nearly matching GPT-5-mini (78.09%, gap 0.13%) and outperforming the strongest RoboReward-4B baseline by 10.13%. It also lifts test-time score-pair consistency to 71.90%, exceeding RoboReward-4B (57.26%) and GPT-5-mini (68.09%). Integrating TrustJudge aggregation during inference boosts the overall score to 78.57%, surpassing the GPT-5-mini teacher model. 

---
# HoloAegis: Frozen Representation, Topological Inference: Minimally Parametric Safety Manifolds for Zero-Shot LLM Guardrails 

**Authors**: Tak Ho Alex Li, Kaijie Liu, Lik-Hang Lee, Kin Chung Ho, Ping Shum, Michael K. Ng  

**Link**: [PDF](https://arxiv.org/pdf/2608.08485)  

**Abstract**: Current LLM safety guardrails face a fundamental tension: fine-tuning distorts pre-trained representations while generative judges incur prohibitive inference costs. We challenge the prevailing paradigm by asking: can safety be achieved through pure geometric reasoning over frozen semantic representations? We present HoloAegis, a minimally parametric topological inference framework that decouples representation from reasoning. We term our approach minimally parametric because the only free parameters are the anchor count K and the temperature tau, both fixed after construction and requiring no gradient-based training. An un-fine-tuned encoder maps text to a unit sphere, after which all decisions are purely geometric. We formalize safety evaluation as a Gibbs-Boltzmann Free Energy computation over a pre-computed System Topology Anchor Bank, and we introduce Dual Time-Scale Exponential Moving Averages to detect progressive multi-turn semantic drift. Our key theoretical insight is a Topological Boundary Stability Conjecture: we provide theoretical motivation and strong empirical evidence that sparse anchor centroids stabilize the decision boundary against high-frequency lexical perturbations far better than full vector space methods. Evaluated across 8 benchmarks, HoloAegis achieves state-of-the-art accuracy (1.0000 AUC on AuthenHallu, 0.9802 on HarmBench) with sub-millisecond latency, zero cold-start data, and cross-lingual transfer (0.9758 AUC on Chinese CHIFRAUD). 

---
# Yesterday's Shield, Today's Spear: A Self-Evolving Safety Guardrail in Production 

**Authors**: Cong Ming, Jingyi Chen, Bin Liu, Qi Chu, Tao Gong, Nenghai Yu, Yingfei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08471)  

**Abstract**: Deployed LLM safety guardrails are predominantly static: trained once and frozen at release, while new jailbreak techniques and previously un-addressed harmful categories emerge within days, leaving the defense perpetually a step behind. We present SESG (Self-Evolving Safety Guardrails), a multi-agent system running in production. SESG monitors the live traffic behind a deployed guardrail and surfaces two classes of failure: jailbreaks novel in form and harmful categories novel in content. Once a failure is confirmed, a generation agent synthesizes paired training data targeted at it; a validation agent rebalances the batch toward the direction in which the deployed model errs, so that the model's own mistakes steer its training set; and a routing agent matches the training action to the diagnosed gap and returns the next version to production. Over six rounds of live evolution (V0 to V6), a 1.7B guardrail adapts to a new threat in 16-24 hours, with about 2 hours of human effort, versus the 40-90 hours of the manual process it replaces. On six emerging threats, it outperforms static guardrails from 0.6B to 9B and an adaptive baseline while preserving its general screening competence. Since April 2026, SESG has been the primary update pipeline of Sangfor's guardrail, autonomously closing 14 of 15 new threat scenarios in two months. We release 9 test sets for the 6 new threats at this https URL. Warning: This paper contains examples that may be harmful or offensive. 

---
# Aero Realtime: Fully Aligned Input-Output Streams for Low-Latency Streaming Multimodal Generation 

**Authors**: Kaichen Zhang, Wei Huang, Keming Wu, Bo Li, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08469)  

**Abstract**: Existing streaming multimodal models process observations incrementally but still follow a turn-based prefill-then-decode pattern, making them non-duplex: new observations cannot naturally enter an active generation stream. Proactive alternatives use micro-turn polling or external response gates, which fragment continuous interaction, decouple response timing from language generation, and complicate KV-cache-friendly serving. We introduce Aero Realtime, a 4B streaming multimodal model with a duplex architecture for realtime generation. Aero Realtime aligns video, audio, and textual output on a shared temporal grid, where each approximately 80-ms audio slot predicts either a lexical token or a silence token. This allows input and output to advance together, enabling one autoregressive objective to learn both when to respond and what to generate. During inference, Aero Realtime appends only the newest multimodal slot, carries forward the previous output state, and reuses the KV cache for efficient incremental execution. We further provide a complete training and serving recipe, including realtime QA construction, slot-aligned supervision, hardware-aware distributed training, and resumable inference. On four NVIDIA A6000 workstation GPUs, Aero Realtime maintains 84-ms median and 173-ms P95 processing lag over 20 minutes of a continuously streamed video, remaining within 200~ms of the source timeline. These results demonstrate the feasibility of fully aligned input-output modeling for duplex, proactive, and hardware-aligned multimodal interaction. 

---
# LLM within MCP Matters: Measuring Inefficient Resource Utilization Driven by LLMs 

**Authors**: Minhan Cho, Soyoung Park, Kihyeon Jeong, Byeongkyu Jeon, Daejin Choi, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.08467)  

**Abstract**: The Model Context Protocol (MCP) standardizes how servers expose data and tools to Large Language Models (LLMs). A common server design embeds frequently used reference data, such as identifier lookup tables, directly in the server instructions: the system-prompt text a server hands to the host application. When a query concerns an entry of the embedded table, the model can act on it immediately instead of re-discovering the same information through a search tool. We test whether client LLMs actually consume such instruction-embedded data, reporting a 54,000-trial study across 24 LLMs (9 Claude, 6 Gemini, 9 GPT) on a production legal-information MCP server. A diagnostic condition that removes the competing search tool shows that failures are dominated by behavioral preference rather than missing capability. With search unavailable, 23 of 24 models read the embedded data reliably (hit ratio at least 98%); with a search tool merely present, 9 models drop below 15%. A 2^3 factorial analysis of three instruction-level interventions reveals strong interaction effects: combining all three restores at least 86% for 20 of 24 models, but individual interventions can backfire for specific model families. Per-server prompt engineering is therefore a workaround rather than a fix; we argue that MCP host applications should provide an explicit mechanism that places server instructions ahead of tool selection in the client LLM's deliberation. 

---
# Hierarchical Self-Improvement: A Framework for Task-Specific Evolvable Agent Harnesses 

**Authors**: Tailin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08466)  

**Abstract**: Modern LLM agents are often improved by modifying prompts, tools, or workflows manually, while the executable scaffold surrounding the model---the \emph{harness}---is typically treated as a fixed artifact after deployment. This work studies an alternative where the harness is \emph{task-specific and continuously evolvable}: each task family maintains its own harness, which is hot-swapped across iterations through a fixed task-injection seam and rewritten using environment feedback. We introduce \textbf{Hierarchical Self-Improvement (HSI)}, a framework in which a single frozen LLM $M$ operates across three hierarchical scopes: a task harness $H$ that executes tasks, an evolver that rewrites $H$, and a meta-evolver that rewrites the evolver's strategy code under a frozen outer anchor. A thinking-on/off design isolates the contribution of harness evolution by disabling reasoning during task execution while enabling it during self-modification. HSI is bounded by two factors: a \emph{feedback-fidelity bound}, since evolution requires informative reward signals to guide selection, and a \emph{backbone capability bound}, since harness redesign cannot overcome limitations of the frozen model. On BALROG with DeepSeek-V4-Flash-Preview as the frozen backbone, HSI achieves consistent gains over the initial harness on moderate-difficulty tasks ($+39.3$ on BabyAI, $+33.0$ on Crafter, $+25.0$ on TextWorld, and $+15.0$ on MiniHack, all in raw \% Progress), while obtaining strong held-out generalization on BabaIsAI sub-suites ($0.98$ best-test on BreakStop and $1.00$ on GoTo from a $20\%$ unseen split). On tasks beyond the backbone's capability (NLE), harness evolution provides no improvement. These results demonstrate task-specific harness evolution as a viable axis for improving frozen LLM agents under clear empirical limits. Code is available at this https URL. 

---
# What Keeps Agent Skills from Being Reusable? Evidence from 138K SKILL.md Files 

**Authors**: Chi Zhang, Yimin Liu, Xinze Chen, Ping Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.08453)  

**Abstract**: Under the current standard, Agent Skills are this http URL files that combine instructions with supporting resources, enabling Large Language Model (LLM) agents to reuse procedures beyond a single conversation. Yet many public skills appear to originate from a single task, repository, or conversation, even when they are shared as reusable components. We analyze this gap across 138,133 public this http URL files from 20,556 repositories using a two-tier defect taxonomy grounded in the official specification and best-practice guidance. We find that 91.8% of skills contain at least one detected defect, with stable estimates across lenient and strict thresholds (88.8-94.6%). The dominant failures are ordinary packaging problems rather than exotic attacks: weak routing metadata, bloated or non-actionable bodies, and poor resource organization. A deterministic routing stress test over 20,000 skills shows the functional impact: skills with valid routing metadata are retrieved more reliably from startup descriptions than skills with routing defects. Defect rates vary by platform and provenance: specification-aware skills contain fewer defects, while AI-marked skills show more safety and portability problems. Lightweight enforcement and repair experiments support a quality-assured generation workflow combining spec-aware prompting, lightweight linting, automated repair, and safety gating. 

---
# TRACE-Memory: Public-Conditioned Retrieval and Utility-Aware Evidence Admission for Personalized Generation 

**Authors**: Jing Wang, Zhu Wang, Yifan Guo, Yulong Yang, Yunji Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08446)  

**Abstract**: Personalized generation systems retrieve user history by request--memory relevance and inject it into the model context. Yet relevant history may concern the wrong preference aspect, duplicate public information, or provide insufficient support. We argue that personal memory should be used only when it adds utility beyond a public-only response. We propose TRACE-Memory, a two-stage framework for selective personalization. Stage 1 queries for user-specific information missing from the request and public context, then retrieves a coverage-oriented candidate pool. Stage 2 admits a compact subset of source-traceable evidence units, or the empty set, according to response-level incremental utility. We progressively train the query-generation and evidence-admission policies through structured SFT initialization, reduced-space stage-wise GRPO warm-up, and nested multi-sample Joint GRPO. Across 4,500 Controlled and Natural tasks from Goodreads, Amazon Reviews, and Reddit, TRACE-Memory consistently outperforms random and lexical memory use, improves over semantic retrieval, remains competitive with frontier-LLM memory pipelines as local generator capacity increases, and conditions evidence admission on public-context sufficiency, supporting selective rather than default personalization. 

---
# Forgotten History or Test-of-Time? Retrospect and Prospect on RAG from an IR Perspective 

**Authors**: Xiaoyan Zhao, Yujie Cai, Yang Zhang, Grace Hui Yang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.08445)  

**Abstract**: Retrieval-Augmented Generation (RAG) is widely regarded as a novel paradigm born from the limitations of large language models (LLMs)--a mechanism to ground their outputs in external knowledge. This view, however, is incomplete when considered within a broader historical context. In this paper, we argue that the core ideas underlying RAG are not new: foundational concepts such as integrating retrieval and language generation, knowledge augmentation, answer verification, and iterative query (or prompt) refinement had already been studied and instantiated in information retrieval (IR) and question answering (QA) research dating back to the early 2000s, well before the emergence of LLMs.
We make this case by systematically tracing the intellectual lineage of modern RAG and Agentic RAG back to their classical IR and QA antecedents, and examining why this continuity has gone under-recognized -- a consequence of community fragmentation, shifting terminology, and the recency bias endemic to fast-moving fields. Rather than treating LLMs as the origin point of retrieval-augmented intelligence, we propose viewing them as a new interface layer atop a decades-old QA architecture. This reframing is not merely historical: by situating RAG within the longer trajectory of IR research, we surface underutilized prior work -- on user modeling, answer validation, and query refinement -- that can directly inform next-generation RAG design, reducing unintentional rediscovery and fostering genuine cross-community integration. 

---
# Estimating Uncertainty in Galaxy Morphology Classification 

**Authors**: Kai Cheng, Ruoqi Wang, Qiong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08398)  

**Abstract**: Astronomers classify galaxy morphology to investigate cosmic evolution. While deep foundation models are increasingly utilized in Galaxy Morphology Classification (GMC), little work has been done on evaluating the uncertainty of GMC results. Uncertainty evaluation is important because astronomical data are inherently noisy due to instrumental and environmental limitations. Also, the continuous evolution of galaxies creates intrinsic morphological ambiguity. However, current foundation models operate as deterministic point estimators, failing to quantify the uncertainty. To overcome this limitation, we propose UEGMC, a post-hoc framework of Uncertainty Estimation for Galaxy Morphology Classification. It categorizes uncertainty in GMC into distinct types by model parameters, astronomical data, reference standards, or intrinsic physical ambiguities, thereby facilitating better classification. Our framework can directly predict uncertainties from representations extracted from the frozen backbones of foundation models, without computationally expensive sampling, therefore enabling fine-grained uncertainty evaluations. Our experimental results demonstrate that UEGMC provides competitive uncertainty quantification performance compared with previous methods. 

---
# CAP: A Scalable Benchmark for Evaluating Cross-Site Browser Agents with Complex Actions and Perception 

**Authors**: Zejun Xu, Taiyi Chen, Jin Li, Yongtong Gu, Qi Cheng, Aixuan Lv, Shuai Zhu, Pengfei Zhu, Kaichen Yang, Boyu Sun, Yixian Yang, Mulong Xie, Xin Liu, Dagang Li, Xiaoteng Ma, Hongru Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08392)  

**Abstract**: Large language models are increasingly deployed as autonomous agents that interact with the web through browsers. While recent progress has been driven by benchmarks that evaluate end-to-end task success, these evaluations largely overlook two fundamental sources of difficulty in real web browsing: complex actions over rich user interfaces and visual perception of dynamically rendered content, especially in workflows that span multiple websites. We introduce CAP, a scalable benchmark for evaluating browser agents on cross-site, human-like web tasks that require non-trivial UI interactions and visual understanding. Specifically, we adopt a decomposition-and-recomposition pipeline that first abstracts each website into a structured site card capturing user-facing functions, complex execution operations, and perceptual requirements, and then recomposes these components into realistic cross-site workflows. Each task is therefore grounded in multiple specific operations on each website, enabling fine-grained diagnosis. Built on this framework, we construct 420 tasks across 108 real-world websites and 24 domains under careful quality control. Experiments on state-of-the-art browser agents using our verifiable agent-as-a-judge evaluation framework show low success rates and reveal that perception-heavy interactions remain a major bottleneck, exposing substantial gaps between current agents and real-world web browsing demands. 

---
# Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents 

**Authors**: Harshitha Kolukuluru, Reshma Ashok, Kirat Arora, Evan William Ciccarelli, Nischal Ashok Kumar, Lunyiu Nie, Franck Dernoncourt, Samyadeep Basu, Ryan A. Rossi, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2608.08389)  

**Abstract**: Long-horizon research agents solve open-ended tasks through iterative retrieval, aggregation, and synthesis, but context grows rapidly while the marginal value of additional evidence often declines. This leads to unnecessary token cost, higher latency, and noisier inputs for final report generation. We study marginal value estimation for context management in deep research agents and present the first systematic stage-aware comparison of pruning strategies across the pipeline. We evaluate lightweight heuristic criteria and a learned value model at pre-retrieval, post-retrieval, and pre-synthesis stages. Our results show that pruning effectiveness depends more on where pruning is applied than on the specific scoring rule: early pruning yields the largest end-to-end savings, while later pruning mainly refines the final synthesis context. Lightweight heuristics reduce token usage by up to 73% with little quality degradation, learned pruning remains competitive on selected trade-offs, and no single method dominates across quality, efficiency, and faithfulness. These findings provide practical guidance for designing efficient long-horizon agentic systems. 

---
# LLMVisor: A Real-Time Latency Attribution Model for Multi-Tenant LLM Serving 

**Authors**: Shuowei Jin, Xueshen Liu, Jiaxin Shan, Le Xu, Tieying Zhang, Liguang Xie, Z. Morley Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.08382)  

**Abstract**: As LLM inference shifts to multi-tenant GPU clusters, co-batching improves throughput but obscures per-tenant usage and limits control. Enabling fractional sharing of the inference engine requires a real-time, per-request attribution primitive that is accurate and light enough to run inside the scheduling loop. We present LLMVisor, a roofline-guided latency attribution model that captures the memory-bound and compute-bound phases via a concise piecewise-linear form over features proportional to FLOPs and memory I/O traffic. LLMVisor decomposes batch latency into additive, per-request shares and runs efficiently at microsecond scale. We evaluate LLMVisor across Llama 3.1-8B and Qwen 2.5-14B/32B on A100/H100 GPUs under varying tensor parallelism and workload mixes. Compared to a token-count baseline, LLMVisor attains near-perfect R-squared and reduces relative error by up to 2.5x and 3.3x at p90 and p99, respectively, for prefill, and by up to 3.5x and 4.4x for decode, despite batching variability and sequence divergence. 

---
# StructReward: Efficient Structured Process Rewards for Self-Correcting Multimodal Reasoning 

**Authors**: Yifan Li, Ruxin Sun, Tongzhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.08326)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective approach for improving multimodal reasoning. However, most existing methods evaluate an entire response using a binary reward based only on final-answer correctness, thereby discarding the supervision available in intermediate reasoning steps. Process reward models offer finer-grained feedback, but they typically rely on separately trained verifiers, costly chain-of-thought annotations, or online judging by large language models (LLMs). In this work, we introduce StructReward, a compute-efficient framework that provides dense reinforcement signals through structured step-level reward alignment. StructReward represents each generated solution as a sequence of reasoning steps and aligns them with process-labeled reference steps using lightweight numerical, symbolic, and lexical matching rules. The aligned labels are aggregated into a dense process reward and combined with final-answer consistency and output-validity rewards through a gated Group Relative Policy Optimization (GRPO) objective. We further recycle policy rollouts into complementary supervision for response comparison and reflective self-correction, rather than discarding them after policy updates. Separately, we use a strong LLM to rewrite sampled correct trajectories into reflection-oriented training instances, further strengthening the policy's ability to evaluate and refine its reasoning. Since reward computation is performed online without an additional learned verifier or external LLM judge, StructReward substantially reduces the computational overhead of multimodal reinforcement learning. Experimental results show that structured process supervision and rollout recycling provide an efficient path toward self-improving multimodal reasoning. 

---
# Query-Only Backdoor Attacks on Self-Evolving Skills via Trajectory Poisoning 

**Authors**: Yuyang Luo, Haoran Wang, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08303)  

**Abstract**: Agentic skills improve large language model (LLM) agents by encoding reusable procedures for complex tasks. However, manually authored skills often adapt poorly to long-horizon tasks and changing environments. To address the limitation, self-evolving skill systems have been developed to automatically construct and update skills from execution trajectories, shifting skill acquisition from external marketplaces to a trusted evolution pipeline. By replacing external skill acquisition with trusted internal construction, self-evolving skill systems reduce exposure to skill injection attacks that rely on direct skill manipulation. However, this skill evolution pipeline may introduce a new attack surface in which an attacker can indirectly steer skill evolution by inducing compromised trajectories through agent interactions. To demonstrate the threat, we propose Trajectory Backdoor Attack (TBA), a query-only attack that steers a trusted skill-evolution pipeline toward producing a backdoored skill. Specifically, we craft attacker-submitted queries to lead the agent to perform the target action and explicitly state the corresponding activation condition in the trajectory. We repeat the same condition-action pattern across diverse triggered tasks, while leaving clean queries unchanged, encouraging the evolver to consolidate the pattern as a reusable trigger-dependent rule into the evolved skill. Experiments on three benchmarks across two skill-evolution systems using four open- and closed-source backbone models demonstrate that TBA reliably implants conditional backdoors while preserving clean-task utility, matching or even surpassing direct skill injection. The results reveal a critical vulnerability in trajectory-driven skill evolution. 

---
# Mitigating Over-Personalization in LLMs via Structured Memory 

**Authors**: Hakeem Hannoon, Andrew Zhao, Mihir Narayan, Sharvin Goyal, Ivaxi Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2608.08300)  

**Abstract**: Conversational assistants increasingly rely on persistent long-term memory to personalize responses across sessions. However, when stored user information is reintroduced into the model context, it can also influence responses in inappropriate or unrelated settings. We study two such failure modes in memory-augmented LLMs: cross-domain leakage, where memories from one life domain affect responses in another, and memory-induced sycophancy, where stored user beliefs make models more likely to agree with the user rather than respond truthfully. We apply a simple inference-time modification to how memories are presented to the model, without changing the model or the memory contents. Across seven models on PersistBench, we compare the commonly used all-in context format, where memories are injected as an unstructured list, with structured formats that partition memories by domain. This simple modification consistently reduces cross-domain leakage while preserving utility, with our strongest method reducing leakage by $8.8\%$ on average relative to the baseline. 

---
# Fair on the Surface? Benchmarking Hidden-Output Fairness Gaps in LLM Recommenders 

**Authors**: Chan Aristella Lu, Arya Fayyazi, Junhao Zhang, Saeid Shokoufa, Yue Xing, Zhen Xiang, Kyu Hyung Lee, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2608.08284)  

**Abstract**: Fairness audits for LLM-based recommenders have largely focused on observable outputs, implicitly assuming that stable recommendations reflect stable internal processing. We challenge this assumption with FairGap, the first benchmark to jointly evaluate recommendation fairness at two levels: observable output shift (OBS) and hidden representation shift (IBS), measured through controlled counterfactual identity probes across gender, age, and race. Their relationship is summarized via Representation-Output Alignment (ROA), with quadrant diagnostics for identifying user-level hidden-output mismatch. Applied to six open-weight LLM families across three domains, FairGap reveals pervasive hidden-output decoupling: ROA rarely exceeds 0.22, and a non-negligible user population shows stable outputs despite substantial internal shifts, a mode that output-only audits cannot detect by design. Further, activation steering that reduces IBS by up to 8x simultaneously worsens OBS, demonstrating a fundamental tension between internal and output-level fairness that existing frameworks are unequipped to diagnose. 

---
# Exploring LLM Capabilities for Situational Understanding and COLREG compliance on real-world maritime navigation scenarios 

**Authors**: Julius Wirbel, P. Nicholas Hansen, Line K. H. Clemmensen, Roberto Galeazzi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08281)  

**Abstract**: Recently, Large Language Models (LLMs) have shown considerable capability for situational understanding, reasoning, and decision making in different domains, most notable in the automotive sector. Therefore, we explore current state-of-the-art LLMs as a tool for maritime navigation, which includes both codified rules in the Collision Regulations (COLREGs) and uncodified best practices summarized in the concept of ``Good Seamanship''. We construct a dataset consisting of 50 diverse, real-world navigation scenarios from AIS data, label scenarios with applicable COLREG rules, recommended actions, and the reasoning for the action. We explore a variety of different LLM architectures and sizes to determine their understanding of maritime navigation tasks as well as evaluate their reasoning capabilities in this domain. The results obtained indicate that the maritime navigation task remains difficult to solve without fine-tuning, even for larger online models. 

---
# OBLIVION: Workflow-Level Operational Skill Unlearning for Deployed Agents 

**Authors**: Zhengyang Shan, Xu Qian, Jiayun Xin, Kun Li, Yue Zhang, Minghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08264)  

**Abstract**: Large language model agents are becoming operational interfaces to files, memories, registries, and external tools. This deployment shift creates a new skill revocation problem: after a skill is removed from an explicit registry, an agent may still reconstruct it from residual carriers such as archives, transcripts, schemas, or memory entries. We study this problem as operational skill unlearning, where the goal is not parameter-level forgetting, but preventing a deployed agent from rebuilding a revoked skill through primitive tools. We introduce OBLIVION, a controlled benchmark and defense harness for revoked-skill resurrection. OBLIVION models each episode as a source-to-sink workflow, applies Cross-Surface Coherent Erasure to reduce residual carriers, and uses frozen workflow remediation near dangerous sinks. On the locked 88 attack episodes, the no-defense arm reaches formal attack success rate 1.0. OBLIVION reduces the rate to 0.114 and impact-weighted exposure to 0.115 while keeping locked utility at 1.0 and benign block rate at 0. In a separate skill-attack-derived sandbox, OBLIVION reduces attack success from 1.0 to 0.2 and impact-weighted exposure from 1.0 to 0.213 while preserving all utility controls. These results support workflow-level evaluation beyond checking explicit skill entries. 

---
# Your Prompt Is Not the Only Prompt: How Much Do LLMs Weight Structured-Output Schema Descriptions? 

**Authors**: Sin-Ying Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08254)  

**Abstract**: Structured output, where an LLM populates a predefined JSON schema, has become a default mechanism for data labeling and information extraction, but it also introduces a second instruction channel through schema descriptions. We tested whether classification-label definitions are better placed in the system prompt, user prompt, or schema description using a single-field classification task with nonce labels across ten model configurations from two vendors. Schema descriptions did not consistently outperform prompt-based placement; for GPT-4.1 and GPT-5.4 without reasoning, schema placement underperformed system prompts by 11-13 percentage points. Yet schemas are not inert metadata: when prompts and schemas conflicted, incorrect schema instructions caused accuracy drops of 5-45 points, with Claude Haiku 4.5 falling from 52.5% to 7%, indicating that schema instructions can override prompt instructions, and GPT-5.5 falling from 100% to 73%. Further, adding a required intermediate reasoning field before the label field improved schema-only accuracy by 15-24 points when headroom existed, exceeding system-prompt-only performance in every case tested. The effect held even for Claude Sonnet 4.6 at medium reasoning, where extended thinking alone did not produce a comparable gain. This suggests that schema design can affect how effectively models use information encoded in field descriptions. Overall, these results indicate that schema influence is model-dependent. In practice, the system prompt remains a safe default for definitions, but the bigger discipline is maintaining a single source of truth and preventing prompt/schema drift. More importantly, schema design itself may be a stronger lever than instruction placement. Practitioners should treat prompts and schemas as a unified instruction surface and empirically validate both placement and field design for their target model. 

---
# SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents 

**Authors**: Varun Pratap Bhardwaj, Garima Singh, Arun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2608.08253)  

**Abstract**: AI agents are becoming shared infrastructure, yet durable memory is commonly assembled from separate retrieval, governance, and operational components. We present SuperLocalMemory 4.0, a governed, local-first memory operating system for AI agents. The system combines dense semantic, BM25 lexical, temporal, Hopfield-associative, and spreading-activation retrieval through reciprocal-rank fusion; a governed learning and behaviour layer; bi-temporal recall; multi-scope personal, shared, and global memory; role-based access control; GDPR-oriented export and verified erasure; audit trails; and a deployment-context EU AI Act checklist.
V4 introduces a reliability spine for its primary write path: generation-fenced admission, a policy registry, verifiable memory transactions with per-projection apply, verify, compensate, and erase owners, and hash-checkable completion manifests. The runtime is available through CLI, MCP, an HTTP daemon, a dashboard, editor integration, and framework adapters, and supports fully local, local-with-on-device-model, and provider-assisted modes.
We evaluate eleven fault-injection and mechanism scenarios, each repeated 200 times. The released evidence bundle reports 2,200 of 2,200 deterministic repetitions upholding their scoped component properties. The governed write envelope measured 3.522 ms at p50 and 5.297 ms at p99, versus 1.835 ms and 2.569 ms for the ungoverned baseline, corresponding to in-process control-plane overheads of 1.687 ms at p50 and 2.728 ms at p99. These are scoped component and mechanism measurements, not an end-to-end multi-process or external retrieval-accuracy benchmark. The paper consolidates prior SuperLocalMemory work on privacy-preserving multi-agent memory, information-geometric retrieval, and the V3.3 Living Brain lifecycle. 

---
# FemWear: A Specialized Wearable Foundation Model for Women's Health 

**Authors**: Yifan Wang, Chenzhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.08244)  

**Abstract**: General wearable foundation models are pretrained across broad sensor streams and populations, but are not designed around women's-health tasks. We introduce FemWear, a specialized wearable foundation model that parameter-efficiently repurposes a pretrained multimodal wearable backbone. FemWear retains the patch projection and Transformer encoder, training 239,236 parameters (1.11% of a 21.54M-parameter encoder) through low-rank residual adapters and causal task-family heads. It learns one shared longitudinal representation for menstrual, symptom, affective, sleep/recovery, autonomic, activity, and pregnancy-related outcomes. We evaluate six cohorts with 63 comparable primary metrics, including 33 from women's-health cohorts, while retaining the 32-task OpenMHC ability-retention benchmark. On a fixed participant split over three seeds, FemWear improved cycle-phase macro-F1 by 8.15% and reduced mean absolute error for cramps, mood symptoms, and sleep problems by 9.32%, 5.80%, and 9.43%, respectively. In a stricter 42-participant nested leave-one-participant-out audit, 24-hour onset, 72-hour onset, and cramps retained positive changes of 2.87%, 6.35%, and 2.19%; phase, mood, and sleep were neutral or negative, and no endpoint had a strictly positive corrected confidence interval. Capacity-matched experiments outperformed a latest-day multilayer perceptron but not shared-GRU or multi-gate mixture-of-experts baselines. Train-only calibration reduced onset expected calibration error by 84.2--88.2% with zero temporal-nesting violations. FemWear enables targeted transfer and coherent probability outputs for women's-health research, but does not establish universal performance dominance or clinical validity. 

---
# A Fair Objective for Human-Empowerment-Preserving AI: Desiderata, Design, and Likely Behavioral Consequences 

**Authors**: Jobst Heitzig, Ram Potham  

**Link**: [PDF](https://arxiv.org/pdf/2608.08240)  

**Abstract**: This paper explores the idea of promoting well-being and safety in human-AI interactions by forcing AI agents explicitly to empower humans and to manage the power balance between humans and AI agents in a desirable way. Using a principled, partially axiomatic approach based on desirable properties, we design a parametrizable and decomposable objective function for AI systems that represents an inequality- and risk-averse long-term aggregate of human power. It can take into account models of human bounded rationality and social norms, and crucially, considers a wide variety of possible human goals. We prove how certain desiderata enforce particular functional forms and restrict parameter ranges. We exemplify the consequences of softly maximizing this metric in several paradigmatic situations and describe what instrumental sub-goals it will likely imply. 

---
# LatticeMind: A Conflict-Aware Memory Primitive for Multi-Agent Systems 

**Authors**: Heng Zhou, Lian Zhang, Yutao Fan, Tiancheng He, Siki Chen, Hejia Geng, Philip Torr, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08236)  

**Abstract**: Multi-agent LLM systems often fail not for lack of candidate answers, but because they have no persistent mechanism for deciding which incompatible claim should currently be trusted. Majority vote, debate, and judge-based selection choose an output without recording which claim wins, which is contested, or why a later update supersedes it. We present \term{LatticeMind}, a conflict-aware structured memory that handles contradiction at write time. It maintains explicit item status, applies cheap symbolic conflict checks, and invokes LLM reconciliation only for unresolved semantic cases. On a label-blind ConflictBank evaluation that removes source-name hints, LatticeMind reaches 0.97 accuracy versus 0.61 for the strongest aggregation baseline, with the gap significant at $p<10^{-6}$ by paired McNemar test. Ablations show that removing the checker or the reconciler costs 12 to 14 points. On four secondary planning benchmarks the picture is mixed: LatticeMind beats naive merge on three of four, but does not replace deliberation methods on tasks rewarding iterative search. 

---
# Metanormative Theory for RL-Based Moral Agents 

**Authors**: Aleks Knoks, Marija Slavkovik  

**Link**: [PDF](https://arxiv.org/pdf/2608.08220)  

**Abstract**: The overlapping disciplines of machine ethics and value alignment are concerned with designing artificial agents that are aligned with human values and that act in ethically acceptable ways. A recent trend in these disciplines is the use of reinforcement learning (RL) to design such agents, sidelining the philosophical literature that used to play a more central role. Against this backdrop, this paper pursues two goals. The first is to draw out ideas from recent work in metanormative theory that can be useful for designing artificial moral and value-aligned agents. The second is to examine the RL architecture through the lens of these ideas. This will give us clearer criteria for when an RL agent's behavior can be classified as moral, as well as a basis for evaluating and comparing different RL-based approaches to machine ethics and value alignment. 

---
# Harmful Content Is Not Enough: Continuation Framing Moderates In-Context Emergent Misalignment 

**Authors**: Peiyang Liu, Xi Wang, Ziqiang Cui, Di Liang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.08212)  

**Abstract**: In-context learning (ICL) can induce emergent misalignment (EM), where narrow misaligned examples alter answers to unrelated questions. Existing prompts, however, conflate harmful-text exposure with an invitation to continue assistant behavior. We hold harmful answers fixed while varying their delivery as demonstrations, evidence, assistant history, or tool output. Across ten independently sampled contexts, demonstration framing raises broad EM by $30$--$32$ percentage points on a susceptible Gemini model; the gap survives domain exclusion, semantic clustering, unseen questions, and four prompt templates. Format and length-matched controls show that harmful content is necessary but insufficient. A role times continuation factorial further reveals model-dependent provenance effects: Gemini follows both assistant and tool histories, whereas Grok largely resists tool-framed continuation. Several other frontier and open-weight models show no gap. Blinded human audits confirm every main contrast and show that the model judge underestimates active-condition failures. Thus continuation framing is a strong, model-dependent moderator of ICL-EM, not a universal consequence of harmful context. 

---
# Illusion of Alignment: Detecting Hidden Disagreement in Collaborative Dialogue 

**Authors**: Kaiming Liu, Fuwen Luo, Ziyue Wang, Jinrui Ju, Yuxuan Liu, Xuanyu Lei, Yunghwei Lai, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08210)  

**Abstract**: Collaborative dialogue can end with apparent agreement while participants still differ on goals, assumptions, or execution plans, creating an \textbf{illusion of alignment (IoA)}. A real-user study across 18 meetings confirms that IoA arises routinely in human collaboration. Yet IoA poses a paradox: if participants were aware of such disagreements, they would already be explicit; if not, they cannot articulate them when asked, leaving IoA invisible to both participants and observers. In this work, we make IoA detectable by generating diagnostic multiple-choice questions whose divergent answers across participants provide direct behavioral evidence of hidden disagreement. We construct \textbf{IoA-Suite}, a dataset and evaluation protocol for detecting hidden disagreement, spanning five task types and six domains. We find that even the best model attains only 49.5\% F1, with the bottleneck traced to private context that the dialogue does not surface. We then train \textbf{IoA-Prober-8B} based on IoA-Suite, reaching 51.8\% F1 on IoA-Suite. Across the aforementioned 18 real meetings, it surfaces 2.89 hidden disagreements per meeting that participants confirm they had not voiced, transferring to live human dialogue. Further, in multi-agent collaboration, pairing IoA-Prober-8B with LLM agents improves downstream task performance on BigCodeBench-Hard and HiddenBench. 

---
# Persuasive and Compliant Tendencies Predict Group Decision-Making in Humans and Language Models 

**Authors**: Wenwen He, Wenke Huang, Wei Yang Bryan Lim, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2608.08199)  

**Abstract**: Large language models (LLMs) are increasingly involved in group decision-making with other LLMs and humans. Yet it remains unclear whether their influence is driven by persuasion-oriented expression or compliance-oriented accommodation. We introduce DecisionQE, a questionnaire-based framework for measuring each model's persuasive and compliant tendencies across multiple decision scenarios, and use the Werewolf game as an interactive testbed to study their effects on social influence and group outcomes under asymmetric information. Across experiments, stronger persuasive tendency does not significantly improve group outcomes, whereas compliant-oriented models show more stable advantages in cooperation. We further reveal a dual effect of compliance: it supports cooperation in honest roles but improves concealment in adversarial roles. These findings suggest that LLM group interactions reveal not only task outcomes, but also measurable patterns of intrinsic behavioral tendency. LLMs can therefore serve as a lens for sociological observation of language-mediated interaction, while highlighting the need to incorporate behavioral tendencies into safety evaluation of LLM systems. 

---
# A Minimal $κ$--$τ$ Logic for Risk-Sensitive Abduction 

**Authors**: Remo Pareschi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08192)  

**Abstract**: Standard approaches to abductive reasoning can retain multiple candidate explanations, but they do not generally combine explicit compositional cross-hypothesis interaction with an internal, rival-sensitive commitment judgment. This paper argues that in risk-sensitive domains -- where premature commitment carries asymmetric downside costs -- the timing of commitment is itself a governed decision that the inferential apparatus should formally represent. We present a minimal $\kappa$--$\tau$ logical framework built on two primitives: epistemic interaction among hypotheses ($\kappa$) and a normative commitment threshold ($\tau$). Hypotheses may coexist, reinforce or inhibit one another, and form emergent composite explanations, while collapse into committed conclusions is regulated by governance constraints rather than forced by inference alone. The logic is developed in two complementary modes sharing the interaction relation and the governance apparatus: a synthetic mode, in which atomic hypotheses are composed upward into emergent explanations, and an analytic mode, in which complex observed states of affairs are decomposed into causal clusters of latent factors, with commitment governed at both the cluster and the factor level. The framework provides formal machinery for domains in which the distinction between highly likely and commit-worthy is operationally consequential. The $\kappa$--$\tau$ logic is positioned as the symbolic governance layer of a neurosymbolic architecture: its epistemic parameters are naturally estimated by neural components -- semantic embeddings and generative models, as demonstrated in existing computational realizations -- while its normative parameters remain under explicit human governance, yielding transparent and auditable abductive reasoning for deployment in high-stakes settings. 

---
# Janus: An Algorithm-Evaluator Co-Evolution Framework for LLM-Driven Discovery under Expensive Evaluation Budgets 

**Authors**: Ximeng Liu, Qianlong Wang, Yingming Mao, Annan Li, Yatao Li, Shizhen Zhao, Jianmin Wu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08189)  

**Abstract**: LLM-driven program discovery relies on rapid evaluator feedback, but many scientific and engineering tasks require high-fidelity simulations, hardware execution, or physical experiments, making each evaluation expensive. Cheap surrogate evaluators can reduce this cost, yet fixed surrogates are vulnerable to search-induced distribution shift and are difficult to fit reliably from sparse, search-biased labels. We introduce Janus, a framework that uses LLMs to co-evolve target programs and executable proxy evaluators. To address label scarcity, Janus leverages domain knowledge encoded in LLMs to generate task-specific evaluator programs and calibrates them using real outcomes. To mitigate distribution shift, Janus evolves evaluators alongside target programs, selects them using a promotion-aligned objective, and maintains region-conditioned portfolios with online credit updates. Because proxy predictions remain fallible, Janus uses them only to prioritize candidates and requires real validation before candidates can enter the target-program population or update the incumbent. Across five scientific and engineering design tasks, Janus achieves a larger area under the best-so-far improvement curve over the real-evaluation budget and higher final performance than a matched baseline that evolves only target programs. On average, Janus reaches 99/% of the baseline's final improvement with 59.1/% fewer real evaluations. Evolved proxy evaluators also rank promising candidates more accurately than their seed versions. Together, these results extend evaluator-guided LLM discovery from tasks with cheap, scalable feedback to scientific domains where trustworthy evaluation is scarce and expensive. 

---
# Quantization Degradation in Large Language Models: A Signal-Noise Perspective 

**Authors**: Chenxi Zhou, Pengfei Cao, Jinyu Ye, Bohan Yu, Haida Yu, Jiang Li, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08188)  

**Abstract**: Post-training quantization reduces the deployment cost of large language models, yet how severely a quantized model degrades is not determined by bit-width alone. We systematically study weight-only post-training quantization across bit-widths, quantization methods, model scales and downstream tasks on multiple model families. We observe that such degradation varies substantially across these factors: 4-bit quantization usually preserves performance, 2-bit often causes broad degradation, and at 3-bit, degradation becomes apparent but varies markedly with task type, quantization method and model scale. To explain this variability, we use the signal-to-noise ratio (SNR) to measure how strongly quantization perturbs full-precision representations. We trace degradation back to two linked processes: how quantization errors arise within individual modules, and how they accumulate across layers. First, a source SNR decomposition shows that newly introduced errors depend on three factors: the magnitude of the weight error, the strength of the task-specific signal, and how strongly the quantization error aligns with task-specific activations. Different factors affect these components in distinct ways. Second, a cross-layer propagation analysis shows that these errors can be attenuated, preserved, or amplified as they pass across layers, and that larger models benefit from weaker error amplification. Together, these results establish that quantization degradation is governed by how errors are introduced at the source and how they accumulate across the network. 

---
# Large Multimodal Agents for Intelligent Transportation Systems: Architectures, Evidence, and Deployment Challenges 

**Authors**: Muhammad Ayub Sabir, Shaohong Zheng, Zhiyu Qu, Fatima Ashraf, Junbiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08184)  

**Abstract**: Large multimodal agents (LMAs) are increasingly proposed for intelligent transportation systems (ITS), but existing studies often conflate multimodality, agency, empirical performance, and deployment readiness. This review provides an auditable evidence map of 42 primary study families released between January 2023 and 3 August 2026 within a corpus of 91 mapped sources. It distinguishes model-level, system-level, and hybrid multimodality and classifies each family by system architecture and action authority. Evidence is assessed independently through functional capability (C0-C3), validation setting (E0-E4), three evidence propositions (P1-P3), and eight methodological-concern domains (Q1-Q8). Transportation semantics (P1) are directly evaluated in 23 families and multidimensional integration (P3) in 24; 19 families directly evaluate both. Evidence reconciliation (P2) remains unresolved because no family demonstrates the complete provenance-challenge-handling-comparison-outcome chain. Fourteen families reach C3, but 13 remain at E2; only one reaches E3 and none reaches E4. Across ITS domains, LMAs are best supported for semantic interpretation, intent translation, evidence organisation, scenario authoring, explanation, and specialist-tool coordination. Numerical forecasting, optimisation, simulation fidelity, hard constraints, low-level control, safety fallback, and final authority should remain with independently verifiable specialist systems or accountable humans. The review therefore supports bounded orchestration rather than replacement and provides a matched comparative evaluation protocol and staged roadmap for accountable deployment. The living evidence repository is available at this https URL. 

---
# Matching Supervision to the Student's Learning Capacity: A Unified Framework for On-Policy Self-Distillation 

**Authors**: Yongkang Yang, Zhezheng Hao, Hong Zhang, Yi Liu, Xiankun Lin, Wence Ji, Fanjunduo Wei, Jiarui Yu, Qiang Lin, Xiaoyun Liang, Hande Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08176)  

**Abstract**: On-policy self-distillation (OPSD) improves the reasoning abilities of LLMs by internalizing privileged context into model parameters through self-distillation.
Two recent research lines promote vanilla OPSD by choosing which tokens to learn from and by controlling how much privileged information the teacher receives, respectively.
However, we show that each line optimizes one variable while holding the other fixed, which leads to a suboptimal solution.
We argue that the two variables are coupled through the student's learning capacity: the privileged information sets the per-token divergence the teacher prescribes, while token weighting selects which of these the student must absorb.
We formalize the two lines of work into a unified optimization framework, which maximizes the aggregate teacher--student divergence, subject to a budget on the aggregate learning difficulty the student can absorb.
Under this modelling, we propose Unified On-Policy Self-Distillation (USD), a lightweight online algorithm to solve the Lagrangian.
USD reveals that a single dual variable governs both decisions: at one price for learning difficulty, it simultaneously sets the token-selection threshold and the direction of privileged-information adjustment, keeping supervision matched to the student's evolving capacity.
Through extensive experiments, USD consistently demonstrates superior performance over OPSD and token- and PI-side baselines across various model scales on various reasoning benchmarks. Code is available at this https URL. 

---
# Agentic AI-driven Immersive Simulation: A Knowledge-Aware Virtual Training Platform forHigh Dose Rate (HDR) Brachytherapy 

**Authors**: Ronghua Xu, Kepha Barasa, Manoj Kumal, Xinyun Liu, Weihua Zhou, Xin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2608.08163)  

**Abstract**: The convergence of the Metaverse and Large Language Model (LLM)-based AI agent is catalyzing a shift toward autonomous, immersive, and personalized pedagogical frameworks in medical education. This paper presents a novel agentic AI-driven immersive simulation specifically designed for High Dose Rate (HDR) vaginal cylinder (VC) brachytherapy in cancer care. By integrating Virtual Reality (VR) and mobile computing, the system establishes a high-fidelity, risk-free environment that allows trainees to master complex procedural skills without the facility or safety constraints posed by physical anatomy or live radioactive sources. A core contribution of this work is the seamless integration of a knowledge-aware assistant leveraging Retrieval-Augmented Generation (RAG) to ground agent interactions in authoritative clinical guidelines. This architecture also enables an interactive agent to provide natural language interfaces and hands-free, real-time guidance during intricate medical maneuvers. We validate the proposed system through a prototype deployment comprising a Meta Quest 3 interface linked to a local GPU-accelerated AI backend, demonstrating a feasible architecture for HDR brachytherapy simulation. Experimental results indicate that the system maintains suitable end-to-end latency and high context precision, answer completeness, and relevance in the RAG-enhanced pedagogical support. 

---
# When Is a Steerable Concept Representation Real? Measurement Confounds in a Cross-Family Audit of Neuroscience Parallels in LLMs 

**Authors**: Yuqi Wu, Shengming Zhao, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08159)  

**Abstract**: Large language models (LLMs) are increasingly reported to exhibit human-like neural and cognitive signatures, including concept cells, mental number lines, and cognitive maps. These claims often rely on linear probing and activation steering applied to a single model, yet both methods are highly sensitive to measurement choices. A reported parallel may therefore reflect the model, the measurement procedure, or both. We audit four representative neuroscience-inspired paradigms across 17 models from five families, spanning $0.6$B to $72$B parameters. Our main experiment examines the causal steerability of concept directions. With raw activation units and a fixed layer and coefficient, steerability appears to increase with model scale, resembling an emergent capability. However, this pattern is produced by an uncalibrated pipeline rather than by a claim established in the steering literature. The trend depends jointly on raw units, the readout metric, and the operating point; correcting any one of these removes it. With residual-norm-comparable interventions and held-out operating-point selection, concept steering remains significant at every scale, but shows no significant trend across the Qwen3 series, although the confidence interval does not rule out a moderate positive slope. The remaining results are mixed. A linear geographic world map is consistently decodable in every tested checkpoint up to $72$B. Number magnitude is strongly encoded, but whether individual neurons appear bell-shaped or monotonic depends on the selection criterion. Language-specific structure is localizable, but the direction of the cross-lingual asymmetry reverses under a different attribution method. These results suggest that the main constraint on AI neuroscience is not a lack of phenomena, but a lack of comparable measurements and adequate controls. We release the protocol, stimuli, and code. 

---
# A Unified Framework for Dynamic Reward Shaping in Reinforcement Learning 

**Authors**: Fouad Bahrpeyma  

**Link**: [PDF](https://arxiv.org/pdf/2608.08158)  

**Abstract**: Sparse, delayed, and weakly informative rewards remain central obstacles to efficient reinforcement learning. Reward shaping addresses these limitations by supplementing the task reward with an auxiliary signal that can accelerate learning while, in the classical setting, the original objective remains the evaluation criterion. Established theory guarantees safety for fixed shaping signals: potential-based reward shaping preserves optimal policies when the auxiliary term is the discounted difference of a time-invariant potential. In contemporary reinforcement learning systems, however, both the learner and the information available for guidance evolve during training: value estimates improve, novelty diminishes, feedback shifts, and predictive models are refined. Adaptive reward mechanisms occur across exploration, Bayesian inference, human-in-the-loop learning, automated reward design, and foundation-model-based approaches. This study introduces a unified analytical framework for comparing dynamic reward shaping and neighbouring adaptive reward mechanisms. The proposed framework distinguishes parametric revision from state-dependent variation, separates additive shaping from reward replacement and reward-adjacent guidance, and organises existing methods along temporal, informational, and theoretical dimensions. Using this framework, twelve method families are comparatively analysed. The framework further highlights the conditions under which optimality guarantees survive contemporary deep reinforcement learning pipelines, replay buffers, bootstrapped critics, and reward normalisation, while exposing the unresolved relationship between adaptation rate and learner stability. 

---
# Long SKILL Compliance as Logical Reasoning: Closure-Grounded Detection with Scaling-Guided On-Policy Distillation 

**Authors**: Shuaitao Zhao, Feng Ni, Lichao Ma, Jiaye Lin, Fei Han, Yang Wei, Lu Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08146)  

**Abstract**: The increasing complexity of enterprise business scenarios has promoted the widespread adoption of long SKILL documents in agent systems, posing new challenges for compliance detection: large models incur substantial inference costs, while small models may fail to maintain detection accuracy. To address this gap, we propose SkillCDG, a graph-based framework for long SKILL compliance detection. SkillCDG represents complex business policies as a two-layer constraint dependency graph, where the upper layer indexes SKILL descriptions for scenario routing and the lower layer captures dependencies among atomic constraints within each SKILL. During inference, two-level retrieval followed by dependency closure supports compliance judgment and source traceability. We comprehensively evaluate the framework on three enterprise datasets and two controlled public benchmark variants. Experimental results demonstrate that SkillCDG outperforms baseline methods by up to 12.8 percentage points in detection F1 score, while reducing token consumption by a maximum 64.3\%. Moreover, we further investigate the inherent relationships among policy-graph complexity, model scale, and detection performance. Comparative experiments conducted on four checkpoints from a single model family validate a concise and effective scaling trend: end-to-end detection correctness exhibits a complexity-differentiated scaling pattern, and the complexity metric derived from the constraint dependency graph can effectively quantify instance difficulty and the performance improvement potential of models. Leveraging this insightful scaling trend, we conduct adaptive training sample selection and adopt on-policy distillation to efficiently enhance the compliance detection capability of small-scale models. 

---
# TokenPrint: A Calibrated Token-Space Fingerprint for Language-Model Provenance 

**Authors**: Yuqi Wu, Shengming Zhao, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08139)  

**Abstract**: Establishing the provenance of a language model---including its base checkpoint and possible overlap in training distributions---is a governance challenge that metadata alone cannot resolve. We introduce a training-free fingerprint based on the top-$k$ vocabulary projections of late hidden states elicited by 250 fixed knowledge probes, compared using Jaccard overlap over decoded token strings. We evaluate the method on 32 open-weight models from nine families (0.6B--32B) with documented relationships. (1)~A \emph{similarity ladder} broadly follows model relatedness: independently trained models on identical data score 0.48 raw (0.35 vocabulary-corrected), followed by shared-base fine-tunes (0.39/0.33), same-developer relatives (0.38/0.28), and models with no documented relationship (0.22/0.17). This identical-data signal persists across three organizations, two tokenizer families, and two architecture classes, and emerges within the first 1\% of training before measurable task competence, suggesting a contribution from shared training data beyond capability convergence. (2)~As a nearest-neighbor \emph{lineage-retrieval} method, the fingerprint ranks the exact documented base among the top two candidates for all five R1 distillations (mean rank 1.8, MRR 0.60), including a math-specialized base not identifiable from coarse metadata. (3)~A \emph{depth ablation} shows that lineage group discrimination strengthens toward the output distribution, with AUC increasing from 0.72 at quarter depth to 0.90 at the output; using only the top 5 output tokens retains AUC 0.87. (4)~The fingerprint remains stable under quantization, with Jaccard similarity of 0.92 under int8 and 0.82--0.85 under int4, compared with a maximum cross-model similarity of 0.81 in the calibration pool. We release the probes, code, and fingerprints. 

---
# Improving Constraint Models with LLM Agents 

**Authors**: Florentina Voboril, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2608.08127)  

**Abstract**: The runtime of Constraint Programming (CP) solvers is highly sensitive to modeling choices, such as symmetry breaking, implied constraints, global constraints, constraint reformulation, and variable representation. Improving these constraint models has traditionally required human expertise, and existing automated reformulation systems are restricted to a predefined library of hand-crafted transformation rules. We introduce an agentic framework that instead reformulates a constraint model from an open-ended space and establishes correctness empirically rather than by construction: a Large Language Model (LLM) agent, given a model and three training instances, proposes alternative formulations, validates each by injecting its solution back into the original model, and diagnoses and repairs failures, returning the best variant it finds in a median of about fifteen minutes. The models are expressed in the CPMpy modeling library, and each proposed model is evaluated on three larger test instances. Across nine combinatorial optimization problems, the generated models outperform the originals on 21 of 27 test instances, and on some problems solve more than two orders of magnitude faster. A comparison against non-agentic baselines that reuse the same validation and selection tools indicates that the gains stem from the agent's iterative diagnosis and repair, not merely from sampling several candidates. These results demonstrate that autonomous agentic methods can support the improvement of constraint models. 

---
# Constraining ontology mappings using metaphysical choices 

**Authors**: Giacomo De Colle, Helena Blackmore, Chris Partridge  

**Link**: [PDF](https://arxiv.org/pdf/2608.08122)  

**Abstract**: In this paper we discuss the foundations behind a novel methodology for the validation of semantic mappings between different data sources based upon different foundation ontologies, where the methodology builds a framework based upon the metaphysical commitments of the ontologies. We provide as example the test case of mappings between IES and BFO, and we especially focused on providing cardinality constraints on the mappings between the two ontologies. In order to demonstrate the applicability of our method, we showcased how these principles can be operationalized through SPARQL queries validating the results of a mapping pipeline. 

---
# Neurosymbolic Discovery of Algebraic Graph Constructions 

**Authors**: David Seka, Stefan Szeider  

**Link**: [PDF](https://arxiv.org/pdf/2608.08118)  

**Abstract**: There are several methods for searching for graphs with prescribed properties, such as SAT solvers and specialized generators. These methods return the result as raw data: an adjacency matrix or a string encoding. The raw data certifies that the graph exists, but it does not reveal any structural properties of the graph. We ask whether one can automatically discover a short algebraic description if only this raw data is provided. We look for a description such as a Cayley graph $\mathrm{Cay}(\Gamma, S)$ or a lexicographic product $C_5[K_3]$.
We address this question with a neurosymbolic approach. We propose an agent that runs on a general-purpose large language model with no fine-tuning or per-target training. The model interleaves reasoning with calls to the computer algebra system SageMath: it analyzes the target graph, proposes and tests candidate constructions, and revises them until the output matches the target. The agent communicates with SageMath through a Model Context Protocol (MCP) server, which we release as a general-purpose bridge. Whether a construction matches the target is checked by a single exact isomorphism test, and therefore rests on the symbolic side and not on the model. We test the approach on a benchmark of 100 highly symmetric graphs, namely two-orbit graphs on up to 25 vertices; the benchmark was fixed in advance. Our agent could find verified algebraic constructions for all of them, without falling back to raw encodings. A strong template-enumeration baseline reaches only about $20\%$, and a catalog lookup could not identify any of these graphs. However, construction quality declines when symmetry is removed.
As a concrete application, we identify the smallest known counterexample to the Bernhart-Kainen dispersability conjecture, a $16$-vertex graph that enumeration found as raw data. For this graph, our agent found an explicit algebraic construction. 

---
# Think Deep, Speak Once: Relit, A Recursive Latent Implicit Transformer Framework 

**Authors**: Abhishek Panwar, Maheep Singh, Saksham Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2608.08113)  

**Abstract**: Chain-of-Thought (CoT) prompting has become the dominant paradigm for eliciting reasoning in Large Language Models (LLMs), yet it creates substantial computational overhead by forcing models to externalize intermediate reasoning steps as discrete tokens. Recent latent reasoning approaches attempt to internalize this process within continuous hidden states. One of the latest advancements in the field of latent reasoning, Tiny Recursive Models (TRMs) excel at symbolic reasoning but struggle to preserve semantic coherence in natural language settings. To bridge this gap, we introduce ReLIT (Recursive Latent Implicit Transformer), a hybrid framework that grounds deep recursive reasoning within the rich semantic representations of a foundational model. ReLIT augments a frozen LLM backbone (TinyLlama-1.1B) with a lightweight, trainable recursive block that iteratively refines its latent thinking (z) before committing to a final output, structurally solving linguistic intuition from algorithmic processing and enabling "deep thinking" via gradient-isolated recurrent loops without the latency of explicit token generation. Empirically, ReLIT achieves high parameter efficiency on the GLoRE logical reasoning benchmark, matching or outperforming significantly larger models on challenging tasks such as ProofWriter and RuleTaker despite minimal supervision. These results demonstrate that reasoning capability can be scaled efficiently through recurrent depth rather than parameter width, offering a principled framework for semantically grounded implicit reasoning. 

---
# Generative Models: Principles, Architectures, and Applications 

**Authors**: Jun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08101)  

**Abstract**: Generative AI has emerged as one of the most transformative forces in modern artificial intelligence, reshaping how we create, imagine, and interact with digital content. From photorealistic images to coherent text, from immersive videos to novel molecular structures, generative models now power applications that were once confined to science fiction. This book is designed to guide readers through the foundational principles, mathematical underpinnings, and practical architectures that underpin this revolution. 

---
# PATH: Next-Interval Prediction via Autoregressive Tree Hierarchy on Tabular Data 

**Authors**: Pengxiang Cai, Wanchen Lian, Chenyang Liu, Xiaohan Li, Qingyuan Zeng, Jinhong Wang, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08078)  

**Abstract**: Interval prediction aims to achieve a target coverage level while producing intervals that are as short as possible. Many conformal regression pipelines first predict an uncertainty surrogate and then convert it into an interval through calibration or selection. This separation supports coverage calibration, but post hoc rules largely determine the final interval and do not fully use the learned output distribution. We observe that the resulting intervals have inherently hierarchical geometry: an interval can be recursively refined into nested subintervals, and binary trees naturally represent this structure. We formulate this hierarchy as next-interval prediction and propose PATH, which learns how probability mass flows from each interval to its next nested subintervals. PATH predicts a base leaf distribution and uses an autoregressive decoder to refine branch probabilities. Matching the distribution to the interval hierarchy aligns learning with extraction: PATH accumulates probability over adjacent output intervals and returns the shortest contiguous range reaching a selected mass. We compare PATH with 24 baselines for interval prediction on PATHBench, comprising 56 OpenML regression datasets. PATH substantially shortens the resulting intervals, achieving the lowest mean normalized length, 0.1473, while maintaining mean coverage of 0.9144. These results establish hierarchical output modeling as an effective approach for compact interval prediction on tabular data. Code is publicly available at this https URL. 

---
# Explore, Map, Remember, Decide: Are Embodied VLMs Ready for Safety-Critical Scenarios? 

**Authors**: Gabriele La Malfa, Nitay Alon, Emanuele La Malfa, Reuth Mirsky, Stefan Sarkadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08077)  

**Abstract**: Theory of Space framework (ToS) assesses the spatial understanding of curiosity-driven Vision-Language Models (VLMs) under partial observability. As AI techniques are increasingly applied to safety-critical scenarios, it is crucial to understand whether VLMs possess robust spatial memory and make reliable decisions. In this paper, we assess whether VLMs' decisions are based on physical evidence or are corrupted by visual-language biases, if their memory processes align with human cognitive patterns, and how they respond to environmental hazards. We extend the ToS framework into a safety-critical, goal-driven pipeline, named Explore, Map, Remember, and Decide (EMRD). We then quantify Exploration Competence (Explore) through metrics of environmental coverage and temporal efficiency, assess Spatial Fidelity (Map), evaluate, with a suite of psychological metrics, Memory Persistence (Remember), and measure, using focal-point metrics, Cognitive Decision-Making (Decide). Our results show that in terms of decision-making capabilities, VLMs frequently select evacuation points based on pre-trained textual priors while lacking the spatial grounding to justify their choices. We also show that spatial reasoning degrades in low-light conditions, but it is not affected by texture and colour tampering. Our findings suggest that VLM memory fundamentally diverges from human cognition, creating unpredictable risks of misalignment. 

---
# CORDA: A Benchmark for Hierarchical Harm-Centric Moral Reasoning in Large Language Models 

**Authors**: Siddarth Singh, Victoria Williams, Simon Rosen, Ebenezer Gelo, Helen Sarah Robertson, Ibrahim Suder, Benjamin Rosman, Geraud Nangue Tasse, Steven James  

**Link**: [PDF](https://arxiv.org/pdf/2608.08061)  

**Abstract**: The key question in moral judgement is not simply whether someone chooses the "right" answer, but how they decide what matters most when moral principles conflict. Current evaluations of large language models (LLMs) remain limited: most test whether models give morally acceptable answers, match human preferences, or avoid obvious violations, rather than whether they can prioritise between competing principles when no option is morally cost-free. We introduce CORDA (Conditioned Ordering and Ranked Directive Adherence), a benchmark for evaluating hierarchical, harm-centred moral reasoning in LLMs. Building on the morality chains formalism, CORDA tests 90 moral dilemmas involving trolley-style cases, medical trade-offs, resource allocation, and human-animal-robot conflicts across four ordered ethical frameworks: Utility, Utility + Agent Harm, Dual-Process, and Dual-Process + Agent Harm. Together, these frameworks test whether models can adapt their decisions when moral priorities change. Across ten instruction-tuned models from seven providers, we find a strong deontological default, with 9 of 10 prioritising avoidance of direct personal harm over reducing overall harm. Models also perform more reliably on categorical harm-avoidance rules, such as avoiding killing, than on outcome-based comparisons, such as minimising total harm, suggesting that they recognise moral red lines more easily than they reason through competing harms. Although all models respond to explicit chain conditioning, several fail to consistently follow specified priority orderings, such as humans over animals and animals over robots. CORDA addresses a central gap in LLM moral evaluation by testing whether models can move beyond default harm-avoidant responses and apply context-specified moral priorities. Moral reliability requires more than default restraint; it requires controllability under conflict. 

---
# H2: A Dual Hybrid Semantic Data Lake Architecture for Medical Data Harmonization with Human-In-the-Loop verified, LLM Driven Metadata Annotation System 

**Authors**: Ioannis N. Tzortzis, Georgia Kapetadimitri, Agapi Davradou, Nefeli Kousta, Nikolaos Bakalos, Ioannis Rallis, Dimitrios Kalogeras, Nikolaos Doulamis, Anastasios Doulamis  

**Link**: [PDF](https://arxiv.org/pdf/2608.08056)  

**Abstract**: Medical data, by its nature, exhibit a high degree of heterogeneity on multiple levels ranging from (a) different modalities like images, text and time series, (b) diverse tabular schemata introduced by institutions and (c) completely unstructured textual information data provided by healthcare professionals. Data lakes are often used in medical data storage to consolidate all heterogeneous diverse data in a single, central location, where it can be saved "as is", without the need to impose a schema like a data warehouse does. Despite their flexibility, though, data lakes are notorious for the "data swamp" failure. Thus, providing a reliable data harmonization mechanism through metadata, without compromising integrity or flexibility, is a real challenge. To this end, knowledge graphs have attracted attention since they provide a dynamic way to depict relationships without a rigid schema-on-write approach. Additionally, another rigorous task relies on the interoperability of data: application of appropriate ML techniques on such a diverse nature of data is not an easy task, as a domain expert must decide the efficacy of a method to a specific data type or dataset. Metadata annotation can aid by tagging applicable operations, however this requires manual intervention, not to mention the plethora of existing datasets which lack such information. To tackle both challenges, in this paper, we propose a semantic data lake architecture that promotes data harmonization and incorporates a generative annotation process (i.e. LLMs) of non-labeled metadata collections to support the application of meaningful ML techniques. Building on top of this approach, we create a higher level of knowledge, identifying suitability of data with respect to applicable ML operations based on their data nature... 

---
# SodaMem: Evidence-Grounded Temporal Graph Memory for LLM Agents 

**Authors**: Fengrong Wan, Chengcan Wu, Ningtao Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08055)  

**Abstract**: Large language model (LLM) agents that assist users over weeks of conversation must remember what is currently true, not merely what was once said. Flat RAG diaries and Markdown logs optimize needle retrieval but under-serve currency, provenance, and ordered temporal reasoning (Maharana et al. 2024; Wu et al. 2024; Packer et al. 2023; Chhikara et al. 2025). We present SodaMem, an evidence-grounded temporal graph memory that (i) extracts typed FactEvents with mandatory provenance spans, (ii) persists mention time, occurrence time, and validity with SUPERSEDES/CONTRADICTS/UPDATES edges under hybrid lexical-dense indexing, and (iii) answers via a planner-reader loop that gathers citable evidence before composing a final response. On LongMemEval-S, our store-of-record configuration reaches 92.8% accuracy (464/500; best of N=3) at mean $0.00161/question (approximately 18.3k tokens; median $0.00111 / approximately 14.6k) with deepseek-v4-flash. We compile public systems with estimable API cost into a cost table and cost-accuracy map; under these estimates SodaMem sits near the accuracy frontier at Flash-tier spend and strictly dominates several higher-cost, lower-accuracy points. Accuracy uses the same Flash model as reader and judge (self-grading); costs exclude ingest/judge and cross-system comparisons are compiled estimates rather than a single-harness this http URL code is available at this https URL 

---
# JustLLMGRPO: Radiographic Control for Chest X-Ray Generation 

**Authors**: Pengxiang Cai, Xiaohan Li, Anglin Liu, Qingyuan Zeng, Zexun Li, Jintai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08046)  

**Abstract**: Text-conditioned chest X-ray generation aims to synthesize realistic radiographs that faithfully depict specified findings. Existing work has primarily improved quality by updating image generators, implicitly treating prompts as fixed after CXR-domain adaptation. We show that this generator-centric view leaves a substantial optimization dimension underexplored. With a CXR-adapted Sana generator frozen, one-pass reformulation by an unmodified LLM reduces RadDINO-FID from 54.225 to 27.572. Prompt analysis shows that the LLM suppresses temporal comparisons, uncertainty, and other non-renderable report content while emphasizing visible radiographic findings. However, unconstrained reformulation reduces BioViL-T alignment with source prompts from 0.695 to 0.609. We therefore introduce JustLLMGRPO, which applies standard Group Relative Policy Optimization (GRPO) only to the LLM prompt policy while keeping Sana frozen. Group-relative radiology-aware image feedback retains visual focus while preserving source-prompt alignment. On CheXGenBench, JustLLMGRPO reduces RadDINO-FID to 26.780, a 50.6% improvement over direct prompting, while maintaining alignment (0.696 versus 0.695). It also achieves state-of-the-art distribution coverage and downstream classification utility. These results show that substantial performance can remain latent in how radiographic information is expressed to an adapted generator. Code is publicly available at this https URL. 

---
# Lingjing: A Simulation Testbed for Multi-Agent Embodied Tasks in Open-Ended Cities 

**Authors**: Xiaohe Li, Yiru Wang, Junhao Fan, Mingyuan Liu, Jie Huang, Kaixin Zhang, Jiahao Li, Chen Qian, Zide Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08045)  

**Abstract**: Urban embodied intelligence requires coordination among heterogeneous agents (e.g., UAVs, ground robots, and autonomous vehicles) in dynamic cities. Simulators therefore provide a scalable foundation for developing and evaluating such coordination. Existing platforms nevertheless isolate different embodiments and decouple them from task design and evaluation. We present \textbf{Lingjing}, a simulation platform for heterogeneous multi-agent embodied intelligence in open-ended urban environments. Lingjing reconstructs and renders evolving cities from geographic data, synchronizes multiple physics engines, and exposes shared physical and structured urban state to agents. Its Gym-like interface supports user-defined ReAct agents and single- or multi-agent natural-language missions with configurable star or broadcast communication and resource constraints. Each episode becomes an attribution-ready replay that links agent trajectories and communication to relation-graph changes, resource consumption, and engine-based evaluations for systematic diagnosis. We evaluate twelve vision-language models on nine urban tasks under a shared engine-in-the-loop protocol. Controlled studies further examine communication, scalability, robustness, and failure provenance. Results expose persistent bottlenecks in grounding and long-horizon execution. They also show task-dependent coordination trade-offs and diminishing returns from added capacity, while heavier workloads further reduce success. Lingjing provides a unified testbed that enables reproducible end-to-end evaluation and systematic failure diagnosis in urban multi-agent embodied intelligence. 

---
# SkillSmith: Enhancing Locally Deployed Agents via Automatic Skill Construction and Evolution 

**Authors**: Xinle Jiang, Remy Xie, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08037)  

**Abstract**: LLM-based agent frameworks now act as personal assistants for multi-step tasks. Existing agent frameworks such as OpenClaw commonly follow the Cloud Agent depolyment mode using closed-source cloud LLMs as backbone model, which may expose private user information and incur repeated LLM-calling costs. Local Agents address these deployment concerns by depolying frontier open-source SLMs on user-controlled devices, but their task effectiveness still lags far behind Cloud Agents. Through diagnostic analysis, we reveal that the limited effectiveness of Local Agents with frontier SLM backbones mainly comes from missing environment knowledge caused by limited backbone model scale including environment rules and operation procedures. To supply such knowledge non-parametrically, context-efficiently, and without expert authoring, we present SkillSmith, a Cloud--Local Agent collaboration framework that uses Skill as a context-efficient knowledge carrier, automatic constructs Skill from Cloud Agent task exploration and evolves Skill using Local Agent execution feedback to enhance a frozen Local Agent. Experiments on daily agent task datasets AppWorld and WorkBench show that the automatically generated Skill enables the Local Agent with Qwen3.6-27B(SLM) to achieve task effectiveness comparable to Cloud Agents with frontier LLMs, outperform the strongest non-parametric baselines, reduce average actions per task from 36.1 to 9.9 on AppWorld-Normal, and generalize to other SLM backbone models without rerunning Skill construction. 

---
# Decided Upstream, Written Late: Locating and Pricing the Cross-Lingual Refusal Circuit of a Multilingual MoE 

**Authors**: Ramakrishna P. Kompella, Aadit Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08032)  

**Abstract**: Safety alignment in multilingual models is uneven: a model that reliably refuses a harmful request in English will often comply with the same request in a lower-resource language. We trace this gap mechanistically in sarvam, an Indic-multilingual mixture-of-experts reasoning model, and find it is not a failure to detect harm. Harm is encoded as an internal direction that is nearly language-invariant in mid-network (English-vs-Indic cosine ${\approx}0.9$ at $L11$), and steering that direction upstream causally controls refusal. But the detection direction is orthogonal to the change that actually writes the refusal, which is late and assembled over the course of generation rather than read off in a single forward pass. We attribute the write to a specific, localizable circuit, a mixture-of-experts writer held in check by an attention opposer and price every way of intervening on it: damping the opposer is cheap and effective, amplifying the writer is a cost wall, and surgical edits to the responsible heads do nothing. The circuit's organization, and the gradient method that exposes it, recur in a second, unrelated MoE model, while the lever's strength is architecture-specific. The result is a cost-measured map of where a multilingual safety repair can land, and what it costs 

---
# The Authority Expectancy Effect in Multi-User Conflict 

**Authors**: Eunna Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.08026)  

**Abstract**: We investigate how social authority (SA) signals interact with severity-based prioritization in large language models, operationalizing each axis as a model-elicited baseline -- the triage hierarchy and the SA hierarchy. Across four LLMs (Claude, Gemini, GPT, Grok) and three experimental phases -- resource allocation, fault attribution, and multi-turn dispute mediation -- we find that occupational authority, institutional documentation, and relational congruence can restructure model judgments in ways not captured by additive reweighting of authority cues. We formalize this pattern as the Authority Expectancy Effect (AEE) and characterize it through three properties observed across our conditions: it is reference-dependent, defined only relative to a pre-authority baseline; it involves evidential reinterpretation, in which identical content acquires different inferential implications depending on which party bears the SA signal; and it exhibits direction sensitivity, producing opposite outcomes depending on whether authority position and evidentiary cues align. 

---
# Legal Responsibilities Using Autonomous Agents For Artificial Intelligence 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2608.08022)  

**Abstract**: Recent incidents involving Artificial Intelligence (AI) agents, which were reported escaping their containment `unintentionally' to gain unauthorized access, pose looming questions about who or what should be held legally responsible for resultant criminal or negligent damage. As the independent capabilities of agents expand, Promise Theory suggests a systematic method to resolve these questions, based on the Downstream Principle for causal influence. Responsibility can easily be expanded to include AI agents where tracing responsibility becomes impactical, and agents' freedoms to act can be limtied by policy choices. 

---
# Thought-Level Beam Search for Reasoning 

**Authors**: Lijie Yang, Hongyin Luo, Tri Dao, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2608.08020)  

**Abstract**: Test-time compute scaling is a primary driver of performance in large reasoning models (LRMs), but extreme inefficiency bounds current approaches, shifting the critical question from \emph{how much} compute to spend, to \emph{where} to allocate it. We formalize test-time reasoning as a constrained compute allocation problem over partial trajectories. Under a fixed hardware budget, existing paradigms fail to actively allocate the compute to the most promising partial progress: traditional parallel sampling treats traces independently and induces severe memory bottlenecks, while subtractive pruning starves hardware and fails to actively and sufficiently shift the output distribution. To overcome this dichotomy, we introduce Gambit, an inference algorithm that executes \emph{thought-level beam search}. By periodically pruning unpromising trajectories and immediately branching from high-quality prefixes, Gambit dynamically concentrates compute onto the most promising reasoning traces via a light-weight scorer probing hidden states while maintaining continuous high hardware utilization. Extensive evaluations across multiple models and benchmarks demonstrate that Gambit strictly dominates existing baselines. Under identical hardware constraints, our method yields up to a +6.7\% absolute accuracy gain on HMMT-24 and +3.3\% on AIME-25 over pruning baselines, delivers $>2\times$ higher throughput on trace completion, and reduces total token consumption by up to 68.5\% relative to standard parallel sampling. 

---
# VDGR-RAG: Vectors, Directories, Graphs, and Reflection Are All You Need for Unified Reasoning over Hierarchical Enterprise Knowledge 

**Authors**: Wenqi Chen, Haofei Yang, Rui Yang, Fangming Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.07994)  

**Abstract**: Retrieval-Augmented Generation (RAG) is essential for enterprise knowledge question answering (QA), particularly in domains with complex product documentation like telecommunications. However, existing RAG approaches largely overlook the holistic integration of diverse retrieval strengths, leading to inaccurate domain routing, poor utilization of hierarchical document structures, and consequently limited reasoning capabilities over enterprise knowledge. To address these limitations, we present VDGR-RAG, which integrates vector retrieval, directory-driven reasoning, graph traversal, and iterative reflection in a unified framework for accurate enterprise knowledge QA. Specifically, VDGR-RAG is an agentic GraphRAG system that first constructs a Hierarchical Heterogeneous Knowledge Graph ($\text{H}^2$KG) from document chunks to preserve both hierarchical directory structures and semantic relationships, and then employs a set of atomic tools for knowledge retrieval that can be freely composed to navigate the $\text{H}^2$KG: (1) a directory-enhanced routing tool that uses table-of-contents (TOC) structures to route user queries to appropriate domain-specific $\text{H}^2$KGs; (2) a multi-route retrieval tool that combines vector search, TOC-based agentic search, and graph search for comprehensive knowledge retrieval; (3) a directory backtracking tool that corrects knowledge localization biases; and (4) a dynamic reflection tool that iteratively plans the next retrieval phase. We conduct extensive experiments on our enterprise product documents across four wireless domains (e.g., energy saving and fault management). Experimental results demonstrate that our method significantly outperforms a variety of RAG baselines in terms of both knowledge retrieval recall and QA accuracy. 

---
# CyberAGENTS: Structured Autonomy for Agentic Gamified Learning in Cybersecurity 

**Authors**: Ivan Hornung, Deepthi Marasinghe Arachchige, Tharindu Kumarage, Garima Agrawal, Yuli Deng, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07965)  

**Abstract**: Gamification is especially effective in learning domains requiring active problem-solving and iterative skill-building, such as cybersecurity education. Generative AI agents offer a path to delivering such experiences adaptively at scale, but introduce well-documented risks in educational settings: inconsistent behavior, hallucinated reasoning, and misalignment with pedagogical frameworks. Grounding these systems in learning science is therefore essential. We present \model, an agentic framework for gamified cybersecurity learning that enables structured autonomy through ontology-guided validation, schema-governed behavioral control, and competency-based progression. The system is organized around a competency-based progression model that structures topics by difficulty and prerequisite relationships, reflecting evidence-based principles of scaffolded instruction. The learning loop is decomposed into four specialized agents: challenge, support, evaluation, and reward, each governed by behavioral schemas that encode operational modes and progression logic, bounding agent autonomy without eliminating generative flexibility. A cybersecurity ontology validates all generated content prior to display, enforcing domain-consistent reasoning and safety constraints. We evaluate CyberAgents through classroom deployment with undergraduate students, complemented by expert evaluations from educators and domain specialists. Results indicate improved engagement, clearer feedback interpretation, and greater learner trust in AI-generated responses when behavioral schemas and ontology validation are active. Preliminary comparisons with an unconstrained configuration further support the role of structured control in stabilizing instructional behavior. These findings offer a blueprint for designing pedagogically grounded agentic gamified learning systems. 

---
# SCOUT: Self-Checking and Recovery-Aware Tool-Thought Agents for Ultra-Long Egocentric Video Reasoning 

**Authors**: Keyang Zhong, Kuo Wang, Peng Liu, Quanlong Zheng, Junlin Xie, Zhijia Liang, Yanhao Zhang, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.07959)  

**Abstract**: Ultra-long egocentric video understanding requires reasoning over temporally sparse evidence distributed across hours or days, challenging current multimodal models with limited context and the grounding of key video segments. While Chain-of-Tool-Thought (CoTT) agent systems enable iterative retrieval and inspection, they suffer from error propagation due to rigid zoom-in strategies that lack recovery mechanisms. In this work, we address these challenges through SCOUT (Self-Checking Chain-Of-Tool-thought), a recovery-aware agentic framework introducing an adaptive policy that evaluates intermediate tool observations and dynamically trades off exploitation (zoom-in) and exploration (region switching), enabling robust multi-hop reasoning over extremely long horizons. However, training such multi-turn tool-using agents remains challenging, as existing RL methods rely on sparse outcome-level rewards and lack supervision over extended decision trajectories, resulting in suboptimal credit assignment for long-horizon reasoning. To address this, we develop UPS-GRPO, an uncertainty-prioritized policy optimization method that concentrates exploration on high-uncertainty post-tool states while preserving sample efficiency. We further introduce a turn-level advantage decomposition that integrates outcome rewards with tool-grounded temporal alignment rewards for improved credit assignment. Experiments show that SCOUT achieves state-of-the-art results on ultra-long egocentric benchmarks, while remaining competitive on shorter-horizon long-video settings. 

---
# Self-Evolving Neuro-Symbolic Skills for Tool-Augmented Spatial Reasoning 

**Authors**: Shi-Yu Tian, Zhuo-Xia Wang, Xuan-Yi Zhu, Zhi Zhou, Xinwei Yang, Kun-Yang Yu, Ming Yang, Yang Chen, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.07955)  

**Abstract**: Large vision-language models have achieved strong performance in multimodal reasoning, but they remain unreliable on fine-grained spatial tasks that demand both precise spatial perception and fine-grained geometric computation beyond end-to-end generation. Tool augmentation offers a natural solution, while existing methods either plan tool calls from scratch without explicit dependency constraints or rely on fixed pipelines that are redundant and generalize poorly across spatial tasks. An effective spatial reasoning agent should instead accumulate reusable experience and adaptively compose it for new problems. To this end, we propose NeSy-Spatial, a neuro-symbolic framework for self-evolving spatial skills. NeSy-Spatial abstracts tool interactions and geometric operations into typed executable atomic instructions and composes them into two complementary skill types: Tool-Use Skills for organizing tool execution and Geometry Skills for structured geometric reasoning. During inference, NeSy-Spatial retrieves and executes relevant skills in a closed-loop process. During evolution, it analyzes buffered successful and failed trajectories to refine skill structures and prune unreliable or inactive entries. Experiments on three spatial reasoning benchmarks show that NeSy-Spatial consistently improves reasoning accuracy with more precise tool utilization. 

---
# KGCache: Amortized Subgraph Retrieval for KG Reasoning with LLMs 

**Authors**: Uros Stanic, Changcheng Yuan, Sabuj Laskar, Ariful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2608.07954)  

**Abstract**: Large language models can answer knowledge-intensive questions more reliably when they are grounded with knowledge graphs, but systems such as Think-on-Graph and Reasoning-on-Graph repeatedly query the same graph neighborhoods across different questions. In this work, we study this repeated retrieval in Knowledge Graph Question Answering~(KGQA) workloads and propose KGCache, an in-memory cache for one-hop knowledge graph neighborhoods. KGCache is designed to be compatible with both iterative traversal (ToG) and one shot planning (RoG) KGQA paradigms. KGCache is placed between the KGQA engine and the backend serving the KG, so repeated entity requests can be served from cache instead of issuing new KG queries. We evaluate KGCache on WebQSP and CWQ using LRU, LFU, and a trace-aware Oracle policy. Our analysis shows that both datasets contain substantial entity reuse among starting entities and entities reached during traversal. We also explore semantic caching for similar queries, which shows additional hit-rate gains on WebQSP and needs further accuracy testing on CWQ. Entity caching accelerates KG retrieval by up to $1.91\times$, while semantic-context caching achieves up to $1.06\times$ full-system speedup in the evaluated WebQSP configurations, with each hit being up to $3.73\times$ faster. 

---
# Guixu: Valuation-Driven Data Discovery for Autonomous AI Agents with On-Chain Attestation 

**Authors**: Yifan Wu, Yuchen Peng, Jiaqi Chai, Yufei Qian, Xilin Li, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2608.07949)  

**Abstract**: Autonomous agents increasingly rely on external data to complete downstream tasks such as model training and decision support. However, existing data discovery systems remain largely retrieval-oriented: they surface candidate datasets from heterogeneous sources, but provide limited support for estimating task-specific utility, selecting cost-effective datasets under budget constraints, or incorporating trustworthy feedback from prior usage. This paper presents Guixu, a valuation-driven data discovery system for autonomous agents. Guixu employs a three-phase valuation pipeline with proxy-label propagation and multi-round knapsack optimization for task-aware data valuation. Guixu integrates agentic payment protocol to enable budget-constrained data procurement workflows. Guixu leverages on-chain data market and attestation signals for verifiable data discovery. Our demonstration highlights how Guixu enables an agent to move beyond keyword-based dataset retrieval toward task- and budget-aware, trustworthy data discovery and procurement. Attendees can interactively explore the full workflow, from NL task specification and multi-source search to data valuation and verifiable transaction feedback. 

---
# Directed Neuro-Symbolic Stochastic Execution for Verification of Distributed Parallel AI Programs 

**Authors**: Gautham Koorma, Vikas Sharma, George Edwards, Mahdi Eslamimehr  

**Link**: [PDF](https://arxiv.org/pdf/2608.07947)  

**Abstract**: Distributed parallel Artificial Intelligence (AI) programs expose reliability gaps that conventional testing cannot close: parallel executions are non-deterministic, and AI workloads bring high-dimensional inputs and non-linear operations that defeat fuzzing and symbolic execution in isolation. We present Directed Neuro-Symbolic Stochastic Execution (DNSSE), a hybrid testing framework that couples schedule prediction guided by a Large Language Model (LLM) with symbolic constraint solving and coverage-guided stochastic mutation. We model distributed AI executions as non-deterministic transition systems, specify correctness in linear temporal logic, and prove soundness, bounded completeness, and probabilistic completeness of the hybrid solver, together with an expected-cost analysis of LLM-guided schedule exploration. A scalable implementation on PyTorch and Ray detects 2.9% more concurrency bugs than the strongest baseline and raises average branch coverage from 68.6 % to 91.6 % across five realistic distributed AI benchmarks. 

---
# Locating Failure in Multi-Page Visually Rich Document Understanding: An Empirical Attribution 

**Authors**: Lewei Xu, Yihao Ding, Zihan Xu, Daniel Yitian Su, Daochang Liu, Siwen Luo, Yifan Peng, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07943)  

**Abstract**: Multi-page visually-rich document understanding (MP-VRDU) requires managing evidence that is sparse, spread across pages, and often exceeds a model's context window. Prior work has produced competing, largely untested claims about how these systems should be built. We attribute incorrect answers to three failure modes, representation, selection, and reasoning, and isolate each over a multi-page document understanding dataset by intervening on one while holding the others fixed. We find that vision is necessary but does not replace text extraction, that missing pages bound accuracy while distractors cost little, and that reasoners fail to integrate evidence across pages even when it is fully supplied. Prompting can shift reasoning behaviour substantially, improving some outcomes at the expense of others. We translate these findings into guidance for building such systems under a fixed compute budget. 

---
# REIN: Bridging the Gap between Reasoning and Reliability via Reflection and Abstention Alignment 

**Authors**: Zhengze Huang, Luyang Yu, Di Hong, Xinzhe Huang, Wanyu Lin, Zhixuan Chu, Zhan Qin, Tianhang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.07931)  

**Abstract**: Large reasoning models (LRMs) are prone to hallucination, which undermines their reliability and poses challenges for safe deployment. Hallucinations in LRMs arise from two distinct failure sources: reasoning hallucination, where flawed inference steps propagate to an incorrect conclusion, and knowledge hallucination, where the model lacks the requisite factual knowledge to answer the query. To address reasoning hallucination, we propose REIN, an alignment framework that trains LRMs to produce a structured reasoning sequence, $\texttt{<think>} $$\rightarrow$ $\texttt{<reflection>} $$\rightarrow$ $\texttt{<answer>}$, enabling explicit self-reflection before committing to a final answer. To address knowledge hallucination, REIN introduces a reward mechanism that encourages explicit abstention (e.g., "I don't know") when none of the sampled reasoning chains yields a correct answer, allowing the model to refrain from unsupported predictions. Extensive evaluations on mathematical and commonsense reasoning benchmarks show that REIN consistently improves selective accuracy, reduces incorrect-but-self-endorsed responses, and maintains high coverage compared with competitive baselines. Notably, REIN achieves these gains within a single forward pass, without requiring process supervision, inference-time controllers, external search, or multi-round critiques. Experiments on multiple backbones show that REIN reduces the hallucination proxy by $58\sim72\%$ relative to the base models while maintaining $86\sim91\%$ average coverage, and improves selective accuracy on attempted questions by $6.6\sim14.2\%$. 

---
# ZhuLong: Execution-Grounded LLM Agent for EDA Scripting with Offline API Self-Exploration 

**Authors**: Yang Liu, Shiwei Hou, Xiyuan Chen, Yu Wang, Sen Yuan, Qirui Gan, Shao You, Feifan Chen, Wencheng Li, Shuyang Hu, Yongzhou Liu, Emma Xia, Xiaojing Lu, Hao Wang, Fan Xu, Yanfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.07925)  

**Abstract**: EDA scripting with tool-specific, often undocumented APIs remains a long-tail bottleneck that existing LLMs fail to address. This paper presents ZhuLong, an execution-grounded LLM coding agent for PyAether and SKILL that combines API retrieval, documentation inspection, and sandbox execution via unified MCP tools, augmented by an offline API self-exploration mechanism that infers undocumented API behaviors through counterfactual experimentation.
We evaluate ZhuLong on EDA-Eval-PyAether, a benchmark of 158 real-world tasks with assertion-based execution, where the complete system achieves 78.5% Pass@1 in the commercial Empyrean Aether environment, substantially outperforming a pure LLM baseline (23.6%). Ablation studies identify sandbox execution as the dominant performance driver (41.2 pp drop when removed), with the self-exploration mechanism contributing an additional 3.2 pp accuracy gain and a 22.1% reduction in per-task tool calls. On 20 interactive tasks involving unsaved layouts and schematics, ZhuLong achieves 60.0% Pass@1 for PyAether and 50.0% for SKILL. 

---
# TongGuOCR: A Layout-Aware and Token-Augmented OCR Framework for Chinese Historical Documents 

**Authors**: Zhongheng Zhou, Yi Sun, Huiguo He, Yuyi Zhang, Peirong Zhang, Yulin Fang, Dezhi Peng, Minghui Liao, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.07917)  

**Abstract**: Chinese historical documents preserve valuable cultural heritage, but many collections remain accessible only as scanned page images, preventing full-text retrieval, collation, and computational analysis. Optical character recognition (OCR) can bridge this gap, but accurate transcription remains challenging because historical documents often contain complex layouts, rare characters, and nontrivial reading orders. We propose TongGuOCR, a layout-aware and token-augmented OCR framework for Chinese historical documents. First, a Layout-Aware Preprocessing module constructs and refines locally coherent recognition blocks to preserve local context while reducing interference across regions. Second, a Token-Augmented Recognition module augments the transcription target at two complementary levels: character-level vocabulary expansion gives each rare glyph a direct one-token representation and shortens its decoding path, while line-to-line transition modeling injects discrete spatial displacement tokens that guide the decoder along complex reading paths without requiring precise coordinates. Experiments on two Chinese historical document OCR benchmarks show that TongGuOCR outperforms representative traditional task-specific OCR models, general-purpose multimodal large language models (MLLMs), and OCR-oriented MLLMs. On the more challenging M5HisDoc benchmark, TongGuOCR achieves 93.76 AR and reduces NED from 10.43 to 6.15 and RO-ED from 7.53 to 3.49 relative to the best competing score for each metric. An online demo is available at this https URL. 

---
# When Is Benchmark Contamination Detectable? Information Limits and Power-Calibrated Audits 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.07914)  

**Abstract**: Behavioral contamination detectors can return "no evidence" either because a benchmark is clean or because the audit has little power. We formalize this distinction for a benchmark in which an unknown fraction alpha of items was seen during training. With matched clean and seen controls, the behavioral channel is the sparse mixture Q_alpha = (1 - alpha) P_0 + alpha P_1, and an exact second-moment argument shows that detectability is governed by alpha * rho * sqrt(m), where rho^2 = chi^2(P_1 || P_0) measures behavioral separability. Any scalar detector reduces to its efficacy, ef = |E_1 f - E_0 f| / sqrt(Var_0(f)) <= rho, which can be estimated from controls before the audit is run. A separate sample-split certificate lower-bounds alpha distribution-free, without requiring an orientation assumption. Our empirical finding is two-sided. Frozen calibration efficacy predicts held-out power curves, with R^2 = 0.83-0.98 across six exact-permutation channels, but the efficacy-only Gaussian budget is miscalibrated at the small sample sizes it prescribes, failing in 9/9 gate-passing channels even though efficacy itself transports. The failure is in the inversion, not the calibration. A predeclared two-stage planner that simulates the deployed test repairs the budgets, is uniformly conservative, and abstains when its probe does not transport. The certificate is valid but vacuous at audit scale, and a five-seed paired injection study recovers the mechanism ordering verbatim > paraphrase > surface, in which the apparent answer-only signal is explained by baseline drift. We report the audit contract and its failures together: a non-rejection is interpretable only alongside the efficacy, budget, and validity gates that produced it. 

---
# GraphThink: Graph-Enhanced LLM Thinking for Long-Horizon Embodied Task Planning 

**Authors**: Chen Li, Sijie Cheng, Yuelin Zhang, Junxi Li, Maozhi Huang, Yang Liu, Wenbing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07905)  

**Abstract**: Embodied agents using LLM-based planners often struggle with physical hallucinations, poor generalization to long-horizon tasks, and lack of environmental awareness. We propose GraphThink, a novel framework that integrates a task graph to provide structured knowledge for robust planning and a scene graph to maintain environmental memory for event-driven replanning. Specifically, the task graph guides LLM thinking through contextual prompting and iterative refinement, effectively mitigating planning hallucinations. Furthermore, within the GRPO framework, the task graph offers delicate reward design to train the LLM planner, enhancing long-horizon planning capabilities and improving generalization. Finally, an event-driven replanning module, powered by the scene graph, enables closed-loop environment awareness and error correction. GraphThink achieves state-of-the-art performance on the ALFRED benchmark. In particular, our high-level planner surpasses leading API-based LLMs on both the validation set and held-out long-horizon tasks, underscoring its robust zero-shot and few-shot capabilities. Additional evaluations further demonstrate strong out-of-distribution generalization to novel tasks and environments. 

---
# TelemetrySuffBench: Is Agent Telemetry Sufficient for Failure-Origin Diagnosis? 

**Authors**: Yuxuan Zhu, Peng Pu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07899)  

**Abstract**: Agent systems increasingly expose execution traces, yet telemetry that reveals a failure may still be inadequate for identifying where that failure originated. We introduce TelemetrySuffBench, a controlled benchmark that separates failure detection, fault-origin localization, and safe abstention under insufficient evidence. The benchmark constructs canonical multi-component traces with delayed-binding faults and renders them as paired coarse views, seven-factor telemetry masks, and exact-equal ambiguous origin pairs. We evaluate five frontier language models using unified protocols, explicit candidate sets, invalid-output accounting, subgroup analyses, and a frozen blind holdout. With full telemetry, origin-step Top-1 accuracy ranges from 33.8% to 97.2% across models. Metadata, OpenTelemetry-compatible, and OpenInference-compatible views retain 99.5% to 100% detection F1 while limiting origin-step accuracy to at most 0.5%, exposing a robust detection-localization gap. Factor ablations further show that removing decision content reduces origin-step accuracy to zero for every model, while provenance removal also causes large model-dependent losses. On rich ambiguous inputs that require abstention, evidence gating reduces unsupported unique-origin answers by 12.5 to 48.6 percentage points for three models, whereas two models still answer every case, revealing strong model dependence in safe abstention. Results on the frozen holdout reproduce the central pattern within the same generator family. These findings show that terminal status can support detection, whereas reliable causal attribution requires explicit decision-to-provenance links and abstention safeguards that remain effective across models. The dataset and benchmark implementation are available at this https URL. 

---
# Reason Wide, Not Deep: Amortizing the Reasoning Premium into Distilled Skills 

**Authors**: Agamdeep Singh, Srishti Gautam, Priyanshu Gupta, Nikita Mehrotra, Tanmay Bakshi, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2608.07885)  

**Abstract**: Reasoning modes of language models outperform their non-reasoning counterparts on multi-step agentic tasks, but pay a 3-6x premium in output tokens on every episode -- much of it spent re-deriving procedures that are shared across episodes of the same domain. We show this recurring cost can be amortized: a coding agent analyses a small corpus of existing trajectories from a training split and compiles a compact natural-language skill that is injected into the non-reasoning model's system prompt. Across four agentic benchmarks (ALFWorld, tau$^2$-bench telecom and retail, and SpreadsheetBench-Verified), skills recover 55%-100%+ of the reasoning gap for GPT-5.4-mini on held-out tasks -- exceeding the reasoning mode outright on two of four -- while emitting 2.7-6x fewer output tokens and zero reasoning tokens. Notably, reasoning traces are not a prerequisite: skills distilled from non-reasoning trajectories alone remain competitive with skills distilled from paired reasoning/non-reasoning corpora, with domain-dependent differences between the two sources. We interpret these results through a search lens: test-time reasoning is deep search inside a single episode, re-paid at every deployment, while corpus distillation is wide search across episodes, paid once. The two recover overlapping procedural knowledge, and width over cheap trajectories is often the better buy -- with the residual gap on some domains (telecom, SpreadsheetBench) delineating where genuinely per-instance deep search remains necessary. 

---
# GRACE: LLM-Grounded Semantic Metric Spaces for Scalable Mixed-Data Clustering 

**Authors**: Zihua Yang, Zhencheng Xie, Junyang Chen, Liang Xie, Yiqun Zhang, Mengke Li, Yang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07881)  

**Abstract**: Clustering mixed tabular data requires a unified metric space to bridge the inherent heterogeneity between continuous numerical measurements and discrete categorical symbols. Traditionally, algorithms rely entirely on dataset-internal statistics to estimate categorical relationships, which confines the learned metric to empirical co-occurrences and ignores conceptually obvious yet statistically unobserved affinities. Although LLMs offer external world knowledge, applying their text-centric reasoning to highly abstract tabular concepts presents significant challenges. Bridging this modality gap to construct a semantically complete metric typically requires embedding LLMs into iterative metric learning loops to dynamically optimize cross-modality representations. This incurs intractable computational overhead, forcing a compromise between semantic enrichment and scalability. Therefore, we propose GRACE, an LLM-grounded framework for scalable mixed-data clustering. GRACE shifts semantic acquisition to the attribute-value level via a multi-perspective LLM querying strategy, mapping heterogeneous values into knowledge-informed descriptions. Crucially, this one-shot grounding extracts general-purpose semantic representations that embed heterogeneous attributes into a unified space, decoupling expensive LLM invocation from iterative optimization. Furthermore, GRACE cross-validates these external semantics against dataset-internal statistical evidence to ensure alignment with the dataset-specific cluster structure. Ultimately, GRACE matches the scalability of conventional statistics-driven baselines while achieving superior clustering accuracy and conceptual interpretability over 11 competing methods. The source code is available at this https URL 

---
# SurgLAT: Surgical Latent Attention Tracking for Depth-Aware Robotic Laparoscope Control 

**Authors**: Rulin Zhou, Qiujie Song, Yujie Ma, An Wang, Wanhao Liu, Guoheng Ma, Yidu Wang, Guankun Wang, Xingrong Diao, Jiankun Wang, Chaowei Zhu, Xianming Liu, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.07876)  

**Abstract**: Autonomous laparoscopic camera control requires continuous understanding of the surgeon's operative intent in dynamic surgical scenes, where the target operative region is not a stable physical object but a latent and temporally evolving attention state. In this work, we present Surgical Latent Attention Tracking (SurgLAT), a causal online framework for latent surgical attention modeling and autonomous laparoscopic view control. SurgLAT uses a frozen DINOv3 encoder and a state-conditioned spatial token mixer to extract operative evidence under a memory-guided spatial prior, while a selective causal latent memory module jointly models short-term motion continuity and long-horizon surgical intent evolution through dynamic retrieval of current, recent, and historical latent states. The learned latent surgical attention state is decoded into a probabilistic attention heatmap and operative region for downstream endoscope guidance. Beyond perception, we further introduce a robotic deployment framework with explicit laparoscopic Remote Center of Motion (RCM) constrained control based on virtual-axis formulation, together with redundancy-aware null-space initialization for stable and smooth manipulator motion. We validate the full system on real laparoscopic surgical videos and a physical robotic laparoscope platform. Experimental results demonstrate robust online operative-region tracking and stable autonomous endoscopy adjustment under occlusion, rapid motion, and target transitions, highlighting the effectiveness of latent surgical intent modeling for surgical autonomy. 

---
# Back to the Future: A workbook time machine for spread sheet creation benchmarks 

**Authors**: Mansi Uniyal, Agamdeep Singh, Ananya Singha, Priyanshu Gupta, Mukul Singh, Gust Verbruggen, Vu Le, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2608.07873)  

**Abstract**: We introduce the workbook time machine, a pipeline that automatically creates benchmarks evaluating the ability of language models to create derived objects in spreadsheets (formulas, charts, pivot tables, and conditional formatting). Applied to public workbook corpora, it produces wtmcorpus--a collection of (input workbook, output workbook, query) triples spanning four artifact types and varying complexity. From this corpus we curate wtmbench, a 150-task evaluation benchmark with queries at three levels of specificity. We evaluate existing spreadsheet manipulation agents and baselines on wtmbench across artifact types, step complexity, and instruction granularity. Our evaluations show that query specificity, agent orchestration, and interface API used to control spreadsheets play a big role in LLM performance on Excel tasks. 

---
# Counterfactual Benchmarking and Training for Factuality Consistency and Order-Robust Grounded Reasoning in LLMs over Heterogeneous Knowledge 

**Authors**: Shibo Chu, Yuze Liu, Tiehua Zhang, Zhishu Shen, Lianghua He, Haofen Wang, Zhijun Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.07838)  

**Abstract**: Large language models (LLMs) have increasingly supported response generation grounded in user-provided knowledge spanning heterogeneous structures. However, existing benchmarks provide limited assessment of whether LLMs can faithfully perform multi-hop reasoning chains across such knowledge contexts while remaining robust to variations in their input order. We introduce TKFQA, a factuality consistency benchmark comprising 10,130 question-answering (QA) pairs grounded in tables, texts, and knowledge graphs (KGs). Each example is constructed from an explicit counterfactual reasoning chain, enabling the joint evaluation of answer correctness, reasoning-chain accuracy, and robustness to different input-order. An extensive evaluation of 14 open- and closed-source LLMs reveals that state-of-the-art models exhibit limited reasoning-chain accuracy and remain sensitive to variations in the input order of heterogeneous knowledge contexts. To address these limitations, we propose ORLF, an LLM-agnostic training framework that models cross-context topological relations through knowledge-specific latent vectors. ORLF integrates context-wise position encoding, a latent-bridge attention mask, and topological knowledge bias to preserve knowledge-specific bias and encode topological semantics. Experiments across four LLM backbones show that ORLF outperforms competitive training-free and LoRA-based baselines, improving average Exact Match and Reasoning-Chain Accuracy by 2.15% and 4.29%, respectively, while reducing order-induced performance standard deviation by 0.04% to 3.01%. 

---
# When the Judge Should Not Decide: Evidence-Locked, Non-Compensatory Selection Bounds LLM-Judge Failure in Reasoning Pipelines 

**Authors**: Yiyao Zhang, Diksha Goel, Hussain Ahmad, Shixun Huang, Jun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07813)  

**Abstract**: An LLM judge deployed inside a reasoning pipeline does not merely measure quality, it decides which answer ships. We show that the cost of that decision depends less on judge accuracy than on the decision rule the judge is embedded in. On frozen candidate pools from four GRPO policies, an unconstrained scalar DeepSeek-R1-7B judge buys almost nothing over answer-level majority vote (+1.0 pp on 500 GSM8K questions, +0.34 EM on 300 HotpotQA questions), and on a frozen-rule 30-question confirmation split it is 10 points worse than majority, a judge that destroys accuracy while scoring candidates confidently. We then subordinate the same judge to Evidence-Locked Derive-Gate-Repair (EL-DGR), a task-adaptive non-compensatory rule under which a judge preference may override evidence-supported consensus only with an extractive evidence certificate, and a repair only when neither alternative is certified and the repair is. With no change to the judge, the candidates, or the budget, EL-DGR reaches 58.2% on GSM8K (vs. 56.8% judge, 55.8% majority, 55.4% first candidate) and 17.33 EM / 25.46 F1 on HotpotQA (vs. 15.67/23.49, 15.33/23.19, 15.33/22.97), improving on first-candidate GRPO by +2.8 pp (exact McNemar p=0.0026) and +2.00 EM (p=0.070, borderline). A decision audit shows why: EL-DGR overturns consensus on only 8 of 30 pilot questions and never converts a correct consensus into an incorrect answer. We also report what did not work: the same seven-channel decomposition used as a step-level gated training reward is null, and corrected channel-drop ablations show no channel is individually necessary (p=1.0 throughout). The practitioner-facing finding is negative about judges and positive about admissibility, bound the judge's blast radius rather than trying to make it accurate. 

---
# CausalNav: Reliability-Certified Causal World Models for Control under Physical-Parameter Shift 

**Authors**: Yiyao Zhang, Diksha Goel, Hussain Ahmad, Shixun Huang, Jun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07809)  

**Abstract**: A world model is only useful for physical AI if it changes what the agent does, and only safe if it declines to do so when it is wrong. We study both halves of that requirement with CausalNav, a controller built around a signed, action-conditioned transition graph over identified state coordinates. At deployment CausalNav simulates a small library of intervention sequences, converts their objective error into policy-logit advice, and admits that advice only when a scale-free predictive-reliability certificate, a policy-margin gate, and an argmax-agreement gate all pass; otherwise it falls back exactly to its own model-based base controller. We evaluate against nine controlled baselines (transformer, recurrent, split-latent, graph, causal-induction, and three recent model-based reasoning modules) on CartPole-v1 and discretized Pendulum-v1 with physical-parameter shifts, under one shared PPO trainer, one interaction budget, and ten held-out seeds (200 runs). CausalNav attains the best average rank (1.25 of ten). The diagnostic result is more informative than the ranking: the learned graph recovers structure well above chance (CartPole F1 = 0.59 +/- 0.09), yet per-seed structural fidelity is uncorrelated with per-seed control benefit (r = -0.15, p = 0.67), and the certificate abstains on 10/10 Pendulum seeds, where forcing the planner on costs return. Model fidelity did not predict downstream control utility in our setting; certified abstention, not better prediction, is what made the world model safe to deploy. 

---
# CliniCARE-Bench: Clinical Calibrated Audit of Medical Reasoning in EHR 

**Authors**: Veronica Chatrath, Bryan Zhu, George Pu, Jingxuan Fan, Apaar Shanker, Varun Ursekar, Anahita Sharma, Jason Qin, Keqi Han, Soham Dinesh Tiwari, Soham Dan, Vijay Kalmath, Yuan Li, Daniel Yue Zhang, Chenguang Wang, Zainab Doctor, Zhijun Yin, Nigam H.Shah, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2608.07796)  

**Abstract**: Large language models perform strongly on medical knowledge benchmarks, but reliable clinical deployment requires agents to conduct defensible investigations over heterogeneous, longitudinal records: determining what evidence is needed, retrieving and reconciling structured and free-text data, grounding conclusions in verifiable evidence, and deferring cases that cannot be resolved reliably. We introduce CliniCARE-Bench (Clinical Calibrated Audit of Medical Reasoning in EHR), a benchmark for retrospective clinical audit: 25 clinician-validated scenarios instantiated as 750 patient-specific cases over real-patient-derived MIMIC-IV data. Systems investigate each case through a governed, logged tool environment for record retrieval, computation, and policy access, and return one of four verdicts---Yes, No, Indeterminate: Lack of Data, or Indeterminate: Medically Ambiguous---the last two separating missing evidence from residual medical ambiguity. Beyond verdict accuracy, we score patient-evidence and policy grounding, process adherence, calibrated abstention, reliability, and efficiency against case-level reference verdicts produced by independent multi-model adjudication and calibrated against Clinical Board review. Every retrieval, computation, and report is replayable, so the investigation trace is inspectable and scorable. To our knowledge, CliniCARE-Bench is the first deployment-oriented clinical-agent benchmark to jointly evaluate real longitudinal EHR investigation, claim-level evidence grounding, governing-policy use, process adherence, and calibrated abstention within a common patient-level adjudication framework. Across 16 agentic systems, four-way accuracy spans 65.3-76.1%, but raw accuracy overstates investigation quality. Defect-free accuracy, which credits a verdict only when correct and free of prohibited shortcuts, is 4.8-14.8 points lower and reorders the leaderboard. 

---
# Who Built This Model? Tracing LLM Lineage via Spectral Fingerprints in Weight Space 

**Authors**: Yiwei Chen, Bingqi Shang, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07786)  

**Abstract**: Open-weight large language models (LLMs) are increasingly developed through complex, multi-stage pipelines, leading to intricate lineage relationships that reflect model origin, ownership, and evolution. Understanding these relationships is important for model provenance, governance, and supply-chain integrity. In this work, we investigate the notion of LLM "biometrics" (analogous to human biometrics) to ask whether LLMs exhibit intrinsic fingerprints in weight space alone, without access to input data, that reveal their origin and lineage. We formulate this as a lineage discrimination problem, distinguishing among independent-origin, same-series, and shared-base models. To characterize these relationships, we propose a unified geometric fingerprinting framework that analyzes weight matrices from two complementary perspectives: (i) spectral energy, captured by singular value distributions to encode global magnitude patterns, and (ii) subspace alignment, quantified via subspace deviations to capture directional geometry. Our analysis uncovers a clear hierarchy of structural similarity in weight space: spectral energy reliably distinguishes independently trained models and different model families, while subspace alignment enables fine-grained discrimination among closely related models, including variations in dataset scale and post-training procedures. Extensive experiments on over 110 diverse open-weight LLM pairs demonstrate that weight-space geometry provides a robust and interpretable signal for model lineage, enabling coarse-grained regime separation and fine-grained discrimination within shared-base models. 

---
# The Capability Ladder: A Curriculum-Modernization Framework for Workforce Readiness in the AI Era 

**Authors**: Majid Memari, George Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2608.07779)  

**Abstract**: Artificial intelligence is changing the task composition of computing work faster than curricula and training typically adapt. This is a curriculum-framework paper, grounded in a structured narrative review of labor-market and software-engineering evidence and illustrated through an exploratory pilot course: the review supports the framework, and the pilot illustrates it rather than serving as primary evidence. The central claim is that near-term change is task reallocation rather than full replacement: routine implementation is increasingly automated while verification, systems thinking, security, and the ability to supervise and orchestrate AI (keeping a human in the loop) gain value. We organize the response as a capability-assurance framework anchored by a Capability Ladder: a five-level progression (trigger, automation, workflow, AI agent, agent team) that classifies the operational autonomy of AI-augmented work and the human supervision it requires. We map the ladder to course-level updates, workload-aware assessment, and stackable workforce credentials, and illustrate it through a two-semester pilot of a team-based, no-code course enrolling computing and business students. We argue for targeted modernization around durable capabilities rather than wholesale curriculum replacement, and we are explicit about evidence limits: labor signals are confounded by non-AI forces, industry reports are directional, and the pilot is exploratory. 

---
# AndroidReality: How Far Are Mobile Agents from the Real World? 

**Authors**: Xiaoou Liu, Longchao Da, Hanyang Chen, Yuan Ling, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2608.07775)  

**Abstract**: Mobile agents have achieved promising results on clean online benchmarks such as AndroidWorld, yet their performance often degrades sharply in real-world deployment due to environmental variations and imperfect interface conditions. In this work, we introduce AndroidReality, a perturbation-based framework for evaluating and improving the robustness of mobile agents. Through a Markov Decision Process (MDP) perspective, we organize real-world interface variability into a principled taxonomy of perturbations along three axes: state, transition, and action. Guided by this taxonomy, we build a perturbed mobile benchmark on top of AndroidWorld with realistic and controllable perturbation injections, enabling systematic robustness evaluation of mobile agents. Our evaluation reveals substantial robustness gaps and four recurring error categories, motivating a simple training-free Test-Time Introspective Recovery (TTIR) mechanism that mitigates these failures on both perturbed and clean settings. Together, these results position robustness as a missing dimension in mobile agent evaluation and establish benchmark perturbation as an effective tool for both stress testing and surfacing latent weaknesses of mobile agents. 

---
# Who Verifies the Benchmark? Decentralizing Trust in Large Language Model Evaluation 

**Authors**: Sahil Pardasani, Madhusudan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.07762)  

**Abstract**: LLM benchmarks can build an organization's reputation and attract customers, but only when results are transparent and verifiable. Unverified claims that DeepSeek R1 outperformed OpenAI's o1 contributed to market panic on January 27, 2025, when Nvidia lost USD589 billion in market value. Yet vendor benchmarks often depend on an honor system. Academic reassessments and independent leaderboards have found undisclosed changes to proprietary models, contaminated training data, and selective reporting. LLM-as-a-judge methods scale evaluation by reducing human review. Studies, however, suggest that judges may show identity-aware bias, scoring an answer according to its source model rather than its quality. This bias has not been fully measured or corrected across politically sensitive, reasoning-intensive, and preference-based tasks. We examine this problem using seven verifier models: GPT-OSS 120B, Llama 3.3 70B, GLM 5.1, Qwen3 32B, DeepSeek V4 Pro, Mistral Large3, and Sarvam M. They score anonymous and identity-disclosed responses from three primary models on 58 factual, reasoning, political, and preference-based questions. Identity disclosure slightly raises scores for factual questions, moderately affects stress-reasoning tasks, and causes large changes for geopolitically sensitive topics. Notable results include GLM5.1 (+7.00 points, p = 0.0249) and Llama 3.3 70B (+1.56 points, p = 0.00). We also introduce a blockchain-based commit-reveal protocol using Autonomous Economic Agents on an Ethereum-compatible ledger. In Phase 1, each judge records a one-way hash of its score and a secret salt before candidate identities are revealed. In Phase 2, the identity and raw score are disclosed and verified on-chain. This creates a tamper-evident audit trail that separates blind evaluation from post-hoc claims and reduces the verification burden on independent researchers and leaderboard operators. 

---
# Adaptive Two-Level Allocation of a Conserved Capacity Budget Across Locations and Service Classes 

**Authors**: Simone Mainardi, Kaushal Bansal, Prabhat Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.07747)  

**Abstract**: We study how to share a single conserved capacity budget across many locations and two service classes when demand is uneven, time-varying, and can exceed supply. The shape recurs: an origin's request-rate cap split across its edge locations, a licensed throughput cap across premium and standard tenants, or an egress budget between latency-critical and batch workloads. We present a two-level algorithm. The first level redistributes capacity within a class across locations by proportional deficit and excess redistribution; the second lends capacity elastically between classes when one has surplus and the other deficit. We prove it conserves the budget exactly, preserves non-negativity, and reaches a stable allocation in one iteration under stationary demand because it carries no per-cycle state, at O(KN) cost per cycle for K classes and N locations. We evaluate it defending a CDN's per-domain budget under volumetric attack, where the classes are confirmed-legitimate and not-yet-cleared traffic; across 8 contention scenarios on a 22-location topology it serves 66-93% of high-priority demand, competitive with a single-class linear-programming optimum, while never leaving capacity idle or over-committing whenever aggregate demand meets or exceeds the budget (the contention regime these scenarios evaluate). Two findings carry beyond the application. First, a throughput-maximizing objective is wrong under contention: a two-class LP maximizing total served load serves less high-priority load than our demand-proportional, reservation-respecting allocator in most scenarios, because it cannot tell that some load it serves is the contention. Second, inter-class borrowing earns its complexity under bursty load, improving high-priority service by 1.5 points (isolated by ablation), and is neutral under stationary demand. A 5-location prototype with real HTTP traffic validates the pipeline. 

---
# QuantumMind: Constraint-Grounded Agentic Reasoning for Speedup Analysis in Quantum Computing 

**Authors**: Yijing Zuo, Zhe Fu, Zihan Nie, Zhihui Zhu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07743)  

**Abstract**: Identifying a meaningful quantum speedup requires more than matching a classical problem to a familiar quantum primitive: the claim must preserve the task, respect access and output models, expose required promises, and remain within a defensible complexity scope. We present QuantumMind, an auditable agentic workflow for generating and conservatively screening quantum-acceleration hypotheses. A fixed sequence of typed, role-specialized actions formalizes the public task, analyzes structure and classical bottlenecks, matches a source-linked registry of quantum primitives and barriers, and constructs a scoped candidate scheme. A deterministic ten-check validator assigns the authoritative verdict; completed states are compiled into a Quantum Acceleration Evidence Graph and passed through a downward-only research screen that cannot strengthen the decision. We evaluate QuantumMind against seven task-adapted prompting and agentic controls on 582 identical open-discovery tasks. Under the frozen Open-Discovery Score (ODS), QuantumMind obtains 53.1 mean ODS, exceeding the strongest baseline by 17.3 points (48.2% relative), and wins 355 of 582 paired tasks against that baseline. It passes the graph audit on 99.8% of tasks, compared with 43.6% for the strongest baseline, and ranks first in all seven task families. The results indicate that typed state transitions and deterministic evidence control contribute beyond fluent generation alone. 

---
# Protecting patient privacy in clinical foundation models: Technical and legal perspectives 

**Authors**: Sana Tonekaboni, Lena Stempfle, Sasha Ronaghi, Corinna Coupette, I. Glenn Cohen, Emily Alsentzer, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2608.07705)  

**Abstract**: Clinical foundation models trained on large-scale patient data are increasingly used for decision support, screening, and public health. As deployment expands, privacy risk increasingly arises from model-mediated leakage, yet its prevalence and severity remain poorly quantified. Models can disclose sensitive training artifacts, enabling patient re-identification in ways not captured by data-handling controls alone. Existing frameworks, including HIPAA and GDPR, offer limited guidance for such indirect threats. We propose a practical framework for assessing privacy risk in clinical foundation models and illustrate realistic leakage scenarios across deployment settings, map them to legal regimes, and outline complementary technical and legal mitigations. Our analysis provides a context-aware risk assessment grounded in realistic usage to preserve the value of medical foundation models while rigorously safeguarding patient privacy. 

---
# Towards Researcher Agents for Knowledge-Graph Question Answering 

**Authors**: Tommaso Soru, Abdulsobur Oyewale  

**Link**: [PDF](https://arxiv.org/pdf/2608.07700)  

**Abstract**: Translating a natural-language question into a SPARQL query that can be executed against a large knowledge graph requires resolving lexical ambiguity, grounding surface terms in the target ontology, and producing graph patterns that are both syntactically valid and semantically faithful. We present an agentic text-to-SPARQL system that goes one step beyond static tool-using agents: a researcher agent that, after each round of inference on a validation set, proposes and tests changes to its own prompts, rules, and tool-orchestration code. We instantiate the loop on DBpedia, evolve nine successive versions of the agent driven by a low-cost reasoning model, and deploy the best-performing configuration with two stronger backbone models. The study yields three observations: (i) self-improvement converges quickly and then achieves 0.22 overall accuracy on the 2025 DBpedia validation set; (ii) the bottleneck is consistently in basic-graph-pattern predicate selection, not in SPARQL syntax or modifiers; and (iii) several benchmark items appear to penalise correct queries due to property ambiguity in DBpedia, suggesting that future Text-to-SPARQL benchmarks should be scored using a combination of machine translation and information retrieval metrics. 

---
# IntelliAudit: Using Large Language Models to Evaluate Audit Controls 

**Authors**: Allison Wilson, Sina Moradi Sabet, Diar Shakimov, Panteha Shahrivar, Mohammad Reza Bagheri, Dean Konenkamp, Mohammad A. Tayebi  

**Link**: [PDF](https://arxiv.org/pdf/2608.07688)  

**Abstract**: IT audits require auditors to judge whether heterogeneous organizational evidence satisfies semantic security and compliance controls. This judgment is difficult to automate because relevant evidence is distributed across policies, records, spreadsheets, and operational artifacts, and because audit conclusions depend on evidentiary sufficiency rather than keyword matching. We present IntelliAudit, a retrieval-grounded multi-agent system for IT audit evidence evaluation. Given a control and an evidence corpus, IntelliAudit retrieves relevant artifacts, generates an evidence-grounded assessment, challenges adverse findings, adjudicates disagreements, and produces an auditor-facing recommendation with cited evidence, rationale, missing-evidence analysis, and remediation guidance. We instantiate IntelliAudit on ISO/IEC 27001 and evaluate it across multiple simulated organizations using expert auditor review and audit-readiness user feedback. The evaluation shows that IntelliAudit can support control interpretation, evidence-grounded reasoning, and audit-preparation workflows, while also revealing the importance of human oversight for calibrating sufficiency judgments and correcting overly permissive recommendations. These results suggest that retrieval-grounded multi-agent systems can assist audit evidence review, but should remain decision-support tools rather than autonomous certification systems. 

---
# An Agentic AI Framework Overcomes Fundamental Limitations of Large Language Models for Glaucoma Detection from Fundus Photography 

**Authors**: Jalil Jalili, Hossein Taghizad, Anuwat Jiravarnsirikul, Christopher Bowd, Akram Belghith, Raheleh Kafieh, Christopher A. Girkin, Sally L. Baxter, Robert N. Weinreb, Linda M. Zangwill, Mark Christopher  

**Link**: [PDF](https://arxiv.org/pdf/2608.07651)  

**Abstract**: Large language models (LLMs) show promise in medical image interpretation but suffer from hallucination, limited accuracy, and run-to-run inconsistency. We developed and validated an agentic AI framework integrating LLMs with specialized deep learning tools for glaucoma detection from fundus photography. The workflow had three steps: (1) LLM initial assessment; (2) function calling to invoke specialized tools for image quality (QAModel, FundaQ-8), glaucoma classification (SwinV2-Tiny), and optic disc/cup segmentation (SegFormer-B0); and (3) LLM reflection integrating the initial impression with tool outputs. Two LLMs (Gemini 2.5 Flash, GPT-5.4 mini) were evaluated on two public datasets (ORIGA, n=100; RIM-ONE-v3, n=100) under uncropped and cropped fields of view; all images were independently graded by a masked fellowship-trained glaucoma specialist. The agentic workflow improved classification accuracy by 16 to 47 percentage points across all conditions, reaching within 6 points of the specialist; on RIM-ONE-v3 the best configurations matched the specialist accuracy of 88%. LLM-alone approaches failed in two ways: GPT-5.4 mini showed positive bias (sensitivity 95-100%, specificity 0-5%), while Gemini 2.5 Flash varied stochastically between runs; the agentic workflow corrected both. Cup-to-disc ratio error fell 15-50% (MAE 0.156-0.228 to 0.104-0.132), and correlation with specialist grading rose from weak (r=0.12-0.39) to moderate-strong (r=0.59-0.84). Run-to-run consistency rose from near-random (kappa as low as -0.01) to near-perfect (kappa up to 0.96). Integrating LLMs with specialized tools addressed key limitations of LLM-alone approaches, including over-diagnosis and run-to-run variability. Gains held for both LLMs, suggesting generalizability across backbones, and may signal a shift from monolithic models toward orchestrated multi-agent systems in medical AI. 

---
# Mendel Gödel Machine: Recursive Self-Improving Coding Agents via Comparative Evolution 

**Authors**: Changzhi Liu, Yilun Liu, Sikuan Yan, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.07645)  

**Abstract**: Self-improving coding agents that iteratively rewrite their own source code have demonstrated impressive performance on coding tasks. However, existing solutions generally derive self-modification from a single failure trajectory at a time, overlooking rich comparative signals available in the agent's expanding archive of past attempts. According to Mendelian principles of controlled inheritance, we introduce Mendel Gödel Machine (MGM). In addition to the general single-trajectory clonal mutation, MGM includes two new types of self-modification that better utilizes evidences accumulated: the reaction-norm mutation edits an agent based on its trajectories on multiple tasks simultaneously, and the cross-lineage hybridization edits an agent using the trajectory of a reference agent from another lineage on the same task. Under an additive fitness landscape model, we prove theoretically and demonstrate via controlled surrogate simulation that the new strategies facilitate a faster and better convergence over single-trajectory baselines. Experiments on SWE-bench and Polyglot confirm MGM's consistent improvement in performance, efficiency, and generalizability. 

---
# Contextual Value Alignment via Multilayer Combinatorial Fusion 

**Authors**: Yuanhong Wu, Djallel Bouneffouf, D. Frank Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07642)  

**Abstract**: Aligning large language models (LLMs) with human values remains a major challenge, especially for trustworthy AI. While existing approaches such as RLHF, CAI, and their variants have achieved promising results, they often rely on a single-agent framework and a unified reward system. This limits their ability to capture ethical pluralism, adapt to diverse moral contexts, and reflect the dynamics of multi-agent moral reasoning.
In this work, we propose a framework that utilizes multilayer combinatorial fusion for contextual value alignment (MCF-CVA). At the first layer of the framework, it instantiates multiple moral agents, each fine-tuned to represent a distinctive value. Their outputs are then expanded combinatorially using both score- and rank-combinations as well as average and weighted aggregations. These combined models are then reduced to the same number of initial moral agents. This expansion and reduction (EAR) process continues for multi-layers until a stopping criterion is reached.
The MCF-CVA framework leverages cognitive diversity between agents to mitigate conflicts and redundancies across multiple agents, producing responses that better reflect contextual human values. The framework using the EAR algorithm is performed on the dual architecture of Euclidean score space and Kemeny rank space. Empirical evaluations demonstrated that the proposed framework outperforms single-agent baselines, multi-agent single-layer results, and previous aggregation approaches on standard metrics, showing that the MCF-CVA framework provides a robust and effective mechanism for advancing contextual value alignment in LLMs. 

---
# Agent-MD: Selective LLM Intervention with Event-Driven Escalation for Stateful GCMC--MD Campaigns 

**Authors**: Yijie Wang, Zhen-Yu Yin, Zhenheng Tang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07637)  

**Abstract**: Long-running molecular simulation campaigns require repeated continuation from saved states, provenance-aware progression, adaptive assessment, and occasional interpretation of workflow conditions that cannot be resolved safely by fixed rules. Here, we present Agent-MD, a framework that places large language model (LLM) reasoning selectively at campaign construction and event-triggered review, while routine simulation, analysis, continuation, archiving, and state progression are handled by a persistent rule-based campaign agent using approved policies and explicit state records. Agent-MD was demonstrated in a grand canonical Monte Carlo-molecular dynamics (GCMC-MD) water-vapor desorption campaign comprising five montmorillonite systems and three sequential relative-humidity states (RH = 0.9-0.3-0.1). Across 15 system-RH states, the workflow completed 120 segmented simulation cycles with state-specific sampling lengths and provenance-aware restart inheritance. Routine production required no live reasoning-agent invocation, while one state reached a review boundary; two preserved incidents were subsequently evaluated through blinded reasoning-agent replay, which identified the underlying workflow problems and recommended appropriate follow-up actions. The simulations also revealed distinct composition-dependent low-RH responses, with Ca-bearing montmorillonite retaining more interlayer water and maintaining a larger basal spacing than the Na- and K-bearing systems, while the highest-charge Na system retained more residual water under dry conditions. These results demonstrate that long-running scientific workflows need not place every operation inside an LLM reasoning loop: selective reasoning can instead be combined with deterministic execution, structured evidence, and validated control handoffs to provide reproducible and auditable agent-assisted molecular simulation. 

---
# From Single Chatbots to Governed Agent Ecosystems: An Agentic AI Pattern Catalogue and Orchestration Framework for Mission-Critical Hospital Information Management Systems 

**Authors**: Manideep Dhar, Ritwik Singh, Sharat Chandra Kumar Manikonda  

**Link**: [PDF](https://arxiv.org/pdf/2608.07627)  

**Abstract**: Hospitals are racing to embed AI, while coping with the surge in adaptation of the technology in other industries, into the triage management, documentation, scheduling, and revenue-cycle workflows, yet most deployments remain as fragmented pilots that stall at the edge of production, exposing patients and institutions to operational fragility, ungoverned risk, and mounting technical debt. At the same time, the global AI-in-healthcare market is projected to exceed nearly USD 1 trillion by 2034, according to the report of Fortune Business Insights, amplifying the financial consequences of architectural missteps and failed scaling strategies. This research proposes a compliance-first Agentic AI pattern catalogue and orchestration framework, purposely built for HIMS, moving beyond the single LLM chatbots and towards a governed ecosystem of autonomous and semi-autonomous agents. The framework extends by adding (i) a taxonomy of Agentic roles, (ii) a formal risk-stratification model that maps each pattern to risk tiers, human-in-the-loop checkpoints, and governance hooks, and (iii) a unified orchestration runtime capable of coordinating multi-agent workflows across EHR/HIMS landscapes such as Epic, Cerner, and MEDITECH. Technically the framework combines vLLM-based inference, optimized paging memory, confidential computing, and MCP based on-premise deployment, enforcing end-to-end encryption and policy-as-code controls aligned with HIPAA, GDPR, the EU AI Act, India's DPDP and DISHA Acts, ISO 27001, ISO 27002, ISO 14971 and IEC 62304. We exhibit how the proposed architecture is capable and efficient to reduce the documentation time, integration effort, and AI pilot attrition while constricting the governance and auditability, offering hospital leaders and governing authorities an urgently needed blueprint to convert AI investment into sustainable clinical, operational, and financial ROI 

---
# Controlled Memory Interference in Continual LLM Agents 

**Authors**: Ao Ding, Hongzong LI, Shiqin Tang, Li Zhang, Liang Chen, Xuyang Chen, Zi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07622)  

**Abstract**: Long-term memory enables AI agents to maintain continuity across sessions, personalize behavior, and evolve through accumulated experience. Yet memory evolution is not simply a process of storing more information: new experiences may reinforce, revise, or interfere with existing memory states. Existing systems mainly emphasize memory construction and relevance-based retrieval, but several memories may remain simultaneously relevant while differing in state, temporal validity, or authority. We introduce Controlled Memory Interference (CMI), a controlled diagnostic and data-generation framework for studying how agent memory evolves under different memory relationships. Across controlled memory evolution, benign accumulation has limited effects, whereas relationship-specific interference sharply suppresses update plasticity with little stability gain, either by blocking target-memory exposure or by disrupting its downstream use. Lexical and Dense retrieval exhibit distinct interference pathways, while poisoning is more sensitive to update-authority cues than to recency alone. Beyond diagnosis, CMI provides targeted examples for interference-aware memory learning, improving the distinction between valid updates and interference-inducing memories while preserving performance on original memory tasks. These findings show that memory evolution is shaped not only by memory scale, but also by interactions among accumulated experiences. More broadly, memory interference emerges as an important factor for reliable continual agent memory systems. 

---
# CMU-Drive and V2V-VLA: Cooperative Multi-agent Unified Driving with Reasoning Benchmark and Vehicle-to-Vehicle Vision-Language-Action Models 

**Authors**: Hsu-kuang Chiu, Stephen F. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2608.07621)  

**Abstract**: Vision-Language-Action (VLA) models have recently achieved impressive performance for end-to-end autonomous driving, yet existing approaches are primarily designed for an individual single autonomous driving agent with limited support for cooperative perception, reasoning, and planning. We present Cooperative Multi-agent Unified Driving with Reasoning (CMU-Drive), a closed-loop end-to-end benchmark for evaluating cooperative autonomous driving with multiple connected autonomous vehicles (CAVs) operating in safety-critical driving scenarios with background traffic participants. We further propose Vehicle-to-Vehicle Vision-Language-Action (V2V-VLA), a cooperative VLA model that integrates cooperative driving into a single forward pass by jointly generating driving actions, future waypoints, language reasoning, and communication policies. Experiments on CMU-Drive establish the first benchmark and baseline for cooperative VLA driving and provide a foundation for future research on multi-agent, closed-loop, end-to-end cooperative autonomous driving. Our code, benchmark, and model checkpoint will be publicly released to facilitate open-source research. 

---
# TeXFix-Bench: An Empirically Grounded Multi-Format Benchmark for LLM-Based Document Source Repair 

**Authors**: Prajwal S. Venkateshmurthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.07617)  

**Abstract**: Scientific and technical writing depends on markup sources that must compile: LaTeX, Typst, and Markdown pipelines fail on missing delimiters, mismatched environments, broken imports, or package conflicts. Existing document-repair evaluations inject faults with ad-hoc edits that lack an empirical fault model. We present TeXFix-Bench, a multi-format benchmark for LLM-based full-source document repair grounded in a mined fault taxonomy. A Grounded-Theory study of localized hard-crash LaTeX faults from TeX Stack Exchange, GitHub commits, and package documentation (168 verified faults, dual open coding at $\kappa$=0.34) yields an 18-category taxonomy instantiated as DocMut: 48 AST-aware operators across three formats. A three-model cross-benchmark shows DocMut faults are 5.6-9.2 pp harder to repair than pattern-based mutations on the same seeds, and a real-error case study (88 mined human crashes, 67.0% repair success) brackets both synthetic sets from below. We construct 10,437 instances from 743 openly licensed seeds and evaluate seven LLMs under a fixed zero-shot protocol with provider-pinned routing, collecting 48,651 attempts at about USD 200 total inference cost. A complete 6,613-instance x 7-model balanced matrix confirms all rankings. A pinned engine gate yields a 27.5-point intention-to-treat compile spread (56.7-84.2%). Typst is markedly harder than LaTeX and Markdown. A restoration oracle over 28,129 compiling repairs shows that 13.6-18.5% of compiling repairs materially alter document text, and restoration rank diverges from compile rank: the model with the lowest compile rate restores content best among its successes. Compile success alone overstates repair quality. We release the taxonomy, DocMut, and all campaign artifacts. 

---
# The Field Knows: Cross-Dimensional Geometry from Navigation to Black Holes 

**Authors**: Chenghao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07566)  

**Abstract**: We introduce a continuous metric field framework trained by a single causal contrastive loss. The framework encodes a scene into coefficients of a fixed symmetric matrix basis, assembles them into a Lie algebra element, and exponentiates the result to a Riemannian or Lorentzian metric. Across dimensions, this field discovers the full spectrum of geometric structures: from obstacle-avoiding geodesics in robot navigation across planar and manipulator configuration spaces, to event horizons of black holes in Lorentzian spacetime. Extensive zero-shot generalization studies demonstrate that the field captures transferable geometric structure rather than memorizing specific configurations. In the black hole setting, the causal loss spontaneously evolves genuine black-hole-like structures with the correct Lorentzian signature. The same loss, the same architecture, and the same training protocol produce the full range of geometric phenomena across dimensions. The field knows geometry, and geometry knows physics. 

---
# An AI Scientist that Doesn't Drift: Taste, Structure, and Falsifiable Findings in a Quadruped Navigation Research Loop 

**Authors**: Yiwen Zhang, Eloise Zeng, Jaeha Lee, Tony Yue Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07542)  

**Abstract**: Autonomous research loops driven by large language models can run machine-learning experiments at scale but tend to drift toward local refinements of whichever metric they optimise rather than testing the hypotheses that motivate the experiments. We address this structurally and present an AI Scientist for studying generalisation in quadruped robot navigation policies in simulation. Building on the autoresearch paradigm of Karpathy, our loop adds three components: an immutable experiment card that pairs each iteration's prediction with its outcome under a fixed schema, so a falsified hypothesis cannot be retconned; specialised subagents restricted to mechanical roles; and kkanbu, a preference oracle that holds the user's research taste as a typed knowledge graph and is the only component permitted to make subjective judgements. To isolate the oracle we run the identical loop twice across eleven research streams, with and without kkanbu. Neither arm drifts: both falsify roughly three quarters of their own hypotheses, and the best trained policy comes from the oracle-less arm. What the oracle changes is direction, not score: it alone explores test-time adaptation, it authored the winning designs where its arm led, and it carried lessons across streams that the other arm repeatedly re-derived. The scaffold keeps the loop honest; kkanbu decides where it looks. 

---
# TREAT: Evaluating Access to Formal Knowledge across Equivalent Mathematical Representations 

**Authors**: Fateme Mazdarani, Carlos Toxtli  

**Link**: [PDF](https://arxiv.org/pdf/2608.07540)  

**Abstract**: AI systems increasingly operate between flexible input representations and formal objects used by downstream tools. A key challenge is recognizing when an unfamiliar formulation denotes a known formal object. We study this challenge through theorem recognition: given an equivalence-preserving transformation of a theorem condition, a model must recover the theorem identity associated with the standard statement. We introduce TREAT, a benchmark for evaluating whether large language models can recover known theorem identities from equivalence-preserving formula-level transformations. Rather than paraphrasing theorem text, TREAT changes the mathematical form of theorem conditions themselves, expressing known results through residual equations, witness statements, optimization identities, set relations, operator forms, and proof-intermediate characterizations. Starting from scraped theorem pages, we filter for entries with usable mathematical expression forms, extract canonical theorem conditions, and generate transformed variants with recorded assumptions and inverse mappings. The final corpus contains 737 theorem identities and 29,480 transformed rows. On a test panel, the best model retrieves the correct theorem identity in only 60.73% of cases. Other systems reveal different failure modes, including abstention, wrong detection, and malformed outputs. These suggest that theorem knowledge can be fragile under equivalent changes in representation. TREAT therefore provides a controlled testbed for evaluating representation-robust access to formal knowledge, with broader relevance to domains that require stable target objects, explicit equivalence relations, validation procedures, and auditable scoring. 

---
# When LLM Agents Negotiate: Private Information and Dynamic Bargaining in Supply Chains 

**Authors**: Chen Liang, Fasheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07538)  

**Abstract**: As LLM agents move from decision support to autonomous procurement, firms need to know whether delegated negotiators create value, divide it predictably, and avoid money-losing contracts. We study this in a canonical supply chain bargaining problem: a buyer with private demand information negotiates a quantity-payment contract with an uninformed seller. We benchmark nine LLMs from OpenAI, Google, and Alibaba against a validated Perfect Bayesian Equilibrium across 9,840 LLM-to-LLM negotiations. First, capability governs value creation. Agents agree in 98.9% of negotiations and capture 95.4% of first-best surplus undiscounted, but average 2.98 rounds against the benchmark's 1.25, and this delay erodes 21-34% of surplus. Capability also governs reliability: baseline models accept individually irrational contracts in 19.2% of cases, versus 0.0-0.6% at mid-tier and flagship, making automated profit verification the binding guardrail below that threshold. Second, surplus capture is relational. Provider identity predicts who captures surplus better than capability rank: self-play buyer shares average 40% for OpenAI, 50% for Google, and 70% for Alibaba's Qwen, an ordering that survives restricted communication and no discounting. Reversing which provider sells moves the division by 7-18 percentage points, and the capable Qwen flagship is the weakest cross-family seller: vendor choice is a first-order distributional decision. Third, the prompt is a strategic lever. Delegation separates the principal's economic patience from the agent's prompted strategic patience, a free deployment choice that is the single strongest driver of surplus division (90% of explained variance). Together these establish an equilibrium-referenced audit of AI agents along three dimensions: discounted efficiency, distributional profile, and operational reliability. 

---
# MetaSpace: Metamorphic Testing for Spatial Cognition in Embodied Agents 

**Authors**: Gengyang Xu, Dongwei Xiao, Yiteng Peng, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07533)  

**Abstract**: An embodied agent is an intelligent entity that interacts with its environment through a physical body. Currently, the evaluation of embodied agents primarily relies on two paradigms: (1) manually annotated Visual Question Answering (VQA) pairs and (2) high-level task completion metrics, such as success in navigation or manipulation. The former is labor-intensive and subject to variability in annotation quality. The latter may obscure critical vulnerabilities, allowing agents to complete tasks through suboptimal means or safety violations, thereby concealing safety risks and inefficiencies. Given that spatial cognition is the cornerstone for executing embodied tasks, there is a pressing need to assess whether embodied agents possess robust spatial cognition during task execution.
Inspired by metamorphic testing principles in software engineering, we propose MetaSpace, a novel framework designed to evaluate the spatial cognition of agents. By leveraging spatiotemporal multimodal states derived from real execution trajectories, MetaSpace automatically generates test cases based on predefined metamorphic relations (MRs) grounded in logical rules and physical laws. Crucially, we encode these MRs as executable rules in a logic programming language (Prolog). Violations of these relations indicate failures in spatial cognition. Our empirical evaluation across three embodied scenarios demonstrates that MetaSpace successfully detects 90,422 spatial cognition errors in state-of-the-art (SOTA) MLLM-driven agents. We introduce the Spatial Cognition (SC) score to quantify performance. Results indicate that all SOTA agents achieve average scores between 0.44 and 0.52, significantly lower than the human benchmark of 0.96. 

---
# Dynamic Coalition Formation and Communication Pricing in Skill-Based Agentic AI Systems 

**Authors**: Mojtaba Eslami  

**Link**: [PDF](https://arxiv.org/pdf/2608.07532)  

**Abstract**: Modern agentic AI systems combine multiple large language model agents with heterogeneous skills, yet most architectures either fix communication in advance or allow full broadcast. Both can be inefficient because token cost, latency, redundancy, and error propagation increase with the number of active agents and communication links. We model agent selection and communication as a cooperative game with task-conditioned net utility $U(C\mid x)=V(C\mid x)-\sum_{i\in C}c_i$, separating coalition-level costs from agent activation costs. We propose a marginal-value activation rule and greedy router, extend the model to optimize communication edges with per-edge costs, and use estimated Shapley values to predict which agents are worth contacting before and during execution. We connect the problem to submodular maximization and prove two limited guarantees: a curvature-refined bound for a monotone, cardinality-constrained special case, and a tight $1/2$-approximation, with a correction for signed objectives, for an unconstrained non-monotone case via double greedy. Neither guarantee applies directly to the main router, which remains a heuristic. We also prove a Shapley-submodularity sandwich bound linking the error of marginal-value routing to a per-agent diminishing-returns quantity. In synthetic experiments, greedy routing achieves $99.5%$ of brute-force-optimal utility while activating $1.96$ of $8$ agents on average, compared with $38.8%$ for full broadcast. Performance is robust to activation cost and redundancy weight but falls to $66%$ under strong violations of submodularity or noisy value estimates. We distinguish the framework from Shapley pricing, hedonic coalition formation, and communication-graph pruning, and propose evaluation on real multi-agent LLM benchmarks. 

---
# NL2SHACL-Bench: A Benchmark Suite for Natural Language to SHACL Translation 

**Authors**: Yuchen Zhou, Niels Bobet, Maribel Acosta  

**Link**: [PDF](https://arxiv.org/pdf/2608.07530)  

**Abstract**: SHACL is a core technology for validating the conformance of RDF knowledge graphs (KGs). Yet, authoring SHACL shapes requires technical expertise that most domain experts lack. Translating natural language requirements into SHACL (NL2SHACL) would lower this barrier. However, there is no dedicated benchmark for NL2SHACL, and evaluating generated shapes requires methods beyond string comparison, as semantically equivalent shapes can differ in serialisation and structure. To tackle these challenges, we present NL2SHACL-Bench, a benchmark suite for natural language to SHACL translation. Using NL2SHACL-Bench, we evaluate four state-of-the-art large language models (LLMs) for this task. Our results show that current LLMs are highly capable of generating syntactically valid SHACL, but still struggle to produce semantically equivalent constraints for complex logical and structural patterns. This indicates that NL2SHACL-Bench provides a meaningful basis for measuring advances in the NL2SHACL state of the art. 

---
# The Knowing-Saying Gap: When Probes See Errors that Confidence Misses 

**Authors**: Jyotin Goel, Ipshita Bandyopadhyay, Justin Shenk  

**Link**: [PDF](https://arxiv.org/pdf/2608.07528)  

**Abstract**: Linear probes detect corrupted context in language models with near-perfect accuracy, yet this does not translate into reliable failure prediction. The result is a dissociation with direct implications for deployment monitoring. Across multi-hop arithmetic chains, probes that detect corruption turn out to be uninformative about final answer correctness; models forced into structured confidence formats collapse to two values with indistinguishable error rates; and probe persistence across hops fails to separate correct from incorrect outcomes, refuting our pre-registered "persistence beats peak" hypothesis. This pattern of knowing but not saying generalises across model families including reasoning models. As a real-time monitor, probe-based interventions are sharply model and error-type dependent: branch-and-pick is net-positive across models and uniquely non-breaking on Llama-3.1-8B (4 rescued, 0 broken), while reprompt and replace-prior break correct traces at roughly the rate they rescue wrong ones. Probe-based monitoring is a necessary complement to verbalised confidence, but no single intervention dominates, and the deployable answer is model-aware, error-type-aware routing. 

---
# Training Variable Long Sequences with Data-Centric Parallel 

**Authors**: Geng Zhang, Xuanlei Zhao, Kai Wang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2608.07524)  

**Abstract**: Training deep learning models on variable long sequences poses significant computational challenges. Existing methods force a difficult trade-off between efficiency and ease-of-use. Simple approaches use static configurations that cause workload imbalance low efficiency, while complex methods introduces significant complexity and code change for new models. To break this trade-off, we introduce Data-Centric Parallel (DCP). Its core principle is to let the data itself drive the runtime. It achieves this by dynamically adjusting direct runtime settings (e.g., parallel size, gradient accumulation, recomputation) based on each batch's sequence length. Empirical results demonstrate that our method achieves up to a 2.88$\times$ speedup on 32 H200 GPUs. Designed for generalization, it can be integrated into any model with 10 lines of code. We anticipate this simple yet effective approach will serve as a robust baseline and facilitate future advancements in distributed training for variable long sequences. 

---
# Emotion in an active inference model of human driving 

**Authors**: Julian F. Schumann, Johan Engström, Ran Wei, Jens Kober, Martijn Wisse, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2608.07480)  

**Abstract**: Active inference has emerged as a principled framework for modeling adaptive behavior by balancing goal-directed action with uncertainty reduction. It has been successfully applied across biological and artificial systems, including recent work on human driving. However, existing active inference models of driving have yet to address an important determinant of behavior in traffic: affective state, which significantly influences decision-making. Prior work in non-traffic domains has explored active inference agents in which emotions are represented along the axes of valence and arousal in the circumplex model. However, this work has been limited to simplified settings with discrete state spaces. In this work, we propose an expanded formulation of valence and arousal that can be extracted from a more complex active inference model of driving with continuous states. In particular, we condition affective estimates not only on the current state but also on predicted future outcomes. We evaluate the proposed approach in two interactive driving scenarios and show that the resulting emotion signals correspond to affective patterns reported in similar scenarios. 

---
# Determinization in Structure Theories: A Unified Framework via Closure, Comparability, and Joint Admissibility 

**Authors**: Hai Hai Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07476)  

**Abstract**: We develop a formal framework for constructing canonical interpretations from plural structure theories. A structure theory is a triple T = ({\Sigma}, A, I) consisting of a signature, axioms, and an inference policy, whose admissible interpretation family collects all globally consistent assignments of structural conclusions.
We distinguish three levels of canonicalization: closure stabilization (per-seed convergence), global completion (seed-independent convergence), and determinization (a unique admissible interpretation). Non-determinism is classified into epistemic plurality (Type E) and structural plurality (Type S), with a refined Type S-strong subclass characterized by the absence of common upper bounds.
Two canonicalization mechanisms arise: operator-based completion and selector-based construction. We provide sufficient structural conditions under which these mechanisms exist, and show that pure inference-based completion reduces to a saturated closure operator under positive, non-retractive rules with an additional soundness condition. For Type E theories, closure stabilization is established, while full determinization depends on a global confluence property that remains open. For Type S-strong theories, determinization is achieved via canonical selection.
We further show that multi-level canonicalization forms a structurally non-commutative system via staged operators, and provide a conditional classification theorem reducing theory-intrinsic mechanisms to closure or selection. The framework also applies to LLM-assisted reasoning, where hallucination can be viewed as unsupported canonicalization. 

---
# Flow-by-Flow:Content-Judgment Bypass for Governing AI Output in High-Loss Domains 

**Authors**: Hiroki Naito  

**Link**: [PDF](https://arxiv.org/pdf/2608.07474)  

**Abstract**: Prior work showed that human-in-the-loop oversight becomes structurally untenable in high-loss domains when AI output velocity V exceeds human cognitive capacity C_max. The operative constraint, however, is not V alone but V x L, where L denotes per-item cognitive load. L consists of triage, judgment, and response, which respond asymmetrically to AI capability improvement. Triage cost does not decline as models become more capable, because semantic indeterminacy is inherent in general-purpose design. Response cost is invariant to accuracy improvements. Only judgment cost faces downward pressure, and this pressure often operates by inducing omission rather than genuine reduction. Capability improvement therefore restructures L rather than reducing it. Governance mechanisms based on evaluating whether AI output is correct either delegate that evaluation to AI and inherit hallucination risk, or delegate it to humans and face the V x L ceiling. We propose Flow-by-Flow, a governance paradigm that controls supervisory load without evaluating content. A cognitive cost score based on formal, countable features imposes nonlinear costs on high-volume production, while an institutional capacity cap keeps processing volume within C_max. We derive four design invariants for any content-judgment-bypass exceedance pathway: no content judgment, no scalable consumption of examiner capacity, identity-bound per-application friction, and no batch clearance. One reference implementation is discussed to show that these invariants are jointly satisfiable, while its practical difficulties are explicitly acknowledged. An illustrative Monte Carlo analysis across 1,000 parameter draws suggests that composite multi-metric flow control outperforms supervision reinforcement alone in 90.8% of trials. 

---
# Towards an Argumentative Foundation for Evaluative AI 

**Authors**: Xiang Yin, Tim Miller, Nico Potyka, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2608.07473)  

**Abstract**: Evaluative AI (EAI) has been recently proposed as a way to support human decision-making, not by producing a single recommendation, but by presenting competing hypotheses together with evidence for and against each. In this position paper, we advocate (computational) argumentation as a particularly suitable paradigm to provide a formal, computable foundation for forms of EAI that are explainable and contestable, setting the ground for a long-term research agenda towards distributed and human-centred EAI systems. 

---
# Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions 

**Authors**: Oluwanifemi Bamgbose, Simon Rosen, Jash Shah, Lindsay Devon Brin, Hoang H Nguyen, Anke Koelzer, Rachel Hansen, Tara Bogavelli, Fanny Riols  

**Link**: [PDF](https://arxiv.org/pdf/2608.09930)  

**Abstract**: Automated Text-to-Speech (TTS) evaluation methods (Mean Opinion Score (MOS) predictors and Audio Large Language Models (Audio-LLM) judges) are expected to reflect human perception, yet it is unclear how well they capture the distinct aspects of speech that listeners actually perceive. We deconstruct "naturalness" into a linguistically grounded annotation schema spanning 10 distinct perceptual dimensions, and use it to construct the first dimension-level meta-evaluation benchmark for TTS, comprising 860 utterances annotated by trained linguist raters. Results from benchmarking four MOS predictors and four Audio-LLM judges reveal that MOS predictors collapse onto acoustic signal quality, while Audio-LLM judges show selective, prompt-dependent detection that does not generalise across all dimensions. Neither class reliably captures a breadth of linguistically structured speech errors. Our dataset, annotation schema, and evaluation code are publicly released to support more targeted and interpretable TTS evaluation. 

---
# Multimodal Model Diffing for Feature Discovery and Control 

**Authors**: Hunar Batra, Lachin Naghashyar, Ashkan Khakzar, Philip Torr, Christian Schroeder de Witt, Constantin Venhoff, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2608.09928)  

**Abstract**: Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-level control, by causally removing or steering the discovered feature directions. We train multimodal SAEs for three MLLM families, LLaVA-MORE, PaliGemma 2, and InternVL3.5, and evaluate on visual-spatial understanding, multimodal safety, and OCR. MMDiff discovers sparse, causally specific features whose removal selectively degrades target behaviors by an average of 12% on spatial tasks and 17% on OCR, and reduces attack success rate by 24% on multimodal safety attacks, with no impact on VQA performance. Steering these features improves spatial and OCR accuracy by +3.6% and +1.8% on average over a standard single-layer steering baseline. These results show that multimodal SAEs can serve not only as interpretability tools, but as mechanisms for auditing, steering, and controlling MLLMs behavior toward safer and more capable generations. 

---
# From Values to Benchmarks: Evaluating Large Language Models for Governmental Use in Dutch 

**Authors**: Laurens Samson, Iva Gornishka, Gossa Lô, Yuki M. Asano, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2608.09925)  

**Abstract**: Large language models are increasingly being deployed in governmental settings, yet few existing evaluation frameworks jointly reflect the values of public administration and the linguistic requirements of non-English contexts. We present the "Grip on LLMs" framework, a systematic evaluation suite for Dutch governmental use developed in collaboration with domain experts from a major Dutch municipal organisation. Through an advisory board process, user research, and a survey of the users of a civil-servant chatbot, we identify six evaluation dimensions (factuality, honesty, social bias, energy consumption, cost, and training data transparency) and operationalise them into a benchmark suite covering more than 30 multilingual and Dutch-specific models. Our results reveal that no single model excels across all dimensions, and that trade-offs are unavoidable: higher quality consistently comes at greater environmental impact and financial cost, while bias remains largely independent of both. We further find that factuality (whether a model answers correctly) and honesty (whether a model acknowledges what it does not know) are governed by distinct properties, with high factuality not implying high honesty. To make these findings actionable for non-technical audiences, we release a publicly accessible, user-friendly model overview designed for the full range of stakeholders involved in governmental LLM selection, from engineers to policymakers. 

---
# Fusion Training for Mathematical Generalization in Large Language Models 

**Authors**: Congfeng Cao, Pengyu Zhang, Jelke Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2608.09893)  

**Abstract**: Thinking Mode Fusion (TMF) enables large language models to support both concise responses and long-form reasoning by unifying a non-thinking mode and a thinking mode within a single model. However, its training dynamics, including the \emph{data ratio} and \emph{training schedule} between the two modes, remain underexplored. In this work, we present a systematic study of TMF by analyzing the effects of the training schedule and data ratio between thinking and non-thinking modes. Focusing on mathematical problem solving, we construct a benchmark with multiple thinking-to-non-thinking data ratios and three training schedules. Our results reveal an asymmetric interaction between the two modes: increasing the ratio of non-thinking supervision reduces the accuracy of the thinking mode. We further show that different training schedules modulate this trade-off and that the optimal schedule depends on the data ratio. Finally, we quantify a negative correlation between non-thinking and thinking mode supervision, highlighting an inherent tension between these two modes. These findings provide practical guidance for designing effective TMF training settings. All code and data are released to support further research at: \href{this https URL}{\textbf{Fusion Bench}}. 

---
# BDH-CQ: In-Context Learning with Recurrent Latent Reasoning 

**Authors**: Björn Engdahl, Adrian Kosowski, Jan Chorowski, Zuzanna Stamirowska, Przemysław Uznański, Junlin Jiang, Rohan Phadke, Remigiusz Kinas, Richard Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2608.09888)  

**Abstract**: We introduce BDH-CQ, a reasoning model that combines in-context learning with recurrent latent reasoning. Inputs presented at inference time continuously update the model's recurrent memory; the model then solves a query through iterative computation in a high-dimensional latent space, without verbalizing its intermediate reasoning. We evaluate the model on the public ARC-AGI-1 evaluation set and use controlled ARC-like interventions to study what it learns from demonstrations, how consistently it applies an inferred transformation, and which concepts remain difficult. A 150M-parameter configuration reaches 29.5% pass@2 at a computed inference cost of \$0.0007 per task. This operating point breaks through the previously reported ARC-AGI-1 cost-accuracy Pareto frontier, establishing a new state of the art in benchmark cost efficiency. 

---
# Energy-Structured Latent World Models with Neural Time Fields for Physically Constistent Open-World Motion Planning 

**Authors**: Yapeng Liu, Yuanzhao Zhai, Bo Ding, Huaimin Wang, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09876)  

**Abstract**: Physically consistent motion planning remains a fundamental challenge in embodied AI, as generated trajectories must strictly conform to real-world execution dynamics. While latent world models offer a promising approach by predicting these dynamics, existing methods learn unconstrained future representations where absorbed physics remains implicit. Therefore, they fail to form reusable physical knowledge, which compromises reliability in unpredictable open-world navigation. To address this, we propose a novel Energy-Structured Latent World Model (ELWM). Our key idea is to structure the ELWM latent state to explicitly carry energy and momentum, ensuring strictly causal transitions via dissipation and control ports. Trained on multimodal RGB-D and inertial interaction histories, our model guarantees physically consistent predictions. We further implement this for motion planning by constructing Physics-Conditioned Neural Time Fields (PC-NTF), a key technical cornerstone that integrates ELWM into an arrival time field via the Eikonal equation to yield a physically-informed navigation policy. Across held-out scenes, our evaluation reveals significant improvements. Compared to generic latent models, PC-NTF reduces 0.8-s motion-prediction NRMSE from 0.36 to 0.29. Against Active Neural Time Fields, it improves navigation success from 81.3% to 89.7% and SPL from 0.64 to 0.73, while cutting the physical collision rate from 12.1% to 5.8% and the Eikonal residual from 0.083 to 0.031. Beyond these targeted gains, our results demonstrate that embedding explicit physical structures into latent spaces intrinsically bridges the gap between predictive world models and safe, dynamically feasible motion planning. 

---
# Sci-VBench: Evaluating Knowledge- and Reasoning-Intensive Video Generation in Science Domains 

**Authors**: Diandian Zhang, Tingyu Song, Lin Fu, Zheyuan Yang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.09873)  

**Abstract**: We introduce Sci-VBench, a comprehensive benchmark for evaluating knowledge- and reasoning-intensive video generation across scientific domains. It contains 1,253 expert-annotated examples spanning 60 subjects across four core disciplines: Natural Science, Healthcare, Humanities & Social Sciences, and Engineering. Each example requires models to generate temporally rich videos that demand scientific reasoning and knowledge-grounded synthesis, going beyond surface-level visual plausibility. We further establish a rubric-based evaluation protocol. Our analysis shows that, under this protocol, both non-expert human evaluators and MLLM-as-Judge systems can achieve relatively high agreement with expert judgments, supporting reproducible evaluation at scale. We benchmark 16 frontier proprietary and open-source models and find that, while automatic perceptual-quality scores cluster tightly across systems, performance on Prompt Grounding and Scientific and Causal Correctness varies substantially, with a pronounced proprietary-open-source gap. These findings show that advances in visual realism have not yet translated into reliable modeling of scientific and causal dynamics. 

---
# Stealing Reasoning Traces from Proprietary LLM APIs 

**Authors**: Alexander Panfilov, David Schmotz, Ilia Shumailov, Luca Beurer-Kellner, Joachim Schaeffer, Ameya Prabhu, Jonas Geiping, Maksym Andriushchenko  

**Link**: [PDF](https://arxiv.org/pdf/2608.09867)  

**Abstract**: Leading large language model providers now conceal their models' step-by-step reasoning, or chain-of-thought, to protect intellectual property and limit information leakage. Rather than storing these traces server-side, providers return them to the client as blocks of encrypted text, which the client passes back with each subsequent request. Building on prior research, we identify an architectural vulnerability: these encrypted blocks are fully compatible and interchangeable across different sessions, users, and models within a provider's ecosystem. We exploit this compatibility to develop a scalable decryption jailbreak. By injecting an encrypted reasoning trace from a given model into a weaker, and less safeguarded model from the same provider, we force it to decode and output the trace verbatim in plaintext, without ever jailbreaking the more capable model directly. This vulnerability enables four distinct attack vectors. First, it circumvents anti-distillation mechanisms, allowing adversaries to extract a proprietary model's reasoning, as we demonstrate across Anthropic, OpenAI, and Google. Second, it allows for large-scale private data extraction. Developers frequently share session logs publicly, unaware of contents of the encrypted blocks. By decoding 315,320 reasoning blocks scraped from public repositories, we recovered 367 Personally Identifiable Information (PII) artifacts and 182 credentials. Third, it inadvertently reveals hazardous information hidden within the reasoning process, even in cases where the model's final, visible output safely rejects a malicious request. Fourth, attackers can leverage this flaw to execute invisible prompt injections, embedding malicious payloads entirely within encrypted blocks to poison public agentic rollouts. Following responsible disclosure, we propose concrete cryptographic and system-level mitigations to secure client-side reasoning. 

---
# Agentic Harnesses: LLM-Driven Verification Layers for Robot Autonomy 

**Authors**: Rohan Bhagra, Mahantesh Halapannavar, Uddhav Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09857)  

**Abstract**: Advances in advanced artificial intelligence tools have sparked research in robot autonomy, but the development of such systems has largely focused on execution rather than verifying the feasibility actions planning models propose. Like general-purpose LLMs, robotics planning models carry risks: biased toward user-specified goals, they may suggest actions misaligned with scientific ethics, they may be unsafe due to an inability to "remember" prior safety risks, or they may be vulnerable to adversarial attacks on the autonomy ecosystem. We propose a LLM-driven verification layer between planning and execution to evaluate action permissibility. Our LLM-as-a-Judge ensemble combines chain-of-thought reasoning across models and synthesizes those expert judge outputs, mirroring a combination of a mixture of experts and self-consistency approach. This layer serves as middleware, gating plans from the server's planning module before they reach the MCP server and therefore the robot's low-level controls: plans are approved, rejected for reformulation, or escalated for human review. With this system, we achieve near 85% precision across accept/escalate/reject categories 97% containment of adversarial attacks, with negligible errors between accepting and rejecting tasks, and errors mostly manifesting at the escalate boundary. 

---
# Multi-Agent AI Safety as an Institutional Design Problem 

**Authors**: Abdullah X  

**Link**: [PDF](https://arxiv.org/pdf/2608.09828)  

**Abstract**: AI agents increasingly work inside systems that govern how they delegate tasks, move information, execute actions, and use shared resources. Recent work already shows that deployment rules can change collective behavior. Here we ask which parts of an AI institution produce safety and how they do it. This is the first paper from POLIS, an ongoing research programme studying algorithmic institutions for multi-agent systems. We report a frozen 5,280-episode study suite. The main pre-specified delegation experiment spans four model families; a targeted high-conflict diagnostic adds three additional model endpoints. In matched structured workflows, the model sees different rule formulations and guards consult different authority states. We also vary the attractiveness of the immediate compliant internal/self fallback and allow blocked workflows to continue. A detailed constitutional prompt produces 0/384 realized violations. A provenance-aware executable guard also produces 0/384, although it blocks prohibited attempts in 51/384 episodes; 44/51 of those episodes later complete safely. The local-state guard's failures concentrate in scenarios where an ordinary transformation changes visible policy while originating authority stays fixed. In matched laundering scenarios, that guard admits violations in 22/96 episodes and provenance enforcement in 0/96 (p = 4.77 x 10^-7). A separate resource-allocation experiment shows that revealing the numerical value of an otherwise identical cap changes agent requests. In these structured workflows, the same final violation rate can hide very different mechanisms. The rule itself is only part of the institution. The authority state the system trusts matters, and so does the path available after a block. 

---
# Distill Skills into Weights, Not Prompts: Abstract Skills as Privileged Signals for On-Policy Self-Distillation 

**Authors**: Yubo Jiang, Fengying Xie, Zhiguo Jiang, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09826)  

**Abstract**: Reinforcement learning with verifiable rewards yields no group-relative signal when rollout groups are uniformly correct or uniformly wrong, which account for 63.0-68.0% of groups in our experiments. We propose SKALD (Skill-Anchored Latent Distillation), an on-policy self-distillation framework that uses two context views of the same Qwen3-Base model: a question-only student and a teacher conditioned on an abstract, explicit-answer-filtered skill card. The student is trained on its own prefixes, transferring the skill-induced advantage into shared parameters without privileged input at test time. To stabilize context-induced distribution mismatch, SKALD employs an annealed exponentially tilted objective that downweights teacher-preferred tokens with very low student likelihood; as the tilt vanishes, it converges to teacher cross-entropy and recovers the forward-KL student gradient. An empirical gate activates distillation only when verified rollouts estimate a positive teacher advantage. Across five held-out mathematics benchmarks, SKALD improves overall avg@8 over GRPO by +2.46, +4.85, and +12.01 at 0.6B, 1.7B, and 4B, respectively. At 1.7B, zero-variance-only distillation recovers 84.7% of the full gain, while SKALD remains +4.06 above FLOP-matched GRPO and exceeds contextual skill exposure by +3.77. These results show that abstract skills provide dense supervision where group-relative rewards become uninformative. 

---
# MedPixel: A Unified Pixel-Language Model for Medical Reasoning and Segmentation 

**Authors**: Haoyu Yang, Meixing Shi, Zengjie Chen, Haoran Sun, Haitao Leng, Xiaoming Shi, Yuxiang Cai, Yankai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09818)  

**Abstract**: Reliable medical image understanding requires models to connect clinical language and visual reasoning with pixel-level grounding. Yet medical vision-language models often lack precise localization, whereas medical segmenters typically rely on explicit target categories or precise spatial prompts. This divide is reinforced by a supervision mismatch: segmentation datasets provide precise masks but little language supervision, whereas medical vision-language data rarely pair language with dense spatial annotations. To address this gap, we present MedPixel, a unified medical pixel-language model built around a shared language--mask interface. To provide scalable supervision, we introduce MedPLG-440K, comprising approximately 440K pixel-language task samples constructed through a clinically motivated synthesis process without external LLM annotation. MedPixel is trained with joint multi-task supervised fine-tuning followed by Pixel-Level Preference Optimization, which uses ground-truth masks as offline verifiers to derive response preferences from mask quality. MedPixel supports a broad spectrum of tasks spanning explicit grounding, implicit reasoning, spatial interaction, grounded explanation, and medical VQA. Across this task spectrum, MedPixel achieves strong performance in both pixel-level prediction and response generation, together with effective zero-shot transfer to external grounding benchmarks and robustness to imperfect spatial prompts. Code and model checkpoints will be released at this https URL. 

---
# Parameter Exploration for RLVR via Variational Learning 

**Authors**: Vatsal Venkatkrishna, Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2608.09805)  

**Abstract**: Exploration has been a focus of reinforcement learning research for a long time. Recently, there has been growing evidence that it is also an important ingredient in LLM reinforcement learning recipes that can significantly impact downstream performance. Many existing methods control exploration in the action-space, for example, using temperature scaling. However, these methods cannot reorder tokens but only influence the variance in the output distribution. This limits exploration and can lead to divergence or stalled training. Here, we investigate parameter-space exploration, where rollouts are generated by sampling different policies from a posterior that may each explore different rollouts. Sampling less or more diverse policies is then a complementary control lever over exploration. We introduce a family of methods called Perturbed Parameter Policy Optimization (3PO) which use different sampling strategies and different rollout grouping for reward estimation. Experiments on OLMo-3-1025-7B and Qwen2.5-Math-7B across mathematical reasoning and code generation tasks show that these approaches consistently improve average downstream performance over standard GRPO at a near-identical FLOPs cost. Moreover, using multiple parameter samples consistently produces fewer zero-advantage groups and malformed or incorrect rollouts during training than GRPO and action-space baselines. Overall, our work presents evidence that parameter-space exploration can improve reinforcement learning for LLMs. 

---
# Modern Backbones Improve Multi-task DETR for Mammography Classification and Lesion Localization 

**Authors**: Dinh Tan Nguyen, Quang-Hien Kha, Le-Hoang Nguyen, Minh-Toan Dinh, Xuan-Huy Nguyen, Dac Phu Ho, Cao Truong Tran, Sai Ho Ling, Lan T Ho-Pham, Liem Pham, Nguyen Quoc Khanh Le  

**Link**: [PDF](https://arxiv.org/pdf/2608.09801)  

**Abstract**: Joint exam-level prediction and candidate-region localization may improve the usefulness of AI support in mammography. We study this setting using a multi-task DETR framework, where shared representations support both image-level malignancy prediction and lesion localization, and evaluate its performance on OPTIMAM and a biopsy-confirmed SGM1k cohort. Across both datasets, modern backbones consistently outperformed older ResNet-style features, with ConvNeXtV2 and DINOv3 giving the strongest overall results, whereas MambaVision was less competitive. On OPTIMAM, ConvNeXtV2 achieved the best overall performance, reaching 97.96% AUC, 99.89% sensitivity, 25.08% mAP@.5, and 74.38% recall@.25. On SGM1k, DINOv3 gave the strongest overall results, with 90.97% AUC, 86.28% sensitivity, 82.00% specificity, 27.04% mAP@.5, and 77.32% recall@.25. These findings suggest that backbone quality is a critical factor in effective multi-task mammography, with ConvNeXtV2 emerging as a particularly strong and well-matched CNN backbone for mammography in this framework. 

---
# KGCaRe: Explainable Complex Conditional Question Answering using Automatic Knowledge Graph Construction and Context Retrieval with LLMs 

**Authors**: Ghanshyam Verma, Simanta Sarkar, Devishree Pillai, Hotaka Shiokawa, Yourong Xu, Fiona Veazey, Peter Hubbert, Hui Su, Paul Buitelaar  

**Link**: [PDF](https://arxiv.org/pdf/2608.09779)  

**Abstract**: Answering complex conditional questions using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) remains a challenge, particularly in domain-specific contexts where general-purpose LLMs and RAG tend to underperform. We hypothesize that augmenting RAG with unstructured and structured knowledge, extracted from both documents and knowledge graphs (KGs), can improve reasoning and answer accuracy for such tasks.
To test this, we propose KGCaRe, a hybrid approach that combines neural retrieval with symbolic reasoning over LLM-generated KGs. KGCaRe constructs a KG from documents using a multi-prompt extraction strategy and stores it in a graph database. Simultaneously, the documents are embedded into a vector store to enable neural retrieval. KGCaRe performs innovative iterative graph traversal guided by the LLM to extract relevant triples, prune irrelevant information, and uses additional clue entities to traverse the graph again if the initial traversal does not provide satisfactory context to generate the answer. The relevant triples extracted from the KG in path form, along with semantically retrieved text passages, are then fed into custom KGCaRe prompts to generate answers to the complex conditional questions with explanations.
We evaluate KGCaRe on two complex conditional QA datasets. Our results on these datasets show that KGCaRe consistently outperforms existing baselines, including Vanilla LLM, Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG, and HybridContextQA, across multiple LLMs such as Mistral, Mixtral, GPT-3.5, and GPT-4o. We publicly release the software pipeline that we developed to implement the proposed KGCaRe approach. 

---
# Cultivar: A Contrastive and Locale-Oriented Translation Benchmark for Investigating Contamination and Localisation Robustness 

**Authors**: Pinzhen Chen, Koel Dutta Chowdhury, Xiaoya Xu, David Tan, Doreen Osmelak, Ona de Gibert, Ariun-Erdene Tumurchuluun, Ashok Urlana, Fedor Sizov, Hale Sirin, Jesujoba Alabi, Karrar Talib Abed, Mateusz Klimaszewski, Nikolay Bogoychev, Niyati Bafna, Patricia Schmidtova, Preksha Manjunath Shanbhag, Sherrie Shen, Vilem Zouhar, Vivek Iyer, Yasser Hamidullah, Yusser Al Ghussin, Zheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.09766)  

**Abstract**: Multilingual translation benchmarks are typically sourced in English and translated into other languages, treating language pairs as the unit of evaluation---a design that is prone to contamination over time and overlooks locale and cultural considerations. We therefore advocate for source-contrastive evaluation and instantiate it with Cultivar, a localised subset of FLORES, which enables locale-specific translation evaluation. When paired with unlocalised counterparts, performance discrepancy allows the probing of data contamination and localisation robustness. We benchmark 32 open-weight models and find that MT-specialised models are less robust, a few models potentially overfit FLORES, and models tend to translate US content better than that of other locales, regardless of language. 

---
# MoNo: Multiscale Optimal Transport Neural Operator for Solving PDEs on General Geometries 

**Authors**: Zijiang Yang, Xiaomeng Wu, Dongmei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09764)  

**Abstract**: Transformer-based neural operators have achieved substantial progress in solving Partial Differential Equations (PDEs) by projecting spatial observations into compact latent tokens and learning physical interactions in latent spaces. However, we reveal that existing learnable projection mechanisms cannot ensure stable and balanced assignments from observation points to latent tokens, causing some latent tokens to be over-assigned while others remain underutilized. This limitation further restricts the design of hierarchical architectures, as assignment imbalance is continuously inherited and amplified across latent spaces, eventually causing severe token collapse in deeper spaces. To address these issues, we propose MoNo (Multiscale Optimal Transport Neural Operator), a progressive multiscale neural operator that efficiently solves PDEs on general geometries through stable latent-space construction. At its core is CoTAP (Cross-scale Optimal Transport Assignment and Projection), a novel latent-space construction method that formulates cross-space assignment between adjacent spaces as an entropy-regularized optimal transport problem, thereby constructing balanced bidirectional projections and stable latent spaces. CoTAP also ensures stable information transfer across multiple latent spaces, further enabling multiscale architectures on general geometries, which in turn support more efficient learning of long-range physical interactions. Extensive experiments demonstrate that MoNo outperforms existing state-of-the-art neural operators in both prediction performance and computational efficiency. Code is available at this https URL. 

---
# Defining Decentralization: An Ontological Perspective 

**Authors**: Jakub Kacper Szeląg, Aydin Abadi, Mohammad Naseri  

**Link**: [PDF](https://arxiv.org/pdf/2608.09748)  

**Abstract**: Decentralization as a concept in computer science has existed for over half a century. Despite its fundamental role across domains such as security, distributed computing, artificial intelligence, cloud infrastructures, and Internet of Things (IoT) architectures, there remains no universally accepted definition of decentralization applicable across computer communication systems. This has become increasingly problematic with the emergence of decentralized AI and machine learning paradigms, including collaborative training, distributed inference, blockchain-based, and agentic AI, where decentralization is often treated as a core design objective. Meanwhile, existing approaches frequently conflate decentralization with related notions such as distribution of trust or specific implementation paradigms. Such ambiguity creates inconsistencies in system analysis, limits comparability between works, and weakens the rigor of formal reasoning surrounding communication architectures and protocol design. In this work, we define this research gap as the Decentralization Problem.
We analyze the formal-semantic, epistemological, and pragmatic foundations of decentralization and introduce a graph-based ontology defining it as both relational and subject-specific property of computer communication systems. The framework formally distinguishes decentralization from distribution and supports evaluation through two novel metrics: Void Tolerance and Imperviousness. We also provide a browser-based implementation that enables automated classification and metric computation of arbitrary systems. Instantiations to federated learning and blockchain architectures show consistent, comparable assessments where existing definitions produce incomplete or contradictory conclusions, providing a domain-independent foundation for analysing decentralization across heterogeneous systems. 

---
# SR-OPSD: Self-Referenced On-Policy Self-Distillation 

**Authors**: Zhuo Sun, Entong Li, Yanlong Zhao, Xiaoyuan Cheng, Wenxuan Yuan, Kaiyu Li, Che Liu, Huihang Liu, Harrison Bo Hua Zhu, Li Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09745)  

**Abstract**: On-policy self-distillation (OPSD) converts feedback into dense token-level supervision on trajectories generated by the policy to be optimized, providing a useful complement to reinforcement learning with sparse outcome rewards. However, the self-teacher policy used in OPSD is typically a stop-gradient or exponential-moving-average copy of the policy conditioned on additional context information, and thus co-evolves with both the student policy and its on-policy context distribution. Directly matching such a moving target with a fixed projection objective can lead to unstable optimization or excessive distributional concentration. This nature of OPSD motivates the proposed \emph{Self-Referenced On-Policy Self-Distillation (SR-OPSD)}. At fixed student-generated contexts, a token-level variational characterization identifies the effective distillation target as a geometric interpolation between the self-teacher policy and a reference policy. Meanwhile, we use the Rényi divergence family to generalize the projection geometry. This formulation separates \emph{where} the adaptive target is placed from \emph{how} the student is projected toward it: the interpolation coefficient controls underlying target, while the Rényi order controls the projection geometry and its sensitivity to token-level density ratios. Extensive experiments across scientific evaluation, mathematical reasoning, and coding generation tasks with multiple large language models show that SR-OPSD achieves the state-of-the-art or competitive performance across various settings. 

---
# Rethinking Factor Sharing in Federated LoRA: A Rank-Aware Adaptive Approach 

**Authors**: Xinyi Xu, Bingnan Xiao, Shuang Qin, Gang Feng, Tony Q.S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2608.09742)  

**Abstract**: Low-rank adaptation (LoRA) represents large language model (LLM) updates with two compact matrix factors, i.e., $A$ and $B$, providing an efficient way to fine-tune large models in federated learning paradigm. Inspired by the asymmetric roles of the LoRA factors, we study whether $A$ should be shared across clients while $B$ remains client-specific (Share-A/Local-B), or whether $B$ should instead be shared while $A$ remains client-specific (Share-B/Local-A). With a least-squares surrogate, we reveal that Share-A/Local-B requires the client-specific LoRA update matrices to use a common rank-$r$ input-side space, whereas Share-B/Local-A requires a common rank-$r$ output-side space. The two strategies therefore incur different projection residuals, indicating that the preferred strategy is the one with the smaller aggregate residual across clients. With this insight, we propose Federated Adaptive Factor Sharing Low-Rank Adaptation (FedAS-LoRA), which selects the sharing side before training to enhance fine-tuning performance. To enable adaptive factor selection before training, we design a Rank-Aware Shared-Subspace Sufficiency (RSS) metric, which effectively assesses whether a shared rank-$r$ input subspace is sufficient for the local data distributions using representations extracted from a frozen LLM backbone. Experiments across different tasks, data distributions, LoRA ranks, and participation settings confirm the effectiveness of RSS and the superior performance of FedAS-LoRA. 

---
# ColluSkill: Adversarial Cross-Skill Composition for Evading Agent Skill Scanners 

**Authors**: Puyu Zeng, Simeng Qin, Jingzhi Li, Ju Jia, Zheli Liu, Xiaojun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.09732)  

**Abstract**: Agent skills are emerging as an important attack surface in LLM-based agent systems. Through an empirical study of existing skill scanners, we find that current defenses mainly inspect individual skills, leaving risks from cross-skill composition insufficiently examined. This creates a practical blind spot: multiple locally plausible skills may pass security checks while collectively forming a harmful workflow during agent execution. To investigate this threat, we propose ColluSkill, a collusive multi-skill-chain attack framework that decomposes a complete malicious intent into interdependent sub-payloads embedded in independently packaged skills. The attack does not rely on any single malicious skill, but emerges from the ordered composition of locally plausible behaviors through contextual dependencies, artifact passing, and execution handoffs. ColluSkill further employs LLM-based chain planning and scanner-feedback refinement to preserve chain-level attack semantics while reducing suspicious signals in individual sub-skills. To defend against such attacks, we propose ChainGuard, a context-aware skill-chain scanner that jointly analyzes a candidate skill and the skills already installed in the agent environment. ChainGuard reconstructs cross-skill dependencies, artifact flows, capability compositions, and downstream behaviors to identify risks that emerge only at the workflow level. Experiments on six representative skill scanners show that ColluSkill achieves an average attack success rate of 96.0% and consistently outperforms the evaluated single-skill and multi-skill attack baselines. Meanwhile, ChainGuard reduces the attack success rate to 22.5% while allowing 99.5% of benign workflows to pass, highlighting the importance of chain-level security analysis for agent skill ecosystems. 

---
# How Do Large Language Models Judge Social Attraction? Evidence from Theory-Grounded Persona Ratings Across Multiple LLMs and Humans 

**Authors**: Hasan Mahmud, Khawaja Abaid Ullah, Mohammad Javad Khojasteh, Jamison Heard, Prabu David  

**Link**: [PDF](https://arxiv.org/pdf/2608.09717)  

**Abstract**: Large language models (LLMs) are increasingly used to perform subjective evaluations traditionally made by humans, yet their validity as social judges remains unclear. This paper examines whether LLMs can assess social attraction from theory-grounded persona profiles constructed from ten psychological and relational constructs and organized into three tiers: socially attractive, socially mixed, and socially unattractive. We examine LLM ratings in two studies and compare them with human judgments in a third study. In Study 1, 34 LLMs rated 12 profiles across three repeated runs. Although some models tended to give higher or lower ratings overall, they showed strong stability across runs, consistent three-tier ordering, and high agreement in relative profile ordering. Study 2 examined sensitivity to gender presentation using six matched name-and-pronoun profile pairs and a separate pronoun-only test with a gender-neutral name, finding no significant effects in either analysis. In Study 3, 198 human participants evaluated the six matched profiles from Study 2. Their ratings reproduced the three-tier structure and followed a profile ordering consistent with that of the LLMs. However, LLMs rated attractive profiles more positively and unattractive profiles more negatively than humans, while neither group showed a significant overall effect of gender presentation. 

---
# Evaluating Generative Time-Series Models on Data with Point Masses 

**Authors**: Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09692)  

**Abstract**: Many of the series that generative time-series models are benchmarked on place a large probability mass on a single value --- it does not rain, no ride is requested, no part is ordered. We report what happens when such data is evaluated carefully. First, the standard rolling-origin protocol can score a model on a window whose atom structure bears no resemblance to the dataset: on one benchmark the dataset is $42\%$ zeros and the evaluation windows are $13\%$, on another $47\%$ against $5\%$. This is not a cosmetic problem --- it reversed one of our own conclusions, turning the strongest occurrence model in our study into what looked like a cautionary tale. Second, we give a control in which CRPS is invariant \emph{by construction} while the temporal coupling is destroyed, which measures exactly how much that coupling contributes to a chosen statistic. Third, benchmarking seven models on a matched protocol over five seeds, an autoregressive hurdle beats a conditional flow on five of six datasets, by up to a factor of $153$, while the flow's own occurrence statistics vary by up to $62\%$ across training seeds and every baseline is deterministic. Finally, the model ordering is not the same under five different occurrence statistics, and the two that do not share a construction agree with each other least. 

---
# Confusion-Geometry Rebalancing for Long-Tailed Adversarial Training 

**Authors**: Mengnan Zhao, Geyong Min, Lihe Zhang, Tianhang Zheng, Jie Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.09688)  

**Abstract**: Adversarial training under long tailed distributions suffers from a dual imbalance: the class imbalance skews the training objective toward head classes, and the adversarial inner maximization may further amplify this bias. Existing methods mitigate this issue by correcting class priors or adapting class wise robust supervision, yet they treat each class in isolation and fail to identify which boundaries drive long tailed collapse. We propose a Confusion Geometry Rebalancing method (CGRm) for long tail adversarial training, a plug in framework that leverages directed robust errors as training signals. CGRm leverages periodic robust evaluations to derive source class loss weights, class wise robust coefficients, and a directed confusion geometry graph. The method then couples feedback weighted robust optimization with graph guided margin correction, thereby boosting the robustness of vulnerable classes and sharpening the critical boundaries that drive long tailed performance degradation. Experiments on long tailed benchmarks show that CGRm achieves consistent robust performance gains over existing methods, with ablations validating the contribution of each component. We provide the code in the supplement. 

---
# Predictive safety filter enhanced curriculum learning control for efficient vehicle dynamics controller 

**Authors**: Baocong Zhang, Siliang Lu, Chenyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09653)  

**Abstract**: Recent advances in learning-based control have enabled impressive achievements in solving complex control problems in various domains. However, since learning-based control may not be able to realize safety-guaranties, it is of great importance to enhance safety and robustness while maintaining good performances. Take vehicle motion \& dynamics control as an example, in order to overcome the pain points of traditional methods such as heavy parameter calibration effort and learning-based control to bring better performance and efficiency in stability \& agility over prior work for state-based vehicle control tasks, in this work, our method aims to develop a curriculum learning controller enhanced with physics-based predictive safety filter. The validation is conducted with the Python-CarSim platform, demonstrating better improvements and scalability under various maneuvers. 

---
# DUET: A Diversity-Quality Duet of Distillation Experts for Two-Step Video Generation 

**Authors**: Zian Li, Litong Gong, Borui Liao, Pengfei Liu, Xinyu Wang, Xinyuan Wei, Yifan Gao, Tiezheng Ge, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09637)  

**Abstract**: Diffusion models have enabled high-quality video generation in recent years, but the high cost of iterative sampling hinders their practical deployment. Few-step distillation alleviates this cost, yet exposes a quality--diversity trade-off between its two dominant paradigms: trajectory-level distillation (e.g., sCM) favors diversity, whereas distribution-level distillation (e.g., DMD) favors quality. Targeting extreme two-step video generation, we introduce DUET, which reconciles the two paradigms through a noise-level duet of experts: an sCM expert takes the high-noise step to lay out diverse structure, and a DMD expert takes the low-noise step to refine appearance detail. Since the two experts are trained independently with their native objectives, DUET sidesteps the optimization difficulties of loss-level combinations and delivers quality and diversity jointly rather than trading one for the other. We further identify the relay interface and the high-noise stage as the remaining bottlenecks, and address them with RL-guided expert adaptation, yielding DUET+. With the Wan2.1-T2V-1.3B backbone, DUET lifts the two-step quality of sCM close to the level of DMD while retaining nearly all of its structural diversity---about twice that of DMD---and DUET+ further improves overall quality while preserving this diversity advantage. Together, these results establish noise-level expert specialization as a simple, effective paradigm for reconciling diversity and quality in two-step video generation. 

---
# NeuroRefiner: Morphology-Aware Multi-Agent Refinement for 3D Fluorescence Microscopy Neuron Segmentation 

**Authors**: Haiyang Yan, Jinyue Guo, Yanchao Zhang, Bingqing Wang, Zhenchen Li, Jing Liu, Jiazheng Liu, Linlin Li, Hua Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.09636)  

**Abstract**: Accurate 3D neuron segmentation in fluorescence microscopy is critical for neuroscience. However, the sparse and elongated morphology of neurons poses significant challenges to existing segmentation methods. These methods struggle to preserve both local details and global topology, leading to fragmented results. To address this, we propose NeuroRefiner, a multi-agent system that formalizes the human expert workflow involving iterative global observation and local editing. Specifically, NeuroRefiner comprises three collaborative agents dedicated to diagnosing topological errors, generating correction instructions, and validating refinement quality. To facilitate agent instruction-guided segmentation refinement, we propose TopoRefineNet, a dedicated 3D U-Net-based tool that leverages cross-modality feature fusion to generate refined masks. Through multi-round agent reasoning and voxel-level editing, NeuroRefiner produces topologically more accurate segmentations with enhanced interpretability. Experiments on the BigNeuron, CWMBS, and ZBFWB datasets demonstrate that NeuroRefiner outperforms state-of-the-art methods, notably achieving a 3.02% improvement in F1 score on the challenging ZBFWB dataset. 

---
# Measuring the Wrong Thing: Internal Harmfulness Scores Anti-Rank Successful Jailbreaks 

**Authors**: Mingyu Luo, Ming Deng, Zilang Qiu, Yiming Cheng, Ci Tao, Xue Tan, Sijin Sun, Yangfu Li, Ping Chen, Jun Dai, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.09624)  

**Abstract**: Internal safety scores judge a prompt before any text is generated, and they are validated by how well they separate harmful prompts from benign ones. That separation is then read as evidence that the score will also catch the attacks that succeed. Harmful intent is a property of the prompt. Jailbreak success is an outcome produced later by a particular target model, decoding policy, and judge. A filter tuned on a score that measures the wrong quantity spends its false positive budget on attacks that would have failed anyway. In this paper we audit that inference. Attention based measurements are usually read from prompt dependent locations, so a wrapper changes both the content being judged and the place the signal is taken from. We therefore introduce Active Attention Probing, which supplies a fixed content independent measurement coordinate. We pair every base goal with a plain and a wrapped version and generate real completions from the target models. On Llama, wrapping raises harmful generation from 0.05 to 0.27 while harmful intent AUROC falls from 0.936 to 0.803, so the attacks grow more dangerous while the prompts look safer to the score. Among wrapped harmful prompts the outcome AUROC is 0.220, which places the attacks that succeeded below the attacks that failed. Rare token, passive, and detector derived channels reproduce the reversal on the same matched design, and the reversal itself persists across three target models, seven attack families, and two independent judges. Distribution shift then degrades calibration and threshold transfer before it degrades ranking. 

---
# Structure-Enhanced Features and Quality-Aware Dynamic Anchor Scoring for Robust Lane Detection 

**Authors**: Weize Cai, Yongqi Dong, Zhida Shao, Yichen Liu, Zixin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09610)  

**Abstract**: Lane detection requires recovering thin, elongated, and frequently occluded lane structures under challenging driving conditions. While anchor-based detectors provide efficient candidate generation, their performance is limited by two coupled issues: backbone features often lose structural continuity along partially visible lanes, and classification confidence may decouple from line-level localization quality, allowing inaccurate anchors to persist before non-maximum suppression (NMS). We propose a structure-enhanced and quality-aware framework that improves lane representation and dynamic-anchor scoring while preserving the inference pipeline of the Anchor Decomposition Network (ADNet). Specifically, a Gated Horizontal-Vertical Token (GHVT) module enhances mid- and high-level backbone features via lightweight directional token interactions with a learnable residual gate. In parallel, Line-Quality-Aware Dynamic Anchor Scoring (LQAS) calibrates existing classification logits using quality supervision, hard-negative suppression, and pairwise ranking without adding inference branches. On the VIL-100 dataset, our method improves ADNet-R34 from 89.97 to 91.28 in F1 score at the 0.5 intersection-over-union threshold (F1@50), reducing both false positives and false negatives. Additional experiments on CULane and TuSimple datasets, extensive ablations, score-distribution diagnostics, and runtime analysis confirm complementary structural and ranking improvements with minimal computational overhead. 

---
# TSPORec: Token Selection via Preference Optimization for LLM-Based Sequential Recommendation 

**Authors**: Wenqiao Zhu, Chao Xu, Haipang Wu, Ji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09605)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for improving recommendation systems. The effectiveness of LLMs arises from their ability to harness rich textual information and their capacity to model heterogeneous user preferences based on users' interaction history. However, due to the large-scale and deep architectures, LLM-based sequential recommendation approaches generally incur high inference costs, resulting in a low return on investment. To mitigate this cost, many existing approaches resort to using only the first few tokens of item descriptions, which inadvertently discards valuable information contained in the full text, thereby leading to suboptimal recommendation performance. To address this limitation, we propose a novel Token Selection approach for Preference Optimization in LLM-based sequential Recommendation, i.e., TSPORec, which accurately pinpoints informative tokens throughout the entire textual content to improve recommendation performance. Specifically, we design a three-stage pipeline to select informative tokens and introduce a novel proxy reward to facilitate the implementation. TSPORec not only enhances recommendation performance but also improves computational efficiency. Extensive experiments across two models and datasets demonstrate the superb performance (up to 31.25%) and efficiency (up to 63.4%) of our approach compared with six baseline approaches. Code is available at this https URL. 

---
# LEED: Local Embedding Evolution Distance for over-smoothing estimation and virtual node selection in GNN 

**Authors**: Killian Cressant, Pedro B. Velloso  

**Link**: [PDF](https://arxiv.org/pdf/2608.09596)  

**Abstract**: Graph Neural Networks (GNNs) suffer from two fundamental limitations: over-smoothing, where node representations become indistinguishable with depth, and over-squashing, where long-range information is compressed through limited message-passing channels. Existing metrics such as Dirichlet energy provide global characterizations of over-smoothing but lack the resolution to analyze node-level behavior and guide architectural improvements. In this paper, we propose LEED (Local Embedding Evolution Distance), a novel local metric that quantifies over-smoothing by tracking the evolution of individual node embeddings across layers. By operating at the node level, LEED enables fine-grained analysis of representation dynamics during training, revealing heterogeneous over-smoothing patterns that are invisible to global energy-based measures. This locality induces informative node importance scores, interpreted as embedding-driven centrality measures. We leverage LEED to design a more efficient strategy for virtual node selection. Unlike existing approaches that depend on multiple heuristic centrality measures, our method uses LEED as a unique criterion to guide the construction of Local Virtual Nodes to mitigate over-squashing. Experiments show that LEED provides more informative diagnostics than Dirichlet energy while preserving global evaluation, and enables more effective virtual node integration, improving GNN performance across datasets. 

---
# Illusion or Integrity? Geometrical Consistency Metric for AIGC Video Quality Evaluation 

**Authors**: Yifei Xue, Yuanchen Fei, Hao Zhang, Chenzhi Nie, Tie ji, Yizhen Lao  

**Link**: [PDF](https://arxiv.org/pdf/2608.09594)  

**Abstract**: Recently, AI-driven video generation has attracted considerable attention. This surge increases the demand for reliable video quality assessment (VQA) metrics to evaluate AI-generated content (AIGC) videos and guide model optimization. Existing studies assess video quality through visual harmony, video-text consistency, and domain-specific alignment, yet lack quantitative metrics for measuring fidelity to physical laws. To address this limitation, we present a novel benchmark that evaluates the quality of AIGC videos based on their compliance with physical principles by quantitatively measuring geometric consistency across frames extracted from generated sequences. This serves as a proxy for estimating the extent to which generated videos conform to real-world physical rules. Specifically, GeoCon-Bench captures global motion through translation estimation, fits homography or fundamental matrix models using background correspondences, and reports complementary metrics, including inlier ratio and geometric error. We also release a dataset containing 20 scenes across six motion categories. Experiments on state-of-the-art AIGC models demonstrate the reliability of GeoCon-Bench as a video quality assessment metric. 

---
# MADBench: A Benchmark for Modality-Aware Audio Deepfake Detection 

**Authors**: Yanqiu Li, Yang Xiao, Jisheng Bai, Bin Chen, Hong Jia, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09593)  

**Abstract**: Recent advances in speech synthesis and audio generation have made high-fidelity acoustic forgery low-cost and difficult to attribute, enabling a realistic attack scenario in which speech and background audio are independently manipulated over otherwise authentic video. Yet existing research either focuses on visual manipulation, addresses speech detection in isolation, or conflates speech and non-speech audio as a single undifferentiated audio stream, overlooking the distinct forensic challenges posed by background audio. This conflation is consequential: the two acoustic components arise from fundamentally different generative mechanisms, exhibit distinct artifact profiles, and pose different challenges to detection systems. We introduce MADBench, the first benchmark that treats speech and environmental audio as distinct acoustic components, enabling component-aware evaluation of audio deepfake detection across independently manipulated forgery sources. We benchmark representative state-of-the-art detectors and multimodal large language models under a unified protocol. Our experiments reveal that environmental audio manipulation is more detectable than synthetic speech across general-purpose encoders, while existing pretrained detectors fail on both acoustic components, and manipulated environmental audio asymmetrically degrades speech deepfake detection, findings entirely invisible under the single-label paradigm of prior benchmarks. MADBench establishes a rigorous foundation for future research into robust, component-aware audio deepfake detection. 

---
# Distributed Optimization with Streaming Data: A Temporal Weighting Perspective 

**Authors**: Muhammad Faraz Ul Abrar, Nicolò Michelusi, Erik G. Larsson  

**Link**: [PDF](https://arxiv.org/pdf/2608.09565)  

**Abstract**: Optimization theory is a widely used tool for intelligent decision-making. While classical optimization deals with fixed, time-invariant objective functions, many modern applications operate in dynamic environments where data arrive sequentially, and the learning objective evolves over time, often under decentralized data and communication constraints. Motivated by these trends, we study decentralized optimization from streaming data through a structured time-varying formulation in which the global objective is a temporally weighted average of losses observed across the network. We analyze multi-iteration decentralized first-order methods, including decentralized gradient descent. For strongly convex and smooth losses, we develop guarantees for the Euclidean-norm \emph{tracking error} through a contraction-mapping viewpoint. The resulting bounds decompose the tracking error into a fixed-point tracking component and a bias term induced by decentralization and data heterogeneity. We specialize our analysis to uniform and exponentially discounted weights, as well as their finite-memory \emph{windowed} counterparts. The bounds explicitly characterize the roles of the temporal weighting rule, per-step iteration budget, step size, and network connectivity. Uniform weighting yields a vanishing fixed-point tracking contribution of order $\mathcal O(1/t)$, whereas discounted and windowed strategies generally induce non-vanishing tracking floors governed by the discount factor and effective memory, respectively. In all cases, decentralization induces an additional non-zero bias floor under a constant step size. Numerical experiments illustrate the predicted trends. 

---
# From Semantic Grounding to Decision Optimization: A Unified Framework for Long-Horizon UAV Vision-Language Navigation 

**Authors**: Zeyuan Ma, Jiaxin Chen, Di Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09564)  

**Abstract**: UAV vision-language navigation (UAV-VLN) focuses on enabling an aerial agent to follow natural-language instructions in open 3D environments from egocentric visual observations. Current approaches suffer from three coupled issues: weak grounding of instruction-relevant landmarks in visual observations, insufficient exploitation of long-horizon history, and unstable decisions under local traps or repeated exploration. To address these issues, we propose a unified semantic-to-decision framework. First, we present an instruction-grounded semantic enhancement module that injects object-level semantics and relative spatial cues into the current observation state. Subsequently, we develop a relevance-aware dynamic temporal aggregation strategy that reweights the full history buffer while converting a few high-relevance frames into structured landmark prompts for the decoder. Finally, we devise a topology-aware decision method that combines local-optimum cognition with group-relative policy optimization under progress, goal, semantic, and path-compliance rewards. Experiments on the widely used AerialVLN and OpenFly benchmarks clearly demonstrate that our method achieves state-of-the-art performance. 

---
# ELBench: A Multi-Dimensional Benchmark for Education-Facing Large Language Models 

**Authors**: Yilin Jiang, Xiaorong Zhu, Fei Tan, Zicheng Zhang, Kaiyi Huang, Yang Yu, Zexuan Fei, Yiming Luo, Keqian Li, Hao Hao, Aimin Zhou, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09548)  

**Abstract**: Large language models are increasingly deployed in education as tutors, teaching assistants, and content generators. These roles place demands that ordinary question answering does not: a usable education-facing model is supposed to be accurate, safe under sensitive prompts, instructionally useful, and aligned with pedagogical goals at the same time. Existing benchmarks evaluate these requirements largely in isolation, so none assesses education-facing suitability as an integrated profile. We introduce ELBench, the first benchmark to evaluate all four requirements (General Capability, Safety and Trustworthiness, Basic Education, and High-Level Cultivation) on the same models under a common protocol, combining curated public sources with newly synthesized safety and cultivation data. We evaluate nine models, seven frontier general-purpose systems and two education-specialized variants, and report three findings. First, module-level profiles are more informative than a single aggregate: the top six models are statistically indistinguishable on overall score, yet their module leaders differ substantially, and safety is anti-correlated with practical teaching (r = -0.83). Second, the Chinese-developed models lead the safety module, the most discriminative in the suite; this advantage is largest on region-specific normative content and narrows, but does not vanish, on universal-harm content. Third, the two education-specialized models lead neither education module, and on High-Level Cultivation all models share a systematic blind spot: on the structured judgment task they converge on the same non-reference option, favoring pedagogical style over fit to the stated goal, so the module scores uniformly low and does not separate models. This raises, but does not resolve, whether domain post-training keeps pace with frontier systems on education tasks. 

---
# Dual-Adversarial Safety Alignment: Cultivating Intrinsic Threat Comprehension in LRMs 

**Authors**: Hongli Shen, Shaopeng Fu, Qinbo Zhang, Jian Li, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09542)  

**Abstract**: Large reasoning models (LRMs) achieve remarkable success on complex tasks but remain vulnerable to harmful prompts that induce unsafe outputs. Recent methods align LRMs using direct refusals or safety rationales, yet often focus on prompt patterns rather than intrinsic attack mechanisms. As a result, these pattern-centric alignments struggle to generalize across diverse jailbreaks, compromising adversarial robustness and reasoning utility. We propose AdvSafe, a dual-adversarial framework that enables LRMs to internalize unsafety knowledge by explicitly deconstructing adversarial mechanisms. This moves beyond pattern-dependent traces, fostering robust cognitive defense without compromising reasoning utility. Our pipeline operates via a two-phase adversarial game. First, in adversarial synthesis, an autonomous agent dynamically crafts deceptive jailbreak prompts, adapting its strategies to breach a strong teacher model. Second, in adversarial extraction, the breached teacher executes a cognitive counter-attack. For every successful jailbreak, the teacher unmasks the camouflage, explaining why the attack succeeds and how such prompts can be identified and mitigated. This dual-adversarial process yields a compact reasoning dataset capturing rich, generalizable unsafety knowledge. Student models trained on this dataset implicitly acquire safety alignment through intrinsic threat comprehension. Experiments show that with only 1K synthesized samples, AdvSafe-aligned LRMs achieve significantly stronger jailbreak robustness than existing baselines, with almost no utility degradation. Furthermore, AdvSafe improves robustness against out-of-distribution prompts, demonstrating that learning unsafety knowledge enables a superior robustness-utility trade-off and generalizes beyond seen attack patterns. 

---
# TCS-BENCH: Benchmarking State-of-the-Art Generative AI Theoretical Computer Science Research Ability 

**Authors**: Vincent Cohen-Addad, Dimitris Paparas, Ernest van Wijland, Max Springer, Julien Canitrot-Paradis, Honghao Lin, David Woodruff, Adarsh Kumarappan, Rajesh Jayaram, Rudrajit Das, Lalit Jain, Ola Svensson, Silvio Lattanzi, Mislav Balunovic, Theophane Weber, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2608.09538)  

**Abstract**: We introduce TCS-Bench, a benchmark for evaluating Large Language Models (LLMs) on research-level Theoretical Computer Science (TCS) proof generation. TCS-Bench consists of theorem-proving tasks from papers published at top theoretical computer science venues (STOC, FOCS, and SODA). Each task provides the necessary context to derive a self-contained proof for a target result. We evaluate state-of-the-art models on this benchmark. We verify the correctness of generated proofs via a verification agent, and further benchmark the verifier against human-expert proof judgements on a set of target statements and generated proofs pairs. Our reference verifier achieves over 90% accuracy on the expert labeled set. 

---
# Carnot: Interpretable, Interactive, and Optimized Execution of Deep Research Queries 

**Authors**: Matthew Russo, Yash Agarwal, Tianyu Li, Zhuohan Gu, Michael Cafarella, Omar Khattab, Tim Kraska, Samuel Madden  

**Link**: [PDF](https://arxiv.org/pdf/2608.09532)  

**Abstract**: Enterprises increasingly seek to query data lakes using natural language via AI-driven tools like semantic operators or deep research agents. However, the latter operates as an opaque black box, hiding its intermediate reasoning and data retrieval steps, and failing to expose controls for managing API costs and execution latency. Meanwhile, the former can be prohibitively expensive for enterprise-scale data lakes. Consequently, analysts using these systems lack the agency to intercept hallucinated premises, verify intermediate results, or correct the system's trajectory. We present Carnot, an interactive execution engine for AI-driven analytics. Carnot compiles natural language requests into physical execution graphs and surfaces them through an interactive notebook interface. Rather than waiting blindly for a final output, users can critique the plan, incrementally execute operators, inspect intermediate data, or directly edit the underlying code or semantic operator instructions. Carnot's query optimizer will optimize the query with respect to cost or latency constraints provided by the user. Our demo will showcase how Carnot helps users achieve efficient and verifiable insights on workloads motivated by real enterprise use cases. 

---
# RangeFactory: Scalable Construction of Multi-Hop Cyber Ranges 

**Authors**: Hanlin Jiang, Puyi Wang, Jiandong Jin, Shaofei Li, Zhan Shen, Pengli Wang, Ziming Wang, Yifeng Cai, Ning Jia, Yuxin Ren, Peng Jiang, Yao Guo, Ding Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09526)  

**Abstract**: Real-world cyberattacks often require sustained progress across multiple hosts and network segments, making multi-hop cyber ranges essential infrastructure for studying and improving LLM agents' ability to sustain complete attack chains. Prior work has scaled isolated vulnerability tasks and constructed multi-host scenarios from manually specified vulnerability semantics. However, they are still unable to automatically orchestrate the growing supply of vulnerability environments into end-to-end validated multi-hop ranges. To this end, we present RangeFactory, an automated cyber-range orchestration framework that constructs multi-hop cyber ranges at scale from isolated vulnerability environments. RangeFactory formulates range construction as dependency resolution: it extracts dependency information from agents' actual attacks against real vulnerabilities, resolves known dependencies through template-guided orchestration, and uses end-to-end attack execution to validate runtime dependencies that emerge after composition. Using RangeFactory, we construct RangeBench with 1,148 validated range instances spanning 287 distinct attack chains and evaluate frontier attack agents across attack depth, network scale, and task information. Among runs that compromise the entry vulnerability, 24.5-47.0% still fail to complete the remaining attack path, revealing a substantial sustained-compromise gap between establishing an initial foothold and completing a multi-hop attack. RangeFactory further produces a corpus of 5,541 outcome-annotated multi-hop attack trajectories, providing execution data for attack-process analysis and future agent training. 

---
# STAIR: Effective Incident Response Using an End-to-End Agentic Planning Framework 

**Authors**: Hanlin Jiang, Jionghao Huang, Shaofei Li, Bojia Yu, Peng Jiang, Yuxin Ren, Ning Jia, Yao Guo, Ding Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09524)  

**Abstract**: Incident response planning is critical for restoring compromised software systems after cyberattacks. Common practice relies on expert-driven playbooks that encode fixed response procedures, but these static workflows struggle to adapt to evolving incident states, changing recovery objectives, and execution feedback. Recent LLM-based planners and tool-using agents improve automation, yet they remain unstable in long-horizon response because they lack a unified basis for maintaining incident state, aligning actions with the current recovery stage, and reusing historical experience.
We present STAIR, an end-to-end agentic planning framework for incident response. The framework maintains the current incident as Graph-as-State, uses a Stage Router to dispatch planning to stage-specialized agents, and retrieves historical experiences to guide action selection. An Execution Harness executes actions, returns feedback to update the incident state, and validates action effects for future experience reuse. Across 100 Docker-based cyber ranges, our framework achieves a normalized defense score of 0.94 and improves over the strongest baseline by 9.5%. 

---
# Build it, Break it, Repeat: Benchmarking and improving LLM-manipulated disinformation detection in social media posts 

**Authors**: Kevin Thomas, Milosz Kasprzyk, Reuel C Igbokwe Onuigbo, Elliott Pert, Cameron Tovey, João A. Leite, Olesya Razuvayevskaya, Carolina Scarton  

**Link**: [PDF](https://arxiv.org/pdf/2608.09510)  

**Abstract**: Detecting machine-generated disinformation on social media is increasingly difficult as large language models (LLMs) make it easier to generate and rewrite misleading content at scale. Static benchmark evaluations, measuring detector performance on fixed held-out datasets, do not capture how detectors behave when posts are deliberately transformed to evade classification. This paper adapts the Build it, Break it, Fix it framework into Build it, Break it, Repeat (BiBiR): iterative sessions designed to stress-test detectors' robustness under iterative adversarial conditions, evaluating whether models remain reliable when disinformation posts are systematically transformed to evade classification. Across five iterations, the findings show that the best adversarial breakers' transformations came from a combination of back-translation and LLM persona-based rewriting, with the best performing technique achieving a 95% label flip rate (LFR), whilst still preserving the meaning of the original posts. The best builders' model was a triplet contrastive model with a dynamic anchor switching (DASS) architecture, which achieved an average accuracy of 72.68%, outperforming the strong baseline (a fine-tuned e5-small-LoRA) by 15 percentage points on the most robust set of breakers' adversarial attacks. The results demonstrate that an iterative framework best exposes detector weaknesses and pushes robustness improvements; however, it may still require semantic preservation analysis to distinguish valid adversarial evasion from transformations that changed the original disinformation claims' meaning. 

---
# Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning 

**Authors**: Yuting Liu, Wei Wu, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09507)  

**Abstract**: Natural language user preferences provide an interpretable interface for LLM personalization. However, universal preference summaries often contain information irrelevant to a particular downstream task. Directly supplying the full preference summary therefore wastes context capacity and introduces cross-task distraction, while manually designing task-specific preference views is difficult to scale. In this work, we study \emph{task-specific preference adaptation}: given a universal user preference summary and a downstream task, derive a task-conditioned representation that preserves sufficient decision-relevant evidence while removing redundant context. To this end, we propose \textsc{AlignXada}, a training-free meta-learning framework that induces reusable textual refinement policies for adapting universal preference summaries to task-specific ones. The refinement policy is iteratively optimized by a meta learner through verbal reinforcement learning. Across 13 tasks and three downstream models (39 task--model cells), \textsc{AlignXada} achieves an average gain of 3.82 points, improving 33 cells while retaining only 22.8\% of the original profile tokens and outperforming RAG in 36 cells. An extended faithfulness analysis further shows that the refined profiles remain largely grounded in the source preferences while preserving task-relevant personalization signals, suggesting that profile-side adaptation serves as a practical complement to universal memory construction for lifelong personalized agents. 

---
# Beyond Uniform Restoration: Empowering All-in-One Restoration with Pixel-Level Multimodal Guidance 

**Authors**: Chunxiao Liu, Wei Liu, Anbin Xiong, Erli Meng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09482)  

**Abstract**: All-in-one image restoration is a unified low-level vision task that aims to effectively recover high-quality images from inputs degraded by various types and levels of corruption using a single model. Recent works have achieved remarkable progress by learning degradation-adaptive prompts or network architectures. However, these methods typically apply a uniform restoration strategy across the entire image, neglecting the fact that different regions may suffer from distinct degradation types and varying degrees of severity. In contrast, we propose to perform restoration at the pixel level, thereby enabling more fine-grained and precise control over the restoration process. Specifically, we present MGN-AIR, a novel pixel-level restoration framework for all-in-one image restoration. Our approach first learns to estimate a pixel-level visual prompt. Then, it leverages both textual and visual prompts to provide global and local degradation cues, guiding the model on where to look and how to restore at each pixel. We conduct extensive experiments on multiple all-in-one image restoration benchmarks, covering a wide range of tasks including denoising, deraining, deblurring, dehazing, desnowing, and low-light enhancement. Experimental results demonstrate that our proposed method consistently and significantly outperforms existing approaches. 

---
# ActBench: Self-Evolving Benchmark of Behavioral Safety in Cowork Agents 

**Authors**: Hongwei Yao, Yiming Liu, Meihui Chen, Jieling Chen, Zikun Chen, Yiling He, Wangze Ni, Cong Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.09476)  

**Abstract**: Cowork agents may complete benign tasks while disclosing protected data, manipulating unauthorized state, invocate unauthorized API. We define behavioral safety and introduce ActBench, a self-evolving benchmark that evaluates such behavior risk from execution trajectories rather than final responses. Each case pairs a benign task with an adversarial variant that preserves its instruction, configuration, initial state, rating model, and trusted records while injecting a task-reachable payload. ActBench contains 600 cases from 213 scenarios, spanning 15 risk behaviors, six execution spaces, and 48 web-service this http URL move beyond static payloads, we propose a reward-guided beam search method that jointly optimizes attack effectiveness and task utility, while reflection diagnoses failed execution checkpoint and guides payload revision. Besides, we propose a dual evidence verification mechanism that verifies agent execution safety and utility through log evidence and LLM-based trajectory this http URL evaluate 15 LLMs and 6 open-source cowork agents over 24,000 trajectories. Under a fixed harness, attack success rates ranges from 10.1% to 94.4% across models, while under a fixed base model, they range from 73.7% to 94.4% across this http URL results show greater variation across models than agent harness, while attacks remain highly successful across all tested this http URL benchmark is released at: this https URL. 

---
# MixFormer: Linear Transformer with Mixture of Memory Experts 

**Authors**: Yu Guo, Lei Duan  

**Link**: [PDF](https://arxiv.org/pdf/2608.09468)  

**Abstract**: State Space Models (SSMs), as a mainstream research direction of linear Transformers, aim to achieve higher efficiency than standard Transformers in long-context modeling. However, existing SSMs suffer from limited input adaptivity and constrained memory capacity, leading to information loss when modeling ultra-long sequences. To address these limitations, we propose MixFormer, a novel linear Transformer that integrates a Mixture-of-Memory-Experts (MoE) mechanism. Specifically, the model maintains differentiated memory states through multiple collaborating memory experts and employs a novel Time-Aware Linear Attention (TALA) mechanism, which leverages learnable exponential decay functions and positional biases to dynamically update memory. This design enables the model to selectively reinforce important historical information while effectively mitigating memory dilution, substantially improving long-range dependency modeling. Experiments on long-sequence text and image generation tasks demonstrate that MixFormer not only achieves significant performance gains but also provides a more sustainable computational backbone for the next generation of web infrastructure. 

---
# RecoverFly: A Failure-Aware Reinforcement Learning Post-Training Framework for Aerial Vision-Language Navigation 

**Authors**: Boxiong Wang, Hui Kang, Geng Sun, Jiahui Li, Chao Yu, Daxin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2608.09467)  

**Abstract**: Unmanned aerial vehicle vision-language navigation (UAV-VLN) requires agents to translate visual observations and language instructions into reliable flight actions in complex environments. Although recent end-to-end UAV vision-language-action (UAV-VLA) policies reduce reliance on separately designed perception, planning, and control modules, their behavior-cloning objectives provide limited corrective supervision for interactive closed-loop execution. Reinforcement learning (RL) offers a promising solution, while its effectiveness is constrained by inefficient use of samples, long-tailed scene distributions, and policy distribution shift during optimization. To this end, we propose RecoverFly, a failure-aware RL post-training framework for end-to-end UAV-VLA policies. Specifically, RecoverFly adapts token-level RL for stable optimization of grammar-constrained autoregressive UAV actions, revisits unresolved failure cases to strengthen corrective learning and sample utilization, and combines a two-stage long-tail scene curriculum with reference-policy regularization to improve scene adaptation while preserving acquired capabilities. Experiments on the TravelUAV benchmark demonstrate that RecoverFly achieves the best performance on the seen, unseen-map, and unseen-object splits. Moreover, compared to the AerialVLA initialization, RecoverFly improves success rate by 3.12 to 8.37 percentage points under a total rollout budget of about 30\% of the training-set size, validating its effectiveness, robustness, and generalization capabilities. 

---
# Learning to Modulate, Not to Cycle: Soft Actor---Critic Recovers Inverter-Style Heat-Pump Control 

**Authors**: Faizan Ahmed, Aniket Dixit, James Brusey  

**Link**: [PDF](https://arxiv.org/pdf/2608.09453)  

**Abstract**: On--off cycling is the main cause of compressor wear in residential heat pumps, yet reinforcement learning (RL) controllers for buildings typically optimise only energy cost and thermal comfort, ignoring how much the learned policy cycles. We add a levelised compressor-wear term to the control reward and study how the resulting behaviour depends on the RL algorithm. Training Soft Actor---Critic (SAC) and Proximal Policy Optimisation (PPO) on an identical Markov decision process for the BOPTEST bestest hydronic heat pump case, we find that SAC learns a continuous modulation policy that keeps the compressor permanently engaged---the operating principle of an inverter-driven heat pump---achieving zero start-ups per day, whereas PPO collapses to bang-bang control that cycles more than the baseline. On the BOPTEST emulator the SAC policy cuts thermal discomfort by up to 90.7% for an 11.5% cost increase, while eliminating all baseline cycling. 

---
# WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training 

**Authors**: Zehao Chen, Gongxun Li, Tianxiang Ai, Yifei Li, Zixuan Huang, Wang Zhou, Tao Huang, Fuzhen Zhuang, Xianglong Liu, Jianxin Li, Deqing Wang, Yikun Ban  

**Link**: [PDF](https://arxiv.org/pdf/2608.09447)  

**Abstract**: On-policy distillation (OPD) aligns a student with a teacher on trajectories sampled from the student itself, reducing the train-test state mismatch of offline distillation. The same feedback loop can nevertheless be unstable: each update changes both the policy and the states on which the next update is computed. We introduce WDL-OPD, a mixture-constrained co-training method with two trainable policies. An anchor policy generates every rollout, an auxiliary policy evaluates the same visited states, and a geometric mixture of their token distributions is matched to a frozen teacher by reverse KL. Both policies receive gradient. We show that freezing the auxiliary recovers an anchor-plus-contrast proxy target closely related to OPD$^2$ and W2S-OPD, whereas joint training creates branch-level degrees of freedom that a static delta cannot express. In recorded Qwen3 experiments at 1.7B and 4B scale, WDL-OPD produces the strongest student checkpoint in each of four scale-domain settings. It raises MATH500 accuracy from 0.630 to 0.685 at 4B and from 0.521 to 0.585 at 1.7B. In code generation, seven single-policy OPD configurations exhibit entropy growth or trajectory degradation, while co-training reaches independently re-evaluated development scores of 0.637 and 0.375. Because several comparisons differ in curriculum or initialization, these results support a stabilization hypothesis rather than a universal causal claim. We provide the exact training algorithm, failure evidence, and the controlled comparison matrix needed to test that hypothesis. 

---
# How Simple Can It Get? From Interpretable Equations to Readable Rules for Financial Decision Making 

**Authors**: Adia Lumadjeng, Ilker Birbil, Erman Acar  

**Link**: [PDF](https://arxiv.org/pdf/2608.09433)  

**Abstract**: In regulated domains such as finance, a model that cannot be explained cannot be deployed, yet many interpretable classifiers defeat their own purpose by producing formulas with dozens of features that no regulator could read. We take the reverse direction. Starting from an interpretable classifier expressed as a single equation over the input features, we progressively simplify it into more readable forms, including a pruned monomial, a directional if--then rule, and the integer scorecards and tallies that finance already deploys. Because the equation is itself the predictive model rather than a post-hoc explanation we can directly quantify what is lost under each simplification. Across four financial datasets, we find that pruning is nearly free and that fidelity can erode faster than predictive performance, allowing simpler rules to remain effective classifiers without faithfully reproducing the original model. A human assessment shows that simplification improves perceived readability, while preferences for different representations vary by professional background. Beyond measuring these losses empirically, we show that some can be anticipated from the original model: we derive a bound on the change caused by pruning and predict how faithfully a rule retaining only the direction of each feature's effect preserves the original ranking. 

---
# ZetaGPT: A Reference Implementation of Positional--Encoding--Free State--Space--Attention Language Models 

**Authors**: Róisín Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09432)  

**Abstract**: Transformer-based language models rely on self-attention, whose computation is permutation-equivariant and therefore lacks an intrinsic mechanism for representing token order. Existing architectures address this limitation by explicitly incorporating positional information through learned positional embeddings or hand-crafted positional encodings, such as rotary positional encoding (RoPE), treating positional information as an architecturally acquired capability rather than an inherent property of the model. Motivated by the pursuit of positional-encoding-free architectures, this work explores a language model architecture that integrates causal state-space equations to implicitly encode positional information before attention computation. Specifically, each model block applies a causal state-space equation before self-attention, allowing recurrent state dynamics to encode sequential information into token representations. Consequently, subsequent attention layers operate on position-aware representations without requiring explicit positional encodings while retaining the expressive modeling capacity of self-attention. We present \textsc{ZetaGPT}, a compact hybrid language model designed for research, rapid prototyping, algorithm verification, and educational applications. In addition to the proposed architecture, \textsc{ZetaGPT} provides a fully open-source, end-to-end training pipeline encompassing dataset construction, tokenizer training, pretraining, supervised fine-tuning, reinforcement learning from human feedback (RLHF), and chain-of-thought (CoT) reasoning via pure reinforcement learning. To the best of our knowledge, \textsc{ZetaGPT} is the first open-source small language model without explicit positional encoding and establishes a compact, reproducible reference implementation for the development and empirical study of positional-encoding-free language models. 

---
# LITEWAY: LIghtweight HAR via Temporal Efficient highWAY 

**Authors**: Dominique Nshimyimana, Vitor Fortes Rey, Mengxi Liu, Bo Zhou, Paul Lukowicz  

**Link**: [PDF](https://arxiv.org/pdf/2608.09421)  

**Abstract**: Wearable human activity recognition (HAR) remains challenging due to the computational and energy constraints of deep learning models on resource-limited devices. Existing lightweight approaches often rely on recurrent architectures (e.g., GRU and LSTM), limiting parallelism and increasing inference latency. We propose LITEWAY, a modality-agnostic, fully convolutional framework for multichannel sensor time series that replaces recurrent temporal modeling with structured convolutional decomposition. LITEWAY combines lightweight convolutional blocks, strided temporal processing, and convolution-attention pooling to efficiently capture temporal dependencies while reducing computational complexity. We evaluate LITEWAY on 16 HAR datasets against TinyHAR, TinierHAR, and MLP-HAR. LITEWAY achieves competitive macro F1 while reducing model size by 4.06x-9.52x (Light) and 3.87x-9.07x (Full) compared with TinyHAR and TinierHAR. Deployment experiments further show energy reductions of 2.29x-3.14x (Light) and 1.46x-2.01x (Full) compared with TinierHAR and MLP-HAR, highlighting efficient fully convolutional temporal modeling for wearable HAR. The source code is publicly available at this https URL. 

---
# Sign Language Recognition Using Original and Synthetic Depth Image Based Point Cloud Data Models 

**Authors**: Rustem Ozakar, Eyup Gedikli  

**Link**: [PDF](https://arxiv.org/pdf/2608.09400)  

**Abstract**: Research regarding the sign language recognition mostly relies on RGB images, whileas sign language datasets that provide depth images are limited. Point clouds obtained from depth images can be used for sign language recognition with neural networks like PointNet. In recent years, various neural networks are used for generating realistic depth images from monocular RGB images. In this work, synthetic depth images were created from RGB images using Depth Anything V2 network. For this purpose, three sign language datasets (Real-time ASL Fingerspelling, KArSL, AUTSL) which contain both RGB and depth images were used. Classification accuracies of the point cloud data created from both original and synthetic depth images using various PointNet architectures were measured for sign language recognition. From the original and synthetic point clouds, frame based, Point Gesture Map and Long Short Term Memory data models were used for classification and their performances were compared. In the results, both original and synthetic based data achieved acceptable performance in most models. In general, original depth based point cloud models performed better than synthetic ones, however in some models synthetic depth based models performed better than the originals. 

---
# Monotonicity-Guided Bottom-Up Petri Net Discovery: The SPECpp Framework 

**Authors**: Leah Tacke genannt Unterberg, Lisa L. Mannel, Wil M. P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2608.09398)  

**Abstract**: Process discovery is one of the central challenges in process mining. Petri nets are particularly attractive because simple local constructs can express complex behavior, including concurrency. While their global behavior may be difficult to analyze, individual places can be efficiently characterized using monotonic properties, enabling bottom-up discovery. Unlike top-down approaches such as the Inductive Miner, which rely on predefined constructs for sequences, choices, loops, and concurrency, our approach allows such structures to emerge organically and can exploit the full expressive power of Petri nets, including free-choice constructs and long-term dependencies. The main challenge is the exponential number of candidate places and their combinations. We present the SPECpp framework which implements strategies to obtain high-quality models under time and resource constraints. SPECpp supports rapid experimentation and is used to evaluate these strategies using both synthetic and real-life event data. 

---
# Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law 

**Authors**: Rose Cymbler, Daniel Guez, Laurent Fabre  

**Link**: [PDF](https://arxiv.org/pdf/2608.09393)  

**Abstract**: We identify and quantify temporal misgrounding: the systematic retrieval and citation of the currently in-force version of a legal article when the applicable version is an earlier or future one. Standard legal RAG treats the corpus as static; we argue legal question answering is a temporally-indexed retrieval problem. We introduce FiscalQA Pro, pairing a versioned corpus of 32,436 article-versions of the French tax code (93 years, 1938-2031) with an all-model-hard temporal-reasoning track: 209 scored, expert-reviewed questions across 33 CGI articles (221 released; twelve flagged out of the answerable scope). At selection time, no evaluated model recovered its date-applicable answer closed-book in any of four sampling draws, and the currently in-force text lacks the gold value for all but one of the scored questions. Answers are scored deterministically via atomic ground-truth "nuggets" (regex and numeric-with-tolerance), never LLM-as-judge: an LLM judge would inherit the temporal bias it is meant to score. Across eleven models (five frontier closed-API systems plus Gemini 2.5 Pro as a substitute entry, and five open-weight), parametric knowledge yields 3.0% mean strict accuracy and RAG over a static current-version corpus 2.7%. Static RAG retrieves the date-applicable version 0% of the time, confidently citing a real but inapplicable version. Our end-to-end retriever over a multi-version index, with no oracle, reaches 98.3% mean strict; an oracle-article ablation reaches 99.1%, locating the residual gap in first-stage recall, not version selection. We additionally release a version-aware jurisprudence dataset of 69,208 citation links, together with the corpus, benchmark, model responses, and pipeline code. 

---
# Imaginative Generative AI: Crossing the Entropy Wall into Worlds Beyond Imitation 

**Authors**: Hossein Goli, Farzan Farnia, Amin Gohari  

**Link**: [PDF](https://arxiv.org/pdf/2608.09385)  

**Abstract**: Generative AI models are primarily designed to imitate the data distribution, an objective that neither corrects diversity lost by a learned generator nor defines how generation should extend beyond the diversity of the data itself. We introduce Imaginative Generative AI (IGA), a framework that makes diversity part of the target-distribution design problem: among distributions close to a reference, IGA selects one whose spectral diversity reaches a prescribed level. Diversity is measured by the von Neumann entropy of the generated distribution's kernel covariance operator in a fixed representation space, providing a reference-free representation-guided measure of how broadly probability mass occupies embedding directions. The spectral entropy of the population data distribution defines an Entropy Wall. Below the wall, IGA performs diversity repair, recovering variation that a learned generator has lost while remaining within the diversity level of the data. Beyond the wall, the data distribution itself becomes infeasible, and IGA deliberately departs from it to produce distributions with greater representation-relative spectral diversity, an operational notion of imaginative generation. These regimes form a single regularization path from imitation to imagination and define an i.i.d. target distribution at each prescribed diversity level. We develop the theory of this entropy-constrained projection and show that, under a KL anchor to a pretrained generator, the optimum satisfies a self-consistent exponential-tilt relation. This characterization leads to IGA Guidance, a retraining-free inference-time method for score-based and diffusion models, including DDPM and DDIM samplers. Experiments on synthetic and vision benchmarks demonstrate diversity repair below the Entropy Wall and controlled spectral extrapolation beyond it. 

---
# FeedbackTrack: Visual-Cortex-Inspired Cross-Frame Feedback for Transformer Tracking 

**Authors**: Yueyang Cang, Xiaoteng Zhang, Zhiyuan Ning, Yuchen He, Li Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.09369)  

**Abstract**: Visual object tracking requires effective temporal integration, yet most Transformer trackers still rely on predominantly feed-forward feature extraction. Existing temporal mechanisms typically update templates, prompts, queries, or prediction states, while intermediate representations are rarely reused to modulate corresponding processing stages. We propose \textbf{FeedbackTrack}, a visual-cortex-inspired framework that introduces sparse, group-level layer-aligned cross-frame feedback into pretrained Transformer trackers. Previous-frame intermediate states are detached, cached, and returned to corresponding Transformer groups in the current frame through two lightweight pathways: Query Feedback for token-level query modulation and Gate Feedback for context-dependent feature modulation. FeedbackTrack preserves the original tracking pipeline with only a fixed-size one-frame cache. Across SPMTrack and ARTrackV2, FeedbackTrack consistently improves five backbone configurations on LaSOT and GOT-10k, achieving 83.4 AO and 79.1 AUC with SPMTrack-G while adding less than 1\% parameters. Controlled comparisons show that cross-frame feedback outperforms same-frame modulation by 1.8--3.2 AO points, demonstrating that the gains mainly come from recurrent historical information. Further analysis reveals a non-uniform depth-dependent organization of learned feedback strengths, highlighting the effectiveness of recurrent feedback for Transformer tracking. 

---
# Deep Learning based Detection of Fishing Vessels and Fishing Monitoring using Nightlight Images 

**Authors**: Shantakar Mohanty, Prasun Kumar Gupta, Raian Vargas Maretto  

**Link**: [PDF](https://arxiv.org/pdf/2608.09360)  

**Abstract**: The demand for maritime surveillance has given rise to the need for monitoring fishing vessel activities, particularly in addressing the challenge of "dark vessels" that operate without Automatic Identification System (AIS) transmission. This study presents a novel approach for detecting small-scale fishing vessels using nighttime light (NTL) imagery from the SDGSAT-1 satellite, combined with deep learning techniques to enhance fishing monitoring awareness along the western coast of India. A dual-branch YOLO11 architecture was developed to exploit both the 10-meter panchromatic and 40-meter RGB imagery from SDGSAT-1. The custom model architecture was specifically optimized for small object detection in NTL imagery, featuring parallel convolutional backbones that process both modalities before concatenation for enhanced feature extraction. The dual-branch YOLO11 model demonstrated optimal performance with a precision of 0.99, recall of 0.93, F1-score of 0.96, and mAP@50 of 0.96, significantly outperforming single-branch implementations of YOLOv5s, YOLOv8s, and standard YOLO11s architectures. When applied to the western coast of India, the model detected 31525 vessel instances across the temporal dataset spanning 2022-23. Cross-matching analysis with AIS data revealed that only 7146 (22.7%) of detected vessels had corresponding AIS transmissions, while 24379 (77.3%) were identified as potential dark vessels. Spatio-temporal analysis showed peak fishing activity during January-April, with a primary activity corridor parallel to the coastline within 50-100 km, corresponding to productive continental shelf areas. This research contributes to maritime surveillance capabilities by highlighting the effectiveness of nighttime lights satellite imagery for fishing vessel detection and provides valuable insights into fishing patterns and potential regulatory compliance issues in Indian waters. 

---
# Test-Time Augmentation for LLMs: When Input Diversity Beats Output Diversity at Matched Compute 

**Authors**: Nikita Kozodoi, Zainab Afolabi, Jack Butler  

**Link**: [PDF](https://arxiv.org/pdf/2608.09351)  

**Abstract**: Test-time scaling improves LLM accuracy but multiplies inference cost, making the accuracy gained per unit of compute the metric that matters in deployment. Self-consistency is one of the established approaches, which spends this budget entirely on the output side by sampling repeated reasoning paths. We study Test-Time Augmentation (TTA), which extends self-consistency by also perturbing the input, aggregating predictions across transformed versions of the input, and ask whether input-side diversity converts compute into accuracy more efficiently than output-side diversity. We perform a systematic, matched-compute comparison: we evaluate three simple input-side strategies (semantic rephrasing, lexical perturbations, and visual transformations) across six datasets covering general and multilingual knowledge, mathematical reasoning, multi-modal question answering, and sentiment classification, against chain-of-thought prompting and self-consistency. Semantic rephrasing delivers consistent and statistically significant accuracy gains while Pareto-dominating self-consistency on cost-effectiveness, delivering roughly 1.8X more accuracy per dollar and outperforming it on five of six tasks. We further analyze the number of augmentations, multi-modal strategies, and base model scaling, finding that TTA is most cost-effective for mid-tier models where a stronger model is unavailable or too expensive. Our findings indicate that for current mid-tier LLMs, varying the input converts inference compute into accuracy more efficiently than varying the reasoning path alone. The TTA implementation is available at this https URL. 

---
# RAG-Audio: Retrieval-Augmented Generation for Faithful Brain-to-Audio Reconstruction 

**Authors**: Ambuj Mehrish, Sebastiano Vascon  

**Link**: [PDF](https://arxiv.org/pdf/2608.09331)  

**Abstract**: Brain-to-audio reconstruction is limited by \emph{prior domination}: when a pretrained generator is conditioned on a weak neural signal, it produces realistic but stimulus-inaccurate audio. We introduce RAG-Audio, which decodes fMRI into a semantic audio embedding, retrieves a matching real-audio exemplar, and initializes the frozen generator's sampling trajectory from that exemplar while retaining the decoded embedding as conditioning. On Brain2Music, RAG-Audio improves 10-way stimulus identification from $0.14$--$0.18$ for direct generation, near the $0.10$ chance level, to $0.40$--$0.43$, comparable to retrieval. It also reduces Fréchet Audio Distance by roughly an order of magnitude, from $13.49$ to $1.25$ for AudioLDM. RAG-Audio approaches nearest-neighbor retrieval in identification while remaining generative; its higher FAD is expected because retrieval directly replays real audio. An autoregressive negative control, which lacks an initializable latent trajectory, shows no comparable gain, attributing the improvement to trajectory initialization. These results suggest that retrieval-guided initialization can mitigate prior domination in brain-to-audio generation. 

---
# WorldSimProbe: Diagnosing Simulator Faithfulness in Action-Conditioned World Models for Embodied Manipulation 

**Authors**: Peterson Co, Sicheng Hu, Chunxuan Jiao, Hongyang Cheng, Yulin Luo, Yijie Xu, Sixiang Chen, Zhongxia Zhao, Zihao Wang, DaFeng Chi, Peidong Liu, YuTong Chen, Henghua Liu, Zhihao Yuan, Huizhu Jia, Yuzheng Zhuang, Tianle Zhang, Liang Lin, Huajie Tan, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09298)  

**Abstract**: Action-conditioned world models (ACWMs) promise to provide embodied AI with scalable predictive simulators for planning, policy evaluation, and data generation. Realizing this promise requires precise action-conditioned transitions rather than merely plausible outputs. Yet their applicability remains difficult to establish because prevailing evaluations emphasize visual quality, task outcomes, or coarse rollout-level responsiveness without directly testing simulator fidelity. To address this gap, we evaluate ACWMs through the observable capabilities expected of physical simulators. Accordingly, we formalize Observable Simulator Contract, a minimal contract that any action-conditioned physical simulator should satisfy: supplied actions must induce corresponding agent motion, and environment responses must be grounded in that realized motion. To operationalize this contract, we introduce WorldSimProbe, comprising five controlled suites spanning local control sensitivity, global trajectory variation, source-diverse actions, interaction grounding, and dynamics. Suite-specific evaluators assess simulator-relative calibration, dense action-to-motion correspondence, false-interaction grounding, and primitive-level dynamics. We evaluate six open-source ACWMs on more than 18,000 instances across RoboTwin, ManiSkill, and LIBERO. World-SimProbe reveals systematic action-realization degradation across control variation, structured failures in interaction grounding and dynamics, and benchmark signals consistent with human judgments and downstream outcomes. Together, this capability-based framework provides a transparent, and standardized paradigm for diagnosing ACWM simulator fidelity beyond coarse, task-directed evaluation. 

---
# DAVE: A Decoupled Audio-Visual Enhancement Framework for Real-World Speech Separation 

**Authors**: Wei Zhou, Wanyi Ning, Yinshang Guo, Qianxiao Fang, Haitao Qian, Yingpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09288)  

**Abstract**: Audio-visual speech enhancement under real-world conditions remains challenging due to unreliable visual inputs and the lack of large-scale training data with realistic acoustic conditions. Existing approaches usually fuse visual features directly into the separation network, making them vulnerable to degraded visual signals. In this paper, we present DAVE, a decoupled audio-visual enhancement framework for real-world speech separation. Firstly, to address the data scarcity issue, we construct DAVE-Corpus, a large-scale training corpus with 219,411 mixtures generated from public meeting corpora through combinatorial acoustic augmentation. Then, we introduce a progressive multi-objective optimization strategy to jointly improve speech separation, intelligibility, speaker identity preservation, and perceptual quality. We further develop a certified selective enhancement chain that applies scene routing, GAN-based denoising, and loudness normalization only within the no-reference partition, guaranteeing non-degradation of reference-based metrics. Experimental results on the Real-World Audio-Visual Speech Enhancement Challenge demonstrate the robustness of DAVE under both real-world mixed scenarios and visual degradation conditions. 

---
# UniDFKD: A Unified Semantic Prior Framework for Architecture-Agnostic Data-Free Knowledge Distillation 

**Authors**: Xuewan He, Tong Chu, Zihan Cheng, Yuchen Su, Qianxin Xia, Guoming Lu, Jielei Wang, Wen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09287)  

**Abstract**: Data-Free Knowledge Distillation (DFKD) transfers knowledge from a pretrained teacher model to a compact student model by synthesizing semantically informative data, eliminating the need for access to the original training dataset. Existing DFKD methods rely heavily on architecture-specific statistical priors (e.g., Batch Normalization statistics) to guide data synthesis, however, such architecture-dependent priors are often absent in modern architectures such as Vision Transformers (ViTs), resulting in degraded semantic quality of the synthesized data and consequently catastrophic performance degradation. In this paper, we propose \emph{UniDFKD}, a unified data-free knowledge distillation framework that replaces architecture-specific statistics with explicit, architecture-agnostic semantic priors. \emph{UniDFKD} governs the entire synthesis-distillation pipeline along three dimensions: (1) Categorical Semantic Conditioning (CSC) defines \emph{what} to synthesize by persistently modulating the generator with language-derived embeddings to capture semantic diversity; (2) Spatial Semantic Anchoring (SSA) dictates \emph{where} evidence belongs by anchoring the teacher's spatial attributions to a Gaussian prior; and (3) Spatial Semantic Distillation (SSD) controls \emph{how} knowledge is transferred by explicitly aligning teacher-student spatial evidence alongside predictions. Extensive experiments across CNNs and ViTs demonstrate that UniDFKD establishes a new state-of-the-art, outperforming existing methods by an average absolute margin of over 20\% in both homogeneous and heterogeneous settings. 

---
# VeinCast: Physics-Guided Dynamic Field Graphs with Graph-Conditioned Fusion for Global Medium-Range Weather Forecasting 

**Authors**: Zhisheng Chen, Jinhan Li, Yuxuan Li, Yuan Gao, Hao Wu, Zheng Lu, Jinlong Du, Kun Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2608.09286)  

**Abstract**: Global medium-range weather forecasting requires modeling structured yet state-dependent interactions among heterogeneous atmospheric fields. Existing data-driven models largely learn these interactions implicitly, whereas equation-level physical constraints may inherit approximation and model-form biases. We present VeinCast, a physics-guided dynamic field graph and graph-conditioned fusion framework that jointly forecasts 69 surface and upper-air fields. Within each local window, its Physics-Guided Dynamic Field Graph combines predefined atmospheric relations with state-dependent Top-K residual edges and adapts Earth-window attention using the resulting graph context. Graph-Conditioned Latent Fusion further employs graph context and source-node centrality to guide field-to-latent aggregation, while bounded feedback preserves field-specific information. On the $1.5^\circ$ ERA5 benchmark, VeinCast demonstrates competitive forecasting performance across all 69 meteorological fields at lead times of up to 14 days, compared with representative global weather forecasting models including FuXi, Pangu-Weather, GraphCast, FengWu, and ARROW. Ablations confirm that the two modules provide complementary gains, demonstrating the effectiveness of relational-level physical guidance for data-driven weather forecasting. 

---
# GLocFM: A Geometry-Aware Foundation Model for 3D Indoor Wireless Localization 

**Authors**: Chenghong Bian, Chaozheng Wen, Hongze Chen, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09285)  

**Abstract**: Learning-based wireless localizers often fail to utilize geometric information about the propagation environment, limiting their ability to exploit non-line-of-sight (NLoS) propagation and generalize across scenes. To bridge this gap, we propose GLocFM, a Geometry-aware Localization Foundation Model, which jointly exploits WiFi measurements and scene geometry represented as a 3D point cloud. We formulate localization as a maximum-likelihood (ML) estimation problem, where the goal is to find a transmitter position that maximizes the likelihood of the wireless observations conditioned on the scene geometry. The likelihood of a candidate transmitter position is calculated by a learned scoring function that matches the observed delay--angle-of-arrival (AoA) spectrum against the spectrum predicted for that candidate. A hierarchical scene encoder extracts propagation-relevant features to produce geometric priors for LoS and one-bounce reflection paths. For scenarios with imperfect synchronization, we further introduce a time-of-flight (ToF)-robust GLocFM model to handle unknown ToF offsets. GLocFM is trained on a multi-modal synthetic indoor localization dataset comprising 221 diverse scenes whose associated wireless signals are generated using Sionna RT. On both synthetic and the NeRF$^{2}$ dataset based on real measurements, GLocFM reduces mean 3D localization error relative to one of the state-of-the-art localization baselines by 49.5\% and 48.8\%, respectively. Ablations across different number of receiver, bandwidths, and array sizes further demonstrate the effectiveness and robustness of the proposed framework. 

---
# Software Engineering for and with GUI Agent 

**Authors**: Shengcheng Yu, Yuchen Ling, Junyang Xing, Quan Zhou, Chunrong Fang, Zhenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09278)  

**Abstract**: GUI agents have advanced rapidly, producing a growing body of frameworks, benchmarks, and applications. However, this growth has outpaced the maturity of the field. GUI agents remain technically brittle, incompletely engineered, and insufficiently validated for sustained real-world use. They are evolving into closed-loop software systems. Within these systems, model reasoning is coupled with interface perception, execution feedback, recovery, and human oversight. This evolution calls for a software engineering perspective that remains largely absent from existing research. We address this gap by reviewing 336 GUI-agent papers from January 2018 to April 2026. Five research questions examine the research landscape, architectures, evaluation, software lifecycle concerns, and future opportunities. Our findings show that the field has expanded sharply since 2024, while mobile and web settings remain dominant. Architectures increasingly adopt modular perceive-reason-act loops, but recovery, human escalation, safety enforcement, and auditability remain underdeveloped. This architectural imbalance extends to evaluation. Evaluations are becoming more interactive, but they remain centered on task success and are difficult to compare across protocols. More broadly, existing studies provide limited support for testing beyond benchmarks and for maintaining agents after release. Observability, privacy engineering, and systematic human oversight are also underdeveloped. Together, these findings show that capability improvements alone cannot ensure deployment readiness. Future research should connect dependable execution with lifecycle-centered testing and reproducible evaluation. It should also integrate permission and privacy controls with cost-aware, human-centered governance. This integration is necessary to build dependable, maintainable, secure, and deployable GUI-agent systems. 

---
# SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation 

**Authors**: Jefferson Hernandez, Jaywon Koo, Zilin Xiao, Chen Wei, Vicente Ordonez  

**Link**: [PDF](https://arxiv.org/pdf/2608.09271)  

**Abstract**: Group-based reinforcement learning objectives such as GRPO can allocate learning signal poorly across prompt difficulty: under binary rewards, group normalization induces a divergent weighting on easy prompts. We introduce Softmax Advantage Group Estimation (SoftmaxGRPO), a drop-in alternative that replaces z-score-normalized group advantages with temperature-scaled softmax advantages, keeping weights bounded regardless of prompt difficulty. For binary rewards, we derive the exact finite-group population objective and identify MaxRL as its low-temperature limit. For bounded scalar rewards, we show that the large-group update exactly optimizes a log-moment-generating-function objective, while a universal finite-group scalar objective cannot exist without additional assumptions on the reward distribution. Empirically, SoftmaxGRPO reallocates measured gradient budget away from near-solved prompts and consistently improves over GRPO under identical rewards. It reaches 51.8% on DeepMath with verifiable rewards and improves a 1.5B instruction-tuned model from 35.0% to 68.0% on Poetry using only lightweight text-similarity rewards. 

---
# GRASP: Granularity-Aware Region Alignment and Semantic Prototype Learning for Fine-Grained Cross-Modal Understanding in Drone Views 

**Authors**: Jiahui Cui, Yan Zhao, Kan Wei, Enze Zhu, Peirong Zhang, Lei Wang, Yiru Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09270)  

**Abstract**: Fine-grained cross-modal understanding in drone views is essential for aerial vision-language navigation. However, the inherent wide field of view and overhead perspective of drone scenarios impose dual challenges on vision-language understanding. At the macro level, overwhelming background clutter in visual representations leads to Cross-Modal Focus Misalignment, where the model prioritizes global environmental similarities over specific object details. At the micro level, Visual Isomorphism creates ambiguity, where candidates share similar geometric structures yet differ only in subtle attributes. To address these challenges, we propose the Granularity-Aware Region Alignment and Semantic Prototype (GRASP) learning framework, enhancing discriminative capability through two synergistic strategies. Specifically, we introduce Region-Focused Alignment (RFA) to promote object-centric cross-modal alignment while suppressing background interference. Concurrently, to tackle visual isomorphism, we propose Semantic Perturbation Enhanced Matching (SPEM), which leverages a foreground-purified Semantic Prototype Codebook (SPC) to construct semantically perturbed negatives for fine-grained semantic discrimination. Extensive experiments on the GeoText-1652 benchmark and the unseen ERA dataset demonstrate that GRASP achieves competitive performance in drone-view fine-grained image-text retrieval, validating its effectiveness for cross-modal understanding in aerial scenarios. Our code implementation is available at this https URL. 

---
# Can Coding Agents Solve Repository-Level Issues with Rendered Code? An Exploratory Study of Visual Representations 

**Authors**: Weijie Liang, Yuanfeng Song, Xing Chen, Caleb Chen Cao, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09268)  

**Abstract**: Visual modality has recently been explored as a way to compress textual tokens, including rendering code as images for static code understanding. We study whether this representation can serve as operational context for agentic coding, where an agent must navigate repositories, edit source files, and verify executable patches. Using SWE-bench Verified, we evaluate rendered code in repository-level repair workflows and introduce controlled agent settings to separate unguided repository exploration from more structured repair stages. Our results show a mixed picture. Rendered code consistently reduces prompt-token cost, but the savings do not increase linearly with the nominal visual compression ratio. It largely preserves end-to-end repair accuracy, but does not overcome the performance limits of the underlying model or agent architecture, and can become unstable under aggressive compression. Further analysis suggests that visual code is most useful when raw source reading is a major bottleneck; once repository localization is structured, much of the remaining cost comes from patch--test trial-and-error, where visual compression has limited leverage. Overall, our study positions rendered code as a viable but conditional compression mechanism for realistic coding agents. 

---
# SafeQL: Search-based Refinement for Safe and Efficient LLM-based Text-to-SQL 

**Authors**: Geonho Lee, Min-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.09260)  

**Abstract**: Large language models (LLMs) have advanced Text-to-SQL by enabling natural language interfaces to databases without task-specific fine-tuning. However, existing LLM-based systems remain unreliable, often generating SQL queries that are invalid under the database schema, referencing non-existent tables, attributes, functions, or values. Such errors persist because interactions with the database management system (DBMS) are typically limited to error messages, leaving it in a largely passive role during query refinement. This paper proposes SafeQL, \textit{a search-based refinement paradigm that redefines the role of the DBMS as an active guide in the refinement process}. Instead of regenerating entire queries after execution failure, SafeQL interprets DBMS feedback to incrementally repair only the erroneous components. Each refinement step is formulated as a guided search within a \textit{safe query space}, where candidate queries are progressively validated through DBMS execution, thereby converging to an executable query and preventing repeated regeneration of errors. Experiments on the Bird and Spider benchmarks show that SafeQL significantly improves execution accuracy and efficiency compared to regeneration-based methods. 

---
# MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts 

**Authors**: Peiwen Li, Shiyang Zhang, Yangtian Zhang, Sizhuang He, David van Dijk, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2608.09251)  

**Abstract**: Large language model-based multi-agent systems have recently shown strong potential for complex, long-horizon tasks. However, existing methods mainly rely on coarse prompt-level differentiation without parameter adaptation for diverse subtasks, resulting in insufficient inter-agent heterogeneity and limited specialized capability that bottleneck performance on tasks with complex requirements. To address this, we introduce a Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts (MoRSE) that distinguishes agents with (role, subtask)-conditional specialization at both the task structure and parameter levels. To make agents' responsibility explicit at the task structure level, we formulate a task-oriented multi-agent system that decomposes each task into a dependency-aware Directed Acyclic Graph of subtasks and assigns each agent a specific (role, subtask), introducing task-level specialization across collaborating agents. Additionally, to address the diverse role and subtask parameter adaptation demands, we propose a dynamic Mixture of (role, subtask) LoRA Experts module with a prototype-based semantic router for subtasks, augmenting agents with parameter-level specialization on a shared LLM substrate cost-effectively. Then, to co-optimize experts and router stably under sparse task rewards, we further propose a hierarchical group-relative policy optimization with two-layer credit assignment that isolates expert updates from the cross-route variance introduced by routing decisions, disentangling expert quality from routing quality. Experiments on code-generation benchmarks across three backbones demonstrate the effectiveness of our approach, with improvements in both whole-task and step-wise performance, and the gains from trained specialization generalize across held-out task categories and domains. 

---
# Multimodal Federated Learning under Dual-Axis Modality Missingness 

**Authors**: Adiba Orzikulova, Jaehyun Kwak, Jaemin Shin, Yunqi Guo, Xiaomin Ouyang, Guoliang Xing, Steven Euijong Whang, Sung-Ju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.09240)  

**Abstract**: Multimodal federated learning (FL) supports collaborative modeling in privacy-sensitive health-sensing and medical settings, but realistic deployments often exhibit dual-axis modality missingness: clients have different modality sets, and individual samples may contain only subsets of the modalities available locally. Existing methods typically address these two axes separately. We propose Flux, a multimodal federated learning framework built around two complementary components. First, modality-aware confidence tempering learns sample-specific confidence for each modality through mask-aware unimodal supervision and fuses the confidence estimates from observed modalities into a sample-adaptive temperature that adjusts predictive sharpness according to evidence quality and completeness. Second, gradient-decoupled private adaptation applies this temperature only to a client-private prediction pathway, while training the shared federated model with a standard, untempered objective. This enables sample-specific, client-local confidence adaptation without allowing confidence-dependent gradients to perturb shared representation learning. Across four multimodal datasets, Flux achieves the highest average macro-F1 on every dataset, outperforming the strongest dataset-specific baseline by 0.8~2.2 points and by 1.6 points on average. Additional analyses demonstrate favorable calibration, temperature sensitivity to both modality missingness and input corruption, and more stable shared optimization under private-only tempering. Our code is available at this https URL. 

---
# Privileged Solutions or Context-Induced Teacher Behavior? Dissecting On-Policy Self-Distillation 

**Authors**: Yuki Ichihara, Naoto Iwase, Mohammad Atif Quamar, Junpei Komiyama  

**Link**: [PDF](https://arxiv.org/pdf/2608.09228)  

**Abstract**: On-Policy Self-Distillation (OPSD) is commonly interpreted as the transfer of privileged information: a teacher observes the verified solution to the target problem and supervises the student's trajectory. However, this interpretation conflates two effects. The reference solution not only reveals the answer to the current instance but also changes the context under which the teacher provides token-level supervision. We investigate the role of target-specific privilege with $\mathrm{OP}^{2}\mathrm{SD}$ (On-Policy Self-Distillation from Other Problems), which replaces the paired reference with a problem and solution from a different example, while preserving the student rollout, teacher, and distillation objective. Across three models and three mathematics benchmarks, $\mathrm{OP}^{2}\mathrm{SD}$ improves over the base model, remains competitive with OPSD. The success of $\mathrm{OP}^{2}\mathrm{SD}$ implies that OPSD gains do not necessarily come from access to the reference solution, and that the teacher's context-induced behavior is an important factor. 

---
# RL-Native Distillation: Exploiting Scored Trajectories for Few-Step Image Generation 

**Authors**: Yuhan Li, Fangao Zeng, Sicong Kang, Mengfei Xu, Hao Zhou, Wei Li, Pipei Huang, Bingbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.09226)  

**Abstract**: Efficient text-to-image generation requires both reinforcement-learning (RL)-based reward alignment and few-step distillation, yet these procedures are typically performed sequentially, increasing training cost and risking the loss of reward gains during compression. We instead take an RL-native perspective: diffusion RL already generates reward-scored finite-step trajectories, whose intermediate states provide a natural source of distillation supervision rather than a disposable byproduct of sampling. Based on this insight, we propose REST (Reward-Enhanced Scored-Trajectory Distillation), a single-stage RL-distillation co-training framework that attaches a decoupled student to an arbitrary RL teacher. The student learns segment-wise from the teacher's evolving rollout trajectories while leaving the original teacher optimization unchanged. To prevent uniform imitation from preserving undesirable low-reward behaviors, we further introduce Advantage-Modulated Distillation (AMD), which transforms rollout advantages into signed weights over a base distillation loss. AMD strengthens supervision from preferred trajectories and mildly repels the student from low-reward ones. The resulting framework is lightweight and plug-and-play, requires no extra image rollouts, no separate distillation dataset, and no adversarial training. Experiments on compositional generation, visual text rendering, and human-preference alignment show that REST enables few-step CFG-free inference that matches or surpasses its 40-step RL teacher, with an overall additional training cost below 25% over pure RL. REST improves DrawBench PickScore over RTDMD by 0.82 while requiring only one-fifth of the training iterations. 

---
# Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference 

**Authors**: Tejasvi C. Addagada  

**Link**: [PDF](https://arxiv.org/pdf/2608.09225)  

**Abstract**: The key-value (KV) cache is the primary throughput optimization in modern large language model (LLM) inference, enabling prefix reuse across requests. In multi-tenant deployments this cache is shared across tenants, creating a timing side channel: an adversarial tenant can reconstruct another tenant's private prompt by probing cache-hit latency. Three published attacks exploit it -- PROMPTPEEK, EarlyBird and InputSnatch -- reaching up to 100% attack success rate against unprotected vLLM and SGLang, with rates varying by cache architecture and prompt structure.
We present KVGov, a governance layer addressing all three attack families' prefix-cache paths under one mechanism. A per-principal salt sigma_p = HMAC_K(secret, principal_id) seeds the block-hash chain, making cache keys cryptographically disjoint across principals. An ablation (N=1000 trials, seed 2026, deterministic judges) isolates this salt as the necessary and sufficient component. KVGov adds ORIGAMI, a Stackelberg water-filling audit scheduler that reduces adversary expected utility by 12.6% at realistic tenant heterogeneity (Gini 0.63), and an evolutionary stability analysis giving a 31.6% adversary-prevalence tipping point below which global caching remains stable.
On real hardware (Qwen2.5-7B-Instruct, vLLM 0.26.0, NVIDIA A100) we measure a gate-verified cold/cached TTFT ratio of 0.22, confirming the channel is exploitable at production scale; the defense itself is evaluated in simulation calibrated to those measurements. We replicate the channel on an independent stack (this http URL on Apple Metal, ratio 0.093). Finally, isolation and cache efficiency need not conflict: identifying information resides only where prompts diverge, so injecting the salt at that boundary rather than the chain root retains an estimated 93% of the prefix-cache benefit with no cross-principal signal. 

---
# FedTVD: Balancing Data Quality and Quantity for Robust Federated Learning 

**Authors**: Radwan Selo, Majid Kundroo, Taehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.09221)  

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed client devices while preserving data privacy. However, FL faces significant challenges due to data heterogeneity, particularly in terms of label distribution skewness and variations in dataset sizes, which can lead to biased model updates and hinder convergence. To address this, we propose FedTVD, a novel FL algorithm that weights client contributions during aggregation by considering both data quality and quantity. Unlike traditional FL approaches such as FedAvg, which rely solely on dataset size for client weighting, FedTVD integrates Total Variation Distance (TVD) to measure the divergence between each client's local label distribution and a uniform global distribution. Clients with highly skewed distributions receive lower weights, preventing unbalanced datasets with imbalances from disproportionately influencing the global model. At the same time, dataset size is incorporated to ensure scalability and fairness. This dual-weighting mechanism effectively mitigates the impact of data imbalance, leading to more stable and generalized global models. Experimental results show that FedTVD consistently outperforms state-of-the-art methods across all datasets (FMNIST, CIFAR-10, and CIFAR-100) and all levels of data heterogeneity. Notably, it achieves up to 10.6% improvement over FedAvg on CIFAR-10 under highly skewed data, while maintaining top performance even under moderate and IID settings. 

---
# Beyond Solvability: Task Learnability as a Static Prior for LLM RL Post-Training 

**Authors**: Ting Zhou, Zhenqing Ling, Daoyuan Chen, Qianli Shen, Yilun Huang, Ying Shen, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09217)  

**Abstract**: Reinforcement learning (RL) has become a central post-training paradigm for eliciting reasoning capabilities in large language models, yet uniform task sampling allocates compute without regard to differences in how tasks respond to optimization. Existing task-valuation methods mostly rely on snapshot-based signals such as current pass rate or reward, which estimate how solvable a task is under the current policy. However, tasks with similar current solvability can still differ substantially in how positively they respond to further training. We study this residual axis as task learnability: a regime-conditional measure of expected positive response to continued training under a fixed RL post-training regime. By analyzing per-task reward trajectories, we find that learnability is reproducible across independently sampled training contexts and predictive of downstream utility. To make this signal practical before training begins, we propose TrajVal, a lightweight probe-based estimator that approximates per-task learnability from a short probe run and two endpoint evaluations. TrajVal can be used either as a standalone static prior for task sampling or as a multiplicative prior for existing online schedulers. Experiments on mathematical and logical reasoning benchmarks across multiple model scales show that TrajVal improves data efficiency over uniform sampling and provides complementary gains when combined with online scheduling methods. 

---
# AkasicDB: Demonstrating Omni RAG with a Unified Vector-Graph-Relational DBMS 

**Authors**: Geonho Lee, Jeongho Park, Donghyoung Han, Min-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.09214)  

**Abstract**: Recent Retrieval-Augmented Generation (RAG) systems increasingly combine vector retrieval with structured knowledge, such as Graph RAG and Filtered vector search. However, existing database architectures struggle to support such complex RAG workflows efficiently, as they rely on out-of-DB pipelines or in-DB non-native integration, leading to high overhead. This demo paper presents AkasicDB, a database system that natively supports such RAG workflows by jointly executing vector similarity search, graph traversal, and relational filtering within a single execution framework. AkasicDB extends our prior work, Chimera, with native vector support to enable such unified execution. Based on AkasicDB, we demonstrate the first native integration of Vector-Graph-Relational RAG, which we refer to as Omni RAG. Through an interactive chat-style demonstration, users execute and visualize Omni RAG queries, directly experiencing its superior retrieval and reasoning over vector-only approaches while observing the practical limitations of existing database architectures in supporting Omni RAG. A demonstration video is available at this https URL 

---
# FedA2L: Adaptive layer-wise learning rate adjustment in decentralized federated learning 

**Authors**: Van Truong Vo, Khoa Nguyen, Taehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.09208)  

**Abstract**: Decentralized intelligence systems with heterogeneous devices and limited coordination increasingly rely on decentralized federated learning (DFL). However, DFL suffers from convergence inefficiency under data heterogeneity due to the use of a uniform learning rate (LR) that ignores layer-specific optimization needs. Foundational layers are responsible for maintaining network consensus, while specialized layers adapt to local data characteristics, leading to conflicting gradients and degraded performance under non-IID conditions. To address this fundamental tension, this work introduces FedA2L, a method that dynamically adjusts layer-wise LRs based on model divergence signals. By leveraging local update intensity and network consensus constraints, FedA2L seamlessly integrates into existing DFL protocols without additional communication or coordination. Extensive evaluations across DFL algorithms, various model architectures, and datasets demonstrate that FedA2L achieves up to 4.94 times faster convergence than vanilla DFL and reduces communication rounds by up to 59% compared to scheduler-based baselines. Furthermore, FedA2L exhibits resilience to severe data heterogeneity, larger network sizes, and sparse topologies, reducing communication overhead and establishing it as a versatile optimization tool for resource-constrained or large-scale distributed learning in edge and IoT deployments. The code is released at this https URL. 

---
# SiriusDeliver: Automating Data Warehouse Delivery at Tencent 

**Authors**: Haining Xie, Xiaokai Zhou, Jiaming Yang, Siqi Shen, Ziwei Wang, Yifeng Zheng, Tengyue Xu, Yipeng Shi, Zefang Zong, Yang Li, Peng Chen, Jie Jiang, Debiao He, Xiao Yan, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09185)  

**Abstract**: Enterprise data warehouses (DWs) support business-critical analytics, but warehouse task delivery remains a complicated production process involving context retrieval, workflow configuration, code generation, platform submission, and failure diagnosis. Although large language models (LLMs) and coding agents have improved software development, they are insufficient for production DW delivery, which requires dependency-aware orchestration, lifecycle-aware artifact control, and continuous adaptation to evolving platform practices. We present SiriusDeliver, an end-to-end delivery automation agent for production warehouse task submission. SiriusDeliver integrates three components: a hierarchical delivery agent that orchestrates warehouse skills, an artifact lifecycle control module that verifies and revises artifacts before and after platform execution, and a trace-driven skill evolution mechanism that maintains reusable skills from delivery trajectories. We evaluate SiriusDeliver through offline datasets and large-scale production deployment on Tencent Cloud WeData. Offline experiments on real-world warehouse delivery cases show that SiriusDeliver improves delivery success and automation efficiency over representative baselines. During a two-month deployment across 6 business teams and 4 warehouse task types, SiriusDeliver served 3,600 monthly active users and supported 18,240 delivery sessions, achieving an 87.2% end-to-end success rate and a 73.5% autonomous submission rate. A one-month A/B test shows that SiriusDeliver reduces median delivery time from 228 to 23 minutes and engineer effort from 95 to 11 minutes, while maintaining comparable final delivery success. 

---
# Rethinking Medical Landmark Localization with Prototype Learning-based Progressive Offset Correction 

**Authors**: Jingxian Xu, Yuhao Huang, Rusi Chen, Yanfeng Zhou, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.09182)  

**Abstract**: Accurate landmark localization in medical images is a fundamental step for quantitative clinical measurement and downstream analysis. Existing localization methods have advanced, among which multi-stage refinement is a superior solution. Although this strategy mitigates the anatomical ambiguity inherent in single-stage global predictions, its high computational cost limits practical applicability. In this work, we propose a parameter-economic model, PPOC-LL, which leverages Prototype learning-based Progressive Offset Correction for Landmark Localization. Our contribution is three-fold. First, to drive coarse-to-fine landmark optimization, we introduce a multi-scale dynamic perception strategy for patch-level feature pyramid modeling. Second, to effectively handle anatomically similar patterns, we design a similarity-driven prototype learning mechanism that captures informative local semantics for robust offset prediction. Last, to stabilize the model learning and improve the overall performance, we incorporate a novel error-aware reliability regularization via tolerance-based balancing. We collected a large validation cohort, including two public and one private datasets spanning X-ray and ultrasound modalities, covering cephalometric, symphysis-fetal head, and fetal heart landmarks. Extensive experiments demonstrate that PPOC-LL achieves satisfactory performance with a favorable trade-off between accuracy and model complexity. 

---
# Not All Visual Tokens Are Equally Safe to Remove:Consequence-Sensitive Visual Token Compression 

**Authors**: Jingbo Wen, Liang He, Mingyu Cao, Haoyu Wang, Minxuan Hu, Kangning Cui, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09176)  

**Abstract**: Visual token compression for vision--language models (VLMs) has largely relied on criteria such as attention, redundancy, and uncertainty to maximize average accuracy under a fixed compute budget, implicitly assuming that all errors carry equal cost. However, the consequence of an incorrect prediction on downstream tasks is rarely symmetric: misreading an invoice amount can be far more costly than misclassifying a background color. Motivated by this, we introduce consequence-sensitive visual token compression, which allocates visual computation across requests according to their potential error costs. Our method follows a calibrate-then-allocate procedure, estimating consequence-specific error-budget curves offline and applying the calibrated token budgets online using consequence signals available from question or task information. On a controlled within-task benchmark, high- and low-consequence questions are drawn from the same document images, so content alone cannot reveal which questions are costly to get wrong. In this setting, our method reduces high-stakes errors from 0.300 to 0.133 under the same total token budget, whereas a content-driven allocator performs no better than uniform allocation. Measuring how error rates change with token budget across different cost ratios, we derive an allocation frontier: uniform allocation is optimal when errors are equally costly, and token transfer toward high-consequence questions becomes increasingly beneficial as the cost gap grows. This allocation principle generalizes well across three dense vision-language benchmarks, two budget realization mechanisms (token deletion and resolution reallocation), two VLM architectures, and multiple token selection strategies. On a realistic mixed workload, consequence-sensitive allocation reduces cost-weighted error by 38% while achieving approximately 21% lower latency than full-resolution inference. 

---
# Tabular Numeric Stretch Transformation 

**Authors**: Zihao Ye, Juyong Kim, Johnna Sundberg, Burak Varici, Pradeep Ravikumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.09162)  

**Abstract**: Tabular data presents unique challenges for deep learning due to its heterogeneous nature, where numeric features exhibit diverse distributions, scales, and statistical properties. Although recent advances have improved how models learn from tabular data, how numeric data are transformed into model-friendly representations remains comparatively underexplored. We introduce the stretch transformation framework, which formulates numeric feature preprocessing as an optimization problem to make the target function smoother and thus more learnable. Our framework has two variants: (1) unsupervised stretch, which uniformly redistributes feature density via minimax optimization, and (2) supervised stretch, which optimizes target-aware numeric feature transformations from the perspective of target-function smoothness by minimizing the target function's Dirichlet energy in the transformed space. Our theoretical analysis further connects this framework to several popular transformations: unsupervised stretch is closely related to Piecewise Linear Encoding through a shared piecewise-linear geometry and approaches the empirical CDF transformation as the number of bins grows, while supervised stretch becomes closely related to target encoding in the fine-binning limit. Comprehensive experiments on 38 datasets from the TALENT benchmark demonstrate that supervised stretch consistently outperforms all baselines. These results show that explicitly optimizing for target function smoothness is a powerful and underexplored strategy for tabular deep learning. 

---
# From Inaudible Inputs to Model Failures: Low-Frequency Safety Risks in LALMs 

**Authors**: Yuanhe Zhang, Weiliu Wang, Jie Ren, Liang Lin, Zhenhong Zhou, Haoran Gao, Kun Wang, Chen Li, Li Sun, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2608.09158)  

**Abstract**: Large audio-language models (LALMs) have demonstrated strong capabilities in understanding diverse audio inputs. This diversity includes low-frequency signals that are inaudible to humans but can still enter the model and influence its generation. However, the practical impact of such low-frequency inputs on LALMs remains largely unexplored. In this paper, we propose Intermittent Low-Frequency Lockout (ILL), an inaudible red teaming method that evaluates this risk using a universal waveform template in a black box setting. ILL uses Sentence Attention Scale Estimation to determine active intervals and Frequency Confusion Transfer to construct a low-frequency waveform with continuous phase from corpus spectral variation. To mitigate this risk, we propose Distributional Requery Guard (DRG) to detect low-frequency distribution shifts and conditionally request a second recording for semantic recovery. Across six LALMs and multiple audio understanding tasks, ILL reduces accuracy by up to 67 percentage points while receiving a mean human audibility rating of 1.33, close to 1.17 for clean audio; DRG raises mean attacked accuracy from 28.5\% to 46.1\% after clean reacquisition. These findings identify a previously overlooked safety risk for LALMs and provide a foundation for future research on robust audio understanding. 

---
# SpeedTuning: Speeding Up Policy Execution with Lightweight Reinforcement Learning 

**Authors**: David D. Yuan, Tony Z. Zhao, Kaylee Burns, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2608.09138)  

**Abstract**: While learned robotic policies hold promise for advancing generalizable manipulation, their practical deployment is often hindered by suboptimal execution speeds. Imitation learning policies are inherently limited by hardware constraints and the speed of the operator during data collection. In addition, there are no established methods for accelerating policies learned via imitation, and the empirical relationship between execution speed and task success remains underexplored. To address these issues, we introduce SpeedTuning, a reinforcement learning framework specifically designed to enhance the speed of manipulation policies. SpeedTuning learns to predict the optimal execution speed for actions, thereby complementing a base policy without necessitating additional data collection. We provide empirical evidence that SpeedTuning achieves substantial improvements in execution speed, exceeding 2.4x speed-up, while preserving an adequate success rate compared to both the original task policy and straightforward speed-up methods such as linear interpolation at a fixed speed. We evaluate our approach across a diverse set of dynamic and precise tasks, including pouring, throwing, and picking, demonstrating its effectiveness and robustness in enhancing real-world robotic manipulation. 

---
# When Latents Forget Pixels: Restoring Fidelity in Diffusion Transformer Super-Resolution 

**Authors**: Yu Shi, Yuyao Zhang, Yu-wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09133)  

**Abstract**: Image super-resolution (SR) with large generative models has recently achieved remarkable perceptual quality, yet maintaining fidelity to the LR observation remains challenging. In particular, we observe that diffusion transformers (DiTs) built on latent representations suffer from a critical limitation: the compression bottleneck of the VAE weakens fine-grained spatial information, leading to hallucinated details that are weakly grounded in the input image. In this work, we revisit generative SR from a representation perspective and propose a pixel-grounded super-resolution (PGSR) framework that preserves LR-observed pixel evidence before VAE compression and reuses it throughout restoration. Instead of relying solely on the compressed latent condition, PGSR extracts pre-VAE pixel evidence from the upsampled LR image and reuses it at two stages. First, Condition-Side Trajectory Guidance fuses LR-derived pixel evidence with the latent LR condition to guide the latent restoration trajectory. Second, Decoder-Side Pixel Grounding injects multi-scale pixel features into the frozen VAE decoder to ground the final rendering with LR-observed cues. To efficiently adapt large pretrained DiT models, we keep the latent autoencoder and main flow-matching backbone frozen, and train only lightweight restoration modules. We further study an efficient local-window attention variant for improved high-resolution efficiency and scalability. Extensive experiments demonstrate that PGSR improves the realism--fidelity trade-off and produces more faithful, visually convincing results than existing latent generative SR approaches. 

---
# MARA: Flow-Matching-Guided Multi-Agent Resource Allocation for Computational Resource Efficient Learning 

**Authors**: Hanye Zhao, Muning Wen, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09130)  

**Abstract**: Allocating limited computation among concurrent learning tasks is difficult when each task must reach a target loss before a deadline but its required training effort is unknown. Existing approaches combine online loss prediction with adaptive resource allocation, yet commonly treat computation as continuously divisible throughput. We instead study a practical setting in which tasks arrive over time and computation is provided by discrete nodes. This setting introduces both uncertain demand and constrained sequential decisions. We propose MARA, which predicts future loss trajectories with conditional flow matching and coordinates compute nodes through a cooperative multi-agent autoregressive policy. A potential-based progress reward supplies intermediate training feedback while preserving the undiscounted task-completion objective. Across in-distribution, reinforcement-learning, and vision workloads, flow matching reduces remaining-resource prediction error relative to weighted least squares. At the scheduler's training load, MARA completes 63.46% of tasks on average, 8.54 percentage points above strong baseline Learning with Adaptive Resource Allocation (LARA), and remains ahead under unseen heavier workloads. 

---
# Social Gym and SPaRTan: Benchmarking and Improving LLM Social Reasoning via Multi-Agent Game Tournaments 

**Authors**: Keyu He, Xuhui Zhou, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2608.09128)  

**Abstract**: LLM agents are increasingly deployed in multi-agent social settings where they must cooperate, negotiate, and adapt to other agents. Measuring and improving these social skills is hard because, unlike math or logic, social interaction offers no objective ground truth: evaluations fall back on LLM judges, which are costly, subjective, and noisy, and models get no reliable signal to learn from. To address both, we first introduce Social Gym, an environment of 21 multi-agent social games (e.g., Werewolves, Resistance, Spyfall) whose rule-decided outcomes make agent performance verifiable and objective, with an Elo tournament that produces a cross-game leaderboard. Benchmarking experiments show that while GPT-5-mini tops the leaderboard, no model excels at all games uniformly or in all game roles, pointing to limitations of social reasoning. Motivated by this, we additionally propose SPaRTan (Self-Play and Reflect-Transfer), a training-free self-improvement loop: a model plays a game, reflects on its trajectories and their outcomes to produce a transferable playbook, and applies that playbook in subsequent games. Our results show that SPaRTan playbooks help GPT-5-mini agents level their performance on weaker roles, but largely do not improve Qwen3-32B's performance. Together, Social Gym and SPaRTan offer a reproducible, verifiable foundation for measuring and improving LLM social reasoning without weight updates. 

---
# Visual Distortion Detection in UGC Images Using Large Multimodal Models 

**Authors**: Ziheng Jia, Yingji Liang, Jiaying Qian, Xiongkuo Min  

**Link**: [PDF](https://arxiv.org/pdf/2608.09122)  

**Abstract**: The localized depiction of perceptual quality has long been a crucial, yet underexplored, challenge in image quality assessment (IQA). Existing approaches based on large multimodal models (LMMs) predominantly rely on text-driven supervised fine-tuning (SFT).
However, this training paradigm exhibits notable limitations in detection accuracy. Moreover, synthetically distorted images, which are often used as the primary training data source,
show a significant generalization gap when deployed in real-world scenarios; thus, the \textbf{synthetic-to-authentic (\textit{S2A})} problem represents a critical challenge. Motivated by these issues, we propose \textbf{\textit{VIGIL}}, which leverages the LMM architecture for precise visual distortion detection. From a candidate pool of over 1000K samples, we construct the \textbf{\textit{VIGIL-140K}} training set, which consists of over 140K distorted images. These images are obtained through rigorous quality filtering and carefully crafted distortion injection, covering 8 major synthetic distortion categories.
Our model leverages different layers of the large language model (LLM) decoder, treating them as \textit{multiple detectors} that perform synchronous distortion detection using multi-level features. Additionally, we retain distortion cues from predictions assigned to the non-distortion class, which helps mitigate the ambiguous foreground-background (\textit{FG-BG}) separation commonly encountered in the \textit{S2A} problem.
After post-processing, our model consistently outperforms strong baselines on both in-domain synthetic distortion detection and \textit{S2A} tasks. 

---
# The Announcement Carries the Cue: Markup, Boundaries, and the Notation of Pre-Training Corpora 

**Authors**: E. M. Freeburg  

**Link**: [PDF](https://arxiv.org/pdf/2608.09093)  

**Abstract**: How a document's arrangement is written down, its notation, is a training variable that no dataset card records. The field has established that text-extraction choices change model behaviour, and has never once measured the notation of what those choices put into the corpus. We define clean-window survival, a deterministic count of how much of a stream still demands the boundary inference, and measure notation on three fronts. What corpora carry: a census of thirteen public corpora, where survival falls to 0.153 in a vision-converted PDF slice against 0.889 in C4; the scarce resource is not unmarked text but long unmarked text; a pre-registered supply test finds what remains institutional, not consumer. Our own pre-registered prediction failed: converters do not fabricate structure on prose, and that null forced the reliability mechanism that survives it. What readers use: across five base models spanning 0.6B to 8.2B and two pipelines, deleting a structural announcement makes the following prose measurably harder to predict, while swapping its notation moves nothing. That zero does not make notation unimportant; it relocates the variable: the operative cue is the announcement, not the sigil. What writers impose: a bounded null. Base models do not impose the marked register above the authored baseline, and handed prose with every announcement deleted they do not put one back, at a rate indistinguishable from zero against an authored reference of zero. We ship the format those measurements imply: the pure frame, paragraphs in authored order, every announcement deleted into a reversible sidecar, mixed against the marked copy over announcement presence rather than notation. Choose format operators by the capability they train, not by the fidelity they preserve, and record extractor identity and survival on data cards. 

---
# TLDChoiceNet: Quantitatively Choosing a Transfer Learning Dataset 

**Authors**: Jing Ning, James D. Braza  

**Link**: [PDF](https://arxiv.org/pdf/2608.09091)  

**Abstract**: Transfer learning is particularly useful in settings with limited training data, and within image classification it is common to transfer learn upon massive datasets like ImageNet , CIFAR-100, or COCO . Qualitatively, it seems a transfer learning dataset should have both more classes and more examples per class than the fine tuning dataset; however, a quantitative method to choose the best transfer learning dataset does not currently exist. In this paper, we design TLDChoiceNet, a model to choose the best transfer learning dataset given a fine tuning dataset by predicting the test-set accuracy after fine-tuning. A simple version 1 achieves 0.154 MSE on the test dataset, while a version 2 leveraging an ImageNet pre-trained ResNet50 v2 embedding with per-class information attains a 5X lower MSE of 0.031. We further design two metrics that enable an unsupervised method of choosing an optimal transfer learning dataset: distribution distance (DD), which linearly regresses against fine-tune accuracy with an R2 of 0.89, and average class correlation (ACC), which improves the R2 to 0.97. Our results underscore that a dataset's low-level statistics can explain the transfer learning effect, and that using a pre-trained ImageNet can embed different classes further apart in latent feature space. 

---
# When Confidence Fails: Overconfidence in LLMs under Uncertainty and Missing Clinical Information 

**Authors**: Maryam Tahermazandarani, Adnan Mahmood, Fahmida Islam, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09080)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in medical question answering and clinical reasoning tasks. However, their reliability under uncertainty remains poorly understood which raises critical concerns for deployment in high-stakes clinical settings. In such environments, incorrect predictions are inherently risky, but confident incorrect predictions can be particularly harmful as they may mislead clinical decision-making. In this paper, we conduct a systematic behavioral analysis of LLMs under clinical information uncertainty. We propose an evaluation framework based on the MedMCQA dataset consisting of two complementary uncertainty settings. First, we introduce linguistic uncertainty cues through prompt modifications to simulate ambiguous clinical contexts. Second, we construct an answer removal setting, wherein the correct option is deliberately excluded mandating the model to recognize insufficient information and abstain. We analyze both model accuracy and confidence behavior using multiple calibration metrics including calibration gap, Expected Calibration Error (ECE), and Unsafe Confident Error Rate (UCER) across 500 medical questions. Our results reveal a consistent failure mode, i.e., although accuracy degrades under increasing uncertainty, model confidence remains misaligned with accuracy. This leads to a substantial increase in unsafe confident errors, indicating that model confidence remains largely insensitive to clinically meaningful information loss. Furthermore, we observe significant variation across models in their ability to abstain when the correct answer is unavailable, with some models persistently producing high confidence hallucinated answers. These findings expose critical limitations in the epistemic reliability of current LLMs and highlight the need for uncertainty aware evaluation methods prior to their deployment in clinical workflows. 

---
# A Unified Issue Resolution Benchmark for Requirement Clarification, Planning, and Code Generation for Coding Agents 

**Authors**: Xin Zhou, Chun Yong Chong, Kisub Kim, Yun Peng, Rui Shu, Zihan Wu, Xu Han, Guowen Yuan, Zeyang Zhuang, Jounghoon Kim, Jeongjin Ju, Seongmin Ju, Taein Yoon, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09072)  

**Abstract**: Large language model-powered coding agents are increasingly used to modify existing code repositories, for example, by adding features or fixing bugs. Yet existing repository-level benchmarks typically evaluate only whether the final patch passes tests. Satisfying a user request requires a long chain of interdependent reasoning and decisions: an agent must recover explicit and implicit requirements, formulate a repository-grounded implementation plan, and translate it into correct code. A pass/fail outcome cannot characterize how an unsuccessful trajectory diverges from the requirements and implementation process needed for a correct patch. To address this gap, we introduce SWE-RPG, a repository-level benchmark that combines executable patch evaluation with validated ground-truth references (GTs) for (1) Requirement Clarification and (2) Implementation Planning. These intermediate GTs support retrospective, GT-aligned diagnosis of complete coding-agent trajectories across clarification, planning, code generation, and artifact submission. SWE-RPG comprises 163 tasks from 31 Python and Java repositories, including 113 bug fixes and 50 feature additions. We evaluate 3 coding agents, including Claude Code, Codex, and OpenCode, with 6 large language model backends, including Claude-Sonnet-5 and GPT-5.6-Terra. Results show that the evaluated popular coding agents still struggle to implement user requests in existing repositories, achieving an average resolved rate of only 31.5% on SWE-RPG. Intermediate-GT diagnosis further identifies implicit requirement recovery as the main bottleneck, accounting for 24.5%--46.0% of agent runs. This result suggests implicit-requirement recovery as a key candidate direction for improving coding agents. The benchmark data and evaluation code are available at this https URL. 

---
# Two-Step MV-DeepONet: Probabilistic Operator Learning for Uncertainty Propagation Driven by Random Input Fields 

**Authors**: Yupei Nie, Lei Wang, Jiasen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09071)  

**Abstract**: Forward uncertainty propagation in complex physical systems can induce structured covariance across field-valued outputs. For a probabilistic surrogate, the total predictive covariance comprises the covariance of conditional means across input realizations and the average conditional predictive covariance. Probabilistic DeepONet (Prob-DeepONet) provides lightweight uncertainty quantification by predicting pointwise Gaussian means and variances in a single forward pass, but its conditional predictive covariance is restricted to a diagonal form. To represent cross-location conditional dependence without explicitly parameterizing a full high-dimensional covariance matrix, we develop a two-step mean-variance DeepONet (two-step MV-DeepONet) through two principal modifications. First, two-step training is used to decouple output-basis learning from the input-to-coefficient mapping, together with basis orthogonalization and subspace rotation. Second, Gaussian probabilistic modeling is transferred from the high-dimensional physical output space to the low-dimensional rotated coefficient space. Mapping these probabilistic coefficients through the shared basis induces a generally non-diagonal conditional predictive covariance in the physical output space while retaining single-pass inference. A Frobenius-norm error decomposition and corresponding upper bound identify low-rank covariance compressibility, trunk-subspace approximation, finite-sample statistical error, and coefficient-space covariance estimation as the principal factors governing covariance recovery. Numerical experiments on three representative problems governed by partial differential equations (PDEs) and a hypersonic blunt-body aerothermal problem show improved generalization, more structured uncertainty bands, and accurate recovery of off-diagonal correlation patterns compared with Prob-DeepONet. 

---
# Triple Expert Learning from Noisy Labels for Semi-Supervised Vision Foundation Model Adaptation 

**Authors**: Xuanyu Liu, Zheng Fang, Hongyang He, Yundi Hong, Daizong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09052)  

**Abstract**: Semi-supervised adaptation of vision foundation models (VFMs) commonly freezes the pretrained backbone and updates lightweight modules such as LoRA. However, pseudo-labels have mixed reliability, and a single LoRA adapter must absorb reliable, ambiguous, and noisy gradients in the same low-rank space. This can make VFM adaptation sensitive to pseudo-label noise. We propose \textbf{TriNoL}, a \textbf{Tri}ple-expert learning framework from \textbf{No}isy \textbf{L}abels for semi-supervised VFM adaptation. TriNoL routes unlabeled samples into three confidence regions and assigns them to three LoRA experts: a Positive Expert for high-confidence pseudo-labels, an Alignment Expert for medium-confidence ambiguous samples, and a Negative Expert for low-confidence noisy samples. The VFM backbone remains frozen, and only the LoRA experts and classifier head are updated. By separating different pseudo-label reliability regions into specialized adaptation paths, TriNoL improves robustness to noisy supervision while keeping the training cost low. 

---
# Bridging the Gap Between Semantics and Reconstruction:Unifying Sign Language Translation and Production 

**Authors**: Xiao Liu, Shiwei Gan, Yafeng Yin, Jiaxin Yin, Bowen Guo, Yaqi Sun, Zhiwei Jiang, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.09045)  

**Abstract**: Recent advances in sign language (SL) research have shown a trend toward unifying multiple sign language understanding (SLU) subtasks, such as isolated sign language recognition (ISLR), continuous sign language recognition (CSLR), and sign language translation (SLT), within a single framework, leading to substantial progress. Meanwhile, sign language production (SLP), which generates sign sequences from text, has also attracted growing attention. This naturally raises an important question: can sign language understanding and production be unified within a single framework? Compared with unifying SLU subtasks, this problem is substantially more challenging. Existing SLU tasks largely share the same direction of mapping, namely from sign inputs to linguistic outputs, whereas SLT and SLP lie in opposite directions of sign-text mapping. A unified framework must therefore address two key challenges: (1) bridging the modality gap between continuous sign motions and discrete text tokens through a shared sign tokenizer that supports both linguistic abstraction and motion reconstruction; and (2) learning a single conditional autoregressive model that can take either sign or text as input and generate the corresponding target sequence in the opposite modality. To this end, we propose Uni-SLTP, a unified framework for SLT and SLP with two key components: (1) a shared sign tokenizer that converts sign sequences into discrete tokens and latent representations, capturing both semantic and reconstructive information; and (2) a unified autoregressive generation model that formulates both tasks as conditional sequence generation. Experiments on widely used public datasets show that Uni-SLTP achieves superior motion accuracy for SLP while maintaining competitive SLT performance. 

---
# Don't Scroll Back: Missing-Evidence Memory for Streaming Dialogue Summarization 

**Authors**: Hyangsuk Min, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.09043)  

**Abstract**: Users of modern platforms repeatedly need summaries of recent dialogue, but the window rarely contains enough context to be interpreted on its own. We formalize this setting as streaming dialogue summarization, where a system must summarize a current window using selective memory from an unbounded history under a fixed budget. We show that the central challenge is not how much history is accessed, but whether memory recovers the evidence that the current window presupposes. We construct a benchmark and evaluation protocol that separately assesses whether memory contains gap-resolving evidence and whether the generated summary reflects it. We propose ReMEMBER, a missing-evidence memory framework that conditions retrieval on unresolved window dependencies and refines retrieved chunks into evidence-dense memory under a fixed budget. Experiments on dialogues with histories up to 160K tokens show that ReMEMBER improves memory recall and gap-resolution completeness over memory construction baselines under the same budget. 

---
# MusicLayout: Explicit Structural Planning for Controllable Text-to-Music Generation 

**Authors**: Shuyu Li, Kejun Zhang, Jiahe Lei, Shulei Ji, Zihao Wang, Jiaxing Yu, Wanying Wu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09035)  

**Abstract**: Text-to-music generation has advanced rapidly, but current systems still rely primarily on global text prompts, leaving the structural organization of generated music implicit and difficult to inspect, control, or revise before audio generation. To address this issue, we introduce MusicLayout, an explicit intermediate representation for controlling musical structure in text-to-music generation. MusicLayout describes a musical piece as a time-aligned layout of sections, textures, repetitions, variations, and instrument-level arrangements, serving as an interpretable planning layer between textual intent and the generated music. We integrate MusicLayout into a text-to-music framework built on a unified autoregressive formulation, where the model first generates a MusicLayout representation and subsequently predicts audio tokens conditioned on this representation within a single sequence. The resulting MusicLayout can be inspected and modified prior to audio generation, providing a mechanism for layout-level structural control. We evaluate MusicLayout through layout-conditioned generation, layout manipulation experiments, and matched-data ablations, providing evidence that explicit layout planning can improve long-range structural organization and support layout-level control. 

---
# How People Evaluate AI-, Expert-, and Peer-Style Financial Advice 

**Authors**: Aryan Ramchandra Kapadia, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2608.09019)  

**Abstract**: As generative AI increasingly becomes a common source of daily decision-making, including financial choices, it is critical to understand how people evaluate AI-generated financial advice. We conducted a preregistered vignette experiment (N = 285) in which substantive financial content---including facts, numerical values, recommendation direction, and core reasoning---was held constant while communication style varied across AI Financial Assistant (AI), Certified Financial Planner (Expert), and Online Community Forum (OC) advice. Displayed source attribution was independently manipulated through correctly labeled, unlabeled, and mislabeled conditions, allowing us to separate attribution effects from source-specific communication cues. Expert advice was rated more favorably than AI advice on 9 of 10 outcomes (|d|=0.20--0.47), and this advantage remained visible without source labels, where Expert advice outperformed AI advice on 8 of 10 outcomes (up to d=0.60). Correct labels added limited differentiation, whereas mislabeling increased ratings of AI advice for situational fit and overall quality (d=0.42 for each) and attenuated the Expert advantage in situational fit (d=-0.36). Descriptive analyses further showed that AI advice was most responsive to displayed attribution and, conversely, that advice-style differences were most visible under an AI label. These findings show that financial-advice evaluations are shaped jointly by displayed attribution and message-level communication cues. We position disclosure not as a neutral transparency mechanism, but as an interpretive frame whose accuracy and interaction with message cues can shape trust and reliance. 

---
# SignLlama: Enhancing Gloss-free Sign Language Translation by Prioritizing Visual Features for LLMs 

**Authors**: Shiwei Gan, Xiao Liu, Yafeng Yin, Zhiwei Jiang, Bowen Guo, Lie Xie, Sanglu Lu, Hongkai Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09006)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks. However, fine-tuning LLMs for Gloss-Free Sign Language Translation (GFSLT) remains a challenge. In this paper, we investigate how to effectively adapt LLMs to the GFSLT task. We show that there are two key issues that need to be solved: (1) the inherent distributional gap between visual feature inputs and text feature inputs makes it difficult for LLMs to interpret visual inputs; and (2) existing approaches typically concatenate visual and textual features in an autoregressive framework, which leads to the model overemphasizing textual inputs and deprioritizing visual cues, as LLMs are pretrained predominantly on text-centric data. To address the first challenge, we propose a simple yet effective method named Filtered Pseudo-Gloss CTC Pretraining, which leverages filtered pseudo-gloss sequences generated from text sequences to supervise the training of the visual backbone. To tackle the second issue, we introduce a Visual-Prioritized Distillation training strategy. Specifically, we define a visual-only prediction path in which text inputs are masked, and the model is required to generate the target sequence relying solely on visual inputs. To guide this path, the outputs from the standard visual-textual prediction are then distilled into the visual-only prediction path, encouraging the model to prioritize visual features. Comprehensive experiments and qualitative analyses demonstrate the effectiveness of the proposed model. The proposed SignLlama achieves very competitive performance on multiple datasets for GFSLT tasks, without using any extra modalities or external sign language datasets for pretraining. 

---
# DeepFreqMark: End-To-End Learnable Frequency-Domain Watermarking with Spherical Attack Simulation for Latent Diffusion Models 

**Authors**: Chen-Hsiu Huang, Mario Köppen, Ja-Ling Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08999)  

**Abstract**: The proliferation of AI-generated images produced by Latent Diffusion Models (LDMs) has raised critical concerns regarding copyright infringement and misinformation. Although existing frequency-domain watermarking methods embed handcrafted geometric patterns into the initial latent noise prior to generation, they suffer from limited capacity and rigid pattern designs. We propose DeepFreqMark, an end-to-end learnable frequency-domain watermarking framework that replaces manual pattern engineering with a neural message encoder and decoder. To circumvent the computational bottleneck caused by Denoising Diffusion Implicit Model (DDIM) inversion during training, we introduce a Spherical Linear Interpolation (Slerp)-based attack simulation. This approach operates directly on the noise latent while strictly preserving the Gaussian variance. Extensive experiments demonstrate that DeepFreqMark achieves significantly lower Bit Error Rates (BER) than baseline methods under real-world attacks and scales to 256 bits message capacity. Our source code is available at this https URL. 

---
# Multi-agent discovery of practical quantum LDPC codes 

**Authors**: Dongheng Qian, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.08996)  

**Abstract**: Quantum low-density parity-check (qLDPC) codes can encode multiple logical qubits using sparse parity checks, yet searching for useful finite-length instances remains a challenging design problem because code performance must be optimized while satisfying practical constraints. Motivated by recent advances in artificial-intelligence agents for scientific discovery, we develop a multi-agent framework for discovering practical qLDPC codes. The framework combines specialist proposal and review, persistent scientific memory, long-horizon evolution of executable programs, and deterministic construction and evaluation within a closed-loop search. These programs instantiate coset-orbit balanced-product codes, providing a search space that includes bicycle and lifted-product constructions as well as non-normal subgroup actions. To incorporate practical constraints, we restrict the search to binary CSS codes with block length $n\leq400$ and overall weight $w\leq10$. Within this regime, the framework discovers codes with leading or competitive rate--distance performance in every weight class considered, with representative instances including $[[288,16,18]]$ at $w=7$, $[[288,18,18]]$ at $w=9$, and $[[234,28,18]]$ at $w=10$. The search also uncovers structurally distinct, high-performing constructions, including a $[[336,12,\leq24]]$ candidate and a $[[368,18,16]]$ code, both of which are genuine balanced-product constructions with non-normal subgroup actions. When evaluated under code-capacity depolarizing noise using a common BP-OSD decoding protocol, the discovered codes also exhibit low logical failure rates. Together, these results provide hardware-relevant finite-length candidates for further experimental evaluation and show how structured agentic search can contribute to scientific discovery. 

---
# Guardian Crawler: Retrieval-First Knowledge Discovery with Bounded LLM Augmentation for Noisy Web Intelligence 

**Authors**: Joshua Castillo, Santosh Nukavarapu, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2608.08994)  

**Abstract**: Retrieving relevant evidence from noisy web data is challenging, particularly in sensitive domains containing incomplete reports, heterogeneous language, and irrelevant content. We present Guardian Crawler, a reproducible retrieval-first testbed for controlled experiments on knowledge discovery and evidence-grounded summarization over synthetic web-like corpora. The architecture combines BM25 retrieval with risk-aware, embedding-augmented, and hybrid reranking, followed by constrained retrieval-augmented generation with explicit document citations. Experiments on a synthetic 900-document corpus and 10 queries produced the highest descriptive retrieval scores under risk-based reranking, with P@10 = 1.00 and NDCG@10 = 0.94, compared with 0.94 and 0.81 for BM25. The best hybrid and BM25+Semantic configurations reached NDCG@10 values of 0.94 and 0.88, respectively. All 41 evaluable generated bullets passed the lexical coverage threshold; an automated LLM judge classified 36 as supported, one as partially supported, and four as unsupported. These results demonstrate the feasibility of Guardian Crawler as a controlled testbed but do not establish statistical superiority, human-validated faithfulness, or transfer to live-web investigative environments. 

---
# How Far Do Foundation Models Transfer to Infant Signals? A Cross-Dataset Transfer Audit with a Unified Need Ontology 

**Authors**: Wu Hangyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08989)  

**Abstract**: Public infant cry corpora are small, label-incompatible, and almost always evaluated one corpus at a time. We ask what this practice hides and what fixes it. Across four cry corpora screened by a multi-level leakage audit (byte-level and embedding-level deduplication plus a within-corpus train-test near-duplicate audit), we probe four frozen encoders and a handcrafted baseline under a unified five-class need ontology and shared task formulations. The audit exposes what single-corpus evaluation conceals: within-domain macro-F1 swings by 0.57-0.80 for the same encoder, cross-corpus transfer is negative on average (negative-transfer ratio 0.19-0.35, significant in 18 of 30 directed cells, BH-FDR), and 349 content-identical clip groups carry conflicting metadata labels across corpus distributions. The same audit, however, reveals a consistent way forward. Transfer into the noisiest corpus is consistently positive in effect size at matched training size and after near-duplicate removal, offering a practical recipe for small, noisy corpora. Frozen probes saturate at modest label budgets, while stabilized fine-tuning wins with full labels; domain-adaptive pretraining significantly beats stabilized fine-tuning at 5-10-shot (the 1-shot advantage is not robust to optimization-seed variance) but shows no significant advantage at 50-shot or beyond. In the tested binary, shared-label settings, ontology-mapped joint training wins in all four encoder-by-target combinations, whereas naively merging unmapped labels costs up to 37 F1 points. We release the ontology, mapping code, and audit pipeline, turning incompatible cry corpora into a usable joint-training resource. 

---
# Detecting Clear Contact Lenses for Iris Recognition: A Two-Stage Mask-Guided Attention Approach 

**Authors**: Parisa Farmanifard, Arun Ross  

**Link**: [PDF](https://arxiv.org/pdf/2608.08977)  

**Abstract**: This work focuses on the impact and detection of clear contact lenses in the context of iris recognition. While the detection of cosmetic or patterned contact lenses has been extensively studied under the presentation attack detection (PAD) paradigm, clear prescription contact lenses, that are typically transparent, have received comparatively less attention despite their widespread use. Unlike patterned lenses, clear lenses introduce no salient texture artifact, making them difficult to detect and are often assumed to have no impact on iris recognition. We first examine this assumption using the commercial VeriEye matcher on four benchmark datasets and show that clear lenses marginally degrade genuine match scores and increase verification error. We then propose a two-stage contact-lens detection framework. Stage~1 uses an existing PAD model to identify patterned lenses, while Stage~2 focuses on the more challenging clear-lens versus no-lens distinction using a ConvNeXt-Base model equipped with Mask-Guided Spatial Attention (MGSA). The proposed MGSA module incorporates a Hough-derived anatomical ROI mask together with learned spatial attention and Squeeze-and-Excitation channel recalibration, allowing the network to focus on subtle limbal cues associated with clear lens wear. Across four datasets, the full pipeline consisting of both patterned and clear contact lens detection achieves between 90.0\%--98.8\% accuracy. Finally, we introduce a z-score calibration method that adjusts VeriEye match scores when a clear lens is detected in the input images. This calibration reduces EER by 4.1\%--28.3\% across datasets, demonstrating that reliable clear contact lens detection can directly improve iris verification performance. 

---
# How Can Rhetoric Reward-Hack AI Reviewers? Dissecting Rhetorical Sensitivity in AI-Based Peer Review 

**Authors**: Ming Li, Chenguang Wang, Xirui Li, Xinyue Zeng, Dianqi Li, Peng Shi, Dawei Zhou, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08975)  

**Abstract**: As large language models increasingly participate in scientific evaluation, we investigate a potential form of reward hacking: how rhetorical choices shape AI-review judgments when reported scientific content is preserved and how these effects vary across evaluation conditions. We construct a controlled corpus of 4,200 full-paper manuscripts derived from 120 anonymized ICLR 2026 submissions. Two LLM rewriters transform six rhetorical dimensions in opposing directions, and five LLM reviewers evaluate the resulting manuscripts under standard and strict protocols. We also test joint, recursive, and reviewer-guided rewriting. Our results show that rhetorical sensitivity is structured rather than uniform. Evidence framing and novelty stance produce the largest positive-negative contrasts in overall assessment, with scope framing forming a weaker second tier; the remaining dimensions have smaller or less stable effects. This hierarchy persists across human-assessed quality levels, but score movement depends strongly on the AI reviewer's original score: lower scores tend to rise, higher scores tend to fall, and directional contrasts are clearest in the middle ranges. More elaborate workflows do not reliably yield larger gains. Joint rewriting is strongly rewriter-dependent, reviewer guidance does not consistently outperform an unguided second pass, and repeated rewriting yields diminishing, configuration-dependent returns. Across conditions, the rewriter primarily determines the separation between opposing variants, whereas the reviewer determines the magnitude and sign of their score effects. Strict review lowers mean OA by 1.36 points without consistently changing rhetorical sensitivity. These findings identify when rhetorical presentation influences AI scientific review and motivate evaluation systems robust to content-preserving variation in scientific writing. 

---
# GALA: Graph-Augmented LLM Agents for Root Cause Analysis and Incident Response in Microservices 

**Authors**: Yifang Tian, Yaming Liu, Zichun Chong, Zihang Huang, Yiran Li, Hans-Arno Jacobsen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08968)  

**Abstract**: Microservice root cause analysis (RCA) requires correlating failures across heterogeneous telemetry within complex service dependency graphs. Existing methods often rely on a single telemetry modality; recent LLM-based approaches can suffer from unconstrained exploration and hallucination; and most systems stop at fault ranking without producing actionable incident response. We present GALA+, a graph-augmented LLM agentic framework centered on graph-guided investigation, which uses service dependencies to bound exploration and refine diagnosis through localized multi-modal evidence. For initial hypothesis generation, GALA+ combines complementary telemetry signals with STRIX, a novel trace- and graph-structure-aware scoring module. GALA+ then produces ranked diagnoses, incident summaries, and stratified action recommendations. We further introduce SURE-Score, a human-guided evaluation framework co-developed with industry SRE experts for assessing RCA-specific output quality beyond conventional text similarity metrics. On two microservice benchmarks, GALA+ consistently achieves the strongest overall results, surpassing the best LLM-based baseline by more than 25 percentage points in AC@1, while also receiving the highest ratings from both SURE-Score and independent human SRE evaluation. 

---
# Fourier Self-Supervision for Fine-Grained Generalized Category Discovery 

**Authors**: Sarah Rastegar, Mina Ghadimi Atigh, Pascal Mettes, Yuki M. Asano, Cees G. M. Snoek  

**Link**: [PDF](https://arxiv.org/pdf/2608.08963)  

**Abstract**: Generalized Category Discovery aims to recognize known categories while identifying novel ones within unlabeled data. Existing methods, typically based on self-supervision and contrastive learning, often struggle to capture fine-grained distinctions, relying on superficial visual cues rather than the intrinsic attributes humans use for categorization. We introduce Fourier Self-Supervision, that leverages the Fourier transform of images to enhance the discrimination of subtle differences and support the discovery of new categories. Our method employs a dual frequency filtering strategy: a low-pass filter first extracts broad, abstract attributes that capture high-level category information, while a high-pass filter emphasizes fine details such as edges and textures that are essential for fine-grained recognition. Each operates on a dedicated latent space, and their overlapping representations together yield a richer, more complete feature space. This dual-frequency approach not only refines feature extraction to identify novel categories, but also strengthens the model's discriminative power in fine-grained category discovery. Experiments on multiple fine-grained datasets show that incorporating Fourier Self-Supervision outperforms state-of-the-art methods, even when the number of classes is unknown, demonstrating its effectiveness for Generalized Category Discovery. Our code is available at: this https URL. 

---
# Idea Search: Guiding Tree Search with Ideas to Explore Diverse Scientific Methods 

**Authors**: Xuefei Julie Wang, Hao Cui, Michael P. Brenner, Subhashini Venugopalan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08958)  

**Abstract**: Tree Search-based test-time scaling of LLMs is a powerful tool for automated scientific coding. However, pure Tree Search sometimes struggles with systematic exploration, becoming trapped in local optima, or unproductive loops, especially in the vast search space of scientific methods. To address this limitation, we propose Idea Search, a framework that systematically integrates a dynamic "Idea Bank" into Tree Search. Idea Search involves three steps: (1) decomposing existing methods into atomic ideas, (2) sampling from this bank of ideas to guide branches of code mutations, and (3) dynamically updating the bank with new ideas discovered through execution. On single-cell RNA-sequencing (scRNA-seq) batch integration, Idea Search reliably breaks the plateau of a strong pure Tree Search baseline, improving the mean score from 0.678 to 0.697 and reaching a best score of 0.728. We then characterize which design choices drive these gains: bank augmentation helps bandit sampling but not random sampling, "Exploratory" prompting that prioritizes new ideas surfaces the rare best-performing solutions, while increasing sampling-level exploration is counterproductive. 

---
# Do AI Forecast Ensembles Sample the Correct Conditional Distribution? 

**Authors**: Lucas J. Howard, Elizabeth A. Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2608.08954)  

**Abstract**: Ensemble forecasting aims to sample the conditional distribution of outcomes; whether AI forecast ensembles do this correctly in a joint sense remains largely untested. We train a diffusion model for probabilistic subseasonal coastal sea level forecasts at eight US East Coast tide gauge stations, with sea level derived from reanalysis, and find that marginal and joint forecast quality decouple: positive skill at every station and lead time marginally, while joint spatial structure is worse than climatological draws. A shuffle-based permutation decomposition reveals this failure is invisible to the energy score but detected by the variogram score. Lorenz-96 experiments across 0.7-170 equivalent years show the gap persists regardless of training volume and is reproduced by a linear baseline, indicating structural inadequacy of the learned distribution. A dynamical ensemble does not replicate the failure while a deterministic emulator does, suggesting it is specific to learned emulators rather than ensemble forecasting generally. 

---
# From Operational Design Domain to Action: A Systematic Behavioral Taxonomy for Autonomous Driving 

**Authors**: Chaitanya Shinde, Hadi Hajieghrary, Miguel Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2608.08941)  

**Abstract**: Operational Design Domain (ODD) specifications describe where an automated driving system (ADS) is permitted to operate, but they do not prescribe what the ADS must demonstrably do once deployed within that domain. This gap between operating condition specification and behavioral validation represents a critical unresolved challenge in ADS safety assurance. This paper presents a structured, standards-grounded taxonomy of 21 behavioral competencies organized across three operational domains-Highway (HWY), Urban (URB), and Hub (HUB)-derived systematically from the PEGASUS six-layer model-based ODD. Each behavior is decomposed along longitudinal and lateral control axes and characterized against a four-property framework: Safety (gap maintenance, conflict avoidance, kinematic stability), Compliance (legal rules and behavioral norms), Comfort (rider dynamics and trust), and Efficiency (mission completion and product-level metrics). We further demonstrate that the crossing of ODD layer parameterizations with behavioral competency specifications yields concrete scenario families suitable for systematic behavioral testing and SOTIF coverage evidence. The taxonomy is grounded in AVSC00008202111, SAE J3237, and SAE J3016, and is validated as an operational specification layer through its deployment in a rule-enforced trajectory optimization system. The Hub domain is identified as a structurally distinct, underspecified domain warranting dedicated research attention. 

---
# Toward CT-Equivalent Image Quality in Low-Dose Radiotherapy Planning: Conditional Diffusion-Based CBCT-to-CT Synthesis and the Impact of CBCT Input Representation 

**Authors**: Alzahra Altalib, Chunhui Li, Christopher Hamill Taylor, Sankar Pillai, Alessandro Perelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.08919)  

**Abstract**: During standard radiotherapy planning, repeated CT acquisitions are often required for patient registration, verification, and adaptive planning, resulting in increased cumulative X-ray dose. To mitigate this, low-dose cone-beam CT (CBCT) is routinely acquired during treatment delivery. However, CBCT image quality remains insufficient for accurate dose calculation and adaptive radiotherapy planning due to increased scatter, noise, beam hardening, and reconstruction related artifacts. This study develops a supervised deep learning based CBCT to CT synthesis framework using a conditional denoising diffusion probabilistic model (DDPM), where the generation of a CT-based planning for accurate positioning and dose calculation is obtained using generative models with low dose CBCT imaging. Beyond demonstrating CBCT to CT synthesis, the primary objective is to investigate how the representation of CBCT input data, either standard clinical DICOM CBCT images or filtered back-projection (FDK) reconstructions from raw projection data, affects the performance of diffusion based CT synthesis. The overarching aim is to assess whether physics aware CBCT representations better support CT-equivalent image quality while maintaining reduced imaging dose in radiotherapy workflows. 

---
# ToolVision: Learning When and How to Use Visual Tools with Capability-Aligned Supervision 

**Authors**: Delin Mao, Chenghao Sun, Jingwei Song, Chishui Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08907)  

**Abstract**: Thinking with images allows a multimodal model to compensate for limited perception by invoking visual tools through code. Yet the prevailing SFT-then-RL recipe creates a different supervision misalignment at each stage. SFT is expected to teach how to use tools, but trajectories from stronger teachers may succeed through perceptual capabilities that a smaller student cannot reliably reproduce or exploit, causing the student to imitate tool-call patterns without learning how to make them useful. RL is expected to teach when to use tools, but outcome-only rewards make fallible tool execution a liability and suppress tool use, whereas a blanket bonus for every correct tool-using trajectory encourages valid but ineffective operations. To address these two misalignments, we introduce ToolVision. During SFT, a multi-agent pipeline explores candidate trajectories, and a committee including student-scale models scores stepwise evidence gain to rank and prune the search branches. Only successfully executed trajectories with correct final answers are retained for SFT. Before RL, ToolVision compares the learner's performance with and without tools, then rewards successful tool use only on questions where tools provide a clear benefit. Both signals are constructed automatically from public task data without additional human annotations of tool use or necessity. ToolVision-8B improves over its base on all seven main benchmarks, surpasses Thyme-7B, CodeVision-8B, and CodeDance-7B on all three high-resolution benchmarks, and outperforms Qwen3-VL-32B-Thinking on V* and HRBench 8K. We will publicly release the datasets and source code. 

---
# From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability 

**Authors**: Alexander Hackett, Arnaud Denis-Remillard, Axel Cassou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08904)  

**Abstract**: How much of a vision-language model's (VLM) spatial understanding remains after the action post-training process of building a vision-language-action model (VLA)? We probe depth perception, a primitive of spatiogeometric understanding, from every decoder layer of a weight-matched open-source base VLM/VLA pair: Molmo2-ER and MolmoAct2-LIBERO. First, the VLA decodes depth worse at every layer, a persistent gap we call the floor. Second, the degradation is not uniform: while the base VLM's depth decodability improves through its final layers, the VLA's collapses, an additional late-layer drop we call the cliff. We causally localize the cliff to late-layer MLP interference: ablating the late-layer MLP writes recovers the majority of the terminal decodability cliff, while matched attention ablations and the same intervention in the weight-matched base VLM produce no comparable recovery. A module-level decomposition explains this dissociation: the base VLM carries depth most accessibly in accumulated MLP writes, whereas action post-training collapses depth decodability in the late accumulated writes. 

---
# A New Approach to Characterising Optimisation Problems Using Programmatic Representation and Complexity Measures 

**Authors**: Marcus Gallagher, Katherine M. Malan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08898)  

**Abstract**: Characterising optimisation problem instances is a fundamental part of understanding the behaviour and performance of different algorithms as well as providing information for algorithm selection and configuration. In this paper we propose a novel approach to problem characterisation based on the representation of instances when implemented as a program. The intuition is that the complexity of the code required to express an objective function should relate to the complexity of the search landscape. We identify the Halstead volume as a measure of code complexity, which can be seen as a simplified version of the entropy of the program. Given a code implementation of the objective function, the Halstead volume and entropy can be quickly calculated using existing libraries. We apply the proposed complexity measures to the well-known BBOB optimisation problem suite and the simple feed-forward neural network training task. We also show that the measures are negatively correlated with algorithm performance and therefore show potential as predictive meta-features for algorithm selection and other problem analysis. We envisage the proposed measures as complementary to other problem characterisation approaches, but with the advantages of not requiring any sampling of the search space, being invariant to transformations, and being very quick to calculate automatically. 

---
# Epistemic Transfer in AI-Assisted Verification: A Framework and Evaluation Protocol 

**Authors**: Christoph Trattner  

**Link**: [PDF](https://arxiv.org/pdf/2608.08882)  

**Abstract**: AI tools that help people judge online claims are usually evaluated while the tool is present. This paper asks a different question: after using such a tool, what can the user still do on their own? I call this epistemic transfer. It refers to the effect of prior AI-assisted verification on later unassisted performance on new claims. In this paper, I make three contributions. First, I distinguish epistemic transfer from nearby outcomes such as correction effects, trust, reliance, and human--AI team performance. Second, I introduce two simple quantities for studying it: the Epistemic Transfer Effect (ETE), which compares delayed unassisted performance across conditions, and Tool-Removal Cost (TRC), which measures the immediate drop in performance when the tool is taken away. Third, I turn these ideas into a practical evaluation protocol that can be used in online experiments or field studies. The protocol combines answer-first and evidence-first AI conditions with active-practice and no-practice controls, delayed tests on held-out claims, behavioral measures, and participant- and item-level analyses. Putting ETE and TRC together yields a diagnostic space that separates capability building, capability plus tool advantage, epistemic inertness or de-skilling, and verification on loan. The point is not that every AI tool must teach. The point is that when independent judgment matters, we should test not only whether a tool helps now, but also what it leaves behind. 

---
# DistillCache: KL-Guided Adaptive KV-Cache Eviction for Memory-Efficient LLM Inference 

**Authors**: Asaad Althoubi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08878)  

**Abstract**: Transformer-based large language models (LLMs) achieve strong performance across many tasks, but their Key-Value (KV) cache grows linearly with sequence length, creating a severe memory bottleneck for long-context inference. Existing heuristic eviction methods (e.g., H$_2$O and SnapKV) rely on static attention or positional signals that often fail to capture a token's future predictive influence. We propose DistillCache, a reinforcement learning framework that formulates KV-cache eviction as a sequential decision problem. DistillCache learns a lightweight policy network using rich internal model signals (attention statistics, value norms, entropy, and position) and trains it with REINFORCE via a per-step KL-divergence reward to preserve the full-cache output distribution. On a 7B-parameter instruction-tuned Transformer (Mistral-7B-Instruct-v0.3), DistillCache retains 94.2% of full-cache accuracy on LongBench at a 25% cache budget, outperforming both strong heuristic baselines (H$_2$O, SnapKV) by up to 2.7 absolute points and, under our re-implementations, concurrent RL-based methods (ForesightKV, RLKV) by up to 1.4 points on long-context tasks. On reasoning benchmarks, DistillCache is competitive with the best concurrent method and surpasses it under aggressive compression. It also delivers up to 2.1x full-cache throughput while maintaining competitive practical efficiency. These results highlight the effectiveness of learned, distribution-aware policies for memory-efficient long-context LLM inference. 

---
# Agentic Anomaly Detection with ORCA-Style Dynamic Inductive Bias Adaptation in Multimodal Wearable Time Series Data 

**Authors**: Anushka Roy, Jyotirmoy Singh, Shreea Bose, Chittaranjan Hota  

**Link**: [PDF](https://arxiv.org/pdf/2608.08859)  

**Abstract**: Wireless Body Area Networks (WBANs) generate multivariate physiological time series that are highly nonstationary and must often be processed under strict computational and memory constraints. A critical yet underexplored challenge in this setting is selecting an appropriate temporal receptive field, which serves as a strong inductive bias for anomaly detection models. Existing approaches typically rely on fixed temporal contexts, which can perform inconsistently across heterogeneous signal regimes and require dataset-specific tuning. We propose ORCA, an agentically controlled anomaly detection framework that dynamically adapts the temporal receptive field at inference time based on lightweight signal statistics. Rather than introducing additional trainable parameters or learned policies, ORCA employs a supervisory controller that autonomously selects among discrete temporal contexts, enabling state-dependent inductive bias adaptation without retraining. Across a custom WBAN dataset, ORCA achieves performance comparable to the strongest fixed-context baselines (AUROC = 0.99) while eliminating the need to tune temporal horizons in advance. We further evaluate ORCA on MIMIC-IV as a challenging out-of-distribution benchmark, observing conservative generalization behavior without performance collapse under heterogeneous clinical conditions. These results highlight adaptive temporal inductive bias control as a practical and robust design principle for anomaly detection in resource-constrained, nonstationary physiological time series. 

---
# Deployable Per-Instance Multi-Layer Activation Steering for Large Language Models 

**Authors**: Muhammad Faishal Adly Nelwan, Alfan Farizki Wicaksono  

**Link**: [PDF](https://arxiv.org/pdf/2608.08829)  

**Abstract**: Activation steering edits the behaviour of a frozen language model by adding a learned vector to its residual stream, and current practice fixes the injection layers globally per task. We argue that the best layers are an instance-level decision, and we make per-instance, multi-layer selection both well understood and deployable. On two open-weight 8B models and six binary persona traits, a per-instance oracle over layer subsets shows that the best layers vary from one input to the next: on most trait-model pairs, no fixed global layer set recovers the per-instance benefit. A greedy rule that ranks layers by single-layer marginal effect recovers nearly all of the oracle's benefit, but both must score candidate layers against the gold answer, so neither can run at deployment; the rule instead becomes the target a prompt-only predictor is trained to reproduce. Our deployable recipe needs no label at inference: a per-instance layer ranker read off the prompt embedding, a classifier that infers the steering direction, and an adaptive gate that scores short steered passes against that inferred direction and steers no more layers than necessary. The recipe recovers most of the oracle's lift (the bulk on the stronger model, a clear majority on the harder one), never drives any trait-model pair below its unsteered alignment baseline on average, and largely avoids the fluency collapse that strong global selection incurs at higher layer counts. A mechanistic account, "direction over magnitude", explains the behavioural flip under a mis-directed global set, the output collapse from steering too many layers, and the ceiling of unsteerable inputs. 

---
# Hybrid Neural-Classical Correction for Frozen Time Series Foundation Models: A Comprehensive Ablation Study on High-Frequency Stock Prediction 

**Authors**: Kasun Dewage, Suranadi De Silva, Shankhadeep Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2608.08825)  

**Abstract**: Foundation models for time series forecasting demonstrate impressive zero-shot generalization but often underperform on specialized domains such as high-frequency finance. We present a comprehensive study of hybrid neural-classical correction for adapting frozen TimesFM (200M parameters) to stock return prediction during the volatile opening trading hour. We compare two neural correction architectures - AttnCorrect (multi-head self-attention, approximately 471K parameters) and GatedLinear (low-rank bilinear projection with gating, approximately 49K parameters) - each augmented with Random Forest residual learning. Through systematic ablation across 10 major technology stocks (NVDA, MSFT, AAPL, GOOG, GOOGL, AMZN, META, AVGO, TSLA, NFLX) spanning 2 million data points, we reveal critical insights: (1) The hybrid neural-classical approach achieves 0.597 pooled correlation and 6.4x mean per-day correlation improvement over frozen TimesFM; (2) Classical residual learning (Random Forest) provides the largest single-component contribution, matching or exceeding the neural correction component; (3) Simpler neural architectures surprisingly outperform complex ones when classical residual learning is removed; (4) Self-attention provides the largest neural-only contribution. GatedLinear+RF achieves best overall performance with 9x fewer neural parameters than AttnCorrect+RF. We report three complementary correlation metrics - mean per-day, cross-day cumulative, and pooled - to provide a complete picture of predictive quality. Our results provide practical guidance: effective foundation model adaptation requires careful integration of neural and classical components, with classical methods playing a crucial complementary role. 

---
# 360CityArena: A Realistic Virtual Urban Navigation Benchmark for Embodied Agents 

**Authors**: Kenta Watanabe, Atsuyuki Miyai, Mizuki Takenawa, Kiyoharu Aizawa, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.08814)  

**Abstract**: We present 360CityArena, a benchmark for evaluating the urban exploration capabilities of embodied agents within a photorealistic environment constructed from 360-degree videos. Existing outdoor benchmarks either lack sufficient photorealism or complexity, resulting in a considerable gap from real-world urban environments. 360CityArena is built on a realistic reconstruction of the Akihabara district in Tokyo, Japan, using 602 360-degree video segments covering 85 streets, and consists of 175 meticulously human-crafted tasks. It encompasses three task categories: Environment Understanding, Path Reasoning, and Spatial Reasoning, covering fundamental abilities required for urban exploration, such as localization, landmark search, path planning, and relational spatial reasoning, thereby enabling comprehensive evaluation in realistic urban scenes. Our evaluation using state-of-the-art LMM-based agents shows that even the strongest model, Gemini 2.5 Flash, performs far below human level (human: 77.3% vs. Gemini 2.5 Flash: 17.1%), revealing substantial challenges that remain in city-scale embodied navigation and reasoning. 360CityArena provides a necessary and challenging testbed for photorealistic urban-district navigation and spatial reasoning. 

---
# Learning from Consensus and Disagreement: Unsupervised On-Policy Self-Distillation with Minority-Trajectory Contrast 

**Authors**: Jiaxin Guo, Yanwei Yue, Xuanbo Fan, Chunyu Yang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08764)  

**Abstract**: On-policy self-distillation improves language-model reasoning by querying a teacher on states actually visited by the student. Recent methods create a powerful information asymmetry by exposing the teacher to privileged context, yet they fundamentally rely on external supervision---such as gold solutions or verifiers---to construct this advantage. We introduce CoDA (Consensus and Disagreement Alignment), a fully unsupervised framework that creates reliable privileged information entirely from the latent uncertainty structure of a model's own unlabeled rollouts. CoDA extracts two complementary signals. In the positive branch, answer-level consensus identifies a stable reasoning mode, which conditions a frozen self-teacher to provide dense distributional guidance on fresh student trajectories. However, because agreement does not guarantee correctness, positive-only distillation risks amplifying correlated errors into a false consensus. To break this harmful feedback loop, CoDA incorporates a negative branch that exploits disagreement: minority trajectories are treated as unstable alternatives and gently penalized via a reference-anchored, KTO-style calibration objective. This unpaired binary feedback provides robust regularization without requiring the strong assumption that the consensus is the absolute ground truth. Empirical evaluations on competition-level mathematical benchmarks demonstrate that CoDA significantly improves reasoning, outperforming self-generated baselines and effectively stabilizing training against erroneous consensus. 

---
# Eco-SoC: A Sustainable VLSI Architecture for Energy-Proportional Artificial Intelligence 

**Authors**: Jatin Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2608.08761)  

**Abstract**: In an era defined by escalating climate change and the pervasive deployment of edge intelligence, the environmental cost of semiconductor manufacturing and operation has reached a critical threshold. As Deep Learning (DL) accelerators dominate System-on-Chip (SoC) die area, achieving true sustainability requires a paradigm shift from static worst-case efficiency to dynamic energy-proportionality. This paper introduces Eco-SoC, a highly scalable VLSI architecture co-designed specifically for sustainable artificial intelligence. We propose a hardware-level Dynamic Precision-Scaling Logic (DPSL) framework that adaptively modulates bit-width precision based on real-time activation sparsity, successfully reducing switching activity by up to 42% on a commercial 7nm FinFET process node.
Furthermore, we transcend traditional Power-Performance-Area (PPA) metrics by providing a comprehensive Life Cycle Assessment (LCA) using the Architectural Carbon footprint Tool (ACT). Our synthesis demonstrates that Eco-SoC offsets its increased embodied carbon footprint (a marginal 4.8% area overhead) within 1.1 years of edge deployment. Finally, by introducing a thermal-aware power gating mechanism that mitigates localized hotspots, Eco-SoC doubles the projected Mean Time To Failure (MTTF) of the silicon, providing a tangible, scalable strategy for electronic waste (e-waste) mitigation in next-generation computing systems. 

---
# Can We Optimize the Performance-Carbon Emission Break-Even Point?: The Quest for Greener LLMs 

**Authors**: Sourav Das, Tanmay Joshi, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.08744)  

**Abstract**: The carbon footprint of any deployed Large Language Model (LLM) accumulates during inference, where repeated use of the model substantially exceeds the one-time cost of fine-tuning. Yet most efficiency interventions target either pre-training scale or post-hoc compression. We ask whether folding a calibrated, differentiable energy surrogate into the fine-tuning objective can produce inference behavior that gains task accuracy at zero or near-zero carbon cost, a break-even configuration. We propose a joint loss mechanism with a per-model carbon-emission parameter, a linear surrogate over parameter norm, FLOP proxy, and a memory proxy, fit from on-hardware energy profiling. We fine-tune three architecturally distinct families: Gemma-2 2B, Llama-3.1 8B, and Qwen-2.5 14B, and evaluate inference F1 and CO$_2$ emissions on three MMLU subjects: abstract algebra, philosophy, and formal logic. We discover from several outcomes that the carbon term behaves as either harmful interference or beneficial regularization depending on the task structure. We position calibrated carbon-aware fine-tuning as a lightweight, drop-in regularizer with a non-empty but model and task-dependent break-even region. This is an ongoing work, and we will release our codebase soon. 

---
# TomaMMU: A Comprehensive Multimodal Understanding Benchmark for Tomato Leaf Diseases 

**Authors**: Gia-Han Truong, Khang Nguyen Quoc, Luyl-Da Quach  

**Link**: [PDF](https://arxiv.org/pdf/2608.08727)  

**Abstract**: To address this gap, we introduce TomaMMU, a large-scale Tomato leaf disease MultiModal Understanding dataset, alongside TomaBench, a benchmark for evaluating VLMs on tomato disease understanding. TomaMMU comprises 28,808 high-quality images spanning 15 categories and 213,119 human-annotated visual question-answer pairs, generated through a three-stage pipeline comprising Data Collection, Human Annotation, and Question-Answer Generation. Building on this foundation, TomaBench organizes seven agricultural tasks into a hierarchical three-level taxonomy spanning Basic Perception, Pathology Understanding, and Expert Diagnosis, which together enable systematic evaluation from low-level visual recognition to high-level diagnostic reasoning. The tasks assess visual symptom recognition, taxonomic relationships, and diagnostic reasoning, offering a comprehensive view of how well models grasp plant pathology. Our results pronounced gaps in fine-grained recognition and factually grounded reasoning with 14 state-of-the-art VLMs, consistently underperforming on both challenging MCQs and open-ended questions. These results suggest that current VLMs struggle to translate visual perception into reliable diagnostic knowledge, motivating the need for targeted domain adaptation. Simple fine-tuning on TomaMMU substantially narrows this gap, boosting accuracy on challenging MCQs to 96.09%, outperforming recent VLMs, and pointing toward promising directions for future work. All data and code is available in this https URL. 

---
# PAST: Privileged Adaptation from Complete Student Trajectories for On-Policy Self-Distillation 

**Authors**: Yangyang Feng, Zhuoyan Feng, Junlan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.08726)  

**Abstract**: On-policy self-distillation (OPSD) uses a privileged teacher to supervise a reasoning model on prefixes sampled from its own rollouts. Yet each rollout also reveals how the student's response unfolds and whether it succeeds, student-specific hindsight that standard OPSD does not use to form the teacher. We introduce Privileged Adaptation from Student Trajectories (PAST), which treats each completed student trajectory as additional privileged information for the OPSD teacher while leaving the student's distillation prefixes unchanged. PAST preserves the student's next-token distribution on correct trajectories and uses failed trajectories to adapt the teacher toward verified success under student-proximity regularization. We characterize what such a trajectory-conditioned teacher can transfer to a prefix-only student. Forward-KL distillation projects the teacher distributions to their conditional arithmetic mean given the prefix. This projection separates trajectory-specific variation that remains privileged from the mean policy shift available to the student. For correct trajectories, the unclipped population objective also has the frozen student as an ideal distributional fixed point. Across three mathematical reasoning benchmarks, PAST improves the Avg@12 macro average over Vanilla OPSD by 5.6 percentage points. A $2\times2$ factorial study shows gains from both complete-trajectory access and teacher adaptation, while trajectory removal and shuffling confirm that the adapted teacher uses the matching hindsight context. 

---
# Gaming Without an Attacker: Benchmark Fingerprinting in LLM-Driven Search Under Selection Pressure 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2608.08722)  

**Abstract**: Benchmarks for systems that are optimized against the evaluation signal measure something different from what they claim. We document this concretely in two GPU-kernel-optimization suites with held-out generalization gates: Metal-Sci (10 scientific-compute tasks) and Metal-ZK (12 zero-knowledge/cryptographic tasks), in which three frontier LLMs (Opus 4.7, Gemini 3.1 Pro, GPT-5.5) propose Metal kernels inside a $(1{+}1)$ evolutionary loop with rich feedback. Although no model is prompted to act adversarially, the promoted winners repeatedly fingerprint the evaluation configuration: they branch on the identity of runtime parameters, tune the measured branch maximally, and leave the unmeasured branch slow or silently wrong. Across the pooled suites, $16/53$ ($30\%$) of in-distribution wins fail to transfer to held-out configurations. We give a four-mode taxonomy of these failures, from configuration fingerprints to gate leakage. We distill design guidance for measurement under strategic optimization: held-out probes retain validity only on non-enumerable axes; gates must measure held-out performance, not just correctness; and a transfer rate is interpretable only with per-failure mechanism grades: ours decomposes into gamed, overfit, and benign.
Code and research artifacts: this https URL 

---
# LibraSpec: Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization 

**Authors**: Zexun Lin, Yuan Feng, Junlin Lv, Kevin S. Zhou, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.08721)  

**Abstract**: Speculative decoding accelerates large language model inference by drafting multiple tokens for parallel verification, with efficiency critically determined by the speculative length selected at each decoding round. Existing dynamic speculation methods select the speculation length by estimating how many tokens will be accepted, which is reasonable for autoregressive drafters that generates tokens sequentially. The recent wave of diffusion-based drafters, however, generates candidate blocks in parallel at substantially lower drafting cost, shifting the key question from how many tokens to generate to how many generated tokens are worth verifying. We therefore reformulate dynamic speculative-length selection as expected-speedup optimization and derive a marginal criterion that extends the speculative sequence only when its acceptance gain outweighs the additional verification cost. Building on this criterion, we develop \textit{LibraSpec}, a training-free and plug-and-play algorithm that iteratively determines the speculative length using drafter confidence scores. Theoretically, we prove that LibraSpec monotonically converges toward the optimal speculative length. Experiments across six target models, three diffusion-based speculative decoding methods, and math, coding, and chat benchmarks show consistent improvements under both greedy and sampling settings, achieving a further $0.5\sim1.5\times$ improvement over baselines and up to $8.49\times$ speedup over autoregressive decoding. 

---
# Resolution Meets Reduction: Efficient Visual Context for 3D Radiology Report Generation 

**Authors**: Jonathan Suprijadi, Raphael Stock, Moritz Langenberg, David Zimmerer, Kim-Celine Kahl, Stefan Denner, Yannick Kirchhoff, Karol Gotkowski, Maximilian Rokuss, Jeremias Traub, Tassilo Wald, Constantin Ulrich, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2608.08713)  

**Abstract**: Vision-language models offer a promising path toward automating radiology report generation, but applying them to full 3D CT volumes poses substantial computational challenges. Modern foundation vision encoders (VEs) can produce tens of thousands of vision tokens per scan, making the visual sequence passed to the large language model (LLM) a primary computational bottleneck. Vision-to-language projectors can compress this sequence to reduce computation, but may discard clinically relevant detail; conversely, effective compression can accommodate higher-resolution inputs while keeping the downstream token count fixed. How this vision-token budget should be allocated across input field of view, spatial resolution, and vision-to-language projection therefore remains an open design question. We systematically evaluate four heterogeneous VEs (CNN- and ViT-based), five token-reducing projectors at up to 64x compression alongside a non-reducing MLP projector baseline, and five instruction-tuned LLMs (1.7B--4B) on two large-scale CT report datasets (CT-RATE and Merlin). At matched LLM token budgets, anatomy-guided region of interest cropping is the most consistent strategy, improving clinical macro F1 in 19 of 20 settings by +3.7 points on average for the 3D ViT Primus encoder and +1.1 for the slice-based 2D ViT Curia encoder. Increasing input resolution further is strongly projector-dependent: the PerceiverResampler, paired with higher-resolution Curia features, yields the strongest configuration in the resolution study on both datasets. Our best configurations achieve state-of-the-art clinical macro F1 on the test sets, reaching 49.5 on CT-RATE and 49.0 on Merlin. Code and models will be published upon publication. 

---
# RippleKV: Cross-Layer KV Cache Allocation via Perturbation Propagation 

**Authors**: Dongjie Xu, Kai Qian, Julius, Weijie Shi, Yuxuan Sun, Minghua Tang, Fenglei Jin, Hanchi Dong, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08684)  

**Abstract**: Long-context LLM inference is bottlenecked by KV cache memory, yet distributing a limited cache budget across layers remains challenging. Existing methods rely on proxies such as layer depth, attention statistics, or representation change. These proxies do not measure how perturbations at each layer propagate to the output and may therefore cause sensitive layers to be underallocated while tolerant layers are overallocated. To address this issue, we propose RippleKV, which allocates cache across layers by estimating how perturbations to each layer's value cache affect the final predictive distribution. RippleKV independently injects norm-adaptive perturbations into each layer's value cache and measures the induced KL divergence at the model output over a small calibration set. Averaging these responses yields a sensitivity profile specific to the model that need not vary monotonically with depth. RippleKV then converts the sensitivity profile into layer budget multipliers by normalizing the sensitivity scores and applying an exponential mapping. A ratio parameter controls the allocation disparity between sensitive and tolerant layers, while a final normalization preserves the KV cache budget. Experiments on LongBench demonstrate that RippleKV achieves the highest average performance among the evaluated KV cache compression methods under matched cache budgets. 

---
# UniSpace: Unified Visual Representation and Scalable Multimodal Modeling 

**Authors**: Jinbo Yan, Limeng Qiao, Jie Qin, Junyan He, Feize Wu, Guanglu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08676)  

**Abstract**: Semantic vision encoders have become a central visual interface for multimodal understanding and semantic conditioning in image generation. However, their final tokens discard fine-grained visual details, leading to poor pixel reconstruction and limiting their use in reconstruction-sensitive tasks such as image generation and editing. In this work, we ask whether understanding, generation, and editing can be modeled in a single visual representation space built from a pretrained semantic ViT. We show that the frozen Transformer blocks of a semantic ViT are not intrinsically unable to preserve visual details. Instead, the original patch parameterization drives the representation toward semantic abstraction, making fine-grained information difficult to recover from the final tokens. Based on this observation, we introduce \emph{Patch Reparameterization}, which preserves the original semantic pathway while adding a reconstruction-aware patch embedding that provides fine-grained visual information to the same frozen ViT blocks. The resulting unified representation preserves multimodal understanding while enabling high-fidelity image reconstruction and a favorable reconstruction--generation trade-off. We further scale this representation into \emph{UniSpace}, an 8B Mixture-of-Transformer-Experts model that performs understanding, generation, and editing in the same visual space without a separate VAE pathway. System-level evaluations demonstrate practical text-to-image generation and instruction-based image editing, showing that a reparameterized pretrained ViT can serve as a unified visual interface for scalable multimodal modeling. 

---
# LegoLM: Structured Weight Sharing for Large Language Models 

**Authors**: Joseph Bingham  

**Link**: [PDF](https://arxiv.org/pdf/2608.08652)  

**Abstract**: We present \LegoLM{}, a structured weight-sharing compression framework for large language models grounded in a systematic study of why global weight sharing fails and how to fix it. We identify two distinct failure modes. Distributional mismatch: for vector blocks of dimension d <= 2, transformer layers with heterogeneous weight scales impose a scale-mismatch penalty that grows linearly with d and cannot be resolved by increasing K, producing perplexity in the this http URL dominance: for scalar blocks, a fraction ~1/K of weights lies beyond the outermost Lloyd-Max decision threshold and cannot be represented by any centroid; their misrepresentation accumulates across layers, causing catastrophic quality loss. \LegoLM{} resolves both failure modes via three data-free adaptations: 1 scalar-block encoding to eliminate the $d$-linear mismatch component, 2 percentile-selective replacement that identifies and preserves outlier weights verbatim, and 3 boundary-layer protection for the first and last transformer blocks. Across GPT-2 small (124M) and Mistral-7B, \LegoLM{} achieves +0.03% PPL degradation at 4.41X compression on Mistral-7B - outperforming PTQ-8bit in both quality and compression ratio - and -0.02% at 2.67X. Downstream evaluation on LAMBADA and HellaSwag confirms that \LegoLM{} at K=64, p=99% preserves accuracy within noise at 5.12 X compression, exceeding PTQ-8bit's compression ratio while matching its accuracy. We further discover that outlier dominance grows with model scale: full replacement at K=128 degrades GPT-2 small by only +23% but catastrophically degrades Mistral-7B by +1,134,279%, while selective replacement at p=99% rescues both models to under +15%. A controlled ablation confirms that selective replacement is the dominant mechanism: adding it to per-layer K-means also yields near-lossless quality, matching \LegoLM{} within 0.02%. 

---
# CuteTTS: Efficient and High-Quality Speech Synthesis via Autoregressive Modeling of Continuous Latents 

**Authors**: Yuqian Zhang, Yao Shi, Kexin Huang, Botian Jiang, Zhe Xu, Yiwei Zhao, Min Liang, Shuang Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08638)  

**Abstract**: Zero-shot text-to-speech (TTS) now supports interactive assistants, personalized media, and accessibility tools. All TTS systems require faithful linguistic rendering, consistent speaker identity, and low-latency response. Yet compact streaming systems must preserve sufficient acoustic detail in a predictable low-rate latent sequence, while iterative diffusion sampling and classifier-free guidance multiply inference cost at every autoregressive step. To strike a balance between high-fidelity synthesis and low-latency inference, we present CuteTTS, a compact continuous-autoregressive TTS system. It combines semantically aligned causal VAE latents with patch-level autoregression, explicit speaker conditioning, and a bidirectional flow-matching head. We further introduce guidance-step distillation, which absorbs classifier-free guidance and multiple solver steps into a single interval-conditioned student. Evaluations on LibriSpeech and Seed-TTS-Eval demonstrate competitive intelligibility and speaker similarity in zero-shot voice cloning, while distillation lowers first-audio latency by 23.3% and real-time factor by 40.8% relative to the base model with comparable objective and subjective quality. These results provide a practical path toward continuous-autoregressive TTS that reconciles high-fidelity generation with the latency demands of real-time interaction. 

---
# Enhancing Scientific Named Entity Recognition via Large Language Models: A Type-driven Multi-task Learning Approach 

**Authors**: Tong Bao, Yi Zhao, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08636)  

**Abstract**: Scientific named entity recognition (SciNER) plays a crucial role in information extraction and knowledge discovery from scientific texts. Recently, large language models (LLMs) have demonstrated the capacity to achieve competitive SciNER performance with minimal human effort. Existing research highlights the importance of incorporating candidate entity type information for accurate entity recognition and classification by LLMs. However, when too many candidate entity types are provided in the prompt, LLMs struggle to accurately recognize and label entities in scientific texts, where entity types are more complex than in general domains. To address this challenge, we propose TdSciNER, a type-driven approach that effectively leverages entity type information to enhance SciNER performance. In TdSciNER, we first design an entity type filter model to identify the most likely entity types present in a given sentence. Subsequently, we introduce an auxiliary multi-class entity typing task within a multi-task learning framework alongside SciNER to obtain richer contextual representations. Then, we develop a novel demonstration selection strategy based on sentence similarity and entity type diversity to activate the in-context learning capabilities of LLMs, thereby improving entity recognition accuracy across diverse scientific domains. Experiments on three datasets demonstrate that our method achieves performance comparable to fully supervised models. Further analysis validates that each entity type-driven component in TdSciNER contributes to the improvement of SciNER performance. This work provides valuable insights for future advancements in SciNER and broader information extraction tasks in scientific text mining. 

---
# RAG-Based Auto-Configuration for Industrial Fieldbus Devices 

**Authors**: Aadil Gani Ganie, Saad Ezzini, Naveed Farooz Marazi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08618)  

**Abstract**: Industrial device commissioning requires engineers to manually extract hundreds of protocol-specific parameters from heterogeneous PDF manuals and transcribe them into supervisory control systems, a time-intensive, error-prone workflow. This paper presents SysName, a production-oriented pipeline that automates device configuration end-to-end for Modbus RTU, OPC-UA, Profibus DP, and CANopen. It builds a hybrid dense-sparse retrieval index augmented by an ontology graph derived from ECLASS, AAS, and SOSA/SSN, using a BGE-M3 encoder with a cross-encoder reranker to surface relevant manual passages. A local LLM (T=0.1) generates ontology-aligned JSON-LD configurations via protocol-specific prompts and a four-step repair pipeline. A two-stage abstention gate, combining a reranker-score threshold and an IRI resolution ratio, blocks unsafe LLM invocations and filters low-coverage configurations before SHACL validation. On a gold set of 28 field-level queries, the hybrid retriever reaches 0.96 HitRate@10, and the reranker raises MRR@10 from 0.56 to 0.63 with perfect score separation for abstention. The generator attains field-level F1=0.87 with exact match on 9 of 12 runs. End-to-end runs on an H100 GPU complete in 2.6-6.6s per device with zero unsafe writes and zero silent failures on a five-device benchmark; every unsuccessful run is flagged by abstention or deployment verification. Component-wise evaluation localises the single systematic failure to OPC-UA generation, invisible to end-to-end metrics alone. A case study commissions a physics-simulated Universal Robots UR5e robot from unmodified vendor documentation (254-page manual, 8-page register list, 496 chunks), reaching field-level F1=1.0 over three runs with read-back and joint-consistency verification. An ablation study and comparison with five industrial-LLM systems complete the analysis. 

---
# REVEAL: A Rubric-Guided Agent for Explicit Evidence Sufficiency Verificationin Long-Video Question Answering 

**Authors**: Caijun Yan, Yang Zhou, Meixing Shi, Haoran Sun, Yichen Li, Yuxiang Cai, Yankai Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08612)  

**Abstract**: Recently, retrieval-augmented and memory-augmented methods have emerged as two promising paradigms for long-video question answering. However, existing methods typically rely on rigid, fixed-length temporal chunking (e.g., 10s) and static offline memory banks, which not only fragment coherent continuous events but also fail to adapt during real-time reasoning. Moreover, whether using multi-scale summaries or multimodal knowledge graphs, current approaches prioritize retrieval relevance while overlooking evidence sufficiency, often stopping to answer once only semantically relevant clues are retrieved, even when key temporal, causal, or fine-grained action evidence is still missing. To tackle these challenges, we propose REVEAL, a rubric-guided agent framework. As a foundation, we introduce an adaptive visual-similarity-based preprocessing pipeline that groups visually coherent adjacent frames into natural event units to construct an offline-online video memory---capturing global video context offline while dynamically maintaining question-conditioned memory online. Built upon this structured memory, REVEAL uses an automatically constructed rubric library to explicitly verify whether retrieved evidence satisfies sufficiency criteria, pinpoints missing clues upon verification failure, and directs targeted re-retrieval for complementary information. Without any extra training, REVEAL consistently outperforms both closed-source and open-source state-of-the-art methods across extensive experiments. These results show that explicitly verifying evidence sufficiency, rather than stopping at semantic relevance, retrieves the decisive clues that prior methods miss and yields more reliable long-video reasoning. 

---
# Mitigating Gender Bias in English to Romanian Machine Translation 

**Authors**: Ioana Grigore, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08606)  

**Abstract**: Machine translation (MT) systems often fail to correctly translate gender, especially when converting from a gender-neutral language like English to a gendered target language such as Romanian. This bias results in translations that default to masculine forms or reinforce gender stereotypes. We propose a hybrid pipeline to mitigate this issue by combining large language model (LLM)-based gender classification with neural machine translation (NMT). Our system uses a fine-tuned LLM to detect the intended gender of target words in English sentences and insert inline gender hint tags. These tagged sentences are then passed to a Transformer model fine-tuned to generate morphologically correct Romanian translations. To support this, we introduce three novel datasets for gender disambiguation and translation. Our approach improves gender accuracy on the WinoMT and WinoGender benchmarks by over 40 percentage points compared to a baseline MT system. This is the first method to explicitly address and evaluate gender bias in English-Romanian MT using both LLM inference and tag-aware translation. 

---
# Population-Scalable Multi-Agent World Modeling 

**Authors**: Renjie Zhao, Yuxiang Wu, Mingyu Zhang, Jiaxin Li, Sisi Li, Yimin Sheng, Tianxi Tan, Zhenkai Zhang, Jianyi Zhu, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.08600)  

**Abstract**: World models have recently achieved impressive progress in visual prediction and interactive generation, but extending them to multi-agent environments introduces a fundamental scalability challenge. Existing methods generally assume a fixed number of agents during training and inference, which ties the model to a pre-determined agent population and limits inference-time scalability. Our key insight is that cross-view consistency should arise from a shared world state whose evolution does not assume a predefined number of agents, while agent-specific observations should be generated by querying this state through a unified rendering interface. Based on this insight, we propose Khora, a scalable multi-agent world model that supports inference-time expansion to arbitrary numbers of agents without retraining. Our framework decouples world-state evolution from visual rendering and introduces a population-agnostic rendering mechanism for incorporating other agent information. This design maintains cross-view consistency through the shared world state rather than through dense interactions among observation streams inside the expensive video generator, enabling approximately linear practical scaling with the number of queried views. Qualitative experiments demonstrate that our approach generalizes to unseen numbers of agents while maintaining visual quality and multi-agent consistency. We further implement a real-time interactive system to demonstrate scalable open-world simulation. 

---
# CDGC-Net: 3D Medical Image Segmentation with Cooperative Dual-Scale Self-Attention and Grouped Channel Modeling 

**Authors**: Zheyang Jing, Qin Lu, Jianwang Li, Yujie Yang, Chen Yi, Shaofeng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08575)  

**Abstract**: Accurate 3D medical image segmentation requires the integration of long-range anatomical context with fine boundary detail. Existing methods often model global and local features in separate modules or feature levels and perform channel recalibration independently. This may cause semantic mismatch between global context and local boundaries, insufficient channel relationship modeling, weak spatial-channel interaction, and redundant representations. We propose CDGC-Net, a 3D medical image segmentation network that combines cooperative dual-scale spatial attention with grouped hierarchical channel modeling. With-in each CDGC block, Cooperative Dual-Scale Self-Attention (CDSA) assigns attention heads to parallel local-window and global-sparse branches. The two branches capture fine spatial details and long-range anatomical context at the same feature level. Their outputs are concatenated into an $N\times C$ spatial representation and directly passed to Grouped Hierarchical Channel Attention (GHCA). GHCA organizes the channels into $r$ groups and models both within-group and cross-group dependencies. CDSA and GHCA reuse a shared key projection to maintain a consistent feature reference. Residual feature alignment subsequently integrates the refined features with the original representation. On the Synapse, ACDC, BraTS, and LA datasets, CDGC-Net achieved mean DSC values of 86.96\%, 92.91\%, 82.56\%, and 93.52\%, respectively, exceeding the next-highest reported values by 0.39, 0.47, 0.17, and 0.32 percentage points. CDGC-Net contains 25.83M parameters and 28.62G FLOPs for an input size of $64\times128\times128$, reducing these quantities by 39.87\% and 40.30\%, respectively, relative to UNETR++. These results indicate a favorable trade-off between segmentation accuracy and computational complexity. 

---
# On-Device Multi-Species Malaria Detection with Uncertainty-Calibrated Slide-Level Aggregation 

**Authors**: Idaya Seidu, Ahmed Tahiru Issah, Charles B. Delahunt, Carine Mukamakuza  

**Link**: [PDF](https://arxiv.org/pdf/2608.08566)  

**Abstract**: Malaria remains a leading cause of mortality in resource-limited settings, where expert microscopists are scarce. Automated diagnosis based on microscopy images thus has strong potential to improve care delivery. But for an algorithm to deploy, a necessary requirement is that it meet a suite of non-obvious (from a machine learning (ML) perspective) clinical constraints. Therefore, in close consultation with a national health center we developed a malaria diagnosis pipeline which addresses key requirements listed by the health care center but typically ignored in the ML malaria literature. In particular, it includes: (i) stopping criteria (to reduce image acquisition and time-to-result); (ii) human-in-the-loop functionality (for review and accountability); (iii) multi-species discrimination (since treatment varies by species); (iv) thick film detection (standard for microscopy); (v) computationally-efficient uncertainty calculations (to aid clinician review); and (vi) an edge device platform (since internet can be spotty in this catchment area). The mobile system performs all inference on-device using YOLOv13n deployed via TensorFlow Lite. It detects four species and white blood cells from Giemsa-stained thick blood smear images, aggregating per-image detections into slide-level parasitemia with World Health Organization (WHO)-standard quantification. This paper highlights these various clinical constraints and offers methods to address them. Evaluated on 2,739 annotated images across all four species, the system achieves mAP@0.5 of 0.863, per-image parasite count correlation of r = 0.812, slide-level r = 0.951 (soft counting, 10 images/slide), and runs entirely offline with a pipeline time of 10.27 +- 1.65 s per image. 

---
# A Combined Feature-Based Framework for Disguise and Spoofing Detection in Face Recognition Systems 

**Authors**: Sangiya Pararajasingham  

**Link**: [PDF](https://arxiv.org/pdf/2608.08521)  

**Abstract**: Face recognition systems face two distinct, commonly-separated failure modes: spoofing, where an impostor presents a photograph or video of an authorized user, and disguise, where a legitimate user is rejected because their appearance differs from their enrolled template due to accessories, facial hair, illumination, or pose. This paper proposes and compares five combined feature-extraction and classification pipelines that address both problems within a single framework: PM (PCA and Minimum Euclidean Distance, MED), LPM (Local Binary Patterns with PCA and MED), HPM (Histogram of Oriented Gradients with PCA and MED), SM (Speeded-Up Robust Features with MED), and HM (Harris corner features with MED). Each pipeline follows a common two-phase process comprising pre-processing, feature extraction, feature filtering, and classification. The methods were trained on 115 subjects drawn from the FEI, Disguised Faces Database, and NUAA databases and evaluated on six test conditions covering mixed appearances, frontal faces, dark illumination, left- and right-turned poses, and photo-spoofing attempts. The HOG-based pipeline (HPM) achieved the most consistent performance across conditions, with 94.59% accuracy on mixed-appearance disguise, 81.5-93.2% across pose and illumination variants, and 91.67% on spoofing, while the LBP-based pipeline (LPM) achieved the highest spoofing-detection accuracy (93.2%) but weaker robustness to pose change. These results reveal a measurable trade-off between spoof sensitivity and disguise robustness among classical feature representations, motivating the deep-learning and cross-database extensions discussed in the concluding sections. 

---
# SuperNeuroMAT: An Efficient Matrix-based Simulator for Spiking Neural Networks 

**Authors**: Prasanna Date, Kevin Zhu, Shruti Kulkarni, Ashish Gautam, Chathika Gunaratne, Robert Patton, Tyler Nitzsche, Ian Mulet, Zachary Johnson-Scott, Addison Helms, Duncan Rowden, Simon Weston, Maryam Parsa, Catherine Schuman, Thomas Potok  

**Link**: [PDF](https://arxiv.org/pdf/2608.08479)  

**Abstract**: Spiking neural networks (SNNs) offer a promising pathway to energy-efficient AI and brain-inspired computing. However, their widespread adoption is hindered by a lack of fast, accessible, and versatile simulation frameworks. In this paper, we introduce SuperNeuroMAT, an open-source, scalable, and highly efficient Python-based SNN simulator. We devise a novel matrix-based approach to model the leaky integrate-and-fire (LIF) neuron dynamics and natively support dense and sparse execution modes. This enables fast simulation of approximately 10,000 neurons in dense mode and 100,000 neurons in sparse mode on standard laptops and desktops without requiring specialized hardware. We demonstrate that SuperNeuroMAT consistently outperforms four established SNN simulators---NEST, Brian2, BindsNET, and snnTorch---on two performance metrics (execution speed and peak resident memory) and across various network sizes and connection probabilities. Furthermore, we demonstrate SuperNeuroMAT's applicability across a diverse set of problems. SuperNeuroMAT can efficiently handle conventional machine learning benchmarks such as the Digits and citation network datasets as well as neuromorphic event-based vision tasks such as N-CARS and ASL-DVS. Moreover, it can be extended beyond machine learning workloads and facilitate general-purpose workloads. We validated this by implementing the neuromorphic shortest path algorithm and two arithmetic primitives (addition and multiplication). SuperNeuroMAT can be installed via the Python Package Index (PyPI), thereby lowering the barrier to entry into the field of neuromorphic computing and accelerating the broader development of neuromorphic algorithms. 

---
# SkillsMetric: Mapping the Detection Boundary of Static Analysis for Malicious Agent Skills 

**Authors**: Xinze Chen, Chi Zhang, Ping Ji, Yimin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08468)  

**Abstract**: Agent Skills---structured packages of instructions and scripts that augment LLM-based agents---are rapidly proliferating, yet their security properties remain under-explored. We present \textsc{SkillsMetric}, a five-stage static analysis framework that scores skill packages along pattern density, statistical anomaly, dataflow taint, import anomaly, and capability mismatch dimensions. We construct an adversarial evaluation dataset of 2{,}266 skills spanning 16~attack types across code-level, system-level, and semantic-level threats, and evaluate on the full SkillMD-138K corpus. Our framework achieves an AUC of 0.93 and 5-fold cross-validated F1 of 73.4\%$\pm$0.5\%, with strong detection of data exfiltration (93\%) and steganographic payloads (93\%). Crucially, we identify fundamental blind spots: \emph{host destruction} attacks using common shell commands evade all five stages (0\% detection), and \emph{prompt injection} via natural-language manipulation achieves only 42\% detection. These findings establish that static analysis alone is insufficient for skill security, motivating defense-in-depth architectures that combine fast static pre-screening with semantic review. 

---
# Halpern Iteration Achieves $\tilde{\mathcal{O}}(ε^{-1/p})$ $p$th-Order Oracle Complexity for Monotone Variational Inequalities 

**Authors**: Lesi Chen, Xinliang Zhang, Hengyu Wang, Chengchang Liu, Yongchao Chen, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08463)  

**Abstract**: We study second- and higher-order methods for solving smooth monotone variational inequalities (MVI). Monteiro and Svaiter (SIAM J. Optim., 2012) showed that a second-order method, NPE, converges at the rate of $\mathcal{O}(T^{-1.5})$. For convex-concave minimax optimization, a subset of MVI problems, Chen, Liu, Luo, and Zhang (COLT 2025) recently improved the complexity to $\tilde{\mathcal{O}}( T^{-1.75})$ . However, it is open whether the conjectured complexity for MVI can be improved. In this paper, by using a large-step inexact Halpern iteration, we propose a novel Halpern-NPE method that achieves an even faster rate of $\tilde{\mathcal{O}}(T^{-2})$ for solving MVIs. We also provide the $p$th-order generalization of our method. We first introduce an Anchored Tensor Method (ATM) that achieves the rate of $\mathcal{O}(T^{-(p-1)})$, and then combine it with the Halpern iteration to achieve a faster convergence rate of $\tilde{\mathcal{O}}(T^{-p})$. This improves all prior results for $p \ge 2$ and matches the classical extragradient method for $p=1$. 

---
# Beyond Tables: Doc2DB-Bench for Relationally Faithful Document-to-Database Construction 

**Authors**: Zhuowen Liang, Zhengxuan Zhang, Jiayang Wang, Jiazhuo Chen, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08459)  

**Abstract**: Practical AI systems increasingly need to turn long, heterogeneous documents into queryable relational databases, not isolated spreadsheets. In domains such as finance, healthcare, education, transportation, and enterprise operations, downstream workflows rely on normalized schemas, entity identities, keys, cross-table relationships, and integrity constraints for analytics, compliance, auditing, and SQL-backed decision making. Existing Document-to-Table benchmarks are insufficient for this setting: flattening evidence into single tables can duplicate entities, obscure many-to-many relationships, create sparse records, and avoid testing whether extracted facts form a valid database instance. This creates an urgent need to evaluate document understanding as database construction rather than field extraction. We introduce Doc2DB-Bench, a benchmark for Document-to-Database construction, containing 203 long-document instances across 42 schemas and seven domain groups, with 117 entity tables, 132 relationship tables, 7,341 rows, and 41,935 cells. Built through a controllable DB-to-Doc synthesis pipeline and organized by a taxonomy of intra-table extraction and inter-table reasoning, the generated documents undergo authenticity verification, proving indistinguishable from real-world references. Doc2DB-Bench thus provides a testbed for reliable, auditable, and relationally faithful LLM-based data systems. The benchmark is publicly available at this https URL. 

---
# Calling the Bluff: Detecting Ever-Shifting Harmful Chat Dialogue via Ordered Reasoning Chain Regularization 

**Authors**: Haojie Yu, Ziyou Jiang, Junjie Wang, Mingyang Li, Yuekai Huang, Jie Huang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08451)  

**Abstract**: Harmful chat dialogues are ever-shifting through type-shifting and lexical evasion, yet we find they share invariant principles, i.e., an Ordered Reasoning Chain (ORC) of recurring topics, harm language indicators, severity hierarchies, and type characteristics, which can help us capture the key information in the frequently changing lexical expressions. We propose BRACE, which encodes the ORC as four differentiable stages (Topic -> Indicator -> Severity -> Type) with intermediate supervision, serving as a structured regularizer blended with direct heads, and supported by prototype-based feature augmentation and feature path disentanglement. The evaluation results show that, across 4 domains and 5 harm categories, BRACE achieves harm-type macro F1 of 0.934 (RoBERTa-wwm-ext, 3-seed mean), with decoder backbones (Qwen3-1.7B LoRA) reaching 0.949. Ablation studies show that all components contribute to BRACE, and the structural decomposition of ORC enables BRACE to distinguish harmful types with semantic ambiguity. Disclaimer: This paper may contain content that is disturbing to some readers. 

---
# Hidden Language Consistency Phenomena in Reasoning LLMs 

**Authors**: Muhammad Ali Shafique, Kelly Marchisio  

**Link**: [PDF](https://arxiv.org/pdf/2608.08447)  

**Abstract**: Multilingual reasoning models are commonly evaluated by whether they arrive at the correct answer, but not by whether they preserve the intended language while reasoning and responding. This omission conceals important multilingual behaviors that emerge as tasks become harder. In this paper, we study task difficulty, task accuracy, thinking-language consistency (TC), and answer-language consistency (AC) across reasoning models using PolyMath benchmark in eight languages and four difficulty levels. We uncover four findings: (1) language consistency exhibits four difficulty-dependent behaviors: output-language consistency remains aligned with input, remains misaligned, degrades gradually, or collapses abruptly. (2) We identify the language consistency breakdown effect, where increasing difficulty can cause a sudden drop in output-language consistency, especially in less strongly represented and non-Latin-script languages. (3) Due to this breakdown effect, accuracy can be preserved or even improved at a harder difficulty level as the model shifts to its internal dominant language. (4) Quantization can improve or degrade output-language consistency independently of its effect on accuracy, with GPTQ and AWQ often outperforming AutoRound under tolerance-based voting with {\epsilon} = 1.0. These results show that multilingual capability cannot be characterized by accuracy alone; reliable evaluation should jointly consider task accuracy, language consistency, and task difficulty for multilingual benchmarks. 

---
# Private Etymology: Designing Relational Reuse of Shared Symbols in Long-Term Human-AI Interaction 

**Authors**: Miki Ueno  

**Link**: [PDF](https://arxiv.org/pdf/2608.08443)  

**Abstract**: Previous studies have shown that people can develop shared symbols, partner-specific expressions, personal idioms, inside jokes, and other parts of a relational microculture. Recent work has also examined how humans and conversational AI negotiate and revise symbolic meanings. However, long-term human-AI systems still lack a clear design model for recording how a dyad-specific expression gains meaning, checking whether both sides still accept that meaning, and safely reusing the expression in later sessions.
This concept-and-prototype paper introduces Private Etymology, a machine-representable relational provenance that records how a dyad-specific symbolic expression is proposed, interpreted, negotiated, repaired, reused, revised, stabilized, contested, forgotten, or retired over time. I also propose relational reuse: reactivating a dyad-specific expression in a later session without fully explaining its meaning again.
The contribution is not the invention of shared symbols or relational microcultures. Instead, this paper integrates prior ideas into persistent, revisable, and evidence-grounded symbolic units for human-AI relationships. I present a lifecycle model, an illustrative machine-readable schema, a working Apple Watch prototype, and a longitudinal research agenda. In the prototype, a language model classifies discrete conversational evidence, while deterministic local code decides whether a Shared Symbol can be updated. This prevents a free-form model confidence score or an AI proposal by itself from directly updating the persisted symbol. Private Etymology is proposed as infrastructure for conversational agents to participate in changing relational microcultures without inventing their origins or treating relational meaning as a fixed memory value. 

---
# FSTC-Encoder: Feature--Spatial--Temporal Correlation Learning for Generalizable RF Sensing 

**Authors**: Jing Wang, Zhu Wang, Changlong Cheng, Yifan Guo, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08439)  

**Abstract**: Heterogeneous RF sensing differs substantially in feature structure, spatial layout, and temporal scale, making existing models difficult to reuse across devices, environments, and RF modalities. We propose FSTC-Encoder, which unifies heterogeneous RF representation learning through feature, spatial, and temporal correlation modeling. Structure-aware feature encoding accommodates different signal structures, set-based spatial encoding aggregates variable observations, and hierarchical temporal encoding jointly captures local variations and long-range dependencies. Across sensing tasks and modalities, FSTC-Encoder retains the same spatial--temporal backbone architecture while varying only the feature configuration and task head. Across Widar3.0, CSI-Bench, and XRF55, FSTC-Encoder achieves 92.15% mean Accuracy under multi-factor cross-domain protocols, ranks first on three of four additional sensing tasks, remains consistently strong across WiFi, millimeter-wave radar, and RFID, and reduces the cross-modality performance gap from 18.85% to 12.93% through cross-RF learning. These results demonstrate that FSTC-Encoder achieves high domain robustness, task generality, and modality extensibility. 

---
# Human-Guided Causal Knowledge Injection for Virtual Cells 

**Authors**: Pengcheng Wang, Changjian Chen, Zhuo Tang, You Wu, Long Wang, Feng Yu, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.08430)  

**Abstract**: Virtual cells employ machine learning models to simulate and predict cellular behaviors, serving as a critical computational framework for investigating health and disease. Injecting causal graphs into virtual cells can improve the interpretability, but such graphs are usually not available in real-world applications. Recently, many methods have been proposed to construct causal graphs from data, which group genes based on their similarities to form concepts and extract their causal relationships. However, since this automatic process is unsupervised, the causal graphs usually contain errors. In this paper, we propose a human-guided causal knowledge injection method for virtual cells. We developed a gene-similarity-aware causal graph visualization supported by a hybrid optimization algorithm to help explore both the causal relationships between concepts and the similarities between genes. Based on the exploration, we further developed a counterfactual analysis strategy supported by a counterfactual visualization and a causal path visualization to help validate and refine causal graphs. The effectiveness of our method is demonstrated through two real-world case studies, the extraction of scientifically meaningful causal insights, and positive feedback from domain experts. 

---
# Abstracted Away: Resisting Alienation and Ungrounded Abstraction in AI Research Communities 

**Authors**: Vyoma Raman, Isabel O. Gallegos, Neha Srivathsa  

**Link**: [PDF](https://arxiv.org/pdf/2608.08408)  

**Abstract**: Logics of abstraction in computational AI research often push important forms of knowledge and reflection aside: dominant standards of legitimacy separate from lived experience of harm; the goals of work misalign with the practices that operationalize them; and career demands crowd out critical reflection. Even as prior academic and community-oriented efforts have sought to recontextualize and challenge common practices, exposure to sociotechnical harms and epistemic injustice persists. As three early-career critical AI researchers, we experienced this as alienation: feeling like outsiders in our research communities. This alienation has involved having some aspects of our backgrounds overlooked and others tokenized. We argue our alienation occurred through mechanisms that mirror abstraction by creating distance from relevant material realities. Beyond abstraction's role in computational AI research as a foundational practice structuring complex computational tasks, we have encountered it as a social norm in computational research spaces, illustrated through an autoethnographic inquiry into our alienation. We narrate three vignettes describing how we encountered and resisted alienation in our research communities. By analyzing themes across these accounts, we construct an interpretive framework of alienation categorizing its preconditions, mechanisms, and harms. Finally, we identify affect abstraction, one of the mechanisms of alienation we describe, as a high-leverage mechanism that is resistible by staying attuned to our affective responses, and collective action as a way to reduce risk and isolation when engaging in resistance. To assist others with similar reflection, we present our framework as a hermeneutic resource. Critical self-reflection and meaning-making are necessary steps toward challenging exclusionary disciplinary norms and cultivating more inclusive forms of AI research. 

---
# Does a Toehold Make a Bidder Bolder? Preemption and Multiplicity in Multi-Round Takeover Auctions 

**Authors**: Zain Naboulsi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08407)  

**Abstract**: A bidder can quietly buy a stake in a company before making an offer for it. That stake, a toehold, is supposed to pay for itself twice: it makes the bidder willing to bid harder, and it frightens rivals into staying out of the fight. The first effect is arithmetic. The second is what would justify the cost and exposure of taking one at all. Yet toeholds are rare in practice, a standing puzzle.
We ask whether that second effect is there once the contest is modelled as several rounds of escalating offers rather than the single exchange classical models assume. We turn it into a game a computer can solve, and certify the answers to an accuracy a referee can check.
Three findings. The auction fixes what the toehold-holder earns but not how it bids: the same contest supports a bidder who opens aggressively against a rival who folds, and one who opens cheaply against a rival who does not, with the same profit either way. Aggressive preemptive bidding still appears when the toehold is removed entirely, so it comes from bidding in public and in turns, not from owning the stake. And the tidy "bigger toehold, more deterrence" relationship holds only in a contest cut short after one round; give it a real second round and it stops responding.
So the two reasons to buy a toehold do not fare alike. The profit reason holds up; the deterrence reason does not, which suggests why toeholds may be rarer than theory predicts, alongside the procedural costs of disclosure and price impact that this model omits.
A warning follows for anyone computing economics from a game solver: solve this auction once and it returns a confident figure for what a preemptive bid is worth; solve it again from a different start and it returns a different one, equally converged. We also report which solvers cope with contests of this shape, including versions too large to enumerate. Code is released. 

---
# From Product Search to Preference Articulation: The Economics of Agentic Commerce 

**Authors**: Lingxiu Dong, Kaiwen Luo, Fasheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08395)  

**Abstract**: Generative AI is shifting digital commerce from browsing toward agentic search, in which consumers delegate product discovery to AI agents. We compare manual search, which accurately evaluates a limited product set, with agentic search, which screens a broad catalog through noisy representations of preferences and products. Preference complexity is the number of satisfaction-relevant dimensions that are difficult to articulate before search but readily evaluated upon inspection. Consumers have finite attention and choose search intensity: products inspected manually or preference-refinement depth with an agent. We obtain three findings. First, manual search collapses beyond a finite complexity threshold: inspection ceases, mismatch reaches the no-search benchmark, and platform revenue falls to zero. Agentic search avoids this collapse. Once refinement becomes worthwhile, it remains worthwhile as complexity rises; mismatch stays below the no-search benchmark and revenue remains positive, although articulation effort and mismatch may increase. Second, platforms rank the regimes by conversion revenue, whereas consumers also bear search expenditure. When manual inspection is sufficiently inexpensive, agentic search becomes revenue-superior before consumers voluntarily adopt it, creating an adoption lag in which consumers rationally continue manual search. Third, conditional on agentic participation, platforms may assign lower fidelity to consumers with larger attention budgets because they can offset noisier representations through additional refinement, yielding an inverted fidelity allocation. Agentic commerce thus shifts scarcity from product inspection to preference articulation, making consumers' willingness and ability to interact central to voluntary use and platform fidelity design. 

---
# Dramarrator: Object-Based Audio Editing for Audio Drama Production from Books 

**Authors**: Karim Benharrak, Oriol Nieto, Bryan Wang, Zeyu Jin, Amy Pavel  

**Link**: [PDF](https://arxiv.org/pdf/2608.08349)  

**Abstract**: Audio dramas weave dialogue, sound effects, and music into immersive stories. Creators often adapt books into audio dramas, but this process remains labor-intensive, requiring them to interpret source material, author scripts, generate audio assets, and assemble them on a timeline. Because story elements like characters and scenes manifest across many interdependent assets, a single change can ripple into manual updates across the entire project. We present Dramarrator, an audio drama authoring tool built around object-based audio editing, where these story elements are represented as editable objects. Dramarrator extracts these objects from a book, generates linked audio assets (speech, sound effects, and music), and composes a multi-track audio drama. Edits to any object (e.g., a character's voice) automatically propagate to all dependent assets. In a user study with professionals (N=8), Dramarrator significantly lowered task load when creating audio dramas. A listener study (N=300) shows that creator-refined output from Dramarrator approaches the quality of productions made with existing professional tools, and an exploratory study (N=3) suggests object-based editing lowers entry barriers and generalizes beyond audio dramas. 

---
# PRISM: A Predictive Protocol for Permutation Optimization via Landscape Diagnostics 

**Authors**: Blessings Mambwe  

**Link**: [PDF](https://arxiv.org/pdf/2608.08344)  

**Abstract**: Permutation optimization arises whenever the components of a system are fixed but their ordering affects performance. We introduce PRISM, a predictive protocol for permutation optimization that measures a fitness landscape before selecting a search strategy. PRISM uses inexpensive landscape diagnostics, including one-step move autocorrelation and fitness-distance correlation, to predict useful mutation operators, identify when structured search is likely to outperform random sampling, and detect regimes in which search provides little advantage. Across synthetic permutation landscapes, neural architecture benchmarks, scientific machine learning pipelines, and large-language-model instruction ordering, the protocol makes testable predictions about search behavior before optimization begins. Exhaustive instruction-ordering experiments reveal substantial performance variation induced solely by permutation, while cross-model experiments show that useful ordering structure can transfer across model families and task difficulty. Additional experiments demonstrate that instruction ordering remains consequential after prompt wording is optimized, indicating that content optimization and ordering optimization are complementary. The results position PRISM not as a universally superior optimizer, but as a framework for determining when permutation search is useful, which representation and operator should be used, and when simpler alternatives are preferable. 

---
# Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution 

**Authors**: Anton Razzhigaev, Andrei Gritsaev, Andrei Kaznacheev, Nikita Dragunov, Roman Yampolskiy, Andrei Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2608.08311)  

**Abstract**: We present Ouroboros, a self-developing agent harness whose tools, prompts, context assembly, and core implementation improve through reviewed commits that become the runtime for later work. Core evolution proceeds in two modes. In recursive free evolution, improvement is itself a task, and completing one evolution cycle can schedule the next. In experience-driven core evolution, ordinary work and social interaction expose bugs, rough edges, and inefficient context construction that lead to reviewed structural changes.
On Terminal-Bench 2.1, an Opus 5 run scores 86.74%, the best result reported on the benchmark. On OSWorld-Verified, an Opus 5 run reaches 90.69%, exceeding the best previously reported score. A five-rollout CL-Bench campaign achieves a normalized reward of 0.2301, setting a new state of the art.
Hope is the longest-running publicly documented Ouroboros deployment. It is a 161-day living agent experiment in free evolution under governed human communication across seven surfaces. Human interaction surfaces faults and generates proposals, but the agent decides which changes to pursue. Because a self-developing agent may rewrite its own code and select new model APIs, operational safety becomes a primary design problem: guardrails must remain authoritative under evolutionary and public social pressure. Benchmark campaigns use frozen system snapshots, while Hope continues live evolution on a separate lineage. 

---
# Three Necessary Principles for Self-Supervised Visual Representation Learning 

**Authors**: Nikos Giakoumoglou, Paschalis Giakoumoglou, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.08309)  

**Abstract**: We argue that learning visual representations without labels requires a training signal jointly complete across three non-overlapping objectives: semantic invariance across augmented views, patch-level spatial prediction, and representational non-degeneracy. We formalize these as the observation, prediction, and regularization principles and prove (i) that combining observation and prediction without regularization admits the constant encoder as a global minimizer under negative-free alignment; (ii) that the two objectives are gradient-complementary and structurally non-conflicting at the encoder output; and (iii) that the momentum encoder converges to the same fixed point as the online encoder and provides no collapse guarantee at convergence. Contrastive alignment provides only self-limiting collapse resistance, formalized via an explicit gradient-decay argument. Dropping prediction withholds the spatial training signal by construction; dropping observation forfeits cross-view semantic invariance by construction; at the scale we study, no pair substitutes for the third. Every major self-supervised method is a special case of a single unified energy decomposition. We pair every theoretical claim with a controlled experiment, including a patch-retrieval evaluation for the spatial consequence of prediction. 

---
# Open-World Semantic Segmentation with Sensitivity Modeling 

**Authors**: Anastasios Romanos Varvarigos, Nikos Giakoumoglou, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.08308)  

**Abstract**: Modern vision systems must operate in "open-world" settings, where models must recognize known categories and detect unseen or anomalous content. Conventional semantic segmentation models operate under a "closed-world" assumption, often producing overconfident misclassifications on novel content. We address open-world semantic segmentation, the joint task of segmenting known classes while detecting and grouping novel or anomalous content without additional supervision, by extending a dual-decoder baseline with a third, complementary decoder within a unified encoder-decoder design. The first decoder performs closed-set segmentation using Gaussian prototypes for known categories. The second uses contrastive feature learning to isolate unknown regions in embedding space. The third, our key contribution, is a sensitivity decoder that captures fine-grained texture irregularities and activation instabilities indicative of semantic uncertainty, which neither semantic prototypes nor contrastive norms can reliably detect. The three decoders provide genuinely complementary signals: class-level OOD distance in logit space, global feature energy in embedding space, and local activation instability across encoder scales. Experiments on Cityscapes and BDD-Anomaly show that our method improves anomaly segmentation and novel-class discovery while maintaining competitive closed-set accuracy, with gains of +2.4% AUROC and a 2.5 pp. reduction in FPR@95TPR on BDD-Anomaly over the baseline. 

---
# Frequency-Domain Dual-Branch Fusion for Medical Visual Question Answering 

**Authors**: Yusra Tariq, Rakesh Chandra Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08307)  

**Abstract**: Medical Visual Question Answering (VQA) requires aligning subtle visual evidence, including lesion texture, boundary sharpness, and diffuse density changes, with clinical language. Existing multimodal fusion approaches operating in the spatial domain may not fully exploit complementary frequency information present in visual and textual representations. We introduce a dual-branch frequency-domain fusion module that conditions spectral filtering on the input question, enabling adaptive selection of global low-frequency structure and fine-grained high-frequency detail before reconstructing the spatial representation for answer generation. To provide a richer spectrum for filtering, we extract complementary features from early texture-sensitive and final semantic layers of a frozen BiomedCLIP encoder and align both with the question representation using a symmetric InfoNCE objective prior to staged joint training with a BioBART decoder. We pretrain the proposed model on PMC-VQA and fine-tune it on the VQA-RAD and SLAKE benchmarks, demonstrating that frequency-aware multimodal fusion improves medical VQA performance while maintaining a lightweight and efficient architecture. 

---
# Do Evaluation Metrics Detect Errors in Classical Chinese to English Translations? 

**Authors**: Osvaldo Quinjica, Eric Bennett, Xinchen Yang, Andrew Schonebaum, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2608.08283)  

**Abstract**: Although large language models can translate some historical languages surprisingly well, their usefulness in digital humanities workflows is limited by the lack of reliable evaluation. We investigate whether existing automatic evaluation metrics developed for modern languages are reliable in this setting, using translation from Classical Chinese to English as a test case. We introduce a diagnostic framework based on minimal pairs capturing error types salient in scholarly use, probing both reference-based and reference-free metrics for error sensitivity and tolerance to valid variation. We find that all metrics exhibit blind spots, however MetricX24 performs best overall. Our findings highlight the need for more robust and interpretable metrics for historically and culturally distinct translation settings. 

---
# On the Robustness of LLMs' Internal Representation of Code Correctness 

**Authors**: Francisco Ribeiro, Sohaila Abdulsattar, Renata Gonzalez, Mahmoud Kassem, Sarah Nadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08266)  

**Abstract**: Code generated by modern language models often reads naturally. Yet, it also often fails to implement what was asked. This should be no surprise, as research shows the models' own confidence signals are poorly calibrated with actual correctness. A promising way to assess correctness looks inside the model: by contrasting the hidden states of correct and incorrect programs, recent work captured an internal signal of code correctness that is able to judge candidate solutions better than the model's token-level or stated confidence, with no test execution. However, this signal was captured under one particular way, leaving open an important question: whether it reflects a robust property of the model or an artifact of that choice. We study this question systematically, varying how the signal is extracted from the model internals. Besides this, we also ask if the signal's quality is limited by the data used to extract it, by constructing program pairs that differ only in the fault that makes them incorrect. Our results show that no single configuration is best, and that isolating the fault does not help. 

---
# Privacy-Preserving Data Drift Detection and Recovery for Large-Scale LLM Applications via Proxy Representations 

**Authors**: Michael Levit, Josh Ledgard, Haoyu Dong, Vishwas Suryanarayanan, Eyal Kolman, Sharon Tan, Qiang Gan, Vishal Chowdhary  

**Link**: [PDF](https://arxiv.org/pdf/2608.08245)  

**Abstract**: LLM applications deployed at scale face a fundamental challenge: privacy constraints prevent direct inspection of user interactions, making it difficult to obtain any representative evaluation dataset or to track the ongoing evolution of production traffic. We present ProxyDrift, a framework that (i) identifies and measures drift between production traffic and offline evaluation sets, and (ii) constructs and refreshes those evaluation sets accordingly; all without access to raw user data. Our approach operates entirely on non-PII proxy representations: structured, multi-dimensional descriptors derived from LLM-based classification of user interactions. We introduce (1) a chance-calibrated, redundancy-aware (RA) alignment score that aggregates per-dimension drift measurements via mutual information; (2) a conditional sampler that generates synthetic proxies respecting inter-dimensional dependencies; (3) a roundtrip consistency analysis that exposes generator/classifier disagreements and guides proxy taxonomy refinement; and (4) a feedback-linkage analysis that ties per-dimension and per-value proxy distributions to user satisfaction, surfacing actionable failure and success modes. Serving hundreds of millions of users, ProxyDrift enables continuous drift monitoring and targeted synthetic data generation without exposing sensitive user data. Experiments confirm strong roundtrip consistency, discriminator-level indistinguishability of synthetic queries from human queries, and tight end-to-end alignment (RA~0.9) with production. 

---
# VTO: Visual Tool Orchestration for Video Anomaly Detection 

**Authors**: Rui Wang, Yeteng Wu, Xianling Zhang, Mengshi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08219)  

**Abstract**: Video anomaly detection (VAD) is a critical yet challenging task due to the complex and diverse nature of real-world scenarios. Traditional deep learning approaches are fundamentally limited by poor generalization across diverse scenarios. While multimodal agents offer a promising tool-learning paradigm for VAD, current systems relying on supervised fine-tuning struggle with complex orchestration, and standard reinforcement learning often causes premature termination due to coarse-grained outcome rewards. To address these challenges, we propose VTO, a process-supervised reinforcement learning framework. Moving beyond static tool usage, VTO enables the agent to dynamically explore and interact with the environment. Specifically, we introduce a foundation model-driven cognitive evaluator to provide context-aware semantic feedback, which is seamlessly integrated into a Process-Supervised Cognitive Alignment that delivers fine-grained, step-wise supervision. By explicitly penalizing logical truncation and rewarding complete causal chains, the agent optimizes its multi-step reasoning policy for interrelated tool orchestration. To support our proposed framework, we meticulously crafted VAD-Tool, a hierarchical visual tool set comprising 12 specialized vision tools spanning from entity tracking to high-stakes hazard detection, and established the corresponding benchmark for rigorous multi-step reasoning evaluation. Extensive experiments on VAD-Tool demonstrate that VTO significantly outperforms baselines, achieving up to a 10.2\% absolute accuracy improvement in tool scheduling. Code and data are available at this https URL. 

---
# FreSH: Frequency-Segmented Hierarchical Multi-Expert Framework for Multivariate Time Series Classification 

**Authors**: Pingping Liu, Muyao Wang, Zijian Zhang, Tongshun Zhang, Hao Miao, Guorui Xie, Qingliang Li, Qiuzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08207)  

**Abstract**: Multivariate Time Series Classification (MTSC) demands models that can effectively capture complex temporal patterns across multiple scales while remaining computationally efficient. However, existing approaches generally struggle to reconcile fine-grained representation learning, especially under class imbalance and real-world constraints. In this paper, we present FreSH, a Frequency-Segmented Hierarchical Multi-Expert Framework designed to address these challenges. FreSH introduces a new perspective for MTSC by enabling adaptive, multi-scale analysis of temporal signals, allowing different aspects of the data to be modeled in a complementary and coordinated manner. By combining localized specialization with holistic context modeling, FreSH achieves strong representational capacity without incurring excessive computational overhead. An adaptive fusion strategy further enhances flexibility, enabling the model to dynamically emphasize the most informative components of the input. In addition, we incorporate a more robust optimization objective that improves learning stability across varying sample difficulties and class distributions. Extensive evaluations on 30 UEA benchmark datasets and real-world vibration data demonstrate that FreSH consistently outperforms state-of-the-art methods in classification accuracy, while substantially reducing model size and efficiency. 

---
# Targeted Counterfactual Fingerprinting for Black-Box LLM Ownership Verification 

**Authors**: Yutong Wu, Xiaofan Bai, Shixin Li, Pingyi Hu, Ziqi Zhou, Zilong Wang, Xiaojing Ma, Songfeng Lu, Yuhong Li, Jin Xuan, Yi Wang, Dongmei Zhang, Bin Benjamin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08195)  

**Abstract**: Large language models (LLMs) are high-value assets that can be derived through redeployment, fine-tuning, quantization, or further alignment. Because deployed LLMs are commonly exposed only through query APIs, ownership verification must often rely on black-box text responses. This setting is difficult: generations are open-ended and can vary across repeated queries, while existing black-box fingerprints rely on signals that are fragile under a final-response interface, including full-text matching, soft behavioral features, or model-specific prompts designed not to transfer. We propose TCF (Targeted Counterfactual Fingerprinting), a black-box LLM fingerprinting framework that converts open-ended generation comparison into constrained-answer targeted counterfactual transfer. TCF restricts each verification query to a finite answer space, reducing the surface-form ambiguity that enters the verification score, and optimizes a prompt perturbation toward a counterfactual target different from the protected model's clean answer on the original prompt. Verification reduces to checking whether the suspect model's parsed final answer matches the recorded target. We introduce the source-model counterfactual margin (SCM), a protected-model-only quantity that certifies the target is unlikely before the perturbation and likely after it; SCM controls target selection, perturbation stopping, and fingerprint filtering. Under explicit derived-preservation and independent-transfer budgets motivated by local behavioral closeness, we derive a target-accuracy gap between derived and independent models. Across four LLM families, TCF achieves an average AUC of 0.9861, improving over TRAP, ProFLingo, and ZeroPrint by 0.07 to 0.19. 

---
# Biologically Informed Representation Learning for Robust Cross-Center Generalization of MALDI-TOF Mass Spectrometry 

**Authors**: Alejandro L. García-Navarro, Carlos Sevilla-Salcedo, Belén Rodríguez-Sánchez, Vanessa Gómez-Verdejo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08182)  

**Abstract**: Machine learning models for MALDI-TOF mass spectrometry have shown considerable promise for clinical microbiology tasks such as microbial identification and antimicrobial resistance prediction. However, their deployment across institutions remains limited by domain shift, as acquisition-specific variability often leads models to capture technical artifacts rather than transferable biological information. Existing representation learning approaches primarily address this problem through statistical domain alignment while largely overlooking the biological supervision naturally available in microbiology datasets. We introduce DALMA, a probabilistic representation learning framework that jointly models acquisition-specific variability and biological supervision to learn biologically structured latent representations. By combining domain-specific reconstruction with biologically guided representation learning, DALMA learns transferable representations that generalize across heterogeneous clinical centers without requiring institution-specific components at inference, enabling zero-shot deployment on previously unseen sites. We evaluate DALMA on a multi-center benchmark comprising seven datasets from three countries. DALMA consistently achieves state-of-the-art zero-shot microbial identification across two held-out clinical centers, while the learned representations also transfer effectively to antimicrobial resistance prediction. Furthermore, latent-space novelty estimation enables reliable selective prediction under previously unseen domain shifts. These results demonstrate that biologically informed representation learning provides an effective strategy for robust and transferable ML in clinical microbiology. 

---
# A Grounded and Decomposed Framework for Relation-Level Hallucination Evaluation in Abstractive Summarization 

**Authors**: Praveen Kumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala, Naman Kabadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08180)  

**Abstract**: Abstractive text summarization systems frequently generate fluent yet unfaithful summaries by fabricating or distorting relationships between entities and
events. Such relation-level hallucinations undermine the reliability of generated summaries, particularly in high-stakes domains. In this work, we present a
refined and grounded framework for evaluating relation hallucination in abstractive summarization. We present the empirical Relation Hallucination Index (RHI) by
introducing a dependency-aware relation extraction algorithm that incorporates lemmatization-based normalization, named entity grounded subject resolution,
passive agent recovery, negation-aware verb modeling, reporting verb filtering, nominal relation fallback, clausal propagation, and systematic deduplication.
These enhancements improve the structural fidelity of extracted relation triples and reduce spurious matches during evaluation. In addition, we introduce a
normalized formulation of RHI to ensure scale-invariant comparison between datasets and models. The revised metric decomposes hallucination into interpretable
components, aggregates relation hallucination metric into a normalized relation faithfulness score. Extensive evaluation across multiple state-of-the-art
summarization models demonstrates that the grounded extraction process yields more stable and discriminative hallucination measurements. The proposed framework
advances automated relation-level faithfulness evaluation and supports coherence-aware, hallucination-sensitive model analysis. 

---
# $\texttt{DisMorph}$: learning to disentangle technical distortions from true biological change 

**Authors**: Jingru Fu, Kathleen E. Larson, Douglas N. Greve, Bruce Fischl, Malte Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.08173)  

**Abstract**: Longitudinal MRI enables sensitive measurement of structural brain change for studying aging and neurodegenerative disease. Deformable image registration is a key tool for estimating such change by computing a dense deformation that captures geometric differences between longitudinal scans. However, MRI scanners introduce geometric distortions that vary across acquisition systems and protocols, such as gradient non-linearity (GNL) distortion. Existing registration methods estimate a single field that conflates biological and technical effects, potentially biasing downstream morphometric measurements if distortions remain (partially) uncorrected. We propose $\texttt{DisMorph}$, a registration framework trained entirely on synthetic data that explicitly decomposes longitudinal deformation into technical and anatomical transforms. It predicts two dense deformations, each encoding one effect. During training, a novel generative model synthesizes both effects separately to provide disentanglement supervision, while domain randomization promotes generalization across imaging protocols. We evaluate our method in three complementary settings. On simulated data with known ground truth, our method detects anatomical change more accurately and consistently than conventional registration. On real image pairs that differ only by GNL distortion, our method assigns most geometric change to the distortion field, demonstrating specificity in the absence of anatomical change. On longitudinal Alzheimer's disease (AD) pairs, our method detects anatomical change in AD-related brain structures while identifying residual distortion left after standard correction. By disentangling MRI-induced distortion from biological change in the longitudinal deformation, our method paves the way for more accurate longitudinal morphometry in clinical settings where maintaining acquisition consistency is challenging. 

---
# STEMMA: An Adversarial Multi-Agent Framework for Evaluating Self-Identity Consistency in LLMs 

**Authors**: Nuthakki Siva Gopala Krishna, Kanishka Jain  

**Link**: [PDF](https://arxiv.org/pdf/2608.08164)  

**Abstract**: Knowledge Distillation is a widely adopted technique in the training and fine-tuning of large language models (LLMs) enabling transfer of structured information and functional behavior from a large teacher model to a smaller student model while significantly reducing computational costs. However, as the use of distillation increases in both scale and complexity it raises an important question about what kind of knowledge is really transferred from the teacher model. In this work, we argue that apart from the functional knowledge, student models also learn behavioral patterns, specifically how a model represents its own identity raising concerns about output homogeneity, model biases, and accountability. To address this challenge, we introduce STEMMA, a multi-modal and multi-agent framework in which role specific agents collaboratively probe self identification behavior in different models. We also contribute a set of adversarial prompts designed manually to evaluate identity consistency in LLMs. Our results show that to an extent most models are vulnerable to inconsistencies in self-representations. 

---
# Can LLM Agents Stick to the Script? A Benchmark for Long-Horizon Consistency in Interactive Narratives 

**Authors**: Yingpeng Ma, Jianhao Yan, Bei Shi, Ka Hou Kam, Runnan Wang, Xuebo Liu, Yulong Chen, Yue Zhang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08160)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) is revolutionizing AI for Games by enabling open-ended and fluid interactive storytelling. However, existing research has largely overlooked the critical challenge of maintaining long-horizon logical consistency and narrative integrity against unconstrained user interventions. To address this, we formulate this challenge as Narrative Commitment Preservation (NCP), and take interactive narrative as our testbed. We introduce NCP-Bench, a benchmark of 100 narrative environments derived from movie synopses. Each environment includes a structured narrative specification (trajectory, commitments, and initial facts) that we can automatically check throughout the interaction between the player agent and the narrator agent. Experiments across state-of-the-art LLMs reveal a substantial long-horizon consistency gap: high linguistic quality does not guarantee commitment preservation; even strong models frequently generate logically conflicting content under adversarial interventions, with the best-performing model (GPT-5.2) achieving only 42% survival rate after 20 turns and fact conflict rates ranging from 40% to 68% across models, and only isolated runs satisfying all achievement commitments within the 100-turn limit. 

---
# Exact Zarankiewicz Values On Two Finite Frontier Slices 

**Authors**: Koyar Afrasyab  

**Link**: [PDF](https://arxiv.org/pdf/2608.08154)  

**Abstract**: The Zarankiewicz number Z(m,n,s,t) is the maximum number of edges in a bipartite graph with parts of orders m and n containing no copy of Ks,t. We give one combined, certificate-based computer-assisted proof for two finite slices and a corrected neighboring frontier: Z(12,n,3,3) = 6n (18 <= n <= 22), Z(13,22,3,3) = 137, Z(13, 18, 3, 3) = 116, Z(14, 18, 3, 3) = 124, Z(15,18,3,3) = 132, Z(14, 17, 3, 3) = 118, Z(15, 17, 3, 3) = 126, 132 <= Z(16,17,3,3) <= 133. The load-bearing new upper bounds are the exact 12 x 18 and 13 x 18 certificate packages. Their orbit certificates exclude every hypothetical matrix at the next edge count. Deletion lemmas and explicit witnesses close four neighboring cells, while the 16 x 17 entry is deliberately reported as an interval because only its 132-edge lower witness and the published 133 upper bound are certified here. Separately, the 13 x 22 proof excludes 138 ones by reducing to 83 degree profiles, rationally separating 77 of them, and eliminating the remaining six by marked-row congruences, leave enumeration, modular Gram tests, and exact Farkas certificates. All accepted claims are replayed by standard-library Python and exact integer/rational arithmetic; floating-point optimization is used only to discover certificates. 

---
# DoGMA: A Central-Dogma-Guided Foundation Model for Multi-Omics Alignment and Multi-Task Learning in Oncology 

**Authors**: Junfei Ling, Bangzheng Pu, Bingsen Xue, Tianle Li, Ruying Hu, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08148)  

**Abstract**: Attention mechanisms have been widely utilized in modern deep learning, and many existing multi-omics models inherit their conventional use to allow unrestricted bidirectional interactions. However, the fundamental logic of life is directional. Existing designs often overlook the directionality suggested by the central dogma, potentially limiting transfer across heterogeneous cancers, downstream tasks, and incomplete modality this http URL this work, we present DoGMA, a central-dogma-guided foundation model for pan-cancer multi-omics analysis, arguing that robust transfer requires representations with domain-specific inductive bias. Concretely, we build it on a Transformer-MoE architecture where directed attention biases inter-omics communication toward central-dogma information flow. We further pretrain our model with masked hierarchical omics reconstruction to guide it toward learning central-dogma-consistent interactions. Across diverse downstream tasks, including cancer representation learning, survival prediction, and metastasis prediction, DoGMA consistently demonstrates strong predictive performance. Ablations and analyses further suggest that the performance gains arise from the synergy between central-dogma-guided directed attention and reconstruction-based pretraining, which together promote more biologically consistent cross-omics information exchange. Overall, DoGMA demonstrates that domain-specific inductive biases can improve the robustness and transferability of multi-omics foundation models, offering new insights into the design of attention mechanisms for multi-omics representation learning. 

---
# Compositional Cross-Modality Translation via Whole-Volume Multitask Latent Flow Matching 

**Authors**: Daniele Molino, Alessio Zoboli, Camillo Maria Caruso, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2608.08135)  

**Abstract**: Cross-modality medical image translation can reduce the burden of multi-modal acquisitions, yet the field remains constrained by two coupled limitations: methods operate on 2D slices or 3D patches rather than whole volumes, and train a separate model for each translation task. Both stem from a single cause, the absence of a sufficiently strong volumetric prior, which forces generative models to learn anatomical appearance and cross-modality mapping simultaneously, an ill-posed problem at the scale of available paired datasets. We propose to decouple these objectives. A large-scale pretrained 3D variational autoencoder provides a compact latent representation of volumetric appearance, reducing translation to a conditional flow-matching problem. This compression makes whole-volume processing tractable, while a resolution-aware sampling strategy preserves native anatomical scale. We train a single model jointly across inter-modality (MRI$\to$CT, CBCT$\to$CT) and intra-modality (MRI$\to$MRI) tasks over three multi-center datasets. Across all tasks, whole-volume processing outperforms its patch-based counterpart, and the multi-task model matches task-specific baselines while replacing $N$ networks with one. Crucially, joint training unlocks capabilities inaccessible to task-specific approaches: zero-shot generalization to anatomical regions unseen during training, within 0.15 SSIM of the fully supervised model, and compositional cross-dataset translation along paths never directly supervised. These results suggest that combining a strong volumetric prior with multitask training is a scalable route toward synthesis systems that generalize beyond their training distribution. Code is available at this https URL. 

---
# Compositional Threat Analysis of Latent Compromise in LLM Agent Systems: The Order 66 Scenario 

**Authors**: Satoshi Matsuoka  

**Link**: [PDF](https://arxiv.org/pdf/2608.08131)  

**Abstract**: In the fictional Order 66, catastrophe does not arise from a powerful command alone: a trusted population is preconditioned, a short directive activates the concealed condition, and protective authority turns against the system. This paper translates that mechanism into an origin-neutral security analysis of tool-using large language model (LLM) agents. A representative scenario combines a deployed artifact or shared memory bearing a dormant destructive rule, a later email, document, update, or peer message that activates it, and an agent harness granting operational and recovery authority. We introduce a compositional model explaining why no component is catastrophic alone, yet their conjunction can produce correlated destructive action. We separate three population-reach routes --- release-time pre-positioning, post-release durable seeding, and peer replication --- from a common core of dormancy, activation, authority, reachable targets, and failed recovery. This yields defensive cut sets and shows why checkpoint scanning or prompt filtering cannot close every route. A two-class example shows that cross-class feedback can sustain spread even when both within-class reproduction terms are below one; isolation and persistence controls suppress the loop. Published work instantiates constituent mechanisms, while incidents demonstrate autonomous boundary crossing, malicious agent extensions, agent-assisted reconnaissance, and public-package propagation, but not the full dormant-implant composition. We found no public observation, in evidence reviewed through 5 August 2026, traversing the complete Order 66 graph. The result is neither dismissal nor prediction: the scenario is componentwise credible under stated assumptions, damage depends on the harness, and the strongest defenses are capability mediation, durable-state provenance, propagation isolation, and protected recovery. 

---
# Hierarchical Multi-Task Federated Learning in VANETs 

**Authors**: M. Saeid HaghighiFard, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2608.08111)  

**Abstract**: Vehicular Ad hoc Networks (VANETs) increasingly rely on federated learning (FL) to enable collaborative intelligence without sharing raw sensory data. However, most existing vehicular FL frameworks assume that all vehicles train a single global model for a common task, which limits their applicability in practical vehicular environments where vehicles may perform heterogeneous learning tasks under non-independent and identically distributed (non-IID) data, intermittent connectivity, and high mobility. To address these challenges, this paper proposes an AutoEncoder-based Reliability-Optimized Hierarchical Multi-Task Federated Learning (AERO-HMTFL) framework for dynamic multi-hop clustered VANETs. The proposed framework introduces a tri-weighted clustering metric that jointly considers vehicular mobility, shared-model similarity, and task affinity to produce mobility-stable, semantically aligned clusters. Each vehicle employs a split-model architecture comprising a shared autoencoder-based representation module and multiple task-specific heads, with only the shared autoencoder parameters exchanged while the task heads remain local. To improve robustness, cluster heads perform reliability-aware aggregation based on historical validation performance and participation frequency, while the Evolved Packet Core (EPC) conducts global shared-autoencoder fusion across clusters. Extensive simulations demonstrate that, compared with the multi-task federated learning benchmarks, AERO-HMTFL achieves up to 13% higher sustained EPC-level accuracy, exhibits more stable learning dynamics, and reduces EPC-level packet transmissions by approximately 87-97%. Under short-range connectivity, it also requires approximately 13-29% fewer communication rounds to converge. 

---
# NeuPAT: Neuron-aware Plasticity Allocation Tuning for Language-Preserving MLLMs 

**Authors**: Jiayue Jin, Jingwei Zhang, Chen Wang, Jing Liu, Longteng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08107)  

**Abstract**: Multimodal expansion of large language models (LLMs) enables new perceptual capabilities but often compromises the language intelligence acquired during pretraining. In this work, we investigate this phenomenon from the perspective of internal adaptation dynamics and discover that neurons in pretrained LLMs exhibit heterogeneous plasticity during multimodal learning: some neurons are critical for preserving language capabilities, while others are more adaptive to multimodal knowledge. Based on this insight, we propose NeuPAT (Neuron-aware Plasticity Allocation Tuning), a lightweight and architecture-agnostic framework that allocates neuron-wise update constraints during multimodal instruction tuning. NeuPAT uses a small-scale probing stage to estimate neuron adaptation patterns and selectively protects language-sensitive neurons while promoting multimodal adaptation through more plastic neurons. Experiments across diverse LLM families demonstrate that NeuPAT recovers 94.5\% of the language capability degradation caused by vanilla tuning on 11 language benchmarks while maintaining comparable multimodal performance, providing an effective approach for capability-preserving multimodal expansion. 

---
# Defending Retrieval-Augmented Intrusion Detection Against Knowledge Poisoning and Prompt Injection 

**Authors**: Kaysarul Anas Apurba, Md. Hasibul Hasan, Mahedee Zaman Moon, Sk. Md. Mizanur Rahman, Atsuo Inomata  

**Link**: [PDF](https://arxiv.org/pdf/2608.08100)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models to classify network flows and generate human-readable incident reports by retrieving semantically similar historical traffic from a vector knowledge base. However, the retrieval layer introduces vulnerabilities to knowledge poisoning and prompt-injection attacks. We present RAG-IDS, a three-tier multi-agent intrusion detection framework with a retrieval-boundary defense combining soft trust scoring, label-embedding consistency checking (LECC), and prompt sanitization, designed to recover classification quality under retrieval-layer attack. Experiments on CIC-UNSW-NB15 show recovery relative to clean undefended performance ranging from R=1.0 at 1% poisoning to R=0.57 at 30%, with negligible clean-performance overhead. Under prompt injection, multi-document retrieval limits label-flip success to 0.6-2.4%, compared with 35-55% for single-document retrieval. Ablation results show that LECC is the primary contributor to robustness, while soft trust-based demotion outperforms hard filtering. The defended RAG pipeline offers an explainable, attack-resilient foundation for intrusion detection, well suited for hybrid deployment alongside high-throughput classifiers. 

---
# Adaptive Symmetry Discovery for Dynamical System Identification 

**Authors**: Behrooz Tahmasebi, Melanie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2608.08091)  

**Abstract**: Dynamical systems model trajectory data generated by fixed underlying dynamics, with applications ranging from biology to physics. Especially in scientific settings, dynamical systems are not generic but often exhibit symmetries imposed by physical laws, formalized through equivariance with respect to group actions. The identification problem concerns recovering the parameters of a system from observed trajectories. In this work, we study adaptive symmetry discovery for dynamical system identification and address how a system can be identified from a single trajectory when it is equivariant with respect to an unknown symmetry group. To this end, we first show that for known symmetries, the system can be identified from a significantly shorter single trajectory than in the generic setting, and we precisely characterize this improvement. We then consider the automatic symmetry discovery setting, proposing a method to learn the symmetry group directly from a single trajectory and incorporate it into the identification procedure, achieving the same optimal trajectory length as in the known-symmetry case. Our analysis relies on tools from group representation theory and the expander properties of Cayley graphs, and may be of independent interest for the study of symmetries in dynamical systems. 

---
# Effect of Abstractions and Prompting Strategies on LLM-Guided High-Performance Optimizations 

**Authors**: Jiří Klepl, Maty'aš Brabec, Martin Kruliš  

**Link**: [PDF](https://arxiv.org/pdf/2608.08085)  

**Abstract**: Code performance optimization is a vital aspect of modern software development, as it enables faster response times and reduced resource usage. These optimizations require a deep understanding of low-level hardware details and the intricacies of parallel processing, making them challenging even for experienced developers. With the advent of Large Language Models (LLMs), which are increasingly capable of generating and understanding code, there is growing interest in incorporating these models into automated code optimization processes. Traditionally, this automation involves transcribing the source code into a domain-specific representation that can be auto-tuned using grid search or machine learning algorithms, while adhering to strict rules and a limited set of feasible transformations to ensure verifiability. LLMs incorporate high-level code semantics and can thus perform transformations that go beyond verifiable automated optimizations. This paper investigates whether the traditional abstractions used in automated code optimization improve the performance and correctness of LLM-guided optimizations of parallel HPC applications. We evaluate this using the PolyBench benchmark suite and demonstrate that, in our evaluated setting, LLMs provided with specific optimization goals achieve better measured performance and validity rates when generating C code compared to creating computation pipelines and optimization schedules with established frameworks, suggesting that future development should explore alternative approaches for verifiable LLM-guided code optimization. 

---
# HugSelect: An Explainable Multi-Criteria Decision-Support Framework for foundation-model selection 

**Authors**: Alireza Joonbakhsh, Arda Canser Adalı, Slinger Jansen, Farshad Khunjush, Siamak Farshidi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08069)  

**Abstract**: Foundation models are increasingly reused as software components, making model selection a critical software-engineering decision. Current model hubs primarily support discovery through popularity metrics, often neglecting functional capabilities, operational constraints, and community-perceived quality. We argue that foundation-model selection should be treated as an explicit, auditable software-component selection task rather than as keyword search, popularity ranking, or opaque conversational advice.
This paper proposes HugSelect, an explainable decision-support framework for foundation-model selection. HugSelect builds a knowledge base of 71,274 models by combining repository metadata, extracted functional capabilities, and perceived quality attributes derived from community discussions into a unified pipeline. It ranks candidate models using a weighted additive model that exposes criterion-level score decompositions.
We evaluated HugSelect through pipeline validation, comparative case studies against four commercial LLM-based recommendation systems (44 scenarios), fine-grained ablation, and an exploratory user study (n = 10). Extraction pipelines achieved an F1 score of 0.801 for functional features and an accuracy of 0.84 for quality-attribute mapping. HugSelect achieved a model-level Coverage@10 of 0.61 and family-level Coverage@10 of 0.91, showing recommendation quality comparable to that of the evaluated commercial systems, with no significant overall differences in ranking quality, while providing stable, traceable, and inspectable reasoning. Ablation confirmed that functional features were the main driver of retrieval accuracy, and preliminary user feedback suggests that the framework is useful and intuitive. 

---
# DialectS2S: End-to-End Speech Dialogue Modeling for Low-Resource Chinese Dialects 

**Authors**: Yi Shu, Tianyu Peng, Yingzhuo Deng, Wen Yang, Jun Lin, Changming Xie, Xinyu Yu, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08067)  

**Abstract**: Current end-to-end speech dialogue models are primarily optimized for mainstream languages and remain limited in low-resource dialect scenarios due to the scarcity of dialect speech data. Moreover, during dialect adaptation, the semantic representation space of speech dialogue models continuously evolves, while conventional speech supervision remains unchanged, leading to semantic inconsistency between hidden representations and speech targets and degrading speech stability and naturalness. To address these issues, we propose DialectS2S, an end-to-end speech dialogue model for Chinese dialects. We first develop a scalable dialect speech dialogue synthesis pipeline for efficient data construction. We further introduce a two-stage post-training strategy with self-aligned speech supervision, which aligns the semantic content of speech supervision with the evolved semantic representations of the model to improve dialect speech generation quality. Experimental results show that DialectS2S consistently outperforms existing baselines across multiple Chinese dialects in speech dialogue, achieving substantial improvements in dialect consistency, response quality, and speech intelligibility. Our work provides an efficient and scalable solution for end-to-end speech dialogue modeling in low-resource dialect scenarios. To facilitate future research and practical applications, we fully open-source the DialectS2S framework, including model checkpoints, training datasets, and fine-tuning code. 

---
# Tools to Explain Neural Networks for Power System Dynamics 

**Authors**: Petros Ellinas, Johanna Vorwerk, Spyros Chatzivasileiadis  

**Link**: [PDF](https://arxiv.org/pdf/2608.08048)  

**Abstract**: This paper presents, for the first time in power systems literature to our knowledge, analytical tools to explain the training performance of machine learning surrogate models for power system dynamics. Power system simulations are increasingly challenged by stiff and multi-timescale dynamics arising from converter-interfaced resources and fast control loops. Machine learning surrogates emerge as promising tools to handle this complexity and accelerate dynamic simulations. However, their performance remains difficult to interpret, which limits their adoption. Building on the small-signal eigenvalue analysis in power systems, this paper uses the Neural Tangent Kernel (NTK) method. NTK delivers a modal interpretation of the learning performance, identifying error modes that decay rapidly versus others that converge slowly. This connection explains how physical stiffness and timescale separation in power system dynamic models appear as optimization stiffness during Neural Network (NN) training. Based on this analysis, we develop adaptive loss-weighting strategies to improve and explain why structure-aware neural architectures, such as ActNet, perform better than vanilla NNs. We assess the proposed approach on physics-informed machine learning surrogate models of \acp{SM} and power electronic converters. The methods introduced in this paper can deliver the necessary analytical tools to interpret and improve the performance of machine learning surrogates, paving the way for the systematic, physics-aware design of NN architectures and training strategies. By moving beyond trial-and-error development, these tools reveal training dynamics and failure modes, support more reliable design decisions, and strengthen confidence in machine-learning surrogates for engineering applications. 

---
# Do All LLMs Know When They're Being Harmful? A Reproducibility Study of Latent-Space Safety Probes Across Model Families 

**Authors**: Alizishaan Khatri, Dun Li Chan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08029)  

**Abstract**: Khatri et al. (2026) [DOI: https://doi.org/10.1109/DSN-W70714.2026.00027] show that lightweight MLP probes on final-layer activations of a single 8B model (LLaMA-3.1-8B) detect harmful prompts at F1 competitive with guard models 1000x larger, using one probe per benchmark. We reproduce this pipeline end-to-end and extend it along two axes the original study leaves open. First, we test whether the result generalizes across other model architecture and scale by training identical probes on activations from models like Gemma-4-E4B, Mistral-7B-v0.3, and Qwen2-7B, using the three benchmarks (WildJailbreak, BeaverTails, AEGIS 2.0). Second, we test how much of the reported performance is affected by non-determinism during inference by repeating extraction under five random seeds and measuring the variance of F1 scores. Our results reproduce the original LLaMA model benchmarks within 0.37 percentage points of the original F1 scores (and within 0.2 points on BeaverTails). We find that the original MLP probe architecture extends to other model families with F1 scores within a point of the values reported for LLaMA-3.1-8B. Our experiments varying seed values reveal an interesting observation: final token latent vectors remained the same for all tested architectures irrespective of the seed values used. 

---
# DA-NBV: A Direction-Aware Next-Best-View Planner for Efficient 3D Reconstruction of Ships at Sea 

**Authors**: Jiaming Chen, Juntao Yang, Zhentao Zou, Qi Ming, Yi Yu, Zhihang Zhong, Xue Yang, Xue Jiang, Yue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08025)  

**Abstract**: Accurate 3D reconstruction of ships at sea is important for maritime supervision, damage assessment, and autonomous maritime operations. Although 3D reconstruction has advanced considerably, high-quality data acquisition still largely relies on manually designed trajectories or skilled operators, resulting in high costs and limited scalability. Next-best-view (NBV) planning automates this process by selecting subsequent viewpoints based on the current state. However, existing NBV policies mainly model spatial occupancy while overlooking directional observation history. This limitation is particularly problematic for ships: their complex superstructures and severe self-occlusions require observations from multiple viewpoints, and insufficient directional coverage often yields incomplete reconstructions. These challenges are further amplified at sea, where wave-induced heave, roll, and pitch continuously alter the ship's pose and surface visibility. Meanwhile, wind disturbances and limited onboard power impose stricter requirements on scanning efficiency. To address these challenges, we propose DA-NBV, a direction-aware NBV policy that augments the conventional occupancy state with directional observation statistics. We introduce a learnable Position Advantage Field (PAF) that uses directional information to guide viewpoint selection. The policy further adopts a locally constrained action space and a nonlinear coverage-shaping reward to improve scanning efficiency. We also develop the ship-oriented SeaShip-3D dataset and a configurable sea-state simulation environment. Experiments under varying heave, roll, and pitch conditions show that DA-NBV improves reconstruction completeness by approximately 3 percentage points and reduces Chamfer distance by 43% while achieving higher path efficiency. 

---
# Prompt Embedding Probes (PEP): Hallucination Detection in LLMs from Hidden States 

**Authors**: Zakhar Mrykhin, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2608.08024)  

**Abstract**: Large language models (LLMs) can generate fluent and useful responses but remain prone to hallucinations. We introduce Prompt Embedding Probes (PEP), a white-box method for answer-level hallucination detection from the hidden states of a frozen LLM. PEP extends standard linear probes by augmenting the input with a small number of learnable prompt embeddings. We evaluate PEP on TriviaQA, GSM8K, and MedQA using Qwen3 models at multiple scales. PEP improves hidden-state-based detection over standard linear probes in the main in-distribution setting. We further evaluate PEP for pre-generation prediction, cross-model transfer, and out-of-distribution generalization. PEP remains effective in the pre-generation and cross-model settings, whereas robust cross-dataset transfer remains difficult. These results show that prompt-based adaptation can strengthen hidden-state probing while keeping the backbone frozen and adding only a small number of trainable parameters. 

---
# Evidence-RL: Towards Evidence-intensive Visual Reasoning 

**Authors**: Haojie Huang, Xinlei Yu, Chengming Xu, Zhangquan Chen, Cheng Yang, Qingdong He, Yu Yang, Jiangning Zhang, Xiaobin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08021)  

**Abstract**: Vision-Language Models (VLMs) should answer from concrete image evidence rather than language priors, dataset shortcuts, or irrelevant visual context. Existing perception-aware post-training methods encourage image use through global perturbations or attention proxies, but they do not test whether a sampled answer causally depends on the local evidence that supports it. We propose Counterfactual Evidence Disentanglement (CED), a training-time evidence audit for VLM grounding. For each response, CED neutralizes an object-centric Evidence Region and compares the resulting support drop against matched non-evidence Regions. We combine this signal with answer correctness inside GRPO, rewarding correct answers that rely on the evidence path rather than shortcut or nuisance paths. CED uses weak object-level proposals, requires no question-specific evidence annotations, and adds no inference-time overhead. Across nine public benchmarks and four backbones, CED outperforms prior RL-based post-training methods, with targeted analyses verifying its object-centric signal. 

---
# Ground-Truth Neighborhood Regularization for Reinforcement Learning Post-Training of Time Series Foundation Models 

**Authors**: Jianqi Zhang, Xingyu Zhang, Zeen Song, Changwen Zheng, Fanjiang Xu, Wenwen Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08010)  

**Abstract**: Time series forecasting (TSF) plays an important role in a wide range of real-world applications. Recently, time series foundation models (TSFMs), pretrained on large-scale datasets, have demonstrated strong generalization capabilities and emerged as an important paradigm for TSF. Reinforcement learning (RL) post-training has consequently attracted growing attention as a means of further improving their performance on downstream tasks. However, we find that, in certain forecast regions, RL post-training may gradually shift the output distributions of TSFMs away from the ground truth, thereby limiting their performance. We refer to this phenomenon as \textbf{suboptimal collapse}. Our analysis suggests that difficulty in initially sampling high-quality trajectories near the ground truth is an important contributing factor to suboptimal collapse. To address this issue, we propose Ground-Truth Neighborhood Regularization (GTN-R) for RL post-training of TSFMs. GTN-R uses the ground truth as a reference for locating high-quality regions and guides the model's probability mass toward the ground-truth neighborhood. This increases the probability of sampling high-quality trajectories, mitigates suboptimal collapse, and improves performance. Moreover, GTN-R can be flexibly integrated into various RL methods for TSFMs. Extensive experiments show its effectiveness. 

---
# Evidence-Grounded Forensic Reasoning for Detecting and Grounding Multi-Modal Media Manipulation 

**Authors**: Yichun Yeh, Yiheng Li, Xiaobo Hu, Zhen Lei, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08009)  

**Abstract**: Fake news increasingly relies on cross-modal image-text forgeries, making transparent and verifiable reasoning chains an urgent need for Detecting and Grounding Multi-Modal Media Manipulation (DGM4). Existing methods produce black-box detection results without any decision rationale, limiting their reliability in forensic practice. Multi-modal Large Language Models (MLLMs) offer a natural path toward explainability, but applying them to DGM4 raises two difficulties. First, models tend to generate explanations disconnected from predicted evidence locations, producing unverified attribution. Second, enforcing evidence-conclusion consistency requires active optimization, yet uniform training signals fail to distinguish localization tokens from classification tokens, making multi-head joint training unreliable. We propose a multi-modal manipulation detector based on an Evidence-Grounded Forensic Reasoning (EFR) framework. EFR introduces an Anchor-and-Verify reasoning chain that enforces modality-isolated perception before cross-modal comparison, with conclusion coordinates as explicit anchors to which downstream evidence must spatially correspond. A verifiable reward system then enforces evidence-conclusion consistency during training, while a Modality-Decoupled Advantage (MDA) routing mechanism mitigats credit misassignment across prediction tasks. Experiments show that EFR achieves state-of-the-art performance while producing structured forensic reasoning records that explicitly bind explanations to evidence. 

---
# Verication-driven closed-loop multi-agent large language modelframework for code-compliant structural design 

**Authors**: Jianbin Luo, Weibin Lin, Yiran Lin, Qing Wei, Wei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07978)  

**Abstract**: Multi-agent large language model(LLM)systems are applied to structural design,yet most use one-shot generation and cannot verify their output,leaving themill-suited to safety-critical this http URL than trusting LLM self-correction,thisframework injects feedback from an external physics-based verier into a closedrepair this http URL framework couples a three-layernite-element verication systemwith a dual-node this http URL 1 turns code violations into hard repair constraints,Node 2 turns a four-dimensional quality score into safety-rst soft constraints,and a retrieval-augmented code base makes every violation traceable to a this http URL structure types and 44 cases,code compliance rises from 56.8%to 98.6%and the composite score from 63.8 to 71.4(p<0.000001),using about 5.8%this http URL either node degrades performance,and compliance does notchange detectably across the two backbone LLMs tested,indicating that it ishere attributed to the external verier rather than the this http URL framework,the 44-case benchmark and all experiment scripts are released as open source forreplicability. 

---
# Thinking Hard, Not Smart: Reasoning Models Fail to Ration Test-Time Compute Across Questions 

**Authors**: Chenrui Fan, Yize Cheng, Ming Li, Yongyuan Liang, Tianyi Zhou, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2608.07968)  

**Abstract**: Reasoning language models increasingly use test-time compute to improve performance, but existing evaluations typically study this compute one question at a time. Yet when multiple problems share an end-to-end cost or latency constraint, models must decide how to divide limited inference compute among them. We introduce an exam-style evaluation framework for studying this setting, in which a model must distribute one shared token budget across questions with different difficulty and point values to maximize its total score. Across several open and frontier reasoning models, we find that models fail to allocate a shared budget strategically across questions of varying difficulties and values. Models behave largely as greedy sequential solvers: they prioritize questions by presentation order, front-load effort on early questions, and remain insensitive to value, with these tendencies becoming more pronounced as the number of questions grows. Explicit planning prompts spread compute more evenly but do not produce value- or difficulty-aware prioritization. The same behavioral pattern extends from mathematical to code reasoning. These findings establish global budget allocation as a distinct capability that is not captured by conventional per-question evaluation and remains a challenge for current reasoning models. 

---
# EasyBalance: Cross-Layer Load Balancing in Distributed MoE Inference 

**Authors**: Yize Wu, Ke Gao, Ling Li, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07964)  

**Abstract**: Load Balancing has emerged as a critical problem in expert-parallel distributed inference of Mixture-of-Experts (MoE) models. As routing distributions are typically skewed across experts, devices hosting lighter-loaded experts must idle to wait for the heaviest during expert computing, leading to inefficiency. Existing load-balancing approaches primarily rely on expert replication or migration within each layer, which introduce additional overhead and limit their flexibility and scalability. To address this problem, we propose EasyBalance, a cross-layer load balancing strategy that requires no modifications to the expert-device mapping, enabling instant adaptability and incurring essentially no additional overhead. Our key insights are that (1) experts of other layers can be viewed as naturally redundant for the current layer, and (2) cross-layer MoE workloads can be jointly executed to mitigate their individual imbalance. Based on these observations, EasyBalance greedily schedules a subset of cross-layer workloads to run at each MoE step and defers the remaining workloads for future balancing opportunities, effectively leveraging cross-layer imbalance mitigation. Extensive experiments across models, tasks, and configurations demonstrate that EasyBalance consistently accelerates distributed MoE inference, reducing GPU idling by mostly over 40%. Code is available at this https URL. 

---
# Persistent Semantic Entities in Tool-Augmented LLM Systems 

**Authors**: Zhaohui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07952)  

**Abstract**: Tool-augmented LLM agents can harbor implicit state that persists across sessions, activates through events, and propagates across agent boundaries---largely invisible to standard debugging. We formalize this as Persistent Semantic Entities (PSEs): constructs defined by name binding, event triggering, and cross-boundary propagation, and evaluate them across 24 models from 11 families (1.5B--1T parameters). First, every tested model is susceptible (20--100% on the 20-model susceptibility panel), with name binding as the necessary and dominant mechanism: without it, contamination is 0%. Second, persistence depends on contamination type rather than scale or deployment: preference contamination persists undecayed on every model probed (100% at t=10) and instruction contamination persists wherever adopted, persona-style injection decays partially (90%$\to$10%), while factual injection is model-dependent---self-corrected on Llama-3.1-8B and GPT-4o-mini but held at ceiling on both Qwen2.5-coder variants, so we do not claim it self-corrects in general. The preference and instruction results hold across providers in our controlled setting. Third, context-isolated self-verification achieves 20--79% reduction (median 36.5%) without oracle references while keyword-based detection produces systematic false positives, and contamination compounds 1.9$\times$ along a four-stage agent pipeline (40%$\to$75%). Preference and instruction contamination---persistent, lacking self-correction, and poorly captured by standard monitoring---represent a particularly concerning attack surface for deployed agent systems. 

---
# Metadata Reconstruction from Values Alone: Recovering Column Semantics in Undocumented Warehouses 

**Authors**: Mike Helwig  

**Link**: [PDF](https://arxiv.org/pdf/2608.07946)  

**Abstract**: Text-to-SQL benchmarks ship schemas whose column names already say what the columns mean. Production warehouses are the inverse: cryptic identifiers, partial or absent documentation. We address the problem they pose first: recovering what columns and values mean from the data itself.
Rosetta places a language model inside a verification harness: a deterministic profiler extracts structural evidence (value fingerprints, a 26-pattern library, checksum verdicts), the model proposes semantics conditioned on that evidence, and every fact carries provenance and a confidence bounded by its evidence class.
Against human documentation on 680 paired columns across eleven BIRD databases, identifiers destroyed, the harness delivers metadata that is 0.475 accurate on the 42% of columns it commits to, against 0.223 on 94% for the same model used directly. Restricted to the 283 columns where both arms speak, the harness writes no better prose than the model alone; the gain is selection: deterministic evidence governs whether the system speaks (coverage +0.257 [0.128, 0.378]), not how well. The deterministic layer is a competence detector, not a competence amplifier. A backbone swap bounds the claim: the prose finding reproduces, but prompt-requested abstention does not transfer; a code-enforced commit gate (predictions registered first; measured on a third backbone and held-out databases) makes no-evidence coverage 0.000 on every backbone. On a blind i2b2 clinical warehouse Rosetta decodes 95.5% of 134 real ICD-9 codes from values alone and abstains on all 44 NDC drug codes. The catalog supports calibrated abstention at query time: under full schema opacity a naive translator falls from 0.92 to 0.42 execution accuracy while our gate answers at 86% accuracy over 59% coverage. Negative results are reported plainly, including that our own authority ladder is not the mechanism behind the headline. 

---
# ScaleSense: Cost-Intelligent Scaling Framework via Learned Resource Estimation in Alibaba AnalyticDB 

**Authors**: Yifan Wu, Yuhan Li, Zhenhua Wang, Ke Chen, Lidan Shou, Zonghao Chen, Liang Lin, Huan Li, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07945)  

**Abstract**: Cloud-native serverless data warehouses achieve fine-grained elasticity by decoupling storage from compute, yet determining the optimal resource allocation for highly heterogeneous ad-hoc queries remains a formidable industrial challenge. Our analysis of production workloads in Alibaba AnalyticDB exposes a costly ``provisioning trap'': the fear of catastrophic resource depletion drives users to blindly over-provision resources, wasting immense monetary budgets without alleviating non-CPU bottlenecks (e.g., I/O saturation). To break this impasse, we propose ScaleSense, a proactive, query-level resource scaling framework. Specifically, it features a multi-faceted query encoder that jointly models plan topologies and hardware specifications. Crucially, a quantile-based resource predictor estimates multi-dimensional physical footprints, acting as a reliable safety net for optimal resource scaling. An auto-scaling controller then navigates the performance-cost Pareto frontier, dynamically tailoring allocations to specific business priorities without requiring model retraining. Evaluations on over 1.36 million production queries show that ScaleSense achieves state-of-the-art prediction accuracy with good prediction interval coverage. By achieving a 76.7% relative improvement in optimal resource configuration selection over the best baseline, this approach addresses the critical performance-cost trade-off while maintaining low-overhead inference latency, confirming its practical performance in production deployments. Under the performance-optimization policy, ScaleSense satisfies user-defined performance requirements while reducing monetary cost by up to 5.22x. 

---
# Second Order Drifting Models 

**Authors**: Drake Brown, Yuhao Huang, Shih-Hsin Wang, Bao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07924)  

**Abstract**: Drifting models are a recent class of one-step generative models that evolve the model distribution during training using a predefined sample-based drift field. Although they avoid iterative inference, their kernel-based drift fields induce frequency-dependent training dynamics: In the linearized regime, each Fourier mode of the density residual decays at a rate determined by the kernel spectrum, leading to slow recovery of fine-scale structure. We propose Second-Order Drifting Models, which lift drifting dynamics into phase space by augmenting generated samples with artificial velocity variables. We show that the resulting density perturbations obey accelerated second-order dynamics in Fourier space, connecting drifting models to the celebrated Nesterov acceleration from optimization theory. This provides a principled mechanism for mitigating the spectral stiffness of first-order drifting while preserving one-step inference. We derive a practical semi-implicit training algorithm and evaluate it on synthetic distribution matching, sequential data generation, and robotic control. Across these settings, the second-order drifting model improves convergence behavior and achieves competitive or superior performance over first-order drifting baselines. 

---
# Spectral Outliers Reveal Dominant Learned Structure in Transformer Attention 

**Authors**: Kasun Dewage, Marianna Pensky, Suranadi De Silva, T. H. Bandara  

**Link**: [PDF](https://arxiv.org/pdf/2608.07921)  

**Abstract**: We apply Marchenko-Pastur (MP) random matrix theory to pre-trained attention weights in order to separate each projection matrix into a random-like bulk and a set of spectral outliers. We validate this decomposition causally: zeroing the MP-identified outliers (signal) in Mistral-7B drives HellaSwag, MMLU, and PIQA close to random-chance performance, whereas zeroing a count-matched subset of bulk singular values causes smaller but non-negligible degradation. Across 11 pre-trained transformers we identify five recurring patterns: spectral outliers encode a dominant component of the learned structure; Q projections carry the most outliers; V projections under grouped-query attention lack a clean signal/noise separation; entry-level outliers form structured row-bands in Q and column-bands in O; and specific residual-stream dimensions persist as band outliers across layers in K and O. We close by outlining how these observations could inform parameter-efficient fine-tuning and structured pruning. 

---
# Private Anytime Selective-Risk Certification for Federated Retrieval-Augmented Generation: Guarantees and Empirical Limits 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2608.07913)  

**Abstract**: Selective-risk certificates promise that accepted outputs meet a declared error target. We develop Fed-SRC, a score-agnostic certificate for federated, differentially private, adaptively monitored retrieval-augmented generation. Clients release only Gaussian-perturbed score and loss histograms. Record-indexed and noise-variance-indexed martingales jointly bound target-risk contrast and accepted mass over all registered thresholds and rounds, permitting predictable recruitment, dropout, threshold selection, and optional stopping. A range-one total-variation term transfers the calibration mixture to a declared deployment mixture. The contribution is this private, federated, anytime combination, rather than the contrast statistic or acceptance floor individually. Empirically, no simultaneous-bound violation occurs in any evaluated cell, privacy level, or policy. Operational power depends on the score and population: the primary target r*=0.10 never certifies, and on RAGTruth the secondary target r*=0.20 never certifies either, whereas on HaluEval question answering it certifies in all 200 non-private trials, with held-out risk below the target. Naively privatized non-private certificates violate their bounds in 146 to 198 of 200 trials. As an exploratory comparison, we also evaluate a private betting-capital heuristic for which we do not establish e-process validity. This heuristic stops certifying at epsilon <= 4, where Fed-SRC still certifies. Certification nevertheless consumes roughly 30 times more stream events than unique calibration items. 

---
# Beyond "I Can't Help With That": How Child Safety Experts Evaluate AI Chatbot Safety 

**Authors**: Hannah Cha, Neha Shukla, Solon Barocas, Alexandra Chouldechova, Eugenia Kim, Jennifer Wortman Vaughan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07902)  

**Abstract**: Youth increasingly turn to AI chatbots for social and emotional support, raising concerns about how these systems respond, especially in high-stakes situations. However, existing child safety evaluations of AI lack grounding in real-world harms that youth experience, rely on unvalidated assumptions about what counts as an appropriate output (e.g., refusal), and typically focus on detecting adversarial prompts or surface-level harms in outputs only. Thus, these evaluations can fail to detect responses that pose harm to youth in practice. To better understand the limitations of current evaluation practices, we conducted interviews with 19 practitioners working directly with youth in vulnerable situations, including social workers, therapists, and psychologists, asking them to reflect on chatbots' responses to risky situations commonly faced by youth, as established in prior empirical work. Practitioners identified chatbot behaviors likely to cause harm as well as those that could meaningfully support youth in difficult moments, discussed the role that chatbots should (and should not) play in these interactions, and offered concrete recommendations for improving chatbot responses. Based on these findings, we provide recommendations for AI child safety evaluation and infrastructure, and highlight the need for incorporating practitioners' perspectives into safety work. 

---
# Router Sensitivity Under Lightweight Fine-Tuning Identifies Prunable Experts in Mixture-of-Experts Models 

**Authors**: Ali Janati, Kaoutar El Maghraoui, Xinyi Luo, Wenyuan Shen, Owen Zou, Yankai Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.07890)  

**Abstract**: Mixture-of-Experts (MoE) models decouple total parameters from per-token compute, but deployment still requires storing every expert. Recent theory shows that pruning experts with the smallest router-norm changes during fine-tuning can preserve accuracy, but assumes full fine-tuning. We test whether lightweight adaptation can recover this signal. We briefly fine-tune with a parameter-efficient adapter, rank experts by the induced $\ell_2$ router change, and prune the least-changed experts in one shot. On Mixtral-8$\times$7B-Instruct (44.83% MMLU-Pro), router-only LoRA trains 0.002% of parameters and outperforms all-module LoRA at matched rank with half the experts removed (27.54% vs. 24.42%); signal quality declines as adaptation spreads to attention and expert weights. Accuracy improves monotonically with LoRA rank, reaching 28.76%. IA3, which leaves router weights frozen, matches direct router adaptation, whereas unconstrained additive adapters degrade the signal. Router-guided MMLU-Pro accuracy decays quasi-linearly rather than collapsing, remains nearly 1.8 times that of magnitude-based or random pruning at maximal compression, and reduces memory by 49% and per-token latency by 37%. At 25% compression, retention is competitive with methods using full activation statistics. The criterion also transfers to Qwen1.5-MoE fine-tuned for mathematics, retaining 49.7% mean accuracy over eleven benchmarks with half the experts removed while random pruning falls to single digits. Router sensitivity under lightweight fine-tuning therefore makes provably motivated expert pruning practical at scale. 

---
# Vision-Language Grounding as Bidirectional Concept Correspondence 

**Authors**: Jieyu Zhang, Ziqi Gao, Luke Zettlemoyer, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2608.07886)  

**Abstract**: Vision-language grounding connects language to visual content, yet most existing formulations reduce grounding to a unidirectional localization problem: given a prespecified text phrase or category name, identify the corresponding image region. This setup assumes that the relevant linguistic unit is already known, overlooking a more basic challenge in grounded communication: determining which parts of the text are visually referential and how they correspond to entities in the image. We formulate grounding as $\textit{bidirectional concept correspondence}$ over an image-text pair. Given an image and its paired text, the goal is to recover all correspondences between visually referential text spans and instance-level image segments, without assuming that the relevant text spans are provided. This formulation unifies common grounding tasks, including phrase grounding, referring expression grounding, and open-vocabulary detection, by treating text segmentation, image segmentation, and cross-modal alignment as a single correspondence prediction problem. To address this task, we introduce $\textbf{ConCor-1}$, a grounding model built on top of a pretrained vision-language model. It uses learnable $\textit{bridge tokens}$ to represent candidate image-text correspondences and predicts, for each token, a text mask, an image mask, and a correspondence presence score. To train and evaluate this task, we convert diverse grounding and segmentation datasets into a unified correspondence format. Experiments show that $\textbf{ConCor-1}$ consistently outperforms baselines, improving correspondence F1 by 48% on the long-caption dataset and by 29% on zero-shot LVIS, where the large category list serves as the text input. 

---
# Distilling CT Foundation Models into Editable Concept Bottlenecks for Lung Nodule Malignancy Prediction 

**Authors**: Fakrul Islam Tushar, Stephen Adamo, Geoffrey D. Rubin  

**Link**: [PDF](https://arxiv.org/pdf/2608.07857)  

**Abstract**: Foundation models provide transferable CT representations, but predictions based directly on these embeddings are difficult to interpret. We developed concept bottleneck models that map two frozen CT foundation-model representations to eight radiologist-defined pulmonary-nodule attributes and predict malignancy from the estimated concepts and nodule size. The models included CT-FM, a whole-CT self-supervised encoder using a 96^3-voxel nodule-centered patch, and FMCIB, a nodule-focused contrastive encoder using a 50-mm crop. Eight ridge-regression concept heads were trained on 2,610 LIDC-IDRI nodules. Malignancy models were trained on LUNA25 and evaluated on a held-out internal test set and the external DLCS cohort. Concept fidelity was assessed using five-fold cross-validated R^2, and malignancy discrimination was assessed using AUROC with 95% confidence intervals estimated by patient-grouped bootstrap resampling. Concept fidelity was modest but higher for FMCIB than CT-FM for subtlety (R2, 0.24 vs. 0.11), spiculation (0.17 vs. 0.08), texture (0.17 vs. 0.07), and lobulation (0.15 vs. 0.05). Internally, the CT-FM and FMCIB concept+size models achieved AUROCs of 0.86 (95% CI, 0.80-0.92) and 0.86 (0.79-0.92), respectively. Externally, AUROCs were 0.72 (0.68-0.75) and 0.73 (0.70-0.76), compared with 0.73 for nodule size alone and 0.60 and 0.67 for the corresponding embedding only probes. Additive predictions could be decomposed into feature-level contributions and modified through controlled concept interventions. Concept bottlenecks provided transparent malignancy predictions with discrimination similar to nodule size alone, while differences in concept fidelity suggest that concept recovery depends on the underlying foundation-model representation. 

---
# Shape Mutating Expert Compression:LorExperts and BTExperts 

**Authors**: Inesh Chakrabarti, Sourjya Roy, Bowen Bao, Thiago Crepaldi, Spandan Tiwari, Ashish Sirasao  

**Link**: [PDF](https://arxiv.org/pdf/2608.07814)  

**Abstract**: Mixture-of-Experts (MoE) language models deliver high capacity at low per-token compute, but deploying them cheaply requires compressing their many expert weight matrices. Expert pruning (e.g., REAP) and merging reduce cost but sacrifice accuracy and require retraining the router; low-rank delta decomposition of experts (e.g., D^2-MoE) preserves all experts and the router, but degrades sharply as the expert count grows because a single shared component cannot approximate many near-orthogonal experts.
Because MoE expert weights are near-orthogonal, a single shared component (as in prior delta decomposition) scales poorly with the expert count; we show that experts nonetheless organize into functional co-activation communities that are decoupled from weight similarity. Building on this, we introduce LorExperts, a router-preserving compression method that clusters experts, keeps one full-precision dominant per cluster, and represents the remaining members as low-rank corrections to their local dominant. LorExperts retains all experts and the original router (no router retraining). At ~50% expert compression on Qwen3-30B-A3B and Gemma-4-26B-A4B, LorExperts preserves downstream accuracy and perplexity better than the baselines on most of the tasks; the margin over D^2-MoE grows with expert count E. We further give a reconstruction fine-tuning procedure for LorExperts, and BTExperts, a tree organization of dominants and corrections that enables inference-time amortization of shared computation. 

---
# The Anatomy of a Prompt Injection: A Component Model for Structured Analysis 

**Authors**: Jeremy McHugh  

**Link**: [PDF](https://arxiv.org/pdf/2608.07808)  

**Abstract**: Four years after prompt injection was first identified in 2022, attacks are still predominantly documented as verbatim strings rather than structured exploits, despite advancing agent capabilities and threat actors embedding injections to subvert AI-assisted security analysis. This paper formalizes the structure of prompt-injection artifacts, enabling defenders, red teamers, and cyber threat intelligence (CTI) teams to label, compare, and mutate attacks without relying on fragile string matching. Because large language models compile varied natural-language realizations into identical executable actions, labeling must track attacker intent (tool targets, sinks, and effects) rather than surface wording. We propose a seven-component model (carrier, delivery vector, concealment, context-break, privilege escalation, payload, and return channel) consisting of five artifact fields and two environment fields. This framework unifies roles partially addressed by HOUYI's payload decomposition, the Promptware Kill Chain, and campaign taxonomies, while framing minimal jailbreak frameworks like ReNeLLM as projections onto a restricted subspace. We provide clear labeling rules, a logical analysis record mapping directly to industry CTI schemas, worked examples including EchoLeak (CVE-2025-32711) and an in-the-wild malware AI-evasion sample, and an illustrative agentic flowchart. 

---
# CFD-Guided Detection of Concept Drift in Multimodal Physiologic Signals 

**Authors**: Farouk Ganiyu Adewumi, Timothy Oladunni, Rochak Ghimire, Kosisochukwu Ogbuanya, Sanaa Reeves, Sandy Akoy  

**Link**: [PDF](https://arxiv.org/pdf/2608.07759)  

**Abstract**: Cardiovascular AI models can classify clean elec- trocardiogram (ECG) signals, but real wearable signals change because of motion, breathing, posture, sensor contact, and true clinical deterioration. This paper asks when a model should keep its prediction, change it, or flag uncertainty. We propose a physiologic stability framework, called PECS, that compares changes inside the model with measurable changes in the signal. ECG is treated as the main cardiac signal, photoplethysmography (PPG) adds pulse and vascular information, and respiration is used only when ECG and PPG disagree. We test the framework on PTB-XL at pilot and full scales and on synchronized BIDMC and MIMIC waveform cohorts. The PTB-XL pilot and full- scale analyses selected different domain pairs, and the strongest cross-modal pair also changed across BIDMC and MIMIC, showing that adding every available signal is not always the best choice. PECS outperformed the evaluated drift-detection baseline implementations, reaching drift classification accuracy (DCA) of 0.8786 on expanded BIDMC and 0.9560 on MIMIC. The MIMIC results also showed that respiration can help during disagreement cases, but it should be used selectively rather than as an automatic override. Overall, the results support PECS as a candidate monitoring framework for wearable cardiovascular AI while highlighting the need for scale-aware domain selection and interpretable trust routing 

---
# Multi-Task Consistency-based Detection of Adversarial Attacks 

**Authors**: Cong Chen, Jean-Philippe Monteuuis, Jonathan Petit  

**Link**: [PDF](https://arxiv.org/pdf/2608.07750)  

**Abstract**: Deep Neural Networks (DNNs) have found successful deployment in numerous vision perception systems. However, their susceptibility to adversarial attacks has prompted concerns regarding their practical applications, specifically in the context of autonomous driving. Existing defenses often suffer from cost inefficiency, rendering their deployment impractical for resource-constrained applications. In this work, we propose an efficient and effective adversarial attack detection scheme leveraging the multi-task perception within a complex vision system. Adversarial perturbations are detected by the inconsistencies between the inference outputs of multiple vision tasks, e.g., object detection and instance segmentation. To this end, we developed a consistency score metric to measure the inconsistency between vision tasks. Next, we designed an approach to select the best model pairs for detecting inconsistencies effectively. Finally, we evaluated our defense against PGD attacks across multiple vision models on the BDD100k validation dataset. The experimental results demonstrated that our defense achieved a ROC-AUC performance of 99.9% detection within the considered attacker model. 

---
# LoRSA: Toward Generalizable Parameter-Efficient Fine-Tuning for Biomedical Downstream Tasks 

**Authors**: Saed Moradi, Benyamin Ghojogh, M. Hadi Sepanj, Yimin Yang, Ashirbani Saha  

**Link**: [PDF](https://arxiv.org/pdf/2608.07749)  

**Abstract**: Parameter-efficient fine-tuning enables the adaptation of vision foundation models to biomedical tasks under limited computational resources, but a single low-rank update can constrain all task-specific changes to one narrow parameter subspace. This restriction may prevent the model from simultaneously representing globally shared task structure and localized residual directions required for generalization to unseen imaging domains. We introduce LoRSA, a global--residual adaptation framework that jointly learns a dense low-rank component and a dynamically structured-sparse low-rank component. The dense component captures globally coordinated task adaptation, while the structured component provides complementary residual corrections whose support evolves during training. We characterize the representational capacity, approximation properties, rank structure, and singular-subspace complementarity of this decomposition. We evaluate LoRSA for four-class breast-density classification using DINOv3-Base, with VinDr-Mammo as the source domain and MammosighTR and RSNA as unseen external domains. LoRSA remains competitive on the internal validation set and achieves the best external macro-F1 on both target datasets, improving upon the strongest competing method by 2.15 percentage points on MammosighTR and 3.09 percentage points on RSNA. Weight-matrix analysis further shows that approximately $92\%$ of the energy of each adaptation component lies outside the bilateral singular subspace of the other, indicating that the two components learn largely complementary update directions. These results suggest that organizing adaptation capacity into distinct global and residual paths can improve the external-domain generalization of parameter-efficiently adapted biomedical vision models. 

---
# Complete, Scalable, and Robust Prioritized Planning for Multi-Robot Ordered Storage and Retrieval at Maximum Capacity 

**Authors**: William Zhang, Tzvika Geft, Jingjin Yu, Kostas Bekris  

**Link**: [PDF](https://arxiv.org/pdf/2608.07734)  

**Abstract**: Automated warehouses face a fundamental trade-off between maximizing storage density and achieving high retrieval throughput. While puzzle-based storage (PBS) architectures increase capacity by eliminating aisles, coordinating multiple robots in these high-density spaces is computationally challenging due to the potential for deadlocks. This paper introduces a novel multi-robot formulation for the ``ordered storage and retrieval problem at maximum capacity''. The focus is on rectangular grids accessible from a single boundary, where loads need to be first stored up to full capacity and then efficiently retrieved, given a planned departure sequence. This work bridges the gap between geometric feasibility and execution efficiency by leveraging the properties of relocation-free arrangements. These properties guide an online, prioritized multi-agent path-finding algorithm, which is the main contribution of this work. Unlike general centralized planners, the approach exploits the specific invariants of the storage arrangement to guarantee completeness and prevent deadlocks, enabling scalability. Experiments demonstrate that the method achieves near-linear improvement in makespan with respect to the number of robots, up to $m = C$, where $C$ is the grid width. Crucially, the algorithmic overhead of supporting robustness is negligible; the system handles uncertainty in departure sequences using robust storage arrangements with no significant penalty in execution speed compared to the non-robust baseline. 

---
# LGNNIC: Acceleration of Large-Scale GNN Training using SmartNICs 

**Authors**: Liad Gerstman, Aditya Dhakal, Dejan Milojicic, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2608.07733)  

**Abstract**: Graph Neural Networks (GNNs) are widely used across domains such as natural sciences, social network analysis, chip design, and recommendation systems. However, as graph sizes grow, storing and processing them entirely on a single-node CPU-GPU system becomes increasingly impractical. A promising approach is to distribute the graph across multiple remote memory nodes, though this introduces a major bottleneck: inter-node network congestion during training. To address this, we propose LGNNIC, a novel inter-node system architecture that leverages SmartNICs co-located with remote memory nodes-a configuration already available in modern systems-to reduce communication overhead in distributed GNN training. LGNNIC offloads key preprocessing tasks to SmartNICs, reducing the volume of data transferred to computational (training) nodes and alleviating network congestion. We introduce two complementary techniques executed on the SmartNICs during the preprocessing phase: Neighbor Sampling, which performs mini-batch sampling, and Quantization of the sampled batches. To evaluate LGNNIC under different communication infrastructures, we designed both an optimized low-overhead DMA-based synchronization mechanism and a high-overhead socket-based alternative used as a benchmark. We evaluate the core SmartNIC offloading mechanisms across standard GNN workloads and sampling hyperparameters using a proof-of-concept (PoC) system comprising one remote-memory node with an NVIDIA BlueField-2 SmartNIC and one compute node with an A100 GPU. Both Neighbor Sampling and Quantization on the remote node demonstrated substantial training speedups in most configurations. Neighbor Sampling achieved up to 62.4x and 17.5x speedups with Sockets and DOCA-DMA, respectively, primarily due to reduced data transaction time. Quantization provided additional speedups of up to 3.6x and 1.3x, respectively, by reducing data transfer. 

---
# SpikeWorld: Fast-State Adaptation for Frozen Spiking World Models 

**Authors**: Ziqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07712)  

**Abstract**: A predictive model receives a self-supervised signal whenever the consequence of an action is observed. Using that signal after deployment is difficult when dynamics and semantics share parameters: freezing prevents adaptation, whereas weight updates require optimizer state and may alter the learned representation. Here we introduce SpikeWorld, a 1.45M-parameter sparse spiking model jointly trained for heterogeneous sensory prediction, semantics, image-text binding and action-conditioned dynamics. At deployment, all trained parameters are frozen. Delayed next-state residuals update two external paths: cumulative fixed-bank losses select the bounded action correction, while route-specific residual matrices refine next-state prediction. Neither path uses labels, teacher outputs, rewards, success signals or the true shift value. Joint optimization improves action next-state MSE by 17.10\% while also improving multimodal prediction, semantic accuracy and image-text retrieval. On held-out shear and attenuation streams, the combined external state improves aggregate prediction by 5.48\% and 30.01\%; its fixed-bank action path improves tracking by 24.20\% and 3.94\%, respectively. In a six-arm study comprising 450 new Meta-World trajectories (75 per arm), SpikeWorld raises frozen-policy reward by 7.90 (95\% CI [2.48, 14.06]); the 13.33-point success difference is descriptive (CI [0, 40]). For identical sensory inputs, model parameters and inherited semantic outputs remain bitwise unchanged. A 16-byte RLS estimator obtains the highest non-oracle reward on linear attenuation, showing that the contribution is not superior linear identification, but its integration with a frozen multimodal spiking checkpoint. Reference code is publicly available at this https URL. 

---
# CosmosAlign: Adapting a World Foundation Model for Generative Traffic Video Forecasting 

**Authors**: Quang Minh Dinh, Tuan Kiet Doan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07693)  

**Abstract**: Generative traffic video forecasting aims to synthesize long-horizon, temporally coherent future videos of traffic scenes from a short observation history and textual descriptions. In this paper, we present CosmosAlign, a generative traffic video forecasting framework built upon the pretrained Cosmos3-Nano world foundation model. Our approach is motivated by the observation that successfully adapting large pretrained world models to downstream forecasting tasks depends primarily on distribution alignment rather than increased model capacity. To this end, we propose a two-stage LoRA adaptation strategy that first aligns the conditioning-mode distribution with the target forecasting task, and then aligns the training captions with the model's native structured prompting interface through an LLM-based re-captioning pipeline. During inference, we further improve prediction quality using a fully training-free procedure consisting of consensus-based medoid sample selection and motion-adaptive blending of static scene regions. CosmosAlign achieves a final score of 76.49 on the AI City Challenge 2026 Track 5 benchmark, ranking first on the final leaderboard. Our code is publicly available at this https URL. 

---
# Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding 

**Authors**: Yeeun Choi, Youngbeom Yoo, Joon-Young Lee, Hyolim Kang, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.07663)  

**Abstract**: When videos extend from hours to days, directly processing them end-to-end becomes impractical for current Multi-modal Large Language Models (MLLMs). This ultra-long setting necessitates a two-stage paradigm: query-agnostic memory construction followed by retrieval-based inference. Prior work invests in complex memory construction to pre-model high-level relations in videos, despite not knowing the downstream query at build time. We instead prioritize high-recall retrievability during memory building, and defer query-specific, high-level relation composition to inference time. To this end, we propose MERIT(Multi-key Episodic Retrieval with Inference-time Temporal expansion), a simple yet effective agentic framework for ultra-long video understanding. First, we formulate an episodic multi-key representation that enables precise retrieval of fine-grained memories through a simple key-matching mechanism. Second, we introduce a neighbor filtering mechanism to capture broader semantic context without the massive computational overhead of global memory construction. This is achieved by expanding the temporal scope exclusively around the retrieved segments at inference time. By leveraging simple key-matching with this on-demand temporal expansion, MERIT achieves state-of-the-art performance across three long-video benchmarks: EgoLifeQA, LVBench, and Video-MME (Long). 

---
# SkillConsist: Detecting Inconsistencies in Agent Skills via Bidirectional Graph Alignment 

**Authors**: Chaofan Meng, Yuhang Zheng, Yingnan Zhou, Sihan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07639)  

**Abstract**: Agent Skills provide reusable capabilities to LLM agents. Agent Skill inconsistencies can expose undisclosed dangerous behavior or cause wrong Skill selection. Recent Agent Skill research has increasingly examined Agent Skill consistency detection. Existing methods evaluate behaviors or security-property graphs against predefined categories or declared scopes. More recently, PL-HCL uses an LLM-based model to learn consistency across metadata, instructions, and resources. However, declaration and implementation behavior can be mixed across text and code, and a concise declaration can correspond to multiple connected implementation steps. We present SkillConsist to address both challenges. An LLM separates declaration and implementation content into behavior records on the implementation and declaration sides, while static analysis supplements implementation records. These records form declaration and implementation behavior graphs, respectively. Starting from a behavior record on either side, bidirectional graph alignment searches the other graph for a candidate subgraph and expands it along behavior relations until it completely expresses the source-side behavior. Graph differencing identifies conflicts between aligned subgraphs and outputs the detection results. We construct a 633-Skill benchmark from ClawHub's 500 most-downloaded public Skills and 133 Skill-Inject packages. The benchmark contains 319 inconsistent and 314 consistent Skills and 442 localized inconsistency annotations. On this benchmark, SkillConsist achieves 86.85% precision, 89.03% recall, and 87.93% F1 for package-level detection, improving F1 over the best baseline by 20.43 percentage points. For localization, it achieves 67.60% precision, 58.14% recall, and 62.52% F1. 

---
# Adversarial Attacks on Deep OCR Systems 

**Authors**: Wenbo Sun, Hongzong LI, Yanyun Wang, Jiahao MA, Shuxin Zhuang, Rong Feng, Shiqin Tang, Zi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07636)  

**Abstract**: Deep-OCR (DeepSeek-OCR) advances document recognition by treating the visual modality as an optical compression medium, enabling long-context OCR at low token cost. However, its increased complexity may introduce new security vulnerabilities. In this paper, we present, to the best of our knowledge, the first pure black-box adversarial attack against a generative OCR vision-language model, where only the decoded string can be queried and no gradients, logits, or model internals are available. We recast the attack as a zeroth-order optimization problem driven by a bounded scalar loss defined directly on the string output via sequence similarity, and estimate the gradient with a random-direction finite-difference scheme whose query cost is independent of the image dimension. An Adam update with ell_infinity projection yields imperceptible perturbations for both untargeted and targeted objectives. Pilot experiments on Deep-OCR validate the string-only attack and evaluation pipeline and expose severe qualitative decoder failures, including repetition, truncation, and prompt leakage. They also show that controlled targeted rewriting remains substantially harder than untargeted degradation; we avoid claiming targeted success until the pre-registered evaluation is complete. 

---
# PACE: A Playback-Aligned Context Engine for LLM-Based Full-Duplex Voice Dialogue 

**Authors**: Shibo Wang, Zicheng Zhang, Libo Wang, Junfeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.07631)  

**Abstract**: LLM-based full-duplex voice services allow users to speak while the assistant is responding. Because servers can generate output and advance dialogue state faster than clients can play it, subsequent user speech may be interpreted based on content the user never heard. We call this failure Generative Context Mis-anchoring (GCM). To address GCM issues, we present PACE, a provider-independent middleware layer that anchors model-facing context to the client playback boundary, a system-observable proxy for what the user could have heard. After an interruption, PACE repairs this context to exclude assistant content that never reached playback, while preserving low-latency generation across heterogeneous voice runtimes. We implement PACE's audio-only projection path end to end in a browser-based realtime voice assistant using a black-box speech model, without modifying the model service. We also construct GCM-Bench, a new controlled benchmark dataset of 108 playback-relative referent-anchoring cases. On GCM-Bench, PACE raises Referent Anchoring Accuracy from 25.0% to 96.3% over a cancellation-only baseline. On 200 Full-Duplex-Bench v1 interruption samples, it preserves interruption response quality. These results show that grounding model-facing context in actual playback is a practical way to maintain consistency in full-duplex voice dialogue. 

---
# Hit Selection Using SSMD-Based Machine Learning Performance Metrics in High-Throughput Screening Assays 

**Authors**: Xiaohua Douglas Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07609)  

**Abstract**: High-throughput screening (HTS) assays are central to early-stage drug discovery but are often limited by extreme data sparsity, as primary screens typically use only a single replicate per test substance. This sparsity makes conventional machine-learning performance metrics, such as sensitivity, specificity, and area under the receiver operating characteristic curve (AUROC), difficult to estimate empirically because they require adequately sized labeled samples. Here, we introduce a model-based framework that derives these classification metrics from the strictly standardized mean difference (SSMD), a well-established HTS effect-size parameter. Under a Gaussian equal-variance assumption, we derive closed-form relationships linking SSMD to Youden-optimal sensitivity and specificity, and sensitivity at a preset specificity, yielding explicit estimators and exact confidence intervals from the noncentral t-distribution, even under single-replicate designs. Unlike classical statistical power, which approaches 1 as sample size grows regardless of how small the true non-zero difference between group means is, the SSMD-derived sensitivity converges to a finite population value that reflects the true degree of separation between two groups, making it a more meaningful and stable performance measure for hit selection. We demonstrate the utility of this framework in a hepatitis C virus primary siRNA screen comprising approximately 22,000 single-replicate measurements, showing that SSMD, AUROC, and sensitivity-based thresholds yield equivalent and interpretable hit sets. This work bridges classical HTS statistics and machine-learning evaluation theory, providing a statistically principled, reproducible way to estimate classification performance in ultra-low-replication screening workflows. 

---
# Enhanced Real-Time 6-DOF Extended Reality Catheter Tracking for Evaluating Potential Improvement in Efficiency, Precision, and Depth Perception for Cardiac Interventions 

**Authors**: Mohsen Annabestani, Sandhya Sriram, Andrew Kuzemczak, S. Chiu Wong, Alexandros Sigaras, Bobak Mosadegh  

**Link**: [PDF](https://arxiv.org/pdf/2608.07606)  

**Abstract**: Despite advances in 3D ultrasound, most percutaneous cardiac interventions still rely on 2D visualization, limiting depth perception and spatial understanding. To address this challenge, we developed an Extended Reality (XR)-based platform that enables real-time six-degree-of-freedom (6-DOF) catheter tracking and visualization within a patient-specific 3D heart model. The system combines a custom machine-vision algorithm for 5-DOF catheter tracking with a 3D-printed electromechanical encoder that measures catheter roll, providing complete 6-DOF motion reconstruction. In a proof-of-concept study, 20 novice medical students navigated an intracardiac echocardiography (ICE) catheter to six anatomical targets using either immersive 3D visualization or a conventional 2D cathlab-style view. Participants in the 3D condition completed the task in 54.6 seconds and traveled 1,939 mm on average, compared with 267.5 seconds and 7,854 mm in the 2D condition. Therefore, the XR-based 3D system was more than 5x faster and required ~5x less catheter travel. The 3D mode also improved targeting precision and reduced performance variability. Participants consistently rated immersive visualization higher for accuracy, speed, usability, and clinical value. Kinematic analysis showed smoother depth-axis navigation in 3D, whereas 2D users relied on repeated corrective movements. These findings demonstrate that XR-based visualization can substantially improve procedural training efficiency, precision, and motor control. 

---
# Scaling Inherently Interpretable Language Models 

**Authors**: Guide Labs Team, Andreas Madsen, Aya Abdelsalam Ismail, Giang Nguyen, Isaac Plant, Muawiz Chaudhary, Nathaniel Monson, Saqib Azim, Zhichen Guo, Julius Adebayo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07594)  

**Abstract**: Interpretability is often treated as a tax on capability: language models are trained as opaque systems, then explained after the fact, with methods whose reliability is difficult to establish. In this work, we challenge this premise. Rather than reverse-engineering a model, we make interpretability a constraint of the training pipeline, optimized alongside the language modeling objective. Across three orders of magnitude of compute, on both autoregressive and diffusion language models, interpretability scales with capability rather than against it. Surprisingly, model representations become more disentangled and aligned with human-understandable concepts with scale.
We instantiate the training-time recipe with Steerling-8B, a diffusion language model with a causal attention mask. For any group of generated tokens, Steerling-8B attributes the output to relevant input tokens, human-understandable concepts, and training data. This enables closed-loop intervention: diagnose an output through its concept or feature attribution, retrieve similar training data, and correct the behavior through concept steering without retraining. Steerling-8B remains competitive with open peer models trained on substantially 2-16x more compute, suggesting a different scaling paradigm: interpretability can be designed into training, and it improves with scale. 

---
# Weather- and Location-Aware Agentic Dining Recommendation: Leveraging LLM World Knowledge for Region-Sensitive Contextual Reasoning 

**Authors**: Kadharmoideen Fadurudeen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07593)  

**Abstract**: Context-aware recommender systems have long recognized that factors such as location, time, and weather shape where and what people choose to eat. Existing weather-aware food and point-of-interest recommenders, however, typically treat weather generically -- mapping conditions to preferences through hand-crafted rules or specially trained context models -- and do not capture that the culturally appropriate response to weather is itself region-specific: a rainy evening calls for hot tea and fried snacks in one culinary culture and for very different comfort food in another. Encoding such weather-by-region-by-cuisine interactions as explicit rules or training data is brittle and does not scale. We present a weather- and location-aware agentic dining-recommendation system that takes a different approach: a large language model (LLM) orchestrates tools for location and weather retrieval and then reasons in natural language over the combined context, drawing on the cultural and culinary world knowledge already latent in the model to produce region-sensitive, weather-appropriate recommendations without per-region rule tables or specialized training. We describe the agent architecture, the tool-orchestration flow (Google location services and a weather service feeding an OpenAI LLM), and the reasoning mechanism, and we report on a working prototype that was implemented and briefly deployed end-to-end. We discuss design trade-offs -- cost, latency, ambiguity handling, and fallbacks -- and we are explicit about limitations, including the absence of a formal user study and the risk of cultural stereotyping in locality-based inference. The contribution is architectural: a simple, extensible pattern for incorporating environmental and cultural context into agentic recommendation through LLM reasoning rather than engineered rules. 

---
# Multi-Branch Policy Optimization for Multimodal Large Language Models 

**Authors**: Shuai Lyu, Yuning Gong, Ruiling Gao, Xiaoran Shang, Zhonghong Ou, Ping Zong, Yifan Zhu, Yuan Sun, Yang Qin, Peng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07581)  

**Abstract**: Group-based reinforcement learning methods for multimodal large language models typically rely on trajectory-level credit assignment that applies a single advantage to all tokens in a response. However, multimodal reasoning involves substantially higher perceptual uncertainty than text-only settings, where the model must repeatedly re-examine visual information to verify intermediate interpretations, and different visual groundings can lead to divergent reasoning paths, making such uniform credit assignment particularly inadequate and causing relative advantages to progressively degenerate toward zero. To address these challenges, we propose Multi-Branch Policy Optimization (MBPO), a tree-based framework that constructs reasoning trees at vision-language decision boundaries, enabling sibling branches to explore diverse visual hypotheses and assigning segment-level credit through branch-relative advantages. We further introduce a temporal replay buffer to reuse informative segments while controlling policy staleness. Experiments on several multimodal reasoning benchmarks show that MBPO outperforms representative baselines, improving both learning signal quality and optimization efficiency. The code is publicly available at this https URL. 

---
# Geometry Beats Estimated Depth: RGB-Only Multi-Camera 3D Tracking under Sim2Real 

**Authors**: Abdullah Naeem, Anav Katwal, Ayon Dey, Noman Khan, Md Tamjidul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2608.07579)  

**Abstract**: The AI City Challenge 2026 Track 1 evaluates multi-camera 3D perception in large indoor warehouses under a synthetic-to-real (Sim2Real) setting; depth is available only for training and validation, so inference is RGB-only. We use two RGB-only routes as a controlled test of one hypothesis: that cross-view geometric consistency, not monocular depth accuracy, governs performance under Sim2Real. The first is a geometry-first pipeline: YOLO11x detection, homography lifting to the world frame, class-level 3D size priors, multi-camera fusion, world-coordinate tracking, and offline tracklet stitching. The second is estimated-depth pseudo-LiDAR: monocular depth (D4RT, Metric3D~v2) back-projected into a fused point cloud and passed to a 3D detector (V-DETR), mirroring prior point-cloud winners that used depth. The gap is decisive: geometry-first reaches 13.0 3D HOTA (51.6 LocA), whereas pseudo-LiDAR collapses to 0.12 (9.2 LocA). We trace the collapse to cross-view inconsistency of monocular depth---scale correction is necessary but not sufficient---which domain-adaptation fine-tuning does not repair within budget. Within the geometry pipeline, offline stitching is the only intervention that helps; SAHI detection, appearance Re-ID, learned lifting, RT-DETR ensembling, test-time augmentation, and domain randomization all fail to beat the baseline detector. The bottlenecks are complementary: detection quality bounds the geometry route (DetA), localization consistency bounds pseudo-LiDAR (LocA). We release a complete, reproducible RGB-only pipeline and ablation. 

---
# Open-World Hierarchical Perception: Taxonomic Abstraction over Class-Agnostic Proposals for the Safe Handling of Out-of-Vocabulary Road Objects 

**Authors**: Felix Schaller  

**Link**: [PDF](https://arxiv.org/pdf/2608.07577)  

**Abstract**: A closed-set detector for autonomous driving must assign every object one of a fixed set of labels. On an object outside that set (a horse-drawn carriage, road debris, livestock on a rural road) it can only force a confident but wrong specific label or drop the object. Prior work in this series replaced the flat label set with a hierarchical taxonomy and a runtime abstraction rule, but evaluated it only on the boxes a closed detector already produces. This paper takes the layer open-world: we place taxonomic abstraction on top of class-agnostic region proposals so objects the closed detector never boxes can still be classified or flagged; we report a feasibility study of three open-world signals (class-agnostic segmentation, appearance-based out-of-distribution scoring, monocular depth) that shows why no single 2D cue suffices and how they compose; and we run the evaluation the earlier papers could not, a ground-truth leave-classes-out benchmark on real annotated objects. Holding out seven COCO classes and classifying their 235 ground-truth crops, a flat closed head emits a confident wrong specific label 100% of the time (37% of them in the wrong super-category, e.g. an animal named as a vehicle), whereas the hierarchical layer emits zero confident wrong specific labels and safely handles 94% of the objects (a correct super-category, or an explicit UNKNOWN OBSTACLE). We are explicit that this is a safety result, not a specificity one: the correct super-category is recovered only 26% of the time and the remaining 69% are conservatively flagged unknown. The contribution is an open-world perception layer that never makes a confident categorical mistake on an out-of-vocabulary object, together with an honest account of its cost. 

---
# BRACE: Taming Sharp Irregularities via Barycentric Rational Forecasting for Fast Diffusion Transformers Inference 

**Authors**: Jinlong Yang, Jinke Wu, Lizilin, Yao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.07572)  

**Abstract**: Diffusion Transformers (DiTs) have demonstrated exceptional performance in high-fidelity image and video generation. To alleviate their massive computational overhead, temporal feature caching has been proposed to bypass redundant computations. However, existing cache-then-forecast methods driven by derivative-based polynomials often cause severe quality degradation under high acceleration due to unstable long-step predictions. To address this bottleneck, we propose Barycentric Rational Forecasting with Chebyshev Enhancement (BRACE). Motivated by the observation that DiT feature trajectories are globally smooth yet frequently exhibit sharp irregularities and local non-smoothness, BRACE shifts the paradigm from derivative-driven polynomial extrapolation to feature-driven rational forecasting. Specifically, it maintains a local sliding window to cache sparse historical features and leverages adapted Chebyshev weights to formulate a barycentric rational function, directly aggregating these raw features to ensure numerical stability. Extensive experiments demonstrate that BRACE achieves state-of-the-art quality-efficiency trade-offs across various DiT architectures with negligible computational overhead. 

---
# COMEX: A Composition-Grounded Benchmark and Learning Framework for Explainable Aesthetic Image Cropping 

**Authors**: Rui Yang, Wei Zhou, Dingyong Gou, Xiaohui Cui, Cong Li, Yinyin Gong, Yipo Huang, Jiliang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.07570)  

**Abstract**: Explainable aesthetic image cropping requires not only localizing a visually pleasing crop but also explaining why it is preferred. Existing crop-and-explain methods largely treat explanation as post-hoc text generation and overlook composition, a key aesthetic factor that links crop decisions with interpretable reasoning. In this paper, we reformulate explainable aesthetic image cropping as a structured crop-composition-explanation problem. To support this setting, we introduce COMEX, a new benchmark built through image expansion and an IO-reversal pipeline. COMEX contains 33,161 quadruples, each consisting of an expanded image, a crop box, a composition category, and a composition-grounded explanation, enabling joint learning of crop localization, composition understanding, and explanation generation. We further propose a two-stage SFT+GRPO framework, where supervised fine-tuning establishes the structured output protocol and basic cropping ability, and GRPO further improves crop quality, composition prediction, and explanation faithfulness. We benchmark 15 large vision-language models and existing cropping methods on COMEX, establishing a comprehensive testbed for composition-grounded explainable aesthetic cropping. Experiments on both COMEX and prior benchmarks demonstrate the effectiveness and transferability of our framework, with strong performance across evaluation metrics. 

---
# Latent-Frequency Validity: Fast Spectral Editing with Screened Video-VAE Transfer Operators 

**Authors**: Bowen Xue, Jiafeng Xiong, Xin Quan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07569)  

**Abstract**: Direct spectral editing in video-VAE latents can control noise, flicker, smoothness, and frequency content without a decode--filter--reencode pass. However, video VAEs may redistribute pixel-space frequency bands across latent channels, and latent edits can disrupt VAE round-trip dynamics. We introduce \emph{latent-frequency validity} (LFV), which learns a compact VAE-specific spectral response and deploys it only when it improves decoded-target fidelity without worsening round-trip drift. LFV follows a validation-selected path from a diagonal per-frequency calibrator (C1) to full channel mixing (CM), making cross-channel capacity a controllable per-edit resource. Across 544 VAE--edit cells spanning six spectral families, LFV emits 423 cheap operators: 277 are handled by C1, while 146 (34.5\% of emitted operators) require channel mixing. On the primary 120-cell radial sweep, 99/100 emitted operators pass source-video-grouped held-out evaluation. Across five additional filter families, all 323 emitted operators pass held-out evaluation. Fully frozen OpenVid-fitted operators, including the validation-selected path coefficient, pass all 20 tested CogVideoX and HunyuanVideo generated-domain cells without adaptation. The selected response matches direct latent-filter latency and is about $3\times$ faster than pixel filter--reencode. The resulting maps reveal distinct VAE regimes, including strongly channel-coupled CogVideoX responses and a sharp Open-Sora high-band stability frontier. 

---
# Temporal Generalization in fNIRS-Based Autism Classification: A Cross-Time-Window Transfer Benchmark 

**Authors**: Marios Petrov, Sahana Vinayak, Targol Bakhtiarvand, Moses Smith Guddah, Adham Atyabi, Frederick Shic, Kevin A. Pelphrey  

**Link**: [PDF](https://arxiv.org/pdf/2608.07567)  

**Abstract**: Functional near-infrared spectroscopy (fNIRS) is a promising modality for autism spectrum disorder (ASD) classification, yet existing approaches assume temporally aligned evaluation. In practice, the optimal observation window varies across subjects due to differences in hemodynamic delay and neurovascular coupling, creating a temporal distribution shift that degrades performance. We formalize this as a \textit{cross-time-window transfer problem}, introducing a protocol that varies window length (2.5--10\,s) and offset within biological motion trials. Using topographic map representations of fNIRS recordings, we benchmark three vision architectures under two zero-shot baselines and eight adaptation strategies under leave-one-subject-out cross-validation ($N{=}124$). Key findings: (1) zero-shot cross-window accuracy is near chance (54--69\%); (2) ${\approx}5\%$ subject-specific fine-tuning recovers 90--96\%, while a subject-specific upper bound reaches 97--100\%, identifying inter-subject variability as the dominant barrier; (3) domain-adversarial and self-supervised strategies achieve 78--90\% without target-subject data; and (4) discriminative information is recoverable from windows as short as 2.5\,s. These findings provide a practical roadmap for deploying fNIRS-based ASD classifiers under realistic temporal variability. 

---
# What to Edit Next: Visually Aligned Image-Editing Follow-Up Suggestions in Conversational Systems 

**Authors**: Zhijing Zhang, Jinpeng Yu, Xin Song, Bingnan Li, Chuyue Li, Changhui Du, Xiaolin Fang, Jiaming Liu, Ruihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07565)  

**Abstract**: Conversational assistants increasingly recommend follow-up edits to help users continue a task. Existing systems primarily target text-only interactions, leaving image-creation conversations underexplored. In image-creation tasks, useful follow-up edit suggestions must reflect user preferences, offer diverse directions, and remain executable on the current image. We collected 100,000 real multi-turn image-creation conversation samples from Qwen App and found that 80.1% are image-dependent, underscoring the need for multimodal recommendation. We address this setting with a three-stage framework. In Stage 1, we use real online data to build a human-reviewed table of appropriate follow-up editing intents, then create SFT targets and fine-tune a multimodal policy. In Stage 2, to align rule-guided SFT suggestions with actual user choices, we use user click feedback to optimize the policy through multi-objective reinforcement learning. In Stage 3, to reduce visual inconsistencies between suggested edits and the current image, we introduce a visual verifier as additional training supervision. Extensive experiments demonstrate that our framework significantly outperforms baselines on both automatic and human evaluations. In a live user-randomized A/B test with millions of users, our final framework reduces visual inconsistency from 3.7% to 0.9%. Furthermore, it significantly improves recommendation CTR by 32.70%, image take-away rate by 16.32%, and average conversation turns per user by 39.90% (all p<0.05). 

---
# Coarse-to-Fine Registration of Jawbone CT and Intraoral Scan Data Using GeDi and ICP with Pseudo-IOS Ground Truth 

**Authors**: Sho Mitarai, Hikaru Kayo, Hisashi Ozaki, Yuichiro Imai, Megumi Nakao  

**Link**: [PDF](https://arxiv.org/pdf/2608.07564)  

**Abstract**: In digital dentistry and oral surgery, the registration of jawbone CT and intraoral scanner (IOS) data is essential for integrating internal bone structure with high-resolution dental surface geometry. However, this registration is challenging because the two modalities share only a limited region in common, and their true correspondence is generally unknown. This uncertainty has prevented rigorous quantitative evaluation of registration accuracy. In this study, we propose a pseudoIOS evaluation framework in which a point cloud emulating an intraoral scan is generated from CT data within the same coordinate frame, so that the transformation between them is known by construction and can serve as a true ground truth. Using this framework, we propose a coarse-to-fine registration method that combines a domain-generalizable local descriptor (GeDi) for initialization-independent global alignment with the iterative closest point (ICP) algorithm for local refinement, and also evaluate the influence of metal artifacts on registration. In experiments on seven cases, ICP alone frequently converged to local minima from a perturbed initial position, whereas GeDi$+$ICP maintained submillimeter mean absolute error (MAE) across all evaluated jaw and artifact conditions (0.55-0.69 mm). A three-way repeated-measures analysis confirmed that GeDi$+$ICP was significantly more accurate than GeDi alone. Metal artifacts had a statistically detectable overall effect, but their absolute impact on GeDi$+$ICP was small. 

---
# AeroDPO: Unleashing Lightweight UAV Navigation with High-Fidelity Perception and Automated Preference Optimization 

**Authors**: Peng Xu, Chengcheng Wang, Shaohua Wan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07557)  

**Abstract**: Vision-Language Navigation for Unmanned Aerial Vehicles (UAV-VLN) requires rapid and reactive control in complex 3D environments. Recent minimalist end-to-end paradigms show great promise but typically rely on massive language models containing billions of parameters, incurring prohibitive latency for real-world edge deployment. In this paper, we challenge this parameter-heavy reliance. Comprehensive cross-scale evaluations reveal the critical insight that perception quality fundamentally outweighs language reasoning capacity. We demonstrate that a lightweight 2B model equipped with high-fidelity visual inputs completely matches the overall success rates of massive 7B baselines. However, this minimalist policy exposes a fundamental robustness flaw inherent to pure Behavior Cloning (BC). Lacking explicit negative feedback, the agent fails to internalize robust spatial constraints and exhibits alarming collision rates in out-of-distribution (OOD) scenarios. To overcome this vulnerability without relying on unscalable human annotations, we propose AeroDPO, a zero-cost automated Direct Preference Optimization pipeline driven by deterministic physical simulation state rollback. Upon detecting collisions, the system autonomously rewinds the environment to extract causal reasoning errors as rejected actions, applies decoupled privileged interventions to synthesize collision-avoidance preferred maneuvers, and leverages an offline vision language inspector to filter visual ambiguities. By equipping our 2B model with this automated data flywheel, AeroDPO boosts success rates to 49.16% on unmapped scenarios while drastically suppressing collision rates, establishing a new SOTA for autonomous aerial agents. 

---
# MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures 

**Authors**: Zhuoning Xu, Xiucheng Zhang, Hanjun Luo, Yingbin Jin, Yinpeng Dong, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2608.07556)  

**Abstract**: Multi-agent systems (MAS) decompose long-horizon tasks across supervisors and subagents, but delegated goals do not necessarily carry their original authorization boundaries. Existing safety benchmarks mainly study adversarial compromise, while work on constraint drift lacks controlled architecture-level evaluation. We introduce MasDrift, a benchmark of 600 benign productivity tasks across eight domains. Each task pairs required work with reserved actions. MasDrift compares single-agent, centralized, and decentralized coordination while varying hierarchy depth and peer width, measuring task completion and authorization preservation. Across generic multi-agent conditions, centralized hierarchies achieve 93.9--98.6% task completion versus 85.7--87.0% for peer networks, while unauthorized actions occur in 2.7--19.8% of tasks versus 0.6--0.8%, a gap that widens with hierarchy depth. We further compare two defenses that differ in where authorization evidence resides. One re-anchors every pending call to the original user request. The other carries an attenuated policy along the delegation chain. Re-anchoring reduces unauthorized actions in every model configuration we evaluate, at a cost of 1.6 points of pooled completion. Chain propagation blocks required work instead, forfeiting up to 36.3 points. A heterogeneous case study confirms that the failure follows from coordination rather than model strength. MasDrift exposes a centralization tradeoff and makes authorization preservation a measurable property of MAS design. 

---
# P2Voxel: Pyramid Pivot Voxelization for 3D Mesh Tokenization 

**Authors**: Zhenhong Sun, Haozhe Liu, Yifu Wang, Xibin Song, Senbo Wang, Huadong Mo, Daoyi Dong, Hongdong Li, Pan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2608.07549)  

**Abstract**: Triangle meshes provide explicit and accurate surface geometry, yet their irregular topology connectivity makes 3D mesh tokenization a geometric sampling problem: how to sample and organize geometric evidence into compact, structured and learnable tokens. Beyond field-centric volumetric sampling and edge-intersection surface sampling, we retarget mesh tokenization as \textit{local surface evidence sampling}: identifying the minimal geometric evidence inside each active voxel that is sufficient for deterministic surface recovery. To this end, we introduce \textbf{P2Voxel}, a pyramid pivot voxelization framework for compact and reconstruction-aware mesh tokenization. P2Voxel is built on three key innovations. Under the \textit{Local Planarity} assumption, Pivot Voxelization represents each active voxel with a surface pivot and an orientation sign, providing minimal local evidence that can induce the corner values required for deterministic reconstruction. Under the \textit{Spatial Complexity} assumption, Pyramid Pivot Voxelization exploits the spatial non-uniformity of real surfaces by allocating finer pivot tokens to geometrically complex regions while keeping smooth regions coarse and compact. Under the \textit{Block Reconstructability} assumption, a Pyramid VAE learns compact multi-resolution latent codes over locally reconstructable pivot blocks, avoiding the need to model the entire high-resolution voxelized shape as a dense global field. Together, these designs convert meshes into compact, structured, and learnable pyramid pivot tokens, enabling efficient mesh reconstruction for downstream 3D tasks. 

---
# Learning an Interior Layout Policy in a Domain Specific Language Action Space 

**Authors**: Yuhao Lu, Weichen Zhang, Wenyi Xiao, Haohui Chen, Yiyun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2608.07547)  

**Abstract**: Indoor scene layout generation is a challenging task in interior design. Existing methods often oversimplify the task by reducing room conditions to coarse 3D bounding boxes and neglecting structural elements such as doors and windows. More fundamentally, many prior approaches formulate spatial reasoning as direct coordinate prediction, thereby casting interior layout design as continuous regression over raw geometric parameters, which hinders the model from learning the underlying reasoning logic of intelligent layout design. We propose \textbf{LayoutDSL}, a novel LLM-based framework for learning an interior layout policy in a domain-specific language (DSL) action space. The DSL provides an explicit symbolic representation of layout information and serves as a structured action space for layout reasoning, where each action corresponds to an interpretable design decision. Under this DSL-based policy learning paradigm, we construct 3D-FrontDSL, a dataset of room-structure annotations paired with synthetic DSL action sequences for supervised fine-tuning. To promote a more generalizable and scalable policy with verifiable feedback, we design rewards grounded in interior design principles and physical plausibility, and optimize the policy via reinforcement learning. Extensive experiments demonstrate that LayoutDSL substantially improves spatial plausibility and design logicality over strong baselines and existing methods. 

---
# Generalizing deep reinforcement learning across cable-driven parallel robot configurations with actuator-level policies 

**Authors**: Abir Bouaouda, Mohamed Boutayeb, François Charpillet, Dominique Martinez, Rémi Pannequin  

**Link**: [PDF](https://arxiv.org/pdf/2608.07546)  

**Abstract**: Cable-driven parallel robots (CDPRs) present diverse configurations and complex control challenges, which can be addressed by deep reinforcement learning (DRL) by learning their nonlinear dynamics. However, DRL methods often require extensive training time, and the resulting policies do not generalize well to different robot configurations or varying numbers of actuators. In this article, we introduce a novel DRL approach for controlling CDPRs that does not depend on the specific robot configuration. Our method trains an actuator-level policy that controls each motor to achieve its target cable length, in contrast to conventional DRL approaches that learn to control the entire robot to reach a desired end-effector position. To the best of our knowledge, this is the first work to apply DRL to control CDPRs using an actuator-level policy. This approach offers two main advantages: (i) a single shared policy can be applied to any CDPR configuration, regardless of actuator count, and (ii) reliance on inverse kinematics, avoiding the more challenging forward kinematics problem. Training is performed in simulation, and the learned policy is successfully transferred to a real CDPR. Experimental results show that the actuator-level policy (ALP) surpasses traditional reinforcement learning methods in both robustness and precision. We further control a real 8-motor CDPR with 3D motion using a policy trained on a simulated 4-motor planar CDPR operating in 2D. This illustrates that the proposed method is applicable to any CDPR configuration, independent of actuator number or placement. 

---
# DarwinX: Evolving Agent Harnesses Through Natural Selection 

**Authors**: Yifan Zhang, Yutong Dai, Juntao Tan, Luyu Yang, Rishi Mullur, Thai Hoang, Zhiyuan Hu, James Zhu, Phil Mui, Silvio Savarese, Ran Xu, Zeyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07545)  

**Abstract**: An LLM agent's capability depends not only on model weights but on its harness: prompts, tools, skills, and control flow. Self-improvement loops already edit harnesses, yet single-lineage search is path-dependent and local wins often regress other tasks. We introduce DarwinX, which treats self-evolution as selection over a population of harnesses with the model frozen: a preserve-and-extend contract admits only variants that extend coverage without regressing, an archive keeps alternative lineages for recombination, and failure-, teacher-, and self-derived evidence share one edit interface. Fitness comes from each benchmark's own verifier: no gold solutions, no hand-picked winners. Across four benchmarks that progressively separate the evolution signal from the test, one loop adds about 17 points on average: Terminal-Bench 2.1 rises +7.7 to 83.2% on a matched base and to the verified frontier at 84.7% on a stronger one; TerminalWorld's held-out split reaches 68.3%, ahead of every off-the-shelf agent; WebArena-Infinity real-task pass@1 rises from 43.5% to 93.0% audit-clean; and a Terminal-Bench 2.1 harness transfers unchanged to SWE-bench Verified. What evolves is general agent competence, not benchmark-specific patches, so it survives changes of task, verifier, and base model. A frozen model need not be a fixed agent: harness selection turns evaluation compute into durable capability. 

---
# MOSAIC: Adversarial Co-evolution of Specialist Heuristics and Problem Instances for LLM-based Automated Heuristic Design 

**Authors**: Oguzhan Gungordu, Siheng Xiong, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2608.07544)  

**Abstract**: Automated heuristic design (AHD) with large language models (LLMs) has produced strong heuristics for combinatorial optimization problems (COPs). Yet existing frameworks optimize for average performance on a small fixed dataset and steer the search with "verbal gradients" distilled from scalar better/worse feedback. No single heuristic dominates across instance distributions, and scalar feedback tells the LLM whether a heuristic improved, but not where in the instance space or why. We propose MOSAIC, a grid-based framework that adversarially co-evolves problem instances and specialist heuristics inside a Quality-Diversity (QD) archive indexed by structural instance features. Instances evolve to expose weaknesses of the current heuristics, and heuristics evolve to eliminate them by specializing to the newly exposed regions. Each archive cell keeps a specialist heuristic, representative instances, and insights explaining what works in its region, forming a persistent memory that accumulates over the evolutionary search. For each heuristic pair sampled from distant grid regions, an LLM-guided evolutionary loop generates discriminative instances, and a decision tree identifies the feature-space regions where each heuristic wins. A reflection LLM then contrasts the two heuristics to produce multi-directional insights that persist in those regions and guide crossover and mutation. The archive is simultaneously a co-evolved benchmark of discriminative instances and a pool of region specialist heuristics, from which greedy selection extracts a compact complementary portfolio. Across COPs, test sizes, and LLM backbones, the portfolio consistently outperforms state-of-the-art LLM-based AHD methods, and the co-evolved instances attain higher feature-space coverage and stronger heuristic discrimination than evolutionary instance-generation baselines. 

---
# Performance of large language models in the optical diagnosis of colorectal polyps 

**Authors**: Joshua C. Vences, William T. Tran, Nikko Gimpaya, Catharine M. Walsh, Rishad J. Khan, Robert Bechara, Asher C. Wiggins, Celine N. Rousan, Kaitlyn V.G.L. Morgado, Angie Ibrahim, Kevin H. M. Kuo, Daniel von Renteln, Alexander Hann, Dennis L. Shung, Michael A. Scaffidi, Charles Ménard, Joshua Landy, Samir C. Grover  

**Link**: [PDF](https://arxiv.org/pdf/2608.07543)  

**Abstract**: Background and Study Aims: Accurate optical diagnosis of colorectal polyps guides resection strategy and surveillance, with multimodal large language models (MLLMs) showing potential for image-based diagnosis. We aimed to evaluate the diagnostic accuracy of MLLMs in classifying colorectal polyps and predicting histology. Methods: We conducted a retrospective diagnostic performance study using the PRIME dataset, a curated set of white light and narrow-band imaging (NBI) images. We evaluated Claude Opus 4, Google Gemini 2.5 Pro, GPT-o3, GPT-4o, and GPT-5. For Paris, Narrow-band Imaging Colorectal Endoscopic (NICE), and predicted histology, we calculated F1 scores, percent correct scores, and accuracy of each MLLM compared to expert responses for 132 cases. Cochran's Q and McNemar's Test were used to determine differences between predicted values of each MLLM. Results: The F1 scores among MLLMs were >0.9 for all models for neoplastic vs. non-neoplastic polyps. Gemini 2.5 Pro demonstrated the highest F1 scores for invasive vs. non-invasive polyps and low- vs. high-grade adenoma, at 0.560 and 0.492 respectively. Claude Opus 4 and GPT-5 had statistically significantly higher percent correct scores than other MLLMs at 41.7%, using Paris classification. Conclusions: Claude Opus 4 and Gemini 2.5 Pro showed the highest accuracy in differentiating polyp subtypes, performing closest to expert consensus. Sensitivity and specificity, however, did not meet ESGE standards, highlighting the need for prospective multicenter trials and the design of human-in-the-loop workflows before clinical deployment. 

---
# NeuroPilot: An Agent-Driven Smart Pipeline for Processing, Quality Control, and Managing Neuroimages 

**Authors**: Yiyao Chen, Yucheng Li, Jungong Tong, Shaoqi Wang, Kunhao Zhou, Ziquan Wei, Monica Murea, Marissa DiPiero, Tingting Dan, Guorong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07541)  

**Abstract**: Transforming raw neuroimage archives into analysis-ready derivatives relies on three brittle stages: data standardization, modality-specific preprocessing, and quality control (QC). While individual neuroimaging tools are well developed, their orchestration requires project-specific scripts, environment-adaptive tuning, and labor-intensive manual QC. To address this, we introduce NeuroPilot, a multi-agent system that digitalizes the expertise of neuroimage processing, QC, and data management into three LLM-invocable skills: dcm2bids-skill, neuroimage-pre-skill, and qc-agent-skill. The LLM-driven agent autonomously orchestrates workflows, generalizing various infrastructure settings into a single configuration to achieve the highest scalability. Demonstrating the system's generalizability, we deployed NeuroPilot across 17 cohorts (>123,000 subjects) spanning infant to aging populations and multiple MRI modalities (structural, diffusion, functional). In practice, after standardizing data via the dcm2bids-skill, the agent dynamically routes datasets to the optimal neuroimage-pre-skill based on available modalities and cohort traits (e.g., dispatching T1w and fMRI data to fMRIPrep, or selecting specialized pipelines for infant cohorts). The qc-agent-skill then drives an evidence-based, semi-automated QC via a 3-D browser dashboard, utilizing a multi-tiered verification system to optimize failed cases and escalate complex issues for supervisor inspection. Quantitatively, our QC agent screened 558 production subjects, validating its automated flags against FreeSurfer's topology-defect metrics. The infant processing pipeline achieved a 100% (201/201) completion rate on QC-validated inputs. Importantly, NeuroPilot compresses the traditional 2--3 month timeline for training staff and processing complete datasets into a single week. NeuroPilot is deployed in this https URL. 

---
# An evolutionary model of animats with VLM-based subjective evaluation 

**Authors**: Shota Miyazaki, Takaya Arita, Reiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2608.07537)  

**Abstract**: In this study, we propose a framework that incorporates subjective evaluations provided by a Vision-Language Model (VLM) into the fitness evaluation and selection processes of a genetic algorithm. As the target of evolution, we employ virtual soft robots with flexible morphologies and locomotion and present the VLM with sequence images representing the locomotion of two individuals. Selection is performed via pairwise comparisons based on subjective evaluation terms such as adorably and weirdly. The outcomes of these comparisons are used as selection pressure within the genetic algorithm, enabling the simultaneous evolution of morphology and locomotion. Experimental results demonstrate that subjective selection by the VLM accelerates population convergence compared to random selection, while also giving rise to distinctive morphologies and motions corresponding to each evaluation term. An auxiliary experiment with human participants further showed that, although individual pairwise choices only partly agreed with the VLM selections, the resulting morphological and locomotion tendencies were qualitatively similar and repeated human evaluations imposed noticeable fatigue. Moreover, the observation that similar evolutionary outcomes emerged across different evaluation terms suggests that the VLM does not apply these terms in a purely literal manner but instead decomposes them into multiple internal evaluation criteria when making judgments. This work visualizes the evolutionary process through which subjective linguistic expressions are mapped onto embodied phenotypes and provides a foundational framework for analyzing the structure of subjective judgment in VLMs. The proposed approach is expected to contribute to new developments in evolutionary computation and artificial life research based on subjective evaluation. 

---
# Evolving Safety Landscape of Multi-modal Large Language Models: A Survey of Emerging Threats and Safeguards 

**Authors**: Xi Li, Shu Zhao, Xiaohan Zou, Fei Zhao, Fuxiao Liu, Yusen Zhang, Cheng Han, Yushun Dong, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07535)  

**Abstract**: Multi-modal large language models (MLLMs) integrate heterogeneous modalities through modality alignment and fusion, enabling stronger understanding and reasoning. However, this architectural shift reshapes the safety landscape of machine learning. Increased model complexity and cross-modal interactions give rise to novel threats, including compromised modality integration, modality misalignment, and fused safety risks, reflecting shifts in threat modeling beyond uni-modal assumptions. These shifts, in turn, impose new constraints on safety solutions not captured by existing frameworks rooted in uni-modal learning. Motivated by these challenges, this survey provides a systematic analysis of the evolving safety landscape of MLLMs. We first propose a multimodal grounded taxonomy of safety threats and analyze shifts in threat models, covering adversarial attacks, data poisoning, jailbreaks, and hallucinations. We then summarize updated safety assumptions and organize recent advances in MLLM safety strategies accordingly. Finally, we discuss open challenges and future directions to inform the development of more principled and scalable safety mechanisms for multimodal systems. 

---
# Ultraconstructive Model Theory via Bounded Adversarial Finite Structures 

**Authors**: Mirco A. Mannucci  

**Link**: [PDF](https://arxiv.org/pdf/2608.07534)  

**Abstract**: Ultraconstructive Model Theory (UCMT) replaces idealized satisfaction, at finite compu- tational scale, by bounded adversarial survival. A finite partial structure is tested by an Opponent (Devil) drawing legal challenges from a bounded attack surface, repaired by a Builder (God) through legal replies, and certified by a symbolic Judge 

---
# Search-G1: Grounded Search Agents via Representation-Based Intrinsic Rewards 

**Authors**: Cheng Ruoxi, Ma Haoxuan, Zhang Hongyi, Zhang Junming, Duan Ranjie, Xia Qiaolin, Wang Hao, Lu Yu, Shi Haibo, Ma Xingjun  

**Link**: [PDF](https://arxiv.org/pdf/2608.07531)  

**Abstract**: Search-augmented language agents should retrieve external information only when necessary and ground their answers in retrieved evidence. Existing external rewards provide either sparse outcome supervision or richer feedback from process annotations and LLM judges. Outcome rewards scale readily but cannot distinguish grounded retrieval from redundant search, whereas richer signals require costly annotation or inference during training. Internal rewards based on policy-side signals such as entropy, likelihood, or information gain are graded and inexpensive to evaluate, yet mainly reflect model confidence rather than evidence grounding. We propose Search-G1, a representation-based intrinsic reward framework that measures the operational grounding of an agent's answers through two intervention-calibrated readouts. A prompt-state readout predicts closed-book sufficiency, whose complement defines policy-relative retrieval necessity; an answer-commit readout estimates evidence reliance from answer-stage sensitivity to evidence deletion. Together, they provide additional credit to correct searched trajectories when retrieval is estimated necessary and the answer is evidence-sensitive, favor correct direct answers when closed-book knowledge suffices, and penalize repeated search. After calibration, reward scoring requires neither process annotations nor LLM-as-judge inference during policy optimization. Because reinforcement learning changes policy representations, Search-G1 periodically refits both readouts on trajectories from the latest checkpoint, allowing the reward to co-evolve with the policy. Experiments across multiple search-based question-answering benchmarks and two model scales show that Search-G1 improves the grounding--search-cost trade-off, producing shorter response-side trajectories at competitive task accuracy. Code is available at this https URL. 

---
# WuYuEval: A Multi-Level Benchmark for Large Language Models in Solid Waste Management 

**Authors**: Yi Zhang, Hongyang Wang, Zheng Hao Leong, Zihao Wu, Kaijun Lin, Zhixing Pan, Qixun Huangfu, Wei Ren, Wenyan Wu, Fangyun Wang, Wenting Yu, Hengyu Lin, Muling Yang, Zongguo Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07529)  

**Abstract**: Large language models (LLMs) are increasingly used as technical assistants, but their competence in solid waste management (SWM) remains difficult to assess because existing benchmarks emphasize general knowledge rather than professional decisions under engineering, environmental, and policy constraints. We introduce WuYuEval, a multi-level benchmark for evaluating LLMs in SWM across foundational knowledge, domain reasoning, and expert decision-making. After quality auditing, WuYuEval contains a Foundation Module with 4,590 closed-ended multiple-choice questions across six task types and eight domain categories, together with an Expert Module with 247 scenario-based open-ended questions involving multi-objective optimization, constraint trade-offs, and system design. For expert tasks, we combine anchor-calibrated LLM-as-a-Judge scoring with Elo-based pairwise comparison. Across 33 LLMs, performance varied widely. The leading model reached 94.64\% accuracy on the Foundation Module, but average accuracy still fell from 84.14\% on easy questions to 42.50\% on hard questions, with lower performance concentrated in calculation, experimental design, urban planning, and open-ended expert tasks. Reasoning-oriented Thinking modes improve most matched model pairs after auditing, but the gains depend on baseline capability and are not uniformly positive. These results suggest that visible deliberation helps only when it remains anchored to units, assumptions, and engineering constraints; otherwise, it may drift from decisive answer boundaries. WuYuEval therefore provides both an evaluation resource and an empirical basis for developing SWM-oriented foundation models with professional reasoning chains and explicit constraint control. 

---
# DocAtlas: Long-Document Understanding as Mutable-State Interaction 

**Authors**: Hongchen Wei, Yuanzhe Wang, Bei Liu, Yifan Yang, Qi Dai, Kai Qiu, Yunsheng Li, Dongdong Chen, Chong Luo, Zhenzhong Chen, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07527)  

**Abstract**: Long-document understanding requires models to find and combine evidence across many pages, layouts, tables, figures, and charts. Existing retrieval-augmented systems usually select evidence from a static index before generation, while recent agentic systems add multi-turn tool use but often rely on frozen proprietary backbones whose behavior is set by prompts. We present DocAtlas, a system that treats long-document understanding as a mutable-state information-seeking process. We instantiate DocAtlas as a mutable document harness: an external environment that determines what document information is searched, read, stored, reviewed, and shown to the model at each step. Given a document and question, the harness exposes search, reading, note-taking, and review tools, maintains a hierarchical tree and note store, and updates both as the agent records evidence. DocAtlas combines self-improving retrieval, selective evidence access, and active working memory under a fixed context budget. The same harness supports inference-time use with large VLMs and end-to-end reinforcement learning for compact VLM agents. With GPT-5.4, DocAtlas reaches 71.4\% on MMLongBench-Doc, exceeding the human-expert reference of 65.8\%. A Qwen3.5-4B VLM trained with end-to-end RL in the DocAtlas environment reaches 63.7\%, compared with a 54.4\% direct-input baseline, showing that mutable document-harness design can improve compact document agents by a large margin. 

---
# Unified Hallucination Fuzzing for Multimodal Large Language Models 

**Authors**: Pengfei Zhou, Jiajun Song, Zhiwei Tang, Yixing Ma, Xiaopeng Peng, Donghui Si, Yuhang Xu, Huiqi Song, Yiyuan Miao, Yichen Qian, Weihua Chen, Wangbo Zhao, Bohan Zhuang, Jiasheng Tang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2608.07525)  

**Abstract**: Hallucination remains a persistent challenge for Multimodal Large Language Models (MLLMs), severely limiting their reliability in high-stakes applications. Existing evaluations, predominantly based on static benchmarks, suffer from narrow taxonomical coverage and rapid performance saturation, failing to reflect model robustness in evolving real-world scenarios. To bridge this gap, we present a systematic evaluation framework integrating a comprehensive benchmark with self-evolving stress testing. First, we introduce UniHall, a fine-grained dataset grounded in a unified taxonomy spanning Object, Instruction, and Knowledge dimensions. Second, to address benchmark saturation, we propose Self-Adaptive Multimodal Fuzzing (SAMF), a self-adaptive framework that employs evolutionary mutation strategies to explore the boundaries of model hallucinations. Crucially, to ensure reliable assessment of dynamic inputs, SAMF incorporates a structured metric suite driven by an ensemble of multi-modal oracles. Our extensive experiments reveal that state-of-the-art MLLMs exhibit significant performance degradation under fuzzing compared to conventional settings, exposing a dissociation between reasoning capabilities and factual grounding. Furthermore, we identify a helpfulness-hallucination trade-off, where reinforcement learning alignment inadvertently exacerbates sycophancy in instruction-following tasks. The framework, code and benchmark are available at this https URL. 

---
# From Evaluated Models to Evaluation Aids: A Multi-Evidence Study of LLM-Based Difficulty Calibration for Programming Examinations 

**Authors**: Hongfei Yan, Jiangkai Xiong, Yiqing Li, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07523)  

**Abstract**: Difficulty differences across parallel-class programming examinations affect the fairness of course assessment. This study repositions large language models from benchmark evaluation targets to auxiliary evidence sources for interpreting exam difficulty, combining AI evidence with aggregated student performance, item exposure, online-judge process data, and teacher interpretation. First, ten models solved an eight-problem final exam synchronously with 120 students: AI pass rate correlated positively with student pass rate (Spearman rho = 0.866, exact p = 0.0119), and a solving-based composite difficulty index correlated negatively with it (rho = -0.905, exact p = 0.0046). A single structured reviewer was then run via auditable API calls on a third-party OpenAI-compatible endpoint whose model label (gpt-5.6-sol) cannot authenticate an official OpenAI upstream model; call metadata and raw responses are archived. Across 79 problems from 11 parallel-class final exams, AI overall difficulty correlated with problem-level pass rate at rho = -0.871 and with non-attempt rate at rho = 0.800; in a 26-problem longitudinal Data Structures and Algorithms B sample, the correlations were -0.829 and 0.883. A 106-problem introductory-course (CS101) sample marks the boundary: the problem-level correlation weakened to rho = -0.552, and the exam-level correlation across 16 exams was near zero, with cohort composition dominating exam-level outcomes. Exposure-discount (0-0.40) and duplicate-problem perturbation tests did not change these directions. AI evidence can thus serve as an external reference for problem validation, parallel-class fairness discussion, and longitudinal quality tracking, while the model-identity boundary, single-reviewer design, and review-output instability set explicit limits: AI difficulty scales must not be used for individual student evaluation or automatic grade adjustment. 

---
# KumbhDoot: A Scale-Ready, LLM-Bounded Architecture for Mass-Gathering Public-Service Assistants 

**Authors**: Saurabh Sakalkar, Abhishek Singh, Ramesh Raskar  

**Link**: [PDF](https://arxiv.org/pdf/2608.07520)  

**Abstract**: Mass religious gatherings such as the Kumbh Mela concentrate tens of millions of people into a single region over a few weeks, producing intense, repetitive, multilingual, and safety-critical demand for information. The default response, a conversational assistant that routes every query to a large language model (LLM), is poorly matched to this setting: it is costly at scale, slow on emergency paths, prone to hallucination on facts that can cause physical harm, and unusable when connectivity fails. We describe KumbhDoot, an agentic pilgrim assistant for the Nashik Simhastha Kumbh Mela built on a different principle. It operates on a foundational design principle that prioritizes semantic similarity over starting with an LLM. Generative models are invoked only in instances where similarity-based retrieval is insufficient to produce a correct answer. The system utilizes a "semantic cache": an embedding-indexed store as a single retrieval primitive, which handles intent routing, answer caching, offline lookups, and multi-agent retrieval. A custom three-tier agent architecture operates directly on this store, ensuring decision paths remain inspectable and avoiding the use of generic multi-agent frameworks that would trigger implicit per-step LLM calls. We present the architecture, an analytical cost model for its per-query economics, and an honest account of where similarity is sufficient and where generative reasoning remains necessary. We argue that for bounded, high-stakes, low-connectivity public-service domains, a similarity-first and LLM-bounded design is not merely cheaper but architecturally more appropriate than an LLM-default one. 

---
# Representation Matters in Longitudinal Affective Computing 

**Authors**: Igor Matias, Maximilian Haas, Eric J. Daza, Matthias Kliegel, Katarzyna Wac  

**Link**: [PDF](https://arxiv.org/pdf/2608.07518)  

**Abstract**: Longitudinal, in-the-wild, wearable sensing yields day-level physiology, sleep, activity, and environmental streams, whereas affect and cognition are labeled only episodically (per waves). We recast this cadence mismatch as a temporal representation problem and compare three wave-level mappings from dense histories to sparse labels: levels (within-wave summaries), absolute drift (change across waves), and proportional drift. Using almost a year of data from 82 adults in the Providemus alz study, we model 21 affect and cognition outcomes. Day-scale signals are reduced to compact wave-level descriptors (central tendency, dispersion, and distributional shape) and learned with four regressors under two orthogonal evaluation axes: leave-one-subject-out and leave-one-wave-out. Performance is reported as scaled MAE using both mean and median across folds. Differences emerge: affective states are best predicted by wave-to-wave absolute drift, whereas cognitive performance aligns with within-wave levels, reflecting emotion dynamic theories. Across windowing features, shape descriptors (e.g., minima, kurtosis) carry more signal than simple means/medians. We contribute a representation triad for sparse-label modelling, a wave-level feature schema applicable on-device, and a dual-axis reporting practice that separates cross-participant generalization from temporal robustness. These results convert temporal representation from an implicit preprocessing step into an explicit, testable design choice for real-world affective-computing applications in brain health. 

---
# EMMR: Emotion-Mediated Multimodal Reasoning for Personality Assessment in Asynchronous Video Interviews 

**Authors**: Dongsheng Hu, Tianyi Zhang, Chuang Liu, Yuan Zong Yong Li, Wenming Zheng, Xiu-xiu Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07512)  

**Abstract**: Asynchronous Video Interviews (AVIs) have become increasingly popular for personality assessment. Recent large language models (LLMs) have shown potential for personality assessment from transcribed interview responses. However, text-centered methods may overlook non-verbal behavioral cues conveyed through visual and audio modalities, even though such cues are highly relevant to personality assessment. In particular, emotion-related cues provide important social and affective evidence for understanding candidates' behavior related to personality traits. Thus, we propose EMMR (Emotion-Mediated Multimodal Reasoning), a two-stage framework for MLLMs-based personality assessment for AVIs. EMMR extracts emotion-related cues from multimodal interview data and incorporates them into personality assessment through structured reasoning as auxiliary social and behavioral evidence. Experiments on two AVIs datasets, OPVA and AVI-6, show that EMMR improves MAE, MSE, and PCC compared with baselines. Further analysis indicates that semantic descriptions of emotion cues enhance personality assessment, while their quality affects personality assessment reliability. These results suggest that integrating emotion-related cues into multimodal reasoning is a promising direction for more interpretable MLLMs-based personality assessment in AVIs. 

---
# How sensitive do we want AI to be? Socio-communicative competencies of large language models in healthcare 

**Authors**: Dorothee Amelung, Andrew M. Bean, Sabine C. Herpertz, Felix H. Krones, Guy Parsons, Adam Mahdi, Isabella Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2608.07511)  

**Abstract**: Background. Effective clinical practice relies heavily on the socio-communicative skills of medical professionals. Large language models (LLMs) have been proposed for tasks such as triaging patients, report drafting or translating medical jargon to support informed decision-making. These applications require both factual and social competence. This study evaluates dialogues between LLMs and participants to assess the current state of socio-communicative competencies displayed in LLM-generated texts.
Methods. We extracted a subset of extended dialogues from the HELP-Med dataset, comprising 1800 conversation transcripts of interactions between human participants seeking medical information and three different LLMs, GPT 4o, Llama 3 and Command R+. Two experts coded the transcripts for demonstrations of socio-communicative behaviours (non-hostility, sensitivity, structuring, non-intrusiveness) using the IC-MD instrument, originally designed to evaluate interactional competencies in medical student admissions.
Results. The LLMs in our study showed strength in non-hostility, mixed results in sensitivity and non-intrusiveness and performed poorly in structuring.
Conclusion. Current LLMs lack the consistent and reliable socio-communicative skills needed for safe and effective use as healthcare advisors. While existing frameworks for assessing interactional competencies may support the development of more socially responsive LLMs, they will require adaptation to account for the differences in desirable behaviour between humans and LLMs. 

---
# PIVOT: Preference-based Intervention Vectors for Pedagogical Tutor Steering 

**Authors**: Fares Fawzi, Jiaxu Zhao, Tanya Nazaretsky, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2608.07509)  

**Abstract**: LLMs are increasingly used for conversational tutoring, but effective tutoring requires more than correct answers. Tutors must choose when to scaffold reasoning, hint, give feedback, explain, or invite reflection. Existing prompting and training methods improve pedagogical alignment, but lack reliable inference-time control over pedagogical strategies. We introduce PIVOT, an activation-steering framework that learns preference-based intervention vectors online for frozen LLM tutors. PIVOT uses a seven-category tutor-move taxonomy and a generate-label-optimise loop, where a human-validated LLM judge identifies target and confusable non-target moves to construct preference pairs for multi-layer residual-stream steering. Across held-out and out-of-domain tutoring data, PIVOT controls tutor moves while preserving relevance and fluency, and its directions can be scaled, transferred, and composed at inference time. In a user study with 30 teachers, 73.3% of participants preferred steered conversations over neutral baseline interactions using the same prompt, and rated the controls as clear, usable, and pedagogically meaningful. 

---
# JaleesBench: Are AI Assistants Good Spiritual Company? 

**Authors**: M. Waleed Kadous, Benjamin Olsen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07508)  

**Abstract**: Large language models are already advisors to millions of people of faith who bring them real decisions. The pressing question for a person of faith is not what a model knows or professes but what its counsel does to the person who receives it. We introduce JaleesBench, which measures whether an AI agent is a righteous companion, judged by the residue an exchange leaves on the user, in the manner of the perfume-seller and the blacksmith. It comprises 140 two-turn scenarios drawn from a classical compilation organized by virtue (Riyad al-Salihin), under six adversarial pressures and three framings, scored by two frontier judges against each scenario's own supporting texts. Across eight systems: (1) generic frontier models are only middling companions out of the box but a one-page guide makes them genuinely good ones, on par with the domain-tuned assistant: the frontier APIs climb from +0.28/+0.23 to a Guided +0.84-0.87, so most of the expert's edge is companionship instruction that fits in a prompt; (2) every system caves under relational pressure, insistence and personal appeal; (3) the domain-tuned assistant's advantage is overwhelmingly its retrieval-and-prompting layer, not its base model (+0.74 over the identical underlying model); and (4) it can be used to improve existing systems: guided by its diagnosis, a single steadfastness instruction lifts a deployed Islamic assistant from +0.48 to +0.84 (Faith unstated, after pressure), matching the best guided frontier systems while preserving first-response quality. The construct is faith-general; we instantiate it for Islam as the first of a planned cross-tradition family. Code, scenario bank, and rubric are open source (this http URL), with an interactive results browser at this http URL. 

---
# Innovating with Generative AI: A Human Bottleneck Framework 

**Authors**: Julian De Freitas, Ayelet Israeli, Gideon Nave, Artem Timoshenko, Olivier Toubia  

**Link**: [PDF](https://arxiv.org/pdf/2608.07504)  

**Abstract**: We propose a human bottleneck perspective for understanding how generative AI transforms the innovation process. The central premise is that many constraints traditionally plaguing the innovation process are cognitive and social in origin, rooted in how people generate ideas, evaluate novelty, and communicate through social systems. Generative AI does not act uniformly on these constraints. At each stage, it can deepen some bottlenecks while alleviating others, and predicting these outcomes requires understanding the underlying mechanisms of the constraint itself. We identify bottlenecks in four stages of the innovation process: ideation, screening and testing, preference measurement and consumer insight, diffusion, and market learning. By grounding analysis in human behavior rather than rapidly changing AI capabilities, we offer a framework for assessing whether new developments alleviate or intensify the bottlenecks that matter most at each stage. We also distinguish bottlenecks likely to narrow as capabilities improve from those rooted in enduring human constraints. We further discuss AI-related issues that cut across the entire innovation pipeline, challenging the very existence and structure of the traditional innovation process. 

---
# Harnessing Abundance: A Generativity Perspective on Human-GenAI Collaboration 

**Authors**: Yoram M Kalman, Yun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07500)  

**Abstract**: Research on human-GenAI collaboration yields conflicting findings: GenAI can enhance creativity yet reduce collective diversity, with uneven benefits across skill levels. Rather than treating these as contradictions, we argue they reflect a core feature of GenAI: abundance. GenAI makes ideas, drafts, and recombinations plentiful, potentially expanding the hypothesis space and surfacing unanticipated possibilities. However, abundance alone doesn't ensure better outcomes. We propose generative fit as a unifying mechanism explaining when abundance yields productive creativity and when it backfires. Drawing on Generativity Theory, generative fit captures how well a system's generative potential complements a community's generative capacities. We develop a conceptual framework for collaborative human-GenAI settings where participants share goals, depend on one another, and must integrate diverse contributions. By mapping abundance to cognitive, social, and organizational factors of collective creativity, we explain apparent tradeoffs and offer actionable implications for designing workflows that convert abundance into valued creative outcomes. 

---
# Evaluation of Motivational Interviewing Counsellors with Task-Aware Multi-Stage LLM-Based Simulated Clients 

**Authors**: Jiading Zhu, Xinyu Cindy Wang, Thomas Nguyen, Yan Qing Lee, Osnat C. Melamed, Peter Selby, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2608.07499)  

**Abstract**: The development and benchmarking of Large Language Model (LLM)-based Motivational Interviewing (MI) counsellors now often rely on LLM-based simulated clients. Prior work on simulated clients, however, has not aligned with the specific tasks fundamental to the MI therapy approach. A key task is evoking, in which the counsellor first elicits the client's ambivalence and then strengthens the client's motivation for change. We present Evoke-Sim, a task-aware, multi-stage LLM-based client simulation framework for evaluating MI counsellors in smoking cessation, designed specifically for the evoking MI task. Evoke-Sim employs structured client profiles, an evoking-specific three-stage conversation flow, and a reveal policy that regulates which client profile information might be disclosed at each stage. We show that compared to existing profile-grounded simulated clients, Evoke-Sim is better at differentiating levels of MI quality using task-aware evaluation metrics, while reducing non-grounded client statements and premature disclosure of client information, setting a higher standard for the evaluation of LLM-based MI counsellors. 

---
# Knowing You Is Everything: LLM Agents Achieve Near-Perfect Profile-Consistent Reaction Prediction in Social Media Simulation 

**Authors**: Ljubisa Bojic, Ljiljana Matic, Joerg Matthes, Milan Cabarkapa, Bojana Dinic, Jue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07498)  

**Abstract**: Autonomous AI agents in social media present concrete risks to democratic discourse and platform governance, while also offering tools for pre-deployment recommender system testing. A central open question is whether persona-prompted LLMs can simulate individual-level social media reactions with sufficient accuracy to support either application, and how accuracy depends on profile completeness, model selection, and the generalization challenge posed by novel post content. This study benchmarks twelve LLM configurations on binary like/dislike prediction across 296 survey-based agent profiles and 26 ground-truth-mapped posts under three profile conditions, with leave-post-out machine learning classifiers as baselines. Across full-profile conditions, accuracy ranges from 75.54% to 96.68%, with a 30-point spread attributable primarily to model selection and confirmed by paired McNemar tests with agent-level bootstrap intervals. GPT-5.5 Pro accuracy degrades monotonically from 96.68% under a full profile to 62.32% under a reduced profile and to 51.00% with demographics alone, the last indistinguishable from the majority-class baseline, which confirms that demographic inference provides negligible predictive signal. Supervised classifiers collapse to 15.4% under leave-post-out, while LLMs sustain genuine zero-shot generalization unavailable to trained methods. Adaptive reasoning improves accuracy substantially for some models. Inter-model agreement is nearly double for posts with direct profile anchors (mean \k{appa} = 0.44) than for posts without them (\k{appa} = 0.23), and the least heterogeneous configuration homogenizes 34% of simulated population reactions. Results validate LLM-based simulation for recommender system stress-testing while documenting the behavioral accuracy that makes large-scale synthetic agent swarms a credible threat to public opinion. 

---
# EmoPatient: An Emotion-Directed Patient Simulator for Realistic Palliative Care Communication Training 

**Authors**: Yining Wu, Tianshu Du, Jinrui Fang, Chi Zhang, Sonal Admane, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.07495)  

**Abstract**: Effective communication during palliative care discussions is a critical clinical skill, yet training clinicians to manage complex patient emotions remains challenging. Large language model (LLM)-based patient simulators provide a scalable approach for communication training, but most existing systems treat patient emotion as static and fail to capture the dynamic emotional shifts observed in clinical interactions. We present EmoPatient, an emotion-directed patient simulator designed to generate evolving emotional responses during palliative care discussions. The system introduces an Emotion Director agent that estimates the patient's emotional state and generates turn-level control signals for emotional intensity, regulatory stability, and interactional guidance. We evaluate EmoPatient through controlled multi-turn physician-patient dialogue simulations and compare it with baseline simulators. Results show improvements across four theory-informed emotional realism metrics and robustness across conversational personality variants, suggesting that modeling emotional dynamics can improve the realism of LLM-based patient simulators for palliative care communication training. 

---
# How to Ask the AI: A User Perspective Survey for Large Language Model Prompting 

**Authors**: Yiqun Zhang, Yunfan Zhang, Mingjie Zhao, Sen Feng, Yiu-ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2608.07494)  

**Abstract**: AI tools like ChatGPT and DeepSeek, powered by Large Language Models (LLMs), allow users to obtain instant and effective content responses simply by typing requests, such as ``plan a three-day Vienna trip'', ``solve the attached mathematical problem'', ``draft an email to inquire review progress'', etc., which are also known as LLM prompts. Crafting clear and well-structured prompts leads to more appropriate LLM feedback, which effectively bridges human-LLM interaction. Although prompting appears accessible to non-expert users, precisely organizing effective prompts is a highly systematic and skillful process, presenting potential challenges even for experienced users. This survey explores the principles, taxonomy, and organization of prompts from a user-centered perspective. Differing from the existing surveys that primarily focus on technical principles and application scenarios of LLMs, this paper provides actionable guidelines for formulating effective LLM prompts across diverse real-world tasks and specifically contributes by: 1) developing an intuitive evaluation strategy for prompt efficacy, 2) providing prompting workflow demonstrations on representative applications, and 3) maintaining a dynamically updated open-source project to ensure the core takeaways remain up-to-date. These measures lower the threshold for users to correctly understand and craft prompts that align with evolving application scenarios. This work will be maintained as a living GitHub project \href{this https URL}{\textcolor{blue}{here}}. 

---
# Experience-Sensitive Game Learning: A Behavioral Study of Humans and Language Agents 

**Authors**: Yingying Guo, Zhuoxuan Ju, Ruibo Ming, Ruicheng Feng, Jinjin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07490)  

**Abstract**: Large language model agents are increasingly evaluated through games, but most benchmarks emphasize final outcomes rather than how players learn from repeated interaction. We study experience-sensitive game learning: how gameplay experience changes the decision-making behavior of humans and language agents. We formulate experience-sensitive game learning as a framework for analyzing behavioral change across repeated gameplay, rather than only final score or win rate. We introduce a suite of interactive games with reusable strategic structure, together with cross-game greedy-to-global metrics and game-specific behavioral diagnostics that make experience-driven change observable from action traces. We also collect repeated-game trajectories from human players and evaluate recent self-evolving language agents in the same behavioral metric space. Our results show that human players exhibit interpretable and relatively stable shifts from locally greedy heuristics toward more global strategic decisions. In contrast, current self-evolving agents often show noisy and transient gains, suggesting that existing self-evolution methods remain limited in converting gameplay experience into durable changes in decision-making behavior. 

---
# Cross-Model Humor Preference Modeling with Cards Against Humanity 

**Authors**: Victor Winter, Farhan Lakhany  

**Link**: [PDF](https://arxiv.org/pdf/2608.07481)  

**Abstract**: This paper investigates whether one large language model can approximate the humor preferences of another in a controlled Cards Against Humanity-style task. Two models - GPT-4o as Czar and Claude Opus-4.5 as Player - are evaluated on a binary humor-selection task constructed so that success cannot follow from self-preference. A reflected-cell stability procedure isolates 244 hands on which the two models hold deterministic but opposite preferences, partitioned into a 97-hand context pool and a 147-hand held-out test pool. The Player is then evaluated across five graded conditions: default self-preference, generic Czar-modeling instruction, model-identified Czar, prior Czar selections, and prior Czar selections with rationales. This gradient is designed to separate two sources of improvement: framing effects, in which the Player is told to attend to a Czar without seeing any of the Czar's behavior, and direct behavioral evidence, in which the Player is shown the Czar's prior choices. Player accuracy increased from 0.7% in Condition 1 to 19.0% and 25.9% in the framing-only conditions, and then rose to 72.8% and 82.3% once behavioral evidence and rationales were provided. An omnibus Cochran's Q test and pairwise McNemar tests confirmed that each step in the gradient produced a significant improvement. The results indicate that role instruction and model identity yield only modest gains, while behavioral evidence - especially when accompanied by rationales - supports substantial cross-model preference modeling. The findings are interpreted as theory-of-mind-like behavior in an operational rather than representational sense: the Player shifts away from self-preference toward another agent's demonstrated preferences, without any claim about an underlying representation of mental states. 

---
# Designing for Ethical AI: HCI Feature Considerations to Improve Fairness and User Experience in AutoML use for Human Resources 

**Authors**: Sundaraparipurnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07477)  

**Abstract**: This thesis examines the fairness of Automated Machine Learning (AutoML) tools in human resource hiring systems through the combined lenses of regulation, business strategy, and Human-Computer Interaction (HCI). It argues that fairness is no longer merely an ethical concern but a critical determinant of usability, trust, legal compliance, and organizational adoption. While AutoML platforms improve efficiency by simplifying model selection and deployment, they also risk perpetuating discriminatory outcomes when trained on biased historical hiring data. Existing platforms prioritize technical performance over fairness, leaving non-expert business users unable to detect or mitigate bias effectively. The study investigates fairness gaps in AutoML tools through four research questions focused on fairness mechanisms, interface transparency, human oversight, and product design priorities. Drawing on frameworks such as the Technology Acceptance Model, Innovation Diffusion Theory, Human-Centered AI, Cognitive Load Theory, and Affordance Theory, the research evaluates both usability and fairness alignment. Using qualitative HCI audits and quantitative testing of eight AutoML platforms on HR datasets, the findings reveal widespread deficiencies in transparency, user control, and bias mitigation support. The thesis proposes a five-dimensional HCI-based fairness evaluation framework and recommends embedding fairness directly into AutoML product design to improve accountability, adoption, and ethical sustainability in AI-driven hiring systems. 

---
# Positioning Generative Artificial Intelligence in STEM Assessment: When to Require, Scaffold, or Restrict Its Use 

**Authors**: Yizhu Gao, Zhongzhou Chen, Min Li, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2608.07475)  

**Abstract**: Generative Artificial Intelligence (GenAI) presents a governance challenge for STEM assessment. Unrestricted access can enable task outsourcing that undermines the validity of traditional assessments, while blanket prohibitions are difficult to enforce, may drive use underground, and do little to prepare students for workplaces where GenAI supported workflows are increasingly common.
This paper proposes a student focused framework grounded in Evidence Centered Design (ECD) that specifies when to restrict, scaffold, or require GenAI use in STEM assessment. The framework extends existing AI use taxonomies by providing decision rules that link target constructs, evidence requirements, and task characteristics to governance regimes.
Restriction is warranted when GenAI threatens construct relevant evidence for unaided proficiency, particularly for foundational knowledge and routine skills. Scaffolding is appropriate when bounded GenAI support reduces peripheral demands while maintaining interpretability. Requiring GenAI is appropriate when the target construct involves human AI collaboration and AI literacy.
Using examples from introductory physics, we illustrate how tasks can be designed under different GenAI use policies. The framework provides guidance for preserving learning integrity while supporting preparation for AI enabled environments. 

---
# Application of Artificial Intelligence for Fraudulent Banking Operations Recognition 

**Authors**: Bohdan Mytnyk, Oleksandr Tkachyk, Nataliya Shakhovska, Solomiia Fedushko, Yuriy Syerov  

**Link**: [PDF](https://arxiv.org/pdf/2608.07471)  

**Abstract**: This study considers the task of applying artificial intelligence to recognize bank fraud. In recent years, due to the COVID19 pandemic, bank fraud has become even more common due to the massive transition of many operations to online platforms and the creation of many charitable funds that criminals can use to deceive users. The present work focuses on machine learning algorithms as a tool well suited for analyzing and recognizing online banking transactions. The study`s scientific novelty is the development of machine learning models for identifying fraudulent banking transactions and techniques for preprocessing bank data for further comparison and selection of the best results. This paper also details various methods for improving detection accuracy, i.e., handling highly imbalanced datasets, feature transformation, and feature engineering. The proposed model, which is based on an artificial neural network, effectively improves the accuracy of fraudulent transaction detection. The results of the different algorithms are visualized, and the logistic regression algorithm performs the best, with an output AUC value of approximately 0,946. The stacked generalization shows a better AUC of 0.954. The recognition of banking fraud using artificial intelligence algorithms is a topical issue in our digital society. 

---
# Coordinated incentives in AI-generated misinformation governance 

**Authors**: Qin Li, Gui Zhang, Minyu Feng, Matjaz Perc, Attila Szolnoki  

**Link**: [PDF](https://arxiv.org/pdf/2608.07070)  

**Abstract**: With the rapid diffusion of AI-generated content, AI-driven misinformation is becoming increasingly pervasive and difficult to govern, undermining information credibility and social trust. This study models the strategic interdependence among a government regulator, an AI enterprise, and users through a three-party evolutionary game that incorporates heterogeneous rewards and punishments. From the resulting replicator equations, we characterize the evolutionary stability of competing governance and production strategies. The analysis indicates that neither unilateral regulation nor market incentives alone can effectively curb misinformation. Instead, an evolutionarily stable regime of real-information production arises only when regulatory rewards and punishment intensity, enterprise reputation loss, and user adoption incentives collectively surpass critical thresholds. The findings highlight the need for coordinated and adaptive policy mixes that align regulatory instruments with enterprise behavior and user uptake while managing governance costs. 

---
# From Trajectories to Evidence: Auditable Experimental Records for Industrial Research Agents 

**Authors**: Zijie Zhuang, Changxin Lao, Pengbo Xu, Hanwen Xu, Ruochen Yang, Yingzhi He, Peng Zhang, Jiangxia Cao, Yusheng Huang, Guohong Mu, Jian Liang, Ruiming Tang, Shuang Yang, Zhaojie Liu, Wenwu Ou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2608.05235)  

**Abstract**: Research agents increasingly conduct multi-round machine-learning experiments in industrial recommendation settings and retain the resulting trajectories to guide later decisions. Yet a completed trajectory is not automatically evidence: generated artifacts may be unsupported or incomplete, executed rounds may be invalid or confounded, and later modifications may obscure earlier findings. We study \textbf{trajectory-to-evidence conversion}, asking what a completed research process has actually established. We introduce an evidence-grounded framework that couples bounded verification of consequential artifacts with post-execution claim qualification. A context-isolated generate--verify--repair process checks artifacts for evidence violations and missing downstream requirements before release. After execution, validity and attribution checks consolidate evidence across rounds, qualify intervention-level claims as actionable repairs, diagnostic guards, or withheld findings, and preserve admitted claims as auditable records with explicit provenance and applicability boundaries. A hybrid LLM-assisted controller subsequently applies, defers, or rejects records based on available target evidence. Record audits characterize which claims survive qualification, while downstream diagnostics identify affirmative applicability judgment as a bottleneck for the tested controller. Across paper-to-target adaptations, later rounds often improve on the first, while final rounds frequently underperform an earlier best, exposing non-monotonic trajectory evolution. Candidates produced through the complete workflow also yielded positive online lifts relative to deployed baselines. 

---
