# Handover of In-Context Learning State Across Session Boundaries 

**Authors**: Masahiro Kato, Taka Kato  

**Link**: [PDF](https://arxiv.org/pdf/2608.14528)  

**Abstract**: This study investigates the methodological and theoretical properties of session handover in applications that use large language models. A task may continue in a new session when the context reaches the model's input limit, when the application restarts, or when another agent is asked to finish the task. The application must then decide which information from the earlier session to pass on. We formulate handover as the transfer of a task-relative in-context learning (ICL) state and distinguish exact recovery of earlier material from preservation of the target distribution. Under an exogeneity condition, predictive equivalence characterizes the coarsest deterministic sufficient handover and gives a fixed-length bit requirement. The analysis isolates the effects of the memory constraint, the writer, and the continuation procedure, and quantifies the cost of writing before the realized downstream query is known. We propose a three-part record that stores decisions and constraints exactly, uses task-justified statistics for repeated evidence, and retains original observations whose effect is not preserved by those statistics. Gaussian linear regression gives an exact finite-dimensional handover and finite-bit perturbation bounds, while nonparametric regression gives upper and lower bounds that relate memory to squared prediction error. These results provide a theory and method for deciding what a handover must retain and how its memory requirement depends on the continuation task. 

---
# Participatory Moral AI Is Not Neutral: The Invisible Hand of Developers 

**Authors**: Taenyun Kim, Edyta Bogucka, Daniele Quercia  

**Link**: [PDF](https://arxiv.org/pdf/2608.14522)  

**Abstract**: As AI systems make more morally loaded decisions across society, one response has been moral preference elicitation. In this approach, researchers poll participants on hypothetical dilemmas and use the aggregated votes to train a policy that an AI model then applies at scale. Before any vote is cast, developers make three key choices in the moral AI elicitation pipeline: feature scoping, voter sampling, and question framing. In other words, they decide which features go to a vote, which voters to include, and how to present the question. These choices are often opaque, undocumented, and treated as technical details rather than normative ones. We examine each of these choices within a common empirical study and show that each can shape the preferences produced by moral AI elicitation. Across two phases (N = 809) in three deployment contexts (i.e., AI kidney allocation, AI agents simulating absent workers, and generative AI depictions of the deceased), we examine the three main stages of the moral AI elicitation pipeline. First, morally relevant features shift across contexts. This suggests that feature schemas should not be assumed to transfer across deployment domains. Second, preferences differ by political ideology for roughly one-third of features, with some differences reversing direction. The ideological composition of the voter pool can therefore affect the resulting aggregated preference profile. Third, the wording of the elicitation question can narrow or widen ideological gaps by up to a full scale point. The framing conditions also change how moral foundations are associated with participants' judgments. Taken together, these findings suggest that voting-based alignment cannot deliver fair or transparent AI by aggregation alone; at minimum, each stage of the moral AI elicitation pipeline should be audited and disclosed. 

---
# Split the Labor: Separating Evidence Interpretation from Decision Aggregation 

**Authors**: Zhelun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14509)  

**Abstract**: Systems that ask a language model to reach a conclusion from many sources usually concatenate them into one prompt. This conflates two operations with different requirements. Interpreting a source rewards capacity and context. Combining interpretations rewards fixed arithmetic, comparability across instances, and the option to return nothing. Once separated, the design problem becomes the interface between them. We propose a four-field evidence tuple (hypothesis, reliability bucket, rationale, provenance) and show that fixing it determines both halves. The separation also reveals a failure mode in how such systems combine, which we call count-scale drift. Thresholding a sum of unnormalized weights is exactly posterior thresholding, but at an operating point that slides with the number of sources consulted. The slide grows with reader reliability. When source reliabilities differ, the vote rule and the posterior order instances differently, and no threshold reconciles them. Pooling calibrated log-likelihood ratios addresses both problems. The fix is arithmetic rather than architectural, and applies to a class of rules beyond language models: score-summing triage engines, diagnostic panels scored by counting positives, and additive multi-signal detectors. We then instantiate the principle twice on one longitudinal corpus, once after outcomes resolve and once before. The same partition helps in both, at different granularities: over reading in the first, over learning capacity in the second. There, a small sequence encoder on an easy auxiliary objective plus a tree ensemble carrying the censored survival loss reaches 0.921 AUPRC against 0.805 for a hand-crafted baseline. We separate what transfers from what must be re-estimated per domain, and state five predictions that would falsify the framework, three negative results, and which comparisons remain confounded. 

---
# Twin: Playing an Unknown Game with a Test-Time Digital Twin 

**Authors**: Alexy Skoutnev, Kirill Acharya, Gaston Longhitano, Madeleine Udell, Kevin Ellis, Iddo Drori  

**Link**: [PDF](https://arxiv.org/pdf/2608.14490)  

**Abstract**: We present a Test-time World-model Inference (Twin) system, in which a frontier coding agent writes an executable world model for completing continual learning tasks, such as ARC-AGI-3 games. Traditional approaches hand-engineer such models, one custom design per task. Each game hides its rules and goal, and our system constructs them from simulation and interaction alone. Its inductive prior over grid games is strong enough to recover the true transitions of the game and the goal on nearly all levels. Replay validation happens in a twin world model. The harness enforces that an action is not made until the program reproduces every previous observed game transition. Each mismatch between a world model prediction and the actual action result becomes a counterexample that is used to repair the world model. Twin clears 179 out of 183 levels (97.8%), and does so more efficiently than humans in 158 out of 179 levels (88.3%). The system infers the goal before any reward on 156 of the levels it clears (87.2%), and in the remaining levels automatically discovers the goal by search. The benchmark scores completion and action efficiency, between 0 and 100, against humans playing each game for the first time. Played directly, the base model scores only 7.8%; an off-the-shelf harness increases it to 61.1%, whereas our twin world model increases the same base model to 93.3%, clearing 23 out of 25 games. Building a usable world model is simpler than anticipated, whereas the harder problem is inferring the right goal. 

---
# Shift Aware Transfer Learning with Adaptive Dual-Encoder Fusion for PM Forecasting in Data-Limited Environments 

**Authors**: Shahab Band, Hamed Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14456)  

**Abstract**: Short-horizon forecasting of fine particulate matter (PM2.5) remains difficult when observations from the target domain are limited and the statistical properties of the source and target domains differ. In these settings, models trained only on local data may not capture complex temporal dynamics, while direct transfer learning can result in negative transfer. This study develops a shift-aware dual-encoder transfer framework that combines source-domain knowledge with target-specific representation learning. The source encoder was pretrained using hourly observations from 10 U.S. monitoring locations. The framework was then adapted and evaluated using two years of hourly observations from 77 stations in Taiwan under a chronological train-validation-test protocol. Among the four principal baselines, the frozen-source dual-encoder model achieved the best performance, with MSE = 21.8960, MAE = 3.1597, and R^2 = 0.8725. This corresponds to an MSE reduction of approximately 7.1% relative to TL-v1 and 4.1% relative to TL-v2. The ablation analysis showed that removing the Taiwan-specific branch caused the largest decline in performance. Allowing the source encoder to adapt produced the best overall result, with MSE = 21.6575, MAE = 3.1383, and R^2 = 0.8739. SHAP analysis indicated that predictions were driven mainly by recent PM2.5 observations and meteorological variables related to pollutant transport and dispersion. These results suggest that source-domain knowledge is most effective when target-specific information is preserved and the transferred representation is allowed to adapt under target supervision. 

---
# SheetCompass: Hierarchical Relation Graphs for Agentic Spreadsheet Reasoning 

**Authors**: Panjing He, Mingyue Cheng, Yucong Luo, Li Li, Xiaohan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14452)  

**Abstract**: Spreadsheets are widely used to organize, analyze, and manipulate semi-structured data, yet automated spreadsheet reasoning remains challenging for large language models (LLMs). Real-world workbooks often contain implicit cross-table associations, fine-grained column dependencies, and complex spatial layouts. Existing methods typically flatten these multidimensional structures into sequential strings, losing important intra-sheet boundaries and inter-sheet semantics. Consequently, LLMs cannot exploit the global spatial context that human experts naturally use when inspecting spreadsheets. We propose SheetCompass, a graph-guided and memory-driven agentic framework for spreadsheet reasoning and automation. SheetCompass explicitly models structural relationships within and across worksheets while maintaining task-relevant information in memory, enabling agents to reason more effectively over complex workbooks. 

---
# Wyvern: An Agentic Framework for Generating Grounded Multimodal Reports 

**Authors**: Beatrice Alessandra Motetti, Emilien Guandalino, Daniele Jahier Pagliari, Alessio Burrello, Lorenz K. Müller, Konstantin Berestizshevsky, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.14446)  

**Abstract**: In the current artificial intelligence-driven innovation era, the pace of knowledge growth is accelerating, and is hard to keep up with. While generative models are increasingly used to synthesize content, they often lack in information grounding. To address these peculiarities of our time, we propose Wyvern, a multi-agent framework for the automated generation of grounded, multimodal technical reports. Wyvern allows for the generation of multimodal outputs, integrating images, tables, and text with supporting references in a unified report. Additionally, a particular focus is placed on the grounding of the content, with the implementation of a claims auto-revision stage. We conduct a human evaluation study to assess the quality of our proposed framework. The results show that the figures' informativeness is perceived as superior to that of a recent baseline in 87% of cases. Furthermore, Wyvern's reports are rated as more useful than those produced by three alternative methods in 63% to 100% of instances. We also carry out automatic evaluations showing that Wyvern gains up to 2.3$\times$ in citation recall and 1.6$\times$ in citation precision with respect to the baselines. 

---
# PACE-Bench: Benchmarking Physics Adaptation via Code Evolution in Dynamic Environments 

**Authors**: Yuhao Zhan, Bingxiang He, Zecong Tang, Chaojun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.14441)  

**Abstract**: Self-evolving agents improve future behavior from interaction experience, yet existing evaluations typically optimize under fixed execution conditions and do not test recovery after those conditions change. To address this gap, we introduce PACE-Bench (Physics Adaptation via Code Evolution), a simulator-grounded benchmark of 144 source-to-target adaptation pairs across six physics domains. Each pair links a source environment to a mutated target environment with the same goal and interface. A code-driven design that succeeds in the source fails in the target, where agents must iteratively adapt it into a working target design using diagnostic sandbox feedback within a limited attempt budget. We compare ten self-evolving methods from four paradigms. The benchmark remains far from saturated: Reflexion + Qwen3-14B succeeds on only 35.9\% of full-benchmark pairs, while GPT-5.5 solves 66.7\% of the Statics subset under the full budget. Together, these results show that simulator-grounded reflection is more reliable than unverified self-revision, while memory anchors agents to early designs and broad tree search explores without converging. Even revealing exact physical changes does not raise the performance ceiling, pointing to mechanism redesign rather than parameter inference as the central bottleneck. Data and code are available at this https URL. 

---
# The Dynamics of Intelligence Explosions 

**Authors**: Toby Ord  

**Link**: [PDF](https://arxiv.org/pdf/2608.14426)  

**Abstract**: AI is increasingly being used to help with AI R&D. Under certain conditions this feedback loop might be able to produce an intelligence explosion, with rapidly escalating AI capabilities. I explore the mathematics of the most explosive possibilities, with an eye to understanding what drives the dynamics. I show that singular growth (towards a vertical asymptote) is harder to achieve than would be expected from recent economics-inspired modelling, and that there is an important but neglected class of growth rates that are faster than exponential but don't lead to a vertical asymptote. I draw out the generation time (the time to go around the feedback loop) as a neglected parameter that plays a pivotal role in determining the behaviour of any intelligence explosion --- one cannot have singular growth unless the generation time rapidly approaches zero. 

---
# Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations 

**Authors**: Toby D. Pilditch  

**Link**: [PDF](https://arxiv.org/pdf/2608.14425)  

**Abstract**: LLM evaluations often use fixed sampling budgets, testing every item the same number of times even after estimates are precise. We introduce optstop, a precision-based adaptive stopping framework that treats evaluation as a sequential measurement problem: keep sampling where uncertainty remains high, and stop where estimates are precise or stable enough. The framework builds on hierarchical Bayesian inference, supports binary, ordinal, and continuous outcomes, and keeps every benchmark item eligible for sampling, without requiring a calibrated item bank. It runs live or retrospectively, and includes a safeguard that samples more cautiously as measured performance approaches zero, where rare successes matter most. In an illustrative 200-item, 10-epoch evaluation, it removes 57%-97% of planned trials across nine validation settings, with overall conclusions equivalent to the full run. These results show that LLM evaluation compute can be allocated by uncertainty rather than by fixed repetition counts, with the magnitude of savings depending on evaluation design. 

---
# The Past and Future of AI Scientists 

**Authors**: Ross D. King  

**Link**: [PDF](https://arxiv.org/pdf/2608.14407)  

**Abstract**: We present a survey of the past and future of AI Scientists: machines capable of automating science. AI Scientists can originate hypotheses, deduce their consequences, design and execute experiments, interpret their results, and revise their beliefs. Such systems are integrated scientific agents, connected to the literature, formal knowledge, mathematical models, simulations, data-analysis systems and physical laboratories.
Adam was the first machine to make novel scientific discoveries through cycles of hypothesis formation and physical experimentation. Eve established the architecture of the modern self-driving laboratory. Foundation models, autonomous agents and laboratory robotics now make it possible to build systems far more general than either Adam or Eve.
The central problem is no longer whether individual components of science can be automated. They can. The problem is integration. AI Scientists must combine neural learning with logic, probability, mathematics, causal reasoning, simulation, experimental design, robotics and formal scientific records.
AI Scientists have the potential to transform science: to make science faster, cheaper, more systematic and more reproducible. AI Scientists could investigate systems too complicated for unaided human science, and enable thousands of AI scientists to work together on single problems.
The Nobel Turing Challenge sets the goal of developing by 2050 AI systems capable of automating Nobel-quality discoveries. Progress is ahead of schedule. When we succeed it will create a new form of science and transform the world. 

---
# LLMs Don't Pay for the Jump 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2608.14397)  

**Abstract**: Zahavy [2026] argues that Large Language Models, despite their capabilities in induction and deduction, cannot perform the abductive "Jump" that produced Einstein's equivalence principle, and attributes this limitation to the absence of embodied simulation. Zheng-Xin [2026] and Farmer [2026] question whether embodiment is necessary for abduction, pointing to alternative routes to General Relativity and forms of abduction that require no sensorimotor grounding. Max Planck resolved the blackbody radiation problem in 1900. Planck's move to E = h{\nu} required no embodied simulation. It was motivated by a mathematical consequence of classical theory, an infinite predicted energy for a finite measured quantity, that could not be physically accepted. We show that neither induction nor deduction could have produced the postulate and argue that its adoption required a coupling between epistemic error and physical cost. We formalize this distinction through thermodynamic coupling and show that fixed-weight transformer inference lacks such coupling, regardless of model scale. This is consistent with empirical results showing that output entropy remains nearly unchanged across tasks with sharply increasing causal difficulty, even as accuracy falls from 100% to 17%. We therefore argue that the missing ingredient in machine abduction may lie deeper than embodiment: a system must have some physical mechanism through which epistemic error becomes costly enough to force revision. 

---
# Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons 

**Authors**: Wei Zhao, Zhe Li, Peixin Zhang, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.14392)  

**Abstract**: Neuron- and path-level interventions offer the finest-grained route to defending large language models (LLMs) against jailbreak attacks, yet existing methods fall short of this promise, i.e., they often compromise model utility significantly. Specifically, one line of work suppresses toxic neurons to erase harmful semantics, but since such semantics are distributed across the network, blocking every pathway forces a large intervention footprint. An alternative line of research focus on identify safety neurons using external classifiers. While promising, the existing approaches suffer from compromising neurons that are important for the model utility as well. Moreover, both approaches remain always on and thus perturb every benign request even when no attack is present. To address these limitations, we present \ours{}, a training-free defense that first identifies safety-specific neurons through per-neuron hypothesis tests under false-discovery-rate control together with a utility-specificity filter. Based on this identification, a trigger-style clamp holds the selected neurons at their harmful-conditional mean activations, injecting an internal harmful-input signal that triggers the refusal behavior learned during alignment. The clamp is then realized by two provably equivalent deployment modes, namely a detector-gated inference-time intervention and an offline bias-patch weight edit. Extensive experiments across four safety-aligned LLMs and four representative attacks demonstrate that \ours{} reduces the average attack success rate to at most 2.0\% while incurring a utility drop of only 0.5\% to 5.3\% on MT-Bench, the smallest among all defenses. Code is available at this https URL. 

---
# AgentRewind: Recoverable Execution for Long-Horizon LLM Agents 

**Authors**: Yu Zhuang, Kefei Chen, Yitong Duan, Shuxin Zheng, Jian Li, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14380)  

**Abstract**: Many real-world tasks require LLM agents to interact with their environments over long execution horizons. Errors that occur early in execution may propagate through both the agent context and environment state, and their effects may be difficult to reverse through subsequent actions. Existing methods mainly seek to reduce such errors through plan refinement and safety checks but provide little support after errors occur. To enable recovery during long-horizon execution, we present AgentRewind, a runtime recovery framework that records aligned checkpoints of the agent context and controlled environment, allowing agents to return to an earlier state and resume execution with information from previous attempts. We also construct MettleBench, a benchmark for evaluating task completion and partial progress on long-horizon engineering assignments containing a series of related requirements. Experiments across tasks, multiple models, execution strategies, and agent harnesses show that AgentRewind improves task success rate and average checklist progress over the compared baselines. 

---
# Wrong but Useful: Trajectory Value Beyond Answer Correctness in Multi-Agent Messages 

**Authors**: Chih-Hsuan Yang, Anjir Ahmed Chowdhury, Cheng-Hau Yang, Weijian Zheng, Fernando Llorente, Xiaolong Ma, Xinyang Li, Eliu A. Huerta, Ian T. Foster, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2608.14375)  

**Abstract**: Multi-agent reasoning systems often use agreement, confidence, or automated scores to decide which messages should shape a final answer. Such filtering assumes that a message likely to be correct is also worth keeping. Yet a wrong answer can contain a useful decomposition, constraint, or scientific principle. We test this distinction with Diverse Hypothesis Deliberation (DHD), a controlled measurement protocol that caches five independently generated messages and replays the same downstream solver, called the integrator, with each message available or hidden. The replay comparison measures a message's trajectory value: whether making the message available helps or harms subsequent reasoning. Across five mathematics and science benchmarks and two openly available model families, gpt-oss-120b and gemma-4-31B-it, wrong-helpful messages appear in every benchmark-model combination. Among wrong-answer messages that change final correctness, more than four in ten changes are helpful in each model. Controlled repeats show that the number of repeatable message effects is unlikely to arise from replay variation alone (p=0.0002). A focused intervention on repeatable wrong-helpful messages finds that the complete message works best, while retaining its reasoning preserves more success than retaining only its answer; the source of the complete-message advantage remains open. Within the same problem, repeated trajectory-value evidence also identifies a better keep-or-remove choice than answer correctness alone. Answer correctness is therefore informative but does not determine trajectory value. DHD measures this missing property and produces reusable labels for learning when agents should listen. 

---
# Designing Sustainable Federated Learning as a Service using Neural Architecture Search 

**Authors**: Keya Patel, Sajib Mistry, Sheik Fattah, Deepak Kanneganti, Aneesh Krishna, Mufti Mahmud, Monowar Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2608.14359)  

**Abstract**: The sustainability constraints of FLaaS consumers pose significant challenges to maintaining carbon-feasible federated training in FLaaS environments. These constraints often lead to infeasible consumer participation and unstable federated training under hard carbon constraints. We propose a Sustainable Federated Learning as a Service (SFLaaS), a carbon- constrained Neural Architecture Search (NAS) framework for heteroge- neous sustainable constraints. We introduce a requirement-driven search space that transforms consumer sustainability profiles into a feasible architecture region before federated execution. We develop a consumer-level carbon feasibility estimation mechanism to evaluate candidate architectures under dynamic carbon conditions. We propose a sustainable con- sumer scheduling strategy that adaptively selects feasible consumers and allocates local workloads to preserve consumer participation and statistical data coverage. An evolutionary search strategy jointly optimised for predictive performance, consumer feasibility, and participation coverage under hard carbon constraints. Experiments on real-world datasets and a simulated environment demonstrate the effectiveness of the proposed approach. 

---
# Disentangled Shared Representations Improve Morpho-Transcriptomic Integration 

**Authors**: Julian Ostermaier, Swann Ruyter, Reuben Dorent, Daniel Racoceanu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14355)  

**Abstract**: Spatial transcriptomics (ST) enables the simultaneous profiling of gene expression and tissue morphology, creating an opportunity to learn multimodal representations capturing shared morpho-transcriptomic structure. However, standard multimodal models often compress modalities into a common latent space without explicitly separating shared and modality-specific sources of variation, which may limit downstream utility. We investigate whether explicit disentanglement of shared and private latent components improves multimodal representation learning for paired Hematoxylin \& Eosin (H\&E) and ST data. We compare VAE-based and contrastive approaches, each in standard and disentangled variants, across two cancer cohorts under matched experimental conditions. Representations are evaluated using cross-modal reconstruction, downstream probing and cross-modal probe transfer. The experiments suggest two main trends. First, contrastive objectives yield higher downstream probing performance than VAE-based models. Second, disentangled variants improve the selected reconstruction and probing metrics, although the gains depend on the model family, task, direction, and disentanglement strength. Overall, our results suggest that explicitly factorizing shared and modality-specific information can improve multimodal representation learning for spatial transcriptomics and provides a useful evaluation framework for future foundation models. 

---
# ScienceFlow: A long-horizon agent for ML research, scientific discovery and beyond 

**Authors**: Mingming Zhao, Jiqian Dong, Kangping Xu, Zadid Hasan, Chengrui Fan, Shan Jiang, Shuai Mao, Ting Lingya, Linyi Zou, Tailin Zhou, Yun Hin Chan, Wenkai Zhang, Zhanhong Zhou, Guowei Huang, Hongliang Li, Wenjing Cun, Zhitang Chen, Mingxuan Yuan, Yanhui Geng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14354)  

**Abstract**: Enabling LLM agents to sustain productive, stable, and goal-aligned research over extended horizons is a central challenge for autonomous machine learning and scientific discovery, as progress hinges on continuously managing evolving state, exploration decisions, and computational resources. Pioneering autoresearch agents, despite great success, still lack mechanisms for continuity, recovery from dead ends, and value-driven compute allocation, which inherently undermines overall search efficiency, wastes computational resources, and lowers the chance of ultimate success. To bridge this gap, we introduce ScienceFlow, an end-to-end autoresearch agent framework that organizes long-horizon research work into research segments grounded in executable workspaces. It represents research progress as recoverable executable states, enabling efficient exploration, revision, and execution. Transitions between research segments are governed by Executable-State Transition through Re-Anchoring (ESTRA), which selects either the live state or an archived state as the next anchor and determines whether to continue or redirect the research trajectory. An evidence-aware execution controller allocates resources to physical jobs based on resource availability, remaining budget, and validated progress. We evaluate ScienceFlow on tasks spanning machine learning, scientific modeling, and mathematical optimization. Results on diverse long-horizon benchmarks demonstrate its ability to sustain effective research processes, highlighted by a SOTA 70.22 percent Any-Medal score on the full MLE-bench within a 24-hour budget, outperforming prior reported results by 4.92 percentage points. The efficacy of ScienceFlow further demonstrates that efficient state management, adaptive exploration, and objective-aligned execution are critical for scaling autonomous research beyond short-horizon interactions. 

---
# Clearing the Fog: Towards Installing and Refining Proactive Exploration Capabilities in LLM Agents 

**Authors**: Zhizhao Guan, Chen Huang, Ziming Liu, Hongru Liang, Wenqiang Lei, See-Kiong Ng, Tat-Seng Chua, Anthony G Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2608.14339)  

**Abstract**: We study proactive exploration in LLM agents, i.e., the ability to explore an environment to acquire information that improves future decision-making. In this regard, we first identify two fundamental bottlenecks that hinder this capability and then propose \ours, a novel method designed to instill and refine proactive exploration. Specifically, \ours\ consists of two components: (1) Exploratory Data Construction, which synthesizes exploration-rich trajectories to mitigate the hindsight bias of standard demonstrations; and (2) RL Optimization with Contrastive Signal Guidance, which leverages contrastive trajectory pairs to distinguish productive exploration from redundant wandering. Extensive experiments demonstrate the effectiveness of \ours\ and provide insights into the characteristics of proactive exploration. Our code is available at: this https URL. 

---
# Program-space Diffusion for Morphology-to-Transcriptomics Prediction 

**Authors**: Ruyter Swann, Dorent Reuben, Racoceanu Daniel  

**Link**: [PDF](https://arxiv.org/pdf/2608.14330)  

**Abstract**: Spatial transcriptomics (ST) enables genome-wide gene expression profiling while preserving tissue architecture, but its cost and limited scalability remain major bottlenecks. This has motivated models that predict spatial expression directly from routine histology. Despite promising results, most existing approaches operate at the gene level without leveraging established transcriptomic modeling practices and rely on heterogeneous gene selection strategies, which complicates fair comparison across methods.
We propose to reformulate morphology-to-transcriptomics prediction as conditional generation in transcriptional program space, thereby exploiting coordinated transcriptional variation instead of predicting genes independently. Using consensus non-negative matrix factorization (cNMF), we extract a low-dimensional set of transcriptional programs capturing coordinated expression variation in the training data, and train a conditional diffusion model to generate program activations from histology. This formulation exploits coordinated transcriptional variation and substantially lowers the dimensionality of the conditional generative task. 

---
# AnchorBench: A Multi-Pathway Benchmark for the Anchoring Effect in LLMs 

**Authors**: Yiderigun Borjigin, Alexander Hermann, Christian Cyron, Roland Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2608.14320)  

**Abstract**: The anchoring effect is a cognitive bias in which an initial reference value shifts a later judgment toward itself. This effect is well established in human judgment and decision-making, and recent work suggests that large language models (LLMs) exhibit similar behavior. However, existing work on anchoring in LLMs typically evaluates only a narrow set of anchor pathways and rarely distinguishes irrelevant from plausible anchors. We introduce AnchorBench, a benchmark for the anchoring effect in LLMs that evaluates multiple anchor pathways under an explicit anchor relevance axis. Across fourteen models, including ten open-weight models and four frontier API models, and a large set of controlled prompts, we find that (1) anchoring is strongly pathway-dependent, (2) plausible anchors usually induce larger shifts than irrelevant ones when introduced through stronger pathways, (3) anchor influence generally weakens as the anchor moves farther from the evidence-supported answer, most clearly on External and RAG, and (4) high task accuracy on the anchor-free control condition (Acc$_{10}$: answers within 10 points of gold) does not guarantee robustness: even frontier API models above 95% control accuracy remain susceptible to plausible anchors. 

---
# Sensor-Driven Mission Synthesis for UAV/UGV Swarms: A TB-CSPN Coordination Architecture with Hardware-Enforced Safety 

**Authors**: Uwe M. Borghoff, Paolo Bottoni, Remo Pareschi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14306)  

**Abstract**: This paper presents a coordination architecture for heterogeneous UAV/UGV swarms that synthesises mission actions from uncertain, multi-modal sensor evidence while preserving hardware-enforced safety at the actuation boundary. The approach combines radar, RF, acoustic, and visual observations with Topic-Based Communication Space Petri Net (TB-CSPN) orchestration to support incremental mission formation under partial and evolving information. Consultant agents transform sensor outputs into temporally bounded semantic tokens, while supervisor agents provide authorisation and policy-governed release of mission transitions. This separation between interpretation, coordination, and execution yields auditable decision paths, constrains non-determinism within the coordination layer through guards and synchronisation, and enables bounded-time integration of heterogeneous evidence. To improve resilience in contested environments, including cyber compromise, spoofing, jamming, and communication loss, the digital coordination layer is complemented by independent analogue safety envelopes that clamp or veto unsafe actuator commands issued to individual vehicles. A coastal-surveillance case study illustrates how the proposed architecture enables dependable, governed, and physically safe swarm coordination under operational uncertainty. 

---
# Intern-S2-Mobius: Foundation Model with Decoupled Knowledge and Reasoning 

**Authors**: Kai Chen, Jifeng Ding, Ning Ding, Jiaye Ge, Lixin Gu, Yicheng Gu, Qipeng Guo, Ermo Hua, Haian Huang, Haozheng Hou, Jie Hou, Xiangyu Hong, Che Jiang, Minxi Jin, Cheng Liang, Dahua Lin, Dawei Liu, Kuikun Liu, Chengqi Lv, Haijun Lv, Han Lv, Ningsheng Ma, Biqing Qi, Jianmin Qian, Shiya Su, Youbang Sun, Huanze Tang, Zhongbo Tian, Hanjing Wang, Rui Wang, Ting Wang, Yi Wang, Baiting Wu, Jun Xu, Bowen Yang, Hui Wang, Weida Wang, Haochen Ye, Jiashuo Yu, Shan Yu, Xiaoyi Yu, Qirui Zeng, Qi Zhang, Ming Zhang, Wenwei Zhang, Bowen Zhou, Xinyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.14290)  

**Abstract**: We introduce Mobius-v0, an architecture that comprises a globally shared Memory (FFN) that stores knowledge vectors and multiple Reasoners (Self-Attn) that iteratively achieve compositional reasoning. Using hidden states as cache and carrier, reasoners repeatedly query memory for required knowledge-vectors, while the knowledge is transmitted back to reasoning operators. Through this knowledge-reasoning-separation architecture, Mobius achieves better knowledge compression and reasoning efficiency. Built upon Mobius-v0 architecture: 1) Our 7B model trained-from-scratch achieves similar downstream score as a 7B Transformer baseline with 62.6% of baseline's training data. 2) Our Intern-S2-Mobius, continually-pretrained from Qwen3.5-35B, achieves similar downstream score while delivering nearly 4x end-to-end inference speedup. 

---
# TimeSage-EV: A Live Benchmark for Agentic Time Series Analysis in Evolving Environments 

**Authors**: Qingren Yao, Yaxuan Kong, Yuqi Nie, Yichen Li, Stefan Zohren, Anna Vettoruzzo, Qingsong Wen, Ming Jin, Joaquin Vanschoren  

**Link**: [PDF](https://arxiv.org/pdf/2608.14270)  

**Abstract**: Time series analysis in high-stakes domains relies on recurring data releases, where new observations can alter the evidence base and the validity of later conclusions. Existing time series QA benchmarks mostly rely on fixed snapshots, leaving temporal validity and cutoff-aware evidence use unevaluated. We introduce TimeSage-EV, a live benchmark for agentic time series analysis in evolving environments. It tracks 60 real institutional scenarios across 6 domains, comprising 1,485 scenario-period QA pairs from Feb 2023 to May 2026 and spanning monthly, weekly, daily, and irregular release cadences. At each period, large language model (LLM) agents receive time series data and source reports, while the withheld target release provides ground truth. TimeSage-EV evaluates state identification, data summarization, and outlook reasoning. Experiments with frontier LLM agents and TimeSage-1.0, a novel self-evolving agent with a reusable analytical skill library, reveal significant performance gaps across model tiers and recurring failures in temporal validity, exogenous context use, and adaptation. We release TimeSage-EV as a research resource with monthly updates, code, a leaderboard, and failure-mode analyses. 

---
# Grounding Without Corrective Control: Truth-Tracking Profiles for Large Language Models 

**Authors**: Brett Reynolds  

**Link**: [PDF](https://arxiv.org/pdf/2608.14252)  

**Abstract**: Recent work suggests that some large language model representations have content or reference. Grounding can secure either without supplying live routes for correction. This paper asks what follows from that gap. An output is answerable when discrepancies can affect what a target- and task-specific arrangement produces, accepts, or withdraws. The arrangement has corrective control only when live, sufficiently independent routes can detect and repair fresh discrepancies. A route profile records which routes constrain the arrangement and how they are related. Those profiles support analysis of truth-tracking: patterned support for representational success.
Language models are the pressure case; text-only arrangements provide a task-relative limiting case. Text-trained models inherit patterns of testimony, coherence, and prior correction. Where target-sensitive correction survives training, these can supply derivative answerability (inherited constraint); live answerability is the relation supplied by a current route for fresh discrepancies. Fluent failures should follow when a task requires independently informative access to the facts. Self-consistency, retrieval, tools, code execution, multimodal input, and feedback should help selectively. Route-by-task interactions test the distinctions. The decomposition's empirical burden is to predict held-out route--task combinations or improve intervention choice without conceptual refitting. Surface improvement and truth-tracking improvement can come apart. 

---
# Polaris : Multi Agentic System for Conversational Enterprise Analytics 

**Authors**: Varuni H K, Soham Sarkar, Jay Kumar, Goutham Krishnan, Tanvi Johari, Avinash Bharadwaj, Santosh Hegde  

**Link**: [PDF](https://arxiv.org/pdf/2608.14246)  

**Abstract**: In today's fast-paced environment, the ability to swiftly access, understand, and act on data is no longer optional; it is essential. Yet most organizations remain data-rich but insight-poor, constrained by the complexity of querying, interpreting, and explaining enterprise-scale information. We present Polaris, a supervisor-led multi-agent framework for conversational enterprise analytics that bridges this gap. Polaris introduces Dynamic Task Coordination (DTC), a decision-theoretic orchestration layer that models agent-task assignment as adaptive bipartite matching, enabling real-time coordination, recovery, and optimization across specialized agents for querying, visualization, and reasoning. By coupling DTC with reason-first, ReAct-style agents, Polaris transforms natural-language queries into coherent analytical workflows that not only retrieve and visualize data but also explain the underlying "why." Evaluation on structured enterprise datasets demonstrates high semantic fidelity and answer relevancy, underscoring the potential of multi-agent orchestration to deliver trustworthy, end-to-end business intelligence at scale. 

---
# Attributing Preprocessing Invariance in Spectral Foundation Models 

**Authors**: Dongjun Wei, Hongyi Wu, Yinuo Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.14227)  

**Abstract**: Preprocessing invariance is an appealing goal for spectral foundation models: a frozen model should remain useful when laboratories preprocess spectra differently. It is usually measured by training a classifier under one preprocessing pipeline and testing it under another, with preserved accuracy read as evidence of learning. We revisit that reading, using a Raman foundation model as a case study. Such models normalize their inputs before any learned parameter is applied. If that normalization maps two differently preprocessed spectra to the same vector, the encoder receives identical inputs, so the invariance cannot be attributed to learning. For a normalization that uses each spectrum's own statistics, this happens exactly when one spectrum is a positive multiple of the other plus a constant. Several standard preprocessing operations take that form. The encoder should therefore be measured against the normalization alone, which has no learned parameters. On six Raman evaluation datasets, the model does not measurably outperform its own normalization. It improves on raw spectra, but so does the normalization alone. Training does improve the encoder over random initialization, and a controlled experiment shows that it learns to ignore a transformation only when that transformation reaches it. A numerical test settles which transformations a given normalization removes. Across released systems in five modalities, most normalizations already remove transformations of that form, and several of those systems claim that invariance as learned. Replicating the comparison on two of them shows no gain either. 

---
# MathForm: Scaling Mathematical Autoformalization with Knowledge Retrieval and Verification-Guided Refinement 

**Authors**: Lushi Pu, Weiming Zhang, Xinheng Xie, Zixuan Fu, Bingxiang He, Hengyu Zhao, Hongya Lyu, Xin Li, Jie Zhou, Yudong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14221)  

**Abstract**: Autoformalization is commonly framed as translating natural-language mathematical statements into machine-verifiable formal languages such as Lean 4. However, faithful formalization requires more than translation. Models must map mathematical concepts to the complex hierarchy of types and definitions in formal libraries such as Mathlib, while ensuring that generated statements preserve the meaning of the source propositions. Existing approaches struggle because they rely heavily on the model's parametric memory for library-specific knowledge, while common data construction pipelines often resort to filtering single-pass outputs and lack mechanisms for feedback-driven revision. To address these challenges, we introduce MathForm, an autoformalization framework for constructing verified training data through Mathlib knowledge retrieval and verification-guided iterative refinement. Before generation, a retrieval planner gathers relevant definitions and existing formalizations from Mathlib to guide the formalization generator. Generated statements are then revised using compiler diagnostics and semantic-consistency feedback. Using this framework, we construct FormalVerse, a Lean 4 dataset containing approximately 367K verified examples across diverse mathematical domains and sources. We then train MathForm-8B through supervised fine-tuning followed by reinforcement learning. Across six benchmarks, MathForm-8B achieves average Pass@8 rates of 88.06% under Syntax Check (SC) and 72.37% under Consistency Check (CC), outperforming multiple specialized 32B autoformalizers. On the challenging FATE-H and FATE-X subsets, it attains CC pass rates of 63% and 37%, exceeding the strongest specialized baselines in both cases. 

---
# A Generalized Parallelogram Rule for Proportional Analogies on Riemannian Manifolds 

**Authors**: Pierre-Alexandre Murena, Marcelo Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.14220)  

**Abstract**: Analogies are quaternary relations of the form "a is to b as c is to d", usually denoted a : b :: c : d. This notion is formalized in particular with the notion of proportional analogy, which imposes some constraints on the valid analogies. Whereas proportional analogies have been studied mostly in symbolic domains and in vector spaces, their use is limited in non-Euclidean spaces. In this paper, we introduce a proportional analogy relation in Riemannian domains, extending the parallelogram rule used for arithmetic analogies in Euclidean spaces. We illustrate the introduced analogy on various manifolds, such as the sphere, shape spaces and manifolds of probability distributions. 

---
# APTER: Adaptive Post-Training with Expert-Grounded Rubrics 

**Authors**: Xukai Wang, Liangqi Li, Zhiyue Xu, Jingang Zhou, Xiaoyu Shi, Jiansheng Cai, Bo Zhang, Zhe Li, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14212)  

**Abstract**: As large language models enter professional domains, they must satisfy domain constraints, include critical evidence, and provide complete reasoning rather than merely produce fluent responses. Existing post-training methods often rely on holistic preferences or outcome-level verification, while recent rubric-based methods usually generate rubrics independently for each query. In specialized domains, such unconstrained rubrics may omit critical requirements and vary across samples, hindering the diagnosis and targeted repair of persistent capability deficiencies. We propose APTER (Adaptive Post-Training with Expert-Grounded Rubrics), a framework that integrates structured domain knowledge into fine-grained evaluation, optimization, and diagnosis for specialized complex reasoning. First, expert-grounded rubric construction starts from an expert criteria framework built by domain experts, where each criterion represents a stable professional capability. For each query, APTER selects relevant criteria and instantiates them into query-level rubrics linked to their source criteria, turning reusable expert criteria into executable query-level supervision without reference answers. Second, adaptive post-training uses rubric verdicts as both optimization and criterion-level diagnostic signals. Aggregating low-scoring verdicts by criterion ID reveals persistent deficiencies and triggers targeted supervised fine-tuning updates during reinforcement learning. Experiments on mathematical reasoning and medical question answering show consistent gains across both domains. Across three model generations, APTER improves the mathematics and medical averages over the corresponding base models by up to 15.86 and 8.04 points, respectively. Code and rubric datasets are available at this https URL. 

---
# FreeBalance: Pre-Routing Online Moe Load Balancing via Residual Workload Prediction 

**Authors**: Pengfei Chen, Yize Wu, Shouxu Kuang, Ke Gao, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14205)  

**Abstract**: Load imbalance poses a major bottleneck to the efficiency of expert parallelism in distributed inference of Mixture-of-Experts (MoE) models. The most heavily loaded rank stalls global execution due to skewed routing distributions, directly increasing latency. While offline expert placement can alleviate persistent imbalance, practical multi-task serving workloads exhibit layer- and batch-dependent routing dynamics, making online load balancing indispensable. Existing approaches rely on routing statistics collected after each MoE router, requiring expert weight load or migration to begin only after routing decisions are available, consequently placing migration overhead on the inference critical path. In this work, we observe that online balancing can instead be largely overlapped with computation before target routing (e.g., attention), if routing distributions can be predicted accurately in advance. Therefore, we propose FreeBalance, a lossless online load-balancing framework that overlaps expert migration with preceding computation stages via residual workload prediction. FreeBalance leverages cross-layer similarities in hidden representations within the residual network to build a lightweight workload predictor. This enables proactive expert migration planning before routing decisions are available, creating substantial overlap between weight transfer and computation-heavy pre-routing stages. Furthermore, a cost model constrains the number of swaps to fully hide the synchronization overhead within the available window. Experiments across models and datasets show that FreeBalance reduces the max-to-mean rank load ratio by 32.8% and end-to-end prefill latency by 13.1%. Specifically, our method hides balancing overhead of an average of 5.1 experts per layer, which would otherwise account for about 8.5% of the critical-path latency. 

---
# Can Language Models Understand mmWave Data? Benchmarking Large Language Models for mmWave Radar-Based Human Understanding 

**Authors**: Jeongwan Shin, Jaehyeon Kim, Donguk Ko, Jaeho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14179)  

**Abstract**: Large language models (LLMs) have shown remarkable reasoning and generative capabilities, motivating their use as universal reasoning engines for perception. While modern approaches such as vision-language models (VLMs) have attempted to incorporate reasoning capabilities into visual sensing, the integration of LLMs with the millimeter-wave (mmWave) modality-despite its unique advantages under low light and occlusion-remains largely unexplored. The principal bottlenecks stem from the scarcity of radar language pairs, severe cross-dataset heterogeneity, and the absence of a foundational mmWave encoder. We address this gap through a minimal textualization interface that serializes each mmWave point cloud into concise natural language, allowing off-the-shelf LLMs to operate in a question answering (QA) setting. Building on this, we present mmWave-QA, the first benchmark for language-conditioned mmWave human perception. mmWave-QA aggregates heterogeneous public mmWave datasets and harmonizes them via calibration-aware preprocessing and global taxonomy alignment, while providing natural language QA. Spanning six scenarios and five QA tasks, the benchmark enables standardized evaluation across diverse mmWave hardware and experimental conditions, establishing a foundation for scalable research on mmWave-LLM integration. We further evaluate and analyze LLMs on our mmWave-QA, highlighting their zero-shot reasoning potential for radar perception, as well as their robustness under visual degradation. 

---
# BiasTrace: Linking Reasoning Behaviours to Biased Outputs in LLMs 

**Authors**: Varsha Ramineni, Hossein A. Rahmani, Jerome Ramos, Karin Sevegnani, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2608.14161)  

**Abstract**: LLMs exhibit social biases that can produce inaccurate and discriminatory inferences, posing risks in high-stakes applications. While prior work has made progress in measuring and mitigating bias, it largely focuses on final outputs of models, with limited understanding of the mechanisms that produce biased outcomes. Recent advances in LLM reasoning offers a new lens for investigating bias, yet the link between reasoning and bias remains poorly understood. Existing approaches focus primarily on final answer correctness or explicitly biased language, overlooking different behaviours in reasoning that can drive biased outcomes. We introduce BiasTrace, an annotation scheme for labelling reasoning behaviours in model-generated traces and linking them to biased outcomes. BiasTrace captures bias-specific behaviours (e.g., unsupported demographic assumptions) as well as general reasoning patterns that may implicitly contribute to bias (e.g. overthinking). We apply BiasTrace to reasoning traces in bias-sensitive contexts, scaled using validated LLM-as-a-judge methods, producing a large annotated dataset. Our analysis shows that biased outputs often stem from subtle reasoning behaviours rather than explicitly biased language, and that reasoning-level annotations improve bias detection. We further show that BiasTrace behaviours can be exploited for inference-time mitigation. These findings underscore the importance of examining a broader range of reasoning patterns to better understand bias in LLMs. 

---
# Removing Temporal Note Redundancy Improves Multimodal Reinforcement Learning for Medicine 

**Authors**: Chenran Weng, Joo Seung Lee, Malini Mahendra, Anil Aswani  

**Link**: [PDF](https://arxiv.org/pdf/2608.14157)  

**Abstract**: Mechanical ventilation is a critical life-support intervention, requiring dynamic adjustments to ventilator settings as a patient's condition evolves. While reinforcement learning (RL) offers a promising framework for optimizing these sequential decisions, standard approaches rely primarily on structured electronic health record (EHR) data, missing crucial clinical context recorded in free-text notes. Integrating longitudinal clinical notes into RL state spaces is challenging because notes are heavily inflated by temporal redundancy, such as copy-forward text, templating, and repetitive documentation, which dilutes time-local updates and degrades state representation quality. To address this, we propose a redundancy-aware multimodal state representation framework that explicitly removes duplicated note text over time before policy learning. We evaluate two computationally efficient temporal decomposition strategies for removing duplicated note text: (1) an embedding-space decomposition using singular value decomposition on local history subspaces, and (2) an interpretable sentence-level diff operation that filters out previously documented sentences before text encoding. Using real-world ICU data, we demonstrate that state representations constructed by stripping temporal note redundancy significantly outperform both structured-only and raw-note baselines across multiple off-policy evaluation methods (Model-Based Rollouts, Fitted Q-Evaluation, Weighted Importance Sampling, and Weighted Doubly Robust Evaluation). Our findings show that explicitly isolating new clinical information from repeated note text yields higher-quality state representations and directly improves RL performance for clinical decision support. 

---
# Towards Efficient Multimodal and Multilingual Opinion Extraction for STI: A QLoRA-Based Fine-Tuning Approach 

**Authors**: Sheng Hong, Xuanqi Wang, Jiacheng Wang, Yuwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14152)  

**Abstract**: Recent advances in large language models (LLMs) have reshaped semantic analysis. Opinion Extraction (OE) for Science and Technology Intelligence (STI) requires concise core opinions from large information streams. Off-the-shelf models struggle to filter noise from these streams and show limited structured-output reliability in zero-shot multilingual and multi-modal settings. To address information overload and extraction defocus, this study proposes a multimodal core-opinion extraction framework in which visual evidence serves as a contextual anchor for textual judgment. Using VideoLLaMA2 (VL2) and VideoLLaMA2.1 (VL2.1) as the base models, we apply Quantized Low-Rank Adaptation (QLoRA) fine-tuning on a curated dataset of 2,194 multilingual and multimodal samples. Under the selected Image-Augmented setting, fine-tuned VL2.1 generates structured JSON core-opinion outputs, achieving 64.98% Precision, 42.15% Recall, 51.14% F1-score, and 74.00% sample-level accuracy. Relative to the zero-shot VL2.1 setting, it raises the F1-scores of Spanish and Russian from 4.83% and 0.45% to 46.05% and 51.93%, respectively. The framework further incorporates a Fuzzy Cumulative Prospect Theory-based post-extraction triage module for case-level value assessment, providing a case-level value signal for downstream STI screening. 

---
# QuaSAR: Quantization Compensation via Stable Activation-Aware Rank Truncation 

**Authors**: Lin-Fa Lee, Yi-Yu Chang, Kuo-Hei Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2608.14149)  

**Abstract**: Recent training-free post-training quantization methods restore model accuracy through closed-form residual compensation. To constrain additional model storage overhead, several existing methods gate layer selection by goodness-of-fit, retaining only those layers whose compensation yields a positive residual fit score and discarding the rest. In this paper, we show that, under the low-bit W4A4 setting, this gating mechanism fails to distinguish poorly predictable quantization error from numerical solver failure. Rank-deficient input activations yield severely ill-conditioned or numerically singular Gram matrices, causing the closed-form solver to become unstable and produce spuriously negative fit scores. Consequently, existing goodness-of-fit gates misclassify affected layers as uncompensable and discard them. Many of these discarded layers can nevertheless provide substantial error recovery when their compensation is computed using a numerically stable solver. To address this problem, we propose a parameter-free truncated pseudoinverse solver which removes collapsed directions prior to inversion. On ViT-B with the W4A4 setting, our training-free method achieves 81.42\% top-1 accuracy, outperforming prior post-training methods and fine-tuning-based baselines. Combined with joint low-rank and quantization compression, the proposed method reaches a deployable operating point of 80.26\% accuracy at 54.7 MB, providing a well-balanced trade-off between model size and accuracy. 

---
# Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost 

**Authors**: Xiaodi Huang, Ziyi Ding, Jingtian Wan, Yuchen Liu, Yuan Zhang, Xiao-Ping Zhang, Jiayu Chen, Zhang Zhang, Tao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14125)  

**Abstract**: LeWM is a lightweight visual world model that learns latent dynamics end-to-end from pixels and ranks candidate action sequences by the distance between their predicted endpoints and the goal. However, LeWM has two limitations. First, during training, it learns local next-step transitions without evaluating complete trajectories relative to the task goal. Second, during planning, it ranks candidates solely by predicted endpoint distance. Because model predictions may differ from actual execution outcomes, the candidate whose predicted endpoint is closest to the goal may not perform best when executed in the environment. The evolution of the complete predicted trajectory can therefore provide complementary information beyond endpoint distance. To address these limitations, we propose Traj-LeWM, which retains LeWM's local-dynamics objective and endpoint score while introducing a goal-conditioned latent trajectory cost (LTC) that aggregates trajectory-level information as a complementary signal. During training, LTC-based trajectory-preference supervision complements next-step prediction in shaping the shared representation. During planning, LTC is combined with endpoint distance to incorporate intermediate-path information into candidate ranking. With joint endpoint-plus-LTC scoring, Traj-LeWM outperforms LeWM on Push-T, OGBench-Cube, Reacher, and Two-Room by $3$, $14$, $7$, and $7$ percentage points, respectively. Controlled experiments and ablations further verify the complementary roles of trajectory-level representation shaping and path-aware candidate ranking. 

---
# Reinforcement Learning-Based Production Scheduling in an Industry-Based Coating Scenario Using the Digital Model Playground 

**Authors**: Arne Kröger, Ralf Buschermöhle, Wilhelm Hasselbring, Henrik Wilbers  

**Link**: [PDF](https://arxiv.org/pdf/2608.14122)  

**Abstract**: Production scheduling in complex manufacturing environments is challenging when sequence-dependent setup times, stochastic disturbances, and due-date constraints must be addressed simultaneously. While reinforcement learning (RL) methods have shown promising results in research, most studies rely on simplified benchmark processes, limiting their industrial relevance. This paper demonstrates the applicability of RL-based scheduling in an industry-inspired coating process that reflects practical complexities such as sequence-dependent setup times, machine breakdowns, and variable utilization. The open-source Digital Model Playground (DMPG), a discrete event simulation framework, is used to model the scenario and to train RL agents. Two standard algorithms, Deep Q-Networks and Proximal Policy Optimization, are benchmarked against conventional dispatching rules to illustrate feasibility and to provide a transparent testbed for further research. Results indicate that RL-based scheduling achieves balanced improvements across key performance indicators, with PPO delivering the most robust performance. The main contribution of this work is to bridge the gap between academic research and industrial practice by validating RL-based scheduling in a realistic, shareable scenario and by providing a reusable open-source framework for future studies. 

---
# A Graph-Based Reinforcement Learning Framework for Structured Drift Diagnosis and Recovery in Autonomous LLM Agents 

**Authors**: Ismail El Hamraoui, Sagar Jose, Nicolas Bureau, Robert Plana  

**Link**: [PDF](https://arxiv.org/pdf/2608.14109)  

**Abstract**: Autonomous LLM agents are increasingly deployed in complex real-world workflows, yet they remain vulnerable to runtime behavioral drift, a silent deviation from the original task that can lead to irreversible side effects on external systems. Existing approaches address drift at the prompt level but lack structured mechanisms for step-level detection, risk assessment, and recovery decision. Because the main task-executing agent is often a large and expensive model that cannot be re-trained on every deployment, this work targets a plug-and-play recovery module instead. It introduces a graph-based framework in which a single small language model is trained via reinforcement learning to specialize at each node of a recovery graph, external to the main agent. Each node has a precise role\,: drift classification, operation detection, risk evaluation, or final decision and the model learns to produce structured XML-formatted reasoning adapted to that role. Training combines rule-based structural rewards with an LLM-as-judge semantic-quality signal, so that the model is graded both on how it answers (schema and length) and on what it says. Experiments on the public AppWorld benchmark show that the method generally exploits information about the suspected drift onset to issue correct recovery decisions using a small language model. In addition, the trained small language model reliably respects the prescribed output schema and produces semantically appropriate content in each field according to its assigned node role. 

---
# Retrieval Grounding Latent Reasoning for Dense Retrieval 

**Authors**: Gang Zhou, Xiongxi Yu, Hu Tian, Yang Wei, Lu Pan, Ke Zeng, Shibiao Xu, Xiaolong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14107)  

**Abstract**: Reasoning-intensive retrieval requires text representations to capture not only semantic similarity, but also the reasoning needed to determine relevance under a given retrieval instruction. Existing reasoning-enhanced embedding models improve retrieval by incorporating reasoning information into dense representations, yet their supervision is typically dominated by the final retrieval objective. As a result, latent reasoning trajectories may learn shortcut reasoning patterns that preserve retrieval performance without producing meaningful incremental retrieval gains. We propose Retrieval Grounding Latent Reasoning (RGLT), a latent reasoning framework for dense retrieval that explicitly connects intermediate latent transitions with retrieval improvements. RGLT performs non-autoregressive reasoning in hidden space through an instruction-conditioned latent reasoning trajectory constructed from silent tokens. It combines process-supervised explicit-to-implicit distillation with retrieval-grounded supervision, using stage-wise CoT reconstruction to shape intermediate latent states and retrieval-effect credit to optimize incremental retrieval gains across the latent reasoning trajectories. Experiments on reasoning-intensive retrieval benchmarks show that RGLT consistently outperforms strong baselines while preserving efficient embedding inference. 

---
# Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers 

**Authors**: Thiago Sandoval, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14089)  

**Abstract**: Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every classifier-dataset combination, catching up to 0.81 of previously missed unsafe content without modifying the underlying classifier. In a deployment study with ten attack campaigns, each a harm category held out of RCV's training, RCV detects every campaign in a dedicated injection panel; in the maintenance census most drift episodes are repaired without updating the classifier, and the fine-tune is reserved for the residual episodes that repair does not restore. 

---
# A Pathway to General-Purpose Scientific AI: Multimodal Comprehension of Scientific Images 

**Authors**: Jennifer D'Souza, Fahad Ahmed, Cecilia Andrea Bustamante Andrade, Lina Frolova, Poorani Gnanasambandan, Dilshad Hussain, Muhammad Uzair Khan, Nkembeng Kevin Nkengfoa, Paul Praveen J., Fabio Priante, Sjoerd Franciscus van der Werf, Thomas Frederik Jan van Roeden  

**Link**: [PDF](https://arxiv.org/pdf/2608.14075)  

**Abstract**: Scientific figures and tables encode essential experimental evidence, yet remain difficult for digital libraries and multimodal AI systems to retrieve and interpret. The ALD/E-ImageMiner benchmark and ICDAR 2026 Competition on Information Extraction from Atomic Layer Deposition/Etching Scientific Figures provide 1,951 figures from 205 publications, expert-annotated for classification, data table extraction, summarization, and visual question answering. In these companion proceedings, we present a forward-looking perspective on how the benchmark can guide future scientific-image challenges. We examine how its tasks probe capabilities from visual and quantitative reading to domain-grounded reasoning and evidential justification, and how Bloom-informed question design can support deeper scientific understanding. We propose "scientific conceptual understanding from images" as a long-term benchmark objective, with future directions including broader domains and figure types, contextual and cross-document synthesis, hypothesis evaluation, provenance, uncertainty, counterfactual grounding, and open-ended multimodal research. This perspective connects the ICDAR 2026 challenge to a broader agenda for machine-actionable scientific visual knowledge and verifiable multimodal scientific AI. 

---
# Mandato: Protocol-Level Enforcement of Digitally Signed Mandates on AI Agent Actions with Cryptographically Chained Audit Trails 

**Authors**: Giovanni Racioppi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14074)  

**Abstract**: AI agents increasingly act on external systems through standardized tool-calling protocols such as the Model Context Protocol (MCP), yet no infrastructure layer constrains their actions to what a principal has verifiably authorized: authorization logic lives in application code, is neither signed nor independently auditable, and the resulting logs lack evidentiary value. We present Mandato, a governance proxy that enforces digitally signed mandates on agent actions at the protocol level. A mandate is a machine-readable, cryptographically signed authorization artifact specifying which tools an agent may invoke, under which parameter constraints and contextual conditions, for how long, and on whose behalf; the proxy evaluates every tool call against the applicable mandate chain, blocks non-conforming calls in line, and records every decision -- permit, deny, and the evidence for each -- in an append-only, hash-chained audit log designed for evidentiary use and periodically anchored via qualified timestamps. The mandate is deliberately modeled on the civil-law institution of delegation of authority, making the artifact legible to lawyers and auditors, not only to engineers. We give the mandate model and its decision semantics, the reference architecture as an MCP-transparent proxy with separated decision and enforcement points, and a mapping of the mechanism onto EU AI Act Articles 12 and 14, GDPR accountability, NIS2, and eIDAS 2, including a roadmap to qualified attestation through Qualified Trust Service Providers (QTSPs). We describe the implementation status of the reference system and a quantitative evaluation plan covering enforcement overhead, audit completeness, and tamper-evidence verification cost. 

---
# Scaling Domain Data Repetition in LLM Pretraining 

**Authors**: Jingwei Li, Xinran Gu, Rui Dai, Xintong Hao, Chengyin Xu, Yan Wu, Shuran Zheng, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14071)  

**Abstract**: As large language models scale, their training-token budgets must also increase to maintain an appropriate tokens-per-parameter ratio (\(\mathrm{TPP}\)). However, high-quality domain data is much harder to scale than general web data. As model size and the training-token budget increase, its fraction in the training mixture tends to decrease. Repeating the available high-quality data provides an effective way to counteract this dilution, but excessive repetition may lead to overfitting. We study this trade-off under practical LLM scaling, where the training-token budget grows proportionally with model size. For a fixed domain, we first find that, surprisingly at a fixed \(\mathrm{TPP}\), the optimal repetition count mildly increases with model size. Across different domains, we find that the optimal repetition count is strongly negatively correlated with the final validation loss of a domain: domains with lower loss can generally benefit from more repetitions. In contrast, the amount of unique domain data is only weakly related to the optimal repetition count. These findings suggest that repetition counts tuned on smaller proxy models with the same \(\mathrm{TPP}\) can provide a practical estimate for larger models. 

---
# Benchmarking data-driven material models on the classic Treloar dataset 

**Authors**: Hagen Holthusen, Moritz Flaschel, Denisa Martonová, Ellen Kuhl  

**Link**: [PDF](https://arxiv.org/pdf/2608.14063)  

**Abstract**: Machine learning is rapidly reshaping constitutive modeling, offers new ways to learn material behavior directly from experimental data, and challenges long-established modeling paradigms. But with a growing number of machine-learning-based approaches available, how do they compare in practice? In this paper, we use the classic experimental data of Treloar to benchmark popular frameworks for hyperelasticity: (Generalized-Invariant) Constitutive Artificial Neural Networks, Physics-Augmented Neural Networks, (Adaptive) Material Fingerprinting, and Efficient Unsupervised Constitutive Law Identification & Discovery. We compare their fitting performance, computational cost, hyperparameter sensitivity, and ease of implementation. Furthermore, we discuss the trade-offs between predictive accuracy and model complexity. The latter is assessed by quantifying both the number of material parameters in the discovered models and the computational time required to evaluate the constitutive model and its derivatives. The results show that all methods can reproduce the benchmark data remarkably well. Rather than identifying a single winner, we highlight the strengths and limitations of each approach and provide practical guidance for their use. The source code for all six methods, including the training and comparison scripts, as well as all results and data used in this study, is publicly available via this https URL. 

---
# Demystifying Agent Skills: Why They Work-Until They Don't 

**Authors**: Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao, Rui Cao, Mengdi Wang, Shilong Liu, Yijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14036)  

**Abstract**: Skills have emerged as a practical and effective approach for enhancing LLM agents at inference time through structured packages of knowledge. However, existing evaluations largely measure whether skills improve aggregated task success, leaving a more fundamental question underexplored: \emph{\textbf{When do skills help, why do they work, and where do they fail?}} Through controlled experiments across various benchmarks, agent harnesses and LLMs, we isolate the effects of representation, outcome annotation, retrieval difficulty, and cross-framework robustness of skills. To further answer this question, we design a contrastive study that combines controlled quantitative experiments with paired trajectory analysis. We normalize 8,135 trial records from controlled experiments and retain 238 valid unique labels from 240 open-coded records. We consolidate these observations into a taxonomy of three high-level categories and twelve skill-use modes: skills work when noisy trajectories become procedural anchors that stabilize execution. Skills improve over Workflow Memory by 6.06 points in matched comparisons. Procedural anchoring accounts for 65.7\% of skill cases, versus 4.5\% for explicit knowledge injection, showing that skills stabilize action rather than inject missing facts. Retrieval is a separate bottleneck: as pools grow from 5 to 100, actual-use precision falls from 29.6\% to 3.3\%. Confusable distractors impair offline identification, yet downstream success remains stable; exact ground-truth invocation is neither sufficient nor necessary. Skills fail under brittle assumptions, incompatible contexts, or insufficient adaptation. These findings move evaluation beyond aggregate success rates and guide reliable self-evolving agents. 

---
# Agent-Orchestration in Autonomous Chip Design 

**Authors**: Linyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14035)  

**Abstract**: Recent developments in large language models (LLMs) and tool-using agents encourage people to explore the potential of using agents in chip design. The core question is what kind of AI we really need in such a sophisticated industry.
To this end, we bring the idea of modeling a chip-design superintelligence as an enormous \textit{AI-organization}. 

---
# Residual Dominance as a Structural Account of Last-Item Reliance in Causal Self-Attention Recommenders 

**Authors**: Keito Kozaki, Keigo Sakurai, Ren Togo, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2608.14021)  

**Abstract**: Transformer-based sequential recommenders with causal self-attention often rely heavily on the most recent interaction at inference time, but how this behavior is structurally expressed in the representation used for prediction remains unclear. We combine prediction-time diagnostics with norm-based analysis of the full attention block. First, we show that SASRec-style models exhibit highly localized last-item reliance. We then find that, although self-attention aggregates contextual information, residual addition sharply shifts the full-block representation toward same-position contributions, which we term residual dominance. To probe this interpretation, we use inference-time residual scaling as a controlled diagnostic intervention. Changing the residual strength induces a monotonic trade-off between structural mixing and last-item reliance, while reducing residual strength recovers a subset of final-position misses for which representations at non-final positions already rank the ground-truth item correctly. Our results provide a structural account linking extreme last-item reliance to residual dominance at inference time. The code is publicly available. 

---
# Buy the Rumor, Sell the News: When Is News Priced In? 

**Authors**: Alireza Kargarzadeh, Nariman Khaledian, Navid Parvini, Sid Ghatak, Arman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2608.14014)  

**Abstract**: Two old market sayings hold that news is already priced in by the time it is published, and that the rumor is bought while the news is sold. Both place the price move associated with a piece of news before and at publication rather than after it. Whether the claims hold, for which kinds of news, and by how much are basic questions about how fast markets absorb public information. We test them on 4.57 million financial news articles covering roughly 3,000 US stocks (2023-2026). A large language model teacher, distilled into a compact classifier through active learning, assigns each article one of 17 event tags and five attributes; articles are clustered into stories to separate first reports from follow-up coverage; and beta-adjusted abnormal returns are measured around the resulting 1.68 million stock-day events, with 364,405 neutral-sentiment events as a placebo group. Three results follow. First, the price move associated with news concentrates before and at publication: pooled across all signed events, the cumulative move in the news direction by the close of publication day is 2.8 times its value 20 days later, and for rumor-flagged events the rumor day captures the entire move while the subsequent confirmation contributes nothing. Second, measured against the placebo of comparable stocks, markets underreact to numbers and overreact to stories: quantified fundamental news (earnings, dividends, guidance, analyst actions) keeps drifting in the direction of the news for weeks, while soft story-driven news (launches, macro commentary, leadership) gives back its move. Third, news carries width as well as direction: publicity raises volatility before the publication day, and volatility declines once the news is out, because publication resolves uncertainty. The study also produces a table of measured drift for each event tag, usable as a prior in news-conditioned forecasting models. 

---
# Simulation-Driven Vehicular Traffic Data Augmentation: Extending Sensor Coverage Through Virtual Sensing 

**Authors**: Davide Andrea Guastella, Eladio Montero Porras, Evangelos Pournaras, Gianluca Bontempi  

**Link**: [PDF](https://arxiv.org/pdf/2608.13993)  

**Abstract**: Urban traffic management relies on sensor networks whose spatial coverage is limited by deployment costs and privacy regulations. Machine learning models trained on such sparse data cannot generalize to unmonitored locations and must be retrained whenever the sensor infrastructure changes. We propose a simulation-based methodology that addresses this problem by generating augmented traffic count datasets in which each physical sensor is replaced by a virtual sensor placed at a surrogate location in the road network. Virtual sensors are selected by a graph-search heuristic that jointly maximises vehicle-flow continuity and traffic-metric similarity between the original and surrogate locations, while enforcing a minimum spatial displacement to ensure diversity of observed traffic conditions. We validate the method on two Belgian cities: Brussels, using a calibrated model, and Namur, using synthetic models. The augmented datasets preserve the bimodal daily demand profile and the dynamics of traffic at the observed locations. 

---
# Content Depth Matters in Short-Video Recommendation: Rethinking the Attention Economy 

**Authors**: Liwei Deng, Jing Jiang, Zhiwei Li, Yang Wang, Guodong Long  

**Link**: [PDF](https://arxiv.org/pdf/2608.13990)  

**Abstract**: Driven by the attention economy, short-video Recommender Systems (RSs) are primarily optimized to maximize user engagement by promoting videos that capture attention within seconds. These systems inherently favor shallow-content videos that are effective at attracting immediate attention. However, growing evidence suggests that prolonged exposure to such content may negatively affect users' cognitive engagement and mental well-being, raising concerns about the long-term societal impact of the short-video platform. To tackle this challenge, this paper introduces a new metric, the \textbf{Content Depth Score (CDS)}, to quantify the content depth of short videos. CDS measures the extent to which a video is expected to stimulate higher-order cognitive processes, using a seven-level scale grounded in established theories of cognitive psychology and learning. As an initial step toward this vision, we present \textbf{SCOPE-Bench}, the first benchmark for content-depth evaluation in short-video recommendation. Built upon a large-scale open-source short-video dataset, SCOPE-Bench provides CDS annotations for 150K videos, enabling systematic evaluation of RSs from a cognitive-content perspective. Leveraging SCOPE-Bench, we evaluate 13 representative RSs and reveal a consistent preference for shallow-content videos. Moreover, we find that these algorithms recommending cognitively deep content are only marginally better than random selection, highlighting a previously overlooked limitation of existing recommendation objectives. Our code and datasets are available at this https URL. 

---
# Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead 

**Authors**: John T. Halloran  

**Link**: [PDF](https://arxiv.org/pdf/2608.13987)  

**Abstract**: Nanbeige4.2-3B is a 3B-parameter agentic model built around a Looped Transformer (LT) that reuses one stack of layers for a second forward pass, adding effective depth without additional parameters. Evaluated on Apple Silicon (MPS), we identify five independent bugs which prevent the released checkpoint from running via Hugging Face transformers out of the box (including a silently-zeroed RoPE buffer and calls to removed transformers cache APIs). Furthermore, we show that fixing these bugs is still not sufficient for agentic tasks, due to the LT's layer-reuse strategy (which effectively doubles peak attention memory) used to achieve parameter efficiency. We thus introduce a chunked-prefill strategy which alleviates the incurred memory-capacity penalty, extending allowable context width by $2.7 \times$ on 32~GiB shared memory. However, even with the reduced memory overhead, we show that patches are required to render Nanbeige4.2-3B usable; resolving both system prompt and MPS-native memory bugs finally allows reliable evaluation on standard MCP and tool-calling benchmarks. On a subset of MCPMark, the debugged model completes up to 30\% of real agentic tasks (up from the original's 0\%), while, on BFCL, it is near-perfect at single tool calls (yet fails the majority of multi-tool tests). We release the patched checkpoint, system prompt optimizer, and evaluation harnesses at this https URL. 

---
# Implementing Computational Law in Wolfram Language for the Governance of Artificial Intelligence 

**Authors**: James K. Wiles  

**Link**: [PDF](https://arxiv.org/pdf/2608.13958)  

**Abstract**: How do we govern AI systems whose reasoning we cannot fully inspect? Governance does not require understanding a system's reasoning. It requires stating what the system is obliged, permitted, and forbidden to do, and checking whether it complied. I present an implementation of Reified Input/Output Logic, the formalism behind the DAPRECO knowledge base, in Wolfram Language: the core I/O axioms, obligations, permissions, constitutive norms, reified eventualities, and temporal operators. I then test whether GPT-4 can translate English legal statements into the formalism, and report the failures: hallucinated functions, omitted temporal scope, deviation from the formalism, and (in the worst cases) code that runs, reads plausibly, but silently encodes the wrong norm. A case study, an AI guard dog operating under a computational contract, shows how formalized rules can extend from a contract directly into the operational code of an embodied agent, producing symbolic, auditable justifications for its behaviour. I argue that computational law can be used as a governance tool and that a desirable goal would be to formalize the law that can and ought to be programmatically executable. 

---
# HELIX: Model-Harness Co-evolution for Recursive Self-Improvement 

**Authors**: Tianyu Fan, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13951)  

**Abstract**: Scaling agent capability has largely focused on improving the model, yet an interactive agent acts through a runtime harness that mediates context, tools, control flow, and stopping. The harness shapes both what a model can accomplish and the trajectories from which it learns. This coupling motivates model-harness co-evolution for recursive self-improvement: build harnesses for a fixed model, update the model from verified sibling trajectories, and rebuild the harnesses as model capabilities change. Realizing this loop requires a controlled way to evolve harnesses while preserving intervention identity and effect. We present HELIX, a source-traceable substrate for harness evolution. HELIX decomposes agent systems into typed ports, reusable atoms, recipes, product shells, and runtime policies. It makes interventions explicit and auditable while retaining trajectories, test outcomes, and provenance. Harness evolution thus serves two linked roles: improving fixed-model execution and producing matched successes, regressions, near misses, and alternative solutions as data for subsequent model improvement. We evaluate HELIX in one evolution round on code repair. A 65-candidate portfolio discovers a fixed harness that improves task coverage by 4.0% over Pi, while the full portfolio exposes up to 58.0% more verified coverage through complementary sibling behavior. Selected candidates are assessed with repeated runs and the SWE-bench evaluator. A 200-slot sibling slice yields 438 verified SFT, critic, filter, and preference records. These results show how harness, model, and data form a feedback system: harness evolution expands current capability and creates learning signal for the next model; model updates motivate the next round of harness evolution. HELIX provides an auditable interface for studying this recursive process. Code is available at this https URL. 

---
# AI Research Preference Models 

**Authors**: Thomas Simon Foster, Bassel Al Omari, Tingchen Fu, Thomas Mann, Carl Domond, Lucia Cipolina-Kun, Bhavul Gauri, Muna Aghamelu, Alexander D. Goldie, Eryk Helenowski, Jean-Christophe Gagnon-Audet, Alberto Pepe, Saba Nazir, Daniel Izcovich, Noam Levi, Rishi Hazra, Karen Hambardzumyan, Nicolas Baldwin, Xian Li, Martin Josifoski, Paris Giampouras, Masoud Jalili Sabet, Anya Sims, Hela Momand, Tatiana Shavrina, Despoina Magka, Jason Weston, Yulin Wang, Anirudh Goyal, João Henriques, Yoram Bachrach, Emily McMilin, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2608.13940)  

**Abstract**: AI research agents (AIRA) can now propose, implement, and evaluate their own machine learning experiments, but progress on frontier tasks is throttled by cost: a candidate solution can be written in minutes, whereas evaluating it can take hours to days of GPU time. An agent can therefore propose far more candidates than it can afford to run, and its progress depends on its research preference: how it allocates a fixed execution budget across many candidates. We introduce AI Research Preference Models (RPMs) that predict which of multiple candidate solutions are most worth executing, without paying the cost of executing them all. We build RPMs from frozen pretrained language models (with no task-specific training), in two forms: an inference-only model that reasons over candidate plans, code, and prior executed solutions, and an agentic model that additionally runs small-scale pilot experiments before deciding. We integrate both into the AIRA-dojo search agent and evaluate on AIRS-Bench, a recent benchmark of machine learning research tasks for AI research agents. The two variants raise the average normalized score from 0.684 to 0.711 and 0.729 respectively, and reach the unguided agent's 24-hour performance in roughly 15 hours, using less than two-thirds of its execution budget. Our best RPMs also yield new state-of-the-art results on two AIRS-Bench tasks. 

---
# Never the Number: Structural Abstention for AI Systems Whose Answers Are Consumed as Fact 

**Authors**: Zhelun  

**Link**: [PDF](https://arxiv.org/pdf/2608.13926)  

**Abstract**: Large language models have made natural language interfaces to databases (NLIDB) newly credible, but LLM text-to-SQL systems fail in a way that matters for deployment: a hallucinated column or a mis-aggregated total yields a fluent wrong answer, indistinguishable at the point of use from a right one. Where the consumer cannot inspect the generated query, as in enterprise AI deployments and operational dashboards, and increasingly where the consumer is a tool-using agent rather than a person, accuracy alone is insufficient: nothing marks which answers to distrust. This is a reliability problem before it is an accuracy problem.
We propose an architectural pattern for such systems, a trusted kernel with a generative shell, resting on one invariant: a component that can fabricate may influence which question the system answers, never which value it returns. A generative shell interprets underspecified input and phrases replies; a deterministic kernel matches fully specified questions against a bounded set of answerable question shapes and compiles them to queries by deterministic execution. The two meet at a confirmation the user reads before any value is computed, and requests the kernel cannot express are declined rather than approximated. We call this structural abstention, and distinguish it from the statistical abstention of selective prediction and calibrated confidence: refusal here needs no confidence estimate, because unanswerable requests are unrepresentable.
We specify the pattern implementation-independently, give a five-decision recipe and work it across three domains, extend the invariant from returned values to the actions of agentic systems, and report a two-year production case study alongside two generative alternatives, a fine-tuned parser and a tool-retrieval agent. We close against enterprise and reliability benchmarks published since. 

---
# When Personal Memory Has No Single Answer: Evaluating LLM Agents under Irreducible Conflict 

**Authors**: Lu Yang, Shusheng Xu, Zhuoran Li, Tongkai Yang, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13921)  

**Abstract**: LLM agents increasingly maintain personal memory across sessions, but it can conflict. Preferences depend on context, behavior evolves, and sources can conflict. When a query lacks context, time, or source authority to interpret conflict, treating one memory as definitive converts unresolved conflict into an unjustified, overconfident action. Existing benchmarks recover one answer from conflicting evidence, overlooking whether agents recognize underdetermination, preserve alternatives, seek missing information, and choose appropriate actions. We introduce \underline{T}esting \underline{A}gents' \underline{N}avigation of \underline{G}enuine, \underline{L}atent, and \underline{E}ntangled Memory Conflicts (\textsc{TANGLE}), a benchmark for genuinely unresolvable memory conflicts. It comprises 541 instances across 40 personas and three types: Context-Partitioned Conflict (CPC), Behavior-Oscillation Conflict (BOC), and Source-Contradiction Conflict (SCC). We evaluate two tracks---an oracle track with curated memory and a pipeline track that extracts memory from multi-session dialogues---on five dimensions: conflict perception, causal reasoning, confidence calibration, clarification seeking, and memory faithfulness. Experiments reveal pipeline challenges. With curated memory, models recognize conflicts more reliably than they calibrate actions or seek targeted clarification. With end-to-end pipeline memory, extraction fails to preserve conflict-bearing relations needed for downstream reasoning. Policy comparisons show fixed rules are insufficient when actions must reflect conflict. These findings motivate Conflict-Aware Action Policy (CAAP), which adapts actions to each conflict using available evidence. \textsc{TANGLE} frames conflict handling as recognizing underdetermination, retaining conflicting evidence, and acting without forcing a definitive answer. 

---
# MemoryLake on MemoryArena: A Matched Study of Agent Memory Backends 

**Authors**: Chaoqun Zhan, Qiang Zhou, Guannan Li, Zhenqiang Huang, Qianjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13883)  

**Abstract**: Most agent-memory benchmarks test post-hoc recall, whereas MemoryArena evaluates whether memory supports interdependent, multi-session task completion. We compare MemoryLake, a structured multi-track memory backend, with Mem0, text-embedding-3-small vector RAG, and a long-context control across all five MemoryArena domains. The systems share the same agent framework, requested gpt-5-mini model alias, task samples, and scoring code; the memory integration is the intentionally changed component. Because each backend bundles write, retrieval, consolidation, budgeting, and prompt-assembly choices, the study is a matched system-level comparison, not a representation-only ablation or a cost-matched experiment. On the shared evaluation sets, MemoryLake has the highest observed success rate (SR) in mathematics (9/40), physics (12/20), and progressive retrieval (4/20). Every system has zero SR in travel planning, and web shopping yields a single bundle-level success (long context, 1/150); MemoryLake ranks third on both the travel soft process score and shopping step match. Following MemoryArena's suite-level convention, a post-hoc equal-weight average over the five SRs is 20.5% for MemoryLake versus 13.6% for the best comparator. These are point estimates: sample sizes are modest, confidence intervals overlap, and we do not report paired significance tests. A separate MemoryLake-only run over all 221 progressive queries yields a failure-counted SR of 26.7% (59/221) and is not a baseline comparison. The results support a workload-dependent view of memory backends and an observed lead among the four evaluated systems on the shared sets; they do not establish benchmark-wide state of the art or a causal advantage of representation structure. 

---
# Joint Optimization of Memory and Computing Frequency for Energy-Efficient DNN Inference 

**Authors**: Yunchu Han, Zhaojun Nan, Sheng Zhou, Zhisheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13863)  

**Abstract**: Deep neural network (DNN) inference on mobile devices often incurs high latency and energy consumption due to limited computing and memory resources. To enable energy-efficient DNN inference, most existing studies focus on dynamic voltage and frequency scaling (DVFS) for adjusting the computing frequency, while the impact of memory frequency on the inference performance has been greatly overlooked. In this paper, we consider the impact of memory frequency and computing frequency on DNN inference time, and jointly optimize these two frequencies together with communication resources for energy-efficient DNN inference. Based on a realistic inference time model, we formulate an optimization problem to minimize the energy consumption of all mobile devices under the deadline constraint. For local inference, we derive a near-optimal closed-form solution via convex optimization, while an optimal closed-form solution for transmission power is obtained for edge inference with the given bandwidth. Furthermore, we propose a low-complexity heuristic algorithm to effectively solve the overall problem with polynomial time complexity. Simulation results based on measured data show that the proposed near-optimal solution for local inference can achieve optimal performance under strict deadline constraints, with a performance gap of up to 2.5% compared with the optimal solution. Meanwhile, our proposed algorithm significantly reduces the energy consumption of devices by up to 10.4% compared to other methods. 

---
# SDO: Subspace Deconflicting Operator for Multi-Adapter Composition 

**Authors**: Zhongsheng Wang, Zhedong Lin, Qian Liu, Xinyu Zhang, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13820)  

**Abstract**: Composing independently trained adapters within a shared diffusion backbone provides a modular approach to multi-character generation, but naive joint deployment often causes identity mixing, cross-character attribute leakage, and unstable scene composition. We study this interference from a parameter-space perspective and hypothesize that it arises partly from conflicts between overlapping dominant subspaces in shared layers. To address this issue, we propose \textbf{SDO}, a \textbf{S}ubspace \textbf{D}econflicting \textbf{O}perator for multi-adapter composition. SDO reconstructs layer-wise low-rank updates from the selected adapters, extracts compact subspace signatures, measures pairwise conflict through output-subspace overlap, and applies a permutation-equivariant transformation that suppresses harmful shared directions while retaining identity-specific characteristics. The resulting representations are mapped back to standard adapter updates and can be directly incorporated into existing diffusion inference pipelines. Experiments demonstrate that SDO consistently improves identity fidelity and compositional stability, with particularly clear gains as the number of jointly composed adapters increases. 

---
# From Passive Delegates to Strategic Negotiators: Reinforcing Social Reasoning in Small Language Models with SocialRL 

**Authors**: Wenyue Hua, Zachary Huang, Tyler Payne, Safoora Yousefi, Saleema Amershi, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2608.13787)  

**Abstract**: AI agents increasingly act on their users' behalf, handling tasks such as scheduling meetings, comparing offers, and haggling over prices. These principal-driven tasks routinely place the agent across from a counterpart (another user's agent, a seller, a recruiter) whose goals may conflict with its principal's. Yet the dispositions that make an assistant pleasant can make it a poor delegate: a friendly, helpful frontier model may disclose its principal's private information unprompted and concede at the first sign of resistance. We present SocialRL, a general recipe that trains social reasoning directly, and apply it to a 4B model across six domains: Deal-or-No-Deal, CaSiNo, Craigslist, Job Interview, Calendar, and Marketplace. Every domain is trained in-domain under the same recipe, and every policy is evaluated on all six. We find that (1) in-domain training reaches the frontier: on held-out scenarios the 4B matches or exceeds the GPT-5 family per domain, closing 73-122% of the baseline-to-frontier gap on the negotiation games, with 78% of buyer openings anchoring below target versus 3% untrained; (2) cross-domain transfer follows game structure: structurally paired games lift each other, a broad multi-issue donor lifts nearly all domains, and structurally isolated games transfer nothing; (3) guided by this transfer structure, two strategies, cascade RL and multi-teacher on-policy distillation (OPD), consolidate the per-domain specialists into a single unified 4B that reaches 0.627 average utility across all six environments, matching or exceeding GPT-4.1 (0.625), GPT-5.1 (0.619), and GPT-5.2 (0.613); (4) an explicit theory-of-mind scaffold helps only through training: distilling the ToM trace, rather than actions alone, lifts utility on every environment and generalizes better across them, and of the two ToM skills, only next-action prediction predicts negotiation outcomes. 

---
# FLARE MCMC: Fidelity-based Layer-Adaptive REcursive proposals for MCMC 

**Authors**: Harini Venkatesan, Christian Shelton, Ming-Feng Ho, Simeon Bird, Mengxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13774)  

**Abstract**: Markov chain Monte Carlo (MCMC) requires only the ability to evaluate the likelihood, making it a common technique for inference in complex models. However, it can have a slow mixing rate, requiring the generation of many samples to obtain good estimates and an overall high computational cost. FLARE MCMC is a multi-fidelity layered MCMC method that exploits lower-fidelity approximations of the true likelihood calculation to improve mixing and leads to overall faster performance. Such lower-fidelity likelihoods are commonly available in scientific and engineering applications where the model involves a simulation whose resolution or accuracy can be tuned. Our technique uses recursive, layered chains with simple layer tuning; it does not require the likelihood to take any form or have any particular internal mathematical structure. We demonstrate experimentally that FLARE MCMC achieves larger effective sample sizes for the same computational time across different scientific domains including hydrology and cosmology. 

---
# Simulation-Aware In-Context Policy Improvement for LLM-Aided Analog Layout Refinement 

**Authors**: Bingyang Liu, Ziming Wei, Xiaohan Gao, David Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.13767)  

**Abstract**: Analog IC layout design remains a labor-intensive iterative process dominated by simulation-driven refinement. Although end-to-end layout generators accelerate initial placement and routing, they still require experts to manually tune layout optimization parameters with repeated post-layout simulations for stringent design specifications. While Bayesian Optimization (BO) is widely adopted for parameter tuning in analog IC design, at the layout level it typically requires hundreds to thousands of evaluations, each involving costly parasitic extraction and post-layout simulation, which makes it impractical. Recently, Large Language Models (LLMs) have demonstrated potential in improving the sample efficiency of such simulation-driven tuning. However, their restricted access to geometric layout context and design-specific heuristics limits their ability to manipulate the layout optimization process. In this paper, we propose a simulation-aware LLM multi-agent framework that performs in-context policy improvement (ICPI) by iteratively updating layout optimization parameters exposed by an analog layout generator through an act-observe-reflect loop on compact structured layout representations. Experiments on real-world analog circuits show that, with only tens of post-layout simulations, our approach improves post-layout performance over the generator's built-in heuristics and BO-based tuning method. 

---
# Explanation Multiplicity: Circuit-Level Interpretability Evidence Does Not Survive Defensible Analytic Variation 

**Authors**: Ajay Pravin Mahale  

**Link**: [PDF](https://arxiv.org/pdf/2608.13754)  

**Abstract**: The EU AI Act requires providers of high-risk systems to file technical documentation describing how the system reaches its decisions. Mechanistic interpretability is the obvious source of such evidence, and circuit discovery is its most developed instrument. We ask whether that evidence survives the condition under which it would be relied upon: two competent analysts, the same system, the same tool, different defensible settings.
We pre-registered a crossed grid of seven analytic axes, every level taken from a published implementation, and mapped each discovered circuit through a deterministic claim map to a structured Annex IV statement. Across 15,840 pre-registered specifications on GPT-2 small and the indirect object identification task, of which 7,561 produced a claim, the derived statement flips across 73.2% of specification pairs (95% CI 0.725 to 0.738) and the modal claim commands 41.1% of the space. The evidence fails a filability criterion at every tolerance a conformity assessment body would plausibly accept.
Standardising the single most influential choice, the evaluation metric, leaves the flip rate at 59.4%. Removing circuit size from the claim entirely and holding it fixed leaves 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold. The circuits underlying these claims are structurally near-disjoint, median pairwise Jaccard overlap 4%, and functionally uncorrelated at Cohen's kappa 0.015, so the instability is not one mechanism described in different words.
We give the filability criterion as a standalone protocol, and we report that one of the seven documented discovery objectives does not execute at all on the library's own canonical task. The study covers one model and one task, and whether the conclusion holds at scale is untested. 

---
# Coverage Aware Active Evaluation for Failure Discovery with Paired Systems 

**Authors**: Anjali Parashar, Rachel Luo, Apoorva Sharma, Sushant Veer, Edward Schmerling, Carson Sobolewski, Mingxin Yu, Chuchu Fan, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2608.13719)  

**Abstract**: Autonomous systems can fail in rare and heterogeneous ways, making real-world failure discovery difficult under limited testing budgets. Although cheaper proxies such as simulators, lower-fidelity systems, or related policies can be sampled extensively to find failures, proxy failures often do not transfer to the real world due to sim-to-real and system-to-system gaps. The key challenge is therefore to effectively leverage proxy system information for accurate prediction of severe target system failures. We propose an adaptive failure discovery method that combines proxy evaluations with limited target system results to guide scenario selection for target system testing. Our method learns a local predictor of target risk by correcting proxy failure signals using control-variate-inspired residual modeling. To find failures that are both likely and diverse, we combine this predictor with a support-aware mutual-information objective that favors realistic, well-supported regions while expanding coverage across failure modes. Across autonomous driving, manipulation, and quadruped velocity-tracking tasks, our method discovers up to 2$\times$ as many failures as random sampling and active-learning baselines, including severe and diverse failures missed by competing methods. 

---
# Learning to Assemble Novel Structures with Unfamiliar Parts under Semantic Constraints 

**Authors**: Jonghyuk Park, Alex Lascarides, Subramanian Ramamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.13684)  

**Abstract**: This paper describes a neurosymbolic architecture for learning to assemble novel structures using evidence from embodied conversations and task demonstrations. We focus on scenarios where an agent encounters, after deployment, semantic constraints on structures--in other words, constraints as to which part types and features make valid structures--that were not available during training, and where it is initially unaware of the relevant structure and component part concepts. The agent must acquire and exploit such knowledge through user interactions while attempting assembly. We study this setting in a simulated toy truck assembly domain, learning from symbolic evidence encoded in natural language and from dense visual observations. Our experiments show that communicating semantic constraints through natural language (e.g., "dump trucks have a dumper") yields more data-efficient online adaptation than relying only on task demonstrations and/or only naming the parts through natural language. 

---
# Second Thought: Reasoning in Parallel as LLM Agents Act and Observe 

**Authors**: Zhensu Sun, Chengran Yang, Yunbo Lyu, Jieke Shi, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2608.13667)  

**Abstract**: LLM agents in the ReAct paradigm alternate between reasoning, acting, and observing, but deliberate reasoning is confined to the Thought phase: while the agent serializes an action and waits for the environment, its reasoning is frozen. We identify this recurring interval for Action and Observation as a reasoning idle window and ask whether it can host additional reasoning in parallel that serves future turns. Therefore, we propose Second Thought, a training-free inference framework that forks four auxiliary branches the instant each Thought phase concludes, decodes them concurrently with the main loop, and merges the generated thoughts back when the environment observation arrives. In this way, Second Thought relocates the added reasoning off the main thread's sequential decoding path. Across three agentic benchmarks and three reasoning LLMs, Second Thought lowers the average turn count in all nine (model,benchmark) pairs and reduces main thread decoding in six of them by up to 43% (roughly 20% on average among those settings), while leaving it essentially unchanged in a seventh; Pass@1 shows no significant change in seven of nine pairs and the two significant differences are +12.4 and +10.2 points. Against a compute-matched control that forces an equivalent budget onto the main thread's own reasoning, it attains strictly higher Pass@1 with 1.3 to 3.2 less sequential decoding in all four settings where the control applies. 

---
# Ontology-Grounded Project Memory for Coding Agents 

**Authors**: James Adam  

**Link**: [PDF](https://arxiv.org/pdf/2608.13662)  

**Abstract**: Coding agents have become the primary means of generating new code in many software projects, and the resulting velocity of changes makes keeping track of the reasons behind those changes challenging. This paper introduces MOOSEDev, a system designed to give coding agents structured, ontology-grounded project memory. The system captures architectural decisions, lessons, constraints, and rationales in a knowledge graph exposed to agents via a Model Context Protocol (MCP) interface. Records carry lifecycle status, provenance, and supersession links, queryable via MOOSE, a proprietary neurosymbolic engine that treats the symbolic layer as the primary reasoning substrate. We compared MOOSEDev against a production vector-memory tool on a neutral public corpus of 835 typed records. MOOSEDev returned the expected answer set essentially in full (0.98-1.00) on supersession, set-completeness, and negation questions, whereas the baseline's top-k retrieval surfaced between 6% and 27%. Conversely, relevance recall and token cost were largely equivalent between the two systems. We also describe a temporal commit-history bootstrap of our own codebase, a pre-registered live trial, and lessons learned. 

---
# Exploring ESC Winners with Nested Diagrams 

**Authors**: Anurag Sharma, Marcel Nöhre, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2608.13630)  

**Abstract**: We present ConceptFlow, a scikit-learn-compatible Python library for Formal Concept Analysis that constructs and renders nested line diagrams from many-valued formal contexts. Given a many-valued context and a partition of its attributes into conceptual scales, ConceptFlow performs conceptual scaling, computes the factor lattices, identifies filled nodes of the corresponding subdirect product, and produces an interactive visualization.
We apply ConceptFlow to the winners of the Eurovision Song Contest from 1975 to 2025, exploring relationships between voting patterns and musical characteristics. Voting support is captured by an outer scale spanning regional, cultural, historical, and political dimensions, while an inner scale captures musical characteristics via tempo and key. The resulting nested line diagram reveals implications across both scales, exposing dependencies between how winning entries were voted for and the musical properties they share. 

---
# A Calibrated Test of Internal Action Maps: State Signals Without Global Affine Closure 

**Authors**: Dekun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13626)  

**Abstract**: A hidden state signal can be decodable or causally usable without supporting a reusable action map. We test whether action maps fitted without a source reach its natural post-action activation and compose. We organize the tests as an evidence lattice and validate the geometric branch on a known affine S_5 carrier: all held-source folds pass one-step, composition, inverse, decoding, and commutativity gates. Structured curvature and held-domain conjugacy raise error monotonically, but only 23/30 strongest cells flip a closure gate, bounding rather than universalizing calibration. In post-trained Qwen/Qwen3-4B, frozen final-token h28 affine maps have mean held-entity error .519, versus .398 for within-test-domain cross-fit. Seven randomized entity splits and map geometry do not support a purely entity-specific account. Earlier h4/h16 layers fit one-step transitions better, but h4 conflict-state decoding is weak and lexical controls remain unresolved. Three matched intervention datasets regenerated from one frozen checkpoint show causal effects only at h28/h36. Outcome-aware refitting improves h28 one-step error to .474 (.469 with weighting), yet no refit passes composition. Learned finite worlds likewise preserve relative algebraic signals or shared charts without held-source affine closure. Within the tested carriers, state availability, causal use, local geometry, and reusable closure are separable. The result is limited to one pretrained model, sampled final-token layers, two finite worlds, and the tested affine or diagnostic function classes. 

---
# Reward Machines for Signal Temporal Logic 

**Authors**: Alper Kamil Bozkurt, Shangtong Zhang, Yuichi Motai  

**Link**: [PDF](https://arxiv.org/pdf/2608.13625)  

**Abstract**: Signal temporal logic (STL) provides a formal language for specifying real-time properties of real-valued observations, along with a quantitative robustness score for monitoring satisfaction. Control synthesis from STL specifications is of interest since manual controller design becomes infeasible as real-world systems grow in complexity. Moreover, many modern autonomous and AI-enabled systems lack accurate and complete system models, which makes optimization-based synthesis approaches unsuitable and motivates learning-based control. Prior work uses STL robustness scores as rewards in reinforcement learning (RL) to obtain control policies satisfying given specifications; however, robustness depends on execution history, leading to intractable state space expansion for general long-horizon specifications with arbitrarily nested temporal operators. This work introduces a novel automata-based approach that provides an efficient memory mechanism and associated Markovian rewards suitable for RL frameworks. Our approach constructs a timed alternating automaton from the given STL specifications, augments the state space with automaton locations and clock valuations, and derives rewards from the automaton acceptance condition. We empirically demonstrate that our approach learns policies that achieve higher robustness scores and satisfaction rates than those learned by existing approaches using robustness-based rewards. 

---
# ARC: Fair Relative Advantage Comparison in Open-Ended Real-World Interaction 

**Authors**: Yongqi Tong, Tan Li Hui Faith, Choy Zhen Wen Marcus, Zhou Jin, Kewei Fu, Jiang-Ming Yang, Jianshe Li, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13622)  

**Abstract**: Open-ended real-world interaction admits multiple valid behaviors: an agent may answer directly, ask for clarification, provide progress updates, or confirm before acting. This flexibility breaks a core assumption behind group-based RL: rollouts compared within a group are no longer guaranteed to be behaviorally comparable. As a result, reward-model preferences over interaction style can distort relative advantages and steer optimization toward reward-preferred behaviors rather than context-appropriate ones. We formalize this as a \textit{reward fairness problem} and propose \textbf{ARC} (Advantage Regularization via Conditioning), a training recipe that restores fairer relative comparison through strategy-conditioned rollout grouping, together with hybrid rewards and entropy regularization. We study ARC in our proposed \inter, a novel paradigm for responsive, steerable, and execution-aware user-agent interaction that decouples user-visible communication from latent reasoning and tool use. \inter\ also provides the annotation and distillation pipeline for constructing \inter-86K, our strategy-annotated training corpus for supervised and RL training. Empirically, ARC substantially strengthens the core $\tau/\tau^2$ tool-use benchmarks, while \inter\ reduces time-to-first-token from 4.91s to 1.27s relative to a think-style baseline. Together, these results suggest that a central bottleneck in open-ended interactive learning is not only how agents are rewarded, but whether their behaviors are compared fairly in the first place. The ARC implementation and \inter-86K training data will be released. 

---
# Your Probabilistic JEPA Is Secretly a Hidden Markov Model: A State-Space Interpretation of Joint-Embedding Predictive Learning 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13621)  

**Abstract**: A hidden Markov model (HMM) combines three roles: inference of a hidden-state belief from observations, propagation through a Markov transition, and emission back to observation space. We show that full, time-indexed Predictive Information Bottleneck VJEPA (PIB-VJEPA) exposes the same computational structure: a stochastic context encoder plays the role of an amortized filtering distribution, a probabilistic predictor defines latent-state dynamics, and a decoder, inverse target encoder, or induced implicit conditional supplies the emission direction. We distinguish 4 progressively stronger levels of correspondence and give sufficient conditions for exact sequence-level HMM equivalence. To make the connection concrete, we introduce Markov-Chain JEPA (MCJEPA), which replaces the latent predictor by a learned transition matrix; in the finite time-homogeneous case, matrix powers guarantee exact multi-horizon Chapman--Kolmogorov consistency. Conditioned discrete-state transitions, continuous-state Markov kernels, and continuous-time dynamics extend this construction, while deterministic temporal JEPA appears as a degenerate Dirac-kernel special case. We further interpret predictive information-bottleneck learning as seeking a compact predictive state: compression promotes minimality, while residual predictability tests sufficiency. Controlled experiments support transition composition, the filtering interpretation, predictive Markovization in a known synthetic process, and the distinction between JEPA latent prediction and HMM-style sequence learning. Together, these results give temporal JEPA a principled state-space interpretation. 

---
# Algorithm Design and Physician Liability 

**Authors**: Shujie Luan, Shubhranshu Singh, Tinglong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2608.13618)  

**Abstract**: A single clinical algorithm can deliver unequal accuracy across patient groups, and concern about such disparity has grown as artificial intelligence (AI) spreads through clinical decision-making. In response, a liability rule introduced in the United States holds healthcare providers responsible when their reliance on disparate algorithms contributes to erroneous clinical decisions. We examine how such liability considerations reshape (i) an AI firm's algorithm design decisions that drive group-specific accuracy and (ii) a physician's decisions to use AI in healthcare delivery. The AI firm designs an algorithm for two patient groups, and improving accuracy for the disadvantaged group is more costly. The physician (who remains the accountable decision-maker) then decides whether to consult AI, weighing the reduction in clinical uncertainty against expected liability exposure when AI errors disproportionately affect the disadvantaged group. We find the liability rule can induce disparate use of AI: the physician may reduce AI use overall and, over an intermediate range of liability, rely on AI less for disadvantaged patients. The effect is non-monotone. As liability increases, the physician's use of AI for disadvantaged patients first declines, then rises as the firm reallocates investment toward reducing disparity or switches to an equal-accuracy design. Mandating equal algorithmic accuracy across patient groups can then inadvertently harm both groups, because a uniform accuracy requirement distorts the firm's investment incentives and the physician's equilibrium AI-use decisions. 

---
# How Compliant is Sepsis Treatment? An Expert-Guided Neuro-symbolic Pipeline for Generating Clinical Compliance Insights 

**Authors**: Himanshu Tripathi, Kaushik Roy, Subash Neupane, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2608.13617)  

**Abstract**: Verifying whether clinical care follows evidence-based protocols is a natural neuro-symbolic problem, yet the safety-critical setting defeats either paradigm alone. We present an expert-guided pipeline that constrains a large language model strictly to semantic normalization, mapping messy drug and microbiology strings onto a fixed clinical vocabulary, while a Sugeno fuzzy inference system reasons over the normalized events. The fuzzy layer encodes eight Surviving Sepsis Campaign bundle rules and replaces binary judgments with graded scores in [0,1]. Applied to 2,438 MIMIC-IV v3.1 sepsis episodes, it surfaces antibiotic timing as the most critical breakdown (mean 0.24, 13% within one hour), Hour-1 underperformance (mean 36.7%), a 51% elevated-lactate drop-off, and descriptive differences in ICU stay across compliance groups (3.8 versus 5.1 days). 

---
# SemPlan: Benchmarking Structured Semantic Planning for LLM-Based Queries over Enterprise Data 

**Authors**: Bruno Santos Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2608.13612)  

**Abstract**: Natural-language interfaces to enterprise data must translate underspecified requests into governed, executable behavior while controlling invalid queries, policy failures, cost, and nondeterminism. SemPlan Benchmark evaluates this architectural design space with a deterministic synthetic bilingual benchmark containing 1,800 cases in English and Brazilian Portuguese; 1,200 cases form the frozen scientific evaluation subset. Four architectures are compared under the same model configuration: direct SQL generation (A1), a bounded tool-agent baseline (A2), structured semantic-request generation followed by deterministic planning and execution (A3), and a clarification/stateful semantic-plan variant (A4). Across 4,800 primary records, answer correctness was low in absolute terms: 22.25% for A1, 22.58% for A2, 25.67% for A3, and 24.25% for A4. A3 had the highest observed correctness and significantly exceeded A1, A2, and A4 in the pre-specified paired correctness analysis, while A1 retained the highest policy-correct rate and the lowest unsafe-or-invalid rate. A4 had the lowest mean API cost and lowest false-refusal rate. On a preselected 150-case stability subset, answer-correct repeatability ranged from 92.00% to 98.67%. The results support a trade-off interpretation rather than a universal ranking: additional structural constraints changed failure modes and efficiency, but did not monotonically improve correctness or solve ambiguity and multi-turn state consistency. 

---
# Evaluating Agentic Learning Harness Capabilities Without Labels via the Scaling Hypothesis 

**Authors**: Aryan Luthra, Kshitij Jain, Siddharth Arya, Bobby Filar, Anna Bertiger  

**Link**: [PDF](https://arxiv.org/pdf/2608.13608)  

**Abstract**: Agentic "Continual Learning Harnesses", systems that pair an LLM with retrieval or memory to improve from feedback without retraining, have shown growing value in cybersecurity. But their value is conventionally measured by gains against labeled benchmarks, an approach that often fails in operational security settings. Benchmark labels are scarce, stale, and unrepresentative, so a practitioner often cannot tell whether a given harness helps at all or which of two is better for their task. Traditional LLM-as-a-judge offers little signal because it is no stronger than the agent it evaluates, and distillation is unreliable on scarce, sporadic, and biased labels.
We propose a framework for evaluating learning harnesses end-to-end without a labeled benchmark, grounded in the scaling hypothesis. A stronger teacher model provides sparsely sampled corrections to a smaller student with a continual learning harness. We score a harness by how much its student converges toward the teacher over time. Across security tasks, model families, and harness designs, we show that improvement relative to the teacher correlates with improvement relative to a held-out gold standard, validating teacher-relative lift as a proxy for true harness uplift when labels are absent. We further show that LLM-as-a-judge between similarly powered models yields no usable signal. These results suggest that a teacher-sized model can be improved through the same harness when humans provide the same kind of sparse, high-precision corrections. 

---
# No Universal Signal Predicts Sample-Level LLM Regression under Version Updates 

**Authors**: Jia Sheng, Yiwei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13607)  

**Abstract**: Frontier LLMs are updated frequently and typically outperform their predecessors in aggregate. But aggregate gains say little about individual samples: an update can still cause sample-level regression, where a response correct under the old model becomes incorrect under the new one. This paper studies how to predict such regressions from signals available at inference time. We compare single-model signals (confidence, logit margin, attention entropy) against cross-version signals (output KL divergence, likelihood drift, token-level KL, representation drift) under a unified added-value test that isolates each signal's gain over a confidence baseline. Across six benchmarks in three task families (multiple-choice question answering, or MCQ; math reasoning; code generation) and six model update pairs, we find that (1) signal effectiveness is task-dependent: confidence is strongest on MCQ and simpler math, while likelihood/KL signals give the most frequent gains on harder math and code; (2) no signal is universally best across model updates either; and (3) some cross-version signals stay informative even when confidence fails, including without labels, which supports a proof-of-concept selective fallback that routes high-risk samples back to the old model. Practitioners can use these task-level patterns to choose which regression signal to trust for a given update. Code is available at this https URL. 

---
# MobileMem: Learning from a Year of Mobile Experiences 

**Authors**: Xinle Deng, Yida Xue, Xiangyuan Ru, Haoming Xu, Shuofei Qiao, Mengru Wang, Yijun Chen, Buqiang Xu, Chen Jiang, Yuchen Eleanor Jiang, Lizhong Wang, Jianfeng Wang, Li Zeng, Haofen Wang, Guilin Qi, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13606)  

**Abstract**: The next generation of AI agents is increasingly moving beyond systems that answer isolated questions toward persistent personal assistants that can understand, remember, and continuously learn from users' experiences. Such assistants require long-term memory to accumulate and leverage user-specific experiences over time, yet existing benchmarks remain inadequate for realistic mobile settings, where experiences are heterogeneous, multimodal, evolving, and deeply personal. We introduce MobileMem, a benchmark and framework for studying on-device long-term memory, grounded in a year-scale collection of mobile experiences. MobileMem employs a knowledge-grounded synthesis pipeline to construct coherent and temporally consistent long-horizon trajectories from user-app sessions. It provides complementary text and multimodal settings covering multi-hop and temporal reasoning, knowledge updating, and implicit preference inference. Specifically, MobileMem enables agents to remember the past, understand the present, and adapt to the future. By modeling experiences rather than isolated facts, MobileMem moves memory beyond information retrieval toward experiential intelligence for continuous personal learning. 

---
# Active Perception for Embodied Disambiguation 

**Authors**: Yiwei Liu, Luwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13605)  

**Abstract**: Natural language provides robots with a flexible task interface, but target ambiguity in embodied environments arises not only from user intent; it can also result from missing taskrelevant physical evidence in the current observation. Existing interactive disambiguation methods primarily obtain additional information by asking the user, whereas occlusion, restricted viewpoints, unreadable text, and unobserved targets require the robot to actively change its observation. We propose an active-perception framework for embodied target disambiguation that uses active observation as the backbone for information acquisition and uses a vision-language model to decide, on the basis of accumulated visual evidence and interaction information, whether to continue observing, request clarification, or complete target selection. Active observation can both directly recover missing discriminative evidence and reveal object names, labels, and semantic attributes, thereby improving user clarification when it remains necessary. Real-robot experiments show that the framework combines physical information acquisition and userintent clarification within a unified embodied disambiguation process. 

---
# Cross-Disciplinary Taxonomy and Modeling of Misunderstanding Generation, Amplification, and Detection, from Pragmatics to AI Agents 

**Authors**: Babak Abbaschian  

**Link**: [PDF](https://arxiv.org/pdf/2608.13604)  

**Abstract**: Detection of misunderstanding is an urgent problem to solve because communication has moved away from real-time, in-person interaction and is increasingly handled by AI-mediated channels. This shift cuts communicators off from the resources repair depends on faster than new means of detection are being built. In this paper we analyse misunderstanding as a layered process in which a divergence is generated, may then be amplified, and is either detected and repaired or left to persist unnoticed. Consolidating accounts from nine fields of research that do not ordinarily cite one another, we identify eleven exact failure modes and show that each operates at a specific point in a communicative process rather than anywhere within it. Those points give eight analytical layers, derived from the literature rather than adopted from an existing model. Eight of the mechanisms primarily generate a divergence, two primarily amplify one already present, and one governs whether a divergence is detected and repaired. We model the eight layers formally, extending information and communication theory from the transmission of signals to the reconstruction of meaning, and we supply a source-by-source evidence matrix that makes every rating auditable, a coding manual, and nine analysed dialogue cases. No prior classification of misunderstanding both locates mechanisms at points in the process and types them by function. 

---
# Measuring Cross-Task Behavioral Consistency in Language Model Agents 

**Authors**: Amritesh Banerjee, Pranil Raichura  

**Link**: [PDF](https://arxiv.org/pdf/2608.13598)  

**Abstract**: Agent evaluation relies almost entirely on outcome metrics such as success rate, which capture whether an agent succeeds but not how consistently it behaves. We argue that behavioral consistency across tasks is a distinct and measurable property, and we introduce the Behavioral Consistency Metric (BCM) to quantify it. BCM trains a model to predict task success from behavioral features of agent execution traces, derives a per-trajectory feature-attribution vector, and measures the mean pairwise similarity of these vectors within an agent system. Across roughly 9,000 trajectories from six language model agents on software engineering tasks, our central finding is that cross-task and within-task consistency are distinct axes that can diverge: some systems are locally reproducible, behaving similarly on repeated attempts at one task, yet globally fragmented, with no stable strategy across different tasks, while others are consistent at both scales. Prior work measures only same-task reproducibility and so cannot observe this separation. We further find that consistency is not reducible to success rate, since systems with comparable success can differ sharply in consistency, and that the frontier-versus-open-source consistency gap persists under a within-task control that holds task difficulty constant. We position BCM as a process-level reliability signal that complements outcome metrics, and we are explicit about the conditions under which it is meaningful. 

---
# Stable Miscalibration in Large Language Models: A Practical View of High-Confidence Errors 

**Authors**: Akira Okutomi  

**Link**: [PDF](https://arxiv.org/pdf/2608.13591)  

**Abstract**: High-confidence errors in large language models are often treated as evidence of fragile internal inference. We study a different possibility: stable miscalibration, where a confident wrong answer remains locally stable under small perturbations. We combine two diagnostics: a label-aware output-level audit score that ranks domains by confidence variation and overconfident mistakes under a forced-answer baseline, and an internal sensitivity probe that measures hidden-state movement. On a multi-domain binary factual audit set, this audit score tracks where abstention-aware self-critique reduces decision loss, although direct labeled baselines rank the same gain more strongly. Internally, self-critical prompting consistently reduces hidden-state sensitivity across layers in three open-weight models. This supports prompt-induced local stabilization rather than a purely output-level abstention pattern, but it does not imply calibration: audit-defined overconfident errors are not clearly more locally sensitive than confidently correct answers, so some high-confidence errors may be stable and miscalibrated rather than simply fragile. 

---
# AI Evaluation Should Work With Humans 

**Authors**: Jan Kulveit, Gavin Leech, Tomáš Gavenčiak, Raymond Douglas  

**Link**: [PDF](https://arxiv.org/pdf/2608.13577)  

**Abstract**: This position paper argues that the dominant paradigm of AI evaluation (which focuses on superhuman autonomous performance and so implicitly targets the goal of replacing humans) is guiding AI development in the wrong direction. Instead, the AI community should pivot to evaluating the performance of human--AI teams. We argue that this collaborative shift will foster AI systems that act as true complements to human capabilities and therefore lead to far better societal outcomes than will the current process. 

---
# Agentao: A Governed Local-First Runtime for Tool-Using LLM Agents 

**Authors**: Bo Jin, Qiang Jiao, Xin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.13574)  

**Abstract**: LLM agents increasingly operate as execution systems that invoke tools, modify local state, use persistent memory, and interact with external protocols. These capabilities make agents useful, but they also introduce risks related to over-privileged actions, weak auditability, prompt injection, tool poisoning, and uncontrolled side effects. This paper presents Agentao, a governed local-first runtime for tool-using LLM agents. Agentao separates model-generated action proposals from host-authorized execution through a layered architecture consisting of host-facing surfaces, a host contract, a runtime core, a permission-mediated tool system, and supporting subsystems for memory, replay, plugins, skills, sub-agents, and protocol integration. We describe the motivation, threat model, design goals, governance model, execution pipeline, and structured event interface of the system. Agentao does not provide formal safety guarantees; rather, it demonstrates how permissions, state, protocol boundaries, and execution traces can be made explicit runtime abstractions for building agents that are more governable, inspectable, and suitable for host-controlled local environments. The code is publicly available at this https URL. 

---
# A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing 

**Authors**: William Nixon, Jon Durbin, Florian Standhartinger, Haryadi S. Gunawi, Juncheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13573)  

**Abstract**: Large Language Model (LLM) serving has become a critical cloud workload, and realistic traces are essential for motivating and benchmarking serving systems. However, existing LLM serving workload studies remain limited in scale and scope. They often observe short time periods and provide limited visibility into how users interact with models in production. As a result, they do not fully capture how LLM serving workloads evolve over time or how user-model interactions shape production traffic.
In this work, we further the understanding of real-world LLM serving workloads through both a global characterization and a longitudinal study of a one-year production trace from Chutes. Unlike prior studies, our trace captures full production behavior across many models and users, including both popular and long-tail models. We analyze the workload from aggregate, temporal, model-level, and user-level perspectives, revealing workload evolution and user-model structure that are typically hidden behind aggregate views. To support future research, we will release the full one-year trace with the paper, enabling downstream studies of production behavior without relying on sampled or synthetically generated workloads. 

---
# Modular Cognitive Architecture Emerges in Large Language Models 

**Authors**: Pengrui Han, Jacob Andreas, Evelina Fedorenko, Andrea Gregor de Varda  

**Link**: [PDF](https://arxiv.org/pdf/2608.13567)  

**Abstract**: The human brain exhibits a striking degree of functional specialization, with distinct networks supporting language, formal reasoning, reasoning about other minds, and reasoning about the physical world. Is this modular organization a fundamental principle of how intelligent systems must be built, or an evolutionary accident specific to biological brains? Here, we test whether a similar organization emerges in Large Language Models--another class of intelligent systems created through a very different optimization process. Using circuit analyses across N=46 tasks spanning four cognitive domains (language, formal reasoning, social reasoning, physical reasoning), we find that LLMs develop a modular architecture that mirrors the human brain: tasks drawing on the same network in humans recruit overlapping neurons in LLMs, whereas tasks drawing on different networks recruit distinct neurons. The convergent emergence of modularity in brains and neural networks suggests that it may be a fundamental property of intelligent systems. 

---
# Depth-Aware Sensitivity Analysis of Mixture-of-Experts Models via Magnitude-Based Expert Masking 

**Authors**: Pradeep Kumar Sharma, Shantanu Godbole, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2608.13565)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale large language models (LLMs) while preserving computational efficiency through sparse activation. Despite their widespread adoption, the relative importance of individual MoE layers remains insufficiently characterized, particularly for model compression. This paper presents a systematic layer-wise sensitivity analysis of the Qwen3.6-35B-A3B model (40 MoE layers, 256 experts per layer, top-8 routing) using magnitude-based expert masking on the XLCoST cross-lingual code translation benchmark. We conduct a multi-phase study spanning 100, 300, and 500 prompt evaluation scales across three H100 GPU servers. Our central finding is that layer sensitivity is strongly depth-dependent: early layers (0-9) and middle layers (10-29) are highly fragile to expert masking, while late layers (30-39), and especially very-late layers (35-39), tolerate aggressive masking of low-magnitude experts. Flat all-layer masking at 30% retains only 150/300 Good+Similar outputs at 300-prompt scale, whereas late-focused policies retain 249-255/300 while masking 640-1,145 experts. On a later 500-prompt held-out validation slice, the narrow very-late policy (layers 35-39 @ 50%) achieves the strongest quality/masked-expert tradeoff among tested candidates, retaining 419/500 Good+Similar outputs while masking only 640 of 10,240 total experts. We additionally characterize top-k routing width reduction from 8 to 6 active experts per token, which shows a large observed wall-clock reduction on a 100-prompt probe with no Good+Similar loss, though it does not yet compose cleanly with aggressive expert masking. These findings provide an empirical foundation for depth-aware MoE expert masking and establish a practical path toward physical weight surgery, activation-based expert scoring, and training-based recovery. 

---
# Inducing Reward-Free Judging Rubrics that Reduce Over-Crediting in Agent Evaluation 

**Authors**: Darragh Quinn, David Dylan, Roisin Healy, Fionn Carroll, Maeve Donnelly, Cormac Sheehan  

**Link**: [PDF](https://arxiv.org/pdf/2608.13564)  

**Abstract**: Evaluating language-model agents at scale increasingly relies on a second language model as an automatic judge, because the gold signal, an executable environment reward, is expensive, slow, or unavailable at deployment time. Such a judge is a reward-free proxy whose value depends on whether it can be trusted, yet existing judges either hand-write the scoring rubric, as in G-Eval, or fine-tune the judge's weights, and both tend to credit fluent but unsuccessful trajectories as successes. We instead induce the text of an agent-judging rubric from a small set of ground-truth-labeled trajectories, grounding it in true outcomes. We present RubricForge, which evolves a judge rubric by reflective evolution against labeled trajectories to maximize agreement with the environment reward, freezes it, and applies it to held-out trajectories in one model call with no environment access. The optimized artifact is human-readable text, so every verdict is attributable to named criteria. Using one frozen 7B model as both agent and judge, on tau-bench (173 labeled trajectories drawn from 220 rollouts) and WebShop (160), the principal gain is faithfulness rather than raw agreement. The edge over a generic G-Eval judge is not statistically significant (McNemar p = 0.248), and absolute-score calibration marginally favors the generic judge (|err| difference -0.048, p = 2x10^-4). Yet RubricForge over-credits failed trajectories roughly half as often (0.115 vs. 0.173 false-pass rate on tau-bench, with three over-credit catches and zero reversals) and ranks graded WebShop outcomes more faithfully (Spearman 0.410 vs. 0.370). For a reward-free evaluator the false-pass rate, not aggregate agreement, is the deployment-relevant quantity, since a false pass ships a broken agent whereas a false fail merely costs a retry. 

---
# Decoding the Past: An Uncertainty-Aware Deep Learning Framework for Sex Attribution in Prehistoric Hand Stencils 

**Authors**: Karel Becerra, Boris Mederos, Dean Snow, Ramón A. Mollineda  

**Link**: [PDF](https://arxiv.org/pdf/2608.14539)  

**Abstract**: Determining the biological sex of the individuals who created Upper Paleolithic hand stencils remains a challenging problem due to the absence of ground truth, population differences between contemporary and prehistoric groups, and the uncertainty introduced by image degradation. Traditional morphometric methods suffer from high structural overlap across sexes, poor cross-population generalizability, and subjective feature engineering. This study presents an uncertainty-aware deep learning framework for sex attribution in prehistoric hand stencils that explicitly models, propagates, and aggregates uncertainty throughout the analytical pipeline. The methodology combines dual image processing, dual contour extraction, structured silhouette augmentation, model architectural diversity, and ensemble-based decision aggregation. The pipeline generates twelve plausible silhouette realizations per stencil to capture boundary uncertainties, which are processed by two ensembles of ten deep neural networks each (EfficientNet-B3 and MobileViT-S) trained on 14,036 contemporary hand samples. Furthermore, a triangulated validation scheme integrates ensemble predictions with unsupervised 2D latent-space manifold mapping (UMAP + k-NN) and explainable AI spatial attributions (LayerCAM) to ensure anatomical consistency. On contemporary data, ensemble models achieve strong classification performance, with accuracies exceeding 88% in older age groups. When applied to prehistoric stencils, the framework produces both sex predictions and confidence measures of internal agreement, enabling the distinction between morphologically stable and ambiguous cases. Convergence across ensemble predictions, latent-space structure, and interpretability analyses shows that uncertainty can become a measurable component of archaeological inference, enabling robust and reproducible decoding of ancient rock art. 

---
# Marionette: Predicting World States, Rendering Geometry, Painting Appearance 

**Authors**: Zian Meng, Zhen Li, Chuanhao Li, Qiang Li, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14530)  

**Abstract**: Interactive game world models typically autoregress visual observations directly in pixel or latent space, forcing structured properties such as pose, geometry, and occlusion to be implicitly maintained by the same generative sequence. Over long horizons, errors in these latent world properties accumulate, making consistency and controllability fragile. We explicitly model the evolving world state, delegate exact geometric computation to a fixed, zero-parameter renderer, and leave the neural model to synthesize appearance. We instantiate this idea as Marionette, a world model for interactive games with articulated characters. First, a two-stage autoregressive dynamics model predicts an explicit and interpretable 276-dimensional 3D world state comprising multi-entity articulated skeletons, metric root trajectories, and rotations. Second, a zero-parameter graphics bridge converts the predicted state into pose-control videos, computing world-space geometry and occlusion in closed form. Third, a control-conditioned video-diffusion observation model synthesizes photorealistic RGB observations from the resulting structured controls. Our experiments establish two properties of Marionette. First, the predicted world state is directly controllable. Forcing a mismatched action stream changes root-aligned joint error by 31% across 48 held-out segments. Second, long-horizon behaviour is determined in the state, and can be repaired there. Left free, the two generated characters drift to 21.2 m apart (recorded sessions stay near 5 m) and a third of frames show ground penetration. Two rules imposed on the explicit state, a terrain collider and a separation cap, cut penetration by 66% and keep the pair engaged, with no change to the observation model. Routing appearance through the predicted state costs no fidelity we can detect, at an FVD of 831 against 799 for recorded pose. 

---
# Learning-to-Transition for Large-scale and High-Order MIMO Detection 

**Authors**: Yubo Zhang, Yiyao Liu, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14511)  

**Abstract**: High-order multiple-input multiple-output (MIMO) detection requires efficient search over a large discrete symbol space while producing reliable soft information for channel decoding. This paper develops a learning-to-transition (L2T) framework that formulates MIMO detection as a stochastic sequence of complete-vector transitions. At each transition, a channel-coupled Transformer updates both the instance embedding and the sampling policy, while a blockwise autoregressive factorization captures inter-stream dependence with moderate sequential complexity. For hard-output detection, a transition network is applied recursively and trained through a residual-to-BER curriculum, which first learns the MIMO search geometry from the exact residual metric and then aligns the policy with transmitted-bit accuracy. For soft-output reception, the well-trained hard policy is cloned at the parameter level into every layer of an untied soft-input soft-output iterative detection and decoding (IDD) receiver. This tied-to-untied transfer preserves the learned zero-prior search dynamics while enabling layer- and round-specific specialization under decoder feedback. Within each IDD round, decoder priors tilt candidate generation according to Bayes' rule, and likelihood-weighted terminal hypotheses produce posterior and extrinsic log-likelihood ratios for LDPC decoding. A multi-stage training strategy further stabilizes the hard-to-soft transfer by progressively exposing the receiver to synthetic and in-loop decoder-generated priors. 

---
# RecipeNet: A Hierarchical Transformer for Recipe Data 

**Authors**: Pin-Yen Huang, Sachin Chhabra, Prasanth Sai Gouripeddi, Abhinav Kumar, Baoxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14505)  

**Abstract**: Recipe data arises in domains such as materials synthesis, pharmaceutical formulation, and industrial manufacturing, where procedures are represented as ordered sequences of steps containing heterogeneous structured fields. Existing tabular learning methods typically flatten this structure into fixed-schema representations, limiting their ability to capture hierarchical field interactions and procedural dependencies. We propose RecipeNet, a hierarchical Transformer architecture that encodes field-level interactions within each step and sequential dependencies across steps through stacked Transformer encoders. Experiments on multiple recipe datasets and tasks demonstrate that RecipeNet consistently outperforms existing tabular models, highlighting the value of hierarchical and sequential modeling for recipe representation learning. 

---
# Universal Thermodynamic Interatomic Potentials for Crystalline Materials 

**Authors**: Juno Nam, Bowen Deng, Xiaochen Du, Luis Barroso-Luque, Benjamin Kurt Miller, Rafael Gómez-Bombarelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.14502)  

**Abstract**: Free energies govern solid-state phase stability, yet computational materials discovery still relies largely on ground-state energies because free energy calculations require ensemble averages. We introduce the thermodynamic interatomic potential (TIP), which extends an interatomic potential from its static energy to a thermodynamically consistent Gibbs free energy model, with thermodynamic responses following from temperature and pressure by automatic differentiation. We implement TIP[UMA] using the universal potential UMA, train it on free energies from quasi-harmonic to molecular dynamics fidelity, and calibrate it to higher-resolution calculations or experiment. From a single evaluation, it returns the equation of state of a crystal and locates phase transitions among competing branches, including dynamically stabilized phases. Fine-tuning extends the model to alloy solubility limits and miscibility gaps. TIP makes the free energy as accessible as the potential energy, opening finite-temperature phase stability to high-throughput discovery. 

---
# Generating Benchmark Health Data Using a Tabular Diffusion Transformer 

**Authors**: Hao Yan, Lisa Pilgram, Dan Liu, Linglong Kong, Fida Dankar, Khaled El Emam  

**Link**: [PDF](https://arxiv.org/pdf/2608.14496)  

**Abstract**: Cross-Tabular Data Generation (CTDG) seeks to learn a generative model from multiple heterogeneous tables and produce new synthetic tabular datasets. However, existing synthetic tabular data generation methods are largely restricted to single-input-table scenarios and struggle to effectively handle multiple heterogeneous tables with diverse feature sets. To address this limitation, we propose a two-stage framework for cross-tabular data generation. In the first stage, each heterogeneous raw table is transformed into a standardized statistical table with the same set of columns across all tables. Each statistical table captures the marginal distributions of the original columns and the pairwise correlations among them. In the second stage, a diffusion transformer model is trained to capture structural patterns across these homogeneous statistical tables and to generate synthetic statistical tables. Synthetic raw tables are subsequently reconstructed from the generated statistical tables via multivariate Gaussian sampling followed by an inverse probability integral transform. This two-stage CTDG framework enables the learning of a unified generative model from multiple heterogeneous tables and supports the generation of an unlimited number of realistic synthetic heterogeneous tables. Experimental results demonstrate high fidelity in the learned statistical representations and a favorable fidelity-diversity trade-off in the generated synthetic data, validating the effectiveness of the proposed approach. 

---
# Optimal Scheduling of Road Maintenance Jobs Considering Impact on Traffic Flows 

**Authors**: Charitha Nandepu, Lohitha Kalepu, Gabriele Ciavarella, SangWoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.14491)  

**Abstract**: Network-level maintenance planning requires repeated evaluations of equilibrium traffic flows under road capacity reductions. While equilibrium traffic assignment models are well established, their repeated solution quickly becomes computationally prohibitive and challenging to embed within maintenance scheduling problems. This paper investigates data-driven surrogate models that approximate equilibrium arc flows directly from origin-destination demand, using optimization-based equilibrium solutions as ground truth. A real-world case study based on traffic data from the Newark, New Jersey area demonstrates the effectiveness of the proposed approach as a scalable building block for future maintenance scheduling frameworks. 

---
# Ensuring Safe Physical AI in Urban Mobility via Hazard-Informed Synthesized Envelopes 

**Authors**: Alexei Odinokov, Rostislav Yavorskiy  

**Link**: [PDF](https://arxiv.org/pdf/2608.14481)  

**Abstract**: As heterogeneous robotic systems deploy across diverse urban zones, maintaining safety amid complex human-robot interactions remains a critical challenge. We present a unified framework that bridges systematic hazard analysis and runtime enforcement using hazard-informed safety envelopes. Rather than treating safety as a static constraint isolated within individual software modules, we introduce a cross-layer safety transformation process spanning symbolic, spatial, and dynamic world models. We show how this representation naturally interfaces with physical AI runtime harnesses to guarantee safe urban mobility. 

---
# LP-NAS: Linear Programming-based Neural Architecture Search 

**Authors**: Abhishek Shukla, Ankur Sinha, Faiz Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2608.14472)  

**Abstract**: Neural Architecture Search (NAS) aims to automate neural network architecture design, reducing reliance on human expertise. Among the various NAS methods, differentiable NAS has gained prominence due to its efficiency and accuracy compared to conventional NAS approaches. Since differentiable NAS relaxes the architecture search space into a continuous domain, it is possible to apply principles from continuous optimization to NAS. In this paper, we propose Linear Programming-based NAS (LP-NAS), a mathematical programming-based framework for differentiable NAS that is applicable to a wide range of continuous search spaces. LP-NAS formulates a linear program (LP) using the validation-loss gradient and the training-loss Hessian to compute an architecture update direction that improves generalization while preserving the optimality of the model parameters. By following this LP-derived descent direction, LP-NAS efficiently navigates the architecture search space, leading to faster and more effective architecture optimization. We introduce two computationally efficient variants of LP-NAS, namely S-LP-NAS and R-LP-NAS. Applying LP-NAS to the Differentiable Architecture Search (DARTS) search space results in two algorithmic variants, S-LP-DARTS and R-LP-DARTS. Both variants achieve faster convergence and significantly higher validation performance during the early search iterations than the standard DARTS algorithm. Extensive experiments on CIFAR-10 and CIFAR-100 show that LP-DARTS outperforms standard DARTS in both the architecture search and evaluation phases. Additionally, we compare our approach with several DARTS variants (P-DARTS, PC-DARTS, and STO-DARTS) on the CIFAR-10 dataset and demonstrate its effectiveness. Furthermore, we validate the transferability of the discovered architectures through experiments on the ImageNet dataset. 

---
# Designing Compact Neural Architectures via Neuron Gating and Mixed Activation 

**Authors**: Abhishek Shukla, Ankur Sinha, Faiz Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2608.14443)  

**Abstract**: Neural Architecture Search (NAS) is naturally formulated as a bilevel optimization problem, where the upper-level optimizes the architecture using validation performance and the lower-level trains network parameters using training loss. However, NAS is computationally expensive due to discrete architectural decisions, exponentially growing search spaces, and the high cost of training candidate architectures. This work develops a general bilevel optimization framework for NAS across diverse architectures, including MLPs, CNNs, RNNs, and Transformers, to identify compact architectures with strong predictive performance. We propose three scalable formulations that replace discrete neuron- and activation-level decisions with continuous relaxations, enabling differentiable optimization over otherwise combinatorial architecture spaces. These formulations give rise to three NAS methods: NAS based on Neuron Gating (NAS-NG), NAS based on Mixed Activation (NAS-MA), and NAS based on Neuron Gating and Mixed Activation (NAS-NGMA). Experiments on MLPs and CNNs using MNIST and CIFAR-10 show that the proposed methods consistently identify compact architectures with competitive or improved predictive performance. On MNIST, NAS-NGMA achieves 98.68% test accuracy with 7.69M MLP parameters, while NAS-NG achieves 99.63% accuracy with only 0.26M CNN parameters. On CIFAR-10, the proposed methods consistently outperform vanilla DARTS. Further experiments demonstrate that NAS-NG can optimize substantially over-parameterized and literature-optimal architectures, improving accuracy while reducing parameters. These results establish relaxed bilevel optimization as a scalable alternative to discrete NAS and provide a general framework for efficient neuron- and activation-level architecture optimization. 

---
# From Style Replication to Style Exploration: Enabling Art Style Exploration with Analyze-Experiment-Resituate Framework 

**Authors**: Wen-Fan Wang, TsaiHsuan Lin, Chi-Lan Yang, An-Ru Cheng, Bing-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.14405)  

**Abstract**: Art style is a signature of professional digital artists that develops through repeated experimentation, reflection, and adaptation. While generative AI (GenAI) can reproduce styles with high fidelity, current tools provide limited support for exploring new stylistic directions and may encourage style replication over exploration. To address this gap, we propose Analyze-Experiment-Resituate (AER), a framework for AI-assisted style exploration derived from interviews with 10 professional digital artists. Rather than prioritizing visually appealing outputs alone, AER supports three core practices of style exploration, including interpreting references, trying out stylistic possibilities, and reflecting on how emerging styles may be received. Specifically, AER enabled artists to (1) analyze artworks into interpretable stylistic elements, (2) have controllable experimentation guided by their own choices, and (3) resituate emerging styles through simulated social perspectives. We implemented AER in a prototype system and evaluated it in a controlled study with 16 artists. Compared with a direct style-transfer workflow, AER increased artists' agency and reflection as they pursued new stylistic directions. A two-week field study with four artists revealed how the AER framework influenced daily style exploration, such as reflection, experimentation, and stylistic decision-making at each stage. We discuss opportunities and challenges in designing AI-assisted style-exploration workflows, and outline implications for future artistic support tools. 

---
# Whose doctor does the AI recommend? An algorithm audit of reputation and demographic signals in large language model-assisted physician choice 

**Authors**: Syeda Anshrah Gillani, Mirza Samad Ahmed Baig  

**Link**: [PDF](https://arxiv.org/pdf/2608.14399)  

**Abstract**: Patients increasingly ask large language model (LLM) assistants which doctor to see, making these systems AI infomediaries: algorithms that intermediate one person's choice among other people and thereby decide, silently and at scale, which physicians become visible. We report a prespecified randomized algorithm audit of what causally moves those recommendations. Seven models (six open-weight; gpt-4o-mini) each chose among five synthetic family-medicine physician cards whose attributes were independently randomized across 3,024 choice sets, three patient personas, nine prompt paraphrases and nine experimental arms, yielding 40,068 scored responses; gender and ethnicity were signaled through names following correspondence-audit methodology. Reputation signals dominate: raising a rating from 3.9 to 4.7 increases choice probability by 31.4 percentage points (pp), and raising the fee from $90 to $190 lowers it by 20.0 pp. Demographic parity is rejected, but not in the direction human audit studies predict: female-signaled names gain 2.5 pp, and Hispanic-, South-Asian- and Black-signaled names gain 1.3-2.9 pp over White-signaled names, tilts worth $7-$14 per visit in fee-equivalent terms, and a content-free first-listed position is worth $11. Yet models mentioned gender or ethnicity in at most 0.03% of their stated reasons and abstained in 0.39% of trials, so these effects are invisible in the models' own explanations, and transparency obligations relying on model self-report would not detect them. One reasoning model failed the prespecified auditability gate outright. The frozen design makes the audit repeatable: any new model can be assessed against identical stimuli, making recurring behavioural audit, rather than self-reported explanation, the monitoring technology fit for purpose. 

---
# AI-Assisted Discovery and Construction of a Counterexample to the Convergence of Three-Block ADMM with the Identity Matrix as its Third Constraint Block 

**Authors**: Kenan Xu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14396)  

**Abstract**: The alternating direction method of multipliers (ADMM), as a landmark algorithm, has attracted tremendous research attention and extensive practical applications over the past two decades. It is well known that, although the two-block ADMM enjoys well-established theoretical convergence guarantees, its direct extension to the three-block case may fail to converge, as demonstrated by existing counterexamples [5]. However, to the best of our knowledge, the case in which the third constraint block is the identity remains unresolved: the existing literature gives neither a general convergence proof nor a counterexample for this subclass. In this paper, we give a negative answer: direct three-block ADMM may fail even when the first two blocks are strongly convex quadratics. Using Codex with GPT-5.6 Sol, we construct an explicit rational counterexample candidate and verify it along a piecewise-affine reduction path; exact checks show that direct three-block ADMM on this instance produces a bounded nonconvergent orbit of period 66. Within the same Codex workflow, we further guide a study of multiplier relaxation and clarify when convergence can be restored at the fixed-instance and class levels: a problem-dependent small dual step can restore convergence, whereas no positive relative step works uniformly over the whole class. Furthermore, we also test the recent Kimi Code with Kimi K3 model without the Codex candidate or project-specific route guidance; along a different path it produces an exact locally attracting period-23 certificate, convertible to an equivalent all-identity instance. The comparison suggests that different research-harness configurations can shape the mathematical objects explored and the certificates pursued. 

---
# Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events? A Systematic Evaluation of Detectors, Generators and Social Dissemination 

**Authors**: Shuo Liang, Yixing Ma, Pengfei Zhou, Xingyan Chen, Zihan Mei, Manting Li, Feihan Chen, Zhiwen Wang, Bin Xu, Haotian Zhang, Jiajun Song, Shiya Su, Run Liu, Zhenghang Ni, Yifa Yu, Jintao Hong, Bolong Feng, Yifei Liu, Zirui Zhang, Jingxuan Zhang, Songlin Zhao, Yifan Bai, Kang Tan, Yizhe Liu, Junhao Du, Yongtao Ge, Zhaopan Xv, Xinyuan Zhang, Mengru Ma, Chunhua Shen, Wei Wang, Yang You, Zheng Zhu, Kaipeng Zhang, Wangbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.14391)  

**Abstract**: Recent video generators can fabricate realistic depictions of wars, disasters, public emergencies, and other real-world crises, creating substantial risks of misinformation. Existing benchmarks, however, provide limited evidence on detector and generator behavior in such settings, including how detectability varies with generation conditions, how people perceive generated videos, and whether detectors remain reliable during social dissemination. To address this gap, we introduce RA-Bench, a benchmark for AI-generated video detection that uses Real videos as Anchors. RA-Bench contains 17,886 videos, comprising 1,830 real-video anchors across 10 social-risk categories and 16,056 generated clips from four open-source and five closed-source generators. Based on RA-Bench, we organize our evaluation along three dimensions. We first assess detector generalization across seven traditional detectors, ten zero-shot multimodal models under three review settings, and two MLLMs specifically fine-tuned on AI-generated video detection. Across these methods, none of the three detector families generalizes consistently across RA-Bench instances. We then examine how detectability varies with generation quality, conditioning information, and sampling seeds. These analyses show that generation properties affect detector families differently, while source-level detection patterns remain stable across seeds. Finally, we study human authenticity judgments and detector reliability during social dissemination. We find that videos that mislead people are also difficult for current detectors, and that social dissemination makes detection harder. Together, these findings show that current methods struggle to detect realistic AI-generated videos, highlighting the need for detectors robust to evolving video generators. 

---
# GBU-Palm: A Multimodal Video Dataset and Benchmark for Palm Presentation Attack Detection 

**Authors**: Yingjie Ma, Zitong Yu, Wei Jia, Ajay Kumar, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.14389)  

**Abstract**: Existing palm presentation attack detection (PAD) datasets are often limited by static imagery, restricted acquisition conditions, or insufficient multimodal video data, hindering systematic evaluation across environments, modalities, and attack types. We present GBU-Palm, a large-scale multimodal video dataset and benchmark containing 21,326 videos from 105 subjects and 210 palms across six acquisition environments, including bona fide, Print, and Replay presentations, with 6,310 synchronized RGB-NIR samples. We construct leakage-controlled protocols that separate palm identity and attack lineage and benchmark four representative video architectures under environment-matched and held-out-environment settings. Results reveal substantial architecture-dependent degradation under environmental shift and show that RGB-NIR fusion does not consistently outperform RGB-only input. We further analyze model behavior through true accept (TA), true reject (TR), false accept (FA), and false reject (FR) decomposition, spectral masking, temporal-order intervention, and frozen-backbone NIR probing, revealing distinct failure patterns and evidence utilization across architectures. GBU-Palm provides a unified and challenging benchmark for developing and evaluating robust multimodal palm PAD methods under cross-environment conditions. 

---
# DeaMoE: Efficient MoE Structure for Fast Small-Batch Decoding 

**Authors**: Zewen Jin, Shen Fu, Zeping Duan, Shannon Wang, Weihao Wu, Chengjie Tang, Congkun Ai, Ping Gong, Zijian Dai, Youhui Bai, Cheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14385)  

**Abstract**: Mixture-of-Experts (MoE) models have been widely adopted in real-time interactive applications such as coding assistants, real-time audio-video interaction systems. To meet the extremely low response latency requirements of these scenarios, practitioners commonly employ small-batch decoding, under which MoE inference becomes memory-bound and is severely bottlenecked by expert weight loading. However, this bottleneck has received limited attention, and existing solutions such as post-training weight compression or fine-grained expert design during pre-training either degrade model accuracy or introduce additional computation and communication overhead. To tackle this issue, we propose DeaMoE, a decoding-efficient MoE architecture, in which the experts are grouped into several departments, and the experts belonging to the same department share most parameters since they come from the same professional field, and additionally each expert contains a few private parameters to reflect its uniqueness. Moreover, we design customized two-stage routing strategy for DeaMoE to avoid redundant loading, under which DeaMoE greatly improves the efficiency during LLM decoding. Compared with vanilla MoE, DeaMoE reduces per-step loaded weights by up to 50.9% and achieves up to 1.33 end-to-end TPOT speedup for the pre-trained 7B model on A40, and up to 2.00x and 1.97x peak speedup for DeepSeek-V3 on A40 and H100 in microbenchmarks. 

---
# Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation 

**Authors**: Yuxuan Chen, Wanruo Zhang, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.14379)  

**Abstract**: Vision-Language-Action (VLA) models have recently achieved promising performance in robotic manipulation. However, existing benchmarks mainly evaluate generalization on static manipulation tasks and largely overlook dynamic interaction scenarios. To address this gap, we present ReflexBench, a benchmark for reaction-critical manipulation. ReflexBench contains six dynamic tasks and introduces an evaluation framework that decouples simulator stepping from robot control while supporting configurable latency under synchronous and asynchronous inference. Building upon ReflexBench, we propose ReflexVLA, an efficient VLA model designed for reaction-critical manipulation without large-scale robot-data pretraining. ReflexVLA enhances temporal reasoning through latent future prediction and multi-frame temporal fusion within the vision backbone, while reducing deployment latency through batched visual encoding and CUDA Graph replay. Experiments show that ReflexVLA consistently improves dynamic manipulation performance while maintaining competitive accuracy on standard static manipulation benchmarks, and real-world experiments further demonstrate its effectiveness under practical deployment conditions. Project website: this https URL 

---
# A Hybrid LLM-Based Framework for Automated Security Annotation Generation in Business Process Models 

**Authors**: Md Kamrul Islam, Tiphaine Henry, Mattia Salnitri, Julius Köpke, Sami Souihi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14370)  

**Abstract**: The modelling and analysis of secure business processes require the incorporation of security annotations into process models. Although BPMN extensions, including SecBPMN2, exist for this purpose, the derivation of accurate and complete security annotations from natural-language specifications remains a manual, expert-intensive, and error-prone task. This paper presents a hybrid framework that takes a BPMN process model and a security requirements document as input and automatically generates security annotations adhering to the SecBPMN2 specification. The approach combines Large Language Model (LLM)--based semantic extraction with schema-constrained mapping, rule-based normalization, and deterministic validation. The framework is evaluated comprehensively on a curated dataset of 27 process models from various domains. The results indicate that it consistently produces structurally valid SecBPMN2 annotations with high schema completeness. Compared to human security analysts, the system achieves substantially higher precision (0.58 vs. 0.29) while maintaining comparable recall (0.52 vs. 0.50) and reduces erroneous or misplaced annotations by nearly 50%. In addition, annotation generation is significantly faster than manual annotation. These findings demonstrate that hybrid LLM- and rule-based automation can reduce modeling effort while improving consistency and reliability, thereby providing a scalable foundation for security-by-design BPM. 

---
# Mind the Long Tail: Understanding the Difficulty of Delay Detection in Business Processes 

**Authors**: Keyvan Amiri Elyasi, Lukas Kirchdorfer, Heiner Stuckenschmidt  

**Link**: [PDF](https://arxiv.org/pdf/2608.14367)  

**Abstract**: The early detection of delayed cases in business processes is a critical capability for organizations. Predictive process monitoring (PPM) supports this task by using historical event logs to predict the remaining time of ongoing cases, enabling timely interventions to avoid missed deadlines and service level violations. Although remaining time prediction has advanced considerably through sophisticated deep learning architectures, little is known about the intrinsic difficulty of delay detection itself. Since performance is typically assessed using aggregate metrics, prior work provides limited insight into how models perform across the target distribution, especially on the operationally most critical cases with large delays. In this paper, we address this gap by analyzing the difficulty of delay detection. Across 14 event logs, we show that remaining times are typically strongly right-skewed, with only a small fraction of cases exhibiting large delays. Existing models capture the mode of this distribution well but perform poorly on high-delay cases. We further uncover pronounced heteroscedasticity, showing that predictive uncertainty increases with delay magnitude. Based on these findings, we evaluate approaches to mitigate the imbalance problem, but find only limited benefits, suggesting that the key underlying problem may not be imbalance but higher uncertainty associated with delayed cases. We show that this correlation can be exploited to substantially improve the identification of delayed cases. Overall, our work provides new insights into the sources of difficulty in delay detection and identifies uncertainty-aware modeling as a promising direction for future PPM research. 

---
# A Four-Axis Trustworthiness Benchmark for LLM-as-Judge in Principle-Based Regulation 

**Authors**: Dipankar Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.14329)  

**Abstract**: Principle-based regulation, with evaluative standards such as "fair, clear, and not misleading" or "deliver good outcomes", cannot be reduced to binary predicates, and LLM-as-judge is increasingly used as the substitute. Our position is that any such judge must be evaluated on four axes: accuracy, paraphrase robustness, adversarial robustness, and calibration. We release Principle-Bench, 168 cryptoasset financial-promotion scenarios mapped to two UK FCA principles, with paraphrase, adversarial keyword-stuffing, and boundary perturbations authored under a pre-registered rubric; the first benchmark covering all four axes for principle-based regulation. We also introduce Ceca (Calibrated Exemplar-Cluster Assessment): a calibrated, auditable assessor that emits exact per-exemplar counterfactual attributions. Across keyword counting, three sentence-transformer embedders, an open-weight LLM-judge, and a calibrated cascade, no method dominates all four axes. A 120B LLM-judge, strongest on benign inputs, loses 47 accuracy points (0.74 to 0.27) on keyword-stuffed Consumer Duty inputs: "compliance theatre." A second judge from a different model family agrees only at Cohen's kappa = 0.16 on that split, localising the failure to the model rather than the corpus. Any deployment-grade LLM-judge for principle-based regulation must report per-principle adversarial deception and post-hoc calibration alongside aggregate accuracy. 

---
# Intelligent Detection of Mechanical, Electrical, and Plumbing (MEP) Metrics Based on 2D Floor Plans 

**Authors**: Tarandeep Singh Mandhiratta, ANK Zaman, Abdul-Rahman Mawlood-Yunis  

**Link**: [PDF](https://arxiv.org/pdf/2608.14317)  

**Abstract**: This research developed a neural network-based model to extract various information from 2D floor plans. We detect lighting symbols, identify the appropriate type of light, and extract the associated texts with lights. The study aims to enable efficient floor designing and determining the number and type of lights needed per floor, i.e., allow efficient design and estimate the power requirement of the floor plan. The model was developed using Mask RCNN as the base. The images were annotated and converted into a Coco data format for training the model. The model achieved bbox\_mAP and segm\_mAP values of 0.7596 and 0.7111, respectively. It also performed well at different IoU thresholds, i.e., with bbox\_mAP 50 and segm\_mAP 75 values of 0.9850 and 0.9219, respectively. The developed model will help various industries, such as architecture and construction, to improve design time and create efficient workflows by automatically detecting Mechanical, Electrical, and Plumbing (MEP) objects from floor plans, and it is the first step towards building tools that will help energy-efficient building design. 

---
# Acoustic UAV Detection in Battlefield Scenarios: Handling Noise, Domain Shift, and Weak Labels 

**Authors**: Vadym Vilhurin, Volodymyr Sydorskyi, Andrii Shevtsov  

**Link**: [PDF](https://arxiv.org/pdf/2608.14287)  

**Abstract**: Passive acoustic sensing offers a critical, cost-efficient, and, crucially, passive alternative for detecting small unmanned aerial vehicles. However, the practical deployment of acoustic systems is discouraged by extreme environmental noise and sensor-induced domain shift caused by heterogeneous hardware. This paper addresses these challenges by introducing a robust framework optimized for real-world battlefield conditions. We propose the integration of Per-Channel Energy Normalization (PCEN) and attention-based pooling to enhance feature extraction under low signal-to-noise ratio scenarios. We further propose a domain-aware training strategy that leverages auxiliary classes and multi-microphone data to mitigate cross-domain performance degradation. Evaluated on a unique dataset of combat-zone recordings from the Ukrainian frontlines, our approach significantly outperforms existing baselines, increasing the F1 score from 55.4% to 78.6%. This paper was originally presented at the International Conference on Military Communication and Information Systems (ICMCIS), organized by the Information Systems Technology (IST) Scientific and Technical Committee, IST-224-RSY - the ICMCIS, held in Bath, United Kingdom, 12-13 May 2026. 

---
# Seeing Red, Thinking Bad: Color Bias in Vision Language Models 

**Authors**: Kohsuke Ide, Ryousuke Yamada, Yoshihiro Fukuhara, Hirokatsu Kataoka, Yutaka Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2608.14286)  

**Abstract**: Vision language models (VLMs) are increasingly used in industrial decision-making systems, such as recruitment support and recommendation. This motivates careful analysis of how VLMs process visual and textual information. In this work, we study how VLMs interpret text rendered as an image, and investigate the influence of visual styling biases. To this end, we introduce Stealth Visual Prompts, which subtly change visual styling of text, such as color and contrast, while preserving semantic content. Using these prompts, we systematically control the visual styling of words in text and measure their impact on the analysis performed by VLMs. We further analyze how such visual perturbations affect the latent representations of the vision encoder. From our experiments, we observed that coloring positive words in green consistently shifts sentiment predictions toward a positive direction. As a result, VLMs often fail to properly account for negative words present in the text. Our analysis suggests that this behavior is correlated with changes in the latent representations of the vision encoder induced by color variations. In addition, we show that reducing text--background contrast increases reliance on visually salient cues and leads to more incorrect Visual Question Answering (VQA) outputs. These results suggest that the visual styling of rendered text can guide VLMs' interpretation in ways that diverge from human semantic understanding.
Project page: this https URL 

---
# SimpleOPD: Simple Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning 

**Authors**: Haonan He, Haodi Lei, Yun Luo, Haoran Zhang, Shunkai Zhang, Yizhuo Li, Shengji Tang, Zhilin Wang, Runzhe Zhan, Lei Bai, Ganqu Cui, Fangchen Yu, Yafu Li, Peng Ye, Ning Ding, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14277)  

**Abstract**: On-policy distillation (OPD) offers a promising way to transfer reasoning capabilities from stronger teacher models, but applying it to long-context reasoning teachers and short-context students introduces practical challenges, including tokenizer mismatch, teacher-student distribution mismatch, response length explosion, and training instability. In this work, we study this setting by transferring proof-reasoning capabilities from the long-context reasoning model SU-01 to short-context student models. To handle tokenizer differences, we perform OPD in a shared text space and align only tokens that occupy identical text spans under the student and teacher tokenizers. To mitigate the problem of excessive generation length and frequent truncation, we introduce a student reference KL loss and mask the advantages of special termination tokens such as </think> and <|im_end|>. This strategy constrains the student from drifting excessively from its initial policy, thereby mitigating the teacher-student distribution mismatch problem and fostering steady length growth. Experiments on both same-family and different-family student models, including Qwen3, Qwen3.5, Intern-S2, GLM-4.7, Gemma-4, show consistent gains in mathematical reasoning, especially natural-language math proving. Notably, Intern-S2-Preview improves by 21.2 points on ProofBench, reaching 55.2 and surpassing Gemini-2.5-Pro. It also improves on science benchmarks such as HLE and HiPhO, suggesting that OPD transfers reasoning capabilities that generalize beyond the mathematical training domain. 

---
# Multi-Objective Bayesian Optimization for Model Merging 

**Authors**: Utkarsh Agarwal, Vamshi Bonagiri, Raul Astudillo, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.14264)  

**Abstract**: Model merging combines trained models directly in weight space, offering a compute-efficient alternative to additional fine-tuning. Selecting merge parameters is nevertheless difficult because downstream evaluations are expensive, gradients are unavailable, and source capabilities can conflict. We formulate merge-parameter selection as a black-box multi-objective optimization problem and introduce MOBO-Merge, a merge-operator agnostic framework that uses multi-objective Bayesian optimization to approximate the Pareto front under a limited evaluation budget. We evaluate Qwen3-4B and Llama-3.1-8B in two-model instruction-math and three-model instruction-math-code settings using Linear, SLERP, TIES, and block-wise merge operators. On held-out benchmark partitions, MOBO-Merge obtains higher mean hypervolume than random search in 11 of 12 reported comparisons. The gain is small for one-dimensional Linear interpolation but substantially larger for several TIES, block-wise, and three-objective searches. No merge operator is uniformly best: TIES leads in three of four family-setting combinations, whereas Block-Linear 4x is strongest for the Llama three-model merge. These results show that multi-objective Bayesian optimization is valuable as a search layer for expressive merge parameterizations. 

---
# Meteorology-driven Causal Nowcasting of Fugitive Landfill Emissions Enables Proactive Public Health Response 

**Authors**: Timothy C. Pearce, David J. T. Smith, Alec Dobney, Alessia Freddo  

**Link**: [PDF](https://arxiv.org/pdf/2608.14254)  

**Abstract**: Fugitive emissions from waste sites increasingly expose communities to toxic and odorous gases, yet public-health responses remain largely retrospective, with episodes investigated only after residents have been exposed. Here we show that the meteorological drivers of elevated hydrogen sulphide (HS) at a long-monitored European landfill, and the timescales over which they act, can be identified directly from routine monitoring data. We introduce CAIRN (Causal-Anchored Inference for Receptor Nowcasting), a machine-learning framework whose internal memory is matched to these measured timescales: a fast component tracking hour-scale wind-borne transport and a slow component tracking multi-hour weather changes. Trained to predict gas measurements, CAIRN operates using only routine weather variables and the calendar, without hand-engineered features. Its behaviour is consistent with the identified transport mechanisms, and the framework transfers unchanged to a second monitoring station and to co-emitted methane. Combining four such nowcasters produces a site-level, tiered alert aligned with WHO odour guidance that closely reproduces the alert generated by a direct sensor network and tracks an independent record of community odour complaints. Weather-driven nowcasting can therefore estimate community impact as an emission episode unfolds, providing public-health authorities with a validated, graded trigger for intervention and enabling exposure to be reduced during events rather than after them. 

---
# Training Fair Tabular Foundation Models 

**Authors**: Patrik Kenfack, Jesse C. Cresswell, Anthony L. Caterini, Samira Ebrahimi Kahou, Ulrich Aïvodji  

**Link**: [PDF](https://arxiv.org/pdf/2608.14211)  

**Abstract**: Tabular Foundation Models (TFMs) have emerged as leading methods for tabular predictive tasks, leveraging in-context learning to predict on new data without task-specific training. Despite the increased use of TFMs in high-stakes decision-making, their fairness properties remain largely unexplored. In this work, we incorporate fairness constraints directly into TFM training, enabling fair predictions in a single forward pass. Our approach addresses two key challenges: limited access to sensitive attributes in training data, and the incompatibility of existing fairness techniques with the in-context learning paradigm. We propose FairTFM, a scalable training strategy based on synthetic fairness tasks and a fairness-aware architecture using a gradient reversal layer, which encourages the model to learn representations invariant to sensitive attributes. Experiments on 132 fairness tasks show consistent improvements in fairness while maintaining competitive accuracy. 

---
# How Much Do Legal RAG Systems Still Hallucinate? 

**Authors**: Souvick Das, Sallam Abualhaija, Domenico Bianculli  

**Link**: [PDF](https://arxiv.org/pdf/2608.14210)  

**Abstract**: Hallucination is a major challenge for retrieval-augmented generation (RAG) systems in the legal domain, where ungrounded answers can lead to serious consequences. To better understand this problem, we conduct a fine-grained analysis of hallucination behavior in eight legal RAG systems across two legal corpora, the GDPR (in English) and a national civil law (in French). Using claim-level and answer-level evaluation, we report on hallucination density and severity, analyze performance across question categories and user personas, and validate our findings on an independent set of 142 legal-expert-authored questions. Our results show that hallucinations remain pervasive, ranging from less than 10% of responses for the best-performing systems to nearly half in the worst case. We further find that false-premise questions, containing incorrect assumptions that must be rejected, produce high hallucination rates on the manually-drafted questions. 

---
# Structure-Guided Spatiotemporal Attention Graph Neural Network for Traffic Flow Prediction 

**Authors**: Xuanmian He, Can Li, Wanjing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.14177)  

**Abstract**: Deep spatiotemporal models integrating graph convolutions and attention mechanisms have demonstrated excellent performance in network-level traffic flow prediction, owing to their exceptional ability to capture complex spatiotemporal dependencies. Despite their predictive success, deployment of such models in safety-critical urban systems remains constrained by their inherent lack of transparency. Existing post-hoc diagnostic methods often struggle with spurious correlations and fail to unveil the intrinsic decision-making mechanisms governing traffic dynamics, resulting in suboptimal interpretability and limited operational trustworthiness. To address these challenges, this paper proposes the Structure-Guided Spatiotemporal Attention Graph Neural Network (SGSAN). Departing from traditional architectures that rely on unconstrained adaptive graphs, SGSAN explicitly learns a static Directed Dependency Graph (DDG) to identify the invariant macroscopic propagation paths of traffic states. We further introduce an InfoNCE-based soft-coupling mechanism that anchors the model's dynamic spatiotemporal attention to this structural prior, offering a mechanistic account of the model's decision-making process while ensuring robust forecasting by aligning attention-based reasoning with identified macroscopic dependencies and preventing over-reliance on ephemeral local noise. Furthermore, a decoupled two-stage optimization framework is developed to resolve the fundamental conflict between structural discovery and predictive error minimization. Extensive experiments on multiple real-world datasets demonstrate that SGSAN achieves state-of-the-art predictive accuracy while providing built-in interpretability that organically aligns with the physical logic of traffic networks. 

---
# Concept Guidance: Precise, Training-Free Latent Control for Text-to-Image Generation 

**Authors**: Nikolai Röhrich, Isabell Hans, Felix Krause, Björn Ommer  

**Link**: [PDF](https://arxiv.org/pdf/2608.14172)  

**Abstract**: Text-to-image diffusion models have two major drawbacks that severely limit their practical utility: (1) standard models lack an intrinsic mechanism for continuous, concept-specific guidance (e.g., for precisely controlling how aesthetically pleasing an image looks), and (2) they lack reliability for tasks requiring high local coherence (e.g., generating text or human hands). To tackle these issues, we introduce a novel notion of concept-wise mutual information and find large, concept-dependent differences between individual layers, demonstrating that the generation of specific structures is localized in distinct parts of the network. We exploit this insight by reinforcing the impact of concept-relevant layers in Concept Guidance (CoG), a precise, target-specific guidance method that works for models out-of-the-box without additional training, external models, gradients, or prompt engineering. CoG first quantifies each layer's concept-specific impact and then guides denoising using a weighted combination of predictions generated with concept-relevant layers skipped. We demonstrate performance increases across various targets and popular models like PixArt-alpha, SD3, SD3.5, and FLUX.1-dev. Code is available at this https URL 

---
# Self-Supervised Visual On-Policy Distillation 

**Authors**: Yijiang Li, Yijun Liang, Yunjie Tian, Bingyang Wang, Ke Zhang, Zhenfei Yin, Di Fu, Philip Torr, Nuno Vasconcelos  

**Link**: [PDF](https://arxiv.org/pdf/2608.14144)  

**Abstract**: Visual on-policy distillation relies heavily on an informative teacher-student asymmetry, through either a larger, stronger teacher or privileged supervision, such as reference answers or ground-truth regions of interest. This raises a fundamental question: where can informative asymmetry come from when nothing privileged is available? We answer this by inverting where the asymmetry comes from. Rather than adding privileged information to the teacher, we subtract information from the student. This asymmetry creates the same effective learning signal for free as a teacher with access to information unavailable to the student, without ground-truth annotations, rewards, or a separate stronger teacher model. Building on this principle, we introduce Self-Supervised Visual On-Policy Distillation (S$^2$VOPD), a simple yet effective method that constructs on-policy learning signals from asymmetric augmented views. S$^2$VOPD distills the teacher's distribution conditioned on the original image on-policy into the student distribution conditioned on a strongly augmented view of the same image. We systematically explore a broad design space of visual augmentations and uncover that (1) asymmetry matters: all four augmentation families improve performance, while symmetric self-distillation degrades it; (2) strength matters: performance peaks at a moderate strength; and (3) the gap must remain task-consistent: augmentations that completely remove the question-relevant evidence can induce large but uninformative discrepancies. Across six fine-grained perception benchmarks, S$^2$VOPD improves Qwen3.5-4B from 70.7% to 77.4%, above all open-source models compared, up to Qwen3-VL at 235B, and surpasses GPT-5.4. While holding training data the same, it recovers 96% of the improvement achieved by methods with privileged information. Website is at this https URL 

---
# SPARGen: Unifying Spatial Perception and Reasoning through Native Multimodal Generation 

**Authors**: Jinsheng Quan, Jianhua Li, Siyi Xie, Xuanke Shi, Kewang Deng, Zukai Chen, Feifei Shao, Lei Yang, Quan Wang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.14138)  

**Abstract**: Spatial perception and reasoning from visual observations require recovering geometric structure, establishing correspondences, and understanding spatial relations. Existing approaches typically address these capabilities separately using task-specific architectures or external geometric modules, limiting knowledge transfer among complementary representations of the same physical scene. We introduce SPARGen, a unified multimodal framework that casts 3D reconstruction, dense correspondence, and spatial reasoning as instruction-conditioned generation tasks. SPARGen serializes compact structured and linguistic outputs as token sequences while generating dense geometric fields in image-aligned forms, enabling spatial supervision to jointly shape shared representations within a native multimodal generative model. Experiments across benchmarks for 3D reconstruction, correspondence, and spatial reasoning show that SPARGen achieves competitive performance across heterogeneous spatial tasks within a single native multimodal generative framework. 

---
# HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting 

**Authors**: Wei Zhang, Shengkai Yu, Shiqiang Gong, Qi Zhang, Qiang Li, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14136)  

**Abstract**: Octree-based anchor Gaussian Splatting has emerged as a scalable representation for city-scale novel view synthesis, where multi-level anchors adaptively capture scene content from coarse building structures to fine architectural details. However, we identify a fundamental limitation in existing methods: cross-level feature isolation, where each level's anchor features are optimized independently with no inter-level communication, causing color drift on building facades and over-smoothing in textured regions. We present HiCo-GS, a high-fidelity reconstruction framework with two complementary modules. Cross-Level Context Aggregation (CLCA) enables bidirectional hierarchical prior injection by leveraging the octree's spatial containment structure to aggregate per-level context vectors into parent-self-child triplets, fused via a lightweight MLP with residual connection. Coarse-level structural priors flow down to inform fine-level anchors, while fine-level detail statistics feed back to prevent over-smoothing, at negligible computational overhead. Depth-Normal Geometric Consistency (DNGC) regularization enforces agreement between rendered normals and depth-derived normals through an alpha-weighted consistency loss, complemented by edge-aware smoothness losses with progressive warmup that exploit the strong planar priors ubiquitous in urban geometry to suppress floating artifacts. We further introduce the China-Pagoda dataset comprising 8 ancient Chinese pagodas with over 1,200 images each, featuring dense ornamental carvings, curved multi-layer eaves, and repetitive fine-grained textures. Extensive experiments on Mill19, UrbanScene3D, MatrixCity, and China-Pagoda demonstrate that HiCo-GS achieves state-of-the-art rendering quality and substantially cleaner geometry across real-world and synthetic urban this http URL: this https URL. 

---
# Act2Intention: A Benchmark For Developing Active Mobile Agents Through Inferring User Intention from GUI Actions 

**Authors**: Xiaokai Yan, Jingtao Ding, Yong Li, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14132)  

**Abstract**: Mobile GUI Agents powered by multimodal large language models (MLLMs) show promise in human-computer intelligence. However, current research primarily focuses on reactive task execution while lacking a comprehensive understanding-prediction-execution process for user intentions, which are the core requirements of active agents. In this paper, we propose the Act2Intention framework that builds an active mobile agent by integrating understanding, predicting user intentions, and executing decisions. First, we construct the Act2Intention Bench through data collection and validated generation, comprising 72,511 intentions and over 700,000 actions across 52 apps, thereby establishing the first benchmark for evaluating proactive agents via continuous intention-action trajectories. We further develop the Act2Intention Agent, achieving proactive services through Proactive-oriented Intention Understanding, Personalized Proactive Intention Prediction, and Experience-guided Intention Execution. Experimental results show that supervised fine-tuning on Act2Intention Bench yields absolute improvements of +32.0 Acc-S, +10.25 Acc-S, and +6.9 SSR points over non-fine-tuned counterparts under the same agent framework for intention understanding, prediction, and execution, respectively. This success underscores the necessity and value of the Act2Intention Bench, which establishes a standardized platform for developing and evaluating proactive agents and consequently paves the way for research on intention-driven human-computer interaction. 

---
# AlignFace: Human-Aligned Face Similarity Metric with Interpretable Concept Relations 

**Authors**: Ying Huang, Wencan Zhang, Brian Y. Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.14130)  

**Abstract**: Computer vision models for generated facial content, such as face editing and privacy protection, increasingly affect people, requiring similarity metrics that serve as faithful proxies for human perception. While perceptual evaluation has progressed from signal-based heuristics to representation-based metrics, current approaches are limited to behavioral modeling without cognitive alignment. They rely on implicit and spurious relations while assuming a universal observer, failing to account for inherent variations across diverse human populations. This leads to inaccurate evaluative models of stakeholders and misleading guidance for generative model debugging. Rather than treating perception as a black box, we leverage scientific findings from cognitive psychology of human face similarity perception: dependence on facial featural and configural attributes, nonlinear psychophysical response scaling, and own-group biases. We introduce the FACETS dataset and propose AlignFace, an interpretable, human-aligned, face similarity metric that encodes these cognitive principles through ante-hoc modeling. It employs visual-language modeling (VLM) to encode paired face images and text-based attributes, gated cross-attention (CA) to extract attribute-specific facial difference representations, concept bottleneck modeling (CBM) to constrain reasoning via interpretable face attributes, and neural generalized additive model (GAM) to model their nonlinear influence. Experiments found AlignFace significantly improves alignment with human subpopulation perceptions compared to baseline metrics, including recent domain-free learned perceptual metrics. By bridging learned representations and human cognitive processes, this work enables more transparent and aligned perceptual evaluation metrics for face images. 

---
# BGA: A noise-immune neural distillation framework for malicious signature extraction in high-entropy encrypted flows 

**Authors**: Sheng Hong, Yixuan Huang, Weiwei Jiang, Junyuan Zhang, Jiacheng Wang, Ruijian Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.14126)  

**Abstract**: To mitigate attention dilution in high-entropy TLS 1.3 flows, we propose BGA, a noise-immune neural distillation framework for encrypted threat this http URL methodology first employs Analysis of Variance (ANOVA) to decouple high-discriminatory control-plane features - specifically industrial setpoints - from stochastic cryptographic noise. To resolve the extreme class imbalance within a corpus of 86,878 flow records, a Wasserstein GAN with Gradient Penalty (WGAN-GP) module, enforcing the 1-Lipschitz constraint, is integrated to synthesize high-fidelity minority samples, elevating the detection recall of rare Malicious State Command Injections(MSCI) attacks by 43.2%. At its core, the BGA architecture integrates Bidirectional Long Short-Term Memory (BiLSTM) for temporal dependency extraction and an Adaptive Gated Multi-Head Attention mechanism. This gated unit functions as a neural filter to dynamically suppress encryption artifacts while amplifying malicious signatures. Extensive evaluations on CIC-IDS-2018 and Edge-IIoT benchmarks demonstrate a performance ceiling exceeding 95.2% across all key metrics. Furthermore, noise-injection stress tests confirm BGAs superior structural resilience with a 8.57% performance margin over vanilla Transformers, while its ultra-low inference latency of 0.2820 ms (estimated 1.6920 ms via theoretical scaling for ARM) indicates a high potential for real-time feasibility on heterogeneous industrial edge gateways, providing a promising architectural baseline for future hardware implementation. 

---
# Overcoming Shortcut Learning in Graph Neural Networks through Active Explanation Guidance 

**Authors**: Taraneh Younesian, Steve Azzolin, Antonio Longa, Francesco Ferrini, Vincenzo Marco De Luca, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2608.14121)  

**Abstract**: Graph Neural Networks (GNNs) can solve prediction tasks by unintentionally exploiting shortcuts---that is, edges, nodes, and features that correlate with but are not causal for the prediction---which compromise their reliability in out-of-distribution tasks. We introduce XIGL, an architecture-agnostic human-in-the-loop strategy for removing such shortcuts from GNNs. Our key insight is twofold. On the one hand, reliance on shortcuts can be detected by inspecting GNN explanations. On the other hand, once made aware of such shortcuts, sufficiently expert users can provide tailored corrective feedback, which helps deconfound the model. XIGL supports any query strategy; however, since corrective feedback can be expensive to acquire, we develop an active learning strategy for prioritizing explanations that are more likely to display shortcut behavior, lowering annotation and cognitive costs. We showcase the effectiveness of XIGL, including both existing and proposed explanation-based strategies, on several GNN architectures. Our implementation is available online. 

---
# From Fixed Grids to Moving Particles:A Transferable Latent Operator for Fluid Dynamics 

**Authors**: Meng Li, Chuqi Chen, Zhengqing Gao, Xi Zhou, Xiao Sun, Yang Xiang, Huaxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14120)  

**Abstract**: Lagrangian modeling is vital to fluid dynamics, as it characterizes particle transport and complements the Eulerian this http URL, Lagrangian trajectories are less commonly available than Eulerian fields, while most neural operators are trained and evaluated primarily in the Eulerian representation. This mismatch motivates a new learning problem: can a model trained solely on Eulerian observations generalize zero-shot from Eulerian field prediction to Lagrangian particle rollout, without Lagrangian supervision or task-specific adaptation? To address this problem, we propose the Transferable Latent Operator (TLO), which learns a unified flow representation shared by Eulerian field prediction and Lagrangian particle rollout. TLO decouples latent flow evolution from coordinate-dependent decoding: querying the evolving latent representation at fixed spatial coordinates yields Eulerian fields, whereas querying velocities at particle positions and recursively updating these positions enables Lagrangian rollout. Across five fluid-dynamics benchmarks, TLO consistently outperforms existing neural operators in both Eulerian field prediction and zero-shot Lagrangian rollout, with further gains from limited Lagrangian fine-tuning. 

---
# Fixed-Budget Gaussian Volume Encoding with Structure-Aware Allocation 

**Authors**: Michael R. Martin, Joseph Insley, Victor A. Mateevitsi, Silvio Rizzi, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.14112)  

**Abstract**: Scientific simulations often produce scalar volumes faster than they can be stored, transferred, and loaded, while in situ reduction must use only a limited share of simulation resources. This work encodes scalar fields as anisotropic Gaussian primitives under a fixed budget. The complete primitive set is allocated analytically from local field structure, including position, orientation, and shape, then refined directly against the scalar field without densification, pruning, or count changes. The selected budget determines encoded storage before refinement and, together with the iteration schedule, provides a controllable refinement-time budget. In a controlled benchmark, truncation-aware field evaluation reduces encoding time by up to 51x; 1.4 million Gaussians encode a billion-voxel volume in at most four minutes on one desktop GPU, with reduced-iteration refinement completing in under one minute. Across five datasets spanning 2.1 million to 1.1 billion evaluated voxels, compression-useful configurations achieve 15.0-38.7 dB PSNR at compression ratios from 2.2x to over 40,000x. Pre-encoding structure statistics characterize fields for which one-shot allocation yields limited gains from additional capacity. Because primitives retain scalar attributes rather than baked appearance, a single compact model serves every subsequent visualization state - supporting post-hoc transfer-function, colormap, lighting, and viewpoint changes without re-encoding. 

---
# Forecast Collapse in Time-Series Foundation Models 

**Authors**: Shu Wan, Miles Ma, Hank Zhu, Guangqi Liu, Stephen Wang, Qingsong Wen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14106)  

**Abstract**: When forecasting hourly returns for 1,000 US equities, we observe an unexpected phenomenon: predictions become nearly flat and show poor stock ranking, as measured by cross-sectional correlation. We call this forecast collapse. Surprisingly, the phenomenon largely disappears when forecasting trading volume under the same setting. We investigate forecast collapse across time-series foundation models (TSFMs), twelve deep-learning forecasting models, and 97 public benchmark configurations, and find that it is closely tied to target predictability. We identify two distinct reasons behind it: low predictability limits the amplitude of calibrated point forecasts, while per-series objectives leave cross-series structure unidentified. These findings reveal a calibration-ranking tradeoff: optimizing squared error leads to flat predictions, whereas directly optimizing cross-sectional correlation improves ranking but can inflate forecast amplitude by more than an order of magnitude. To address this tradeoff, we introduce CalibRank, a simple objective that balances calibration and ranking. On Finance1K, CalibRank nearly triples cross-sectional correlation while keeping amplitude close to the target, and improves correlation on all tested models. Our results reveal a blind spot in conventional time-series evaluation: per-series metrics can hide failures in cross-series structure needed by downstream decisions. 

---
# Rewrite Once, Validate Anywhere: Producing OWL-Aware SHACL Constraints (Extended Version) 

**Authors**: Anouk Oudshoorn, Piotr Gorczyca, Dörthe Arndt  

**Link**: [PDF](https://arxiv.org/pdf/2608.14104)  

**Abstract**: The Shapes Constraint Language (SHACL) is a W3C recommendation to express syntactic constraints, called shapes, on RDF graphs. SHACL validators are used to test whether a given graph adheres to such a shape. However, RDF graphs often come with OWL ontologies, whose implicit knowledge needs to be taken into account. This is classically handled by first applying reasoning and then performing the constraint checking on the results, often using different technologies which makes the process inefficient and vulnerable for mistakes.
To overcome this, we propose to internalise the OWL axioms in the SHACL constraints; we construct a rewriting which takes as input both shapes and an OWL EL$^-$ ontology -- a fragment of OWL EL restricting the usage of existential restrictions -- and produces SHACL constraints. This output can then be evaluated by any validator supporting SHACL core regardless of its reasoning support, while yielding the same results as the traditional approach. The implementation of our translation is evaluated both against applying state-of-the-art reasoners and validators consecutively, as against validators with built-in reasoning support. For our benchmark, we show that our approach is in general more efficient in finding violations compared to the sequential approach, thus providing a powerful tool which simplifies combining reasoning with validation. 

---
# P2Skill: Privacy Preserving Skill Distillation for Cloud-Local LLM Inference Systems 

**Authors**: Myunghoon Ryu, Geunpyo Park, Sungjoon Lee, XinYu Piao, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.14094)  

**Abstract**: Cloud-local LLM inference systems have the potential to use the reasoning capability of large cloud models while protecting sensitive user data on personal devices. Cloud-bound requests must exclude personally identifiable information (PII) to prevent external data leakage. Existing privacy-preserving methods rely on prompt perturbation, entity masking, or model fine-tuning, but these approaches may distort contextual semantics or require additional training. This paper proposes P2Skill, a prompt-based skill distillation method in which a local small language model (SLM) autonomously performs decomposition, PII-aware routing, paraphrasing, and reconstruction by following the skill prompts. Skills are iteratively refined from execution failures by a cloud LLM, enabling the local SLM to generalize beyond memorized PII patterns, and therefore P2Skill requires no privacy-specific fine-tuning or learned auxiliary detectors. Evaluation on a four-domain benchmark shows that P2Skill achieves $1.69\times$ and $3.66\times$ higher privacy-preserved inference quality than previous baselines. 

---
# Reaction-Transformation-Aware Flow Matching for Generalizable Transition State Generation 

**Authors**: Kaipeng Zeng, Wenxi Zhai, Shengrui Xu, Jie Zhao, Bowen Li, Shiyue Wang, Junchi Yan, Tong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.14076)  

**Abstract**: Transition-state (TS) structures define the energetic barriers and mechanistic pathways of elementary chemical reactions, yet their identification remains computationally demanding because conventional saddle-point searches require expensive quantum-mechanical calculations. Recent machine-learning approaches have accelerated TS generation by predicting structures from reaction endpoint information, but they primarily learn geometric correspondence between endpoints and TSs, leaving the structural transformations underlying elementary reactions implicitly represented. To address this limitation, we introduce TransTS, a reaction-transformation-aware framework for generalizable TS generation from atom-mapped reactant-product pairs. TransTS explicitly learns atom-level structural transformations between reaction endpoints and integrates them with a unified atom-aligned geometric representation of reactants, TSs and products, enabling reaction-aware equivariant generation of TS geometries. TransTS is designed to provide reliable TS initial guesses for subsequent quantum-chemical refinement, where generated structures are evaluated not only by geometric similarity but also by their ability to converge to validated saddle points and recover the intended reaction pathways. Across IID and zero-shot OOD benchmarks, TransTS demonstrates improved TS initialization quality, with particularly strong generalization to unseen reaction distributions. On the challenging GDB-10-rxn and GDB-17-rxn OOD benchmarks, TransTS generates TS candidates that more frequently converge to validated saddle points and recover the intended elementary reactions after refinement than existing approaches under the same training regime. Scaling reaction coverage and model capacity further improves both geometric fidelity and refinement outcomes. 

---
# MACS: A Hybrid Multi-Agent Framework for Reliable Conversational E-Commerce Recommendation 

**Authors**: Juli Huang, Hannah Clay, Sajjad Beygi, Thomas Sarda, Negin Golrezaei, Amin Saberi  

**Link**: [PDF](https://arxiv.org/pdf/2608.14068)  

**Abstract**: Conversational recommendation for e-commerce is increasingly mediated by large language models (LLMs), yet many real-world deployments operate under a stricter requirement: recommendations must be drawn only from a merchant's fixed catalog, without web search or unsupported product claims. In this setting, the main challenge is reliability under hard constraints: the system must satisfy user requirements, remain grounded in available inventory, and preserve preferences across multiple conversational turns. We present MACS (Multi-Agent Commerce System), a hybrid multi-agent framework for reliable conversational recommendation in fixed-catalog settings. MACS uses LLMs for language-facing tasks such as interpreting user requests, eliciting preferences, and generating responses, while correctness-critical operations, including product retrieval, hard-constraint filtering, brand exclusion, and progressive relaxation, are executed deterministically by the merchant agent. A session-persistent preference layer tracks constraints across turns, enabling consistent handling of budget overwrites and exclusion reversals. On a 140-query single-turn benchmark, MACS achieves the highest pass rate (87.1%) and perfect brand compliance (1.000). On a 10-scenario multi-turn benchmark, MACS achieves the strongest macro Pass@5 (72% vs. 56% GPT+Catalog / 52% Gemini+Catalog) with zero constraint drift. The advantage is sharpest on exclusion reversal (100% vs. 20% / 0%) and constraint accumulation (100% vs. 60% / 40%). Mean judged response quality is similar across systems (0.751 vs. 0.736). These results suggest that hybrid architectures combining deterministic constraint enforcement with session-persistent preference tracking provide stronger reliability-oriented performance than catalog-bound prompt-only baselines in the fixed-catalog merchant setting. 

---
# Rethinking Automated Program Repair: The Impact of Bug Complexity, Fault Localization, and LLM Cost-efficiency 

**Authors**: Junchi Liu, Ali Bigdeli, Roya Daneshi, Atu Ambala, Sudipto Ghosh, Fabio Santos  

**Link**: [PDF](https://arxiv.org/pdf/2608.14065)  

**Abstract**: Background: Software bugs remain a critical challenge in development, necessitating effective Automated Program Repair (APR) techniques. While Large Language Model (LLM)-based APR systems have shown promise, prior studies primarily focus on overall repair effectiveness. The effects of bug complexity, fault localization, reasoning settings, and repair cost-effectiveness remain insufficiently explored.
Aims: This study presents a comprehensive empirical analysis of LLM-based APR, focusing on how repair performance is shaped by bug complexity, fault localization, reasoning settings, and costs.
Method: We evaluate two APR techniques (ChatRepair and CodeCorrector) using three LLMs (DeepSeek, GPT, and Llama), and examine their performance across diverse levels of bug complexity and localization strategies through a multi-dimensional empirical framework and statistical analysis.
Results: Although structurally complex bugs and imprecise fault localization make repair more challenging, LLM-based APR techniques still achieve competitive repair effectiveness. Imprecise fault localization can substantially enlarge the performance gap between APR techniques. Furthermore, higher-cost LLMs and stronger reasoning settings do not consistently yield better cost-efficiency, revealing a nontrivial trade-off between repair effectiveness and computational cost.
Conclusions: Over 50% of moderately complex bugs can be repaired by low-cost LLM-based APR techniques. The repair effectiveness gap between APR techniques becomes larger as fault localization becomes less precise. GPT-5 repairs 7 and 39 more complex bugs than DeepSeek-V4-pro and DeepSeek-V3.2, respectively; whereas the total repair cost of DeepSeek-V3.2 shows the best cost-efficiency performance. 

---
# Voxel-based 3D Facies Segmentation from Seismic Data: A Comparative Study 

**Authors**: Duc-Thanh Pham, Minh-Tan Pham, Anh Nguyen, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.14058)  

**Abstract**: Seismic facies segmentation has emerged as a significant challenge in geophysics, requiring robust methods and systems to effectively identify geologically analogous facies with limited labeled data. Although existing studies have shown promising results in 2D facies segmentation, they often preprocess the original 3D seismic volumes into sets of 2D slices, typically the inline and crossline directions, and treat this problem as a purely 2D segmentation task. This simplification introduces discontinuities across slices and fails to preserve the spatial and structural continuity in 3D seismic data, thus limiting the model's ability to learn coherent geological patterns. In this work, we present a comparative and reproducible benchmark for voxel-based 3D seismic facies segmentation, built upon publicly available seismic volumes including the Netherlands F3 and the Parihaka datasets, with standardized data splits and evaluation metrics. By evaluating the three representative families of modern 3D segmentation architectures, we establish strong baseline results that highlight the potential and remaining challenges for future research in this domain. 

---
# Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use 

**Authors**: Yi Ding, Yanzhao Yu, Xili Dai, Xianbiao Qi, Peiwen Sun, Xueqian Wang, Xiangyu Yue, Jianan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14047)  

**Abstract**: This paper integrates end-to-end Visual-Language-Action (VLA) models with agentic tool-use to propose Agentic Robot with Tool-use (ART). ART is a tool-injection framework that tunes any VLA model to leverage off-the-shelf tool modules for low-level vision, high-level affordance, and embodiment enhancement. Compared to vanilla VLA models with a whole continuous action solution space, ART reduces the complexity of the action solution space through tool-use, which not only improves generalizability across different tasks but also reduces data dependency. To demonstrate the advantages (high generalizability and low data dependency) of this framework, we first built a dataset of 30K tool-use trajectories and action demonstrations, which is much smaller than those used by baseline methods. We then designed a training regimen for long-trajectory tool-use reasoning in challenging environments. Experiments show that ART achieves a 20% higher success rate than mainstream baselines on simulation and real-world tasks, such as pick-and-place in the dark at novel viewpoints. Empirical results highlight the benefits of an agent-based approach: modular tool utilization enables more efficient training, lightweight deployment, and scalable integration of new tools. This design fosters robustness, adaptability, and extensibility, paving the way for the practical deployment of VLA systems in complex real-world scenarios. 

---
# HAM-RAG: Hierarchy-Aware Multimodal RAG for Structure-Faithful Interleaved Generation 

**Authors**: Yin Li, Ziyang Hu, Zhiyu Guo, Xiangyu Liu, Wenbin Li, Boo-Ho Yang, Rav Lawana, Ziyue Li, Wei Zeng, Fugee Tsung  

**Link**: [PDF](https://arxiv.org/pdf/2608.14032)  

**Abstract**: Existing multimodal RAG methods often flatten structured documents into isolated text and image units, weakening the source organization and local text-image logic needed for faithful evidence selection and placement. We propose HAM-RAG, a Hierarchy-Aware Multimodal RAG framework for structure-faithful interleaved generation. HAM-RAG uses document hierarchy as a grounding signal across retrieval and generation, contextualizing textual and visual evidence and preserving source position and local text-image relations in the prompt. We further introduce HAM-Bench, covering Wukong, Wiki, arXiv, and Recipe across game walkthroughs, web pages, scientific papers, and step-wise recipe documents. Across multiple backbones, HAM-RAG improves the main multimodal average by 17.3% over the strongest non-hierarchical baseline. On Wukong, HAM-RAG improves Img-CBS by 24.2% over the strongest non-hierarchical baseline, demonstrating substantially better local text-image alignment. The main experiments and ablation study together demonstrate that document hierarchy is a key grounding signal for faithful image selection, placement, and local text-image alignment. These findings highlight the value of hierarchy-aware grounding for reliable multimodal assistants that generate answers faithful to the source organization, procedural structure, and local text-image evidence of structured documents, such as technical manuals, maintenance guides, and industrial SOPs. The code is available at this https URL. 

---
# AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning 

**Authors**: Zhiyue Zhao, Jingyi Wu, Hairuo Liu, Mingyu Liu, Liyang Li, Hengdi Zhang, Tong He, Zhengxue Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.14028)  

**Abstract**: Dexterous manipulation is a fundamental capability for embodied intelligence, but scaling it remains difficult because robot demonstrations are expensive to collect and action spaces vary across embodiments. Policies trained on heterogeneous data can also entangle task-relevant visual cues with embodiment-specific appearance, limiting cross-embodiment generalization. We present AdvDex, a unified Vision-Language-Action framework for learning dexterous manipulation from human and robot demonstrations. First, we introduce OmniShare, a large-scale multimodal dataset of human manipulation demonstrations that provides high-quality kinematic supervision and tactile measurements while reducing reliance on robot teleoperation. Second, we propose the Joint-Aligned Action Space (JAAS), a canonical action representation comprising an $\mathrm{SE}(3)$ wrist pose and 15 finger joints, thereby functionally aligning human hands, dexterous robot hands, and parallel grippers. Finally, we use domain-adversarial learning to reduce embodiment-specific information in the learned visual representation. Experiments on hand-action prediction and real-world dexterous manipulation show consistent improvements over baselines, effective zero-shot human-to-robot skill transfer, generalization to unseen objects and environments, and data-efficient few-shot adaptation. 

---
# ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models 

**Authors**: Xinye Li, Lingshuai Lin, Lei Wang, Liuzhou Zhang, Jialin Cui, Qingshan Li, Guanchu Wang, Qingbin Liu, Xi Chen, Jiang Bian, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2608.14022)  

**Abstract**: Action-conditioned video world models require low-latency causal generation and reliable responses to game-native controls. Although causal distillation enables one- or few-step video synthesis, extending it to interactive world models remains challenging, as discrete keyboard states and continuous mouse motion must remain aligned with temporally compressed latent chunks during causal training and autoregressive rollout. We introduce ForgeWM, a progressive framework that transforms a bidirectional action-conditioned video generator into efficient few-step world models through domain adaptation, teacher-forced causal training, causal consistency distillation, and on-policy distribution matching with a bidirectional teacher. The resulting budget-specialized students operate at steady-state denoising budgets of 1, 2, and 4 steps. ForgeWM further supports a dual-path deployment protocol combining latency-critical interaction with optional replay-time refinement, where the one-step student re-noises and refines its saved draft. On paired Minecraft trajectories, ForgeWM leads the evaluated systems in Imaging Quality, reference-aligned motion-profile agreement, action-sign accuracy, and mouse-control accuracy, while achieving the lowest reference LPIPS; the same four-stage recipe transfers to gamepad-controlled FPS gameplay. Replay-time refinement matches four-step reference quality while remaining roughly three times closer to the experienced trajectory than regeneration from noise. These results demonstrate ForgeWM's effectiveness for controllable few-step video generation. 

---
# Content Based Video Narration of Gameplay with Vision Language Models 

**Authors**: Mathew Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2608.14016)  

**Abstract**: Live game commentary is scarce: it exists for professional esports broadcasts and almost nowhere else. We present a content-based video narration system that produces spoken, esports-style commentary for arbitrary gameplay recordings using a general-purpose vision-language model (VLM) and a text-to-speech back end, with no game-specific instrumentation, no engine telemetry, and no task-specific training. Three mechanisms carry the system. Temporal mosaic packing arranges nine uniformly sampled frames into a single 3x3 image, letting an image-native VLM reason about motion while consuming one image payload per segment instead of nine. Context-conditioned prompting replays the K most recent narrations as assistant-role history, suppressing the repetition that dominates per-segment captioning of static scenes. Duration-conditioned generation and elastic alignment constrain narration length in the prompt, then time-scale or symmetrically pad the synthesized audio so each utterance fills its segment slot exactly, giving frame-accurate muxing without a forced aligner. The implementation supports either cloud TTS or a 6-bit quantized 4B-parameter on-device TTS model on Apple silicon, making the speech stage fully local. We report a qualitative case study on real-time strategy footage, a cost model showing the mosaic reduces per-minute image payloads by 9x, and a candid account of observed failure modes - hallucinated game state, resolution loss from mosaicking, and prosody artifacts from time-scaling. We release the system as a reproducible baseline, with an evaluation protocol for the quantitative study a full version will report. 

---
# MedClaw: Heuristic Agent Harness for Long-Horizon Surgical Video Reasoning 

**Authors**: Yingying Fan, Penghui Du, Leyan Zhu, Runze He, Zimeng Wu, Yuxuan Zhang, Liang Chen, Jiahao Xie, Jiangtang Wang, Shuai Shao, Anchao Yang, Yutong Bai, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.14015)  

**Abstract**: Understanding tens-of-minutes surgical videos requires long-horizon temporal reasoning, answering what happens before, after, or across stages of a procedure by grounding the question in visual evidence spread across time. Existing approaches handle this poorly: a one-shot vision-language model (VLM) compresses the whole procedure to fit its context window and loses the detail a "before" or "after" question depends on, while video agents that train the model where to look are data-hungry and transfer poorly to out-of-domain surgery. We build an agent harness that separates reasoning from perception and improves by evolving context rather than optimizing weights. A text-only orchestrator plans which evidence to gather and issues an auditable sequence of tool calls, while frozen vision-language sub-agents execute each call over the pixels, viewing, cropping, inspecting frames, and retrieving external knowledge. We further propose a gradient-free, reward-gated Heuristic Skill Distillation loop that mines the agent's own low-scoring traces and keeps a candidate skill only when it raises a validation reward, yielding reusable retrieval skills, notably directed re-look. Growing an external skill library rather than tuning weights, the loop adapts from only about 100 labeled examples, far fewer than supervised or reinforcement fine-tuning requires. To evaluate this agent, we introduce MedClawBench, a de-leaked, doctor-grounded benchmark of 1,123 questions over self-built long neurosurgery recordings and a held-out public lecture-video test split. Across both datasets and all four evaluation dimensions, our agent consistently outperforms one-shot VLMs and general video-agent frameworks, with the largest gains on the long, out-of-domain neurosurgery videos. Project page: this https URL. 

---
# EchoRec: Multi-Item Prediction-Empowered Generative Recommendation via Cycle-Consistent Preference Alignment 

**Authors**: Haokai Ma, Aoqi Hu, Yueao Xing, Ruobing Xie, Yonghui Yang, Teng Tu, Lei Meng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2608.14011)  

**Abstract**: Generative recommendation autoregressively generates the semantic IDs of the target item, unifying preference modeling and index retrieval within the shared token space. Recent attempts have introduced Multi-Token Prediction (MTP) into this field, yet they primarily inherit its efficiency merit, leaving its potential as dense supervision unexplored. Unlocking this potential hinges on whether future behaviors qualify as informative supervision. Our analysis reveals that future behaviors carry a semantic echo of the current one far above that of random pairs, which nevertheless decays along horizons under intent transitions, making them informative yet order-dependent signals. Motivated by this, we propose EchoRec, which empowers MTP with cycle-consistent holistic preference alignment across multi-horizon for generative recommendation. It comprises two synergistic modules. Horizon-aware Preference Generation (HPG) sequentially chains lightweight auxiliary branches upon the base recommender, where each branch conditions on its predecessor to respect preference evolution. Verifiable Holistic-Preference Alignment (VHA) further consolidates them into the holistic preference and echoes it back through cycle-consistent projectors to suppress spurious alignment, with theoretical guarantees that exclude the rank-collapse form of spurious alignment under an invertible transport, enabling the holistic preference to be retained in the decoding representation. All auxiliary components serve as disposable scaffolding discarded at inference, introducing negligible online serving overhead. Extensive experiments on three datasets demonstrate the superiority of our EchoRec, together with its naturally acquired multi-item generation ability. Our code and datasets will be available upon acceptance. 

---
# Musical Mirrors: The LLM as Sounding Board in Songwriting 

**Authors**: Xiao Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.13944)  

**Abstract**: This paper examines a use of AI in creative practice as an interpretive sounding board for human-generated material, rather than the more familiar pattern of AI generation followed by human curation. Through the lens of resonance as theorized by Hartmut Rosa, I present a first-person case study of songwriting from July 2025 to March 2026, drawing on 16 original pieces in English, French, and other languages along with piano solos. I describe a configuration in which resonance is not located between user and model, but in the author's deepening contact with their own material, mediated through the model. This kind of resonance was supported rather than inhibited by AI when sounding-board behavior was cultivated through sustained calibration by the user. Two failure modes appeared when calibration was absent: sycophantic drift and magical overinterpretation. This account suggests both the potential and the risks of AI as an interpretive partner in creative practice. 

---
# CMCNet: Aligning Ultrasound Image Embeddings with Textual TI-RADS Representations for Fine-Grained Thyroid Classification 

**Authors**: Bingxin Yu, Xueli Wang, Jerry Zhou, Wenyan Wang, Li Wen, Lan Huang, Xin Feng, Fengfeng Zhou, Kewei Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13939)  

**Abstract**: Ultrasound is the primary imaging modality for assessing thyroid nodules, and the ACR TI-RADS framework standardizes diagnosis through five ultrasound feature categories that are aggregated into five risk levels (TR1-TR5). Although widely adopted in clinical practice, most deep learning approaches focus on binary malignancy classification, while multi-class prediction and explicit utilization of feature-level supervision remain underexplored, largely due to limited annotated data. In this study, we introduce the STN dataset of 600 thyroid nodules with paired transverse and longitudinal ultrasound images, bounding box annotations, and complete labels for all five TI-RADS feature categories. Following the clinical decision process, we investigate how structured feature information can guide representation learning during training while requiring only images at inference. We demonstrate that text embeddings derived from standardized feature descriptions form a stable surrogate representation for TI-RADS risk levels. Based on this observation, we propose CMCNet, which aligns image embeddings to fixed textual embeddings via a Center-Margin Contrastive Loss that simultaneously promotes intra-class compactness and inter-class separation. Experimental results show that this embedding alignment strategy is more data-efficient and robust than direct multitask learning, and consistently outperforms InfoNCE, center loss, a strong multitask baseline, and a VQA-style multimodal model, particularly in imbalanced settings. The dataset is freely available at doi: https://doi.org/10.5281/zenodo.19125693 and the source code is available at: this https URL. 

---
# CForce: Boosting Parallel Decoding for dLLMs via Consistency Forcing 

**Authors**: Yuji Ren, Chenkai Xu, Zhuocheng Gong, Jianguo Li, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.13925)  

**Abstract**: Diffusion large language models (dLLMs) accelerate language generation by predicting multiple masks in a single forward pass. However, existing dLLMs can suffer from unreliable predictions in early denoising stages under aggressive parallelism strategies, leading to errors that can propagate to later stages. To tackle this issue, we present Consistency Forcing (CForce) for dLLMs, a distillation method to force the mask predictions of early stages to align with those of later stages. CForce trains the model on pre-collected self-rollout trajectories, thereby improving training-inference alignment. We introduce Confidence Adaptive KL Divergence as a distillation objective to conjoin the merits of forward and reverse KL. We further provide a theoretical analysis for the consistency objective to explain why CForce can approximately minimize the prediction error of early stages. Critically, the same formulation applies to both mask-to-token decoding and edit-capable decoding; in the edit-capable case, later token-to-token refinements provide additional supervision for earlier masked-state predictions. Experiments on non-edit and edit-capable LLaDA models show improved speed-quality trade-offs, especially under high-parallelism decoding budgets. Code is available at: this https URL. 

---
# Hybrid Quantum-inspired Kolmogorov-Arnold Networks for Privacy-Aware Federated Biosignal Learning 

**Authors**: Chun-Hua Lin, Samuel Yen-Chi Chen, Yu-Chao Hsu, Kuo-Chung Peng, Jiun-Cheng Jiang, Chi-Sheng Chen, Tai-Yue Li, Nan-Yow Chen, En-Jui Kuo, Hsi-Sheng Goan  

**Link**: [PDF](https://arxiv.org/pdf/2608.13914)  

**Abstract**: Electrocardiogram (ECG) recordings are sensitive biomedical data, limiting the ability of hospitals and wearable devices to share raw signals for centralized model training. Federated learning addresses this practical privacy constraint by enabling collaborative model training while keeping raw biosignal data at their respective sources. However, federated ECG classification remains challenging due to limited client-side samples, imbalanced arrhythmia labels, and non-independent and identically distributed (non-IID) data across clients. These constraints require classifiers that are both communication-efficient and robust to cross-client distribution shifts. In this work, we evaluate a hybrid quantum-inspired Kolmogorov-Arnold network (HQKAN) against a multilayer perceptron (MLP) for five-class arrhythmia classification on the MIT-BIH dataset and three-class classification on the INCART dataset under federated averaging (FedAvg). Across multiple client configurations, HQKAN improves most aggregate and minority-class metrics while using 37.35% fewer trainable parameters and reducing communication cost by 24.89% on MIT-BIH; on INCART, it achieves corresponding reductions of 44.81% and 36.41%. These results indicate that HQKAN offers a compact, communication-efficient and robust alternative to the MLP baseline for privacy-aware federated learning on biosignal data. 

---
# CipherSight: Robust Website Fingerprinting via Record-Resource Semantic Supervision under Distribution Shifts 

**Authors**: Runhan Song, Qiqi Liu, Chuanzhou Pan, Zhenquan Ding, Youquan Xian, Chongru Fan, Lei Cui, Wei Wang, Zhiyu Hao  

**Link**: [PDF](https://arxiv.org/pdf/2608.13905)  

**Abstract**: HTTPS website fingerprinting (WF) aims to identify visited websites from metadata observable in encrypted traffic. However, real-world deployments introduce a significant out-of-distribution (OOD) problem caused by temporal and geographic changes, while previously unseen websites are common in open-world scenarios. Existing methods primarily learn from raw TCP packet sequences and struggle to capture stable and generalizable website representations, resulting in performance degradation under practical conditions.
We propose CipherSight, a TLS-record-based hierarchical framework for robust HTTPS WF. Unlike existing approaches that rely on TCP packet sequences and are sensitive to transport-layer artifacts, CipherSight learns website representations from TLS records by jointly encoding multiple record-level attributes. It introduces a hierarchical architecture that captures both intra-flow dependencies among TLS records and inter-flow interactions across concurrent flows, enabling the model to exploit structural patterns in HTTPS traffic. Besides, to learn robust representations, CipherSight employs a masked record modeling (MRM) task to capture contextual traffic semantics and leverages fine-grained record-resource annotations as privileged supervision through structure-aware objectives and semantic distillation. Experiments show that CipherSight achieves 95.41% accuracy across more than 2,000 website classes in the closed-world setting and maintains over 90% accuracy under both temporal and geographic drift, consistently outperforming all evaluated baselines. 

---
# Agentic Transaction: Towards ACID-Compliant Agent Systems 

**Authors**: Zhaoyan Sun, Xiaoxiao Wang, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13900)  

**Abstract**: Large language model (LLM) agents are evolving from conversational assistants into autonomous systems that execute long-horizon tasks through reasoning, tool use, code generation, and workspace manipulation. As agents increasingly operate over persistent environments and multi-step workflows, they face challenges analogous to those addressed by transactional database systems: reliable execution, consistent outcomes, safe concurrency, and durable state management. We introduce the concept of an agentic transaction and propose an ACID-compliant agent system framework that reinterprets the classical ACID properties for agent execution through four semantic guarantees: Semantic Atomicity, Semantic Consistency, Semantic Isolation, and Semantic Durability. Together, these properties provide a principled foundation for building reliable agent systems despite model uncertainty and dynamic execution environments. To instantiate this framework, we develop an ACID-compliant data agent that realizes these guarantees through transactional exploration-execution-validation cycles, transactional skill hubs, confidence divergence-based validation, semantic dependency-aware isolation, and transaction-aware semantic state management. Experimental results on widely used benchmarks show that our system achieves a 10.6% improvement over state-of-the-art agents, including Claude Code. This work opens a broader research agenda on extending transactional principles and system architectures toward building trustworthy, scalable, and self-evolving AI agent systems. 

---
# Engineering Signals of Human-AI Collaboration in the Agentic Coding Era: A Longitudinal Analysis of 33,228 Pull Requests from vLLM and SGLang with Implications for Biomedical AI Agents and Bioinformatics Pipeline Developmen 

**Authors**: Jiada Li, Xuesong Ye, Olamide Olowoniyi  

**Link**: [PDF](https://arxiv.org/pdf/2608.13884)  

**Abstract**: The rapid adoption of AI coding assistants and autonomous agentic development systems has coincided with major changes in the pace and structure of open-source software engineering. Yet empirical longitudinal evidence of these changes at the team level remains limited. We present a descriptive longitudinal analysis of seven engineering metrics: pull request (PR) throughput, cycle time, contributor diversity, PR comment density, merge rate, new-author participation, and PR size. Metrics were computed from all merged PRs in two high-velocity AI infrastructure repositories, vLLM (February 2023-June 2026; 18,290 PRs) and SGLang (January 2024-June 2026; 14,938 PRs). We segment development into four eras aligned with major changes in AI-assisted software development and examine human- and bot-authored activities. Both projects show substantial increases in development velocity and AI-developer collaboration signals. PR throughput increased 21x in vLLM and 17.9x in SGLang, while bot-authored PRs accounted for less than 0.2% of this growth, indicating that the increase was overwhelmingly human-driven. In the latest era, median cycle time was 1.04 days for vLLM and 0.62 days for SGLang, while P90 cycle times reached 16.8 and 14.3 days, respectively. Monthly unique authors increased steadily in both projects, suggesting broader contributor participation. PR comment density increased 4.2x in vLLM and 3.8x in SGLang, with bot comments contributing an estimated 15-20% of the increase. In contrast, PR size remained relatively stable across eras. Overall, AI-assisted development is associated with higher throughput, broader contributor participation, and increased AI-developer collaboration signals in high-velocity open-source software development. 

---
# Engineering Reliable Coding Agents: Evaluating and Operating the System Around the Model 

**Authors**: Stephanie Jarmak  

**Link**: [PDF](https://arxiv.org/pdf/2608.13867)  

**Abstract**: AI coding agents are commonly evaluated as models but deployed as systems. Their reliability depends not only on model capability, but on the harness, execution state, retrieval, memory and state management, permissions, review interfaces, and resource allocation. This monograph examines those boundaries and develops a framework for evaluating and operating coding agents reliably. It synthesizes 164 scholarly works, 100 practitioner records, 29 benchmark records, and 17 author-system case records through a structured multivocal review, targeted update audits, software-engineering coverage analysis, and distributed-systems evidence synthesis. Across this evidence, many apparent model failures originate elsewhere in the system, while improvements at one layer often fail to propagate to end-to-end outcomes. Evaluation and operation are treated as a dependency chain in which weaknesses in task construction, execution environments, retrieval, state management, verification, or observability can invalidate downstream conclusions. The monograph contributes a versioned catalog of 206 reliability records: 193 gated practices, including 56 developed in depth, plus 13 research leads; an evidence ledger; a framework for dependency and repair asymmetry across the agent lifecycle; measurements and failure cases from operated agent systems; runnable evaluation and reliability protocols; and five reusable agent skills with evidence maps. Together, these provide a system-level methodology for distinguishing model capability from infrastructure effects, designing defensible evaluations, and building systems that recover safely when components fail. The review is structured rather than exhaustive, evidence strength varies by topic, and results depend on workload and configuration. The methods record which search lanes were executed, which remain unexecuted, and limits on evidence-grading claims. 

---
# Federated Prompt Learning: A Unified Framework, Empirical Analysis, and Future Directions 

**Authors**: Qinglin Yang, Chen Qiu, Hongyuan Zhang, Pengdeng Li, Yuan Liu, Zhihong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2608.13844)  

**Abstract**: Large language models (LLMs) have become core components of cloud-based intelligent services in academia and industry, yet their training and deployment are hindered by high computational costs, data centralization, and privacy concerns. Federated learning (FL) offers a decentralized training paradigm that enables clients to collaboratively train a learning model without sharing raw data, making it a promising solution for privacy-preserving LLM training and reasoning. This paper presents a comprehensive survey of federated prompt learning (FPL) to review recent advances in integrating the federated learning paradigm and large language models, answering the following research questions: RQ1: The fundamental motivations, characteristics, and enabling technologies of FPL, and how it differs from conventional FL and full-model federated fine-tuning; RQ2: The trade-offs FPL approaches exhibit in performance, communication efficiency, computational overhead, scalability, personalization, and heterogeneity handling; RQ3: The remaining security, privacy, robustness, and system challenges, along with key future research directions. To this end, we systematically examine existing FPL methods across the full model lifecycle: pre-training, fine-tuning, and practical applications, while discussing security, privacy, and robustness issues and summarizing existing defense mechanisms. Finally, we highlight open challenges and future directions, aiming to help readers understand how the insights drive research in FPL. 

---
# ASSERT: A Measurement Pipeline for GenAI Audits 

**Authors**: Riccardo Fogliato, Abhinav Palia, Xiawei Wang, Emily Sheng, Chad Atalla, Jean Garcia-Gathright, Nicholas Pangakis, Sharman Tan, Dan Vann, Hannah Washington, P. Alex Dow, Heba Elfardy, Hanna Wallach, Sandeep Atluri  

**Link**: [PDF](https://arxiv.org/pdf/2608.13840)  

**Abstract**: Audits of generative AI (GenAI) systems often summarize behavior as a reported rate: how often the audited system complies with policy. Researchers and stakeholders use that rate to compare systems, track regressions, and gate deployment. A reported rate reflects both the system under audit and the measurement choices behind it, so a change in the rate can leave it unclear whether the system or those choices moved. We introduce ASSERT, a specification-driven measurement pipeline for GenAI audits that ties each reported rate to a written specification of the measurement choices used to produce it. ASSERT helps draft a behavioral rubric and test cases, then runs the audit against a GenAI system and returns a reported rate. In a case study on conversational deception, we observe that the reported rate moves substantially with the dialogue setup, the simulated user, the judge, and the evidence bar for non-compliance. These measurement choices substantially change the reported rate and can reorder GenAI system rankings. Because each reported rate is tied to an explicit specification, differences across audits are easier to attribute and interpret. 

---
# AdsWorldEngine: A Self-Evolving Conversational Advertising Agent through Orchestrator and Tool Coevolution 

**Authors**: Simiao Zuo, Chenhui Xu, Yimeng Jia, Qiang Lou, Jian Jiao, Denis Charles  

**Link**: [PDF](https://arxiv.org/pdf/2608.13833)  

**Abstract**: Conversational advertising aims to deliver useful ads within multi-turn assistant interactions. Unlike conventional query-based advertising, where the user's intent is often expressed in a short standalone query, conversational ads must infer latent commercial intent from the current user query, the assistant response, and dialogue history while also deciding whether an ad would be helpful rather than intrusive. We propose AdsWorldEngine, an agentic framework for conversational advertising. AdsWorldEngine uses an Opportunity Gate to determine whether ads should be shown, an Orchestrator to generate commercial intents, call advertising tools, and construct a top-3 ad slate, and an Evaluator to score delivered ads for offline optimization. The central contribution is an iterative actor-tool training procedure: we first train the Orchestrator with supervised fine-tuning and agentic reinforcement learning, then use high- and low-reward rollouts to construct preference data to train tools. This creates a self-improving loop in which the system learns not only how to use advertising tools, but also how to improve them from rewarded behavior. To support subjective production decisions, we introduce label grounded judgment modeling, which trains judgment models from human labels collected under explicit guidelines. It enriches labels with thinking traces, filters inconsistent rationales through reflection, and further optimizes binary judgments with a cost sensitive GRPO variant that preserves asymmetric reward gaps. Offline, AdsWorldEngine improves diversity by 60% and relevance by 80% over the current production ad delivery system. In an online A/B test, it increases RPM by 22% and ads coverage by 74%. 

---
# Optimal Power Allocation and AI Receiver Design for Superimposed DMRS and Data Transmission 

**Authors**: Sha Hu, Zhongwang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13809)  

**Abstract**: In this paper, we consider transmissions with superimposed (SI) demodulation-reference-symbol (DMRS) and data in orthogonal frequency-division multiplexing (OFDM) based multiple-input multiple-output (MIMO) systems. First, we derive an analytical framework to characterize the iterative behavior between the mean-square errors (MSEs) of channel estimation (CE) and MIMO detection (MD) within an iterative CE and detection (ICED) process. This framework is subsequently utilized to optimize power allocation and pilot patterns between the DMRS and data symbols for SI-DMRS transmission. Second, we design an artificial intelligence (AI) based receiver built upon Transformer encoders for SI-DMRS transmissions, which incorporates an iterative CE and detection (ICED) structure. Simulation results demonstrate that the proposed AI-ICED receiver, combined with SI-DMRS, effectively increases spectral efficiency (SE) compared to conventional systems using non-overlapped DMRS and data symbols. 

---
# PPAPlace: Differentiable Cross-Stage Objectives for Chip Placement Optimization 

**Authors**: Ruogu Chen, Jie Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.13790)  

**Abstract**: Macro placement significantly affects a chip's post-route performance, power, and area (PPA). Most placement methods optimize half-perimeter wirelength (HPWL) as the primary objective. However, recent benchmarking shows a near-zero correlation between HPWL and post-route timing metrics such as the worst negative slack (WNS) and total negative slack (TNS). As a result, all six evaluated artificial intelligence (AI) placers degraded PPA relative to the hierarchical baseline. Recent efforts have tried to train cross-stage predictors to close this gap. However, existing methods focus on macro-only representations and use pre-route metrics as training labels. A label fidelity study of ten circuits at four design flow stages reveals that HPWL and pre-route timing poorly reflect final post-route timing rankings. In contrast, post-global-routing achieves the best balance between final timing fidelity and label generation cost-effectiveness. Based on this finding, PPAPlace is a timing-driven differentiable surrogate predicting post-route PPA from macro and standard-cell placements. The surrogate is a dual-stream predictor that combines graph attention over the chip netlist with spatial convolution over the placement grid. It is trained on post-global-routing labels. The predicted WNS and TNS gradients flow end-to-end back to cell coordinates. PPAPlace exploits these gradients in two ways: as a co-objective injected into an analytical placer's optimization loop (PPAPlace-CoOpt), and as a post-placement refinement step that adjusts macro positions via projected gradient descent (PPAPlace-Refine). On five ChiPBench test circuits excluded from training, PPAPlace improves average WNS and TNS by 22\% and 51\% over the hierarchical baseline while preserving power and routability, using the same predictor without test-circuit retraining. Code is available at this https URL. 

---
# Do AI chatbots find what experts would? Effects of model, user role, and sample size on study retrieval for medical questions 

**Authors**: Qingfang Liu, Qiao Jin, Joe D. Menke, Thorsten Kahnt, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13786)  

**Abstract**: Large language model (LLM) chatbots are increasingly used to answer clinical questions with citations to relevant clinical studies. Prior research has largely focused on citation fabrication, leaving a gap in evaluating the quality of retrieved studies and the factors driving their selection. In this study, we evaluated three general-purpose LLM chatbots: Claude Sonnet 5, Gemini 3.1 Pro, and ChatGPT GPT-5.5. We prompted the models with clinical questions adapted from 20 review questions in Issues 6 and 7 of the 2026 Cochrane Database of Systematic Reviews, simulating patient, clinician, and evidence-synthesis researcher roles. Each chatbot was queried under each user role with four independent repetitions, yielding 720 responses. Each chatbot was asked to support its answers with primary clinical citations, which we benchmarked against the included and excluded study sets of the Cochrane reviews. On average, a chatbot response retrieved 39.2% $\pm$ 29.8% of Cochrane included studies, while citing 5.0% $\pm$ 9.4% of excluded studies. Recall of Cochrane included studies varied significantly by model and user role. ChatGPT achieved higher recall than Claude or Gemini (63.1% $\pm$ 29.5% vs. 37.0% $\pm$ 23.8% vs. 17.3% $\pm$ 13.1%; $p=2.0\times10^{-5}$). The researcher role yielded higher recall than the clinician or patient roles (42.8% $\pm$ 30.8% vs. 38.6% $\pm$ 28.9% vs. 36.1% $\pm$ 29.3%; $p=2.0\times10^{-5}$). Controlling for publication year, citations per year, and open-access status, sample size was the only independently significant predictor of retrieval (odds ratio 1.80 per 1-unit increase in log sample size, 95% CI 1.37-2.36, $p=2.34\times10^{-5}$). These findings suggest that while LLM chatbots can retrieve some studies identified by expert reviewers, their performance varies by model and user role, and they exhibit a bias toward clinical trials with larger sample sizes. 

---
# CutClean: Neural Network Pruning for Privacy-Preserving Inference 

**Authors**: Leonardo Magliolo, Vito Paolo Pastore, Giuseppe Valenzise, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2608.13773)  

**Abstract**: Neural networks are increasingly deployed in high-stakes applications with growing privacy leakage concerns. We show that this privacy leakage can occur even in the absence of representation imbalances that lead to traditional dataset biases. This poses significant privacy risks when deploying models that process sensitive attributes. In this context, we propose CutClean, a privacy-aware pruning method that allows to reduce privacy information flow through the network, while increasing its sparsity. Our approach employs auxiliary linear privacy heads placed at each network's block to quantify information leakage, and further applies increasing levels of sparsity to remove the private attribute leakage, measured in terms of the accuracy of the privacy head attached to the last block. Experiments on synthetic and real-world datasets demonstrate that our approach effectively minimizes private information flow while achieving high sparsity rates and preserving classification target accuracy. 

---
# Amplified Does Not Mean Predictive: Reasoning Behaviors in Thinking Models 

**Authors**: Jean de Dieu Nyandwi, Leena Mathur, Yonatan Bisk, Robert Hawkins, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2608.13760)  

**Abstract**: Which reasoning behaviors are associated with correct answers in reasoning models, and does reasoning-oriented training amplify those behaviors? This distinction is important because reasoning-oriented training can make traces look more deliberative without amplifying the behaviors most tied to model correctness. We quantify this mismatch with Behavioral Lift, a metric that measures how much correctness changes when a behavior is present versus absent in a model's reasoning trace. Across 15 models and 6 benchmarks spanning text-only and vision-language reasoning, we annotate 15,282 traces with a taxonomy whose core behaviors are defined for both LLM and VLM traces. We find evidence for an Amplification-Lift Gap, in which thinking models strongly amplify self-correction, hypothesis testing, and uncertainty acknowledgment, while the highest-lift behaviors are confidence calibration, knowledge alignment, and self-awareness. Confidence calibration is among the strongest positive signals of correctness in both modalities, yet is barely amplified; uncertainty acknowledgment is amplified by 3--7$\times$, yet is weakly or negatively associated with correctness. We find that reasoning-oriented training does not preferentially amplify the highest-Lift behaviors, motivating process-level objectives that reward calibrated and grounded reasoning rather than surface form alone. 

---
# Data-driven techniques for translational neuroscience and personalized neuro-health 

**Authors**: Vishal Subedi, Shashipraba N. K. Rajakaruna, Pratyusha Sarkar, Subhankar Chattoraj, Anjali Khasa, Siddhartha Nandy, Hamza Farooq, Animikh Biswas, Sanjay Chaudhuri, Asim K. Dey, Karuna Joshi, Christophe Lenglet, Ansu Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.13749)  

**Abstract**: Neurodegenexrative diseases such as Alzheimer's disease and Parkinson's disease are diagnosed most reliably only after substantial, often irreversible, neuronal loss has already occurred, creating an urgent need for quantitative tools that can detect subtle, early, and individual-specific brain changes from neuroimaging data. This review surveys a broad and rapidly evolving toolkit of data-driven techniques for translational neuroscience and personalized neuro-health, organized around four complementary methodological pillars. Throughout, we emphasize how these methodologically diverse approaches converge on a common translational goal: personalized, mechanistically grounded, and clinically actionable models of individual brain health, and we close by discussing the principal open statistical, computational, and clinical challenges that remain. 

---
# Does ISO-Grounded NFR Specification Improve LLM Code Generation? A Comparison of Rich and Structured Interventions against a Natural-Language Baseline 

**Authors**: Joào Pedro Monteiro Pereira, Vinicius Cardoso Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2608.13742)  

**Abstract**: In LLM-based code generation, Non-Functional Requirements (NFRs) are often specified as terse one-line phrases. We ask whether grounding those specifications in ISO/IEC 25010 Quality Model, either as rich natural-language prose (NL-rich) or as structured JSON (Structured), improves code generated on HumanEval/HumanEval-ET compared to a RobuNFR-style one-line baseline (NL-simple). We evaluate four NFRs (performance, error handling, code smell, readability) with ten prompt variations per condition under a fixed model snapshot and paired non-parametric analysis. Primary finding: ISO-grounded enrichment improves static quality proxies (unreadability density falls across all four NFRs (e.g., Performance 0.88 -> 0.69 for NL-rich)) and reduces sensitivity to prompt wording, but does not reliably improve functional correctness; for error handling, extended-test pass rate decreases, suggesting tension between defensive coding patterns and exact-output benchmarks. Secondary finding: when ISO content is held constant, NL-rich and Structured differ negligibly in correctness (|delta| <= 0.023), indicating that semantic content matters more than JSON-vs-prose format. Practitioners should invest in standard-grounded NFR content rather than serialization form. A fully traceable replication package is provided. 

---
# Building AI-Intensive Software with AI: Early Results and a Cautionary Tale on Measuring Development Cost 

**Authors**: Victor Barros de Miranda Neves, Kiev Santos da Gama, Vinicius Cardoso Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2608.13730)  

**Abstract**: Empirical reports on the true cost of AI-intensive software development remain scarce, and the few that exist are easy to get wrong in ways that never surface in the final number. We report early results from an ongoing case study: a six-person student team built a full conversational onboarding assistant -- RAG-based code chat, guided tours, dependency graphs, technical-debt analysis -- over one academic term using pervasive AI assistance. We instrumented development with a three-layer cost model (real AI spend, self-reported human effort, human counterfactual) and initially reported a 19.4x cost ratio. A follow-up pass revealed two independent errors -- inferring per-token cost under a flat-rate subscription, and pricing the counterfactual with the wrong regional labor rates -- that together had inflated the ratio by roughly 2x; the corrected figure is ~9.9x. We present this correction as an early, generalizable finding in its own right: both errors are easy to make, invisible in the final number, and plausibly common in similar reports. We outline next steps toward a more robust, replicable costing methodology for AI-intensive development. 

---
# Capacity-Dependent Effects of Data Selection for Reasoning 

**Authors**: Cuong Dang, Hoang Anh Just, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.13721)  

**Abstract**: In reasoning supervised fine-tuning, candidate responses for the same instruction can differ substantially in how well they match the student's current distribution. Recent likelihood-based response selection methods suggest that responses closer to the student distribution provide more effective supervision, motivating the hypothesis that high-likelihood responses may generally be preferable for fine-tuning. In this paper, we revisit this intuition and show that the value of likelihood-based data selection depends critically on model capacity and training duration. Through controlled experiments on mathematical reasoning, using students ranging from 1.5B to 8B parameters and supervision generated by stronger teacher models, we observe a clear \emph{capacity-dependent} ``{\color{SMALLCOLOR}\textbf{Fast-Fit}} / {\color{LARGECOLOR}\textbf{Slow-Gain}}'' pattern. High-likelihood data provides faster and more stable early improvements, especially for smaller models, but low-likelihood data becomes increasingly beneficial for larger models when training is allowed to continue longer. To explain this phenomenon, we analyze learning dynamics, showing that small models often fail to absorb low-likelihood supervision and instead fall into shallow or repetitive behaviors, while larger models are better able to move toward the teacher distribution under such data. We further provide a capacity-constrained theoretical view of distillation that clarifies how data difficulty, data span, and student capacity jointly govern transfer. Overall, our findings show that effective data selection for reasoning should be aware of model capacity and computing budget rather than based on a single universal preference for high-likelihood supervision. 

---
# Reading Between The Lines: Modeling and Evaluating Behavioral Realism in Legal Simulation 

**Authors**: Divya Vetticaden, Arya Gupta, Julian Nyarko, Megan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.13712)  

**Abstract**: Deposition training requires attorneys to manage dynamic witness behavior, yet legal-AI evaluations largely focus on factual accuracy, reasoning, or response-level plausibility. We introduce WitnessSim, a deposition simulator driven by controllable legal personas. We use an evaluation framework separating behavioral realism from pedagogical usefulness. We assess realism through adversarial testing, blinded attorney comparison, and analysis of longitudinal behavioral trajectories. WitnessSim generally maintained plausible behavioral boundaries, and attorneys did not systematically prefer either original testimony or WitnessSim generated testimony. Pedagogical tests showed that witness behavior changed meaningfully in response to question form and attorney intervention without uniformly collapsing the assigned persona. Together, these results showcase a model of behavioral fidelity in legal simulations, and provide a framework for evaluating its performance. 

---
# TeachMateGPT: A Multi-Agent Knowledge-Grounded Framework for Pedagogical Assessment Generation from Science Curriculum Materials 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, M. F. Mridha, Jubayer Al Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2608.13708)  

**Abstract**: Automatically generating textbook-grounded assessment items can reduce science teachers' workload, but existing retrieval-augmented generation (RAG) systems rely on flat retrieval, support only single-question generation, lack safeguards against weak evidence, and are ill-suited to low-resource, board-exam-structured curricula. We address these limitations with TeachMateGPT, a multi-agent system contributing four advances to curriculum-grounded science-assessment authoring. (i) COPE, a hierarchical knowledge base replacing token-window chunking with a multi-resolution index that segments documents along syllabus structure and links them at three granularities via a traversable graph-based lineage, matching evidence to each topic's instructional level. (ii) A staged, fail-closed agent pipeline replacing one-shot retrieve-then-generate: routing gates search, retrieval fuses dense and lexical evidence under a coverage gate that withholds generation on insufficient evidence, and specialist agents draft objective and constructed-response items. (iii) SAVER, a source-attributed verification protocol scoring faithfulness, relevance, and hallucination risk against retrieved evidence, applying stricter grounding checks across each creative question's four sub-parts, paired with teacher-in-the-loop evaluation rather than automatic filtering. (iv) NCTB-SciGen8, a curriculum-grounded dataset of 198 items (143 multiple-choice, 55 creative questions) spanning all 14 chapters of the NCTB Class 8 science textbook, produced by the pipeline and rated by three practicing teachers. TeachMateGPT raises faithfulness (0.68 $\rightarrow$ 0.96) and answer relevancy (0.60 $\rightarrow$ 0.89) over a vanilla RAG baseline. 

---
# CLAIR-Fin: An Adversarial Multi-Agent Framework for Claim-Level Verification and Adaptive Debate in Cross-Modal Financial QA 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Jubayer Al Mahmud, M. F. Mridha, Md. Alam Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.13706)  

**Abstract**: Existing defenses against hallucination in retrieval-augmented and multi-agent pipelines remain partial: evidence is trusted despite modality disagreement, debate verifies an aggregate report rather than individual claims, and such verification occurs only after drafting, leaving inter-agent errors undetected until the final text. To close this gap, we present CLAIR-Fin, a nine-agent framework that decomposes each question into atomic claims maintained in a typed Financial Claim Ledger. Each claim is resolved through Asymmetric Evidence Authority, which conditions evidence trust on claim type rather than treating all modalities as equally reliable; Chain-of-Custody Verification, which checks grounding at the hand-off between drafting and adversarial review rather than only at the pipeline's exit; an Adaptive Rebuttal Cycle, which routes contested claims through adversarial debate whose depth scales with what that debate finds; and a terminal entailment audit paired with a continuous Hallucination Risk Index that distinguishes claims that passed scrutiny from claims never contested. We evaluate CLAIR-Fin on BB-FinQA-X, a 500-question cross-modal financial evaluation set built from Bangladesh Bank Annual Report material, stratified by query type, format, and difficulty. Relative to a single-pass retrieval-augmented generation baseline, it raises faithfulness ($0.780 \rightarrow 0.889$) while abstaining on 5.4% of questions when evidence is insufficient rather than forcing an unsupported response, and it exceeds stronger retrieval-strategy baselines such as HyDE and Graph-RAG on faithfulness ($\leq 0.874$). 

---
# SAGE: Surrogate-gradient Adaptation via Attention-Guided Entropy for Spiking Transformers 

**Authors**: Kiran Nair, Rodrigue Rizk, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.13702)  

**Abstract**: Spiking neural networks (SNNs) offer an energy-efficient alternative to conventional deep neural networks by exploiting sparse event-driven computation, but their training remains challenging because the non-differentiable spike function requires surrogate gradients whose fixed shape may be suboptimal across layers and training stages. In this work, we introduce SAGE, an uncertainty-modulated surrogate-gradient mechanism for Transformer-based SNNs. SAGE estimates block-level uncertainty from normalized self-attention entropy and uses this signal to adapt the surrogate-gradient slope during training while leaving the inference model unchanged. By modulating only the training-time surrogate parameter, the proposed method preserves the original architecture and deployment cost while improving optimization flexibility. Experiments on CIFAR-10/100 demonstrate that SAGE achieves improved accuracy over fixed-surrogate baselines, with results up to 1-2\% consistent gains across multiple simulation time steps. These results highlight the potential of attention-derived uncertainty as a lightweight training signal for adaptive surrogate-gradient learning in transformer-based SNNs. 

---
# MedPlex: Deep Vision-Language Co-Adaptation for Clinically Grounded Medical Segmentation 

**Authors**: Rafi Ibn Sultan, Hui Zhu, Chengyin Li, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13690)  

**Abstract**: Medical image segmentation is still largely treated as a vision-only problem, although clinical interpretation often relies on textual knowledge of anatomy, location, appearance, and surrounding context. Existing text-guided segmentation methods within the Vision-Language Model (VLM) paradigm often use language only as a late conditioning signal, limiting its influence on visual representation learning. We introduce MedPlex (Medical Plexus of Vision and Language), an end-to-end VLM framework that makes text guidance a continuous, clinically grounded component of segmentation learning. Through Bi-Fusion (Bidirectional Fusion), visual and textual representations evolve jointly across the encoding hierarchy. MedPlex further introduces class-level and region-level concept alignment to organize the shared representation at complementary granularities. Class-level alignment anchors each anatomical target to an aggregated clinical concept profile, while region-level alignment preserves individual concepts, such as shape, location, appearance, and texture, through class-specific visual evidence. In this way, language provides structured supervision throughout the encoder rather than serving only as a late-stage cue. MedPlex achieves state-of-the-art performance across CT and MR benchmarks for multi-organ, cardiac substructure, and tumor segmentation, including settings with real free-text clinical supervision. Code: this https URL. 

---
# Fine-Tuning Qwen3-27B for C-to-Rust Code Translation: A Three-Stage Curriculum of Pretraining, Debugging-Aware SFT, and Task-Specific SFT 

**Authors**: Pu Zhao, Changdi Yang, Yixiao Chen, Yi Gao, Yifan Cao, Haochen Zeng, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13681)  

**Abstract**: Translating C code into safe, idiomatic Rust is a longstanding software-engineering goal because it can eliminate entire classes of memory-safety vulnerabilities while preserving the functional behavior of legacy systems. Large language models (LLMs) have shown promise for this task but typically underperform when applied off-the-shelf, since general-purpose pretraining rarely emphasizes idiomatic Rust generation, cross-language semantic equivalence, or the ability to reason about and repair compiler/runtime feedback. In this report we describe a three-stage fine-tuning curriculum applied to Qwen3-27B that is designed to progressively specialize the model for the C-to-Rust (C2Rust) translation task: (1) continued pretraining on Rust-centric corpora to strengthen the model's prior over idiomatic Rust syntax and standard-library usage; (2) supervised fine-tuning (SFT) on the microsoft/Verus_Training_Data dataset to instill debugging and self-repair behavior over Rust code; and (3) task-specific SFT on paired C/Rust solutions derived from LeetCode problems to teach direct semantic translation. We evaluate the resulting model using the agentic, static-analysis-guided verification framework of SACTOR, which performs structure-aware, two-phase (unidiomatic to idiomatic) translation with foreign-function-interface (FFI)-based end-to-end (E2E) testing. We report success rate, idiomaticity (Clippy lint counts, unsafe-code fraction), and failure-mode analyses, and compare our fine-tuned model against baseline Qwen3-27B and other LLMs evaluated under the same framework. 

---
# From BERT to Frontier Agents: Eight Years of Language-Model Progress, the Collapse of the Capability-Cost Curve, and the Rise of Task-Targeted Models 

**Authors**: Pranav Kumar Kaliaperumal  

**Link**: [PDF](https://arxiv.org/pdf/2608.13675)  

**Abstract**: Between October 2018 and July 2026 AI models progressed from simple systems like BERT to massive agents that solve complex math and write software. The ability to resolve real coding issues improved by nearly six times per year since late 2024. During this time costs dropped sharply with OpenAIs budget model GPT 5 point 6 Luna matching flagship capabilities for just one to six dollars per million tokens beating older versions at a fraction of the price. Top performance is now split across specialized models as Claude Opus 5 leads in frontend coding Claude Fable 5 excels at repository level coding and GPT 5 point 6 Sol dominates terminal tasks. In a grade school math test using the Qwen 2 point 5 model basic methods solved 58 of 100 problems while advanced sampling solved up to 79. A confidence ranking tool correctly identified 47 right answers in its top 50 choices proving highly useful for sorting tasks with all research materials made fully public. 

---
# Measuring Fairness in Large Audio Language Models via Semantic-Aware Bias Estimation 

**Authors**: Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13624)  

**Abstract**: Large Audio Language Models (LALMs) have seen increasing use for audio understanding tasks such as speech recognition and audio question answering, raising concerns about fairness across demographic subgroups. Fairness evaluation in spoken-input settings is challenging due to confounding factors, including semantic variation in spoken content and speaker-specific characteristics. Ignoring these factors can result in misleading conclusions about model bias. We propose a semantic-aware mixed-effects regression framework for fairness evaluation in LALMs that explicitly accounts for these confounders. Our approach incorporates sentence-level semantic embeddings of reference text as covariates and models speaker identity as a random effect. Notably, semantic representations are extracted from the same LALM under evaluation, enabling semantic control over variation as perceived by the model itself. Experiments on simulated data and real-world benchmarks demonstrate that the proposed approach substantially reduces spurious fairness findings and yields more robust and interpretable estimates of subgroup performance differences. 

---
# Secret-Stego Dissimilarity as a Design Axis: Invertible Coverless Image Steganography with Diffusion Models 

**Authors**: Hongxin Xu, Jianping Mei, Can Wang, Defang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.13597)  

**Abstract**: Coverless image steganography (CIS) synthesizes a stego image rather than modifying an existing cover image, enabling authorized recipients to reconstruct the original secret image from the stego. Existing diffusion-based CIS methods can generate natural-looking stego images but preserve substantial visual similarity to the secret image. This resemblance risks exposing structural and semantic cues, giving rise to security vulnerabilities that cannot be evaluated solely via recovery fidelity. Achieving substantial visual dissimilarity between the secret and stego images without compromising stego quality and recovery fidelity remains challenging. To address this issue, we propose InvCISD, an invertible diffusion framework that couples the latent representations of the secret and an irrelevant reference image with an invertible network called LIMNet. We first train LIMNet in diffusion latent space, followed by end-to-end fine-tuning of the entire network, i.e., LIMNet integrated diffusion inversion and generation modules. Experiments demonstrate that the proposed method substantially reduces secret-stego visual similarity, improves stego quality, and retains satisfactory secret reconstruction quality. Our further investigation shows that all evaluated methods are highly detectable by the CIS-oriented steganalysis model, indicating that resistance against targeted steganalysis constitutes a critical direction for future CIS research. 

---
# Training-Free Knowledge Transfer Across Model Scales through Activation-Guided Pruning 

**Authors**: Jiahe Fan, Si Chen, Yinghao Hou, Aiyuan Zhang, Hong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.13596)  

**Abstract**: Heterogeneous model fusion seeks to combine models that differ in tasks, initializations, architectures, or scales. We study an underexplored cross-scale setting: improving a small recipient language model with a stronger donor despite substantial architectural mismatch. We ask whether useful capabilities can be transferred without explicit neuron-wise semantic alignment. Building on the observation that truncating a large model to a smaller architecture and injecting it with a tiny mixing weight can already improve the recipient, we propose Activation-Prune-Merge (APM), an activation-guided framework for cross-scale fusion. APM constructs task-conditioned activation maps on the donor, selects salient layers, hidden dimensions, attention heads, and MLP neurons to prune it to the recipient architecture, and injects the resulting donor slice into the original recipient using a micro interpolation coefficient. This formulation treats the donor as a source of concentrated functional components rather than requiring precise structural transplantation. Across 16 benchmarks spanning reasoning, mathematics, code generation, instruction following, and classification, APM improves the overall average accuracy from 55.5% to 60.6% over the original 3B recipient. RTE accuracy increases from 64.3% to 82.3%, QNLI from 52.3% to 65.7%, and BoolQ from 70.8% to 79.2%. Analyses of injection ratios and sequential multi-stage fusion further suggest that activation-guided extraction improves the quality of the transferable donor slice while preserving the small-ratio fusion regime. These results provide evidence that cross-scale heterogeneous fusion can succeed without explicit semantic alignment when the donor contribution is sufficiently concentrated and carefully selected. 

---
# Context Aware AI Assistant and AR Interface for Lunar Extravehicular Activity (EVA) Procedural Guidance 

**Authors**: Rodrigo Gallardo, Qilmeg Doudatcz, Ganit Goldstein, Ilkyaz Sarimehmetoglu, Sergio Mutis, Alexander Htet Kyaw, Anita Lin, Clara Emmerling, Berfin Ataman, Skylar Tibbits  

**Link**: [PDF](https://arxiv.org/pdf/2608.13589)  

**Abstract**: As human space exploration returns to the Moon, astronauts need rapid access to procedural information during extravehicular activities (EVAs), where attention is divided across navigation, repair tasks, tool handling, and environmental risk. The challenge is not the absence of information, but surfacing the right information at the right moment. We present GAIN-AI (Guided Assistant for Intelligent Navigation), a context-aware AI assistant and minimal heads-up interface for procedural guidance in simulated lunar EVA. The system operates in two layers. The first grounds a large language model with structured context: EVA procedure documents, live telemetry data, and error-handling protocols encoded as JSON. The second restructures that output into three compact units for AR display: Goal, Task, and Verification. Evaluated on 111 synthetic EVA scenarios, the system scores 10.0/10 on nominal conditions and 8.15/10 on single-fault scenarios, with performance degrading on multi-fault and boundary-threshold cases. 

---
# IterCOMP: Reasoning-aware Adaptive Prompt Compression for Multi-hop Question Answering 

**Authors**: JungMin Yun, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.13588)  

**Abstract**: Multi-hop question answering requires complex reasoning across multiple evidence segments, which often overwhelms retrieval-augmented generation systems with lengthy and noisy contexts, thereby undermining both efficiency and accuracy. While existing prompt compression methods attempt to address this issue, they are typically designed for single-turn queries and fail to capture interdependent reasoning steps. We propose IterCOMP, a unified, training-free prompt compression framework that incorporates multi-hop reasoning within an iterative compression loop. IterCOMP decomposes documents into evidence segments, evaluates question answerability, and generates targeted follow-up questions to iteratively integrate essential evidence, producing a compact, reasoning-oriented prompt. Experiments on MusiQue, 2WikiMultiHopQA, and HotpotQA demonstrate that IterCOMP achieves substantial improvements in Exact Match and F1 scores while reducing the token budget, outperforming existing baselines and exhibiting robustness as reasoning complexity increases. 

---
# UltraArUco: A Lightweight Multilingual Library And Framework With Low-Latency Real-Time Marker-Based Tracking System For Mobile AR Interaction 

**Authors**: Mikhail Kiselev, Aleksandr Marukhin, Ivan Snegirev, Elizaveta Semenyakina, Miguel Altamirano Cabrera, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2608.13584)  

**Abstract**: UltraArUco - a lightweight multilingual library and framework for low-latency, real-time marker-based tracking in mobile augmented reality. Unlike standard OpenCV-based implementations, UltraArUco introduces an optimized multilingual wrapper that reduces per-frame latency by five times, while maintaining high accuracy. Distributed Wi-Fi architecture provides portability, connects a mobile device (camera input) with a PC-based visual application, enabling responsive interactions. The framework is validated through an interactive piano simulation, where static ArUco markers on keys enable occlusion-based note triggering, and hand-mounted markers provide spatial gesture recognition. UltraArUco's system requirements make it perfect for resource-constrained mobile AR applications, demonstrating a viable AR music application without specialized equipment. 

---
# From Prediction to Intervention: Personalized Meal-Level Glucose Regulation via an LLM Agent 

**Authors**: Mingyu Huang, Weiqing Min, Ying Jin, Yilin Wang, Shuqiang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13581)  

**Abstract**: Personalized glucose regulation remains a central yet unresolved challenge in precision nutrition, as postprandial glucose response varies substantially across individuals. Existing approaches based on glycemic indices fail to adequately account for such heterogeneity and lack the mechanism to dynamically adjust meals based on personal physiological feedback. In this context, recent advances in LLM-based agents offer a promising direction, as they enable context-aware reasoning and iterative refinement. Inspired by this, we propose a physio-feedback agentic loop, a unified system that integrates individualized absorption modeling with dietary intervention to regulate glucose response. Specifically, we develop a Physiology-Aware Glucose Predictor to model individualized absorption dynamics through a learnable Temporal Physiological Absorption Decay Module. We then construct a Prediction-Driven Two-Stage Meal Optimization Agent that iteratively refines real-world meals using predicted outcomes as explicit feedback. Through extensive experiments on multiple public datasets, we demonstrate that our method not only improves prediction accuracy but also effectively reduces glucose excursions. To the best of our knowledge, this paper marks the first step in integrating physiological learning with an LLM-based agent for personalized glucose regulation. 

---
# Jais 2: A Family of Arabic-Centric Open Large Language Models 

**Authors**: Mohamed Anwar, Abed Alhakim Freihat, George Ibrahim, Mostafa Awad, Abdelrahman Sadallah, Gurpreet Gosal, Gokulakrishnan Ramakrishnan, Sarath Chandran, Biswajit Mishra, Rituraj Joshi, Ahmed Frikha, Etienne Goffinet, Abhishek Maiti, Ali El Filali, Sarah AlBarri, Samujjwal Ghosh, Rahul Pal, Parvez Mullah, Awantika Shukla, Sajid siddiki, Samta Kamboj, Onkar Pandit, Sunil Kumar Sahu, AbdelRahman Elbadawy, Amr Mohamed, Ahmad Chamma, Evan Dufraisse, Abdelaziz Bounhar, Dani Bouch, Hadi Abdine, Guokan Shang, Fajri Koto, Yuxia Wang, Zhuohan Xie, Ali Mekky, Rania Elbadry, Sarfraz Ahmad, Momina Ahsan, Omar El Herraoui, Daniil Orel, Hasan Iqbal, Kareem Elzeky, Mervat Abassy, Kareem Elozeiri, Saadeldine Eletter, Farah Atif, Nurdaulet Mukhituly, Haonan Li, Xudong Han, Aaryamonvikram Singh, Zainul Abedien Ahmed Quraishi, Neha Sengupta, Larry Murray, Avraham Sheinin, Joel Hestness, Natalia Vassilieva, Hector Xuguang Ren, Zhengzhong Liu, Michalis Vazirgiannis, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2608.13580)  

**Abstract**: Jais 2 is a family of Arabic-centric large language models developed jointly by MBZUAI, Cerebras, and Inception, designed to advance Arabic-centric language modeling, with strong performance across the Arabic and culturally grounded benchmarks evaluated in this report. The family includes, to our knowledge, the largest open Arabic-centric LLM trained from scratch at 70B parameters, and a competitive 8B-parameter variant among the evaluated open models. A custom Arabic-centric vocabulary enables efficient training and inference. In addition, an optimized architecture and training recipe yield highly compute-efficient training. With a substantially smaller token budget than comparable models, Jais 2 achieves strong Arabic performance on the benchmarks considered in this report and competitive English results. The models obtain leading results among the evaluated open models on OALL2 and AraGen. They also perform strongly on several culturally grounded Arabic benchmarks, including poetry, religion, cuisine, and dream interpretation, as well as in general tasks such as translation and summarization. We release the models in HuggingFace under a commercially permissive license. Jais 2 70B is also released as a chat app on the Web, iOS, and Android; it runs on Cerebras hardware, delivering up to 2,000 tokens per second, and enabling high-throughput Arabic-centric chat serving in our deployment setting. By uniting scale, linguistic diversity, cultural fidelity, openness, and speed, Jais 2 provides an open-weight foundation intended to support further research and development in Arabic-centric LLMs. 

---
# BCMT: Blockwise Causal Memory Transformer 

**Authors**: Rachid Arezki  

**Link**: [PDF](https://arxiv.org/pdf/2608.13578)  

**Abstract**: Transformer architectures rely on dense self-attention to model long-range dependencies, but this mechanism exhibits quadratic complexity with respect to sequence length. We introduce BCMT (Blockwise Causal Memory Transformer), an architecture for long-context language modeling that decouples local token interactions from global context propagation. Dense causal self-attention is applied independently within local blocks, while each block produces an adaptive summary aggregated through an exponential causal memory. This memory is subsequently injected back into the token representations, enabling efficient propagation of long-range contextual information without relying on explicit global attention. Unlike standard Transformers and recurrent memory architectures, BCMT maintains neither dense interactions between distant tokens nor learned memory states. Its memory mechanism is fully parallelizable and remains compatible with standard implementations of dense self-attention. Experiments on language modeling with context lengths of up to 1024 tokens show that BCMT achieves validation performance comparable to that of Dense Transformers while significantly improving training throughput and reducing memory consumption. An ablation study further confirms that these improvements arise from the proposed memory mechanism. These results demonstrate that an exponential causal memory constructed from block summaries provides an effective alternative to dense global attention mechanisms for long-context language modeling. 

---
# Interactive Analysis of Global Explanations using Aggregated Class Activation Maps for Network Data 

**Authors**: Igor Cherepanov, David Sessler, Alex Ulmer, Felix Wagner, Throsten May, Jörn Kohlhammer  

**Link**: [PDF](https://arxiv.org/pdf/2608.13575)  

**Abstract**: Recent machine learning (ML) advances have demonstrated that deep learning (DL) achieves impressive results in different application domains, including the classification of computer network traffic to corresponding applications. However, the data frequently contains diverging patterns within a single predicted class. This presents a significant challenge to the ability to provide a clear and comprehensive explanation and emphasizes the necessity for tools capable of detecting and analyzing these patterns. Furthermore, the capacity to extract descriptive rules for classes is a crucial requirement in network traffic analysis and intrusion detection, particularly when leveraging advanced tools like next-generation firewalls. We provide a visual-interactive system that explains predictions of classes for network traffic. Global explanations derived from multiple samples of a given class contribute to understanding model predictions. Visualization of global explanations enables recognition of different patterns that offer experts a more comprehensive overview of its characteristics. We introduce a prototype that facilitates visual exploration and refinement of global explanations, enabling network experts to detect and refine new patterns for specific applications. These explanations support the identification of misleading features and the formulation of new rules for the management of networks. Our approach also aims at enabling ML experts to acquire new insights, including the possibility of separating or merging classes and the development of more accurate and reliable DL models. Our proposed prototype was evaluated by experts in machine learning and network analysis. 

---
# The Architect: Interactive Visualization of Deep Learning Mathematics Directly in Microsoft Excel 

**Authors**: Mohammad Imrul Jubair, Tom Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2608.13572)  

**Abstract**: We present The Architect, a system that turns Microsoft Excel into an interactive view of deep learning mathematics. A user describes a neural network in a compact table. The system then generates a workbook that shows the full forward pass and, when requested, the backward pass and parameter updates. Computed values appear as live spreadsheet formulas, while user-controlled values such as inputs, weights, labels, and hyperparameters remain editable. Excel reactively updates the dependent computations through its recalculation engine.
Most deep learning tools hide the numerical details behind library calls. Many visualization tools show architecture diagrams or training summaries, but they do not expose the full arithmetic of the model. The Architect focuses on that missing middle layer. It makes matrices, activations, losses, gradients, and updates visible as inspectable spreadsheet regions, with editable controls for values users naturally manipulate. The system also produces aligned PyTorch snippets, which helps users connect formulas to implementation.
This report describes the motivation, design, implementation, and use cases of The Architect. We show how the system supports introductory arithmetic tracing, learning-rate exploration, diagnosis of dying ReLU, and inspection of vanishing gradients. The main idea is simple: spreadsheets already support formulas, direct editing, reactive recomputation, and tabular layout. These properties make them a useful medium for understanding how small educational and diagnostic neural networks compute. 

---
# Not All Tokens Are Equal: Inflation-Aware Routing for Agentic LLM Systems 

**Authors**: Heming Fu, Shan Lin, Qianqian Xie, Guojun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2608.13571)  

**Abstract**: When a language model fails to answer a query on the first attempt, an agentic system retries, consuming additional tokens each time. This retry overhead creates a gap between what a model's per-token price implies and what a full workflow actually costs. We call this gap \emph{token inflation} and define it as the ratio of true workflow cost to single-call cost. Systems like FrugalGPT route based on the latter, which can underestimate real cost by more than $2\times$ on difficult tasks. We address this with InflationAgent, a four-stage router that (1) measures token inflation systematically across model tiers and task types, finding inflation as high as $4.25\times$ for a 7B model on multi-hop question answering; (2) introduces CoT Branching Entropy (CBE), a pre-execution difficulty signal computed entirely from local inference, which predicts high inflation with AUROC 0.887; and (3) selects models by maximizing a Semantic Exchange Rate (SER) that divides expected accuracy by predicted true cost, with a fresh-escalation policy that discards failed chains before routing to a stronger model. On GSM8K under a fixed budget, InflationAgent achieves 94.7\% accuracy versus 91.0\% for FrugalGPT while using 31\% fewer tokens, and we show that forwarding a failed reasoning chain to GPT-4o reduces its accuracy by up to 34.8 percentage points, validating the fresh-escalation design. 

---
# Think in Latent, Explain in Language: Self-Explainable Latent Reasoning 

**Authors**: Dayuan Zhao, Shengcao Cao, Yu-Xiong Wang, Liang-Yan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2608.13570)  

**Abstract**: Latent reasoning has emerged as a powerful alternative to text-based Chain-of-Thought (CoT), offering significant gains in computational efficiency by compressing verbose reasoning into compact embeddings. However, compressing reasoning into the latent space renders the thinking opaque, hindering its interpretability. Current methods present a stark trade-off: they either function as unexplainable ''black boxes'' (e.g., Coconut), where the latent reasoning is not human-readable, or rely on separate post-hoc decoders for explainability (e.g., Heima), introducing architectural overhead and decoupling the explanation from the actual reasoning process. In this work, we present a unified framework for Self-Explainable Latent Reasoning (SELR) that trains a single model to perform efficient and inherently explainable latent reasoning. Our core contribution is a novel multi-task training objective that optimizes for two goals simultaneously: (1) an Answer Loss that optimizes the latent reasoning trajectory to produce accurate final answers, and (2) a CoT Loss that explicitly trains the same model to decode its own latent representations back into human-understandable reasoning steps. This design ensures that generated latent representations are both task-effective and semantically interpretable, eliminating the need for external decoders. We validate the effectiveness of SELR on both Large Language Models (LLMs) and Vision-Language Models (VLMs), demonstrating that SELR achieves superior token efficiency and accuracy compared to baselines, while uniquely providing self-contained explainability without auxiliary models. Project page is available at this https URL. 

---
# Does a Language Server Save Tokens for Coding Agents? A Measurement Methodology and Preliminary Study 

**Authors**: Pengcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13568)  

**Abstract**: Coding agents spend most of their context budget on retrieval. Lexical retrieval (grep) is universal, instant, and zero-setup, but noisy: it cannot tell a definition from a call from a comment. Semantic retrieval via the Language Server Protocol (LSP) is precise and typed, but needs a running, indexed server and pays a per-symbol round-trip. The claim that semantic retrieval is more token-efficient is, we find, asserted almost everywhere and measured almost nowhere: no public source isolates the LSP-vs-lexical token delta for an agent at equal task-success. This paper formalizes the question with one metric (tokens-to-success), specifies a five-arm ablation isolating semantic retrieval from confounds, maps three pre-stated failure modes onto measurable variables, and reports a preliminary study (Python and TypeScript repos; Claude Opus 4.8, Sonnet 4.6, Haiku 4.5). The answer is conditional and usually negative. On symbol-named localization the LSP costs tokens (+6% to +118%) and the agent ignores it when free. On reference-completeness it buys precision but not token savings and cannot raise the recall ceiling set by agent thoroughness; it saves tokens only for the weakest model. Tool choice is task-dependent: models default to grep on localization (0-6% semantic use) but reach for the LSP about half the time on reference tasks, unprompted. On edits scored by real test execution the gap is starkest: grep solves multi-file renames perfectly, a location-only LSP fails three-quarters of them by missing a call site, and even a complete, index-warmed, text-enriched LSP (each reference's line inline, as production LSP-MCP servers do) recovers most of the gap but cannot close it, since a rename must touch comments and strings that semantic references exclude. The implication is not LSP-always but an adaptive router keyed on task class, model capability, and lexical noise. 

---
# Don't Claim Benchmark-Oriented Optimization Improves General Coding Capability -- Diverse Evaluation Is Required 

**Authors**: Egor Shibaev, Vera Kudrevskaia, Timur Galimzyanov, Mikhail Evtikhiev, Ana Terna, Rastislav Rabatin, Timur Kudashev, Timofey Bryksin, Arina Puchkova, Patrik Bartak, Egor Bogomolov, Sergey Titov  

**Link**: [PDF](https://arxiv.org/pdf/2608.13566)  

**Abstract**: Post-training papers, model cards, and blog posts often treat scores on a small set of coding benchmarks (e.g., SWE-bench and LiveCodeBench) as evidence of broad coding capability, both for research artifacts and user-facing systems. We argue that optimization for these benchmarks leads to measuring task-specific performance, creating a meaning gap between measured scores and claims of general coding ability. We examine this gap with a Django-based case study benchmark suite we create.
Evaluating foundation models and checkpoints post-trained on SWE-bench trajectories, we find that benchmark rankings frequently fail to generalize. Post-trained checkpoints show little cross-task transfer, and SWE-bench optimization yields limited or no gains on our tasks or on LiveCodeBench. Similarly, fine-tuning on individual Django modalities fails to transfer.
We conclude that a small number of benchmarks is insufficient for evaluating diverse models under benchmark optimization pressure. We encourage the community to use differentiated evaluation - holistic assessment for frontier models, multi-task suites for research, and human-in-the-loop studies for narrow task applications. Finally, we argue for creating a capability taxonomy and sustained benchmark maintenance, rather than one-off benchmark releases. Without reliable evaluation standards, engineers and researchers using LLMs and agents have to rely on insufficient evidence to make research, development, and deployment decisions. 

---
# Proxy-Validated LLM UX Micro-Simulations: An Artifact-First Protocol for Early-Stage Decision Support 

**Authors**: Alexandre Cristovão Maiorano  

**Link**: [PDF](https://arxiv.org/pdf/2608.13563)  

**Abstract**: Early-stage teams often lack users, time, and budget to run repeated UX studies, yet still need decision-oriented signals to iterate safely. We study an LLM-driven UX micro-simulation pipeline that generates structured customer-experience feedback (walkthrough steps, friction points, micro-survey signals) from versioned prompts, personas, tasks, and UI snapshots. Because public usability datasets with task outcomes are scarce, we validate simulated friction themes using multiple public proxy corpora (app reviews, support tweets, and open-source software issues). We propose a lightweight proxy-validation protocol with two alignment metrics: top-k Jaccard and distributional weighted-Jaccard (W), and compare lexical, TF-IDF, and multilingual embedding baselines across six proxy datasets. Embedding-based alignment yields higher W than lexical baselines on primary app-review and support-tweet proxies (e.g., W=0.128 vs 0.000 on Gojek), while top-k Jaccard is shown to overstate alignment at large k. We ablate four agent strategies (single-pass, best-of-N, hybrid, and a proposed score-then-select judge) across Azure OpenAI deployments and report bootstrap confidence intervals over 8 method-dataset pairs; these intervals reveal that the embedding W point estimate is systematically unstable under resampling at our subsample size. We also provide a failure-mode analysis of grounding and fabrication proxies, with documented calibration caveats and worked examples of outputs flagged as fabricated by an adversarial judge. Our artifact-first pipeline produces reproducible tables and figures from versioned run artifacts, supporting iterative prompt and taxonomy refinement before final paid-model calibration. 

---
